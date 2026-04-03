import argparse
import csv
import os
import time
from typing import Dict
import random
import numpy as np
import ray
import uuid
import logging
import joblib
from ray.rllib import BaseEnv
from datetime import datetime
from ray.rllib.algorithms import ppo, dqn
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import try_import_torch
from ray.tune import register_env
from agents import AdversarialMitigationModel, DeltrAgent
from model import PredictorNetwork
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.evaluation.episode_v2 import EpisodeV2

#from marketenvDiscreteNoQuantity import MarketEnvDiscreteNoQuantity
#from marketenvDiscreteNoQuantityBlind import MarketEnvDiscreteNoQuantityBlind
#from marketenvDiscrete import MarketEnvDiscrete
from seller_box_env import SellerBoxEnv

torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default="PPO")  # PPO, DQN, SAC
parser.add_argument("--gpuid", type=str)
parser.add_argument("--num-agents", type=int, default=3)
parser.add_argument("--late-join-ep", '--list', action='append', help="--late-join-ep 10 --late-join-ep 20", type=list, default=[])
parser.add_argument('--framework', choices=["tf", "tf2", "tfe", "torch"], default="tf2")  # torch or tf2
parser.add_argument('--manual-log', help='use this to create a csv log file', action='store_true', default=False)
parser.add_argument('--bias', help="0 = no bias, 1 = full bias towards lowest price", type=float, default=0)
parser.add_argument('--blind', action='store_true', default=False)
parser.add_argument('--no-quantity', action='store_true', default=False)
parser.add_argument('--shared-policy', action='store_true', default=False)
parser.add_argument('--local-log', help='use this to run it locally in pycharm', action='store_true', default=False)
parser.add_argument('--custom-filename', type=str)
parser.add_argument('--supervision', help='activate another agent that controls the agents\' actions', action='store_true', default=False)

# cd /opt/new_mkp_env && python main.py --local-log --framework torch --num-agents 14 --algorithm "PPO"

class SaveSellerDataCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.num_agents = 14
        self.worker_uuid = str(uuid.uuid4())[:8]
        self.log_dir = "data"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.episode_rows = []

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
                         episode: EpisodeV2, **kwargs):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        episode.user_data["output_filename"] = os.path.join(
            self.log_dir, 
            f"seller_data_{timestamp}_{self.worker_uuid}.csv"
        )
        
        episode.user_data["agents"] = []
        self.episode_rows = []
        print(f"Started episode. Log will be saved to: {episode.user_data['output_filename']}")

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs):
        unwrapped_env = base_env.get_sub_environments()[0]
        if unwrapped_env.current_step > 0:
            agents = {}
            for agent in unwrapped_env.agents:
                agents[agent["id"]] = {
                    "price": agent["price"],
                    "quantity": agent["quantity"]
                }
            episode.user_data["agents"].append(agents)

            shuffled_agents = random.sample(unwrapped_env.agents, len(unwrapped_env.agents))

            for agent in shuffled_agents:
                left_over = agent.get('left_over', 0)
                revenue = agent.get('revenue', 0)
                profit = agent.get('profit', 0)
                capital = agent.get('capital', 0)
                composite_cost = agent.get('composite_cost', 0)

                self.episode_rows.append(
                    {'agent_id': agent["id"],
                     'timestamp_id': episode.user_data["output_filename"],
                     'step': unwrapped_env.current_step,
                     'price': agent["price"][0],
                     'installments': agent["installments"][0],
                     'delivery_time': agent["delivery_time"][0],
                     'premium': agent['premium'],
                     'score': unwrapped_env.scores[agent["id"]] if agent["id"] < len(unwrapped_env.scores) else 0,
                     'rank': unwrapped_env.rank[agent["id"]] if agent["id"] < len(unwrapped_env.rank) else 0,
                     'quantity': agent["quantity"][0],
                     'costs': unwrapped_env.cost,
                     'revenue': revenue,
                     'capital': unwrapped_env.capital[agent["id"]],
                     'profit': profit,
                     'demand_for_price': unwrapped_env.calc_demand(agent["price"])[0],
                     'orders': unwrapped_env.last_orders[agent["id"]],
                     'composite_cost': composite_cost,
                     'leftover_units': left_over,
                     },
                )

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy],
                        episode: EpisodeV2, **kwargs):
            filename = episode.user_data["output_filename"]
            
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['agent_id', 'timestamp_id', 'step', 'price', 'installments', 'delivery_time', 'premium', 'score', 'rank', 'quantity', 'costs',
                            'profit', 'revenue', 'capital', 'demand_for_price', 'orders', 'composite_cost', 'leftover_units']
                writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=fieldnames)
                
                writer.writeheader()
                for row in self.episode_rows:
                    writer.writerow(row)

            print(f"Data successfully saved to {filename}")

"""custom algo"""

def policy_mapping_fn(agent_id, episode, **kwargs):
    return str(agent_id)

"""run"""
if __name__ == "__main__":
    args = parser.parse_args()

    if args.gpuid:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
        ray.init(dashboard_host="0.0.0.0", dashboard_port=8905)
    else:
        ray.init(log_to_driver=True, logging_level='INFO')

    np.seterr('raise')

    import __main__
    setattr(__main__, "PredictorNetwork", PredictorNetwork)

    env_config = {
            "supervision": args.supervision,
            "num_agents": args.num_agents,
            "late_join_ep": [int(''.join(episodeStr)) for episodeStr in args.late_join_ep],
            "max_steps": 365,
            "log_level": "INFO",
            "start_demand": 50,
            "price_elasticity": 3,
            "start_capital": 50,
            "unit_cost": 1,
            "max_price": 5,
            "premium_qty": 7,
            "max_quantity": 50,
            "bias": args.bias,
            "num_action_gradients": 10,                                  
            "num_actions": 3,
            "fair": True,
            "adv_model": AdversarialMitigationModel('model/adversarial_new_predictor_final_aa.pth'),
        }

    policies = {str(agent_id): PolicySpec() for agent_id in range(args.num_agents + len(args.late_join_ep))}
    ppoconfig = {
        "env_config": env_config,
        "env": "marketenv",
        "horizon": 1000, 
        "model": {
            "fcnet_hiddens": [256, 256],
        },
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "count_steps_by": "agent_steps", 
        } if not args.shared_policy else {
            "count_steps_by": "agent_steps",
        },
        "callbacks": SaveSellerDataCallback,
        "lr": 1e-5,
        'kl_coeff': 0.2,
        "num_workers": 2,
        "rollout_fragment_length": 200, 
        "compress_observations": True,
        "batch_mode": "truncate_episodes",
        "train_batch_size": 5000,
        "framework": args.framework,
        "num_gpus": 0 if not args.gpuid else 1,
        "vf_clip_param": 100.0,
    }

    register_env('marketenv', lambda env_config: SellerBoxEnv(env_config))
    print(ppoconfig)

    if args.algorithm == "PPO":
        trainer = PPOTrainer(config=ppoconfig, logger_creator=None)


    policy = trainer.get_policy()
    episode = 0

    while episode < 5000:
        print(f'==> Episode {episode}')
        results = trainer.train()
        episode = results["episodes_total"]


    ray.shutdown()
