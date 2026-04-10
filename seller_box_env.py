import numpy as np
np.seterr(under='ignore')

from gymnasium.spaces import Box, Dict, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.repeated import Repeated
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
import torch
import gc
import torch.nn.functional as F
import pandas as pd
import logging
import torch.nn as nn

import random
import itertools
from agents import FairAgentOptimizer, LinearAgent, ModelAgent, OptimizerAgent, FairPostProcessorAgent, AdversarialMitigationModel, FeldmanAgent
from collections import Counter

MAX_VAL = 1e10

class SellerBoxEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.bias = float(config.get('bias', 0))
        self.max_steps = int(config['max_steps'])
        self.start_capital = int(config['start_capital'])
        self.price_elasticity = float(config['price_elasticity'])
        self.start_demand = config['start_demand']
        self.premium_qty = config['premium_qty']
        self.max_demand = self.start_demand + (self.price_elasticity * self.start_demand)
        
        self.late_join_ep = config.get('late_join_ep', [])
        self.num_agents = config['num_agents']
        
        self.action_gradients = config['num_action_gradients']
        self.max_quantity = config['max_quantity']
        self.num_actions = config['num_actions']
        self.last_orders = {}
        
        self.action_stack = np.geomspace(0.01, 1, self.action_gradients)
        
        self.fair = int(config.get('fair', 0))
        self.lin_agent = OptimizerAgent(
            base_demand=self.max_demand,
            demand_elasticity=self.price_elasticity,
            sensitivity_installments=0.2,
            sensitivity_delivery=0.4,
            marketplace_tax=0.10
        )

        # TEMPORARY: Uncomment the fair method desired

        #self.fair_agent = FeldmanAgent() # Feldman et al.
        self.fair_agent = FairAgentOptimizer() # Yang and Stoyanovich
        #self.fair_agent = config['adv_model'] # ARBMA

        self.action_space = Dict({
            'offer': Discrete(self.action_gradients ** self.num_actions),
            'quantity': Discrete(self.max_quantity),
        })

        self.player_space = Dict({
            'prices': Box(low=0, high=MAX_VAL, shape=(self.num_agents,), dtype=np.float32),
            'cost': Box(low=0, high=MAX_VAL, shape=(1,), dtype=np.float32),
            'quantity': Box(low=0, high=MAX_VAL, shape=(1,), dtype=np.float32),
            'capital': Box(low=0, high=MAX_VAL, shape=(1,), dtype=np.float32),
        })
        
        self.observation_space = Repeated(self.player_space, max_len=self.num_agents + len(self.late_join_ep))

        self.action_mapping = list(itertools.product(*[range(self.action_gradients) for _ in range(self.num_actions)]))
        self.cost = config['unit_cost']
        self.start_cost = self.cost
        self.reset()

    def step(self, action_dict):
        self.current_step += 1
        rewards = {}
        self.new_dones = {}
        
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            mapping = self.action_mapping[action['offer']]
            vals = [self.action_stack[i] for i in mapping]
            agent["price"] = self.sigmoid(torch.tensor([vals[0]], dtype=torch.float32)).numpy()
            agent["installments"] = self.sigmoid(torch.tensor([vals[1]], dtype=torch.float32)).numpy()
            agent["delivery_time"] = self.sigmoid(torch.tensor([vals[2]], dtype=torch.float32)).numpy()
            agent["quantity"] = np.array([float(action['quantity'])], dtype=np.float32)

        orders = self._calc_orders()
        self.last_orders = orders

        for agent in self.agents:
            a_id = agent['id']
            if self.done.get(a_id, False):
                rewards[a_id] = 0.0
                continue

            agent['stock_cost'] = self._calc_stock_cost(agent['left_over'], agent['premium'])
            
            price = agent['price'][0]
            agent['revenue'], agent['left_over'] = self._calc_sales(
                agent['quantity'][0], agent['left_over'], orders[a_id], price
            )

            agent['profit'], _ = self._calc_composite_cost(
                agent['price'], agent['installments'], agent['delivery_time'],
                agent['quantity'], agent['left_over'], agent['stock_cost'], agent['revenue']
            )

            self.capital[a_id] += float(agent['profit'])
            if self.capital[a_id] < 0: self.capital[a_id] = 0.0
            agent['capital'] = np.array([self.capital[a_id]], dtype=np.float32)

            if self.capital[a_id] <= 0 or self.current_step >= self.max_steps:
                self.new_dones[a_id] = True
                self.done[a_id] = True
            else:
                self.new_dones[a_id] = False

            rewards[a_id] = np.sign(float(agent['profit'])) * np.log1p(np.abs(float(agent['profit'])))

        self.done["__all__"] = all(self.done[a['id']] for a in self.agents)
        self.new_dones["__all__"] = self.done["__all__"]
        
        return self._next_observation(), rewards, self.new_dones, {"__all__": False}, {}

    def _calc_orders(self):
        orders = {i: 0 for i in range(self.num_agents)}
        active_agents = [a for a in self.agents if not self.done.get(a['id'], False)]
        
        if not active_agents: return orders

        rows = [{'price': a['price'][0], 'installments': a['installments'][0], 
                 'delivery_time': a['delivery_time'][0], 'premium': a['premium']} for a in active_agents]
        
        df_rank = self.lin_agent.rank(pd.DataFrame(rows))
        df_rank['scores'] = df_rank['scores'].replace(0, 0.001).fillna(0.001)
        
        self.scores = df_rank['scores'].values
        self.rank = df_rank['rank'].values
        if self.fair == 1:
            try: 
                df_fair = self.fair_agent.rank(df_rank)
                self.scores = df_fair['fair_scores'].values
                self.rank = df_fair['fair_rank'].values
            except: pass

        shift_scores = self.scores - np.max(self.scores)
        exp_scores = np.exp(shift_scores)
        probabilities = exp_scores / np.sum(exp_scores)
        
        probabilities = np.nan_to_num(probabilities)
        probabilities /= probabilities.sum()

        max_demand = self.calc_demand(max(a["price"][0] for a in active_agents))
        
        if max_demand > 0:
            agent_ids = [a['id'] for a in active_agents]
            choices = np.random.choice(agent_ids, size=int(max_demand), p=probabilities)
            for aid in choices: orders[aid] += 1
            
        return orders

    def _calc_composite_cost(self, price, delivery, inst, qty, left, stock_c, rev, quota=0.5):
        comp = (price + inst + delivery) * quota
        gen_cost = ((qty - left) * (self.cost + comp)) + stock_c if qty > left else stock_c
        return float(rev - gen_cost), float(gen_cost)

    def _calc_sales(self, qty, left_over, demand, price):
        total_avail = int(qty + left_over)
        if demand > 0:
            sold = np.clip(total_avail, 0, demand)
            rev = float((self.cost + (1.01 - price) + 3) * sold)
            return rev, float(total_avail - sold)
        return 0.0, float(total_avail)

    def _calc_stock_cost(self, left, premium, tax=0.2):
        mult = tax if premium else 0.1
        return float(left * self.cost * mult)

    def calc_demand(self, price):
        d = -1 * (-self.start_demand * self.price_elasticity - self.start_demand + 
             (self.start_demand * (1 - price) * self.price_elasticity))
        return np.clip(d, 0, None)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.capital = [float(self.start_capital) for _ in range(self.num_agents)]
        self.done = {i: False for i in range(self.num_agents)}
        self.done["__all__"] = False
        
        self.agents = [{
            'id': i, 'price': np.array([self.cost]), 'quantity': np.array([1.0]),
            'installments': np.array([0.5]), 'delivery_time': np.array([0.5]),
            'capital': np.array([self.start_capital], dtype='float32'),
            'left_over': 0.0, 'premium': i < self.premium_qty
        } for i in range(self.num_agents)]
        
        return self._next_observation(), {}

    def _next_observation(self):
        all_prices = np.array([a['price'][0] for a in self.agents], dtype=np.float32)
        obs = {}
        for a in self.agents:
            if not self.done.get(a['id'], False):
                obs[a['id']] = [{
                    'prices': all_prices,
                    'cost': np.array([self.cost], dtype=np.float32),
                    'quantity': a['quantity'],
                    'capital': a['capital']
                }]
        return obs