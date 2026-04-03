# Marketplace Simulation and ARBMA

A Python-based marketplace simulation environment and implementation of the ARBMA algorithm.

## Prerequisites

Before running the simulation, ensure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Simulation

1. **Prepare the logging directory:** Create a `data/` folder in the root directory to store the simulation logs.
2. **Configure the environment:** Open `main.py` and edit the `env_config` dictionary to parametrize your simulation as needed.
3. **Toggle Fairness (Temporary Step):** * Open `seller_box_env.py` and manually uncomment the fairness mitigation method if you want a **Fair** simulation. 
   * Leave it commented out for an **Unfair** simulation. 
   * *(Note: This will be moved to a configuration parameter in future updates).*
4. **Execute the simulation:** Run the following command in your terminal. You can adjust the `--num-agents` flag to your preferred number of agents.

```bash
python main.py --local-log --framework torch --num-agents 14 --algorithm "PPO"
```

## Running ARBMA Training

To train the ARBMA model, it must be trained on the logs of an **unfair** simulation.

1. Ensure you have run a simulation with the fairness mitigation disabled (see step 3 above).
2. Configure `train_arbma.py` to point to the `data/` folder containing the logs from that specific unfair simulation.
3. Run the training script:

```bash
python train_arbma.py
```
```