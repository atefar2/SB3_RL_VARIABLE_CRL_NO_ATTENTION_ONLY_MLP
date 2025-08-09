amen@amens-Laptop RL_MultiHeadAttention % cd /Users/amen/Documents/RL_MultiHeadAttention && python test_variable_mlp.py   cd /Users/amen/Documents/RL_MultiHeadAttention && python test_variable_mlp.py
zsh: command not found: python
amen@amens-Laptop RL_MultiHeadAttention % python3 train_simple_mlp.py --algorithm TD3 --mlp-size heavy --timesteps 1000     python3 train_simple_mlp.py --algorithm TD3 --mlp-size heavy --timesteps 1000
Traceback (most recent call last):
  File "/Users/amen/Documents/RL_MultiHeadAttention/train_simple_mlp.py", line 9, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
amen@amens-Laptop RL_MultiHeadAttention % source venv/bin/activate && python train_simple_mlp.py --algorithm TD3 --mlp-size heavy --timesteps 1000     source venv/bin/activate && python train_simple_mlp.py --algorithm TD3 --mlp-size heavy --timesteps 1000
✅ RESTORED: Increased features per coin from 14 to 28
✅ Total observation dimension: 84 (was 42)
⚠️ WARNING: Using placeholder min/max values for observation space. These must be updated.
🎯 USING TF-AGENTS EXACT TRAINING METHODOLOGY
🔧 This replicates the successful learning paradigm:
   • Collect 100 steps → Train 1 iteration → Repeat
   • Evaluate every 4 iterations (400 steps)
   • Exact hyperparameters from working implementation
🎯 CONSTRAINED REINFORCEMENT LEARNING ENABLED
   • Profile: balanced
   • Adaptive Lagrange multiplier λ for smoothness constraints
   • Automatic penalty adjustment based on constraint violations
🎯 CRL Profile 'balanced': Balance between stability and responsiveness
   Constraint threshold: 0.050
   Initial λ: 0.100
   λ learning rate: 1.0e-03
   • Balance between stability and responsiveness
💰 REWARD SYSTEM: TRANSACTION_COST
   • Net Return = Profit - Transaction Costs
   • Transaction Cost Rate: 0.200 (20.0%)
   • Agent will learn to minimize unnecessary trading
💾 Models and checkpoints will be saved in: ./models/TD3_tf_agents_style_heavy_TRANSACTION_COST_variable_CRL_balanced_20250809_032856
🎯 TF-AGENTS EXACT REPLICATION WITH CONSTRAINED RL
📊 Reward Type: TRANSACTION_COST (Net Return = Profit - Transaction Costs)
📊 Transaction Cost Rate: 0.200 (20.0%)
🎯 CRL Profile 'balanced': Balance between stability and responsiveness
   Constraint threshold: 0.050
   Initial λ: 0.100
   λ learning rate: 1.0e-03
🎯 CRL Configuration:
   Profile: balanced
   Use CRL: True
   Constraint threshold: 0.0500
   Initial λ: 0.1000
   λ learning rate: 1.0e-03
📊 Iterations: 1000
📊 Steps per iteration: 100
📊 Total steps: 100000
📊 Evaluation every 4 iterations (400 steps)
📊 Episodes per evaluation: 4
🎯 Creating environments with reward_type: TRANSACTION_COST
🔄 Portfolio mode: Variable
🎯 Portfolio Environment initialized with reward_type: TRANSACTION_COST
🔧 Variable portfolio setup:
   Action space: Box(0.0, 1.0, (4,), float64)
   Observation space: Dict with 84 features + 3 mask
🎲 Episode coins: ['LTC', 'STR', 'DASH'] (3 coins)
💰 Randomized initial allocation (variable): ['8.22%', '52.73%', '23.06%', '15.99%']
🎯 Portfolio Environment initialized with reward_type: TRANSACTION_COST
🔧 Variable portfolio setup:
   Action space: Box(0.0, 1.0, (4,), float64)
   Observation space: Dict with 84 features + 3 mask
🎲 Episode coins: ['STR', 'DASH', 'LTC'] (3 coins)
💰 Randomized initial allocation (variable): ['34.39%', '6.14%', '35.82%', '23.66%']
🔧 Step 1.1: Setting up SB3 logger for CSV generation...
Logging to ./logs/TD3_tf_agents_style_heavy_TRANSACTION_COST_variable_CRL_balanced_20250809_032856
✅ SB3 logger configured with CSV output to: ./logs/TD3_tf_agents_style_heavy_TRANSACTION_COST_variable_CRL_balanced_20250809_032856
⚡ Using OrnsteinUhlenbeckActionNoise (sigma=0.2, theta=0.15) for better exploration.
🔧 Using separate, decaying learning rates for actor and critic:
   - Actor LR: Starts at 1e-4, decays to 5e-5 over 90% of training.
   - Critic LR: Starts at 5e-4, decays to 1e-4 over 90% of training.
🎯 CRL Profile 'balanced': Balance between stability and responsiveness
   Constraint threshold: 0.050
   Initial λ: 0.100
   λ learning rate: 1.0e-03
Using cpu device
Wrapping the env in a DummyVecEnv.
🎯 CRL Mode Enabled:
   Constraint threshold (d): 0.0500
   Initial λ: 0.1000
   λ learning rate: 0.0010
🔧 Overriding optimizers with custom learning rates: Actor=0.0001, Critic=0.001
✅ CRL Mode: Using adaptive Lagrange multiplier for smoothness constraints
✅ Constraint: E[action_change²] ≤ 0.0500
✅ STABILITY FIX: Using Huber loss for the critic.
✅ STABILITY FIX: Using constant learning rate for direct TF-Agents replication
🔧 Step 1.2: Attaching SB3 logger to model...
✅ Logger attached to model - CSV logging now enabled
🔧 Step 2: Setting up SB3 callbacks for TF-Agents style training...
✅ Step 2.2: PlotGeneratorCallback configured (generate plots every 400 steps)
🧮 Initial evaluation...
🎲 Episode coins: ['LTC', 'STR', 'DASH'] (3 coins)
💰 Randomized initial allocation (variable): ['18.82%', '0.32%', '53.55%', '27.31%']
