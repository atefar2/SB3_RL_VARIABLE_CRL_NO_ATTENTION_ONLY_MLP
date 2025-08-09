💡 Recommended First Training Run
python train_with_attention.py --algorithm PPO --attention coin_attention --config medium --timesteps 50000

# Activate virtual environment
source .venv/bin/activate
python train_variable_attention.py --algorithm PPO --attention variable_coin_attention --config heavy --timesteps 200000 --envs 4 --variable

# --- Recommended First Training Run ---
# A good starting point to verify the setup works.
echo "Running initial PPO training run..."
python train_with_attention.py --algorithm PPO --attention coin_attention --config medium --timesteps 50000 --variable

# === SIMPLE MLP TRAINING (NO ATTENTION) ===
# ✅ RECOMMENDED: Use train_simple_mlp.py with clear --mlp-size parameter
python train_simple_mlp.py \
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 10000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --use-fixed-portfolio


# === VARIABLE PORTFOLIO MLP TRAINING ===
# FOR VARIABLE MLP, USE USE_VARIABLE_PORTFOLIO

python train_simple_mlp.py \
    --algorithm TD3 \
    --mlp-size heavy \
    --timesteps 10000 \
    --reward-type STRUCTURED_CREDIT \
    --crl-profile adaptive \
    --use-variable-portfolio
    


# ------------------------------------------------------------

# Train with CRL (adaptive penalty)
python train_simple_mlp.py --algorithm TD3 --crl-profile balanced

# Train with different profiles
python train_simple_mlp.py --algorithm TD3 --crl-profile conservative
python train_simple_mlp.py --algorithm TD3 --crl-profile aggressive

# Train with CRL (adaptive)
python train_simple_mlp.py --algorithm TD3 --crl-profile adaptive

# Disable CRL (use fixed penalty)
python train_simple_mlp.py --algorithm TD3 --disable-crl

# Test the integration
python test_crl_integration.py

# Profile |	Constraint Threshold |	Description
# conservative | 0.01 | Minimal trading, very stable
# balanced | 0.05 | Balance stability and responsiveness
# aggressive | 0.15 | Allow quick market responses
# adaptive | 0.03 | Start conservative, adapt over time

# ------------------------------------------------------------

# Test MLP functionality (recommended first step)
python train_simple_mlp.py --test

# Main MLP training with clear parameter naming # light is 256, 128
python train_simple_mlp.py --algorithm DDPG --mlp-size light --timesteps 10000

# heavy is 1024, 512, 256
python train_simple_mlp.py --algorithm DDPG --mlp-size heavy --timesteps 100000

# Different MLP sizes (architectures)
python train_simple_mlp.py --algorithm PPO --mlp-size light --timesteps 50000    # Small: [256, 128] → 128 features
python train_simple_mlp.py --algorithm PPO --mlp-size heavy --timesteps 100000   # Large: [1024, 512, 256] → 512 features

# MLP with different algorithms (NO ATTENTION)
python train_simple_mlp.py --algorithm DDPG --mlp-size medium --timesteps 100000
python train_simple_mlp.py --algorithm SAC --mlp-size medium --timesteps 100000



# Compare all algorithms with MLP
python train_simple_mlp.py --compare
# Quick test (600 steps) - recommended for verification
python train_simple_mlp.py --test-issue3
# Full training with model saving
python train_simple_mlp.py --tf-agents-style --algorithm DDPG --mlp-size heavy

# Alternative approach (confusing naming - "config" refers to MLP architecture when using --attention mlp)
# python train_with_attention.py --algorithm PPO --attention mlp --config medium --timesteps 100000

# fixed coin attention no variable
python train_variable_attention.py \
    --algorithm DDPG \
    --attention coin_attention \
    --config heavy \
    --timesteps 200000 \
    --envs 1 \
    --fixed

# --- Production Model Training (PPO) ---
# Example of training a more robust model with more steps and parallel environments.
echo "Starting production PPO training..."
python train_variable_attention.py \
    --algorithm PPO \
    --attention variable_coin_attention \
    --config heavy \
    --timesteps 200000 \
    --envs 4 \
    --variable


# --- TD3 Training Example ---
# TD3 is an off-policy algorithm suitable for continuous action spaces.
echo "Starting TD3 training..."
python train_variable_attention.py \
    --algorithm TD3 \
    --attention variable_coin_attention \
    --config heavy \
    --timesteps 10000 \
    --envs 1 \
    --variable


# --- DDPG Training Example ---
# DDPG is another off-policy algorithm, a precursor to TD3.
echo "Starting DDPG training..."
python train_variable_attention.py \
    --algorithm DDPG \
    --attention variable_coin_attention \
    --config heavy \
    --timesteps 200000 \
    --envs 1 \
    --variable


# --- Other Alternative Configurations ---
echo "Running other configurations..."
python train_variable_attention.py --algorithm SAC --config heavy
python train_variable_attention.py --algorithm DDPG --config medium
python train_variable_attention.py --attention coin_attention --config light


# # Quick test training (5K steps)
# python train_with_attention.py --algorithm PPO --timesteps 5000

# # Full training with coin attention (recommended)
# python train_with_attention.py --algorithm PPO --attention coin_attention --config medium --timesteps 100000

# # Train with SAC (continuous control algorithm)
# python train_with_attention.py --algorithm SAC --attention coin_attention --config heavy --timesteps 200000

# # Compare different algorithms
# python train_with_attention.py --compare



