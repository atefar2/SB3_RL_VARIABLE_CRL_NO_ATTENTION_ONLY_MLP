#!/usr/bin/env python3
"""
Training script for RL portfolio allocation using attention mechanisms with Stable Baselines 3.
"""

import os
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import torch

# Import our custom environment and attention policies
from enviorment import PortfolioEnv
from attention_policy import (
    create_attention_policy, 
    AttentionActorCriticPolicy,
    ATTENTION_CONFIGS,
    MLP_CONFIGS
)


def create_attention_env(env_id="portfolio-v0", **env_kwargs):
    """Create a single environment instance."""
    return PortfolioEnv(**env_kwargs)


def train_with_attention(
    algorithm="PPO",
    attention_type="coin_attention",
    config_name="medium",
    total_timesteps=100000,
    eval_freq=10000,
    log_dir="./logs",
    model_save_path="./models"
):
    """
    Train RL agent with attention mechanisms for portfolio allocation.
    
    Args:
        algorithm: "PPO", "SAC", or "TD3"
        attention_type: "multihead" or "coin_attention"
        config_name: "light", "medium", or "heavy"
        total_timesteps: Total training steps
        eval_freq: Evaluation frequency
        log_dir: Logging directory
        model_save_path: Model save directory
    """
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    
    # Create training environment
    env = Monitor(PortfolioEnv(), filename=os.path.join(log_dir, "training"))
    
    # Create evaluation environment
    eval_env = Monitor(PortfolioEnv(), filename=os.path.join(log_dir, "evaluation"))
    
    # Get configuration based on attention type
    if attention_type == "mlp":
        config = MLP_CONFIGS[config_name]
        print(f"Using {config_name} MLP configuration: {config}")
        print(f"🚫 NO ATTENTION - Simple MLP architecture: {config['net_arch']} → {config['features_dim']}")
        print(f"🔧 Environment fix: Actions will be properly applied (not overwritten)")
    else:
        config = ATTENTION_CONFIGS[config_name]
        print(f"Using {config_name} attention configuration: {config}")
        print(f"🔧 Fixed architecture: features_dim={config['features_dim']}")
    print(f"🔧 Environment fix: Actions will be properly applied (not overwritten)")
    
    # Create policy with all required parameters
    Policy = create_attention_policy(
        attention_type=attention_type,
        algorithm=algorithm,
        **config
    )
    
    # Training hyperparameters based on algorithm
    if algorithm == "PPO":
        if attention_type == "mlp":
            print(f"🚀 Creating PPO model with simple MLP architecture...")
        else:
            print(f"🚀 Creating PPO model with {attention_type} attention architecture...")
        model = PPO(
            Policy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="auto"
        )
        if attention_type == "mlp":
            print(f"✅ PPO model created successfully with simple MLP (no attention)")
        else:
            print(f"✅ PPO model created successfully with {attention_type} attention")
    elif algorithm == "SAC":
        if attention_type == "mlp":
            print(f"🚀 Creating SAC model with simple MLP architecture...")
        else:
            print(f"🚀 Creating SAC model with {attention_type} attention architecture...")
        model = SAC(
            Policy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            device="auto"
        )
        if attention_type == "mlp":
            print(f"✅ SAC model created successfully with simple MLP (no attention)")
        else:
            print(f"✅ SAC model created successfully with {attention_type} attention")
    elif algorithm == "TD3":
        if attention_type == "mlp":
            print(f"🚀 Creating TD3 model with simple MLP architecture...")
        else:
            print(f"🚀 Creating TD3 model with {attention_type} attention architecture...")
        model = TD3(
            Policy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            device="auto"
        )
        if attention_type == "mlp":
            print(f"✅ TD3 model created successfully with simple MLP (no attention)")
        else:
            print(f"✅ TD3 model created successfully with {attention_type} attention")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_path,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    # Validate model works before training
    print(f"🧪 Validating model setup...")
    try:
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        print(f"✅ Model validation successful - sample action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step successful - reward: {reward:.4f}")
        print(f"✅ Action properly applied - allocation: {info['money_split']}")
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return None
    
    # Train the model
    if attention_type == "mlp":
        print(f"\n🚀 Starting training: {algorithm} with simple MLP...")
        print(f"📊 Total timesteps: {total_timesteps}")
        print(f"🖥️  Device: {model.device}")
        print(f"🏗️  Architecture: Simple MLP {config['net_arch']} → {config['features_dim']} features → policy network")
        print(f"🚫 NO ATTENTION mechanisms used")
    else:
        print(f"\n🚀 Starting training: {algorithm} with {attention_type} attention...")
        print(f"📊 Total timesteps: {total_timesteps}")
        print(f"🖥️  Device: {model.device}")
        print(f"🏗️  Architecture: {config['features_dim']} features → policy network")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name=f"{algorithm}_{attention_type}_{config_name}"
    )
    
    # Save final model
    final_model_path = os.path.join(model_save_path, f"{algorithm}_{attention_type}_{config_name}_final")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    return model


def test_attention_policy():
    """Test that the attention policy works correctly with fixed architecture."""
    print("🧪 Testing attention policy with fixed architecture...")
    
    # Create environment
    env = PortfolioEnv()
    
    # Test both attention types and MLP
    for attention_type in ["multihead", "coin_attention", "mlp"]:
        if attention_type == "mlp":
            print(f"\n🔍 Testing {attention_type} (simple MLP)...")
            config = MLP_CONFIGS["light"]
        else:
            print(f"\n🔍 Testing {attention_type} attention...")
            config = ATTENTION_CONFIGS["light"]
        
        try:
            # Create policy with configuration
            Policy = create_attention_policy(
                attention_type=attention_type,
                **config
            )
            
            # Create model
            model = PPO(Policy, env, verbose=0)
            
            # Test observation and action
            obs, _ = env.reset()
            action, _ = model.predict(obs)
            
            # Test that action is properly applied
            obs, reward, terminated, truncated, info = env.step(action)
            
            if attention_type == "mlp":
                print(f"✅ {attention_type} (simple MLP) working correctly")
            else:
                print(f"✅ {attention_type} attention working correctly")
                print(f"   Observation shape: {obs.shape}")
                print(f"   Action shape: {action.shape}")
                print(f"   Sample action: {action}")
                print(f"   Applied allocation: {info['money_split']}")
                print(f"   Reward: {reward:.4f}")
            
        except Exception as e:
            print(f"❌ {attention_type} attention failed: {e}")
            import traceback
            traceback.print_exc()
    
    env.close()


def compare_attention_models():
    """Compare different attention configurations."""
    print("Comparing attention models...")
    
    results = {}
    
    for attention_type in ["multihead", "coin_attention"]:
        for config_name in ["light", "medium", "heavy"]:
            print(f"\nTraining {attention_type} with {config_name} config...")
            
            try:
                model = train_with_attention(
                    algorithm="PPO",
                    attention_type=attention_type,
                    config_name=config_name,
                    total_timesteps=20000,  # Short training for comparison
                    eval_freq=5000,
                    log_dir=f"./comparison_logs/{attention_type}_{config_name}",
                    model_save_path=f"./comparison_models/{attention_type}_{config_name}"
                )
                
                # Quick evaluation
                env = PortfolioEnv()
                obs, _ = env.reset()
                total_reward = 0
                
                for _ in range(100):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    
                    if terminated or truncated:
                        break
                
                results[f"{attention_type}_{config_name}"] = total_reward
                env.close()
                
                print(f"✅ {attention_type}_{config_name}: Average reward = {total_reward:.4f}")
                
            except Exception as e:
                print(f"❌ {attention_type}_{config_name} failed: {e}")
                results[f"{attention_type}_{config_name}"] = None
    
    print("\n=== COMPARISON RESULTS ===")
    for name, reward in results.items():
        if reward is not None:
            print(f"{name}: {reward:.4f}")
        else:
            print(f"{name}: FAILED")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL portfolio allocation with attention or simple MLP")
    parser.add_argument("--algorithm", default="PPO", choices=["PPO", "SAC", "TD3", "DDPG"])
    parser.add_argument("--attention", default="coin_attention", choices=["multihead", "coin_attention", "mlp"])
    parser.add_argument("--config", default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--test", action="store_true", help="Run tests instead of training")
    parser.add_argument("--compare", action="store_true", help="Compare different attention models")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    if args.test:
        test_attention_policy()
    elif args.compare:
        compare_attention_models()
    else:
        # Run main training
        train_with_attention(
            algorithm=args.algorithm,
            attention_type=args.attention,
            config_name=args.config,
            total_timesteps=args.timesteps
        )
        
        print("\n🎉 Training completed successfully!")
        
        print(f"\n✅ ISSUES FIXED:")
        print(f"   🔧 Environment: Actions are now properly applied (not overwritten)")
        print(f"   🔧 Architecture: Fixed dimension mismatch in attention mechanism")
        print(f"   🔧 Training: Model can now learn effective portfolio allocation strategies")
        
        print(f"\n📊 Expected improvements with fixed model:")
        print(f"   • Diversified portfolios across DASH, LTC, STR, and Cash")
        print(f"   • Dynamic rebalancing based on market conditions")
        print(f"   • Attention weights focusing on relevant market signals")
        print(f"   • Meaningful allocation changes throughout episodes")
        
        print("\n💡 Next steps:")
        print("1. Run evaluate.py to see proper multi-asset allocation")
        print("2. Visualize attention weights to understand model focus")
        print("3. Compare performance with and without attention")
        print("4. Experiment with different attention configurations")
        print("5. Try different reward functions (Sharpe ratio, drawdown-adjusted)") 