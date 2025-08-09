#!/usr/bin/env python3
"""
Enhanced training script for variable portfolio allocation with attention mechanisms.
Supports dynamic portfolio sizes and advanced attention-based policies.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Tuple, List
from tqdm import tqdm

# Stable Baselines 3 imports
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# Local imports
from enviorment import PortfolioEnv
from attention_policy import create_attention_policy, ATTENTION_CONFIGS
import config


class EnhancedTrainingCallback(BaseCallback):
    """
    Enhanced callback that provides TF-Agents style training with:
    - Periodic evaluation and metrics display
    - Model saving every N steps
    - Best model tracking
    - Training history and visualization
    - Colored output and HTML logging
    """

    def __init__(
        self, 
        eval_env,
        eval_freq: int = 500,  # Lower frequency for shorter runs
        log_freq: int = 50,    # More frequent logging
        model_save_freq: int = 12,  # Save every 12 steps as requested
        n_eval_episodes: int = 5,   # Fewer episodes for faster evaluation
        model_save_dir: str = "./models/",
        log_dir: str = "./logs/",
        model_name: str = "model",
        verbose: int = 1,
        use_variable_portfolio: bool = True  # Add this parameter
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.model_save_freq = model_save_freq
        self.n_eval_episodes = n_eval_episodes
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        self.model_name = model_name
        self.use_variable_portfolio = use_variable_portfolio  # Store the parameter
        
        # Training history tracking
        self.training_history = {
            'steps': [],
            'returns': [],
            'eval_rewards': [],
            'losses': [],
            'portfolio_sizes': [],
            'episode_lengths': []
        }
        
        # Best model tracking
        self.best_mean_reward = -np.inf
        self.best_model_path = None
        
        # Create directories
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup HTML logging
        self.html_log_path = os.path.join(self.log_dir, f"{self.model_name}_training_log.html")
        self._init_html_log()
        
    def _init_html_log(self):
        """Initialize HTML log file."""
        html_header = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Log - {}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .step {{ background-color: #f0f0f0; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .eval {{ background-color: #e6f3ff; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .best {{ background-color: #e6ffe6; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .loss {{ color: #ff6600; font-weight: bold; }}
                .reward {{ color: #0066cc; font-weight: bold; }}
                .best-reward {{ color: #009900; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Training Log: {}</h1>
            <p>Started: {}</p>
        """.format(self.model_name, self.model_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        with open(self.html_log_path, 'w') as f:
            f.write(html_header)
    
    def _colored_print(self, message: str, color: str = "white"):
        """Print colored output to terminal."""
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'reset': '\033[0m'
        }
        
        if color in colors:
            print(f"{colors[color]}{message}{colors['reset']}")
        else:
            print(message)
    
    def _log_to_html(self, message: str, css_class: str = "step"):
        """Log message to HTML file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        html_entry = f'<div class="{css_class}">[{timestamp}] {message}</div>\n'
        
        with open(self.html_log_path, 'a') as f:
            f.write(html_entry)
    
    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model and return metrics."""
        episode_rewards = []
        episode_lengths = []
        portfolio_sizes = []
        
        # Create a simple non-vectorized environment for evaluation to avoid Dict observation issues
        from enviorment import PortfolioEnv
        simple_eval_env = PortfolioEnv(use_variable_portfolio=self.use_variable_portfolio)
        
        try:
            for episode in range(self.n_eval_episodes):
                obs, _ = simple_eval_env.reset()
                episode_reward = 0.0
                episode_length = 0
                terminated = truncated = False
                
                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    next_obs, reward, terminated, truncated, info = simple_eval_env.step(action)
                    
                    obs = next_obs
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Extract portfolio size if available
                if isinstance(info, dict) and 'episode_coins' in info:
                    portfolio_sizes.append(len(info['episode_coins']))
                else:
                    portfolio_sizes.append(0)  # Default if not available
        
        finally:
            simple_eval_env.close()
        
        # Calculate metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        mean_portfolio_size = np.mean(portfolio_sizes) if portfolio_sizes else 0
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'mean_portfolio_size': mean_portfolio_size,
            'episode_rewards': episode_rewards
        }
    
    def _save_model(self, step: int, is_best: bool = False):
        """Save model with step number."""
        if is_best:
            algorithm = self.model_name.split('_')[0]
            model_path = os.path.join(self.model_save_dir, f"{algorithm}_best_model.zip")
            self.best_model_path = model_path
        else:
            model_path = os.path.join(self.model_save_dir, f"{self.model_name}_step_{step}.zip")
        
        self.model.save(model_path)
        return model_path
    
    def _save_training_history(self):
        """Save training history to CSV and generate plots."""
        # Save CSV - handle mismatched array lengths
        try:
            # Ensure all arrays have the same length by padding with None/NaN
            max_length = max(len(v) for v in self.training_history.values() if isinstance(v, list))
            
            # Pad shorter arrays
            padded_history = {}
            for key, values in self.training_history.items():
                if isinstance(values, list):
                    if len(values) < max_length:
                        # Pad with NaN for numeric data, None for others
                        if key in ['eval_rewards', 'episode_lengths', 'portfolio_sizes', 'losses']:
                            padded_values = values + [np.nan] * (max_length - len(values))
                        else:
                            padded_values = values + [None] * (max_length - len(values))
                        padded_history[key] = padded_values
                    else:
                        padded_history[key] = values
                else:
                    padded_history[key] = values
            
            df = pd.DataFrame(padded_history)
            csv_path = os.path.join(self.log_dir, f"{self.model_name}_training_history.csv")
            df.to_csv(csv_path, index=False)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save training history CSV: {e}")
            # Save what we can
            try:
                # Just save the basic metrics that should be available
                basic_history = {
                    'steps': self.training_history.get('steps', []),
                    'eval_rewards': self.training_history.get('eval_rewards', []),
                }
                df = pd.DataFrame(basic_history)
                csv_path = os.path.join(self.log_dir, f"{self.model_name}_training_history.csv")
                df.to_csv(csv_path, index=False)
            except:
                pass
        
        # Generate plots
        if len(self.training_history['steps']) > 1:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Training Returns
            plt.subplot(2, 3, 1)
            plt.plot(self.training_history['steps'], self.training_history['eval_rewards'], 'b-', linewidth=2)
            plt.title('Average Return During Training')
            plt.xlabel('Training Steps')
            plt.ylabel('Average Return')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Episode Lengths
            plt.subplot(2, 3, 2)
            plt.plot(self.training_history['steps'], self.training_history['episode_lengths'], 'g-', linewidth=2)
            plt.title('Average Episode Length')
            plt.xlabel('Training Steps')
            plt.ylabel('Episode Length')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Portfolio Sizes
            plt.subplot(2, 3, 3)
            plt.plot(self.training_history['steps'], self.training_history['portfolio_sizes'], 'r-', linewidth=2)
            plt.title('Average Portfolio Size')
            plt.xlabel('Training Steps')
            plt.ylabel('Portfolio Size')
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Training Losses (if available)
            if self.training_history['losses']:
                plt.subplot(2, 3, 4)
                plt.plot(self.training_history['steps'][:len(self.training_history['losses'])], 
                        self.training_history['losses'], 'orange', linewidth=2)
                plt.title('Training Loss')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
            
            # Plot 5: Reward Distribution (last evaluation)
            if self.training_history['returns']:
                plt.subplot(2, 3, 5)
                plt.hist(self.training_history['returns'][-1], bins=20, alpha=0.7, color='skyblue')
                plt.title('Last Evaluation Reward Distribution')
                plt.xlabel('Episode Reward')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
            
            # Plot 6: Cumulative Best Reward
            plt.subplot(2, 3, 6)
            cumulative_best = []
            current_best = -np.inf
            for reward in self.training_history['eval_rewards']:
                if reward > current_best and not np.isnan(reward):
                    current_best = reward
                cumulative_best.append(current_best)
            plt.plot(self.training_history['steps'], cumulative_best, 'purple', linewidth=2)
            plt.title('Best Reward Over Time')
            plt.xlabel('Training Steps')
            plt.ylabel('Best Average Return')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.log_dir, f"{self.model_name}_training_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._colored_print(f"📊 Training plots saved to: {plot_path}", "cyan")
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        
        # Note: CSV dumping is now handled by ConsistentCSVCallback to avoid misalignment
        # We'll only handle our custom logging here
        
        # Capture training losses ONLY at evaluation frequency to avoid mismatched arrays
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            # Try to get training loss from logger
            loss_value = None
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                try:
                    # Get the most recent loss value
                    if hasattr(self.model.logger, 'name_to_value'):
                        # Try different loss keys for different algorithms
                        loss_keys = ['train/loss', 'train/actor_loss', 'train/critic_loss', 'train/value_loss', 'train/policy_loss']
                        for key in loss_keys:
                            if key in self.model.logger.name_to_value:
                                loss_value = self.model.logger.name_to_value[key]
                                break
                except:
                    pass
            
            # Store loss value for training history (only at evaluation frequency)
            if loss_value is not None:
                self.training_history['losses'].append(loss_value)
            else:
                # Try to get loss from model attributes (for TD3/DDPG)
                try:
                    if hasattr(self.model, 'actor_loss') and self.model.actor_loss is not None:
                        self.training_history['losses'].append(float(self.model.actor_loss))
                    elif hasattr(self.model, 'critic_loss') and self.model.critic_loss is not None:
                        self.training_history['losses'].append(float(self.model.critic_loss))
                    else:
                        # If no loss available, append None to maintain array alignment
                        self.training_history['losses'].append(None)
                except:
                    self.training_history['losses'].append(None)
        
        # Log training progress periodically (but don't store losses here)
        if self.n_calls % self.log_freq == 0:
            # Try to get current loss for display only
            current_loss = None
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                try:
                    if hasattr(self.model.logger, 'name_to_value'):
                        loss_keys = ['train/loss', 'train/actor_loss', 'train/critic_loss', 'train/value_loss', 'train/policy_loss']
                        for key in loss_keys:
                            if key in self.model.logger.name_to_value:
                                current_loss = self.model.logger.name_to_value[key]
                                break
                except:
                    pass
            
            message = f"🔄 Step {self.n_calls}: Training in progress..."
            if current_loss is not None:
                message += f" | Loss: {current_loss:.6f}"
            
            self._colored_print(message, "yellow")
            self._log_to_html(message, "step")
        
        # Enhanced evaluation display (runs after standard SB3 evaluation)
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            # Let the standard SB3 evaluation run first, then add our enhancements
            
            try:
                # Get evaluation results from SB3 logger if available
                mean_reward = None
                std_reward = None
                mean_length = None
                
                if hasattr(self.model, 'logger') and self.model.logger is not None:
                    try:
                        if hasattr(self.model.logger, 'name_to_value'):
                            mean_reward = self.model.logger.name_to_value.get('eval/mean_reward')
                            std_reward = self.model.logger.name_to_value.get('eval/std_reward')
                            mean_length = self.model.logger.name_to_value.get('eval/mean_ep_length')
                    except:
                        pass
                
                # If SB3 evaluation data not available, run our own
                if mean_reward is None:
                    self._colored_print(f"🧪 Running additional evaluation at step {self.n_calls}...", "blue")
                    eval_results = self._evaluate_model()
                    mean_reward = eval_results['mean_reward']
                    std_reward = eval_results['std_reward']
                    mean_length = eval_results['mean_length']
                    mean_portfolio_size = eval_results['mean_portfolio_size']
                else:
                    # Get portfolio size from our evaluation
                    eval_results = self._evaluate_model()
                    mean_portfolio_size = eval_results['mean_portfolio_size']
                
                # Store in training history
                self.training_history['steps'].append(self.n_calls)
                self.training_history['eval_rewards'].append(mean_reward)
                self.training_history['episode_lengths'].append(mean_length)
                self.training_history['portfolio_sizes'].append(mean_portfolio_size)
                if 'episode_rewards' in eval_results:
                    self.training_history['returns'].append(eval_results['episode_rewards'])
                
                # Save training history immediately after each evaluation
                self._save_training_history()
                
                # Display results (TF-Agents style)
                message = f"Step {self.n_calls}: Average Return = {mean_reward:.4f} ± {std_reward:.4f}"
                self._colored_print(message, "blue")
                
                # Additional metrics
                metrics_msg = f"   📊 Episode Length: {mean_length:.1f} | Portfolio Size: {mean_portfolio_size:.1f}"
                self._colored_print(metrics_msg, "cyan")
                
                # HTML logging
                html_msg = f"<span class='reward'>Average Return: {mean_reward:.4f}</span> ± {std_reward:.4f} | Length: {mean_length:.1f} | Portfolio: {mean_portfolio_size:.1f}"
                self._log_to_html(html_msg, "eval")
                
                # Check if this is the best model (in addition to SB3's best model saving)
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    
                    saved_path = self._save_model(self.n_calls, is_best=True)
                    best_message = f"🏆 NEW BEST MODEL! Reward: {mean_reward:.4f}. Saved to {saved_path}"
                    self._colored_print(best_message, "green")
                    
                    html_best_msg = f"<span class='best-reward'>NEW BEST MODEL! Reward: {mean_reward:.4f}</span>"
                    self._log_to_html(html_best_msg, "best")
                    
            except Exception as e:
                error_msg = f"❌ Enhanced evaluation failed at step {self.n_calls}: {str(e)}"
                self._colored_print(error_msg, "red")
                self._log_to_html(error_msg, "step")
        
        # Save model periodically
        if self.n_calls % self.model_save_freq == 0 and self.n_calls > 0:
            model_path = self._save_model(self.n_calls)
            save_message = f"💾 Model saved at step {self.n_calls}: {model_path}"
            self._colored_print(save_message, "magenta")
            self._log_to_html(save_message, "step")
        
        return True
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        self._colored_print("🎉 Training completed!", "green")
        
        # Save final training history and plots
        self._save_training_history()
        
        # Close HTML log
        with open(self.html_log_path, 'a') as f:
            f.write(f'<p>Training completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
            f.write('</body></html>')
        
        # Final summary
        if self.training_history['eval_rewards']:
            final_reward = self.training_history['eval_rewards'][-1]
            summary_msg = f"📊 Final Summary:\n"
            summary_msg += f"   Final Average Return: {final_reward:.4f}\n"
            summary_msg += f"   Best Average Return: {self.best_mean_reward:.4f}\n"
            summary_msg += f"   Best Model Path: {self.best_model_path}\n"
            summary_msg += f"   Training History: {len(self.training_history['steps'])} evaluations"
            
            self._colored_print(summary_msg, "green")


class VariablePortfolioProgressCallback(BaseCallback):
    """
    Enhanced callback for monitoring variable portfolio training progress.
    Tracks metrics across different portfolio sizes and attention patterns.
    """

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.portfolio_stats = {
            'sizes': [],
            'rewards': [],
            'allocations': [],
            'attention_weights': []
        }
        
    def _on_step(self) -> bool:
        # Only log portfolio metrics at the same frequency as other metrics
        if self.n_calls % self.check_freq == 0:
            # Collect portfolio size distribution
            if hasattr(self.training_env, 'get_attr'):
                try:
                    episode_coins = self.training_env.get_attr('episode_coins')[0]
                    portfolio_size = len(episode_coins)
                    self.portfolio_stats['sizes'].append(portfolio_size)
                    
                    # Log current episode info (print only, not to CSV)
                    if self.verbose > 0:
                        print(f"Step {self.n_calls}: Portfolio size = {portfolio_size}, "
                              f"Coins = {episode_coins}")
                        
                        # Calculate portfolio size distribution
                        if len(self.portfolio_stats['sizes']) >= 10:
                            recent_sizes = self.portfolio_stats['sizes'][-100:]
                            size_dist = pd.Series(recent_sizes).value_counts().sort_index()
                            print(f"Recent portfolio size distribution: {dict(size_dist)}")
                    
                    # Log to CSV only at the same time as other metrics
                    if len(self.portfolio_stats['sizes']) > 0:
                        recent_sizes = self.portfolio_stats['sizes'][-100:]
                        avg_size = np.mean(recent_sizes)
                        
                        # Record portfolio metrics (will be written with other metrics)
                        self.logger.record("portfolio/avg_size", avg_size)
                        self.logger.record("portfolio/min_size", np.min(recent_sizes))
                        self.logger.record("portfolio/max_size", np.max(recent_sizes))
                            
                except Exception as e:
                    if self.verbose > 1:
                        print(f"Could not retrieve portfolio info: {e}")

        return True

    def _on_rollout_end(self) -> None:
        """Log rollout summary - but don't force CSV dump here."""
        # This method is called by PPO but not by TD3/DDPG
        # We'll handle portfolio logging in _on_step to ensure consistency
        pass


class PortfolioEvalCallback(EvalCallback):
    """
    Enhanced evaluation callback for variable portfolio environments.
    Tests performance across different portfolio sizes.
    """
    
    def __init__(self, eval_env, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.portfolio_performance = {}
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # Additional evaluation for different portfolio sizes
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._evaluate_portfolio_sizes()
            
        return result
    
    def _evaluate_portfolio_sizes(self):
        """Evaluate performance for each possible portfolio size."""
        if hasattr(self.eval_env, 'envs'):
            env = self.eval_env.envs[0]
        else:
            env = self.eval_env
            
        # Test with different forced portfolio sizes
        for size in range(1, min(4, config.MAX_COINS + 1)):  # Test sizes 1-3
            try:
                # Force specific portfolio size for evaluation
                original_range = config.COINS_PER_EPISODE_RANGE
                config.COINS_PER_EPISODE_RANGE = (size, size)
                
                rewards = []
                for _ in range(5):  # 5 episodes per size
                    obs, _ = env.reset()
                    episode_reward = 0
                    done = False
                    steps = 0
                    
                    while not done and steps < 500:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        done = terminated or truncated
                        steps += 1
                    
                    rewards.append(episode_reward)
                
                avg_reward = np.mean(rewards)
                self.portfolio_performance[f"size_{size}"] = avg_reward
                self.logger.record(f"eval/portfolio_size_{size}_reward", avg_reward)
                
                # Restore original range
                config.COINS_PER_EPISODE_RANGE = original_range
                
            except Exception as e:
                print(f"Error evaluating portfolio size {size}: {e}")


def create_variable_env(
    env_id: str = "VariablePortfolio",
    use_variable_portfolio: bool = True,
    seed: int = None
) -> PortfolioEnv:
    """Create a variable portfolio environment with proper configuration."""
    
    def _init():
        env = PortfolioEnv(use_variable_portfolio=use_variable_portfolio)
        if seed is not None:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            # Set numpy seed for the environment
            np.random.seed(seed)
        env = Monitor(env)
        return env
    
    return _init


def setup_training_environment(
    n_envs: int = 4,
    use_variable_portfolio: bool = True,
    use_multiprocessing: bool = True
) -> DummyVecEnv:
    """Setup vectorized training environment."""
    
    if use_multiprocessing and n_envs > 1:
        env_fns = [create_variable_env(
            use_variable_portfolio=use_variable_portfolio,
            seed=i
        ) for i in range(n_envs)]
        return SubprocVecEnv(env_fns)
    else:
        env_fns = [create_variable_env(
            use_variable_portfolio=use_variable_portfolio,
            seed=i
        ) for i in range(n_envs)]
        return DummyVecEnv(env_fns)


def train_variable_attention_model(
    algorithm: str = "PPO",
    attention_type: str = "variable_coin_attention",
    attention_config: str = "medium",
    total_timesteps: int = 200000,
    use_variable_portfolio: bool = True,
    n_envs: int = 4,
    eval_freq: int = 10000,
    model_save_freq: int = 12,
    model_name: str = None,
    log_dir: str = "./logs/",
    model_save_dir: str = "./models/",
    verbose: int = 1
) -> Tuple[Any, str]:
    """
    Train a variable portfolio attention model.
    
    Args:
        algorithm: "PPO", "SAC", or "TD3"
        attention_type: "variable_coin_attention", "coin_attention", or "multihead"
        attention_config: "light", "medium", or "heavy"
        total_timesteps: Number of training steps
        use_variable_portfolio: Whether to use variable portfolio sizes
        n_envs: Number of parallel environments
        eval_freq: Frequency of evaluation
        save_freq: Frequency of model saving
        model_name: Custom model name
        log_dir: Directory for logs
        model_save_dir: Directory for saving models
        verbose: Verbosity level
    
    Returns:
        Tuple of (trained_model, model_path)
    """
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_name is None:
        portfolio_type = "variable" if use_variable_portfolio else "fixed"
        model_name = f"{algorithm}_{attention_type}_{attention_config}_{portfolio_type}_{timestamp}"
    
    log_path = os.path.join(log_dir, model_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Setup logging
    logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    
    # Clean up any existing progress.csv to ensure fresh start
    progress_csv_path = os.path.join(log_path, "progress.csv")
    if os.path.exists(progress_csv_path):
        try:
            os.remove(progress_csv_path)
            print(f"🧹 Cleaned up existing progress.csv for fresh start")
        except:
            pass
    
    print(f"🚀 Starting Variable Portfolio Training")
    print(f"📊 Configuration:")
    print(f"   Algorithm: {algorithm}")
    print(f"   Attention Type: {attention_type}")
    print(f"   Attention Config: {attention_config}")
    print(f"   Portfolio Type: {'Variable' if use_variable_portfolio else 'Fixed'}")
    print(f"   Total Timesteps: {total_timesteps:,}")
    print(f"   Parallel Environments: {n_envs}")
    print(f"   Model Name: {model_name}")
    print(f"   Log Directory: {log_path}")
    print(f"   Model Save Directory: {model_save_dir}")
    
    # Create environments
    train_env = setup_training_environment(
        n_envs=n_envs,
        use_variable_portfolio=use_variable_portfolio,
        use_multiprocessing=True
    )
    
    eval_env = setup_training_environment(
        n_envs=1,
        use_variable_portfolio=use_variable_portfolio,
        use_multiprocessing=False
    )
    
    # Get attention configuration
    if attention_config not in ATTENTION_CONFIGS:
        raise ValueError(f"Unknown attention config: {attention_config}")
    
    config_params = ATTENTION_CONFIGS[attention_config].copy()
    
    # Create policy
    PolicyClass = create_attention_policy(
        attention_type=attention_type,
        algorithm=algorithm,
        **config_params
    )
    
    # Algorithm-specific parameters
    if algorithm == "PPO":
        model_params = config.get_algorithm_config("PPO")
        ModelClass = PPO
        
    elif algorithm == "SAC":
        model_params = config.get_algorithm_config("SAC")
        ModelClass = SAC
        
    elif algorithm == "TD3":
        model_params = config.get_algorithm_config("TD3")
        ModelClass = TD3
        
    elif algorithm == "DDPG":
        model_params = config.get_algorithm_config("DDPG")
        ModelClass = DDPG
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create model
    print(f"🧠 Creating {algorithm} model with {attention_type} attention...")
    
    model = ModelClass(
        PolicyClass,
        train_env,
        verbose=verbose,
        tensorboard_log=log_path,  # Enable tensorboard logging
        **model_params
    )
    
    # Set custom logger
    model.set_logger(logger)
    
    # For TD3 and DDPG, ensure we log training metrics more frequently
    if algorithm in ["TD3", "DDPG"]:
        from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
        model.train_freq = TrainFreq(1, TrainFrequencyUnit.STEP)  # Train every step
        model.gradient_steps = 1  # One gradient step per training step
    
    # Adjust frequencies for short runs
    actual_eval_freq = min(eval_freq, max(100, total_timesteps // 10))  # At least 10 evaluations
    actual_log_freq = actual_eval_freq  # Sync logging with evaluation to avoid empty rows
    
    print(f"📊 Adjusted frequencies for {total_timesteps:,} timesteps:")
    print(f"   Evaluation frequency: {actual_eval_freq}")
    print(f"   Logging frequency: {actual_log_freq}")
    print(f"   Model save frequency: {model_save_freq}")
    
    # IMPORTANT: Use smaller evaluation frequency like successful implementation
    # The successful implementation evaluates every 4 training iterations
    # With collect_steps_per_iteration=100, that's every 400 steps
    # Let's use a similar pattern: evaluate every 500-1000 steps for better learning feedback
    if total_timesteps <= 50000:
        actual_eval_freq = 500  # More frequent evaluation for shorter runs
        actual_model_save_freq = 2500  # Save every 5 evaluations
    else:
        actual_eval_freq = 1000  # Standard frequency for longer runs
        actual_model_save_freq = 5000  # Save every 5 evaluations
        
    print(f"📊 Using evaluation frequency optimized for learning: {actual_eval_freq}")
    print(f"📊 Using model save frequency: {actual_model_save_freq}")
    print(f"   This matches the successful implementation's frequent evaluation pattern")
    
    # Setup standard SB3 evaluation callback first
    from stable_baselines3.common.callbacks import EvalCallback
    standard_eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=None,  # We use our EnhancedTrainingCallback for this
        log_path=log_path,
        eval_freq=actual_eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=verbose
    )
    
    # Add a custom callback to ensure consistent CSV logging
    class ConsistentCSVCallback(BaseCallback):
        def __init__(self, log_freq=100, verbose=0):
            super().__init__(verbose)
            self.log_freq = log_freq
            
        def _on_step(self) -> bool:
            # Only log at consistent intervals to avoid CSV misalignment
            if self.n_calls % self.log_freq == 0:
                try:
                    # Ensure all required columns have values
                    if hasattr(self.model, 'logger') and self.model.logger is not None:
                        # Fill in missing standard metrics with defaults
                        logger = self.model.logger
                        
                        # Time metrics (always available)
                        logger.record("time/total_timesteps", self.num_timesteps)
                        logger.record("time/time_elapsed", time.time() - getattr(self, '_start_time', time.time()))
                        
                        # Initialize start time if not set
                        if not hasattr(self, '_start_time'):
                            self._start_time = time.time()
                        
                        # Force dump to CSV
                        logger.dump(self.num_timesteps)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"CSV logging error: {e}")
            return True
    
    csv_callback = ConsistentCSVCallback(log_freq=actual_eval_freq, verbose=verbose)
    
    # Setup enhanced callback for TF-Agents style training
    enhanced_callback = EnhancedTrainingCallback(
        eval_env=eval_env,
        eval_freq=actual_eval_freq,
        log_freq=actual_log_freq,
        model_save_freq=actual_model_save_freq,  # Save model every N steps as requested
        n_eval_episodes=5,  # Fewer episodes for faster evaluation
        model_save_dir=model_save_dir,
        log_dir=log_path,
        model_name=model_name,
        verbose=verbose,
        use_variable_portfolio=use_variable_portfolio
    )
    
    # Keep the portfolio progress callback for additional portfolio-specific metrics
    progress_callback = VariablePortfolioProgressCallback(
        check_freq=actual_eval_freq,  # Sync with evaluation frequency
        verbose=verbose
    )
    
    # Combine all callbacks: standard SB3 eval + CSV consistency + enhanced + portfolio progress
    # All callbacks now log at the same frequency to maintain CSV structure
    callbacks = CallbackList([standard_eval_callback, csv_callback, enhanced_callback, progress_callback])
    
    # Train model
    print(f"🎯 Starting training for {total_timesteps:,} timesteps...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Save final model
        model_path = os.path.join(model_save_dir, f"{model_name}_final.zip")
        model.save(model_path)
        
        print(f"💾 Final model saved to: {model_path}")
        
        # Enhanced callback already handles best model saving and evaluation
        print(f"💾 Best model saved by enhanced callback during training")
        print(f"📊 Training history and plots saved in: {log_path}")
        print(f"🌐 HTML training log available at: {os.path.join(log_path, f'{model_name}_training_log.html')}")
        
        return model, model_path
        
    except KeyboardInterrupt:
        print(f"⚠️ Training interrupted by user")
        model_path = os.path.join(model_save_dir, f"{model_name}_interrupted.zip")
        model.save(model_path)
        print(f"💾 Interrupted model saved to: {model_path}")
        return model, model_path
        
    finally:
        train_env.close()
        eval_env.close()


def evaluate_variable_model(
    model,
    eval_env,
    n_episodes: int = 10,
    deterministic: bool = True
) -> Dict[str, float]:
    """
    Evaluate a variable portfolio model across different portfolio sizes.
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    results = {
        'total_reward': [],
        'episode_length': [],
        'portfolio_sizes': [],
        'final_values': []
    }
    
    # Check if we have a vectorized environment
    is_vectorized = hasattr(eval_env, 'envs')
    
    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        
        # Extract observation from vectorized format if needed
        if is_vectorized:
            # For Dict observations in vectorized env, extract first element of each key
            if isinstance(obs, dict):
                obs = {key: value[0] for key, value in obs.items()}
            else:
                obs = obs[0]
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Handle vectorized environment outputs
            if is_vectorized:
                # Extract from vectorized environment
                if isinstance(next_obs, dict):
                    obs = {key: value[0] for key, value in next_obs.items()}
                else:
                    obs = next_obs[0]
                    
                if isinstance(reward, (list, tuple, np.ndarray)):
                    reward = reward[0]
                if isinstance(terminated, (list, tuple, np.ndarray)):
                    terminated = terminated[0]
                if isinstance(truncated, (list, tuple, np.ndarray)):
                    truncated = truncated[0]
                if isinstance(info, (list, tuple)):
                    info = info[0]
            else:
                obs = next_obs
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        results['total_reward'].append(episode_reward)
        results['episode_length'].append(episode_length)
        results['portfolio_sizes'].append(len(info.get('episode_coins', [])))
        results['final_values'].append(info.get('portfolio_value', 1.0))
    
    # Calculate summary statistics
    summary = {
        'mean_reward': np.mean(results['total_reward']),
        'std_reward': np.std(results['total_reward']),
        'mean_episode_length': np.mean(results['episode_length']),
        'mean_portfolio_size': np.mean(results['portfolio_sizes']),
        'mean_final_value': np.mean(results['final_values']),
        'sharpe_ratio': np.mean(results['total_reward']) / (np.std(results['total_reward']) + 1e-8)
    }
    
    return summary


def main():
    """Main training function with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Variable Portfolio Attention Models")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC", "TD3", "DDPG"])
    parser.add_argument("--attention", type=str, default="variable_coin_attention", 
                       choices=["variable_coin_attention", "coin_attention", "multihead"])
    parser.add_argument("--config", type=str, default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--variable", action="store_true", default=False)
    parser.add_argument("--fixed", dest="variable", action="store_false")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="./models/", help="Directory to save models")
    parser.add_argument("--model_save_freq", type=int, default=12, help="Save model every N steps")
    parser.add_argument("--eval_freq", type=int, default=1000, help="Evaluate model every N steps")
    parser.add_argument("--verbose", type=int, default=1)
    
    args = parser.parse_args()
    
    # Train model
    model, model_path = train_variable_attention_model(
        algorithm=args.algorithm,
        attention_type=args.attention,
        attention_config=args.config,
        total_timesteps=args.timesteps,
        use_variable_portfolio=args.variable,
        n_envs=args.envs,
        eval_freq=args.eval_freq,
        model_save_freq=args.model_save_freq,
        model_name=args.name,
        model_save_dir=args.model_dir,
        verbose=args.verbose
    )
    
    print(f"🎉 Training completed! Model saved at: {model_path}")


if __name__ == "__main__":
    main() 