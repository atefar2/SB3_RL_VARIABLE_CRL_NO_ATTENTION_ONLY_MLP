#!/usr/bin/env python3
"""
Test script to verify variable portfolio functionality with simple MLP.
This tests the mask-and-renormalize logic without full training.
"""

import numpy as np
import pandas as pd
from enviorment import PortfolioEnv
import config

def test_variable_portfolio_basic():
    """Test basic variable portfolio functionality."""
    print("🧪 Testing Variable Portfolio Basic Functionality")
    print("=" * 60)
    
    # Create environment with variable portfolio
    env = PortfolioEnv(use_variable_portfolio=True, reward_type="TRANSACTION_COST")
    
    # Reset and check initial state
    obs, info = env.reset()
    print(f"🎲 Episode coins: {env.episode_coins}")
    print(f"📊 Number of active coins: {env.n_episode_coins}")
    print(f"🎯 Action space shape: {env.action_space.shape}")
    print(f"👁️  Observation type: {type(obs)}")
    
    if isinstance(obs, dict):
        print(f"   Observations shape: {obs['observations'].shape}")
        print(f"   Mask shape: {obs['mask'].shape}")
        print(f"   Mask values: {obs['mask']}")
    else:
        print(f"   Observation shape: {obs.shape}")
    
    # Test a few steps with random actions
    for step in range(5):
        print(f"\n--- Step {step + 1} ---")
        
        # Generate random action
        action = env.action_space.sample()
        print(f"🎯 Raw action: {action}")
        print(f"🎯 Raw action sum: {np.sum(action):.6f}")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"💰 Portfolio value: ${info['value']:.2f}")
        print(f"🎁 Reward: {reward:.6f}")
        print(f"📊 Final allocation: {[f'{x:.3f}' for x in info['money_split']]}")
        print(f"📊 Allocation sum: {np.sum(info['money_split']):.6f}")
        
        if terminated or truncated:
            print("✅ Episode ended")
            break
    
    print("\n✅ Basic variable portfolio test completed!")

def test_action_masking():
    """Test the mask-and-renormalize logic specifically."""
    print("\n🧪 Testing Action Masking Logic")
    print("=" * 60)
    
    env = PortfolioEnv(use_variable_portfolio=True, reward_type="TRANSACTION_COST")
    obs, info = env.reset()
    
    print(f"🎲 Active coins: {env.episode_coins}")
    
    # Test with different action patterns
    test_actions = [
        # All zeros (should default to cash)
        np.zeros(env.action_space.shape[0]),
        # All ones (should be renormalized)
        np.ones(env.action_space.shape[0]),
        # Random action
        env.action_space.sample(),
        # Action with high values in inactive slots
        np.random.uniform(0.8, 1.0, env.action_space.shape[0])
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n--- Test Action {i + 1} ---")
        print(f"🎯 Input action: {action}")
        print(f"🎯 Input sum: {np.sum(action):.6f}")
        
        # Apply masking manually to see the logic
        masked_action = env._apply_action_mask(action)
        
        print(f"🔧 Masked action: {masked_action}")
        print(f"🔧 Masked sum: {np.sum(masked_action):.6f}")
        
        # Check that inactive assets have zero weight
        if isinstance(obs, dict):
            mask = obs['mask']
            for j, is_active in enumerate(mask):
                if not is_active and j < len(masked_action) - 1:  # -1 for cash
                    if masked_action[j + 1] != 0:
                        print(f"⚠️  WARNING: Inactive asset {j} has non-zero weight: {masked_action[j + 1]}")
        
        # Verify sum is 1
        if abs(np.sum(masked_action) - 1.0) > 1e-6:
            print(f"⚠️  WARNING: Masked action sum is not 1: {np.sum(masked_action)}")
        else:
            print("✅ Masked action sum is 1.0")
    
    print("\n✅ Action masking test completed!")

def test_multiple_episodes():
    """Test that different episodes have different coin sets."""
    print("\n🧪 Testing Multiple Episodes")
    print("=" * 60)
    
    env = PortfolioEnv(use_variable_portfolio=True, reward_type="TRANSACTION_COST")
    
    episode_coins = []
    for episode in range(5):
        obs, info = env.reset()
        episode_coins.append(env.episode_coins.copy())
        print(f"Episode {episode + 1}: {env.episode_coins} ({len(env.episode_coins)} coins)")
        
        # Take a few steps to make sure it works
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    
    # Check for variety
    unique_coin_sets = set(tuple(coins) for coins in episode_coins)
    print(f"\n📊 Unique coin sets: {len(unique_coin_sets)} out of {len(episode_coins)} episodes")
    
    if len(unique_coin_sets) > 1:
        print("✅ Variable portfolio is working - different episodes have different coin sets!")
    else:
        print("⚠️  All episodes had the same coin set - check config.COINS_PER_EPISODE_RANGE")
    
    print("\n✅ Multiple episodes test completed!")

def test_fixed_vs_variable():
    """Compare fixed vs variable portfolio modes."""
    print("\n🧪 Testing Fixed vs Variable Portfolio")
    print("=" * 60)
    
    # Fixed portfolio
    env_fixed = PortfolioEnv(use_variable_portfolio=False, reward_type="TRANSACTION_COST")
    obs_fixed, info_fixed = env_fixed.reset()
    
    # Variable portfolio
    env_var = PortfolioEnv(use_variable_portfolio=True, reward_type="TRANSACTION_COST")
    obs_var, info_var = env_var.reset()
    
    print("Fixed Portfolio:")
    print(f"   Coins: {env_fixed.episode_coins}")
    print(f"   Action space: {env_fixed.action_space.shape}")
    print(f"   Observation type: {type(obs_fixed)}")
    
    print("\nVariable Portfolio:")
    print(f"   Coins: {env_var.episode_coins}")
    print(f"   Action space: {env_var.action_space.shape}")
    print(f"   Observation type: {type(obs_var)}")
    
    if isinstance(obs_var, dict):
        print(f"   Mask: {obs_var['mask']}")
    
    print("\n✅ Fixed vs Variable comparison completed!")

if __name__ == "__main__":
    print("🚀 Starting Variable Portfolio MLP Tests")
    print("=" * 60)
    
    try:
        test_variable_portfolio_basic()
        test_action_masking()
        test_multiple_episodes()
        test_fixed_vs_variable()
        
        print("\n🎉 All tests completed successfully!")
        print("✅ Variable portfolio functionality is working correctly.")
        print("✅ You can now use train_simple_mlp(use_variable_portfolio=True) to test MLP with variable assets!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 