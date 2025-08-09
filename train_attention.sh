#!/bin/bash

# =============================================================================
# ATTENTION TRAINING SCRIPT - Uses CustomTD3_AC with AttentionTD3Policy
# =============================================================================
# Updated to use your custom CustomTD3_AC implementation without --tf-agents-style
# Key fixes applied:
# - CRL constraints disabled for exploration
# - Correct AttentionTD3Policy selection
# - Variable portfolio support with masking
# - Fixed action normalization
# =============================================================================

echo "🚀 Attention Training (Variable Portfolios) with CustomTD3_AC"
echo "============================================================="
echo "✅ Algorithm: TD3 with CustomTD3_AC"
echo "✅ Policy: AttentionTD3Policy (variable_coin_attention)"
echo "✅ Portfolio: Variable (1-3 coins per episode)"
echo "✅ CRL: Adaptive profile (ENABLED for smooth allocations)"
echo "✅ Constraint: λ adapts to enforce action smoothness"
echo ""

# Activate virtual environment
source venv/bin/activate

# Run attention training with CRL enabled
python train_simple_mlp.py \
    --algorithm TD3 \
    --attention variable_coin_attention \
    --timesteps 10000 \
    --mlp-size heavy \
    --variable \
    --reward-type TRANSACTION_COST \
    --crl-profile adaptive \
    --enable-crl

echo ""
echo "📊 Attention Training Completed!"
echo "Check results in logs/TD3_simple_mlp_* directories"