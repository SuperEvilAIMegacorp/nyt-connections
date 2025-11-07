#!/bin/bash
# Generate predictions for the ORIGINAL (untrained) base model on test and validation sets
# Model: unsloth/Qwen3-4B-Thinking-2507 (before any fine-tuning)

set -e

# Create output directories
mkdir -p data/predictions
mkdir -p data/predictions_validation
mkdir -p logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "=========================================="
echo "Generating ORIGINAL model predictions at $TIMESTAMP"
echo "Model: unsloth/Qwen3-4B-Thinking-2507 (base model)"
echo "=========================================="

# Test set predictions (GPU 0)
echo ""
echo "GPU 0: Generating predictions on global_test.jsonl..."

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/generate_predictions.py \
    --model unsloth/Qwen3-4B-Thinking-2507 \
    --test data/global_test.jsonl \
    --output data/predictions/original_model_test.json \
    --gpu 0 \
    > logs/pred_original_test_${TIMESTAMP}.log 2>&1 &
TEST_PID=$!
echo "  [GPU 0] Original model on test set (PID: $TEST_PID)"

# Validation set predictions (GPU 1)
echo ""
echo "GPU 1: Generating predictions on global_validation.jsonl..."

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/generate_predictions.py \
    --model unsloth/Qwen3-4B-Thinking-2507 \
    --test data/global_validation.jsonl \
    --output data/predictions_validation/original_model_validation.json \
    --gpu 1 \
    > logs/pred_original_validation_${TIMESTAMP}.log 2>&1 &
VAL_PID=$!
echo "  [GPU 1] Original model on validation set (PID: $VAL_PID)"

echo ""
echo "=========================================="
echo "Original model predictions launched!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/pred_original_test_${TIMESTAMP}.log"
echo "  tail -f logs/pred_original_validation_${TIMESTAMP}.log"
echo ""
echo "Process PIDs:"
echo "  Test set (GPU 0): $TEST_PID"
echo "  Validation set (GPU 1): $VAL_PID"
echo ""
echo "Check if still running:"
echo "  ps -p $TEST_PID,$VAL_PID"
echo ""
echo "Output files:"
echo "  data/predictions/original_model_test.json"
echo "  data/predictions_validation/original_model_validation.json"
echo ""
echo "All logs saved with timestamp: ${TIMESTAMP}"
echo "=========================================="
