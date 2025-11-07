#!/bin/bash
# Generate validation predictions for all 11 trained models across 8 H100 GPUs
# Distributes models to avoid OOM: GPU 0-2 get 2 models, GPU 3-7 get 1 model

set -e

# Create output directory
mkdir -p data/predictions_validation
mkdir -p logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "=========================================="
echo "Launching all VALIDATION predictions at $TIMESTAMP"
echo "Distributing across 8 GPUs (0-7)"
echo "=========================================="

# GPU 0: 2 models
echo ""
echo "GPU 0: Launching 2 models..."

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/generate_predictions.py \
    --model models/exp1_baseline --test data/global_validation.jsonl \
    --output data/predictions_validation/exp1_baseline.json --gpu 0 \
    > logs/pred_val_exp1_baseline_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 0] exp1_baseline (PID: $!)"

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/generate_predictions.py \
    --model models/exp1_full --test data/global_validation.jsonl \
    --output data/predictions_validation/exp1_full.json --gpu 0 \
    > logs/pred_val_exp1_full_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 0] exp1_full (PID: $!)"

# GPU 1: 2 models
echo ""
echo "GPU 1: Launching 2 models..."

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/generate_predictions.py \
    --model models/exp1_permutation --test data/global_validation.jsonl \
    --output data/predictions_validation/exp1_permutation.json --gpu 1 \
    > logs/pred_val_exp1_permutation_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 1] exp1_permutation (PID: $!)"

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/generate_predictions.py \
    --model models/exp2_mixed --test data/global_validation.jsonl \
    --output data/predictions_validation/exp2_mixed.json --gpu 1 \
    > logs/pred_val_exp2_mixed_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 1] exp2_mixed (PID: $!)"

# GPU 2: 2 models
echo ""
echo "GPU 2: Launching 2 models..."

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/generate_predictions.py \
    --model models/exp1_synthetic --test data/global_validation.jsonl \
    --output data/predictions_validation/exp1_synthetic.json --gpu 2 \
    > logs/pred_val_exp1_synthetic_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 2] exp1_synthetic (PID: $!)"

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/generate_predictions.py \
    --model models/exp2_structured --test data/global_validation.jsonl \
    --output data/predictions_validation/exp2_structured.json --gpu 2 \
    > logs/pred_val_exp2_structured_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 2] exp2_structured (PID: $!)"

# GPU 3: 1 model
echo ""
echo "GPU 3: Launching 1 model..."

CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/generate_predictions.py \
    --model models/exp2_sequential/phase2_structured_final --test data/global_validation.jsonl \
    --output data/predictions_validation/exp2_sequential.json --gpu 3 \
    > logs/pred_val_exp2_sequential_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 3] exp2_sequential (PID: $!)"

# GPU 4: 1 model
echo ""
echo "GPU 4: Launching 1 model..."

CUDA_VISIBLE_DEVICES=4 nohup python -u scripts/generate_predictions.py \
    --model models/exp2_unstructured --test data/global_validation.jsonl \
    --output data/predictions_validation/exp2_unstructured.json --gpu 4 \
    > logs/pred_val_exp2_unstructured_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 4] exp2_unstructured (PID: $!)"

# GPU 5: 1 model
echo ""
echo "GPU 5: Launching 1 model..."

CUDA_VISIBLE_DEVICES=5 nohup python -u scripts/generate_predictions.py \
    --model models/exp3_warmup/phase2_full_final --test data/global_validation.jsonl \
    --output data/predictions_validation/exp3_warmup.json --gpu 5 \
    > logs/pred_val_exp3_warmup_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 5] exp3_warmup (PID: $!)"

# GPU 6: 1 model
echo ""
echo "GPU 6: Launching 1 model..."

CUDA_VISIBLE_DEVICES=6 nohup python -u scripts/generate_predictions.py \
    --model models/exp3_staged/phase3_nyt_final --test data/global_validation.jsonl \
    --output data/predictions_validation/exp3_staged.json --gpu 6 \
    > logs/pred_val_exp3_staged_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 6] exp3_staged (PID: $!)"

# GPU 7: 1 model
echo ""
echo "GPU 7: Launching 1 model..."

CUDA_VISIBLE_DEVICES=7 nohup python -u scripts/generate_predictions.py \
    --model models/exp3_no_warmup --test data/global_validation.jsonl \
    --output data/predictions_validation/exp3_no_warmup.json --gpu 7 \
    > logs/pred_val_exp3_no_warmup_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 7] exp3_no_warmup (PID: $!)"

echo ""
echo "=========================================="
echo "All 11 VALIDATION predictions launched!"
echo "Distributed across 8 GPUs:"
echo "  GPUs 0-2: 2 models each (6 models total)"
echo "  GPUs 3-7: 1 model each (5 models total)"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/pred_val_*_${TIMESTAMP}.log"
echo ""
echo "Check completion:"
echo "  ls -lh data/predictions_validation/"
echo "  watch -n 10 'ls -lh data/predictions_validation/ | wc -l'"
echo ""
echo "All logs saved with timestamp: ${TIMESTAMP}"
echo "=========================================="
