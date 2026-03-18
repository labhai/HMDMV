#!/bin/bash
# Example training script for Hotels-8k benchmark
set -e

echo "Running experiment"
python main.py \
    --device 0 \
    --seed 42 \
    --dataset hotels8k \
    --num_view 3 \
    --num_classes 7774 \
    --method HMDMV \
    --model_name vit_small_r26_s32_224 \
    --hmd_loss True \
    --batch_size 64 \
    --num_workers 8 \
    --epochs 100 \
    --patience 20 \
    --optim SGD \
    --learning_rate 0.01 \
    --warmup_steps 5 \
    --weight_decay 5e-4 \
    --momentum 0.9 \
    --temp 4.0 \
    --lambda_param 0.1 \
    --alpha 1.2 \
    --grad_clip_norm 80.0
