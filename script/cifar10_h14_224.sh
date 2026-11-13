#!/usr/bin/env bash

# Data: CIFAR-10 | N_train=50k | N_test=10k | x:32²×3→224²×3 | classes=10
# Model: ViT-H/14 | L=32 | D=1280 | heads=16 | P=14² | params≈632 M
# Effective batch size: 64 samples × 2 gradient-accumulation steps = 128
# Out: ./runs/cifar10_h14_224/${RUN_ID}/{args.txt,metrics.json,training_curves.png,logs/,checkpoints/}

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RUN_NAME="cifar10_h14_224"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="./runs/${RUN_NAME}/${RUN_ID}"

mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs"

train_args=(
    # Data arguments
    --dataset cifar10 \
    --data-path ./data \
    --image-size 224 \
    --batch-size 64 \
    --num-workers 4 \
    --in-channels 3 \
    --num-classes 10 \

    # Training arguments
    --device cuda:5 \
    --seed 42 \
    --lr 3e-4 \
    --min-lr 1e-6 \
    --warmup-epochs 5.0 \
    --weight-decay 0.05 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --num-epoch 200 \
    --grad-accum-steps 2 \

    # ViT arguments: ViT-H/14
    --ebd-dim 1280 \
    --n-head 16 \
    --head-dim 80 \
    --patch-size 14 \
    --n-layer 32 \
    --ffn-dim 5120 \
    --ffn-type gelu \
    --dropout-img-ebd 0.1 \
    --dropout-att-weight 0.0 \
    --dropout-att-out 0.1 \
    --dropout-ffn-inner 0.1 \
    --dropout-ffn-out 0.1 \
    --ffn-bias \

    # Output arguments
    --checkpoint-dir "${RUN_DIR}/checkpoints" \
    --output-dir "${RUN_DIR}"
)

.venv/bin/python train.py "${train_args[@]}" \
    2>&1 | tee "${RUN_DIR}/logs/train.log"
