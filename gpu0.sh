#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_vtab_mopdo.py \
  --dataset_name caltech101 \
  --dataset_dir ./data/vtab-1k \
  --pretrained_dir ViT-B_16.npz \
  --output_dir output/vtab_mopdo
