#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python inference_vtab_mopdo.py \
  --dataset_name caltech101 \
  --dataset_dir ./data/vtab-1k \
  --pretrained_dir ViT-B_16.npz \
  --output_dir output/vtab_mopdo_eval \
  --resume output/vtab_mopdo/caltech101/caltech101_94.44.pth \
  --max_batches 100
