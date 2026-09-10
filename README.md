# MoPDO

This is the cleaned MoPDO workspace.  
It keeps only the MoPDO model code, VTAB data loaders, and the public training and inference entry points.

## Abstract

Parameter-Efficient Fine-Tuning (PEFT) has become a widely used paradigm for adapting large vision models to downstream tasks. However, many existing PEFT methods rely on low-rank or linear transformations, which may restrict the flexibility of feature adaptation when downstream visual distributions differ from those seen during pre-training. In this paper, we propose MoPDO, a Mixture of PDE-Derived Operators for parameter-efficient vision fine-tuning. MoPDO reformulates visual feature adaptation as a lightweight mixture of PDE-derived operators, enabling frequency-domain modulation with structured spatial responses beyond standard low-rank adaptation. As an initial exploration, we instantiate MoPDO with three representative responses inspired by heat diffusion, wave propagation, and a steady-state Poisson component. These responses induce different spatial patterns in the frequency domain, while task-adaptive, layer- and head-specific routing coefficients balance their contributions. To implement them efficiently, we derive a unified formulation based on the Discrete Cosine Transform (DCT), making MoPDO easy to insert into Vision Transformers as a plug-and-play adapter with only a small number of trainable parameters. Using MAE pre-trained ViT-B, MoPDO improves PEFT accuracy by up to 2.1% on VTAB1K image classification with a comparable number of trainable parameters. Extensive experiments across supervised and self-supervised vision backbones demonstrate the effectiveness, efficiency, and adaptability of MoPDO, suggesting PDE-derived operators as a practical direction for parameter-efficient vision model adaptation.

## Layout

- `models_mopdo/`: MoPDO model implementation
- `datasets_mopdo/`: VTAB loaders and task configs
- `train_vtab_mopdo.py`: training entry point
- `inference_vtab_mopdo.py`: evaluation and visualization entry point
- `scripts/run.sh`: minimal training launcher

## Setup

```bash
conda env create -n MoPDO -f environment.yaml
```

## Data

Put VTAB-1k under:

```text
./data/vtab-1k/<dataset_name>/
```

The scripts read the standard split files from each dataset folder.

## Checkpoint

Download the ImageNet pre-trained ViT checkpoint you want to use and pass it with `--pretrained_dir`.

## Train

```bash
python train_vtab_mopdo.py \
  --dataset_name caltech101 \
  --dataset_dir ./data/vtab-1k \
  --pretrained_dir ViT-B_16.npz \
  --output_dir output/vtab_mopdo
```

## Inference

```bash
python inference_vtab_mopdo.py \
  --dataset_name caltech101 \
  --dataset_dir ./data/vtab-1k \
  --pretrained_dir ViT-B_16.npz \
  --resume output/vtab_mopdo/caltech101/caltech101_94.44.pth \
  --output_dir output/vtab_mopdo_eval \
  --max_batches 100
```
