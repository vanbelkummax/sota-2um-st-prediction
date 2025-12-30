# SOTA 2μm Spatial Transcriptomics Prediction

State-of-the-art prediction of gene expression from H&E histology images at 2μm Visium HD resolution.

## Project Goal

Achieve **SSIM > 0.60** for predicting spatial gene expression from H&E images at 2μm resolution, surpassing the current baseline of **SSIM = 0.5699** (Prov-GigaPath + Hist2ST + MSE loss).

### Scientific Motivation

Visium HD enables 2μm resolution spatial transcriptomics, but sequencing costs remain prohibitive for large cohorts. This project develops deep learning methods to predict gene expression directly from H&E histology images, enabling:

1. **Cost reduction**: H&E staining costs ~$5 vs. ~$5,000+ for spatial transcriptomics
2. **Retrospective analysis**: Apply to existing H&E slide archives (millions available)
3. **Clinical translation**: Enable spatial gene expression inference in routine pathology

### Key Challenge: Extreme Sparsity

At 2μm resolution, gene expression is **95-98% sparse** (zeros). Standard MSE loss fails because:
- MSE equally weights all predictions
- Model learns to predict zeros everywhere (trivial solution)
- Non-zero expression patterns are never captured

**Solution**: Zero-Inflated Negative Binomial (ZINB) and Focal losses that handle count data sparsity.

## Methods

### Multi-Scale Cross-Attention Fusion

Unlike global pooling approaches (iSCALE), we use **cross-scale attention** where:
- 2μm features **QUERY** coarser scale context (8μm, 32μm, 128μm)
- Spatial structure is **preserved** (not pooled away)
- Coarse scales provide tissue-level context without losing cellular detail

```
2μm patches → Encoder → 2μm features ─────────────────┐
                                                       ├─→ Cross-Scale Attention → Decoder → Predictions
8μm patches → Encoder → 8μm features (context) ───────┘
```

### Loss Functions

| Loss | Formula | Rationale |
|------|---------|-----------|
| **MSE** | `mean((pred - target)²)` | Baseline, fails on sparse data |
| **ZINB** | `-log P(y; μ, θ, π)` | Models count data with excess zeros |
| **Focal** | `(1-p)^γ * MSE` | Downweights easy zeros, focuses on hard cases |

### Foundation Model Encoders

| Encoder | Dimension | Training Data | Source |
|---------|-----------|---------------|--------|
| Prov-GigaPath | 1536 | 1.3B pathology tiles | Microsoft/Providence |
| Virchow2 | 1280 | 3M WSIs | Paige AI |
| UNI2-h | 1536 | 100M+ patches | Mahmood Lab |
| H-optimus-1 | 1024 | Large pathology corpus | Bioptimus |
| CONCH v1.5 | 768 | Path + text pairs | Mahmood Lab |
| ResNet50 | 2048 | ImageNet (baseline) | torchvision |

## Repository Structure

```
sota-2um-st-prediction/
├── README.md                 # This file
├── docs/
│   └── plans/
│       ├── 2025-12-26-sota-2um-st-prediction-design.md  # Original design document
│       └── MONTH_2-3_IMPLEMENTATION_ROADMAP.md          # Implementation timeline
├── configs/
│   └── default.yaml          # Training configuration
├── src/
│   ├── __init__.py           # Package exports
│   ├── data.py               # Multi-scale data loading (lines 1-501)
│   ├── losses.py             # ZINB, Focal, MultiTask losses
│   ├── fusion.py             # CrossScaleAttention, GatedScaleFusion, MultiScaleHist2ST
│   └── encoders.py           # Foundation model registry and loading
├── scripts/
│   ├── train_multiscale.py   # Main training script
│   ├── run_ablation.py       # Ablation experiment runner
│   ├── extract_2um_patches.py      # Initial extraction (deprecated)
│   └── extract_2um_patches_v2.py   # Correct coordinate mapping
└── experiments/              # Experiment outputs (gitignored)
```

## Data Locations

All data stored on fast ext4 filesystem (NOT NTFS /mnt/ drives):

### Source Data
```
/home/user/work/enact_data/
├── GSM8594567_P1CRC_tissue_image.btf   # P1 whole slide image
├── GSM8594568_P2CRC_tissue_image.btf   # P2 whole slide image
└── GSM8594569_P5CRC_tissue_image.btf   # P5 whole slide image
```

### Preprocessed Patches
```
/home/user/work/encoder-loss-ablation-2um/data/
├── P1_precomputed_labels_v2/           # P1 gene expression labels
│   ├── patch_XXX_YYY.npy               # Shape: (128, 128, 50) per patch
│   ├── patches.json                    # Patch metadata
│   └── metadata.json                   # Scale factors, gene list
├── P2_precomputed_labels_v2/           # P2 labels
├── P5_precomputed_labels_v2/           # P5 labels
└── multiscale/                         # Extracted image patches
    ├── P1/
    │   ├── scale_2um/                  # 576 patches @ 256×256 PNG
    │   ├── scale_8um/                  # 576 patches
    │   ├── scale_32um/                 # 576 patches
    │   └── scale_128um/                # 576 patches
    ├── P2/                             # 601 patches per scale
    └── P5/                             # 570 patches per scale
```

### Dataset Statistics

| Patient | Patches | Genes | Label Shape | Sparsity |
|---------|---------|-------|-------------|----------|
| P1 | 576 | 50 | (128, 128, 50) | 95.4% |
| P2 | 601 | 50 | (128, 128, 50) | 94.8% |
| P5 | 570 | 50 | (128, 128, 50) | 95.1% |
| **Total** | **1,747** | 50 | - | ~95% |

## Coordinate System

Critical for correct patch extraction from WSI:

```python
# Scale factor from metadata.json (coordinate_scale, NOT tissue_hires_scalef)
SCALE_FACTORS = {'P1': 0.275, 'P2': 0.2750, 'P5': 0.2735}

# Patch grid to WSI pixel mapping:
fullres_patch_size = 256 / scale_factor  # ~930 pixels in WSI
start_x = patch_col * 256 / scale_factor
start_y = patch_row * 256 / scale_factor
```

## Installation

```bash
# Clone repository
git clone https://github.com/vanbelkummax/sota-2um-st-prediction.git
cd sota-2um-st-prediction

# Create conda environment
conda create -n sota-st python=3.11 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate sota-st

# Install dependencies
pip install timm transformers pillow tifffile pandas numpy tqdm pyyaml
```

## Usage

### Quick Verification

```python
import sys
sys.path.insert(0, '/home/user/sota-2um-st-prediction')

from src.data import MultiScaleSTDataset
from src.losses import ZINBLoss, FocalMSELoss
from src.encoders import get_encoder, get_encoder_dim

# Load dataset
dataset = MultiScaleSTDataset(
    patients=['P1', 'P2'],
    scales=['2um', '8um'],
    max_patches_per_patient=100
)

# Get encoder
encoder = get_encoder('resnet50')
dim = get_encoder_dim('resnet50')  # 2048

# Initialize losses
zinb = ZINBLoss(n_genes=50)
focal = FocalMSELoss(gamma=2.0)
```

### Training

```bash
# Phase 1: Encoder ablation (frozen encoders)
python scripts/run_ablation.py --phase 1 --encoder prov-gigapath

# Phase 2: Loss function ablation
python scripts/run_ablation.py --phase 2 --loss zinb

# Phase 4: Multi-resolution ablation
python scripts/run_ablation.py --phase 4 --scales 2um 8um 32um
```

### Single Training Run

```bash
python scripts/train_multiscale.py \
    --train_patients P1 P2 \
    --test_patient P5 \
    --encoder resnet50 \
    --loss zinb \
    --scales 2um 8um \
    --epochs 50 \
    --batch_size 8
```

## Experimental Design

### Ablation Phases

| Phase | Variable | Options | Metric |
|-------|----------|---------|--------|
| 1 | Encoder | ResNet50, GigaPath, Virchow2, UNI2, H-optimus, CONCH | SSIM, PCC |
| 2 | Loss | MSE, ZINB, Focal, Multi-task | SSIM, PCC |
| 3 | Decoder | Hist2ST, MiniUNet, THItoGene, sCellST | SSIM, PCC |
| 4 | Resolution | 2μm, 2+8μm, 2+8+32μm, all | SSIM, PCC |

### Evaluation Protocol

- **Leave-one-patient-out**: Train on 2 patients, test on held-out patient
- **Metrics**: SSIM (structure), PCC (correlation), MSE (error)
- **Per-gene analysis**: Identify which genes are predictable

## Baseline Results

From prior encoder-loss ablation study:

| Encoder | Loss | SSIM | Notes |
|---------|------|------|-------|
| Prov-GigaPath | MSE | 0.5699 | Current best |
| Virchow2 | MSE | 0.5491 | -3.7% vs GigaPath |
| ResNet50 | MSE | 0.4823 | ImageNet baseline |

**Target**: SSIM > 0.60 with ZINB/Focal loss + multi-scale fusion

## Key Implementation Details

### Why ZINB Loss?

Gene expression follows negative binomial distribution with excess zeros:
- **Zero-inflation (π)**: Probability of structural zero (no expression possible)
- **Dispersion (θ)**: Gene-specific overdispersion parameter
- **Mean (μ)**: Predicted expression level

```python
# ZINB probability
P(y=0) = π + (1-π) * NB(0; μ, θ)
P(y>0) = (1-π) * NB(y; μ, θ)
```

### Why Cross-Scale Attention (not pooling)?

Global pooling loses spatial information:
```
[tumor][stroma][tumor] → pool → [mixed context]  # BAD: location lost
```

Cross-attention preserves spatial structure:
```
2μm query at position (i,j) attends to 8μm context at (i,j) → spatial preserved
```

### Why Frozen Encoders?

With only n=3 patients, fine-tuning foundation models leads to overfitting:
- Frozen: SSIM 0.5699 (generalizes)
- Fine-tuned: SSIM 0.61+ on train, 0.48 on test (overfit)

## Hardware Requirements

- **GPU**: 24GB VRAM (RTX 3090/4090/5090, A5000, etc.)
- **RAM**: 64GB+ recommended
- **Storage**: 50GB for data, SSD/NVMe preferred

## License

MIT License - see LICENSE file.

## Citation

If you use this code, please cite:

```bibtex
@software{sota_2um_st_prediction,
  title = {SOTA 2μm Spatial Transcriptomics Prediction},
  author = {Van Belkum, Max},
  year = {2024},
  url = {https://github.com/vanbelkummax/sota-2um-st-prediction}
}
```

## Acknowledgments

- Visium HD data from GEO accession GSE280318
- Foundation model weights from respective authors (Microsoft, Paige AI, Mahmood Lab, Bioptimus)
- Vanderbilt University Medical Center computational resources
