# Method Inventory - Spatial Transcriptomics Prediction from H&E

**Date**: 2025-12-26
**Phase**: Month 1 Week 1 - Literature Discovery
**Status**: Initial inventory complete

---

## Overview

This document catalogs all methods discovered for predicting spatial transcriptomics from H&E histology images. Methods are categorized by type and prioritized based on:
1. Claims at 2μm or subcellular resolution
2. Code availability and reproducibility
3. Novelty of approach
4. Publication venue and recency

**Total Methods Found**: 30+

---

## Tier 1: High Priority (Claims 2μm/Subcellular + Code Available)

### GHIST (2025)
- **Paper**: Nature Methods 2025, Vol 22, pages 1900-1910
- **DOI**: https://doi.org/10.1038/s41592-025-02795-z
- **PMID**: 40954301
- **GitHub**: Not yet found (search pending)
- **Key Innovation**: Multitask deep learning for single-cell resolution prediction
- **Architecture**: Leverages cell nuclei morphology, cell-type info, neighborhood info, and gene expression jointly
- **Resolution**: Single-cell level (subcellular spatial transcriptomics)
- **Dataset**: Validated on public datasets + TCGA
- **Status**: Paper acquired ✅ (`papers/paywalled/GHIST_NatureMethods2025.pdf`)
- **Priority**: **CRITICAL** - claims 2μm SOTA

### Img2ST-Net (2025, Huo Lab)
- **Paper**: Published 2025
- **PMID**: 41210922
- **Authors**: Huo lab (Vanderbilt)
- **Key Innovation**: Efficient high-resolution ST prediction via fully convolutional image-to-image learning
- **GitHub**: Search pending
- **Priority**: **CRITICAL** - from our own lab, likely tested on similar data

### sCellST (2024)
- **Paper**: bioRxiv 2024.11.07.622225v1
- **GitHub**: Search pending
- **Key Innovation**: Multiple Instance Learning for single-cell resolution
- **Architecture**: Uses cell morphology alone (single-cell images, not patches)
- **Resolution**: Single-cell level
- **Dataset**: PDAC (pancreatic ductal adenocarcinoma)
- **Status**: Preprint acquired ✅ (`papers/biorxiv/2024.11.07.622225v1.pdf`)
- **Priority**: **HIGH** - subcellular approach, novel MIL framework

### stImage (2025, Huo Lab)
- **Paper**: Published 2025
- **PMID**: 40905789
- **Authors**: Huo lab (Vanderbilt)
- **Key Innovation**: Versatile framework with customizable deep histology and location-informed integration
- **GitHub**: Search pending
- **Priority**: **HIGH** - from our lab, customizable framework

---

## Tier 2: High Priority (SOTA Methods, CVPR/Top Venues)

### TRIPLEX (2024)
- **Paper**: CVPR 2024
- **arXiv**: 2403.07592 (March 2024)
- **GitHub**: https://github.com/NEXGEM/TRIPLEX ✅
- **Key Innovation**: Multi-resolution features (target spot, neighbor view, global view)
- **Architecture**: Fusion of 3 resolutions capturing cellular morphology → local context → global tissue organization
- **Metrics**: Outperforms SOTA in MSE, MAE, PCC
- **Dataset**: 3 public ST datasets + Visium (10X Genomics)
- **Priority**: **HIGH** - multi-scale approach relevant to 2μm

### THItoGene (2024)
- **Paper**: Briefings in Bioinformatics 2024, Vol 25, Issue 1
- **DOI**: https://doi.org/10.1093/bib/bbad464
- **PMID**: 38145948
- **GitHub**: https://github.com/yrjia1015/THItoGene ✅
- **Key Innovation**: Hybrid dynamic CNN + capsule networks
- **Architecture**: Adaptively senses molecular signals in histology
- **Dataset**: Human breast cancer + cutaneous squamous cell carcinoma
- **Priority**: **HIGH** - claims "improved Hist2ST", capsule networks novel

### Hist2ST (2022)
- **Paper**: Briefings in Bioinformatics 2022
- **GitHub**: https://github.com/biomed-AI/Hist2ST ✅
- **Key Innovation**: CNN + Transformer + GNN hybrid
- **Architecture**:
  - CNN for 2D vision features
  - Transformer for global spatial relations
  - GNN for neighbor patch relations
  - ZINB distribution for gene expression
- **Dataset**: HER2+ breast cancer + cutaneous squamous cell carcinoma
- **Status**: **BASELINE** - current decoder in our 0.5699 SSIM system
- **Priority**: **CRITICAL** - our current baseline, understand internals

---

## Tier 3: Benchmark Studies (Critical Context)

### HEST-1K Benchmark (NeurIPS 2024)
- **Paper**: NeurIPS 2024 Spotlight
- **GitHub**: https://github.com/mahmoodlab/HEST ✅
- **Key Innovation**: Comprehensive benchmark of 11 pathology foundation models
- **Tasks**: 9 tasks for gene expression prediction (50 HVGs)
- **Resolution**: 112×112 μm regions at 0.5 μm/px
- **Organs**: 9 organs, 8 cancer types
- **Priority**: **CRITICAL** - establishes baseline for foundation model comparison
- **Action**: Apply for access (note in `USER_ACCESS_NEEDED.md`)

### Nature Communications Benchmark (Feb 2025)
- **Paper**: Nature Communications 2025
- **DOI**: https://doi.org/10.1038/s41467-025-56618-y
- **Key Innovation**: Reproduced 11 methods on 5 SRT datasets
- **Validation**: External validation on TCGA
- **Priority**: **HIGH** - comprehensive method comparison

### SpaRED Benchmark
- **GitHub**: https://github.com/BCV-Uniandes/SpaRED ✅
- **Key Innovation**: Spatial transcriptomics completion for benchmarking
- **Priority**: **MEDIUM** - complementary benchmarking resource

---

## Tier 4: Integration/Transfer Methods

### Tangram (2021)
- **Paper**: Biancalani et al. 2021
- **GitHub**: https://github.com/broadinstitute/Tangram ✅
- **Key Innovation**: Deep learning alignment of scRNA-seq to ST
- **Priority**: **MEDIUM** - label transfer, not direct H&E prediction

### SpaGE (2020)
- **Paper**: Abdelaal et al. 2020
- **GitHub**: https://github.com/tabdelaal/SpaGE ✅
- **Key Innovation**: Spatial gene enhancement using scRNA-seq
- **Priority**: **MEDIUM** - enhancement, not direct prediction

### stLearn
- **GitHub**: https://github.com/BiomedicalMachineLearning/stlearn_interactive ✅
- **Key Innovation**: H&E image integration for spot analysis
- **Priority**: **MEDIUM** - exploratory analysis, not prediction-focused

---

## Tier 5: Additional Methods (To Investigate)

### HistoSPACE (2024)
- **PMID**: 39521362
- **GitHub**: https://github.com/samrat-lab/HistoSPACE ✅
- **Priority**: **MEDIUM** - recent method, needs investigation

### STMCL (2025)
- **Paper**: Multimodal contrastive learning framework
- **GitHub**: https://github.com/wenwenmin/STMCL ✅
- **Key Innovation**: Integrates histology, gene expression, and location
- **Priority**: **MEDIUM** - contrastive learning approach novel

### HE2ST / DeepSpot
- **GitHub**: https://github.com/ratschlab/he2st ✅
- **Key Innovation**: Leverages spatial context for enhanced prediction
- **Priority**: **MEDIUM** - spatial context methods

### hist2RNA
- **GitHub**: https://github.com/raktim-mondol/hist2RNA ✅
- **Key Innovation**: Predicts gene expression from digital histology
- **Priority**: **LOW** - older method, lower priority

---

## Foundation Models (Encoders)

### Tested in Our Lab
1. **Prov-GigaPath** (current best, SSIM 0.5699 @ 2μm, frozen)
2. **Virchow2** (frozen better than fine-tuned, -3.7% vs Prov-GigaPath)
3. **CONCH v1** (ViT-B/16, tested at 8μm Ridge only, #5/7 encoders, r=0.467)

### Untested at 2μm with Neural Decoder (Priority)
4. **UNI2-h** (ViT-H/14, 1536-dim, HF access acquired ✅)
5. **CONCHv1.5** (ViT-L, larger than v1, **PRIORITY for Month 2**)
6. **H-optimus-1** (search needed)
7. **GigaPath** (non-Prov version, search needed)
8. **Phikon** (search needed)
9. **DenseNet** (baseline, likely tested)

---

## Cross-Disciplinary Inspiration (Month 1 Week 4)

### Super-Resolution Imaging
- **CARE** (Content-Aware Image Restoration)
- **CSBDeep** (framework for CARE)
- **Noise2Void** (self-supervised denoising)

### Weather Forecasting
- **GraphCast** (Google DeepMind, graph neural networks for global weather)
- **FourCastNet** (NVIDIA, Fourier neural operators)

### Protein Structure
- **AlphaFold 2/3** (attention mechanisms, MSA processing)
- **ESMFold** (language model for structure)

### Neural Rendering
- **NeRF** (Neural Radiance Fields)
- **Gaussian Splatting** (3D scene representation)

---

## Methods to Search (Pending)

Based on design document list, still need to find:
- [ ] HisToGene (different from THItoGene?)
- [ ] MERGE (hierarchical graph GNN)
- [ ] SpatialScope
- [ ] DeepST
- [ ] TESLA
- [ ] STAND
- [ ] Tissue-ViT
- [ ] ConvNeXt variants for pathology
- [ ] Spatial-GLUE
- [ ] stDiff (diffusion models for ST)
- [ ] Others from GitHub awesome lists

---

## Key Papers from Huo Lab (Vanderbilt)

1. **PMID:41210922** - Img2ST-Net (2025) ⭐
2. **PMID:40905789** - stImage (2025) ⭐
3. **PMID:41280008** - Computer Vision Methods for ST Survey (2025)
4. **PMID:40375953** - Spatial Pathomics Toolkit (2024)
5. **PMID:41323019** - PySpatial toolkit (2025)
6. **PMID:41256883** - Cell AI foundation models in kidney (2025)
7. **PMID:41286516** - Cell foundation models w/ human-in-the-loop (2025)
8. **PMID:41079265** - Assessment of Cell Nuclei AI Foundation Models (2025)

---

## Key Papers from Lau Lab (CRC Spatial Atlas)

1. **PMID:35794563** - CRC spatial atlas, CAF-TME crosstalk (2022) ⭐
2. **PMID:36669472** - Multiplexed 3D atlas of CRC (2023) ⭐
3. **PMID:38065082** - Molecular cartography in sporadic CRC (2023) ⭐
4. **PMID:34910928** - Pre-malignant programs in CRC polyps (2021)
5. **PMID:40026233** - Multiomic spatial atlas DMBT1 in dysplasia (2025)
6. **PMID:41042257** - Elemental imaging + ST in colon TME (2025)

---

## Key Papers from Sarkar Lab (ST Methods)

1. **PMID:39849132** - Mapping topography with interpretable DL (2025) ⭐
2. **PMID:37873258** - Same paper, preprint version (2023)

---

## Action Items

### Immediate (Week 1)
- [x] Complete initial method inventory
- [ ] Clone Tier 1 GitHub repos (TRIPLEX, THItoGene, Hist2ST)
- [ ] Acquire papers via PubMed/bioRxiv APIs
- [ ] Update `CODE_ARCHAEOLOGY.md` with initial findings
- [ ] Search for remaining methods in "Pending" list

### Week 2
- [ ] Deep extraction of GHIST architecture (paper acquired)
- [ ] Deep extraction of TRIPLEX architecture
- [ ] Deep extraction of THItoGene architecture
- [ ] Deep extraction of sCellST architecture
- [ ] Test CONCHv1.5 empirically at 2μm

### Week 3
- [ ] Identify common architectural patterns
- [ ] Statistical analysis: performance vs complexity
- [ ] Document "critical ingredients" for 2μm success

### Week 4
- [ ] Mine cross-disciplinary literature
- [ ] Generate 10-15 novel hypothesis candidates

---

## Notes

- **Polymax database**: synthesis_run_id = 77 (initialized for this project)
- **Memory graph**: 8 entities created with baseline findings
- **GitHub access**: Some repos require authentication (use `gh` CLI if needed)
- **HEST-1K**: Requires application for access - add to `USER_ACCESS_NEEDED.md`
- **Papers acquired**: GHIST (paywalled), sCellST (bioRxiv), 1 arXiv preprint
- **Prioritization rule**: Methods claiming 2μm/subcellular + code > SOTA methods > benchmarks > baselines

---

**Last Updated**: 2025-12-26
**Next Update**: After Week 2 deep extraction phase
