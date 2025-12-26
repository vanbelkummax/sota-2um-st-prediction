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

**Total Methods Found**: 45+

**Last Update**: 2025-12-26 (Critical infrastructure + generative methods added)

**Status**: Week 1 complete with optimized execution order

---

## Critical Reality Check: What "2μm" Actually Means

⚠️ **Important**: Visium HD is a grid of 2×2 μm barcoded squares, but **10X recommends starting at 8×8 μm bins**. When papers claim "2μm," verify:
- (a) **Raw 2μm bins** (extremely sparse, ~90% zeros)
- (b) **Cell-level** maps rendered onto pixels
- (c) **8μm bins** (most common)

**Implication**: True 2μm prediction requires:
1. **Bin→cell assignment** (cells overlap bins, bins overlap cells)
2. **Sparsity handling** (regression/MSE collapses to zeros)
3. **Generative heads** (diffusion/flow/discrete counts)

---

## Tier -1: Infrastructure (The Missing Piece)

### ENACT ⭐⭐⭐ CRITICAL - START HERE ⭐⭐⭐
- **Paper**: Bioinformatics 2025 (OUP Academic)
- **PMID**: TBD
- **GitHub**: https://github.com/Sanofi-Public/enact-pipeline ✅
- **Why CRITICAL**: **First tissue-agnostic pipeline for Visium HD bin→cell assignment**
- **What It Does**:
  - Cell segmentation (StarDist)
  - Multiple bin-to-cell assignment strategies
  - Outputs AnnData format
- **The Problem It Solves**: At 2μm, cells overlap bins and bins overlap cells - you need assignment logic BEFORE prediction
- **Priority**: **MUST RUN FIRST** - this defines your ground truth
- **User Insight**: "Bin→cell is part of the model, not preprocessing"
- **Status**: Production-ready pipeline, Python-based
- **Note**: Without this, you're not truly working at 2μm resolution

---

## Tier 0: Bleeding Edge (2024-2025 SOTA Breakthroughs)

### Stem (ICLR 2025) ⭐ GENERATIVE BASELINE - START HERE FOR DIFFUSION
- **Paper**: ICLR 2025
- **GitHub**: https://github.com/SichenZhu/Stem ✅
- **Key Innovation**: **Conditional Diffusion for ST Prediction**
- **Paradigm Shift**: Generative (not regression) - learns distribution of possible expressions for given morphology
- **Why Critical**: Handles "one-to-many" biological reality (same morphology → multiple possible cell states)
- **Implementation**: Clean repo, HEST integration, runnable
- **Resolution**: 2μm capable
- **Priority**: **HIGHEST for generative baseline** - cleanest diffusion implementation
- **User Recommendation**: "If you want fast proof you can win, run Stem first"
- **Borrowed From**: Generative AI / Latent Diffusion Models (Stable Diffusion, DALL-E)

### STFlow (ICML 2025) ⭐ SLIDE-LEVEL JOINT MODELING
- **Paper**: ICML 2025
- **GitHub**: https://github.com/Graph-and-Geometric-Learning/STFlow ✅
- **Key Innovation**: **Whole-slide flow matching** - models joint distribution across entire slide
- **Paradigm**: Spots are NOT independent - captures cell-cell interactions
- **Why Critical**: "If your thesis is 'slide-level structure matters', run STFlow"
- **Resolution**: Whole-slide scale
- **Priority**: **HIGH** - addresses spatial dependence explicitly
- **User Recommendation**: Alternative to Stem if slide-level structure is key hypothesis
- **Borrowed From**: Flow matching / Normalizing flows

### GenAR (Late 2025) ⭐ DISCRETE COUNT GENERATION
- **Paper**: OpenReview 2025
- **GitHub**: https://github.com/oyjr/genar ✅ (verify)
- **Key Innovation**: **Multi-scale autoregressive discrete count generation**
- **Why Critical**: Treats outputs as **discrete tokens/counts** (not continuous)
- **Addresses**: "Counts aren't continuous" - raw 2μm bins are integer counts
- **Priority**: **HIGH** - direct counter to sparsity/zero-inflation
- **User Recommendation**: "Put GenAR on your radar specifically for discrete counts"
- **Note**: DeepSpot2Cell uses MSE (easy to beat with better likelihood/discrete head)
- **Borrowed From**: Autoregressive language models (GPT-style)

### PixNet (2025) - DENSE CONTINUOUS MAPPING
- **Paper**: arXiv 2025
- **GitHub**: Code availability unclear
- **Key Innovation**: **Dense continuous gene expression map** - aggregate into any spot size
- **Paradigm**: Implicit neural representation f(x,y,context)→expression
- **Why Critical**: Conceptually perfect for HD (dense outputs, sparse supervision)
- **Resolution**: Continuous (can aggregate to any bin size)
- **Priority**: **MEDIUM** - novel approach, code availability uncertain
- **Borrowed From**: NeRF / Coordinate MLPs (implicit neural representations)

### CarHE (2024)
- **Paper**: Published 2024
- **GitHub**: https://github.com/Jwzouchenlab/CarHE ✅
- **Key Innovation**: **Contrastive Alignment for Visium HD**
- **Architecture**: CLIP-style contrastive learning + Transformer
- **Resolution**: **Explicitly designed for Visium HD 2μm bins + Xenium (subcellular)**
- **Performance**: Predicts 17,000+ genes at 2μm resolution
- **Why Critical**: First method explicitly validated on Visium HD 2μm data
- **Priority**: **CRITICAL** - our exact use case (Visium HD 2μm)
- **Borrowed From**: Web-scale image search (CLIP/OpenAI)

### DeepSpot2Cell (2024)
- **Paper**: Published 2024
- **GitHub**: https://github.com/ratschlab/DeepSpot2Cell ✅
- **Key Innovation**: **Virtual Single-Cell Prediction via DeepSets/MIL**
- **Architecture**: Multiple Instance Learning + Pathology Foundation Models (UNI/Hoptimus0)
- **Resolution**: Virtual single-cell (~2μm nuclear scale)
- **Paradigm**: Treats "spot" as bag of cells, deconvolves aggregate → single-cell
- **Why Critical**: Super-resolution from coarse data (train on standard Visium, predict single-cell)
- **Priority**: **CRITICAL** - deconvolution approach novel
- **Borrowed From**: DeepSets / Multiple Instance Learning

### TissueNarrator (2024)
- **Paper**: Published 2024
- **GitHub**: Search pending
- **Key Innovation**: **LLM/Tokenization for Spatial Biology**
- **Architecture**: Treats tissue as "spatial sentences" (cells = words)
- **Paradigm**: Uses LLM architectures to predict based on "grammar" of cellular neighborhoods
- **Why Critical**: Most novel approach - applies NLP to spatial biology
- **Priority**: **HIGH** - extremely novel, may be overkill
- **Borrowed From**: Large Language Models (ChatGPT/BERT)

### Diff-ST (MICCAI 2024)
- **Paper**: MICCAI 2024
- **GitHub**: Search pending
- **Key Innovation**: **Diffusion for Image Super-Resolution → ST**
- **Architecture**: Treats ST prediction as image super-resolution task
- **Why Critical**: Proven diffusion framework adapted for biology
- **Priority**: **HIGH** - diffusion approach established
- **Borrowed From**: Image restoration / Super-resolution

### STPath (2025) ⭐ FOUNDATION MODEL - GENERATIVE PARADIGM
- **Paper**: npj Digital Medicine 2025
- **GitHub**: Search pending (likely released)
- **Key Innovation**: **Generative Foundation Model for Spatial Transcriptomics**
- **Scale**: Pretrained on **~1,000 WSIs** paired with spatial transcriptomes across **17 organs**
  - Training corpus: **~4 million spatial spots** from combined public datasets
- **Architecture**:
  - Geometry-aware Transformer (attends across all spots in whole slide)
  - **Masked Gene Modeling**: Masks gene expression at some locations, predicts from image + context
  - Incorporates metadata (organ type, platform modality) as inputs
- **Resolution**: Can predict **~39,000 genes** at spot/cell level without fine-tuning
- **Paradigm**: Zero-shot "virtual sequencing" foundation model (like GPT for spatial biology)
- **Why Critical**:
  - First true foundation model for H&E → RNA
  - Generalizes across tissues and sequencing platforms
  - Enables virtual sequencing for any archived pathology slide
- **Priority**: **CRITICAL** - paradigm-defining foundation model
- **Use Cases**:
  - Zero-shot gene prediction on new slides
  - Imputation of missing spatial spots
  - Survival prediction as transfer task
- **Borrowed From**: Foundation models (GPT-style masked modeling)
- **User Insight**: "A model that can predict transcriptomes for any tissue, at any resolution, without retraining"

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

## Tier 1.5: Advanced Architectures (Novel Techniques from Other Fields)

### LUNA (2024)
- **Paper**: Search pending
- **GitHub**: Search pending
- **Key Innovation**: **Masked Language Modeling for Spatial Transcriptomics**
- **Architecture**: Spatial tokenization + Masked modeling
- **Technique**: Mask 20% of cells, predict from remaining 80% context
- **Why Critical**: Forces model to learn tissue organization rules
- **Priority**: **HIGH** - self-supervised learning approach
- **Borrowed From**: BERT/Masked Language Models (NLP)

### BLEEP (2024)
- **Paper**: Search pending
- **GitHub**: Search pending
- **Key Innovation**: **Shared Latent Spaces (H&E ↔ RNA)**
- **Architecture**: Dual encoders + InfoNCE loss
- **Capability**: Zero-shot gene queries
- **Priority**: **MEDIUM** - contrastive learning variant
- **Borrowed From**: CLIP / Contrastive Learning

### HIPT (Hierarchical Image Pyramid Transformer)
- **Paper**: Search pending
- **GitHub**: Search pending
- **Key Innovation**: **Multi-scale hierarchical learning**
- **Architecture**: Zoom-out encoder (256μm) + Zoom-in encoder (2μm)
- **Why Critical**: Captures tissue architecture AND nuclear morphology
- **Priority**: **HIGH** - addresses context vs resolution trade-off
- **Borrowed From**: Satellite imaging / Gigapixel learning

### iStar (2024)
- **Paper**: Search pending
- **GitHub**: Search pending
- **Key Innovation**: **Hierarchical gigapixel learning**
- **Priority**: **MEDIUM** - hierarchical approach
- **Borrowed From**: Remote sensing

### ResSAT (2024)
- **Paper**: Search pending
- **GitHub**: https://github.com/StickTaTa/STGIN_main ✅ (verify)
- **Key Innovation**: **Graph Self-Supervised Residual Learning**
- **Architecture**: Spatial context + batch correction
- **Priority**: **MEDIUM** - GNN-based
- **Borrowed From**: Graph representation learning

### H&Enium (2024)
- **Paper**: Search pending
- **GitHub**: https://github.com/cbib/DeepSpot ✅ (verify branch)
- **Key Innovation**: **Foundation Model Alignment (H&E ↔ scRNA-seq)**
- **Architecture**: Soft alignment target for biological noise tolerance
- **Resolution**: Cell-level/subcellular
- **Priority**: **MEDIUM** - alignment approach
- **Borrowed From**: Multi-modal contrastive learning

### SciSt (2024) ⭐ BIOLOGICAL PRIOR INJECTION
- **Paper**: Briefings in Bioinformatics 2024
- **GitHub**: Not publicly released, similar to sCellST approach
- **Key Innovation**: **Single-cell reference-informed framework**
- **Architecture**:
  - Step 1: Cell segmentation + counting (Hover-Net cell detector)
  - Step 2: Construct "Initial Expression (IE)" vector by weighting canonical gene profiles from scRNA atlas
  - Step 3: Dual encoders (ResNet34 + self-attention for image, MLP for IE)
  - Step 4: Fusion decoder refines gene prediction
- **Why Critical**: **Uses single-cell reference as biological prior** ("prompt engineering" for spatial biology)
- **Performance**: Outperformed HisToGene and THItoGene on HER2+ breast cancer (higher Pearson correlation)
- **Resolution**: Spot-level (55μm), adaptable to subcellular
- **Priority**: **HIGH** - biological grounding significantly boosts accuracy
- **User Insight**: "By using single-cell reference prototypes as a 'prompt,' SciSt provides a biologically grounded prior"
- **Borrowed From**: Few-shot learning + biological knowledge injection

### DANet (2025) ⭐ DYNAMIC ALIGNMENT
- **Paper**: Bioinformatics 2025 (OUP)
- **GitHub**: Available on GitHub
- **Key Innovation**: **Dynamic Alignment Network with novel architectural components**
- **Architecture**:
  - **Densely connected CNN**: Captures intricate cellular details (preserves local info better than plain CNN)
  - **State-space model module**: Treats gene expression as sequence, models gene-gene dependencies
  - **Residual Kolmogorov-Arnold Network (RKAN)**: Learnable activation function for bimodal alignment
    - Dynamically adjusts how image features map to gene features during contrastive training
- **Performance**: ~20% gains in gene correlation metrics on public datasets
- **Datasets**: GSE240429 (human brain Visium), HER2+ breast cancer
- **Why Critical**: Novel activation learning (RKAN) for cross-modal alignment
- **Resolution**: Spot-level, adaptable
- **Priority**: **HIGH** - architectural innovation (RKAN) transferable to other methods
- **Borrowed From**: Kolmogorov-Arnold representation theorem + residual connections

### PRTS (2025) ⭐ ZERO-INFLATION HANDLING
- **Paper**: Science China Life Sciences 2025
- **GitHub**: Check PMC article for availability
- **Key Innovation**: **Two-head loss for sparse single-cell data**
- **Architecture**:
  - **Binary head**: Predicts whether gene is expressed at all (addresses dropout)
  - **Continuous head**: Predicts expression level if gene is present
- **Why Critical**: Explicitly tackles **zero-inflation** in high-resolution platforms (2μm bins)
- **Problem Solved**: At 2μm, most bins have zero counts for most genes (capture stochasticity, not biology)
- **Resolution**: Single-cell spatial transcriptomic maps
- **Priority**: **HIGH** - critical for handling 2μm sparsity
- **User Insight**: "Addresses zero-inflation in high-resolution platforms by using a two-head loss"
- **Borrowed From**: Zero-inflated statistical models (ZINB, hurdle models)

### STMCL (2025) - CONTRASTIVE MULTI-SLICE
- **Paper**: Methods 2025
- **GitHub**: Search pending
- **Key Innovation**: **Inferring multi-slice spatial gene expression via contrastive learning**
- **Architecture**: Contrastive framework for aligning expression across serial sections
- **Why Critical**: Handles 3D reconstruction / serial section alignment
- **Resolution**: Multi-slice integration
- **Priority**: **MEDIUM** - useful for 3D spatial reconstruction (Month 5+)
- **Borrowed From**: Contrastive learning + cross-slice alignment

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

## Key Architectural Insights (User-Provided SOTA Analysis)

### Paradigm Shift: Regression → Generative
- **Old Approach**: CNNs with regression heads → predict mean expression (blurry, misses rare cells)
- **New Approach**: Diffusion models (Stem, Diff-ST) → learn distribution of expressions
- **Impact**: Handles "one-to-many" biology (same morphology → multiple possible states)

### Critical Techniques Borrowed from Other Fields

1. **From Generative AI**: Latent Diffusion Models (Stable Diffusion → Stem, Diff-ST)
   - Generate sharp, multimodal distributions
   - Predict Cell Type A OR B (not averaged muddy signal)

2. **From NLP**: Tokenization + Masked Modeling (BERT → TissueNarrator, LUNA)
   - Cells = words, tissue = sentences
   - Mask 20% of cells, predict from 80% context
   - Learns "tissue grammar"

3. **From Web-Scale Search**: Contrastive Learning (CLIP → CarHE, BLEEP, H&Enium)
   - Shared latent spaces (H&E ↔ RNA)
   - InfoNCE loss
   - Enables zero-shot gene queries

4. **From Satellite Imaging**: Hierarchical Gigapixel Learning (HIPT, iStar)
   - Zoom-out (256μm tissue architecture) + Zoom-in (2μm nuclear morphology)
   - Addresses context vs resolution trade-off

### The "Winning Frankenstein Architecture" (User Recommendation)

**Do NOT use simple CNN (ResNet/EfficientNet) - that approach has plateaued.**

**Recommended Stack**:
1. **Encoder**: Pathology Foundation Model (UNI, CONCH, Virchow) - do NOT train from scratch
2. **Context**: Graph Neural Network OR Transformer to aggregate neighbor features (spatial context)
3. **Prediction Head**: Diffusion Head (NOT regression head) to preserve variance + biological noise

**Critical for 2μm**:
- **Input**: Large context patch (256×256 px, ~100+ μm) - NOT just 2μm crop
- **Architecture**: Vision Transformer OR Hierarchical CNN
- **Output**: Dense prediction map (pixel-wise or 2μm bin-wise) where center pixel inferred from surrounding context

---

## Tier 6: Cross-Disciplinary Techniques (Mathematical Isomorphisms)

**Source**: Project Aether strategic analysis - identifying mathematical parallels from remote sensing, medical imaging, and computer vision

### Remote Sensing: Hyperspectral Pansharpening

**The Isomorphism**:
- **Remote Sensing**: Panchromatic image (high spatial, low spectral) + Hyperspectral image (low spatial, high spectral) → Fused high-res hyperspectral
- **Our Domain**: H&E (high spatial, 3 channels) + Visium (low spatial, 20K genes) → Fused high-res gene expression

#### HyperPNN (Hyperspectral Pansharpening Neural Network)
- **Paper**: IEEE Transactions on Geoscience and Remote Sensing
- **GitHub**: Search "HyperPNN" + author names from papers
- **Key Innovation**: Spectral predictive branch - predicts high-level spectral ratios from high-res image
- **Adaptation**: Predict "transcriptomic residuals" (how cell deviates from neighborhood mean)
- **Priority**: MEDIUM - architectural inspiration for attention mechanisms
- **Status**: Conceptual transfer, not direct code reuse

#### HSpeNet (Hyperspectral Network)
- **Paper**: Remote sensing literature
- **GitHub**: Search "HSpeNet pansharpening"
- **Key Innovation**: **Dual-Attention Fusion Block (DAFB)**
  - Spatial Attention: Identifies "where" (nuclei vs stroma)
  - Spectral Attention: Identifies "which genes" correlate with textures
- **Adaptation**: Replace Cross-Attention in Stem with DAFB layers
  - H&E embedding = "PAN" input
  - Gene embeddings = "HS" input
  - Enforces: Gene variance must be grounded in morphological variance
- **Priority**: HIGH - direct architectural component for "Clean Frankenstein"
- **User Insight**: "Spatial-Spectral Attention to attend to spatial textures and correlate them with spectral bands"

### Medical Imaging: Implicit Neural Representations (INRs)

#### STINR (Spatial Transcriptomics via Implicit Neural Representation)
- **Paper**: "Deciphering Spatial Transcriptomics via Implicit Neural Representation"
- **GitHub**: Search for STINR implementations
- **Key Innovation**: Models gene expression as continuous function f(x,y) → gene vector
  - Parametrized by MLP (Multilayer Perceptron)
  - Provides "infinite resolution" - query at any (x,y) coordinate
- **Adaptation**: **Image-Guided INR** - condition MLP on both coordinates AND image features I(x,y)
  - Similar to Local Implicit Image Functions (LIIF) in computer vision
  - Use as decoder: Encoder extracts latent grid → INR queries at ENACT cell centroids
- **Priority**: HIGH - solves spatial discontinuity (point cloud of cells vs fixed pixel grid)
- **Benefits**:
  - Decouples image resolution from prediction resolution
  - Train on Visium HD, deploy for single-cell prediction
  - 3D reconstruction: Stack sections, query continuous volume

#### SUICA (Spatial Unsupervised Image-based Clustering and Analysis)
- **Paper**: Related to STINR, verify exact method
- **Key Innovation**: INR for spatial clustering
- **Priority**: LOW - STINR more relevant for our use case

#### LIIF (Local Implicit Image Functions)
- **Paper**: Computer vision (CVPR 2021)
- **GitHub**: https://github.com/yinboc/liif
- **Key Innovation**: Learn continuous image representation, super-resolution at arbitrary scales
- **Adaptation**: Template for image-guided INR decoder in our architecture
- **Priority**: MEDIUM - reference implementation for INR architecture

#### JIIF (Joint Implicit Image Function)
- **Paper**: MRI super-resolution literature
- **Key Innovation**: Continuous 3D volume representation, interpolate between slices
- **Adaptation**: Future 3D reconstruction (stack H&E sections, predict continuous ST volume)
- **Priority**: LOW (Month 5+) - for 3D spatial reconstruction

### Computer Vision: Continuous Representations

#### NeRF-style Approaches
- **Concept**: Neural Radiance Fields - continuous 3D scene representation
- **Adaptation**: PixNet appears to use similar paradigm (dense continuous gene map)
- **Priority**: LOW - PixNet already captures this approach in Tier 0

---

## Tier 7: Evaluation Frameworks and Tools

### scIB-E (Single-cell Integration Benchmarking - Extended)
- **Paper**: Nature Methods (original scIB), verify scIB-E extension
- **GitHub**: https://github.com/theislab/scib (search for extended metrics)
- **Purpose**: **Biological Conservation Metrics**
  - Beyond correlation - does model preserve cell types and states?
- **Metrics**:
  - **Adjusted Rand Index (ARI)**: Cluster agreement with ground truth
  - **Normalized Mutual Information (NMI)**: Information shared between clusterings
  - **Local Inverse Simpson's Index (LISI)**: Local diversity/mixing
  - **Silhouette Score**: Cluster separation on cell types
- **Implementation**: Use `scib-metrics` Python library
- **Priority**: **CRITICAL for Day 0-1** - defines evaluation harness
- **User Insight**: "Ultimate test is whether predicted expression clusters correctly"

### POT (Python Optimal Transport)
- **GitHub**: https://github.com/PythonOT/POT
- **Purpose**: **Wasserstein Distance / Earth Mover's Distance**
- **Why Critical**:
  - PCC/SSIM punish spatial shifts harshly (B-cell 10µm away = complete mismatch)
  - Wasserstein recognizes nearby cells, assesses geometrically appropriate penalty
- **Use Cases**:
  - Loss function: Sinkhorn Divergence for training
  - Evaluation: Wasserstein distance between predicted/true distributions
- **Implementation**: `ot.sinkhorn()`, `ot.emd()`
- **Priority**: **CRITICAL for Day 0-1** - core evaluation metric
- **User Insight**: "OT measures 'effort' to transform predicted distribution into ground truth"

### SpatialQC (Spatial Fidelity Metrics)
- **Purpose**: Spatial pattern validation beyond per-pixel metrics
- **Metrics**:
  - Spatially Variable Gene (SVG) recovery
  - Spatial autocorrelation (Moran's I)
  - Cell-type spatial distribution
- **Priority**: MEDIUM - complementary to scIB-E

---

## Mathematical Formulations (from Aether Report)

### DeepSet (MIL Architecture)
```
f(X) = ρ(Σ_{x∈X} φ(x))
```
- X = set of cells in region
- φ = encoder network (per cell)
- ρ = decoder network (per spot)
- Summation ensures permutation invariance

### Diffusion Model (Stem)
- Problem: p(y|x) is multimodal (same morphology → multiple states)
- Regression: minimizes ||ŷ - y||² → converges to E[y|x] (conditional mean = blur)
- Diffusion: learns distribution p(y|x) itself, samples from manifold of valid states

### Optimal Transport Loss
- Wasserstein distance W(P,Q) = "effort" to transform distribution P into Q
- Sinkhorn Divergence: differentiable approximation for training
- Better than MSE for sparse, discrete count data

---

### Optimized Execution Order (User-Recommended)

**"If you told me 'win within 8-12 weeks,' I'd do:"**

**Step 0 (Days 0-1): Lock the Eval Definition** ⚠️ CRITICAL
- **Decide**: Raw 2μm bins vs 8μm bins vs cell-level?
- **Metrics**: Gene-wise correlation + SVG recovery + cell-type recovery (NOT just SSIM)
- **Why**: Prevent trivial smoothing, reward biological signal
- **Reference**: GHIST shows cell-type accuracy style evaluation

**Step 1 (Days 1-2): Steal ENACT's Bin→Cell Machinery** ⭐ THE MISSING PIECE
- **Goal**: Produce cell-by-gene matrices from Visium HD + segmentation
- **Action**: Clone https://github.com/Sanofi-Public/enact-pipeline
- **Extract**: Cell segmentation, bin-to-cell assignment, AnnData output
- **Why**: "Without this, you're not truly working at 2μm"

**Step 2 (Days 2-3): Get a Generative Baseline Running**
- **Option A (fast proof)**: Run **Stem** (clean repo, HEST integration)
- **Option B (slide-level structure)**: Run **STFlow** (whole-slide joint modeling)
- **Why**: Generative > regression for sparse/zero-inflated 2μm bins

**Step 3 (Days 3-4): Extract MIL/DeepSets Aggregator**
- **Source**: DeepSpot2Cell
- **Goal**: "Bag of subspots/cells → spot" logic as drop-in module
- **Why**: Handles partial cells/bins correctly

**Step 4 (Days 4-5): Decide Count Handling**
- **Problem**: MSE plateaus on HD sparsity (DeepSpot2Cell uses MSE = easy to beat)
- **Option A**: Swap to **GenAR** (discrete counts/tokens)
- **Option B**: Keep Stem diffusion (continuous but generative)
- **Option C**: Report both

**The "Clean Frankenstein" (High Probability of Working)**:
```
PFM Encoder → Multiscale Context → MIL Aggregator → Generative Head
     ↓              ↓                    ↓                  ↓
   UNI/Virchow   HIPT/Local        DeepSpot2Cell      Stem/STFlow/GenAR
```

**The "Aggressive Frankenstein" (Highest Upside, Highest Risk)**:
1. **Contrastive pretrain** (BLEEP/H&Enium/CarHE) → stable H&E↔RNA latent
2. **Generative finetune** (diffusion/flow/AR counts) conditioned on latent
3. Optional: Retrieval of RNA prototypes (nearest-neighbor gene vectors) during decoding

---

### Novel Hypotheses Generated

**From Initial SOTA Analysis**:
- H_diff_001: Replace our MSE loss with conditional diffusion head
- H_contrast_001: Use contrastive learning (CLIP-style) instead of supervised regression
- H_hier_001: Hierarchical encoders (tissue + cellular scales)
- H_mil_001: Treat Visium HD 2μm bins as bags of sub-cellular features
- H_token_001: Spatial tokenization + masked modeling for self-supervised pretraining

**From Updated Analysis**:
- H_enact_001: ENACT-style cell-level ground truth vs raw bin-level
- H_flow_001: Flow matching (STFlow) vs diffusion (Stem) for slide-level structure
- H_discrete_001: Discrete count generation (GenAR) vs continuous (Stem/MSE)
- H_franken_001: Full Frankenstein (PFM→HIPT→MIL→Diffusion)
- H_2stage_001: Two-stage (contrastive pretrain → generative finetune)

---

### Predictable Failure Modes (Watch For These)

1. **Stain/Batch Artifact Leakage**: Model "wins" metrics by learning slide-level artifacts
   - **Prevention**: Slide-level train/test splits, cross-stain validation
2. **Registration Noise Dominates**: Model learns blur, not biology
   - **Prevention**: ENACT jitter stress tests, alignment QC
3. **Degenerate Loss (All Zeros)**: Raw 2μm sparsity → loss collapses
   - **Prevention**: Generative heads (diffusion/flow/discrete counts), NOT MSE
4. **Trivial Smoothing**: High SSIM from Gaussian blur
   - **Prevention**: SVG recovery, cell-type accuracy, pathway alignment metrics

---

## Notes

- **Polymax database**: synthesis_run_id = 77 (initialized for this project)
- **Memory graph**: 8 entities created with baseline findings
- **GitHub access**: Some repos require authentication (use `gh` CLI if needed)
- **HEST-1K**: Requires application for access - add to `USER_ACCESS_NEEDED.md`
- **Papers acquired**: GHIST (paywalled), sCellST (bioRxiv), 1 arXiv preprint
- **Prioritization rule**: Methods claiming 2μm/subcellular + code > SOTA methods > benchmarks > baselines

---

## Meta-Resources (Stay Comprehensive Without Drowning)

### Curated Lists
1. **Awesome Vision-driven Models for Spatial Omics**
   - GitHub: https://github.com/hrlblab/computer_vision_spatial_omics
   - Living list of ST prediction methods
   - Includes methodological categorization

2. **HEtoSGEBench** (Benchmarking Pipeline)
   - GitHub: https://github.com/SydneyBioX/HEtoSGEBench
   - Forces honest eval hygiene
   - Useful for spot-based prediction pipelines

### Foundation Model Hubs
- **HuggingFace Mahmood Lab**: UNI, UNI2-h, CONCH, CONCHv1.5, FEATHER models
- **HuggingFace Bioptimus**: H-optimus-0, H0-mini
- **Paige AI**: Virchow, Virchow2 (Azure marketplace)

### Datasets
- **HEST-1K** (NeurIPS 2024): 1000+ paired ST + WSI, 11 FM benchmark
- **10X Visium HD Public**: Human Colon Cancer (https://github.com/10XGenomics/HumanColonCancer_VisiumHD)
- **Ken Lau CRC Atlas**: 6 papers, potential validation cohort

---

**Last Updated**: 2025-12-26 (Post-user intelligence integration)
**Next Update**: After Week 2 code archaeology + eval definition locked
**Total Methods**: 45+ (Tier -1 to Tier 5)
**Execution-Ready**: YES ✅
