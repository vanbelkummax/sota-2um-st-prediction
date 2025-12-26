# Comprehensive Synthesis: SOTA 2μm ST Prediction Project

**Date**: 2025-12-26
**Phase**: Month 1 Week 1 Complete → Week 2 Day 0-1 Complete
**Status**: Pre-implementation intelligence gathering complete
**Decision Point**: Go/No-Go for Day 1-2 (ENACT extraction)

---

## Executive Summary

**Project Goal**: Build SOTA system for predicting spatial transcriptomics at 2μm resolution from H&E histology, surpassing current baseline (Prov-GigaPath + Hist2ST, SSIM 0.5699).

**Current Status**:
- ✅ **Week 1 Complete**: 70+ methods cataloged across 14 priority tiers
- ✅ **Day 0-1 Complete**: Evaluation protocol locked, SpatialEvaluator implemented
- ✅ **Intelligence Integrated**: 3 major reports synthesized (initial discovery, Aether, SOTA+HF)
- 🟢 **Ready**: Proceed to Day 1-2 (ENACT extraction) - all prerequisites met

**Key Achievement**: Identified paradigm shift from regression → generative and critical infrastructure (ENACT) that was missing from original plan.

---

## Intelligence Gathering Summary

### Three Major Intelligence Drops Integrated

#### 1. Week 1 Initial Discovery (30+ methods)
- Standard literature search (PubMed, arXiv, GitHub)
- Huo lab papers (Img2ST-Net, stImage)
- Lau lab CRC spatial atlas
- Established baseline methods (GHIST, TRIPLEX, THItoGene, sCellST)

#### 2. Project Aether Strategic Analysis (Mathematical Rigor)
**Paradigm Shift**: Regression → Generative
- Regression predicts conditional mean E[y|x] → blurry averages
- Generative models sample from p(y|x) → sharp, multimodal distributions
- Critical at 2μm where identical morphology → multiple cell states

**Mathematical Isomorphisms Identified**:
- **Hyperspectral Pansharpening**: ST prediction ≅ fusing PAN (H&E high-res, 3 channels) + HS (Visium low-res, 20K genes)
  - Techniques: HyperPNN, HSpeNet (Dual-Attention Fusion Blocks)
  - Direct transfer: Spatial Attention (where) + Spectral Attention (which genes)
- **Implicit Neural Representations**: STINR, LIIF - model gene expression as continuous f(x,y)
  - Benefit: Query at arbitrary coordinates, decouple image/prediction resolution
- **Optimal Transport**: Wasserstein distance for spatial predictions
  - Better than PCC/SSIM - recognizes nearby cells vs punishing shifts

**ENACT Identified as "THE MISSING PIECE"**:
- Bin→cell assignment at 2μm (cells overlap bins, bins overlap cells)
- Without this: training on noisy, contaminated ground truth
- Tier -1 priority (infrastructure layer, must run FIRST)

#### 3. SOTA + HuggingFace Database (Foundation Models + Datasets)
**Foundation Models**:
- **STPath**: Pretrained on 1,000 WSIs, 4M spots, 17 organs (zero-shot generalization)
- **PAST**: Pathology + scRNA foundation model (trained for H&E → genes)
- **Threads**: Molecular-driven FM (trained with genomic/transcriptomic supervision)
- **MIRROR**: Modality alignment + retention (don't destroy modality-specific signal)

**Critical Datasets**:
- **HEST-1k**: 1,000+ ST-WSI pairs, 9 organs, 8 cancer types (gold standard benchmark)
- **HESCAPE**: Batch effects benchmark (most methods fail across sites/stains)

**Evaluation Insights**:
- **MIL Transfer Study**: Evidence on whether pretrained MIL aggregators transfer
- **hist2RNA**: Documents common pitfalls to avoid

---

## Current State: Method Inventory

**Total Methods Cataloged**: **70+** across **14 priority tiers**

### Tier -1: Infrastructure (THE MISSING PIECE)
- **ENACT** (Sanofi-Public) - Bin→cell assignment for Visium HD

### Tier 0: Bleeding Edge (2024-2025 SOTA)
- **Stem** (ICLR 2025) - Conditional diffusion, generative baseline
- **STPath** (npj Digital Medicine 2025) - Foundation model, 1000 WSIs, zero-shot
- **CarHE** - Contrastive alignment, Visium HD 2μm (our exact use case)
- **DeepSpot2Cell** - Virtual single-cell via MIL/DeepSets
- **STFlow** (ICML 2025) - Whole-slide flow matching
- **GenAR** - Discrete count generation (autoregressive)
- **PixNet** - Dense continuous mapping (NeRF-style)
- **TissueNarrator** - LLM/tokenization for spatial biology
- **Diff-ST** (MICCAI 2024) - Diffusion for image super-resolution

### Tier 1: High Priority (Claims 2μm + Code Available)
- **GHIST** (Nature Methods 2025) - Multitask learning, single-cell resolution
- **SciSt** - Single-cell reference-informed (biological prior injection)
- **DANet** - RKAN (learnable activation), ~20% gains
- **PRTS** - Two-head loss for zero-inflation (critical for 2μm sparsity)
- **Img2ST-Net** (Huo Lab) - Efficient high-res prediction
- **sCellST** - MIL for single-cell
- **stImage** (Huo Lab) - Huo lab ST prediction framework
- **TRIPLEX** (ICLR 2024) - Multi-resolution fusion
- **THItoGene** - Improved Hist2ST
- **Hist2ST** - CNN + Transformer + GNN
- **STMCL** - Contrastive multi-slice learning

### Tier 1.5: Advanced Architectures
- **LUNA** - Masked token modeling
- **BLEEP** - Shared latent spaces (CLIP-style)
- **HIPT** - Hierarchical Image Pyramid Transformer
- **H&Enium** - Foundation model alignment
- **ResSAT** - Graph self-supervised learning

### Tier 6: Cross-Disciplinary Techniques
**Remote Sensing**:
- **HyperPNN** - Spectral predictive branch
- **HSpeNet** - Dual-Attention Fusion Blocks (Spatial + Spectral)

**Medical Imaging - INRs**:
- **STINR** - Implicit neural representation, continuous f(x,y)
- **LIIF** - Local implicit image functions
- **JIIF** - Joint implicit (3D MRI super-resolution)

### Tier 7: Evaluation Frameworks
- **scIB-E** - Biological conservation metrics (ARI, NMI, Silhouette, LISI)
- **POT Library** - Optimal Transport (Wasserstein distance)
- **SpatialQC** - Spatial fidelity metrics

### Tier 8: Datasets and Benchmarks
- **HEST-1k** ⭐ - 1,000+ ST-WSI pairs, 9 organs (external validation benchmark)
- **STimage-1K4M** - 4M sub-tiles, dense supervision
- **HESCAPE** ⭐ - Batch effects benchmark (critical for generalization)

### Tier 9: Foundation Models (Extended)
- **Threads** ⭐ - Molecular-driven (genomics + transcriptomics)
- **mSTAR** - 100+ tasks, gene expression pretraining
- **PathOrchestra** - 100+ task FM
- **PAST** ⭐ - Pathology + scRNA (directly trained for H&E → genes)
- **UNI** (already had) - Self-supervised pathology FM
- **CONCH** (already had) - Visual-language pathology FM

### Tier 10: Direct Gene Expression Prediction
- **hist2RNA** ⭐ - Baseline + pitfalls documentation
- **Gene-DML** - Dual-pathway discrimination
- **SToFM** - Multi-scale extraction

### Tier 11: Cross-Modal Alignment
- **MIRROR** ⭐ - Alignment + retention (don't destroy modality-specific signal)

### Tier 12: MIL and WSI Aggregation
- **Do MIL Models Transfer?** ⭐ - Transfer learning study
- **MambaMIL** - Long-sequence state-space models

### Tier 13: Generative Models
- **PixCell** - Diffusion for histopathology (data augmentation)

---

## Current Baseline and Target

### Baseline Performance (What We Must Beat)
- **Model**: Prov-GigaPath (frozen) + Hist2ST decoder + MSE loss
- **Performance**: SSIM 0.5699 @ 2μm
- **Dataset**: 3 CRC patients (P1, P2, P5), 50 genes, Visium HD
- **Limitations**:
  - Regression-based (predicts mean, not distribution)
  - No bin→cell assignment (training on contaminated ground truth)
  - MSE loss (collapses to zeros for sparse 2μm bins)
  - No biological priors

### Target Performance
- **Primary**: SSIM > 0.60 (95% CI excludes 0.5699)
- **Strong**: SSIM > 0.65 + external cohort validation
- **Exceptional**: SSIM > 0.70 + ≥2 tissue types

### Success Metrics (Locked)
1. **Primary**: SSIM (per-gene, aggregated), Wasserstein distance
2. **Gene-centric**: Pearson r, Spearman ρ (per gene)
3. **Biological conservation**: ARI, NMI, Silhouette, LISI (scIB-E)
4. **Spatial fidelity**: SVG recovery, Moran's I
5. **Negative controls**: Label shuffle, spatial jitter, random encoder

---

## The "Clean Frankenstein" Architecture Blueprint

**Synthesis of Best Components Across All Methods**:

### Module 1: Bio-Geometry Preprocessor (Tier -1)
**Engine**: ENACT
- **Input**: Raw Visium HD 2μm bins + H&E WSI
- **Process**:
  - Nuclei segmentation (StarDist, fine-tuned on H&E)
  - Cell boundary expansion (Watershed)
  - Bin assignment: Weighted by Area (fractional overlap)
- **Output**: Cell-level AnnData (clean, uncontaminated ground truth)
- **Status**: Ready to extract (Day 1-2)

### Module 2: Multi-Scale Visual Encoder (Tier 1 + Tier 9)
**Options** (in priority order):
1. **PAST** - Pretrained on H&E → genes (most aligned with task)
2. **Threads** - Molecular supervision (genomics + transcriptomics)
3. **UNI** - Robust general pathology features
4. **CONCH** - Visual-language capabilities

**Context Extraction** (from DeepSpot2Cell):
- Cell-level crop (local texture, nuclear morphology)
- Spot-level crop (55μm neighborhood context)
- Global encoding (positional, tissue-level semantics)
- **Fusion**: Multi-head attention or HSpeNet DAFB

### Module 3: Generative Core (Tier 0)
**Options** (in priority order):
1. **Stem** (FAST WIN) - Clean diffusion baseline, HEST integrated, proven
2. **STFlow** - Whole-slide flow matching (if slide-level structure hypothesis)
3. **GenAR** - Discrete counts (if zero-inflation dominates)

**Modifications**:
- Replace simple patch embedding with Module 2 multi-scale context
- Inject HSpeNet DAFB into cross-attention layers
- Use Optimal Transport loss (Sinkhorn) alongside denoising loss

### Module 4: Continuous Decoder (Tier 6) [OPTIONAL]
**Engine**: STINR (Image-Guided INR)
- MLP conditioned on (x,y) coordinates + Module 2 latent features
- Enables querying at arbitrary cell centroids
- Decouples image resolution from prediction resolution

### Module 5: Loss Functions (Tier 7 + Insights)
**Primary**: Denoising score matching (Stem standard)
**Auxiliary**:
- **Optimal Transport**: Sinkhorn Divergence (geometric appropriateness)
- **Zero-Inflation Head** (PRTS): Binary (is expressed?) + Continuous (level)

### Module 6: Evaluation Harness (Day 0-1 Complete)
**Engine**: SpatialEvaluator (implemented)
- Wasserstein distance (POT library)
- Bio-conservation (scIB-E metrics)
- Spatial fidelity (SVG recovery, Moran's I)
- Hierarchical bootstrap (valid for n=3)

---

## Resource and Tool Assessment

### Have (✅ Ready to Use)

**Code and Tools**:
- ✅ SpatialEvaluator class (comprehensive evaluation harness)
- ✅ POT library (Optimal Transport)
- ✅ scib-metrics (biological conservation)
- ✅ esda/libpysal (spatial statistics)
- ✅ Memory graph (37+ entities, 39+ relations)
- ✅ polymax-synthesizer MCP (paper extraction)
- ✅ vanderbilt-professors MCP (local paper search)
- ✅ latex-architect MCP (figure generation)

**Data**:
- ✅ 3 CRC patients (P1, P2, P5) - Visium HD 2μm, 50 genes
- ✅ Baseline predictions (Prov-GigaPath + Hist2ST, SSIM 0.5699)
- ✅ H&E WSIs (pyramidal tiffs)

**Models (Accessible)**:
- ✅ UNI (HuggingFace gated, access granted)
- ✅ CONCH (HuggingFace gated, access granted)
- ✅ CONCHv1.5 (mentioned in model card)
- ✅ Prov-GigaPath (current baseline encoder)
- ✅ Virchow2 (tested baseline)

**Papers**:
- ✅ GHIST (Nature Methods 2025) - acquired
- ✅ 70+ methods documented with GitHub links
- ✅ 3 major synthesis reports integrated

### Need (🔴 HIGH / 🟡 MEDIUM Priority)

**Code Repositories (Day 1-2)**:
- 🔴 **ENACT** (github.com/Sanofi-Public/enact-pipeline) - THE MISSING PIECE
- 🔴 **Stem** (github.com/SichenZhu/Stem) - Fast win baseline
- 🔴 **CarHE** (github.com/Jwzouchenlab/CarHE) - Visium HD 2μm specific
- 🔴 **DeepSpot2Cell** (github.com/ratschlab/DeepSpot2Cell) - MIL aggregator
- 🟡 STFlow (github.com/Graph-and-Geometric-Learning/STFlow)
- 🟡 GenAR (github.com/oyjr/genar) - verify exists

**Model Weights**:
- 🔴 **PAST** (check HuggingFace/GitHub) - Best pretrained for H&E → genes
- 🟡 **Threads** (check availability) - Molecular supervision
- 🟡 **mSTAR** (check availability) - Gene-aware pretraining

**Papers (Full Text)**:
- 🟡 TRIPLEX supplementary (ICLR 2024) - Architecture details
- 🟡 sCellST supplementary (Nature Biotech 2024) - Upsampling strategy
- 🟡 STPath code/paper details - Foundation model implementation

**Datasets (Month 5)**:
- 🟡 **HEST-1k** (github.com/mahmoodlab/HEST) - External validation
  - Not blocking for Months 1-4 (use CRC data)
  - Critical for Month 5 generalization testing

### Don't Need (Low Priority or Redundant)

- ❌ Additional CRC data (3 patients sufficient for proof-of-concept)
- ❌ Synthetic data generation (PixCell) - won't help with 3 patients
- ❌ Bulk RNA-seq integration - not in current scope
- ❌ 3D reconstruction tools (JIIF) - Month 5+ only
- ❌ Additional foundation models beyond PAST/Threads/UNI/CONCH

---

## Gap Analysis and Risks

### Critical Gaps (Must Address)

1. **ENACT Bin→Cell Assignment**
   - **Gap**: Not yet extracted/implemented
   - **Risk**: Without this, training on noisy ground truth (20% contamination)
   - **Mitigation**: Day 1-2 priority extraction
   - **Impact**: BLOCKS all downstream work

2. **PAST Model Weights**
   - **Gap**: Don't know if weights are available
   - **Risk**: May be best pretrained backbone, unavailable = suboptimal encoder
   - **Mitigation**: Search HuggingFace/GitHub for PAST weights (Day 1)
   - **Fallback**: Use UNI (proven, available)

3. **Zero-Inflation Handling**
   - **Gap**: MSE loss collapses to zeros at 2μm
   - **Risk**: Model learns to predict zero for everything
   - **Mitigation**: PRTS two-head loss or Optimal Transport loss
   - **Status**: Conceptually understood, needs implementation (Day 3-4)

### Medium Risks (Monitor)

4. **Batch Effects**
   - **Gap**: 3 patients may have scanner/site/stain variation
   - **Risk**: HESCAPE shows most methods fail across batches
   - **Mitigation**: Nested CV per patient, hierarchical bootstrap
   - **Future**: HEST-1k validation will reveal batch sensitivity

5. **Hyperparameter Search**
   - **Gap**: Budget ~15 configurations per method
   - **Risk**: Underfitting hyperparameter space with limited compute
   - **Mitigation**: Use Bayesian optimization, focus on critical hyperparams
   - **Timeline**: Month 5 Week 1-2

### Low Risks (Accept)

6. **External Validation Unavailable**
   - **Gap**: HEST-1k may not be accessible
   - **Risk**: Can't claim cross-tissue generalization
   - **Mitigation**: Be conservative in claims ("CRC-specific SOTA")
   - **Acceptable**: Primary goal is 2μm CRC, generalization is bonus

7. **Small Sample Size (n=3)**
   - **Gap**: Only 3 patients limits statistical power
   - **Risk**: High variance, uncertain generalization
   - **Mitigation**: Hierarchical bootstrap (valid for n=3), conservative CI
   - **Acceptable**: Proof-of-concept, not production model

---

## Decision: Go/No-Go for Day 1-2

### Checklist for Proceeding to ENACT Extraction

- ✅ **Evaluation protocol locked** (cell-level, rigorous metrics)
- ✅ **SpatialEvaluator implemented** (comprehensive harness)
- ✅ **Methods cataloged** (70+ across 14 tiers, prioritized)
- ✅ **Baseline understood** (Prov-GigaPath + Hist2ST, SSIM 0.5699)
- ✅ **Architecture blueprint** (Clean Frankenstein modules specified)
- ✅ **Data available** (3 CRC patients, H&E WSIs, existing predictions)
- ✅ **Tools ready** (Memory graph, MCPs, evaluation harness)
- ✅ **Risks identified** (gaps analyzed, mitigations planned)

### Blockers: **NONE**

All prerequisites for Day 1-2 are met. We can proceed to ENACT extraction.

---

## Recommended Execution Strategy

### Week 2 Day-by-Day Plan (Optimized)

**Day 0-1** ✅ COMPLETE:
- Evaluation definition locked (cell-level target)
- SpatialEvaluator implemented (Wasserstein, scIB-E, hierarchical bootstrap)
- Metrics harness ready

**Day 1-2** 🔴 NEXT (THE MISSING PIECE):
1. Clone `Sanofi-Public/enact-pipeline`
2. Extract bin→cell assignment module (weighted-by-area)
3. Process 3 CRC patients → cell-level AnnData
4. Visualize: Naive bins vs ENACT assignment
5. **Critical Validation**: Jitter stress test (perturb coords, measure SSIM drop)
6. **Checkpoint**: If jitter-robust → model ignores spatial info (red flag)

**Day 2-3** 🔴 HIGH PRIORITY (FAST WIN):
1. Clone `SichenZhu/Stem` (clean diffusion baseline)
2. Search for PAST weights (best pretrained backbone)
3. "Hotwire" Stem data loader → ENACT-processed H5AD
4. Run training on 1 fold (P1+P2 train, P5 test)
5. Generate samples (observe stochasticity vs regression blur)
6. Run SpatialEvaluator on outputs
7. **Checkpoint**: Compare to baseline (Prov-GigaPath + Hist2ST, SSIM 0.5699)

**Day 3-4** 🔴 CRITICAL (MIL UPGRADE):
1. Clone `ratschlab/DeepSpot2Cell`
2. Extract DeepSet aggregator (bag-of-cells logic)
3. **Experiment**: Feed DeepSet embeddings into Stem as conditioning
4. **Hypothesis**: DeepSet captures sub-spot structure better than raw ViT patch
5. **Checkpoint**: SSIM improvement vs baseline?

**Day 4-5** 🟡 DECISION POINT (COUNT HANDLING):
- **Option A**: Stick with Stem (continuous diffusion)
- **Option B**: Swap to GenAR (discrete counts/tokens)
- **Decision Criteria**: Does Stem handle zero-inflation adequately?
- **Mitigation**: Implement PRTS two-head loss if zeros dominate

**Day 5** 📊 SYNTHESIS:
- Compile results: ENACT baseline, Stem baseline, DeepSpot-Stem hybrid
- Run negative controls (label shuffle, spatial jitter, random encoder)
- Write Day 1-5 report
- **Go/No-Go**: Proceed to Week 3 architectural synthesis?

---

## Alternative Execution Paths

### Conservative Path (Lower Risk, Slower)
1. Day 1-2: ENACT only
2. Day 3-5: Test existing methods (GHIST, TRIPLEX) on ENACT data
3. Week 3: Implement Stem if baselines fail
4. Week 4: Architectural synthesis

**Pros**: Less code debt, understand landscape first
**Cons**: Slower to novel architecture, may miss fast wins

### Aggressive Path (Higher Risk, Faster Innovation)
1. Day 1: ENACT + PAST weights search in parallel
2. Day 2: Implement ENACT + Stem + DeepSpot simultaneously
3. Day 3-4: Full "Clean Frankenstein" prototype (all modules)
4. Day 5: Benchmark vs baseline
5. Week 3: Ablation studies

**Pros**: Fast prototype, early evidence of approach
**Cons**: High code complexity, potential wasted effort if wrong

### **RECOMMENDED: Middle Path (Balanced)**
- Follow Day 0-5 plan above (sequential with checkpoints)
- Each day has clear success criteria
- Allows pivots at Days 2, 3, 4 based on results
- Minimizes rework while maintaining momentum

---

## Success Criteria (Pre-Registered)

### Day 1-2 Success
- ✅ ENACT pipeline extracted and functional
- ✅ 3 CRC patients processed → cell-level AnnData
- ✅ Visualization: ENACT vs naive assignment shows clear difference
- ✅ Jitter stress test: SSIM degrades appropriately with misalignment

### Day 2-3 Success
- ✅ Stem running on ENACT-processed data
- ✅ Predictions generated (observe sharp distributions vs regression blur)
- ✅ SpatialEvaluator results: SSIM, Wasserstein, bio-conservation computed
- ✅ At least **ONE metric** shows improvement vs baseline (even if small)

### Week 2 Success (Day 0-5 Complete)
- ✅ Baseline metrics established (ENACT + existing methods)
- ✅ Generative baseline established (Stem on cell-level data)
- ✅ Negative controls validate model learns biology (not artifacts)
- ✅ **Primary Criterion**: Evidence that generative > regression OR ENACT preprocessing > raw bins

### Week 2 Failure Modes (Pivot Triggers)
- ❌ ENACT processing shows no improvement over naive bins → re-evaluate bin assignment strategy
- ❌ Stem SSIM << baseline → diffusion may not be suitable, try STFlow or regression with better loss
- ❌ Negative controls fail → model learning artifacts, need architectural changes
- ❌ All metrics show high variance (per-patient SSIM range > 0.10) → need more data or better batch correction

---

## Long-Term Resource Needs (Month 3+)

### Month 3: Ablation Studies
- **Compute**: GPU cluster for encoder ablation (UNI, CONCH, Virchow2, PAST, Threads)
- **Code**: Modular architecture for swapping components
- **Time**: ~15 configurations × 3 folds = 45 training runs

### Month 5: External Validation
- **Data**: HEST-1k access (if available) OR Ken Lau CRC cohort
- **Metrics**: Cross-tissue/cross-cohort generalization
- **Claim Scope**: Conservative if no external validation

### Month 6: Publication
- **Figures**: LaTeX Architect MCP for IEEE/MICCAI style
- **Code**: Clean release, documentation, README
- **Weights**: HuggingFace model upload
- **Dataset**: Zenodo upload (if permitted)

---

## Conclusion and Recommendation

### Assessment: **READY TO PROCEED** ✅

**All prerequisites met**:
1. ✅ Evaluation protocol locked (rigorous, small-n safe)
2. ✅ Methods cataloged (70+, prioritized, GitHub links)
3. ✅ Architecture blueprint (Clean Frankenstein modules specified)
4. ✅ Implementation plan (Day 0-5 execution strategy)
5. ✅ Tools ready (SpatialEvaluator, MCPs, memory graph)
6. ✅ Data ready (3 CRC patients, baseline predictions)

**Critical gaps identified and mitigated**:
- 🔴 ENACT: Day 1-2 extraction (THE MISSING PIECE)
- 🔴 PAST weights: Search Day 1, fallback to UNI
- 🔴 Zero-inflation: PRTS two-head loss or OT loss (Day 3-4)

**No blockers**: Can proceed to Day 1-2 immediately.

### **RECOMMENDATION: PROCEED TO DAY 1-2 (ENACT EXTRACTION)**

**Next Steps**:
1. Clone `Sanofi-Public/enact-pipeline`
2. Extract weighted-by-area bin→cell assignment
3. Process P1, P2, P5 → cell-level AnnData
4. Visualize and validate (jitter stress test)
5. Establish cell-level baseline metrics

**Expected Outcome**: Clean, uncontaminated cell-level ground truth enabling all downstream generative modeling work.

**Timeline**: Day 1-2 complete by EOD 2025-12-27, ready for Stem baseline Day 2-3.

---

## Appendix: Method Count by Tier

| Tier | Category | Methods | Priority |
|------|----------|---------|----------|
| -1 | Infrastructure | 1 | 🔴 CRITICAL |
| 0 | Bleeding Edge | 9 | 🔴 HIGH |
| 1 | High Priority SOTA | 11 | 🔴 HIGH |
| 1.5 | Advanced Architectures | 5 | 🟡 MEDIUM |
| 2 | CVPR/Top Venues | 4 | 🟡 MEDIUM |
| 3 | Established Methods | 5 | 🟢 LOW |
| 4 | Encoder Baselines | 3 | 🟢 LOW |
| 5 | Pending Methods | 15+ | 🟢 LOW |
| 6 | Cross-Disciplinary | 6 | 🟡 MEDIUM |
| 7 | Evaluation Tools | 3 | 🔴 CRITICAL |
| 8 | Datasets/Benchmarks | 3 | 🟡 MEDIUM |
| 9 | Foundation Models | 5 | 🔴 HIGH |
| 10 | Gene Prediction Methods | 3 | 🟡 MEDIUM |
| 11 | Cross-Modal Alignment | 1 | 🟡 MEDIUM |
| 12 | MIL/WSI Aggregation | 2 | 🟡 MEDIUM |
| 13 | Generative Models | 1 | 🟢 LOW |
| **TOTAL** | | **70+** | |

---

**Status**: Synthesis complete, ready for execution.
**Author**: Claude Sonnet 4.5
**Project**: sota-2um-st-prediction
**Phase**: Month 1 Week 2 Day 0-1 → Day 1-2 transition
