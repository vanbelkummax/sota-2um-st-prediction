# Day 0-1: Evaluation Definition Lock

**Date**: 2025-12-26
**Phase**: Month 1 Week 2 - Day 0-1
**Status**: Decision document

---

## Critical Decision: Evaluation Target

### Decision: **Cell-Level Prediction**

**Rationale**:
1. **2μm bins are too sparse** for direct prediction
   - ~90% zeros for most genes (capture stochasticity, not biology)
   - MSE regression collapses to zero predictions
2. **8μm bins** aggregate away the resolution we're trying to achieve
3. **Cell-level** provides:
   - Biological ground truth (ENACT assigns bins to cells)
   - Proper aggregation reduces noise while preserving resolution
   - Valid statistical units (cells, not arbitrary grid squares)

**Implementation**:
- Use ENACT preprocessing to aggregate 2μm bins → cell-level gene counts
- Train models to predict cell-level expression
- Evaluate at cell-level using biological metrics

---

## Dataset Selection

### Primary Dataset: **3 CRC Patients (P1, P2, P5)**

**Justification**:
- **Existing baseline**: Prov-GigaPath + Hist2ST = SSIM 0.5699 on this exact data
- **Immediate comparability**: Can directly compare new methods to baseline
- **Domain match**: CRC is our target application (Lau lab collaboration)
- **HEST-1k availability uncertain**: If HEST-1k becomes available later, can validate generalization

**Dataset Specifications**:
- **Platform**: Visium HD 2μm resolution
- **Tissue**: Colorectal cancer (3 patients)
- **Genes**: 50-gene panel (epithelial, immune, stromal, housekeeping)
- **Resolution**: 128×128 spatial bins per patient
- **Ground Truth**: After ENACT processing → cell-level AnnData

### Secondary Dataset (Month 5): **HEST-1k** (if accessible)

**Purpose**: Cross-tissue generalization validation
- 9 organs, 8 cancer types
- 11 foundation model benchmarks
- 50 HVG prediction tasks at 112×112 μm @ 0.5 μm/px

---

## Evaluation Protocol (Nested Cross-Validation)

### Outer Loop (Final Evaluation): 3-Fold LOOCV

- **Fold 1**: Train P1+P2 → Test P5
- **Fold 2**: Train P1+P5 → Test P2
- **Fold 3**: Train P2+P5 → Test P1

### Inner Loop (Hyperparameter Tuning): Per Outer Fold

**Example for Fold 1**:
1. Use P1 for development (train subset)
2. Validate hyperparameters on P2 (validation)
3. Select best hyperparameters
4. Retrain on full P1+P2 with best hyperparameters
5. Evaluate **once** on P5 (locked test set)

**Critical Rule**: Test patient (P5 in this example) is **never** used for hyperparameter selection.

---

## Evaluation Metrics

### Primary Metrics (Always Report)

1. **SSIM @ Cell-Level**
   - Computation: Per-gene structural similarity on spatial expression maps
   - Aggregation: Mean across 50 genes (primary), Median (robustness check)
   - Per-category means: Epithelial, immune, stromal, housekeeping

2. **Wasserstein Distance (Optimal Transport)**
   - Purpose: Geometric appropriateness of spatial predictions
   - Advantage: Recognizes nearby cells (B-cell 10μm away ≠ complete mismatch)
   - Implementation: POT library (`ot.wasserstein_distance`)
   - **Lower is better** (unlike SSIM/Pearson)

### Gene-Centric Metrics

3. **Pearson Correlation (per-gene)**
   - Correlation across all cells
   - Report: Mean, median, per-category means
   - Count: Genes with r > 0.5, r > 0.8

4. **Spearman Correlation (per-gene)**
   - Rank correlation (robust to outliers)
   - Report: Mean, median

### Biological Conservation Metrics (scIB-E Framework)

5. **Adjusted Rand Index (ARI)**
   - Cluster agreement between predicted and true expression
   - Tests: Does model preserve cell types?

6. **Normalized Mutual Information (NMI)**
   - Information shared between clusterings

7. **Silhouette Score (on cell types)**
   - Cluster separation quality

8. **Local Inverse Simpson's Index (LISI)**
   - Local diversity/mixing of cell types

**Implementation**: `scib-metrics` Python library

### Spatial Fidelity Metrics

9. **Spatially Variable Gene (SVG) Recovery**
   - Do predicted genes show correct spatial patterns?
   - Method: Identify top SVGs in ground truth, check if predicted data ranks them similarly

10. **Moran's I (Spatial Autocorrelation)**
    - Per-gene spatial clustering
    - Tests: Do spatially correlated genes in ground truth remain correlated in predictions?

### Biological Validity

11. **Marker Concordance**
    - Test: EPCAM (epithelial) vs VIM (stromal) anticorrelation
    - Validates biological plausibility

12. **Pathway Alignment**
    - Gene Ontology enrichment of top predicted genes
    - Should match CRC biology (proliferation, immune response, EMT)

---

## Statistical Testing (Small-n Safe)

### DO NOT USE: Paired t-test with n=3 (underpowered, invalid)

### USE: Hierarchical Bootstrap

**Procedure**:
1. Bootstrap at patient level (n=3 with replacement)
2. For each bootstrap sample:
   - Compute metric for novel method
   - Compute metric for baseline
   - Calculate difference: Δ = novel - baseline
3. Repeat 10,000 times
4. Report:
   - **Effect size**: Mean Δ
   - **95% Confidence Interval**: [2.5th percentile, 97.5th percentile]
   - **Cohen's d**: Standardized effect size

**Example Report**:
```
Novel method: SSIM 0.XXX
Baseline:      SSIM 0.5699
Mean difference: +0.XXX (95% CI: [0.XXX, 0.XXX])
Cohen's d: XXX (small/medium/large effect)
```

**Interpretation**:
- If 95% CI excludes zero → statistically distinguishable improvement
- Effect size magnitude matters more than p-value

---

## Negative Controls (Proof of Biological Learning)

Run alongside all evaluations to prove model learns biology, not artifacts:

1. **Label Shuffle Control**
   - Randomly permute gene labels
   - Expected: SSIM should collapse
   - Interpretation: If SSIM remains high, model ignoring gene identity

2. **Spatial Jitter Control**
   - Add random coordinate offsets (±2, 5, 10, 20μm)
   - Expected: SSIM should drop with increasing jitter
   - Interpretation: If SSIM robust to 20μm jitter → model ignoring spatial info

3. **Random Encoder Control**
   - Replace pretrained encoder with random ViT initialization
   - Expected: SSIM << baseline
   - Interpretation: If SSIM similar → morphology not being used

4. **Smooth Random Field Control**
   - Generate synthetic spatial patterns (Gaussian processes)
   - Expected: Low SSIM, no biological pathway enrichment
   - Interpretation: Baseline for spatial structure without biology

---

## Data Quality Control

### Registration QC (Critical at 2μm)

1. **Manual Alignment Inspection**
   - Overlay H&E patches on coordinates
   - Inspect 10 random spots per patient
   - Flag misalignments > 10μm (5 bins at 2μm)

2. **Jitter Stress Test**
   - Perturb coordinates by 5, 10, 20μm
   - Measure SSIM degradation
   - **Warning**: If SSIM robust to 20μm jitter → model ignoring spatial information

### Tissue Masking

1. **Background Removal**
   - Exclude bins with UMI count < 10
   - Report % bins excluded per patient

2. **Outlier Detection**
   - Flag cells with z-score > 3 (total UMI or specific genes)
   - Report % cells flagged per patient

---

## Claim Scope (Conservative, Pre-Registered)

### Primary Claim (Always Safe)

**"Best performance for Visium HD CRC at 2μm on 50-gene panel"**

- Domain: Colorectal cancer
- Platform: Visium HD 2μm
- Genes: 50-gene panel (epithelial, immune, stromal, housekeeping)
- Sample size: 3 patients

### DO NOT Claim (Without External Validation)

- ❌ "Universal SOTA for all tissues"
- ❌ "Generalizes to all genes"
- ❌ "Works at all resolutions"

### Conditional Claims (Only if Validated)

- ✅ **IF tested on Ken Lau CRC cohort** → "Generalizes across CRC cohorts"
- ✅ **IF tested on HEST-1K** → "Generalizes across tissues"
- ✅ **IF tested on full transcriptome** → "Scales to 18K genes"

### Success Thresholds (Pre-Registered)

- **Minimum viable**: SSIM > 0.60, 95% CI excludes 0.5699
- **Strong claim**: SSIM > 0.65 + external cohort validation
- **Exceptional**: SSIM > 0.70 + ≥2 tissue types

### Heterogeneity Check

**If per-patient variance > 0.10 SSIM**:
- Treat as patient heterogeneity
- Report per-patient results
- Do NOT make broad generalization claim

---

## Implementation Checklist (Day 0-1)

- [ ] Create `SpatialEvaluator` class
- [ ] Implement `compute_wasserstein()` using POT library
- [ ] Implement `compute_bio_conservation()` using scib-metrics
- [ ] Implement `compute_svg_recovery()`
- [ ] Implement `compute_morans_i()`
- [ ] Implement hierarchical bootstrap function
- [ ] Create negative control generators (label shuffle, spatial jitter, random encoder)
- [ ] Document metric computation formulas
- [ ] Create example usage notebook

---

## Next Steps (Day 1-2)

Once evaluation harness is complete:
1. Clone ENACT pipeline
2. Process 3 CRC patients → cell-level AnnData
3. Visualize cell-level ground truth vs naive 2μm bins
4. Establish baseline metrics on existing predictions (Prov-GigaPath + Hist2ST)

---

## References

- **Optimal Transport**: POT library (https://github.com/PythonOT/POT)
- **Bio Conservation**: scIB-E framework (https://github.com/theislab/scib)
- **Nested CV**: Prevents test leakage in hyperparameter search
- **Bootstrap**: Valid for small n (3 patients)
- **Project Aether**: Strategic blueprint for mathematical rigor

---

**Evaluation Lock Date**: 2025-12-26
**Signed Off By**: Claude (sota-2um-st-prediction project)
**Status**: LOCKED - Do not modify metrics after implementation begins
