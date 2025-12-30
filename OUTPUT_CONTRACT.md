# SOTA 2μm ST Prediction - Output Contract

**Date**: 2025-12-27
**Status**: REVISED - PCC is primary metric, not SSIM

---

## Critical Finding: SSIM is Inappropriate for Sparse ST Data

### Evidence from Label Shuffle Control (2025-12-27)

| Metric | Baseline | Global Shuffle | Expected | Verdict |
|--------|----------|----------------|----------|---------|
| **PCC** | 0.1001 | 0.0007 | ~0 | **PASS** |
| SSIM | 0.5710 | 0.4931 | <0.35 | FAIL |

**Root Cause Analysis:**
- ST data is 94.39% zeros (extremely sparse)
- Different ground truth patches have SSIM ~0.85 with each other (zeros match zeros)
- SSIM drop from correct→wrong label is only 6% (not sensitive to correctness)
- PCC correctly distinguishes correct from incorrect labels

**Conclusion**: SSIM measures structural texture similarity, not gene-specific spatial correspondence. PCC is the correct metric.

---

## The Problem

We're switching from bin-level to cell-level ground truth (via ENACT). If we change
what we're predicting, evaluation numbers become incomparable.

**Current baseline** (H_013: Prov-GigaPath + Poisson + Hist2ST):
- PCC @ 2μm: 0.1001 (primary)
- SSIM @ 2μm: 0.5699 (secondary, but unreliable)

**Target**: PCC ≥ 0.13 (+0.03 absolute) with clean targets beating old targets

---

## Output Contract

### We WILL:

1. **Continue predicting 2μm bin-level expression images**
   - Model output: Gene expression values at 2μm grid locations
   - Same spatial resolution as before
   - Same evaluation grid

2. **Use ENACT as a denoising/reassignment operator**
   - ENACT produces cell-level counts
   - We rasterize back to 2μm bins via area-weighted overlap
   - This creates "clean" 2μm targets

3. **Evaluate SSIM exactly as in H_013 baseline**
   - Same evaluation code
   - Same normalization
   - Same genes

### We WILL NOT:

1. Change the prediction task to per-cell prediction (yet)
2. Compare SSIM numbers that use different grid resolutions
3. Claim improvement without held-out patient validation

---

## Rasterization Method (Cell-level → 2μm bins)

### Area-Weighted Rasterization

For each 2μm bin `b`:

1. Find all cells `c` that overlap with `b`
2. Calculate overlap area `A(b,c)` between bin and cell
3. Compute weighted average:

```
target(b, gene) = Σ_c [A(b,c) × count(c, gene)] / Σ_c A(b,c)
```

This is essentially reversing ENACT's `weighted_by_area` assignment.

### Implementation

```python
def rasterize_cells_to_bins(cells_gdf, cell_counts, bins_gdf):
    """
    Project cell-level counts back to 2μm bin grid.

    Args:
        cells_gdf: GeoDataFrame with cell geometries
        cell_counts: sparse matrix (n_cells, n_genes)
        bins_gdf: GeoDataFrame with 2μm bin geometries

    Returns:
        bin_counts: sparse matrix (n_bins, n_genes)
    """
    from shapely.ops import intersection

    # Spatial join to find overlapping cells for each bin
    joined = gpd.sjoin(bins_gdf, cells_gdf, how='left', predicate='intersects')

    # For each bin, compute area-weighted average
    bin_counts = []
    for bin_idx in bins_gdf.index:
        overlapping = joined[joined.index == bin_idx]
        if len(overlapping) == 0:
            bin_counts.append(np.zeros(cell_counts.shape[1]))
            continue

        # Compute overlap areas
        bin_geom = bins_gdf.loc[bin_idx, 'geometry']
        weights = []
        cell_indices = []
        for _, row in overlapping.iterrows():
            cell_geom = cells_gdf.loc[row['index_right'], 'geometry']
            overlap_area = bin_geom.intersection(cell_geom).area
            weights.append(overlap_area)
            cell_indices.append(row['index_right'])

        # Weighted average
        weights = np.array(weights) / sum(weights)
        weighted_count = sum(w * cell_counts[i] for w, i in zip(weights, cell_indices))
        bin_counts.append(weighted_count)

    return np.array(bin_counts)
```

---

## Success Criteria (Revised)

### Phase A Success = Clean targets beat old targets

| Metric | Requirement | Priority |
|--------|-------------|----------|
| **PCC clean vs PCC old** | **+0.03 absolute (0.10 → 0.13)** | PRIMARY |
| SSIM clean vs SSIM old | Any improvement (secondary) | Secondary |
| Held-out patient (P5) | Clean model > old model on PCC | Required |
| Negative control (label shuffle) | **PCC craters to ~0** ✓ PASSED | Validated |
| Segmentation stability | Mask IoU >0.70 across methods | Required |

### Phase A Fail Conditions

- ~~Label shuffle doesn't crater~~ → **RESOLVED: PCC correctly drops, SSIM is broken**
- Clean targets don't help → label noise wasn't the problem
- Segmentation highly unstable → cell-level GT unreliable

---

## SSIM Evaluation Contract

### Current SSIM Computation (must match exactly)

```python
from skimage.metrics import structural_similarity

def compute_ssim(pred, target, data_range=None):
    """
    Compute SSIM for gene expression prediction.

    Both pred and target should be:
    - Shape: (H, W, n_genes) or (H, W) for single gene
    - Normalized to [0, 1] or consistent range
    """
    if data_range is None:
        data_range = max(target.max() - target.min(), 1e-8)

    return structural_similarity(
        pred, target,
        data_range=data_range,
        multichannel=(len(pred.shape) == 3)
    )
```

### Normalization (must be consistent)

```python
# Per-gene normalization to [0, 1]
def normalize_expression(expr):
    """Normalize expression per gene to [0, 1]."""
    for g in range(expr.shape[-1]):
        gene_expr = expr[..., g]
        min_val, max_val = gene_expr.min(), gene_expr.max()
        if max_val > min_val:
            expr[..., g] = (gene_expr - min_val) / (max_val - min_val)
    return expr
```

---

## Comparability Checklist

Before claiming any improvement, verify:

- [x] Same evaluation code as H_013 baseline
- [x] Same genes (50 panel)
- [x] Same held-out patient (P5)
- [x] Same normalization
- [x] Same spatial resolution (2μm)
- [x] **Label shuffle control passes - PCC drops to ~0** (PASSED 2025-12-27)
- [ ] Segmentation stability verified across methods

---

## PCC Evaluation Contract (NEW)

### Primary Metric: Pearson Correlation Coefficient

```python
from scipy.stats import pearsonr

def compute_pcc_per_gene(pred, target, mask=None):
    """
    Compute PCC for gene expression prediction.

    Args:
        pred: (H, W, n_genes) predicted expression
        target: (H, W, n_genes) ground truth expression
        mask: (H, W) tissue mask (optional)

    Returns:
        per_gene_pcc: (n_genes,) PCC for each gene
        mean_pcc: scalar mean PCC across genes
    """
    pcc_values = []
    n_genes = pred.shape[-1]

    for g in range(n_genes):
        p = pred[..., g].flatten()
        t = target[..., g].flatten()

        if mask is not None:
            m = mask.flatten() > 0.5
            p = p[m]
            t = t[m]

        if p.std() > 1e-6 and t.std() > 1e-6:
            r, _ = pearsonr(p, t)
            if not np.isnan(r):
                pcc_values.append(r)

    return np.array(pcc_values), np.mean(pcc_values)
```

### Why PCC over SSIM for Sparse ST Data

1. **PCC measures correlation** - directly tests if spatial patterns match
2. **Invariant to scale** - handles sparse count data correctly
3. **Validated by shuffle test** - correctly drops to ~0 with random labels
4. **Interpretable** - r=0.10 means 10% of variance explained

---

## Comprehensive Metrics Stack for 94%-Zero ST Data

### TL;DR

For **94%-zero** ST targets, the primary scoreboard should be **per-gene spatial correlation (PCC + rank metric)**, plus **a zero-aware detection metric** and **a count-likelihood metric**. SSIM stays as a *sanity/visual* metric only—the shuffle test proved it's not a decision metric.

---

### 1. Primary: Per-Gene Spatial PCC (on stabilized transform)

**Compute:**
- For each held-out **slide/patient** and each **gene**:
  - Transform counts: `x = log1p(counts)` (or `sqrt(counts)`)
  - Compute **Pearson r across spatial positions** (bins/cells) between prediction and truth

**Aggregate:**
- **Macro-average across genes** (each gene equal weight)
- Use **Fisher z-transform** when averaging r's (prevents weirdness at extremes)
- Report distribution: mean, median, q25, q75

```python
import numpy as np
from scipy.stats import pearsonr

def fisher_z(r):
    """Fisher z-transform for correlation averaging."""
    return 0.5 * np.log((1 + r) / (1 - r + 1e-8))

def inverse_fisher_z(z):
    """Inverse Fisher z-transform."""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def compute_pcc_fisher_averaged(pred, target, mask=None):
    """
    Compute per-gene PCC with Fisher z-averaging.

    Args:
        pred: (H, W, n_genes) predicted log1p expression
        target: (H, W, n_genes) ground truth log1p expression
        mask: (H, W) tissue mask (optional)

    Returns:
        dict with mean, median, q25, q75 of per-gene PCC
    """
    pcc_values = []
    n_genes = pred.shape[-1]

    for g in range(n_genes):
        p = pred[..., g].flatten()
        t = target[..., g].flatten()

        if mask is not None:
            m = mask.flatten() > 0.5
            p, t = p[m], t[m]

        if p.std() > 1e-6 and t.std() > 1e-6:
            r, _ = pearsonr(p, t)
            if not np.isnan(r):
                pcc_values.append(r)

    pcc_array = np.array(pcc_values)

    # Fisher z-transform for proper averaging
    z_values = fisher_z(pcc_array)
    mean_z = np.mean(z_values)
    fisher_mean = inverse_fisher_z(mean_z)

    return {
        'pcc_fisher_mean': fisher_mean,
        'pcc_median': np.median(pcc_array),
        'pcc_q25': np.percentile(pcc_array, 25),
        'pcc_q75': np.percentile(pcc_array, 75),
        'pcc_per_gene': pcc_array,
    }
```

### 2. Rank Backstop: Spearman ρ (Anti-Gaming)

Rank metrics are less sensitive to scaling and "global shrinkage" tricks.

```python
from scipy.stats import spearmanr

def compute_spearman_per_gene(pred, target, mask=None):
    """Spearman correlation as anti-gaming backstop."""
    spearman_values = []
    n_genes = pred.shape[-1]

    for g in range(n_genes):
        p = pred[..., g].flatten()
        t = target[..., g].flatten()

        if mask is not None:
            m = mask.flatten() > 0.5
            p, t = p[m], t[m]

        if p.std() > 1e-6 and t.std() > 1e-6:
            rho, _ = spearmanr(p, t)
            if not np.isnan(rho):
                spearman_values.append(rho)

    return np.mean(spearman_values), np.median(spearman_values)
```

### 3. Zero-Aware Detection: AUPRC (Critical for Sparse Data)

With 94% zeros, we must separately score:
1. "Did you predict **where it's nonzero**?" (detection)
2. "Given it's nonzero, did you predict the **magnitude**?" (conditional)

**Use AUPRC (not AUROC)** because positives are rare—AUROC will lie to you.

```python
from sklearn.metrics import average_precision_score

def compute_auprc_per_gene(pred, target, mask=None, threshold=0):
    """
    AUPRC for nonzero detection.
    Catches models that inflate PCC by predicting smooth low-amplitude haze.
    """
    auprc_values = []
    n_genes = pred.shape[-1]

    for g in range(n_genes):
        p = pred[..., g].flatten()
        t = target[..., g].flatten()

        if mask is not None:
            m = mask.flatten() > 0.5
            p, t = p[m], t[m]

        # Binarize ground truth: nonzero = positive
        y_true = (t > threshold).astype(int)

        # Only compute if we have both classes
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            auprc = average_precision_score(y_true, p)
            auprc_values.append(auprc)

    return np.mean(auprc_values), np.median(auprc_values)
```

### 4. Conditional Magnitude: PCC on Nonzero-Only Bins

```python
def compute_conditional_pcc(pred, target, mask=None):
    """PCC restricted to bins where ground truth is nonzero."""
    cond_pcc_values = []
    n_genes = pred.shape[-1]

    for g in range(n_genes):
        p = pred[..., g].flatten()
        t = target[..., g].flatten()

        if mask is not None:
            m = mask.flatten() > 0.5
            p, t = p[m], t[m]

        # Only nonzero ground truth
        nonzero_mask = t > 0
        if nonzero_mask.sum() > 10:  # Need enough samples
            p_nz = p[nonzero_mask]
            t_nz = t[nonzero_mask]

            if p_nz.std() > 1e-6 and t_nz.std() > 1e-6:
                r, _ = pearsonr(p_nz, t_nz)
                if not np.isnan(r):
                    cond_pcc_values.append(r)

    return np.mean(cond_pcc_values), np.median(cond_pcc_values)
```

### 5. Count Fidelity: Poisson NLL (Proper Scoring Rule)

```python
import torch

def compute_poisson_nll(pred_rate, target_counts, mask=None):
    """
    Poisson negative log-likelihood.
    Provides a proper scoring rule that correlation metrics don't.
    Catches cases where PCC improves but predictions are miscalibrated.
    """
    pred = torch.tensor(pred_rate, dtype=torch.float32)
    target = torch.tensor(target_counts, dtype=torch.float32)

    if mask is not None:
        mask_t = torch.tensor(mask, dtype=torch.float32)
        # Flatten and mask
        pred = pred.reshape(-1)[mask_t.flatten() > 0.5]
        target = target.reshape(-1)[mask_t.flatten() > 0.5]

    # Poisson NLL: -log P(k|λ) = λ - k*log(λ) + log(k!)
    # We use PyTorch's implementation
    nll = torch.nn.functional.poisson_nll_loss(
        pred.clamp(min=1e-6).log(),
        target,
        log_input=True,
        full=True,
        reduction='mean'
    )
    return nll.item()
```

### 6. Hotspot Overlap: Top-k Precision (Hard to Fake)

```python
def compute_topk_precision(pred, target, mask=None, k_percent=5):
    """
    Top-k% overlap (Precision@k).
    Hard to fake with smooth predictions.
    """
    precision_values = []
    n_genes = pred.shape[-1]

    for g in range(n_genes):
        p = pred[..., g].flatten()
        t = target[..., g].flatten()

        if mask is not None:
            m = mask.flatten() > 0.5
            p, t = p[m], t[m]

        k = max(1, int(len(t) * k_percent / 100))

        # Top-k indices in each
        pred_topk = set(np.argsort(p)[-k:])
        true_topk = set(np.argsort(t)[-k:])

        # Precision: what fraction of predicted hotspots are true hotspots
        precision = len(pred_topk & true_topk) / k
        precision_values.append(precision)

    return np.mean(precision_values), np.median(precision_values)
```

---

## Standard Reporting Block (Per Held-Out Patient)

```python
def compute_full_metrics(pred, target, mask=None):
    """
    Complete evaluation metrics for ST prediction.

    Args:
        pred: (H, W, n_genes) predicted expression (raw counts or rates)
        target: (H, W, n_genes) ground truth expression
        mask: (H, W) tissue mask

    Returns:
        dict with all metrics
    """
    # Transform to log1p for correlation metrics
    pred_log = np.log1p(pred)
    target_log = np.log1p(target)

    results = {}

    # 1. Primary: PCC with Fisher averaging
    pcc_results = compute_pcc_fisher_averaged(pred_log, target_log, mask)
    results.update(pcc_results)

    # 2. Rank backstop: Spearman
    results['spearman_mean'], results['spearman_median'] = \
        compute_spearman_per_gene(pred_log, target_log, mask)

    # 3. Zero detection: AUPRC
    results['auprc_mean'], results['auprc_median'] = \
        compute_auprc_per_gene(pred, target, mask)

    # 4. Conditional magnitude
    results['cond_pcc_mean'], results['cond_pcc_median'] = \
        compute_conditional_pcc(pred_log, target_log, mask)

    # 5. Count fidelity: Poisson NLL
    results['poisson_nll'] = compute_poisson_nll(pred, target, mask)

    # 6. Hotspot overlap
    results['topk5_precision_mean'], results['topk5_precision_median'] = \
        compute_topk_precision(pred, target, mask, k_percent=5)
    results['topk1_precision_mean'], results['topk1_precision_median'] = \
        compute_topk_precision(pred, target, mask, k_percent=1)

    return results
```

---

## Gene Set Stratification

Report metrics on three gene sets:

1. **All genes** (50 panel)
2. **HVGs** (highly variable genes)
3. **Top-K predictable genes** (secondary view only)

```python
def stratify_by_gene_sets(results_per_gene, gene_sets):
    """
    Aggregate metrics by gene set.

    Args:
        results_per_gene: dict with per-gene metric arrays
        gene_sets: dict mapping set_name -> gene indices
    """
    stratified = {}
    for set_name, indices in gene_sets.items():
        for metric_name, values in results_per_gene.items():
            if isinstance(values, np.ndarray) and len(values) > max(indices):
                subset = values[indices]
                stratified[f'{metric_name}_{set_name}'] = {
                    'mean': np.mean(subset),
                    'median': np.median(subset),
                }
    return stratified
```

---

## Updated Success Criteria (Complete)

### Minimum "Claim a Win" Gate

| Metric | Baseline | Target | Requirement |
|--------|----------|--------|-------------|
| **PCC (Fisher mean, log1p)** | 0.10 | **≥0.13** | Must improve |
| **AUPRC (nonzero detection)** | TBD | Higher than baseline | Must improve |
| Spearman | TBD | Higher than baseline | Anti-gaming check |
| Top-5% Precision | TBD | Higher than baseline | Hotspot quality |
| Poisson NLL | TBD | Lower than baseline | Calibration check |

### Negative Control Requirements

| Control | Expected |
|---------|----------|
| Label shuffle PCC | ~0 (validated ✓) |
| Label shuffle AUPRC | ~prevalence baseline (~6%) |
| Label shuffle top-k | ~random (~k/N) |

### What NOT to Use as Primary

| Metric | Use as | Why |
|--------|--------|-----|
| SSIM | Visual sanity only | Shuffle test proved broken for sparse data |
| Global MSE/RMSE | Secondary | Dominated by zeros |
| AUROC | Never | Lies with imbalanced classes |

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2025-12-27 | Initial contract defining SSIM comparability |
| 2.0 | 2025-12-27 | REVISED: PCC as primary metric after label shuffle analysis |
| 3.0 | 2025-12-27 | COMPREHENSIVE: Full metrics stack for 94%-zero ST data (PCC+AUPRC+Top-k+NLL) |

---

## References

1. [Gene expression prediction from histology images via hypergraph neural networks](https://academic.oup.com/bib/article/25/6/bbae500/7821151) - OUP Academic
2. [A visual–omics foundation model](https://pmc.ncbi.nlm.nih.gov/articles/PMC12240810/) - PMC
3. [GHIST: Spatial gene expression at single-cell resolution](https://www.nature.com/articles/s41592-025-02795-z) - Nature Methods
4. [Spatial Transcriptomics Analysis of Spatially Dense Gene Expression](https://arxiv.org/html/2503.01347v2) - arXiv
5. [DeepSpot2Cell: Predicting Virtual Single-Cell Spatial Transcriptomics](https://openreview.net/pdf/e128509ef03f6950b767510ff242522be4807a70.pdf) - OpenReview
6. [Scaling up spatial transcriptomics for large-sized tissues](https://www.nature.com/articles/s41592-025-02770-8) - Nature Methods
