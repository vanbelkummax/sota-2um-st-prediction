# Critical Metrics Discovery - 2025-12-27

## Executive Summary

**SSIM is fundamentally inappropriate for sparse spatial transcriptomics evaluation.**

PCC (Pearson Correlation Coefficient) is the correct primary metric.

---

## Evidence

### Label Shuffle Negative Control

| Metric | Baseline | Global Shuffle | Change | Verdict |
|--------|----------|----------------|--------|---------|
| **PCC** | 0.1001 | 0.0007 | -99.3% | **CORRECT** |
| SSIM | 0.5710 | 0.4931 | -13.6% | BROKEN |

**Expected**: Both metrics should crater with shuffled labels.
**Observed**: PCC correctly drops to ~0, but SSIM barely moves.

### Root Cause Analysis

1. **ST data is 94.39% zeros** (extremely sparse)
2. **Different patches have SSIM ~0.85 with each other** (zeros match zeros)
3. **SSIM drop from correct→wrong label is only 6%** (not sensitive to correctness)
4. **Model predictions have ~50% pairwise SSIM** (similar patterns)

### Why SSIM Fails for Sparse Data

SSIM was designed for dense images (photos, medical scans):
- Computes similarity within 7×7 local windows
- In sparse data, most windows are all zeros
- Zeros match zeros → inflated similarity
- Structural texture dominates over actual gene correspondence

---

## Updated Evaluation Protocol

### Primary Metric: PCC (Pearson Correlation Coefficient)

```python
from scipy.stats import pearsonr

def compute_pcc_per_gene(pred, target, mask=None):
    pcc_values = []
    for g in range(pred.shape[-1]):
        p = pred[..., g].flatten()
        t = target[..., g].flatten()
        if mask is not None:
            m = mask.flatten() > 0.5
            p, t = p[m], t[m]
        if p.std() > 1e-6 and t.std() > 1e-6:
            r, _ = pearsonr(p, t)
            if not np.isnan(r):
                pcc_values.append(r)
    return np.mean(pcc_values)
```

### Secondary Metrics (Optional)

- **SSIM**: Keep for backwards compatibility, but don't use for success criteria
- **Poisson Deviance**: More appropriate for count data
- **Gene-wise Spearman**: Robust to outliers

---

## Updated Success Criteria

### Phase A: ENACT Clean Targets

| Metric | Baseline | Target | Priority |
|--------|----------|--------|----------|
| **PCC** | 0.10 | **0.13** (+0.03) | PRIMARY |
| SSIM | 0.57 | Any improvement | Secondary |

### Validation Checklist

- [x] Label shuffle control passes (PCC drops to ~0)
- [ ] Clean targets improve PCC on held-out patient
- [ ] Segmentation is stable across methods

---

## Implications for ENACT Work

1. **The label shuffle "failure" is actually a success** - we found a real problem
2. **Continue with ENACT** - the hypothesis that clean targets help is still valid
3. **Focus on PCC improvement** - target +0.03 absolute (0.10 → 0.13)
4. **Keep SSIM for reference** - but don't gate progress on it

---

## Comprehensive Metrics Stack (Added v3.0)

For **94%-zero** ST targets, use this multi-metric scoreboard:

| Category | Metric | Purpose |
|----------|--------|---------|
| **Primary** | PCC (Fisher z-avg, log1p) | Spatial pattern correlation |
| **Rank Backstop** | Spearman ρ | Anti-gaming check |
| **Zero Detection** | AUPRC (not AUROC!) | Nonzero bin prediction |
| **Conditional** | PCC on nonzero-only | Magnitude accuracy |
| **Count Fidelity** | Poisson NLL | Proper scoring rule |
| **Hotspot** | Top-k% Precision | Hard to fake |

### Success Gate
- PCC improves **AND**
- AUPRC improves **AND**
- Negative control passes (shuffle → ~0)

### What NOT to Use
- SSIM: Visual sanity only (shuffle test proved broken)
- Global RMSE: Dominated by zeros
- AUROC: Lies with imbalanced classes

See `OUTPUT_CONTRACT.md` v3.0 for full implementation code.

---

## Files Updated

- `/home/user/sota-2um-st-prediction/OUTPUT_CONTRACT.md` - Version 2.0 with PCC focus
- `/home/user/work/encoder-loss-ablation-2um/scripts/diagnose_ssim_failure.py` - Diagnostic analysis
- `/home/user/work/encoder-loss-ablation-2um/scripts/analyze_model_predictions.py` - Prediction analysis

## Diagnostic Scripts

```bash
# Run SSIM diagnostic
cd /home/user/work/encoder-loss-ablation-2um
python scripts/diagnose_ssim_failure.py

# Analyze model predictions
python scripts/analyze_model_predictions.py
```
