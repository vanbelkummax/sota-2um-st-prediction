# REVISED Day 1-2 Plan: Conservative, Test-Driven Approach

**Date**: 2025-12-26 (Revised post-critical-feedback)
**Status**: Addresses Trap A (cell-level overconfidence) and Trap B (n=3 fragility)

---

## Critical Feedback Integration

### Identified Traps

1. **Trap A**: Treating ENACT outputs as "clean ground truth" (they're derived estimates with segmentation/registration/capture errors)
2. **Trap B**: Deferring robustness to Month 5 (with n=3, will overfit to batch effects)

### Strategic Revisions

1. **Measurement Model Paradigm**: Keep bins, model mixtures with overlap weights (don't collapse to "cell labels")
2. **Minimum Viable SOTA First**: ZINB baseline BEFORE diffusion/INR/Longformer
3. **Killer Validations**: Segmentation sensitivity + registration optimization checks
4. **Robustness Earlier**: Stain augmentation + TTA in Week 2, not Month 5

---

## Revised Week 2 Goal

**Beat baseline (0.5699 SSIM) + Show spatial sensitivity + Show robustness curves**

NOT: "Implement full Clean Frankenstein with 6 modules"

---

## Day 1: ENACT Installation + Segmentation Sensitivity

### Morning: Full Pipeline Run (Don't Extract Yet)

**Tasks**:
1. Clone ENACT:
   ```bash
   cd /home/user/work/code_archaeology
   git clone https://github.com/Sanofi-Public/enact-pipeline
   ```

2. Environment setup:
   ```bash
   conda create -n enact python=3.9
   conda activate enact
   pip install -r enact-pipeline/requirements.txt
   # Key dependencies: stardist, geopandas, scanpy
   ```

3. **Run full ENACT pipeline on P1** (single patient, test run):
   - Input: H&E WSI (pyramidal tiff) + Visium HD bin counts (HDF5)
   - Output: Cell-level AnnData with overlap weights
   - **DO NOT extract code yet** - validate outputs first

### Afternoon: Killer Validation #1 - Segmentation Sensitivity Sweep

**Hypothesis**: If ENACT outputs are stable across segmentation methods, can use as weak labels. If unstable, MUST use measurement model.

**Procedure**:
1. Run ENACT with **StarDist default** (standard nuclei model):
   ```python
   # Output: cells_stardist_default.h5ad
   ```

2. Run ENACT with **StarDist fine-tuned** (if H&E model available):
   ```python
   # Output: cells_stardist_finetuned.h5ad
   ```

3. Run ENACT with **Alternative segmenter** (CellPose as fallback):
   ```python
   # Output: cells_cellpose.h5ad
   ```

4. **Quantify cell expression variance**:
   ```python
   # For each gene, compute:
   # - Pearson r between methods (cell-wise expression)
   # - Coefficient of variation across methods
   # - Silhouette score difference (cell-type clustering)
   ```

**Decision Criteria**:
- ✅ **Stable** (Pearson r > 0.9, CV < 20%): Can use ENACT cells as weak labels
- ❌ **Unstable** (Pearson r < 0.8, CV > 30%): MUST implement measurement model

### End of Day 1 Deliverables

- ✅ ENACT pipeline functional on P1
- ✅ Segmentation sensitivity quantified
- ✅ **Go/No-Go Decision**: Use ENACT cells directly OR implement measurement model

**Checkpoint**: If segmentation unstable, Day 2 shifts to implementing bin-level mixture model instead of cell-level baseline.

---

## Day 2: Registration Check + Baseline Metrics

### Morning: Killer Validation #2 - Registration Optimization

**Hypothesis**: If metrics improve with small affine shift, registration error dominates (not model error).

**Procedure**:
1. Grid search affine transformations (H&E ↔ Visium grid):
   ```python
   import scipy.ndimage as ndi

   # Translation sweep
   for dx in [-20, -10, -5, 0, 5, 10, 20]:  # pixels
       for dy in [-20, -10, -5, 0, 5, 10, 20]:
           shifted_coords = coords + [dx, dy]
           ssim_shifted = compute_ssim(pred, true, shifted_coords)
           # Track: (dx, dy, ssim)

   # Rotation sweep (if translation insufficient)
   for angle in [-5, -2, -1, 0, 1, 2, 5]:  # degrees
       rotated_coords = rotate_coords(coords, angle)
       ssim_rotated = compute_ssim(pred, true, rotated_coords)
   ```

2. **Quantify registration error ceiling**:
   ```python
   best_shift = argmax(ssim_grid)
   registration_gain = ssim[best_shift] - ssim[identity]

   # If registration_gain > 0.05 SSIM → alignment is a major bottleneck
   ```

**Decision Criteria**:
- ✅ **Registration OK** (gain < 5% SSIM): Proceed with standard pipeline
- ⚠️ **Registration Problem** (gain > 10% SSIM): Add registration optimization to preprocessing

### Afternoon: Naive vs ENACT Baseline Comparison

**Goal**: Quantify ENACT improvement over naive centroid assignment.

**Procedure**:
1. Generate **Naive Baseline** (centroid-based bin→cell):
   ```python
   # Assign bin to cell if centroid inside segmentation polygon
   naive_cells = assign_bins_naive(bins, segmentation_masks)
   naive_cells.write_h5ad('cells_naive.h5ad')
   ```

2. Generate **ENACT Baseline** (weighted-by-area):
   ```python
   # Already have from Day 1
   enact_cells = sc.read_h5ad('cells_enact_weighted_area.h5ad')
   ```

3. **Compute quality metrics**:
   ```python
   def compare_baselines(naive, enact):
       metrics = {}

       # Sparsity
       metrics['sparsity_naive'] = (naive.X == 0).mean()
       metrics['sparsity_enact'] = (enact.X == 0).mean()

       # Entropy (distribution uniformity)
       metrics['entropy_naive'] = scipy.stats.entropy(naive.X.mean(axis=0))
       metrics['entropy_enact'] = scipy.stats.entropy(enact.X.mean(axis=0))

       # Cell-type separability (if cell types available)
       sc.pp.neighbors(naive); sc.tl.leiden(naive)
       sc.pp.neighbors(enact); sc.tl.leiden(enact)
       metrics['ari_naive'] = ari(true_labels, naive.obs['leiden'])
       metrics['ari_enact'] = ari(true_labels, enact.obs['leiden'])

       return metrics
   ```

**Expected Results**:
- ENACT should reduce sparsity (fewer zeros)
- ENACT should increase entropy (less uniform, more biological)
- ENACT should increase ARI (better cell-type separation)

### End of Day 2 Deliverables

- ✅ Overlap weights w_bc extracted (for measurement model)
- ✅ Registration error quantified (know alignment ceiling)
- ✅ Naive vs ENACT baseline metrics (quantify improvement)
- ✅ **Decision Point**: Proceed to ZINB baseline (Day 3) OR pivot to full measurement model

**Checkpoint Criteria**:
- ✅ **Proceed to ZINB**: If segmentation stable AND registration error < 10%
- ⚠️ **Implement Measurement Model First**: If segmentation unstable OR registration > 10%

---

## Day 3-4: Minimum Viable SOTA (ZINB Baseline)

**NOTE**: Only proceed here if Day 1-2 checkpoints pass. Otherwise, implement measurement model.

### Architecture: Simplified (Not Full Frankenstein)

```python
class MinimumViableSOTA(nn.Module):
    def __init__(self, n_genes):
        self.encoder = load_pretrained('UNI')  # Frozen
        self.encoder.requires_grad_(False)

        # Simple spatial context (k-NN aggregation)
        self.context_agg = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # ZINB parameters (per cell)
        self.mu_head = nn.Linear(256, n_genes)      # Mean
        self.theta_head = nn.Linear(256, n_genes)   # Dispersion (log scale)
        self.pi_head = nn.Linear(256, n_genes)      # Zero-inflation prob

    def forward(self, patches, neighbor_indices):
        # Encode patches
        features = self.encoder(patches)  # (batch, 1024)

        # Aggregate neighbors (simple mean)
        neighbor_feats = features[neighbor_indices]  # (batch, k, 1024)
        context = neighbor_feats.mean(dim=1)  # (batch, 1024)

        # Context fusion
        fused = self.context_agg(context)  # (batch, 256)

        # Predict ZINB parameters
        mu = torch.exp(self.mu_head(fused))  # Ensure positive
        theta = torch.exp(self.theta_head(fused))  # Ensure positive
        pi = torch.sigmoid(self.pi_head(fused))  # [0, 1]

        return mu, theta, pi

def zinb_loss(y_obs, mu, theta, pi):
    """
    Zero-Inflated Negative Binomial loss.

    Args:
        y_obs: Observed counts (batch, n_genes)
        mu: Predicted mean (batch, n_genes)
        theta: Dispersion (batch, n_genes)
        pi: Zero-inflation probability (batch, n_genes)
    """
    # Mixture of point mass at zero + NB
    prob_zero = pi + (1 - pi) * (theta / (theta + mu)) ** theta
    prob_nonzero = (1 - pi) * nb_logprob(y_obs, mu, theta)

    # Negative log-likelihood
    nll = -torch.where(
        y_obs == 0,
        torch.log(prob_zero + 1e-10),
        torch.log(prob_nonzero + 1e-10)
    )

    return nll.sum(dim=1).mean()
```

### Training Loop

```python
# If using measurement model (bins with mixture weights)
for epoch in range(n_epochs):
    for batch in dataloader:
        patches, bin_counts, overlap_weights = batch

        # Predict cell latents
        mu_cells, theta_cells, pi_cells = model(patches, neighbor_indices)

        # Aggregate to bin level (mixture model)
        mu_bins = (overlap_weights @ mu_cells.T).T  # Sum over cells
        theta_bins = (overlap_weights @ theta_cells.T).T
        pi_bins = (overlap_weights @ pi_cells.T).T

        # Loss on observed bins
        loss = zinb_loss(bin_counts, mu_bins, theta_bins, pi_bins)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# If using ENACT cells directly (if segmentation stable)
for epoch in range(n_epochs):
    for batch in dataloader:
        patches, cell_counts = batch

        # Predict cell expression
        mu, theta, pi = model(patches, neighbor_indices)

        # Loss on ENACT cells (with uncertainty weighting)
        loss = zinb_loss(cell_counts, mu, theta, pi)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Success Criteria (Day 3-4)

1. ✅ ZINB model trains and converges (loss decreases)
2. ✅ ZINB SSIM > baseline (0.5699)
3. ✅ Negative controls pass:
   - Label shuffle → SSIM collapses
   - Spatial jitter (20μm) → SSIM degrades
4. ✅ **NEW**: NB likelihood better than MSE baseline
5. ✅ **NEW**: Stain augmentation doesn't cause catastrophic failure

---

## Day 5: Synthesis + Decision Point

### Morning: Component Attribution

**Goal**: Understand which components drive performance.

**Ablations**:
1. Encoder only (no context) vs Full model
2. MSE loss vs ZINB loss
3. ENACT preprocessing vs Naive
4. With stain augmentation vs Without

**Output**: Table of ablation results, identify critical components.

### Afternoon: Robustness Curves

**Goal**: Quantify generalization before proceeding to complex models.

**Tests**:
1. **Stain perturbation sweep**:
   ```python
   for h_shift in [0, 0.05, 0.1, 0.2]:
       for s_scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
           augmented = apply_stain_jitter(images, h_shift, s_scale)
           ssim_aug = evaluate(model, augmented)
           # Plot: augmentation severity vs SSIM
   ```

2. **Cross-patient variance**:
   ```python
   # Compute per-patient SSIM
   ssim_p1, ssim_p2, ssim_p5 = evaluate_per_patient(model)
   variance = np.var([ssim_p1, ssim_p2, ssim_p5])

   # If variance > 0.10 → high patient heterogeneity
   ```

### Decision Point: Proceed to Diffusion?

**Criteria**:
- ✅ **Stick with ZINB** if:
  - ZINB SSIM > 0.60 (strong baseline)
  - Robustness curves flat (no catastrophic failure)
  - NB likelihood competitive

- ⚠️ **Add Diffusion** if:
  - Evidence of multimodality (ZINB underfits)
  - ZINB < 0.60 (need more expressivity)

- ❌ **Debug First** if:
  - ZINB < baseline (implementation bug)
  - Negative controls fail (model not learning biology)
  - High patient variance (batch effect problem)

---

## Implementation Files (Organized)

### Directory Structure
```
/home/user/work/sota-2um-st-prediction/
├── data/
│   ├── raw/
│   │   ├── P1_H&E.tif
│   │   ├── P1_visium_hd.h5ad
│   │   ├── P2_H&E.tif
│   │   ├── P2_visium_hd.h5ad
│   │   ├── P5_H&E.tif
│   │   └── P5_visium_hd.h5ad
│   ├── enact_outputs/
│   │   ├── P1_cells_naive.h5ad
│   │   ├── P1_cells_enact_weighted.h5ad
│   │   ├── P1_overlap_weights.npz
│   │   └── ... (P2, P5)
│   └── baselines/
│       ├── P1_baseline_pred.h5ad  # Prov-GigaPath + Hist2ST
│       └── ... (P2, P5)
├── code/
│   ├── preprocessing/
│   │   ├── enact_wrapper.py  # ENACT pipeline interface
│   │   ├── segmentation_sensitivity.py
│   │   └── registration_optimization.py
│   ├── models/
│   │   ├── minimum_viable_sota.py  # ZINB model
│   │   ├── zinb_loss.py
│   │   └── measurement_model.py  # If needed
│   └── evaluation/
│       ├── spatial_evaluator.py  # Already have
│       └── robustness_curves.py
├── results/
│   ├── day1_segmentation_sensitivity.csv
│   ├── day2_registration_check.csv
│   ├── day2_naive_vs_enact.csv
│   └── day5_ablations.csv
└── docs/
    └── plans/
        └── REVISED_Day1-2_Conservative_Plan.md  # This file
```

---

## Risk Mitigation

### If Segmentation Unstable (Day 1 Checkpoint Fail)
- **Action**: Implement full measurement model (keep bins, model mixtures)
- **Timeline**: Adds 1-2 days to Week 2
- **Benefit**: More principled, avoids overfitting to ENACT artifacts

### If Registration Error High (Day 2 Checkpoint Fail)
- **Action**: Add registration optimization preprocessing step
- **Implementation**: Grid search + affine transform before ENACT
- **Timeline**: +0.5 days

### If ZINB < Baseline (Day 3-4 Checkpoint Fail)
- **Action**: Debug (check data loading, loss computation, hyperparams)
- **Fallback**: MSE baseline + ENACT (simpler, known to work)
- **Timeline**: +1 day debugging

---

## Success Metrics Summary

### Day 1-2 Success
1. ✅ ENACT pipeline functional
2. ✅ Segmentation sensitivity quantified (CV, Pearson r)
3. ✅ Registration error quantified (gain from shift)
4. ✅ Naive vs ENACT improvement demonstrated
5. ✅ Overlap weights w_bc extracted

### Day 3-4 Success (Minimum Viable SOTA)
1. ✅ ZINB model trains
2. ✅ ZINB SSIM > 0.5699
3. ✅ NB likelihood > MSE baseline
4. ✅ Negative controls pass
5. ✅ Stain robustness demonstrated

### Day 5 Success (Synthesis)
1. ✅ Component attribution (know what drives performance)
2. ✅ Robustness curves (quantify generalization)
3. ✅ Go/No-Go decision on diffusion (data-driven)

---

## Conservative Claims (Pre-Registered)

**DO NOT Claim** (even if metrics good):
- ❌ "SOTA across all tissues" (only have CRC)
- ❌ "Generalizes to all genes" (only 50-gene panel)
- ❌ "Production-ready" (only n=3, no deployment testing)

**CAN Claim** (if validation passes):
- ✅ "ZINB handles 2μm sparsity better than MSE"
- ✅ "ENACT preprocessing reduces contamination vs naive"
- ✅ "Measurement model propagates uncertainty"
- ✅ "Robustness to stain variation demonstrated (in vitro)"

---

## End of Day 2 Go/No-Go Decision Tree

```
Day 1: Segmentation Sensitivity
├─ Pearson r > 0.9, CV < 20%
│  └─ ✅ STABLE → Can use ENACT cells as weak labels
│     └─ Proceed to Day 2 (Registration Check)
└─ Pearson r < 0.8, CV > 30%
   └─ ❌ UNSTABLE → MUST use measurement model
      └─ Pivot to bin-level mixture model (adds 1-2 days)

Day 2: Registration Optimization
├─ Gain < 5% SSIM
│  └─ ✅ REGISTRATION OK → Proceed to Day 3 (ZINB baseline)
└─ Gain > 10% SSIM
   └─ ⚠️ REGISTRATION PROBLEM → Add optimization step
      └─ Proceed to Day 3 with registration fix

Day 3-4: ZINB Baseline
├─ SSIM > 0.5699
│  └─ ✅ BEATING BASELINE → Proceed to Day 5 (synthesis)
└─ SSIM < 0.5699
   └─ ❌ UNDERPERFORMING → Debug (data, loss, hyperparams)
      └─ Fallback: MSE + ENACT if ZINB unfixable

Day 5: Decision Point
├─ ZINB SSIM > 0.60 + Robust
│  └─ ✅ STICK WITH ZINB → Month 2: Ablations, not diffusion
├─ ZINB < 0.60 + Evidence of multimodality
│  └─ ⚠️ ADD DIFFUSION → Month 2: Implement Stem on top of ZINB
└─ High variance or control failures
   └─ ❌ DEBUG FIRST → Don't add complexity until baseline solid
```

---

**Status**: REVISED, CONSERVATIVE, TEST-DRIVEN
**Ready**: Addresses Trap A (measurement model) + Trap B (early robustness)
**Next**: Await user confirmation, then execute Day 1 plan

