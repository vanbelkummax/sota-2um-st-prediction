# Month 2-3 Implementation Roadmap: SOTA 2μm Strategy

**Date**: 2025-12-26
**Phase**: Post-Literature Synthesis → Execution Planning
**Timeline**: 8 weeks (Weeks 5-12, January-February 2026)
**Current Baseline**: Prov-GigaPath + Hist2ST + MSE = SSIM 0.5699 @ 2μm

---

## Executive Summary

### High-Level Strategy

**Conservative, test-driven approach with mandatory checkpoints**. We build on **Agent 1's literature synthesis** (70+ methods, ENACT infrastructure gap identified), **Agent 2's hypothesis prioritization** (ZINB/Focal loss for sparsity), and **Agent 3's revised conservative plan** (measurement model approach, robustness-first).

**Core Philosophy**:
1. **Infrastructure before innovation** - Fix ground truth contamination (ENACT) before testing fancy models
2. **Simple before complex** - ZINB baseline before diffusion/flow matching
3. **Robustness early** - Validate generalization at Week 6, not Month 5
4. **Checkpoints prevent waste** - Every week has Go/No-Go criteria

**Target Outcome by End of Month 3**:
- SSIM >0.60 (beating 0.5699 with statistical significance)
- Understand **why** improvement happened (ablations identify critical components)
- Identify top 3 architectural components for Month 4 novel system
- Robustness curves quantified (stain, patient heterogeneity, registration)

---

## Week-by-Week Breakdown

### Week 5 (Days 1-5): Infrastructure + Baseline Reproduction

**Goal**: Establish clean evaluation harness and reproduce baseline exactly before making changes.

#### Day 1-2: ENACT Installation + Segmentation Sensitivity ✅

**Implementation**:
1. Clone `Sanofi-Public/enact-pipeline`
2. Install dependencies (StarDist, geopandas, scanpy)
3. **Run full ENACT pipeline** on P1 (test patient):
   - Input: H&E WSI + Visium HD 2μm bin counts
   - Output: Cell-level AnnData with overlap weights `w_bc`

**Killer Validation #1: Segmentation Sensitivity**
```python
# Run ENACT with 3 segmentation methods
segmentation_methods = ['stardist_default', 'stardist_finetuned', 'cellpose']

results = {}
for method in segmentation_methods:
    cells = run_enact(P1_wsi, P1_bins, segmenter=method)
    results[method] = cells

# Quantify stability
for gene in genes:
    pearson_r = correlation(results['stardist_default'][gene],
                           results['cellpose'][gene])
    cv = coefficient_of_variation([results[m][gene] for m in methods])

    # DECISION CRITERIA:
    # ✅ Stable: r > 0.9, CV < 20% → Can use ENACT cells as weak labels
    # ❌ Unstable: r < 0.8, CV > 30% → MUST use measurement model
```

**Expected Runtime**: 1 day
**Compute**: CPU-bound (segmentation), RTX 5090 idle

**Success Criteria**:
- [ ] ENACT pipeline functional on P1
- [ ] Segmentation stability quantified (Pearson r, CV per gene)
- [ ] **Go/No-Go Decision**: Use ENACT cells directly OR implement measurement model

**If This Fails**:
- Segmentation unstable → Pivot to bin-level mixture model (adds 1-2 days)
- Implementation bug → Debug, fallback to naive centroid assignment

**Deliverables**:
- Code: `/home/user/work/sota-2um-st-prediction/code/preprocessing/enact_wrapper.py`
- Results: `/home/user/work/sota-2um-st-prediction/results/week5/day1-2_segmentation_sensitivity.csv`
- Data: `P1_cells_enact.h5ad`, `P1_overlap_weights.npz`

---

#### Day 2-3: Registration Check + Naive vs ENACT Baseline

**Implementation**:

**Killer Validation #2: Registration Optimization**
```python
# Grid search affine transformations (H&E ↔ Visium coordinates)
best_ssim = 0
best_shift = (0, 0)

for dx in [-20, -10, -5, 0, 5, 10, 20]:  # pixels
    for dy in [-20, -10, -5, 0, 5, 10, 20]:
        shifted_coords = coords + np.array([dx, dy])
        ssim_shifted = compute_ssim(baseline_pred, ground_truth, shifted_coords)

        if ssim_shifted > best_ssim:
            best_ssim = ssim_shifted
            best_shift = (dx, dy)

registration_gain = best_ssim - ssim_identity

# DECISION CRITERIA:
# ✅ OK: gain < 5% → Alignment sufficient
# ⚠️ PROBLEM: gain > 10% → Add registration optimization preprocessing
```

**Naive vs ENACT Comparison**:
```python
# Generate naive baseline (centroid-based assignment)
cells_naive = assign_bins_naive(bins, segmentation_masks)

# Compare metrics
metrics = {
    'sparsity_naive': (cells_naive.X == 0).mean(),
    'sparsity_enact': (cells_enact.X == 0).mean(),
    'entropy_naive': scipy.stats.entropy(cells_naive.X.mean(axis=0)),
    'entropy_enact': scipy.stats.entropy(cells_enact.X.mean(axis=0)),
}

# Expected: ENACT reduces sparsity, increases biological entropy
```

**Expected Runtime**: 1 day
**Compute**: CPU for registration sweep, GPU for SSIM computation

**Success Criteria**:
- [ ] Registration error quantified (gain from optimal shift)
- [ ] Naive vs ENACT metrics computed (sparsity, entropy, ARI)
- [ ] Overlap weights `w_bc` extracted for measurement model
- [ ] **Checkpoint**: If registration gain >10%, add optimization step

**If This Fails**:
- High registration error → Add affine optimization to preprocessing (+0.5 days)
- ENACT shows no improvement over naive → Re-evaluate bin assignment strategy

**Deliverables**:
- Results: `week5/day2-3_registration_check.csv`, `week5/naive_vs_enact.csv`
- Data: `P1_cells_naive.h5ad`, `P1_P2_P5_all_enact.h5ad`

---

#### Day 3-4: Baseline Reproduction (Sanity Check)

**Goal**: Reproduce Prov-GigaPath + Hist2ST baseline **exactly** before making changes.

**Implementation**:
```python
# Load existing baseline
baseline_config = {
    'encoder': 'prov-gigapath',  # Frozen ViT-G
    'decoder': 'hist2st',         # CNN + Transformer + GNN
    'loss': 'mse',                # No activation, linear output
    'data': 'raw_2um_bins'        # NOT ENACT-processed yet
}

# Train on P1+P2, test on P5
model = train_baseline(baseline_config, train_patients=['P1', 'P2'])
ssim_reproduced = evaluate(model, test_patient='P5')

# CRITICAL: Must match 0.5699 ± 0.01 SSIM
assert abs(ssim_reproduced - 0.5699) < 0.01, "Baseline not reproduced!"
```

**Why This Matters**: If we can't reproduce the baseline, we can't trust improvements.

**Expected Runtime**: 6-8 hours (frozen encoder = fast training)
**Compute**: RTX 5090 24GB, batch size 16

**Success Criteria**:
- [ ] Baseline SSIM within 0.01 of published result (0.5699)
- [ ] Per-gene correlations match expected distribution
- [ ] Negative controls pass (label shuffle → SSIM collapse)

**If This Fails**:
- SSIM mismatch → Debug data loading, loss computation, hyperparameters
- Can't proceed to new methods until baseline reproduced

**Deliverables**:
- Code: `code/models/baseline_hist2st.py`
- Results: `week5/day3-4_baseline_reproduction.csv`
- Checkpoint: `pretrained_weights/baseline_hist2st_reproduced.pth`

---

#### Day 5: Week 5 Synthesis + Go/No-Go Decision

**Tasks**:
1. Compile all Week 5 results:
   - Segmentation sensitivity (stable or unstable?)
   - Registration error (alignment OK or needs optimization?)
   - Naive vs ENACT improvement (quantified delta)
   - Baseline reproduction (exact match or mismatch?)

2. **Go/No-Go Decision Tree**:
```
Week 5 Success?
├─ All 4 checkpoints pass
│  └─ ✅ PROCEED to Week 6 (ZINB baseline)
├─ Segmentation unstable
│  └─ ⚠️ PIVOT to measurement model (add 1-2 days to Week 6)
├─ Registration error high
│  └─ ⚠️ ADD registration optimization (add 0.5 days to preprocessing)
└─ Baseline not reproduced
   └─ ❌ DEBUG FIRST - Don't proceed until baseline solid
```

**Deliverables**:
- Document: `docs/synthesis/week5_synthesis.md`
- Decision: `docs/plans/week6_execution_plan.md`

---

### Week 6 (Days 6-10): Minimum Viable SOTA (ZINB Baseline)

**Goal**: Beat baseline (0.5699) with simplest improvement: Better loss function for 2μm sparsity.

**Rationale** (from Agent 2 hypothesis H_20241224_012):
- 95% zero sparsity = extreme class imbalance
- MSE treats all examples equally → overfits to predicting zeros
- ZINB models zero-inflation explicitly (structural zeros vs sampling zeros)
- Proven in scRNA-seq (where sparsity is similar)

---

#### Day 6-7: ZINB Loss Implementation

**Architecture**:
```python
class MinimumViableSOTA(nn.Module):
    def __init__(self, n_genes=50):
        super().__init__()

        # Frozen encoder (same as baseline)
        self.encoder = load_pretrained('prov-gigapath')
        self.encoder.requires_grad_(False)

        # Simple spatial context (k-NN mean pooling)
        self.context_agg = nn.Sequential(
            nn.Linear(1536, 512),  # Prov-GigaPath outputs 1536-dim
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # ZINB parameter heads (per cell/bin)
        self.mu_head = nn.Linear(256, n_genes)      # Mean
        self.theta_head = nn.Linear(256, n_genes)   # Dispersion
        self.pi_head = nn.Linear(256, n_genes)      # Zero-inflation prob

    def forward(self, patches, neighbor_indices):
        # Encode patches
        features = self.encoder(patches)  # (batch, 1536)

        # Aggregate k=8 neighbors
        neighbor_feats = features[neighbor_indices]  # (batch, 8, 1536)
        context = neighbor_feats.mean(dim=1)  # (batch, 1536)

        # Context fusion
        fused = self.context_agg(context)  # (batch, 256)

        # Predict ZINB parameters
        mu = torch.exp(self.mu_head(fused))        # Positive
        theta = torch.exp(self.theta_head(fused))  # Positive
        pi = torch.sigmoid(self.pi_head(fused))    # [0, 1]

        return mu, theta, pi

def zinb_loss(y_obs, mu, theta, pi):
    """Zero-Inflated Negative Binomial NLL."""
    # Probability of zero (mixture)
    nb_zero_prob = (theta / (theta + mu)) ** theta
    prob_zero = pi + (1 - pi) * nb_zero_prob

    # Probability of non-zero (NB component)
    log_nb_nonzero = (
        torch.lgamma(y_obs + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y_obs + 1)
        + theta * torch.log(theta / (theta + mu))
        + y_obs * torch.log(mu / (theta + mu))
    )
    prob_nonzero = (1 - pi) * torch.exp(log_nb_nonzero)

    # Negative log-likelihood
    nll = -torch.where(
        y_obs == 0,
        torch.log(prob_zero + 1e-10),
        torch.log(prob_nonzero + 1e-10)
    )

    return nll.sum(dim=1).mean()
```

**Training**:
```python
# Train on P1+P2, test on P5
model = MinimumViableSOTA(n_genes=50)
optimizer = torch.optim.AdamW([
    {'params': model.context_agg.parameters(), 'lr': 1e-3},
    {'params': model.mu_head.parameters(), 'lr': 1e-3},
    {'params': model.theta_head.parameters(), 'lr': 1e-4},  # Slower for stability
    {'params': model.pi_head.parameters(), 'lr': 1e-3}
])

for epoch in range(50):
    for batch in train_loader:
        patches, neighbor_indices, y_true = batch

        mu, theta, pi = model(patches, neighbor_indices)
        loss = zinb_loss(y_true, mu, theta, pi)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate on P2 subset
    val_ssim = evaluate(model, val_data)

# Final test on P5 (NEVER used for hyperparameters)
test_ssim = evaluate(model, test_data='P5')
```

**Expected Runtime**: 1.5 days (implementation + training)
**Compute**: RTX 5090 24GB, batch size 16, ~8 hours per fold

**Success Criteria**:
- [ ] ZINB model trains and converges (loss decreases smoothly)
- [ ] ZINB SSIM > 0.5699 (beating MSE baseline)
- [ ] ZINB NLL < MSE loss (better likelihood fit)
- [ ] Negative controls pass (label shuffle, spatial jitter)

**If This Fails**:
- SSIM < baseline → Debug (check data, loss, hyperparameters)
- Training unstable → Reduce learning rates, add gradient clipping
- Fallback: MSE + ENACT (simpler, known to work)

**Deliverables**:
- Code: `code/models/minimum_viable_sota_zinb.py`
- Results: `week6/day6-7_zinb_baseline.csv`
- Checkpoint: `pretrained_weights/zinb_baseline_p5_test.pth`

---

#### Day 8-9: Focal Loss Experiment (Quick Win Hypothesis)

**Rationale** (from Agent 2 hypothesis H_20241224_012):
- Focal loss down-weights easy examples (zeros) with $(1-p)^\gamma$ weighting
- Proven in object detection for 99% background vs 1% foreground imbalance
- 95% zeros ≈ 99% background (transfer from computer vision)
- **Fast test**: Modify ZINB loss, grid search $\gamma \in \{1, 2, 3\}$

**Implementation**:
```python
def focal_zinb_loss(y_obs, mu, theta, pi, gamma=2.0):
    """Focal-weighted ZINB loss."""
    # Standard ZINB NLL
    zinb_nll = zinb_loss(y_obs, mu, theta, pi)

    # Focal weighting: (1-p)^gamma where p = exp(-nll)
    p = torch.exp(-zinb_nll.clamp(max=20))  # Stability
    focal_weight = (1 - p).pow(gamma)

    # Focal ZINB loss
    focal_zinb = focal_weight * zinb_nll
    return focal_zinb.mean()

# Grid search gamma
for gamma in [1.0, 2.0, 3.0]:
    model = MinimumViableSOTA(n_genes=50)
    train(model, loss_fn=lambda y, mu, th, pi: focal_zinb_loss(y, mu, th, pi, gamma))
    ssim_gamma = evaluate(model, test='P5')

    # Track: (gamma, ssim)
```

**Expected Runtime**: 1 day (3 runs × 8 hours)
**Compute**: RTX 5090 24GB, batch size 16

**Kill-Shot Criteria**:
- If SSIM improvement <3% over plain ZINB → Focal not helping, abandon
- If optimal $\gamma = 1$ (no focal weighting) → Class imbalance not the issue
- If training unstable → Reduce $\gamma$ or abandon

**Expected Results** (from hypothesis):
- SSIM improvement: 0.5699 → 0.60-0.65 (+10-20%)
- Optimal $\gamma$ likely = 2 (standard focal loss parameter)
- Non-zero-heavy genes benefit most from focal weighting

**Deliverables**:
- Code: `code/models/focal_zinb.py`
- Results: `week6/day8-9_focal_gamma_sweep.csv`
- Analysis: `week6/focal_ablation_per_gene.csv`

---

#### Day 10: Week 6 Synthesis + Robustness Check

**Tasks**:
1. **Component Attribution**: Which component drives performance?
   ```python
   ablations = {
       'Baseline (MSE)': ssim_baseline,
       'ENACT preprocessing': ssim_enact_only,
       'ZINB loss': ssim_zinb,
       'Focal ZINB': ssim_focal_zinb,
   }

   # Isolate contributions
   enact_contribution = ssim_enact_only - ssim_baseline
   zinb_contribution = ssim_zinb - ssim_enact_only
   focal_contribution = ssim_focal_zinb - ssim_zinb
   ```

2. **Stain Robustness Curves** (NEW - from Agent 3 conservative plan):
   ```python
   # Stain perturbation sweep
   for h_shift in [0, 0.05, 0.1, 0.2]:  # Hue
       for s_scale in [0.8, 0.9, 1.0, 1.1, 1.2]:  # Saturation
           augmented = apply_stain_jitter(images, h_shift, s_scale)
           ssim_aug = evaluate(best_model, augmented)

   # Plot: augmentation severity vs SSIM
   # Expected: Gradual degradation (NOT catastrophic failure)
   ```

3. **Cross-Patient Variance**:
   ```python
   ssim_p1, ssim_p2, ssim_p5 = evaluate_per_patient(best_model)
   variance = np.var([ssim_p1, ssim_p2, ssim_p5])

   # If variance > 0.10 → High patient heterogeneity (batch effect problem)
   ```

**Go/No-Go Decision for Week 7**:
```
Week 6 Results?
├─ ZINB SSIM > 0.60 + Robust stain curves
│  └─ ✅ STICK WITH ZINB → Week 7: Encoder ablations (not diffusion yet)
├─ ZINB < 0.60 + Evidence of multimodality
│  └─ ⚠️ ADD GENERATIVE → Week 7: Implement Stem diffusion on top of ZINB
├─ High patient variance (>0.10) OR control failures
│  └─ ❌ DEBUG FIRST → Week 7: Batch correction + robustness before complexity
└─ Focal gives large gains
   └─ ✅ USE FOCAL ZINB → Week 7: Ablations with Focal ZINB as new baseline
```

**Deliverables**:
- Document: `docs/synthesis/week6_synthesis.md`
- Results: `week6/component_attribution.csv`, `week6/stain_robustness_curves.png`
- Decision: `docs/plans/week7_execution_plan.md`

**Success Criteria for Week 6**:
- [ ] ZINB SSIM > 0.5699 (beating baseline)
- [ ] Focal loss tested (improvement quantified)
- [ ] Stain robustness curves plotted (no catastrophic failure)
- [ ] Cross-patient variance quantified
- [ ] Component attribution complete (know what drives performance)

---

### Week 7-8: Encoder Ablations + Decoder Experiments

**Goal**: Identify best encoder and test decoder architectural improvements.

**Rationale**:
- Baseline uses Prov-GigaPath (ViT-G, 1B params, pathology pretrained)
- Agent 1 literature synthesis identified better options:
  - **PAST**: Pretrained on H&E → genes (most aligned with task)
  - **Threads**: Molecular supervision (genomics + transcriptomics)
  - **UNI**: Robust general pathology features
  - **CONCH**: Visual-language capabilities

---

#### Week 7 Day 1-3: Encoder Comparison

**Experimental Design**:
| Experiment | Encoder | Decoder | Loss | Expected SSIM | Time (hrs) |
|------------|---------|---------|------|---------------|------------|
| Baseline | Prov-GigaPath | Hist2ST | MSE | 0.5699 | 0 (already have) |
| ZINB Baseline | Prov-GigaPath | Hist2ST | ZINB | 0.60? | 0 (Week 6) |
| Test 1 | **PAST** | Hist2ST | ZINB | 0.62? | 8 |
| Test 2 | **Threads** | Hist2ST | ZINB | 0.61? | 8 |
| Test 3 | **UNI** | Hist2ST | ZINB | 0.60? | 8 |
| Test 4 | **CONCH** | Hist2ST | ZINB | 0.59? | 8 |

**Implementation**:
```python
encoders_to_test = {
    'PAST': 'path/to/past/weights',     # Search HuggingFace Day 1
    'Threads': 'path/to/threads',       # Check availability
    'UNI': 'mahmoodlab/UNI',           # Already have access
    'CONCH': 'mahmoodlab/CONCH',       # Already have access
}

results = {}
for encoder_name, encoder_path in encoders_to_test.items():
    model = MinimumViableSOTA(n_genes=50)
    model.encoder = load_pretrained(encoder_path)
    model.encoder.requires_grad_(False)  # Keep frozen

    train(model, loss_fn=focal_zinb_loss, data='ENACT_processed')
    ssim = evaluate(model, test='P5')

    results[encoder_name] = {
        'ssim': ssim,
        'feature_dim': model.encoder.embed_dim,
        'params': count_parameters(model.encoder),
    }
```

**Expected Runtime**: 3 days (4 encoders × 8 hours each, run 2 in parallel)
**Compute**: RTX 5090 24GB, can run 2 encoders simultaneously if feature dims allow

**Success Criteria**:
- [ ] All 4 encoders tested on same data/loss/decoder
- [ ] Per-encoder SSIM, PCC, per-category performance computed
- [ ] Best encoder identified (highest SSIM + lowest variance)

**If This Fails**:
- PAST weights unavailable → Use UNI as best alternative
- All encoders similar → Encoder choice not critical, focus on decoder/loss

**Deliverables**:
- Results: `week7/encoder_ablation.csv`
- Analysis: `week7/encoder_per_gene_performance.csv`
- Checkpoint: `pretrained_weights/best_encoder_zinb.pth`

---

#### Week 7 Day 4-5 + Week 8 Day 1-3: Decoder Architectures

**Rationale**:
- Baseline Hist2ST decoder has 3 parallel paths (CNN + Transformer + GNN)
- Agent 1 synthesis identified improvements:
  - **THItoGene**: Improved attention mechanism over Hist2ST
  - **sCellST**: Subcellular UNet upsampling
  - **DeepSpot2Cell**: MIL aggregator for multi-scale context
  - **HyperPNN**: Dual-Attention Fusion Blocks (from hyperspectral pansharpening)

**Experimental Design**:
| Experiment | Encoder | Decoder | Loss | Expected SSIM | Time (hrs) |
|------------|---------|---------|------|---------------|------------|
| Best from Week 7 | Best encoder | Hist2ST | Focal ZINB | 0.62 | 0 (baseline) |
| Test 1 | Best encoder | **THItoGene** | Focal ZINB | 0.63? | 10 |
| Test 2 | Best encoder | **MiniUNet** | Focal ZINB | 0.60? | 6 |
| Test 3 | Best encoder | **DeepSpot2Cell MIL** | Focal ZINB | 0.64? | 12 |
| Test 4 | Best encoder | **HyperPNN DAFB** | Focal ZINB | 0.65? | 12 |

**Implementation Priority**:
1. **THItoGene** (FAST WIN): Small modification of Hist2ST attention
   - Claimed improvement over Hist2ST in paper
   - Easy to implement (replace attention module)
   - Expected: +1-2% SSIM

2. **DeepSpot2Cell MIL** (HIGH CEILING):
   - Multi-scale context aggregation (cell + spot + global)
   - Proven on single-cell deconvolution
   - Expected: +3-5% SSIM if multi-scale hypothesis correct

3. **HyperPNN Dual-Attention** (NOVEL TRANSFER):
   - From hyperspectral pansharpening (spatial + spectral attention)
   - Direct mathematical isomorphism (H&E = PAN, ST = hyperspectral)
   - Expected: +4-6% SSIM if attention mechanism transfers

**Code Archaeology** (Week 7 Day 4):
```bash
# Clone decoder repos
cd /home/user/work/code_archaeology/
git clone https://github.com/liyichen1998/THItoGene
git clone https://github.com/ratschlab/DeepSpot2Cell

# Extract architectures
grep -r "class.*Decoder\|class.*Net" --include="*.py"
# Read forward pass, extract hyperparameters
```

**Expected Runtime**: 4 days (3 decoders × 10-12 hours each)
**Compute**: RTX 5090 24GB

**Success Criteria**:
- [ ] At least 2 decoder architectures tested
- [ ] Per-decoder SSIM improvement quantified
- [ ] Best decoder + encoder combination identified
- [ ] Ablation study: Decoder contribution vs encoder contribution

**If This Fails**:
- All decoders similar → Decoder choice not critical, focus on loss/training
- Implementation too complex → Stick with Hist2ST, focus on other components

**Deliverables**:
- Code: `code/models/thitogene_decoder.py`, `code/models/deepspot2cell_mil.py`
- Results: `week8/decoder_ablation.csv`
- Checkpoint: `pretrained_weights/best_encoder_decoder_combo.pth`

---

#### Week 8 Day 4-5: Multi-Scale Hypothesis Test

**Rationale** (from Agent 1 literature synthesis):
- 2μm bins may be **information-rich** (cellular detail), not information-limited
- Current methods use single resolution (2μm patches only)
- **TRIPLEX**: Multi-resolution fusion (claims SOTA with 2μm+8μm joint training)
- **DeepSpot2Cell**: Cell + spot + global context aggregation

**Hypothesis**: Adding 8μm context improves 2μm predictions (tissue architecture signal).

**Experimental Design**:
```python
# Test 1: Single-scale (2μm only) - Baseline
model_2um = MinimumViableSOTA(
    encoder=best_encoder,
    decoder=best_decoder,
    loss=focal_zinb_loss,
    input_scales=['2um']  # 256x256 patches
)

# Test 2: Dual-scale (2μm + 8μm joint)
model_dual = MultiScaleSOTA(
    encoder=best_encoder,
    decoder=best_decoder,
    loss=focal_zinb_loss,
    input_scales=['2um', '8um'],  # 256x256 + 512x512
    fusion='attention'  # Or 'concat', 'add'
)

# Compare
ssim_single = evaluate(model_2um, test='P5')
ssim_dual = evaluate(model_dual, test='P5')

delta = ssim_dual - ssim_single
# If delta > 0.02 → Multi-scale helps, adopt for Month 4
```

**Expected Runtime**: 1.5 days (2 experiments × 10 hours each)
**Compute**: RTX 5090 24GB (may need gradient checkpointing for dual-scale)

**Success Criteria**:
- [ ] Multi-scale model trains without OOM
- [ ] SSIM improvement >2% → Multi-scale hypothesis validated
- [ ] Per-gene analysis: Which genes benefit from multi-scale?

**Deliverables**:
- Code: `code/models/multiscale_sota.py`
- Results: `week8/multiscale_ablation.csv`

---

### Week 9-10: Generative Models (IF Justified by Week 6-8 Results)

**CRITICAL**: Only proceed here if ZINB shows evidence of **underfitting multimodality**.

**Decision Criteria from Week 6**:
- ✅ **Proceed to generative** IF:
  - ZINB predictions look "averaged/blurry" (visual inspection)
  - Per-spot variance too low (model underestimates uncertainty)
  - Evidence of multimodal distributions in ground truth
- ❌ **Skip generative, stay with ZINB** IF:
  - ZINB SSIM >0.65 (already strong)
  - Predictions look sharp and biologically plausible
  - Variance well-calibrated

---

#### Week 9 Day 1-3: Stem Diffusion Baseline (If Proceeding)

**Rationale** (from Agent 1 synthesis):
- **Stem** (ICLR 2025): Clean diffusion baseline, HEST integration, proven
- Generative models sample from $p(y|x)$, not just predict $E[y|x]$
- Better for multimodal distributions (identical morphology → multiple cell states)

**Implementation**:
```bash
# Clone Stem
cd /home/user/work/code_archaeology/
git clone https://github.com/SichenZhu/Stem

# Hotwire data loader for ENACT-processed H5AD
# Replace Stem's patch embedding with our best encoder
```

**Architecture Modification**:
```python
class StemModified(nn.Module):
    def __init__(self):
        super().__init__()

        # Replace Stem's simple patch encoder with our best encoder
        self.encoder = load_pretrained(best_encoder)
        self.encoder.requires_grad_(False)

        # Keep Stem's diffusion U-Net
        self.diffusion_unet = load_stem_unet()

        # Conditioning: Inject encoder features into U-Net
        self.condition_proj = nn.Linear(encoder_dim, unet_dim)

    def forward(self, x_noisy, t, patches):
        # Encode morphology
        morph_features = self.encoder(patches)
        condition = self.condition_proj(morph_features)

        # Diffusion denoising step
        noise_pred = self.diffusion_unet(x_noisy, t, condition)

        return noise_pred

# Training (simplified)
for epoch in range(n_epochs):
    for batch in dataloader:
        patches, y_true = batch

        # Sample timestep
        t = torch.randint(0, 1000, (batch_size,))

        # Add noise to y_true
        noise = torch.randn_like(y_true)
        y_noisy = diffusion_scheduler.add_noise(y_true, noise, t)

        # Predict noise
        noise_pred = model(y_noisy, t, patches)

        # Denoising loss
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Expected Runtime**: 2.5 days (implementation 1 day + training 1.5 days)
**Compute**: RTX 5090 24GB, diffusion requires more memory (use gradient checkpointing)

**Success Criteria**:
- [ ] Stem trains on ENACT-processed data
- [ ] Samples show diversity (not collapsed to mean)
- [ ] SSIM competitive with ZINB (within 5%)
- [ ] Wasserstein distance better than ZINB (captures spatial distribution)

**If This Fails**:
- Diffusion doesn't improve SSIM → Stay with ZINB
- Training unstable → Adjust noise schedule, learning rate
- OOM errors → Reduce U-Net depth, use FP16

**Deliverables**:
- Code: `code/models/stem_modified.py`
- Results: `week9/stem_diffusion_results.csv`
- Samples: `week9/stem_samples_visual_inspection.png`

---

#### Week 9 Day 4-5 + Week 10 Day 1-2: Hybrid ZINB + Diffusion

**Hypothesis**: Combine strengths of both approaches.

**Architecture**:
```python
class ZINBDiffusion(nn.Module):
    """
    Hybrid model:
    1. ZINB predicts mean and zero-inflation (fast, stable)
    2. Diffusion models residuals/uncertainty (captures multimodality)
    """
    def __init__(self):
        super().__init__()

        # ZINB component (from Week 6)
        self.zinb_model = MinimumViableSOTA(n_genes=50)

        # Diffusion component (models residuals)
        self.diffusion_unet = StemModified()

    def forward(self, patches, mode='zinb'):
        if mode == 'zinb':
            # Fast inference: ZINB mean prediction
            mu, theta, pi = self.zinb_model(patches)
            return mu

        elif mode == 'diffusion':
            # Slow inference: Diffusion sampling conditioned on ZINB
            mu_zinb, _, _ = self.zinb_model(patches)

            # Sample residuals via diffusion
            residuals = self.diffusion_unet.sample(
                shape=mu_zinb.shape,
                condition=mu_zinb
            )

            # Final prediction: ZINB mean + diffusion residuals
            y_pred = mu_zinb + residuals
            return y_pred

# Training
# Stage 1: Train ZINB (already done Week 6)
# Stage 2: Freeze ZINB, train diffusion on residuals
for epoch in range(n_epochs):
    for batch in dataloader:
        patches, y_true = batch

        # Get ZINB prediction
        with torch.no_grad():
            mu_zinb, _, _ = model.zinb_model(patches)

        # Compute residuals
        residuals_true = y_true - mu_zinb

        # Train diffusion on residuals
        loss_diffusion = train_diffusion_step(
            model.diffusion_unet,
            residuals_true,
            condition=mu_zinb
        )

        optimizer.zero_grad()
        loss_diffusion.backward()
        optimizer.step()
```

**Expected Runtime**: 2 days
**Compute**: RTX 5090 24GB

**Success Criteria**:
- [ ] Hybrid model trains successfully
- [ ] SSIM improvement over ZINB alone
- [ ] Samples show controlled diversity (not too wild)
- [ ] Faster inference than pure diffusion

**Deliverables**:
- Code: `code/models/zinb_diffusion_hybrid.py`
- Results: `week10/hybrid_results.csv`

---

#### Week 10 Day 3-5: Generative Model Synthesis

**Tasks**:
1. **Compare all approaches**:
   - ZINB (deterministic, fast)
   - Stem Diffusion (stochastic, slow)
   - ZINB + Diffusion Hybrid (controlled stochasticity)

2. **Metrics**:
   - SSIM (primary)
   - Wasserstein distance (spatial fidelity)
   - Prediction variance (calibration)
   - Inference time (practical deployment)

3. **Decision**: Which approach for Month 4 novel architecture?

**Deliverables**:
- Document: `docs/synthesis/week9-10_generative_synthesis.md`
- Results: `week10/generative_comparison_table.csv`

---

### Week 11-12: Best Combination + Optimization

**Goal**: Combine best components from Weeks 5-10, optimize hyperparameters, validate rigorously.

---

#### Week 11: Component Integration

**Best Components from Weeks 5-10**:
1. **Preprocessing**: ENACT cell-level (or measurement model if segmentation unstable)
2. **Encoder**: Best from Week 7 encoder ablation (likely PAST or UNI)
3. **Decoder**: Best from Week 8 decoder ablation (likely DeepSpot2Cell MIL or HyperPNN DAFB)
4. **Loss**: Focal ZINB (if Week 6 validated) or pure ZINB
5. **Multi-scale**: 2μm+8μm if Week 8 validated
6. **Generative**: Hybrid ZINB+Diffusion if Week 9-10 validated

**Implementation**:
```python
class BestCombinationSOTA(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Component 1: Best encoder (frozen)
        self.encoder = load_pretrained(config['best_encoder'])
        self.encoder.requires_grad_(False)

        # Component 2: Multi-scale if validated
        if config['multi_scale']:
            self.encoder_8um = load_pretrained(config['best_encoder'])
            self.encoder_8um.requires_grad_(False)
            self.scale_fusion = AttentionFusion(dim=encoder_dim)

        # Component 3: Best decoder
        self.decoder = load_best_decoder(config['best_decoder'])

        # Component 4: Loss function
        self.loss_fn = focal_zinb_loss if config['use_focal'] else zinb_loss

        # Component 5: Generative component (optional)
        if config['use_diffusion']:
            self.diffusion = StemModified()

    def forward(self, patches_2um, patches_8um=None):
        # Encode
        features_2um = self.encoder(patches_2um)

        if patches_8um is not None:
            features_8um = self.encoder_8um(patches_8um)
            features = self.scale_fusion(features_2um, features_8um)
        else:
            features = features_2um

        # Decode
        mu, theta, pi = self.decoder(features)

        return mu, theta, pi
```

**Expected Runtime**: 2 days (integration + debugging)

**Success Criteria**:
- [ ] All components integrated without conflicts
- [ ] Model trains end-to-end
- [ ] Preliminary SSIM on P5 test set

**Deliverables**:
- Code: `code/models/best_combination_sota.py`
- Config: `code/configs/best_config.yaml`

---

#### Week 11 Day 3-5: Hyperparameter Optimization

**Strategy**: Bayesian optimization (not grid search - more efficient).

**Critical Hyperparameters**:
1. Learning rates (decoder_lr, theta_lr, pi_lr)
2. Batch size / gradient accumulation
3. Decoder depth (number of layers)
4. Graph k-neighbors (if using GNN)
5. Multi-scale fusion weights
6. Focal loss gamma (if using focal)
7. Diffusion noise schedule (if using diffusion)

**Implementation**:
```python
import optuna

def objective(trial):
    # Sample hyperparameters
    config = {
        'decoder_lr': trial.suggest_loguniform('decoder_lr', 1e-4, 1e-2),
        'theta_lr': trial.suggest_loguniform('theta_lr', 1e-5, 1e-3),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
        'decoder_depth': trial.suggest_int('decoder_depth', 2, 6),
        'k_neighbors': trial.suggest_int('k_neighbors', 4, 16),
        'focal_gamma': trial.suggest_uniform('focal_gamma', 1.0, 3.0),
    }

    # Train with sampled hyperparameters
    model = BestCombinationSOTA(config)
    train(model, train_data='P1+P2', val_data='P2_subset')

    # Evaluate on validation set (NOT test set P5!)
    val_ssim = evaluate(model, val_data='P2_subset')

    return val_ssim

# Run Bayesian optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

best_config = study.best_params
```

**Budget**: ~15 trials × 8 hours = 5 days GPU time (run overnight)
**Expected Runtime**: 3 days wall time (parallel trials if possible)

**Success Criteria**:
- [ ] Optimal hyperparameters found
- [ ] Validation SSIM improvement >2% over default hyperparameters

**Deliverables**:
- Results: `week11/hyperparameter_optimization.csv`
- Config: `code/configs/optimized_config.yaml`

---

#### Week 12: Full Cross-Validation + Final Ablations

**Goal**: Rigorous evaluation with nested CV, statistical significance testing, ablation studies.

**Day 1-2: 3-Fold LOOCV**
```python
# Outer loop: 3-fold LOOCV
folds = [
    {'train': ['P1', 'P2'], 'test': 'P5'},
    {'train': ['P1', 'P5'], 'test': 'P2'},
    {'train': ['P2', 'P5'], 'test': 'P1'},
]

results = []
for fold in folds:
    # Train with best config
    model = BestCombinationSOTA(best_config)
    train(model, train_data=fold['train'])

    # Evaluate on test patient (NEVER used before)
    test_ssim = evaluate(model, test_data=fold['test'])
    test_pcc = compute_pcc(model, test_data=fold['test'])
    test_wasserstein = compute_wasserstein(model, test_data=fold['test'])

    results.append({
        'fold': fold['test'],
        'ssim': test_ssim,
        'pcc': test_pcc,
        'wasserstein': test_wasserstein,
    })

# Aggregate results
mean_ssim = np.mean([r['ssim'] for r in results])
std_ssim = np.std([r['ssim'] for r in results])
```

**Statistical Testing**:
```python
# Hierarchical bootstrap (valid for n=3)
def bootstrap_comparison(novel_results, baseline_results, n_bootstrap=10000):
    deltas = []

    for _ in range(n_bootstrap):
        # Bootstrap at patient level (n=3 with replacement)
        sampled_indices = np.random.choice(3, size=3, replace=True)

        novel_sample = [novel_results[i] for i in sampled_indices]
        baseline_sample = [baseline_results[i] for i in sampled_indices]

        delta = np.mean(novel_sample) - np.mean(baseline_sample)
        deltas.append(delta)

    # 95% Confidence Interval
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)

    # Effect size (Cohen's d)
    cohens_d = np.mean(deltas) / np.std(deltas)

    return {
        'mean_delta': np.mean(deltas),
        'ci_95': [ci_lower, ci_upper],
        'cohens_d': cohens_d,
    }

# Compare to baseline (0.5699)
baseline_ssim_per_patient = [0.5699, 0.5699, 0.5699]  # Approximate
novel_ssim_per_patient = [results[i]['ssim'] for i in range(3)]

stats = bootstrap_comparison(novel_ssim_per_patient, baseline_ssim_per_patient)

print(f"Mean SSIM improvement: {stats['mean_delta']:.4f}")
print(f"95% CI: [{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}]")
print(f"Cohen's d: {stats['cohens_d']:.2f}")
```

**Day 3-4: Component Ablation Study**
```python
# Ablate each component to understand contribution
ablations = {
    'Full Model': best_model,
    'No ENACT (naive bins)': best_model_naive,
    'No Focal (plain ZINB)': best_model_no_focal,
    'No Multi-scale (2μm only)': best_model_single_scale,
    'Best Encoder → ImageNet': best_model_imagenet_encoder,
    'Best Decoder → MiniUNet': best_model_simple_decoder,
}

ablation_results = {}
for name, model in ablations.items():
    ssim = evaluate(model, test_data='P5')  # Use single test patient
    ablation_results[name] = ssim

# Quantify contributions
enact_contribution = ablation_results['Full Model'] - ablation_results['No ENACT']
focal_contribution = ablation_results['Full Model'] - ablation_results['No Focal']
# ... etc
```

**Day 5: Robustness + Negative Controls**
```python
# Negative controls
negative_controls = {
    'Label Shuffle': evaluate(best_model, shuffle_labels=True),
    'Spatial Jitter (20μm)': evaluate(best_model, jitter_coords=20),
    'Random Encoder': evaluate(best_model_random_encoder, test_data='P5'),
}

# Expected: All negative controls should show SSIM collapse

# Robustness curves
stain_robustness = []
for augmentation_severity in [0, 0.1, 0.2, 0.3, 0.4]:
    ssim_aug = evaluate(best_model, stain_augmentation=augmentation_severity)
    stain_robustness.append(ssim_aug)

# Plot: augmentation vs SSIM (should degrade gradually, not catastrophically)
```

**Success Criteria for Week 12**:
- [ ] Mean SSIM > 0.60 (95% CI excludes 0.5699)
- [ ] Cohen's d > 0.5 (medium effect size)
- [ ] Per-patient variance < 0.10 SSIM
- [ ] Negative controls show appropriate SSIM collapse
- [ ] Stain robustness curves show gradual degradation
- [ ] Component ablations quantify each contribution

**Deliverables**:
- Results: `week12/final_3fold_cv_results.csv`
- Stats: `week12/statistical_significance_tests.csv`
- Ablations: `week12/component_ablation_study.csv`
- Figures: `week12/robustness_curves.png`, `week12/negative_controls.png`
- Document: `docs/synthesis/month2-3_final_synthesis.md`

---

## Parallel Execution Strategy

### Weeks That Can Run in Parallel

**Week 7-8 Encoders + Decoders**:
- **Day 1-3**: Run 2 encoder tests simultaneously (UNI + CONCH in parallel)
  - RTX 5090 24GB can handle 2 frozen encoders with batch size 8 each
  - Saves 1 day
- **Day 4-5**: Decoder tests run sequentially (more complex, need full GPU)

**Week 9-10 Generative (If Running)**:
- **Day 1-3**: ZINB training (already done Week 6) vs Stem diffusion in parallel
  - ZINB just inference for baseline comparison
  - Diffusion uses full GPU
- **Day 4-5**: Hybrid model needs both, sequential

**Week 11 Hyperparameter Optimization**:
- Run 3 Optuna trials in parallel if using multiple GPUs
- Single RTX 5090: Sequential trials, but run overnight

### Sequential Dependencies

**MUST be sequential**:
1. Week 5 Day 1-2 (ENACT) → Day 3-4 (baseline reproduction)
   - Need ENACT outputs before testing baseline on ENACT data
2. Week 6 Day 6-7 (ZINB) → Day 8-9 (Focal ZINB)
   - Need plain ZINB results to compare Focal improvement
3. Week 7 Encoders → Week 8 Decoders
   - Need best encoder before testing decoder combinations
4. Weeks 5-10 → Week 11 (integration)
   - Need all component results before selecting best combination
5. Week 11 optimization → Week 12 final CV
   - Need optimized hyperparameters before final evaluation

**Total Parallel Savings**: ~2-3 days over 8 weeks

---

## Resource Allocation

### Compute Budget

**Training Time per Experiment** (frozen encoder):
- Simple model (ZINB): 6-8 hours
- Complex model (Diffusion): 10-12 hours
- Hyperparameter trial: 8 hours

**Total Experiments Planned**:
- Week 5: 3 experiments (baseline, ENACT variants)
- Week 6: 5 experiments (ZINB + Focal gamma sweep)
- Week 7-8: 7 experiments (4 encoders + 3 decoders)
- Week 9-10: 3 experiments (Stem + Hybrid + ablations) - IF running
- Week 11: 15 hyperparameter trials
- Week 12: 3 folds + 6 ablations

**Total**: ~36 experiments × 8 hours avg = **288 GPU-hours** = **12 GPU-days**

**Timeline**: 8 weeks = 56 days
**Utilization**: 12 / 56 = **21% GPU utilization** (conservative, allows debugging)

**Bottlenecks**:
- Week 7-8: Encoder/decoder ablations (most experiments)
- Week 11: Hyperparameter optimization (sequential trials)

**Mitigation**:
- Run overnight experiments
- Parallelize encoder tests where possible
- Use early stopping if experiments clearly failing

---

## Risk Mitigation

### Technical Risks

| Risk | Likelihood | Impact | Mitigation | Timeline Cost |
|------|------------|--------|------------|---------------|
| **ENACT segmentation unstable** | Medium | High | Use measurement model instead of cell labels | +1-2 days (Week 5) |
| **Registration error >10%** | Medium | Medium | Add affine optimization preprocessing | +0.5 days (Week 5) |
| **ZINB < baseline** | Low | High | Debug implementation, fallback to MSE+ENACT | +1 day (Week 6) |
| **PAST weights unavailable** | Medium | Medium | Use UNI as best alternative encoder | No delay (Week 7) |
| **Decoder repos broken** | Low | Medium | Implement from paper descriptions | +2 days (Week 8) |
| **Diffusion doesn't improve** | Medium | Low | Skip generative, stay with ZINB | Save 1 week (Week 9-10) |
| **OOM with multi-scale** | High | Medium | Use gradient checkpointing, FP16 | +0.5 days (Week 8) |
| **Hyperopt finds no improvement** | Low | Medium | Use default hyperparameters from literature | No delay (Week 11) |

### Timeline Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Implementation harder than expected** | High | Start simple (Week 5-6), add complexity gradually |
| **Code archaeology takes too long** | Medium | Budget 1 day per repo, skip if unavailable |
| **Bugs in integration** | High | Unit tests, modular components, checkpoints |
| **Hyperparameter search too slow** | Medium | Limit to 15 trials, use Bayesian optimization |

### Knowledge Continuity Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Forget what worked in previous weeks** | Critical | Weekly synthesis documents MANDATORY |
| **Lose track of best models** | High | Save checkpoints with descriptive names |
| **Results not comparable** | High | Fix evaluation harness Week 5, never change |

---

## Success Metrics (End of Month 3)

### Primary Success (Publication-Worthy)

- [ ] **SSIM >0.60** (95% CI excludes baseline 0.5699)
- [ ] **Cohen's d >0.5** (medium or large effect size)
- [ ] **Per-patient variance <0.10** (generalizes across patients)
- [ ] **Negative controls pass** (label shuffle, spatial jitter show appropriate degradation)
- [ ] **Component attribution complete** (know which components drive improvement)

### Secondary Success (Understanding Achieved)

- [ ] **Top 3 critical components identified** for Month 4 novel architecture
- [ ] **Ablation studies quantify** each component's contribution
- [ ] **Robustness curves** demonstrate generalization to stain variation
- [ ] **Per-gene analysis** reveals which gene categories benefit most

### Stretch Goals (If Everything Goes Well)

- [ ] **SSIM >0.65** (+15% over baseline)
- [ ] **Generative model** shows controlled diversity
- [ ] **Multi-scale hypothesis** validated (8μm context improves 2μm predictions)
- [ ] **Novel architectural insight** discovered (e.g., focal loss critical for sparsity)

### Failure Modes (Pivot Triggers)

- ❌ **SSIM <0.60 by Week 10** → Pivot to ensemble of existing methods, abandon novel architecture
- ❌ **High patient variance (>0.15)** → Focus Month 4 on batch correction, not architecture
- ❌ **Negative controls fail** → Model learning artifacts, need architectural changes
- ❌ **All hypotheses fail** → Re-evaluate problem formulation, may need more data

---

## Deliverables Checklist (End of Month 3)

### Code
- [ ] `code/preprocessing/enact_wrapper.py` (ENACT integration)
- [ ] `code/models/minimum_viable_sota_zinb.py` (ZINB baseline)
- [ ] `code/models/focal_zinb.py` (Focal loss variant)
- [ ] `code/models/best_combination_sota.py` (Final integrated model)
- [ ] `code/evaluation/spatial_evaluator.py` (Comprehensive evaluation harness)
- [ ] `code/configs/best_config.yaml` (Optimized hyperparameters)

### Results
- [ ] `results/week5/` (ENACT validation, baseline reproduction)
- [ ] `results/week6/` (ZINB baseline, focal loss sweep)
- [ ] `results/week7/` (Encoder ablations)
- [ ] `results/week8/` (Decoder ablations, multi-scale tests)
- [ ] `results/week9-10/` (Generative models - if run)
- [ ] `results/week11/` (Hyperparameter optimization)
- [ ] `results/week12/` (Final 3-fold CV, ablations, robustness)

### Documentation
- [ ] `docs/synthesis/week5_synthesis.md`
- [ ] `docs/synthesis/week6_synthesis.md`
- [ ] `docs/synthesis/week7-8_encoder_decoder_synthesis.md`
- [ ] `docs/synthesis/week9-10_generative_synthesis.md` (if applicable)
- [ ] `docs/synthesis/month2-3_final_synthesis.md` (COMPREHENSIVE)

### Figures (for Month 6 publication)
- [ ] Encoder comparison bar chart (SSIM per encoder)
- [ ] Decoder comparison bar chart (SSIM per decoder)
- [ ] Component ablation study (contribution breakdown)
- [ ] Robustness curves (stain augmentation vs SSIM)
- [ ] Negative controls (expected SSIM collapse)
- [ ] Per-gene performance heatmap

### Checkpoints
- [ ] `pretrained_weights/baseline_hist2st_reproduced.pth`
- [ ] `pretrained_weights/zinb_baseline.pth`
- [ ] `pretrained_weights/best_encoder_decoder_combo.pth`
- [ ] `pretrained_weights/final_optimized_model.pth`

---

## Next Steps After Month 3

### If Success (SSIM >0.60)

**Month 4: Novel Architecture Design**
- Take top 3 components from Month 2-3 ablations
- Design modular novel architecture (e.g., "Clean Frankenstein" from Agent 1 blueprint)
- Implement architectural innovations:
  - If multi-scale worked → HyperPNN Dual-Attention Fusion
  - If generative worked → Flow matching or autoregressive
  - If MIL worked → Multi-level aggregation

**Month 5: Optimization + External Validation**
- Full hyperparameter search on novel architecture
- Test on HEST-1k (if accessible) or Ken Lau CRC cohort
- Cross-tissue generalization

**Month 6: Publication Preparation**
- Methods paper: Novel architecture
- Benchmark paper: Comprehensive comparison
- Code release, pretrained weights, dataset

### If Partial Success (SSIM 0.58-0.60)

**Month 4: Focused Improvement**
- Identify weakest component from ablations
- Deep dive on that component only
- E.g., if decoder is bottleneck → Try 5 more decoder architectures

**Month 5: Ensemble + Calibration**
- Ensemble of best models
- Uncertainty quantification
- Calibration techniques

**Month 6: Publication**
- Benchmark paper (comprehensive comparison)
- Methods note (lessons learned)

### If Failure (SSIM <0.58)

**Month 4: Root Cause Analysis**
- Is 2μm fundamentally limited by signal-to-noise?
- Is n=3 too small (batch effects dominate)?
- Is ground truth quality the bottleneck?

**Pivot Options**:
1. Switch to 8μm resolution (more signal)
2. Acquire more patients (reduce batch effect variance)
3. Focus on specific gene categories (epithelial markers only)
4. Reframe as uncertainty quantification problem (not point prediction)

---

## Appendix: Weekly Time Budget

| Week | Days | Focus | GPU-hours | Critical Path |
|------|------|-------|-----------|---------------|
| **Week 5** | 5 | Infrastructure + Baseline | 24 | ENACT validation → Baseline reproduction |
| **Week 6** | 5 | ZINB + Focal Loss | 40 | ZINB implementation → Focal sweep → Robustness |
| **Week 7** | 5 | Encoder Ablations | 32 | Encoder comparison (can parallelize 2) |
| **Week 8** | 5 | Decoder + Multi-scale | 36 | Decoder comparison → Multi-scale test |
| **Week 9** | 5 | Generative (if justified) | 30 | Stem baseline → Hybrid (OPTIONAL) |
| **Week 10** | 5 | Generative Synthesis | 20 | Comparison → Decision (OPTIONAL) |
| **Week 11** | 5 | Integration + HyperOpt | 40 | Component integration → Bayesian opt |
| **Week 12** | 5 | Final CV + Ablations | 36 | 3-fold CV → Ablations → Robustness |
| **TOTAL** | **40 days** | | **258 GPU-hours** | **~11 GPU-days** |

**Slack Time**: 8 weeks = 56 days, 40 used = **16 days slack** for debugging, pivots, sick days

---

## Appendix: Decision Trees for Each Week

### Week 5 Go/No-Go
```
Day 1-2: ENACT Segmentation Sensitivity
├─ Pearson r > 0.9, CV < 20%
│  └─ ✅ STABLE → Use ENACT cells as weak labels
│     └─ Proceed to Day 3-4 (Baseline Reproduction)
└─ Pearson r < 0.8, CV > 30%
   └─ ❌ UNSTABLE → Implement measurement model
      └─ Add 1-2 days to Week 5, proceed to Week 6

Day 2-3: Registration Check
├─ Gain < 5% SSIM
│  └─ ✅ OK → Proceed to Day 3-4
└─ Gain > 10% SSIM
   └─ ⚠️ PROBLEM → Add registration optimization (+0.5 days)

Day 3-4: Baseline Reproduction
├─ |SSIM - 0.5699| < 0.01
│  └─ ✅ REPRODUCED → Proceed to Week 6
└─ |SSIM - 0.5699| > 0.01
   └─ ❌ MISMATCH → Debug before Week 6 (BLOCKING)
```

### Week 6 Go/No-Go
```
Day 6-7: ZINB Baseline
├─ SSIM > 0.5699
│  └─ ✅ BEATING BASELINE → Proceed to Day 8-9
└─ SSIM < 0.5699
   └─ ❌ UNDERPERFORMING → Debug, fallback to MSE+ENACT

Day 8-9: Focal Loss
├─ Improvement > 3% over ZINB
│  └─ ✅ FOCAL HELPS → Use Focal ZINB as new baseline
└─ Improvement < 3%
   └─ ❌ FOCAL NOT CRITICAL → Use plain ZINB

Day 10: Robustness Check
├─ SSIM > 0.60 + Stain curves stable + Patient variance < 0.10
│  └─ ✅ STRONG BASELINE → Week 7-8: Encoder/Decoder ablations (not generative)
├─ SSIM < 0.60 + Evidence of multimodality
│  └─ ⚠️ NEED GENERATIVE → Week 9-10: Add diffusion
└─ High variance OR control failures
   └─ ❌ DEBUG FIRST → Week 7: Focus on robustness, not complexity
```

### Week 7-8 Go/No-Go
```
Week 7: Encoder Ablations
├─ One encoder >> others (>5% gain)
│  └─ ✅ CLEAR WINNER → Use for all future experiments
└─ All encoders similar (<2% difference)
   └─ ⚠️ ENCODER NOT CRITICAL → Stick with Prov-GigaPath, focus on decoder/loss

Week 8: Decoder Ablations
├─ One decoder >> others (>5% gain)
│  └─ ✅ CLEAR WINNER → Use for Week 11 integration
└─ All decoders similar (<2% difference)
   └─ ⚠️ DECODER NOT CRITICAL → Stick with Hist2ST, focus on loss/training
```

### Week 11-12 Go/No-Go
```
Week 11: Integration + HyperOpt
├─ HyperOpt finds >2% improvement
│  └─ ✅ USE OPTIMIZED CONFIG → Week 12: Final CV
└─ HyperOpt finds <2% improvement
   └─ ⚠️ USE DEFAULT CONFIG → Week 12: Final CV

Week 12: Final Evaluation
├─ Mean SSIM > 0.60, CI excludes 0.5699, variance < 0.10
│  └─ ✅ SUCCESS → Proceed to Month 4 (Novel Architecture)
├─ Mean SSIM 0.58-0.60, high variance
│  └─ ⚠️ PARTIAL SUCCESS → Month 4: Focus on robustness
└─ Mean SSIM < 0.58 OR controls fail
   └─ ❌ FAILURE → Month 4: Root cause analysis + pivot
```

---

## Conclusion

This roadmap provides a **conservative, test-driven path** from current baseline (SSIM 0.5699) to publication-worthy performance (SSIM >0.60) over 8 weeks.

**Key Principles**:
1. **Infrastructure first** (ENACT, evaluation harness)
2. **Simple before complex** (ZINB before diffusion)
3. **Checkpoints prevent waste** (Go/No-Go decisions every week)
4. **Robustness early** (Week 6, not Month 5)
5. **Understanding over black-box** (ablations identify critical components)

**Expected Outcome**:
- Beat baseline with statistical significance
- Understand **why** improvement happened
- Identify top 3 components for Month 4 novel architecture
- Quantified robustness to stain variation and patient heterogeneity

**Total Timeline**: 8 weeks (56 days), 288 GPU-hours (21% utilization), 16 days slack for debugging/pivots.

---

**Roadmap Status**: READY FOR EXECUTION
**Author**: Claude Sonnet 4.5 (Strategic Planning Agent)
**Date**: 2025-12-26
**Next Action**: Await Director approval, then begin Week 5 Day 1 (ENACT installation)
