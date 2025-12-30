# SOTA 2μm Spatial Transcriptomics Prediction: Master Roadmap

**Date**: 2025-12-27
**Author**: Max Van Belkum, MD-PhD Student
**Status**: ACTIVE - Ready for Execution
**Synthesized From**: 5-Agent Strategic Analysis + Phase 1-3 Experimental Results

---

## Executive Summary

### Current State
- **Best Result**: H_013 (Prov-GigaPath + Poisson + Hist2ST) = **SSIM 0.5699** @ 2μm
- **Phase 3 Multi-task**: SSIM 0.5635 (+0.18% vs Phase 1, -1.1% vs H_013)
- **Target**: SSIM >0.60 (95% CI excludes baseline)
- **Stretch Goal**: SSIM >0.70 (Nature Methods-caliber)

### Critical Insight
**Phase 3 underperformed expectations** (+0.18% vs expected +10-14%) because we're training on contaminated ground truth. The 5-agent synthesis correctly identified **ENACT infrastructure as the blocking issue**.

### Strategic Recommendation
Execute a **4-phase, 8-week plan**:
1. **Phase A**: Infrastructure (ENACT) - Fix ground truth contamination
2. **Phase B**: Loss Function Optimization - ZINB/Focal for 95% sparsity
3. **Phase C**: Encoder/Decoder Ablations - Find best components
4. **Phase D**: Integration & Validation - Combine best elements

**Expected Outcome**: SSIM 0.65-0.75 by Week 8, with clear path to 0.80+

---

## Part 1: Current State Assessment

### 1.1 Performance Hierarchy (All Experiments to Date)

| Experiment | SSIM @ 2μm | PCC @ 2μm | Per-Gene PCC | Status |
|------------|------------|-----------|--------------|--------|
| **Target** | **>0.60** | >0.35 | >0.30 | Goal |
| H_013 (Prov-GigaPath + Poisson + Hist2ST) | **0.5699** | 0.3345 | 0.2566 | **Current Best** |
| H_014 Phase 3 (Multi-task w/ nuclei) | 0.5635 | 0.3275 | 0.2552 | -1.1% vs best |
| H_014 Phase 1 (Poisson only) | 0.5625 | 0.3250 | 0.2551 | -1.3% vs best |
| Baseline (Virchow2 + Poisson) | 0.5494 | 0.3507 | 0.2499 | Original |
| H_012 (ZINB + Virchow2) | 0.5212 | 0.3245 | 0.2497 | Failed |
| H_011 (ZINB + Prov-GigaPath + MiniUNet) | 0.4655 | 0.2398 | 0.2315 | Failed |

### 1.2 Key Findings from Experiments

**What Worked:**
1. **Prov-GigaPath > Virchow2**: +3.7% SSIM (whole-slide pretraining helps)
2. **Poisson >> ZINB**: ZINB consistently underperformed (optimization issues)
3. **Hist2ST decoder robust**: Works well across encoders

**What Didn't Work:**
1. **ZINB loss**: -5% to -15% SSIM (collapses to predicting zeros)
2. **MiniUNet decoder**: Poor with ZINB (-15.3%)
3. **Multi-task on noisy data**: Only +0.18% (expected +10-14%)

### 1.3 Gap Analysis: Why We're Stuck at 0.57

| Gap | Impact | Evidence | Solution |
|-----|--------|----------|----------|
| **Ground truth contamination** | HIGH | Multi-task only +0.18% vs expected +14% | ENACT bin→cell assignment |
| **Loss function mismatch** | MEDIUM | ZINB failed, Poisson plateau | Focal loss for class imbalance |
| **Limited encoder diversity** | LOW | Only tested Virchow2, Prov-GigaPath | Test PAST, Threads, UNI, CONCH |
| **No generative baseline** | MEDIUM | Regression may hit ceiling | Test Stem diffusion |

---

## Part 2: The Blocking Issue - Ground Truth Contamination

### 2.1 The Problem

At 2μm resolution, **bins and cells don't align**:
- A single 2μm bin may contain parts of multiple cells
- A single cell may span multiple bins
- Current approach: Assign bin expression to center coordinate
- **Result**: ~20% label noise in ground truth

### 2.2 Why This Matters

From 5-agent synthesis:
> "ENACT identified as THE MISSING PIECE (Tier -1 priority). Without this, all downstream experiments inherit contaminated ground truth. Multi-task learning on noisy labels will show minimal improvement."

**Phase 3 Validation**: Multi-task nuclei segmentation only gained +0.18% SSIM
- Expected from GHIST evidence: +10-14%
- Actual: +0.18%
- **Explanation**: Model learns noisy spatial patterns, auxiliary task can't fix fundamental data quality issue

### 2.3 The Solution: ENACT Pipeline

**ENACT** (Sanofi-Public) provides proper bin→cell assignment:
1. **Nuclei Segmentation**: StarDist/Cellpose on H&E
2. **Cell Boundary Expansion**: Watershed to cytoplasm
3. **Weighted Assignment**: Fractional overlap between cells and bins
4. **Output**: Cell-level AnnData with proper ground truth

**Repository**: `github.com/Sanofi-Public/enact-pipeline`

---

## Part 3: Phase-by-Phase Implementation Plan

### Phase A: Infrastructure (Days 1-5)

**Goal**: Fix ground truth contamination, establish clean baseline

#### Day 1-2: ENACT Setup
```bash
# Clone and install
git clone https://github.com/Sanofi-Public/enact-pipeline
cd enact-pipeline
conda create -n enact python=3.9
conda activate enact
pip install -e .

# Verify installation
python -c "from enact import pipeline; print('ENACT ready')"
```

**Tasks**:
- [ ] Clone ENACT repository
- [ ] Install dependencies (StarDist, scipy, anndata)
- [ ] Test on P1 ROI (verify output structure)
- [ ] Document any installation issues

**Success Criteria**: ENACT runs on P1 without errors

#### Day 2-3: Process All Patients
```bash
# Process each patient
python run_enact.py --patient P1 --resolution 2um --output /home/user/work/enact_data/P1_cells/
python run_enact.py --patient P2 --resolution 2um --output /home/user/work/enact_data/P2_cells/
python run_enact.py --patient P5 --resolution 2um --output /home/user/work/enact_data/P5_cells/
```

**Tasks**:
- [ ] Run ENACT on P1 (full slide, not just ROI)
- [ ] Run ENACT on P2
- [ ] Run ENACT on P5 (held-out test)
- [ ] Validate output: cell count, gene count, spatial coordinates

**Success Criteria**:
- Cell-level AnnData files for all 3 patients
- Reasonable cell counts (10K-100K cells per patient)
- All 50 genes present

#### Day 3-4: Segmentation Sensitivity Analysis
**Critical Validation**: Verify segmentation stability

```python
# Test segmentation sensitivity
segmentation_methods = ['stardist_default', 'stardist_finetuned', 'cellpose_cyto2']
results = []

for method in segmentation_methods:
    cells = run_enact(P1, segmentation=method)
    expr_matrix = cells.X
    results.append({
        'method': method,
        'n_cells': cells.n_obs,
        'mean_expr': expr_matrix.mean(),
        'sparsity': (expr_matrix == 0).mean()
    })

# Decision criteria
cv = np.std([r['mean_expr'] for r in results]) / np.mean([r['mean_expr'] for r in results])
if cv < 0.20:
    decision = "STABLE - Use ENACT cells as ground truth"
else:
    decision = "UNSTABLE - Use measurement model instead"
```

**Tasks**:
- [ ] Test 3 segmentation methods on P1
- [ ] Compare cell counts, expression distributions
- [ ] Calculate coefficient of variation
- [ ] Make Go/No-Go decision

**Success Criteria**:
- CV < 20% across segmentation methods
- OR pivot to measurement model approach

#### Day 4-5: Clean Baseline Establishment
**Goal**: Re-run H_013 architecture on ENACT-processed data

```python
# Training on cell-level data
config = {
    'encoder': 'prov-gigapath',
    'decoder': 'hist2st',
    'loss': 'poisson',
    'data': 'enact_cell_level',  # NEW
    'train_patients': ['P1', 'P2'],
    'test_patient': 'P5'
}

model = train_img2st(config)
metrics = evaluate(model, P5_cells)
```

**Tasks**:
- [ ] Modify data loader for cell-level AnnData
- [ ] Train H_013 architecture on ENACT data
- [ ] Evaluate on P5
- [ ] Compare to bin-level baseline (SSIM 0.5699)

**Success Criteria**:
- Training completes without errors
- SSIM on ENACT data >= 0.5699 (should improve with clean labels)

#### Phase A Checkpoint (End of Day 5)

| Metric | Required | Stretch |
|--------|----------|---------|
| ENACT pipeline working | YES | - |
| Segmentation CV | < 20% | < 10% |
| Cell-level baseline SSIM | >= 0.5699 | >= 0.60 |
| All 3 patients processed | YES | - |

**Go/No-Go Decision**:
- ✅ All required met → Proceed to Phase B
- ❌ Segmentation unstable → Pivot to measurement model
- ❌ SSIM drops → Investigate ENACT processing

---

### Phase B: Loss Function Optimization (Days 6-10)

**Goal**: Find optimal loss for 95% sparse 2μm data

#### Day 6-7: Focal Poisson Loss

**Rationale**: 95% zeros = extreme class imbalance. Focal loss proven for imbalanced classification (object detection: 99% background).

```python
class FocalPoissonLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_rate, target):
        # Standard Poisson NLL
        poisson_nll = pred_rate - target * torch.log(pred_rate + 1e-8)

        # Focal weighting: down-weight easy (zero) examples
        p = torch.exp(-poisson_nll)  # "confidence"
        focal_weight = (1 - p) ** self.gamma

        # Alpha weighting for non-zeros
        alpha_weight = torch.where(target > 0, self.alpha, 1 - self.alpha)

        return (alpha_weight * focal_weight * poisson_nll).mean()
```

**Tasks**:
- [ ] Implement FocalPoissonLoss
- [ ] Grid search: γ ∈ {1, 2, 3, 5}, α ∈ {0.1, 0.25, 0.5}
- [ ] Train on ENACT data (P1+P2 → P5)
- [ ] Compare to baseline Poisson

**Success Criteria**: SSIM improvement > 2% over Poisson baseline

#### Day 8-9: Negative Binomial (without Zero-Inflation)

**Rationale**: ZINB failed due to zero-inflation parameter (π) collapsing. Try NB without ZI.

```python
class NegativeBinomialLoss(nn.Module):
    def __init__(self, theta_init=1.0):
        super().__init__()
        self.log_theta = nn.Parameter(torch.tensor(theta_init).log())

    def forward(self, mu, target):
        theta = self.log_theta.exp()

        # NB NLL
        nll = (
            torch.lgamma(target + theta)
            - torch.lgamma(theta)
            - torch.lgamma(target + 1)
            + theta * torch.log(theta / (theta + mu))
            + target * torch.log(mu / (theta + mu))
        )
        return -nll.mean()
```

**Tasks**:
- [ ] Implement NB loss (learnable dispersion θ)
- [ ] Train on ENACT data
- [ ] Monitor θ convergence (should stabilize, not collapse)
- [ ] Compare to Focal Poisson

**Success Criteria**: θ stabilizes (not → 0 or → ∞)

#### Day 10: Robustness Validation

**Critical**: With n=3 patients, must validate robustness early

```python
# Stain augmentation test
augmentations = ['none', 'color_jitter', 'stain_normalize', 'randaugment']
results = {}

for aug in augmentations:
    model = train_with_augmentation(aug)
    results[aug] = {
        'P5_ssim': evaluate(model, P5),
        'variance': compute_per_patch_variance(model, P5)
    }

# Cross-patient variance
patient_ssims = [evaluate(model, P1), evaluate(model, P2), evaluate(model, P5)]
cross_patient_cv = np.std(patient_ssims) / np.mean(patient_ssims)
```

**Tasks**:
- [ ] Test 4 augmentation strategies
- [ ] Compute cross-patient variance
- [ ] Identify best augmentation for robustness
- [ ] Document any patient-specific failure modes

**Success Criteria**:
- Cross-patient CV < 15%
- No single patient SSIM < 0.50

#### Phase B Checkpoint (End of Day 10)

| Metric | Required | Stretch |
|--------|----------|---------|
| Best loss function identified | YES | - |
| SSIM improvement over Poisson | > 2% | > 5% |
| Cross-patient CV | < 15% | < 10% |
| Robustness to augmentation | Stable | - |

**Go/No-Go Decision**:
- ✅ SSIM > 0.60 → Consider skipping to Phase D
- ✅ SSIM 0.58-0.60 → Proceed to Phase C
- ❌ SSIM < 0.58 → Debug loss function, check data

---

### Phase C: Encoder/Decoder Ablations (Days 11-20)

**Goal**: Find optimal encoder and decoder combination

#### Days 11-13: Encoder Ablation

**Encoders to Test** (priority order):
1. **Prov-GigaPath** (current best, 1.3B tiles)
2. **PAST** (pretrained on H&E → genes, if weights available)
3. **Threads** (molecular supervision)
4. **UNI** (100M patches, general pathology)
5. **CONCH** (vision-language, 1.17M pairs)

```python
encoders = {
    'prov-gigapath': {'dim': 1536, 'weights': 'hf://prov-gigapath'},
    'uni': {'dim': 1024, 'weights': 'hf://MahmoodLab/UNI'},
    'conch': {'dim': 512, 'weights': 'hf://MahmoodLab/CONCH'},
    # Add PAST if weights found
}

results = {}
for name, config in encoders.items():
    model = Img2STNet(encoder=name, decoder='hist2st', loss='focal_poisson')
    model.train(P1_cells, P2_cells)
    results[name] = evaluate(model, P5_cells)
```

**Tasks**:
- [ ] Search for PAST weights (HuggingFace, GitHub)
- [ ] Search for Threads weights
- [ ] Test each encoder with best loss from Phase B
- [ ] Run 2 encoders in parallel (GPU utilization)

**Success Criteria**: Identify encoder with best SSIM

#### Days 14-16: Decoder Ablation

**Decoders to Test**:
1. **Hist2ST** (current, Transformer + GNN)
2. **THItoGene** (Dynamic ConvNets + CapsNet)
3. **DeepSpot2Cell MIL** (Multi-Instance Learning)
4. **HyperPNN** (Dual-Attention Fusion from remote sensing)

```python
decoders = {
    'hist2st': Hist2STDecoder,
    'thitogene': THItoGeneDecoder,  # From GitHub
    'deepspot_mil': DeepSpotMILDecoder,  # From GitHub
    'hyperpnn': HyperPNNDecoder  # Cross-disciplinary
}

for name, decoder_class in decoders.items():
    model = Img2STNet(encoder=best_encoder, decoder=decoder_class, loss='focal_poisson')
    results[name] = evaluate(model, P5_cells)
```

**Tasks**:
- [ ] Clone THItoGene repository, extract decoder
- [ ] Clone DeepSpot2Cell repository, extract MIL aggregator
- [ ] Implement HyperPNN dual-attention fusion
- [ ] Test each decoder with best encoder

**Success Criteria**: Identify decoder with best SSIM

#### Days 17-18: Multi-Task with Clean Data

**Re-test multi-task now that data is clean**

```python
class MultiTaskImg2STNet(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.shared_decoder = decoder.shared_layers

        # Task heads
        self.gene_head = decoder.gene_head
        self.nuclei_head = nn.Conv2d(256, 1, 1)  # Binary segmentation
        self.celltype_head = nn.Conv2d(256, n_celltypes, 1)

    def forward(self, img, nuclei_mask=None, celltype_labels=None):
        features = self.encoder(img)
        shared = self.shared_decoder(features)

        pred_genes = self.gene_head(shared)
        pred_nuclei = self.nuclei_head(shared)
        pred_celltype = self.celltype_head(shared)

        # Multi-task loss (from GHIST: unweighted sum works)
        loss = (
            focal_poisson_loss(pred_genes, target_genes) +
            dice_loss(pred_nuclei, nuclei_mask) +
            ce_loss(pred_celltype, celltype_labels)
        )
        return pred_genes, loss
```

**Tasks**:
- [ ] Add nuclei segmentation head
- [ ] Add cell-type prediction head (derive labels from clustering)
- [ ] Train multi-task model on ENACT data
- [ ] Compare to single-task baseline

**Expected Improvement**: +5-10% SSIM (now that data is clean!)

#### Days 19-20: Multi-Scale Fusion

**Combine 2μm + 8μm features**

```python
class MultiScaleImg2ST(nn.Module):
    def forward(self, img_2um, img_8um):
        feat_2um = self.encoder(img_2um)  # Local morphology
        feat_8um = self.encoder(img_8um)  # Tissue context

        # Upsample 8um to match 2um
        feat_8um_up = F.interpolate(feat_8um, size=feat_2um.shape[-2:])

        # Fusion (attention or concatenation)
        fused = self.fusion(torch.cat([feat_2um, feat_8um_up], dim=1))

        return self.decoder(fused)
```

**Tasks**:
- [ ] Create multi-scale data loader (extract 8μm patches)
- [ ] Implement feature fusion module
- [ ] Train multi-scale model
- [ ] Compare to single-scale baseline

**Success Criteria**: SSIM improvement > 2%

#### Phase C Checkpoint (End of Day 20)

| Metric | Required | Stretch |
|--------|----------|---------|
| Best encoder identified | YES | - |
| Best decoder identified | YES | - |
| Multi-task improvement | > 5% | > 10% |
| Multi-scale improvement | > 2% | > 5% |
| Combined SSIM | > 0.62 | > 0.68 |

**Go/No-Go Decision**:
- ✅ SSIM > 0.65 → Skip generative, proceed to Phase D
- ✅ SSIM 0.60-0.65 → Consider generative (optional)
- ❌ SSIM < 0.60 → Debug, check for implementation errors

---

### Phase D: Integration & Validation (Days 21-28)

**Goal**: Combine best components, rigorous validation

#### Days 21-23: Component Integration

```python
# Final model configuration
final_config = {
    'encoder': best_encoder,  # From Phase C
    'decoder': best_decoder,  # From Phase C
    'loss': 'focal_poisson',  # From Phase B
    'multi_task': True,
    'multi_scale': True,
    'augmentation': best_augmentation  # From Phase B
}

final_model = build_final_model(final_config)
```

**Tasks**:
- [ ] Integrate best encoder + decoder + loss
- [ ] Add multi-task heads
- [ ] Add multi-scale fusion
- [ ] Hyperparameter tuning (lr, batch_size, epochs)

#### Days 24-25: 3-Fold Cross-Validation

```python
# Leave-One-Out CV (n=3)
folds = [
    {'train': ['P1', 'P2'], 'test': 'P5'},
    {'train': ['P1', 'P5'], 'test': 'P2'},
    {'train': ['P2', 'P5'], 'test': 'P1'}
]

cv_results = []
for fold in folds:
    model = train(fold['train'])
    cv_results.append(evaluate(model, fold['test']))

mean_ssim = np.mean([r['ssim'] for r in cv_results])
std_ssim = np.std([r['ssim'] for r in cv_results])
```

**Tasks**:
- [ ] Train 3 models (one per fold)
- [ ] Evaluate each on held-out patient
- [ ] Compute mean ± std SSIM
- [ ] Identify any patient-specific failures

#### Days 26-27: Statistical Testing & Ablations

```python
# Hierarchical bootstrap (valid for n=3)
def hierarchical_bootstrap(novel_results, baseline_results, n_bootstrap=10000):
    deltas = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(3, size=3, replace=True)
        delta = np.mean([novel_results[i] for i in idx]) - np.mean([baseline_results[i] for i in idx])
        deltas.append(delta)

    ci_lower, ci_upper = np.percentile(deltas, [2.5, 97.5])
    cohens_d = np.mean(deltas) / np.std(deltas)

    return {
        'mean_improvement': np.mean(deltas),
        'ci_95': (ci_lower, ci_upper),
        'cohens_d': cohens_d,
        'significant': ci_lower > 0
    }
```

**Tasks**:
- [ ] Run hierarchical bootstrap (novel vs H_013)
- [ ] Compute 95% CI
- [ ] Compute Cohen's d effect size
- [ ] Run component ablations (remove each component, measure drop)

**Success Criteria**:
- 95% CI excludes 0 (significant improvement)
- Cohen's d > 0.5 (medium effect size)

#### Day 28: Final Report & Negative Controls

```python
# Negative controls
controls = {
    'label_shuffle': shuffle_gene_labels(P5),
    'spatial_jitter': add_coordinate_noise(P5, sigma=10),
    'random_encoder': RandomEncoder()
}

for name, control in controls.items():
    control_ssim = evaluate(final_model, control)
    assert control_ssim < 0.40, f"Model passes {name} control"
```

**Tasks**:
- [ ] Run 3 negative controls
- [ ] Verify model learns biology, not artifacts
- [ ] Generate final performance report
- [ ] Create visualizations (predictions vs ground truth)

#### Phase D Checkpoint (End of Day 28)

| Metric | Required | Stretch |
|--------|----------|---------|
| Mean CV SSIM | > 0.60 | > 0.68 |
| 95% CI excludes baseline | YES | - |
| Cohen's d | > 0.5 | > 0.8 |
| All negative controls pass | YES | - |
| Per-patient SSIM std | < 0.08 | < 0.05 |

---

## Part 4: Success Metrics Summary

### Primary Success Criteria

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| SSIM @ 2μm | 0.60 | 0.65 | 0.75 |
| PCC @ 2μm | 0.35 | 0.40 | 0.50 |
| Per-gene PCC mean | 0.28 | 0.32 | 0.40 |
| 95% CI excludes baseline | YES | - | - |
| Cohen's d effect size | 0.5 | 0.8 | 1.2 |

### Secondary Metrics (Multi-task)

| Metric | Target |
|--------|--------|
| Nuclei Dice | > 0.30 |
| Cell-type F1 | > 0.50 |
| Cross-patient SSIM std | < 0.08 |

### Negative Controls (All Must Pass)

| Control | Expected SSIM |
|---------|---------------|
| Label shuffle | < 0.35 |
| Spatial jitter (10px) | < 0.45 |
| Random encoder | < 0.30 |

---

## Part 5: Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation | Contingency |
|------|-------------|--------|------------|-------------|
| ENACT installation fails | Low | HIGH | Docker container | Use measurement model |
| ENACT segmentation unstable | Medium | HIGH | Sensitivity analysis | Measurement model fallback |
| Focal loss doesn't help | Low | Medium | Try NB, Huber | Stick with Poisson |
| PAST weights unavailable | Medium | Low | Use UNI instead | Already have backup |
| Multi-task hurts performance | Low | Medium | Ablate each task | Use single-task |
| OOM with multi-scale | High | Low | Gradient checkpointing | FP16, reduce batch |
| Statistical power insufficient | Medium | Medium | Hierarchical bootstrap | Effect size reporting |

### Decision Trees

```
ENACT Segmentation Sensitivity
├── CV < 10%: Use ENACT cells directly
├── CV 10-20%: Use ENACT with measurement model uncertainty
├── CV 20-30%: Use measurement model, investigate segmentation
└── CV > 30%: Abandon ENACT, use alternative approach

Loss Function Selection
├── Focal Poisson > Poisson + 3%: Use Focal Poisson
├── NB > Focal + 2%: Use NB
├── All similar: Stick with Poisson (simplicity)
└── All worse: Debug implementation

Multi-task Decision
├── +5% SSIM with clean data: Keep multi-task
├── +2-5% SSIM: Keep if compute allows
├── <2% SSIM improvement: Single-task (simpler)
└── Hurts performance: Remove auxiliary tasks
```

---

## Part 6: Resource Requirements

### Compute

| Resource | Available | Required | Status |
|----------|-----------|----------|--------|
| GPU VRAM | 24 GB (RTX 5090) | 16-24 GB | ✅ |
| CPU cores | 24 | 8+ | ✅ |
| RAM | 196 GB | 64+ GB | ✅ |
| Storage | ~1 TB | 200 GB | ✅ |

### Time Budget

| Phase | Days | GPU-Hours | Notes |
|-------|------|-----------|-------|
| Phase A: Infrastructure | 5 | 20 | ENACT processing |
| Phase B: Loss Functions | 5 | 40 | Grid search |
| Phase C: Ablations | 10 | 120 | Multiple encoders/decoders |
| Phase D: Integration | 8 | 60 | 3-fold CV, bootstrap |
| **Total** | **28** | **240** | ~10 GPU-days |

### Data

| Dataset | Size | Status |
|---------|------|--------|
| P1 CRC | 33 GB | ✅ Ready |
| P2 CRC | 33 GB | ✅ Ready |
| P5 CRC | 33 GB | ✅ Ready |
| ENACT outputs | ~5 GB | Pending |

---

## Part 7: Timeline

### Week 1 (Days 1-7)
- [x] Day 1-2: ENACT installation and setup
- [ ] Day 2-3: Process all patients
- [ ] Day 3-4: Segmentation sensitivity analysis
- [ ] Day 4-5: Clean baseline establishment
- [ ] Day 6-7: Focal Poisson implementation

### Week 2 (Days 8-14)
- [ ] Day 8-9: NB loss testing
- [ ] Day 10: Robustness validation
- [ ] Day 11-13: Encoder ablation
- [ ] Day 14: Best encoder identified

### Week 3 (Days 15-21)
- [ ] Day 15-16: Decoder ablation
- [ ] Day 17-18: Multi-task with clean data
- [ ] Day 19-20: Multi-scale fusion
- [ ] Day 21: Component integration begins

### Week 4 (Days 22-28)
- [ ] Day 22-23: Final model training
- [ ] Day 24-25: 3-fold CV
- [ ] Day 26-27: Statistical testing & ablations
- [ ] Day 28: Final report & negative controls

---

## Part 8: Immediate Next Steps

### Today (Day 1)

```bash
# Step 1: Clone ENACT
cd /home/user/work
git clone https://github.com/Sanofi-Public/enact-pipeline
cd enact-pipeline

# Step 2: Create environment
conda create -n enact_sota python=3.9 -y
conda activate enact_sota
pip install -e .

# Step 3: Test installation
python -c "from enact import pipeline; print('ENACT ready')"

# Step 4: Run on P1 ROI (quick test)
python run_enact.py --input /home/user/work/enact_data/P1/ --output ./P1_test/
```

### Decision Points

1. **End of Day 5**: ENACT working? Baseline established?
2. **End of Day 10**: Best loss function? Robustness confirmed?
3. **End of Day 20**: Best encoder/decoder? Multi-task helping?
4. **End of Day 28**: Final SSIM? Significant improvement?

---

## Appendices

### Appendix A: Code Templates

See `/home/user/sota-2um-st-prediction/code/templates/` for:
- `focal_poisson_loss.py`
- `nb_loss.py`
- `multitask_model.py`
- `multiscale_encoder.py`
- `evaluation_harness.py`

### Appendix B: Reference Documents

- `/home/user/Desktop/STRATEGIC_PLAN_SOTA_2UM_COMPLETE.md` - 5-agent synthesis
- `/home/user/work/enact_data/Architectural_Pattern_Analysis_2025-12-26.md` - 27-method analysis
- `/home/user/sota-2um-st-prediction/docs/synthesis/COMPREHENSIVE_SYNTHESIS_2025-12-26.md` - Literature synthesis
- `/home/user/work/encoder-loss-ablation-2um/RESULTS.md` - Phase 1-3 results

### Appendix C: Key Papers

| Paper | PMID | Relevance |
|-------|------|-----------|
| Img2ST-Net (Huo) | 41210922 | Our baseline |
| GHIST (Nat Methods) | Pending | Multi-task evidence |
| DeepSpot (2025) | Pending | Foundation + MIL |
| ENACT (Sanofi) | Pending | Bin→cell assignment |

---

## Document Metadata

**Version**: 1.0
**Created**: 2025-12-27
**Last Updated**: 2025-12-27
**Author**: Max Van Belkum
**Advisor**: Yuankai Huo, PhD
**Status**: ACTIVE - Ready for Execution
**Next Review**: End of Week 1 (Day 7)

---

**EXECUTE THIS PLAN STARTING TODAY**
