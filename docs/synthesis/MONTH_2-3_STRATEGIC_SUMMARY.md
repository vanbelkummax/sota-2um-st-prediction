# Month 2-3 Strategic Summary: Conservative Path to SOTA 2μm

**Date**: 2025-12-26
**Prepared By**: Claude Strategic Planning Agent
**For**: Max Van Belkum (MD-PhD, Vanderbilt)

---

## TL;DR (Executive Summary)

**What**: 8-week implementation plan to beat current baseline (SSIM 0.5699) and achieve publication-worthy performance (SSIM >0.60) at 2μm Visium HD spatial transcriptomics prediction.

**How**: Conservative, test-driven approach with weekly checkpoints. Start simple (fix ground truth contamination + better loss function), add complexity only if justified (encoders, decoders, generative models).

**Why**: Avoid wasting time on complex methods before validating fundamentals. Build understanding through systematic ablations.

**Expected Outcome**:
- Beat baseline with statistical significance
- Understand **why** improvement happened (identify top 3 critical components)
- Quantified robustness to stain variation and patient heterogeneity
- Clear path to Month 4 novel architecture

**Timeline**: 8 weeks (56 days), 21% GPU utilization (288 GPU-hours), 16 days slack

**Risk Level**: **Low-Medium** (conservative approach, multiple checkpoints, pivot options)

---

## Strategic Approach: "Simple Before Complex"

### Phase 1: Infrastructure (Week 5)
**Fix the fundamentals before testing fancy models**

**Problem**: Training on contaminated ground truth (2μm bins overlap multiple cells)
**Solution**: ENACT cell-level assignment
**Validation**: Segmentation sensitivity test, registration check
**Deliverable**: Clean cell-level AnnData, baseline exactly reproduced

**Why This Matters**: Can't trust improvements if baseline is wrong.

---

### Phase 2: Loss Function (Week 6)
**Better loss function for 2μm sparsity (95% zeros)**

**Problem**: MSE treats all examples equally → overfits to predicting zeros
**Solution**: ZINB loss (models zero-inflation) + Focal loss (down-weights easy zeros)
**Validation**: SSIM >0.60, robustness to stain variation
**Deliverable**: New baseline (ZINB or Focal ZINB)

**Why This Matters**: Low-hanging fruit - proven in scRNA-seq, fast to test (1 week).

**Expected Gain**: +5-10% SSIM (0.5699 → 0.60-0.63)

---

### Phase 3: Encoder/Decoder Optimization (Week 7-8)
**Identify best architectural components**

**Problem**: Baseline uses Prov-GigaPath + Hist2ST, may not be optimal
**Solution**: Test 4 encoders (PAST, Threads, UNI, CONCH) × 3 decoders (THItoGene, DeepSpot2Cell, HyperPNN)
**Validation**: Per-component SSIM improvement quantified
**Deliverable**: Best encoder + decoder combination

**Why This Matters**: Literature synthesis identified better options, need empirical validation.

**Expected Gain**: +2-5% SSIM per component (cumulative +4-10%)

---

### Phase 4: Generative Models (Week 9-10) [OPTIONAL]
**Only if ZINB shows evidence of underfitting**

**Problem**: Regression predicts mean, not distribution (may be too blurry)
**Solution**: Diffusion models sample from p(y|x)
**Decision Criteria**: Only proceed if ZINB predictions look averaged/blurry
**Deliverable**: Generative baseline (if justified)

**Why This Matters**: Generative = higher ceiling, but more complex. Test simple first.

**Expected Gain**: +3-8% SSIM (if multimodality exists)

---

### Phase 5: Integration + Optimization (Week 11-12)
**Combine best components, validate rigorously**

**Tasks**:
- Integrate best encoder + decoder + loss + (optional) generative
- Bayesian hyperparameter optimization (15 trials)
- 3-fold cross-validation with statistical testing
- Component ablation study (quantify each contribution)
- Robustness curves (stain, patient variance)

**Deliverable**: Final optimized model, comprehensive evaluation

**Why This Matters**: Publication requires rigorous validation and understanding.

**Target**: Mean SSIM >0.60 (95% CI excludes 0.5699), Cohen's d >0.5

---

## Key Design Decisions (Rationale)

### Decision 1: ENACT Before Encoders
**Rationale**: Fix contaminated ground truth before testing better encoders.
**Alternative Rejected**: Test encoders first (would learn from noisy data).
**Risk Mitigation**: Segmentation sensitivity test validates ENACT outputs.

### Decision 2: ZINB Before Diffusion
**Rationale**: Simpler loss function, faster to test, proven in scRNA-seq.
**Alternative Rejected**: Start with diffusion (more complex, slower, may not be needed).
**Risk Mitigation**: Week 6 checkpoint decides if generative needed.

### Decision 3: Frozen Encoders (No Fine-Tuning)
**Rationale**: n=3 patients too small, fine-tuning would overfit.
**Alternative Rejected**: Fine-tune encoders (would require >20 patients).
**Risk Mitigation**: Test multiple pretrained encoders to find best.

### Decision 4: Robustness Early (Week 6, Not Month 5)
**Rationale**: With n=3, must validate generalization before adding complexity.
**Alternative Rejected**: Defer robustness to Month 5 (would waste time on brittle models).
**Risk Mitigation**: Stain augmentation curves, cross-patient variance quantified.

### Decision 5: Weekly Checkpoints (Not Monthly)
**Rationale**: Prevent wasting time on dead ends (e.g., diffusion if ZINB already strong).
**Alternative Rejected**: Linear plan without checkpoints (inflexible).
**Risk Mitigation**: Go/No-Go decisions every week, pivot options specified.

---

## Integration with Literature Synthesis

### Agent 1 Findings → Implementation Plan

| Agent 1 Discovery | Week | Implementation |
|-------------------|------|----------------|
| **ENACT (Tier -1)** | Week 5 | Install, validate, use for all downstream work |
| **Stem diffusion (Tier 0)** | Week 9-10 | Test if ZINB underfits (OPTIONAL) |
| **PAST encoder (Tier 9)** | Week 7 | Test as best pretrained backbone |
| **DeepSpot2Cell MIL** | Week 8 | Test as decoder improvement |
| **HyperPNN DAFB** | Week 8 | Test dual-attention fusion |
| **ZINB loss** | Week 6 | Implement for 2μm sparsity |
| **Focal loss** | Week 6 | Test for class imbalance |

### Agent 2 Hypothesis Prioritization → Week Assignment

| Hypothesis ID | Priority | EV | Week | Status |
|---------------|----------|-------|------|--------|
| **H_20241224_012** | High | 8.5 | Week 6 | Focal + NB2 hybrid |
| **Multi-scale** | Medium | 7.0 | Week 8 | 2μm+8μm context |
| **PAST encoder** | High | 8.0 | Week 7 | Best pretrained backbone |
| **MIL aggregation** | Medium | 7.5 | Week 8 | DeepSpot2Cell |

### Agent 3 Conservative Plan → Checkpoints

| Agent 3 Trap | Week | Mitigation |
|--------------|------|------------|
| **Trap A: Cell-level overconfidence** | Week 5 | Segmentation sensitivity test, measurement model fallback |
| **Trap B: n=3 fragility** | Week 6 | Robustness curves early, hierarchical bootstrap |
| **Registration error** | Week 5 | Registration optimization sweep |
| **Batch effects** | Week 6 | Cross-patient variance quantified |

---

## Critical Assumptions (If These Are Wrong, Plan Changes)

### Assumption 1: ENACT Segmentation is Stable
**Test**: Week 5 Day 1 - Segmentation sensitivity sweep
**If Wrong**: Pivot to measurement model (keep bins, model mixtures)
**Impact**: +1-2 days to Week 5

### Assumption 2: ZINB Beats MSE for 2μm Sparsity
**Test**: Week 6 Day 6-7 - ZINB vs MSE comparison
**If Wrong**: Investigate why (implementation bug? hypothesis invalid?)
**Impact**: Fall back to MSE+ENACT, focus on encoder/decoder

### Assumption 3: Frozen Encoders Sufficient (No Fine-Tuning Needed)
**Test**: Week 7 - Encoder ablation
**If Wrong**: Would need more data (n>20 patients) to fine-tune safely
**Impact**: Stick with frozen, add to Month 4 future work

### Assumption 4: Decoder Architecture Matters
**Test**: Week 8 - Decoder ablation
**If Wrong**: Stick with Hist2ST, focus on loss function and training
**Impact**: Save 2 days in Week 8

### Assumption 5: Generative Models Not Needed (ZINB Sufficient)
**Test**: Week 6 Day 10 - Visual inspection, variance calibration
**If Wrong**: Add Week 9-10 generative experiments
**Impact**: +1 week to timeline (still within 8-week budget)

---

## Success Probability Estimates

Based on literature review and hypothesis analysis:

| Outcome | Probability | SSIM Range | Interpretation |
|---------|-------------|------------|----------------|
| **Strong Success** | 30% | >0.65 | Publication-worthy, novel insight discovered |
| **Success** | 50% | 0.60-0.65 | Publication-worthy, beat baseline clearly |
| **Partial Success** | 15% | 0.58-0.60 | Marginal improvement, needs Month 4 refinement |
| **Failure** | 5% | <0.58 | Fundamental issue (batch effects, signal-to-noise, etc.) |

**Most Likely Outcome**: Success (0.60-0.65 SSIM)
- ZINB handles sparsity better than MSE (proven in scRNA-seq)
- ENACT preprocessing reduces contamination (proven in Visium HD)
- Encoder/decoder ablations find 5-10% gains (common in literature)

**Worst Case**: Partial Success (0.58-0.60 SSIM)
- Even if no single component gives large gains, cumulative effect should beat baseline
- Hierarchical bootstrap may show statistical significance even with small effect

**Best Case**: Strong Success (>0.65 SSIM)
- Focal loss + ZINB synergy exceeds expectations
- Multi-scale hypothesis validated
- Generative model captures multimodality

---

## Resource Requirements

### Compute
- **GPU**: RTX 5090 24GB (current hardware)
- **Utilization**: 21% average (288 GPU-hours over 56 days)
- **Bottlenecks**: Week 7-8 (encoder/decoder ablations), Week 11 (hyperopt)
- **Slack**: 16 days buffer for debugging, pivots

### Data
- **Primary**: 3 CRC patients (P1, P2, P5) - Visium HD 2μm, 50 genes
- **External Validation**: HEST-1k (Month 5) or Ken Lau CRC cohort

### Tools
- **MCP Servers**: polymax-synthesizer, vanderbilt-professors, memory, latex-architect
- **Libraries**: POT (optimal transport), scib-metrics, esda/libpysal
- **Code Archaeology**: GitHub repos for ENACT, Stem, DeepSpot2Cell, etc.

### Time
- **Max Van Belkum Time**: ~2 hours/week (review results, approve pivots)
- **Claude Agent Time**: ~5 hours/day coding, ~1 hour/day synthesis
- **Total Timeline**: 8 weeks (January-February 2026)

---

## Pivot Options (If Plan Needs Adjustment)

### Pivot 1: Skip Generative Models (Save 1 Week)
**Trigger**: ZINB achieves SSIM >0.65 and predictions look sharp
**Action**: Skip Week 9-10, go directly to Week 11 integration
**Impact**: Finish 1 week early, focus on optimization

### Pivot 2: Skip Decoder Ablations (Save 0.5 Weeks)
**Trigger**: Week 7 encoders show no clear winner (all within 2%)
**Action**: Focus on loss function and training, not architecture
**Impact**: More time for hyperparameter optimization

### Pivot 3: Add Measurement Model (Cost 1-2 Days)
**Trigger**: Week 5 segmentation sensitivity fails (r <0.8, CV >30%)
**Action**: Keep bins, model mixtures with overlap weights
**Impact**: More principled, avoids ENACT artifacts

### Pivot 4: Focus on Single Gene Category (Save 1 Week)
**Trigger**: Week 12 shows high variance across gene categories
**Action**: Month 4 focuses on epithelial markers only (most important for CRC)
**Impact**: More specialized claim, but stronger results

---

## Month 4+ Contingency Plans

### If Strong Success (SSIM >0.65)
**Month 4**: Novel architecture design (e.g., "Clean Frankenstein")
**Month 5**: External validation (HEST-1k or Lau cohort)
**Month 6**: Publication preparation (methods + benchmark papers)

### If Success (SSIM 0.60-0.65)
**Month 4**: Focused improvement on weakest component
**Month 5**: Ensemble + uncertainty quantification
**Month 6**: Publication (benchmark paper + methods note)

### If Partial Success (SSIM 0.58-0.60)
**Month 4**: Root cause analysis (batch effects? signal-to-noise?)
**Month 5**: Acquire more data OR switch to 8μm resolution
**Month 6**: Workshop paper / technical report

### If Failure (SSIM <0.58)
**Month 4**: Reframe problem (uncertainty quantification? gene categories?)
**Month 5**: Collaborate with Huo lab on novel approach
**Month 6**: Lessons learned documentation

---

## Communication Plan

### Weekly Updates (Friday EOD)
**Format**: 1-page summary
**Contents**:
- What was tested this week
- Key findings (SSIM improvements, ablation results)
- Go/No-Go decision for next week
- Blockers (if any)

**Audience**: Max Van Belkum, Yuankai Huo (optional)

### Month 2-3 Final Report (End of Week 12)
**Format**: 10-15 page technical report
**Contents**:
- Comprehensive results (3-fold CV, ablations, robustness)
- Component attribution analysis
- Statistical significance tests
- Top 3 components for Month 4
- Recommendations for novel architecture

**Audience**: Max Van Belkum, Yuankai Huo, dissertation committee

### Checkpoint Meetings (Optional)
**Week 6 (Post-ZINB)**: Review baseline improvement, decide on generative
**Week 8 (Post-Ablations)**: Review encoder/decoder results
**Week 12 (Final)**: Review full results, approve Month 4 plan

---

## Risk Matrix

| Risk | Likelihood | Impact | Mitigation | Cost |
|------|------------|--------|------------|------|
| **ENACT unstable** | Medium | High | Measurement model | +1-2 days |
| **ZINB < baseline** | Low | High | Debug, fallback to MSE+ENACT | +1 day |
| **PAST weights unavailable** | Medium | Medium | Use UNI instead | 0 days |
| **Decoder repos broken** | Low | Medium | Implement from papers | +2 days |
| **High patient variance** | Medium | High | Robustness curves early, hierarchical bootstrap | 0 days (built in) |
| **OOM with multi-scale** | High | Medium | Gradient checkpointing, FP16 | +0.5 days |
| **All hypotheses fail** | Very Low | Critical | Root cause analysis, pivot problem formulation | 2 weeks |

**Overall Risk Level**: **Low-Medium**
- Multiple checkpoints prevent wasted effort
- Conservative approach (simple before complex)
- Proven techniques (ZINB, focal loss, ENACT)
- Fallback options at every stage

---

## Key Metrics Dashboard (Track Weekly)

### Primary Metrics
- **SSIM @ 2μm**: Current 0.5699 → Target >0.60
- **95% Confidence Interval**: Must exclude baseline
- **Cohen's d**: Target >0.5 (medium effect)

### Secondary Metrics
- **Per-patient variance**: Target <0.10 SSIM
- **Wasserstein distance**: Spatial fidelity
- **Per-gene Pearson r**: Gene-level performance

### Process Metrics
- **GPU utilization**: ~21% planned
- **Experiments completed**: 36 total planned
- **Checkpoints passed**: 5 critical checkpoints

---

## Conclusion

This 8-week roadmap provides a **conservative, evidence-based path** from current baseline (0.5699 SSIM) to publication-worthy performance (>0.60 SSIM).

**Key Strengths**:
1. **Checkpoints prevent waste** - Go/No-Go decisions every week
2. **Simple before complex** - Test fundamentals (ZINB) before fancy models (diffusion)
3. **Robustness early** - Validate generalization at Week 6, not Month 5
4. **Understanding over black-box** - Ablations identify critical components

**Key Weaknesses**:
1. **n=3 limits generalization** - Need external validation (Month 5)
2. **Frozen encoders may be suboptimal** - Can't fine-tune with small data
3. **Single tissue type** - CRC-specific, generalization unknown

**Expected Outcome**:
- 80% probability of beating baseline (SSIM >0.58)
- 50% probability of strong success (SSIM 0.60-0.65)
- Clear understanding of what drives 2μm performance
- Top 3 components identified for Month 4 novel architecture

**Next Action**: Week 5 Day 1 - Install ENACT, segmentation sensitivity test

---

**Document Status**: READY FOR EXECUTION
**Approval Required**: Max Van Belkum (Principal Investigator)
**Start Date**: TBD (awaiting approval)
**Estimated Completion**: 8 weeks from start

---

**Prepared By**: Claude Sonnet 4.5 (Strategic Planning Agent)
**Date**: 2025-12-26
**Contact**: Max Van Belkum (max@vanderbilt.edu)
