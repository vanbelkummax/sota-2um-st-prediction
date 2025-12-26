# SOTA 2μm ST Prediction - Project Roadmap

**Last Updated**: 2025-12-26
**Current Phase**: Month 1 Complete ✅ → Starting Month 2 (Week 5)
**Current Baseline**: Prov-GigaPath + Hist2ST + MSE = SSIM 0.5699 @ 2μm
**Target (Month 3)**: SSIM >0.60 (95% CI excludes baseline)

---

## Overview Timeline

- ✅ **Month 0** (Dec 2025): Design phase complete
- ✅ **Month 1 Week 1** (Dec 2025): Literature synthesis (70+ methods cataloged)
- ✅ **Month 1 Week 2 Day 0-1** (Dec 2025): Evaluation protocol locked, SpatialEvaluator implemented
- 🚀 **Month 2-3** (Jan-Feb 2026): Implementation + ablations (8 weeks, conservative approach)
- ⏸️ **Month 4** (Mar 2026): Novel architecture design
- ⏸️ **Month 5** (Apr 2026): Optimization + external validation
- ⏸️ **Month 6** (May 2026): Publication preparation

---

## Month 1: Literature Synthesis (Weeks 1-4)

### Week 1: Broad Discovery
- [ ] Use polymax-synthesizer to search 27+ methods
- [ ] Search vanderbilt-professors MCP for Huo lab papers
- [ ] Find GitHub repos for all methods
- [ ] Acquire papers via arXiv API, Europe PMC, bioRxiv
- [ ] Add paywalled papers to USER_ACCESS_NEEDED.md
- **Output**: Complete method inventory

### Week 2: Deep Extraction
- [ ] Extract GHIST architecture (Nature Methods 2025) ✅ Paper acquired
- [ ] Extract TRIPLEX architecture (ICLR 2024)
- [ ] Extract THItoGene architecture (Bioinformatics 2024)
- [ ] Extract sCellST architecture (Nature Biotech 2024)
- [ ] Test CONCHv1.5 empirically (ViT-L, acquired)
- **Output**: Architectural specs in polymax DB + memory graph

### Week 3: Pattern Analysis
- [ ] Identify common patterns across SOTA methods
- [ ] Statistical analysis: performance vs complexity
- [ ] Document "critical ingredients" for 2μm success
- **Output**: `docs/synthesis/architectural_patterns.md`

### Week 4: Cross-Disciplinary Mining
- [ ] Mine super-resolution imaging literature (CARE, CSBDeep)
- [ ] Mine weather forecasting (GraphCast)
- [ ] Mine protein structure (AlphaFold)
- [ ] Mine neural rendering (NeRF)
- [ ] Generate 10-15 novel hypothesis candidates
- **Output**: `docs/hypotheses/novel_architecture_ideas.md`

---

## Month 2: Implementation (Weeks 5-8)

**Goal**: Fix ground truth contamination, reproduce baseline exactly
- [ ] Day 1-2: ENACT installation + segmentation sensitivity test
- [ ] Day 2-3: Registration check + naive vs ENACT comparison
- [ ] Day 3-4: Reproduce baseline (must match 0.5699 ± 0.01 SSIM)
- [ ] Day 5: Week 5 synthesis + Go/No-Go decision
- **Deliverable**: Clean cell-level AnnData, baseline reproduced

### Week 6: ZINB Baseline + Focal Loss (Days 6-10)
**Goal**: Beat baseline with better loss function for 2μm sparsity
- [ ] Day 6-7: Implement ZINB loss, train on P1+P2, test on P5
- [ ] Day 8-9: Focal loss sweep (γ ∈ {1, 2, 3})
- [ ] Day 10: Robustness check (stain curves, patient variance)
- **Target**: SSIM >0.60
- **Deliverable**: New baseline (ZINB or Focal ZINB)

### Week 7-8: Encoder/Decoder Ablations (Days 11-20)
**Goal**: Identify best encoder and decoder architectures
- [ ] Week 7 Day 1-3: Test 4 encoders (PAST, Threads, UNI, CONCH)
- [ ] Week 7 Day 4-5: Start decoder experiments
- [ ] Week 8 Day 1-3: Test 3 decoders (THItoGene, DeepSpot2Cell, HyperPNN)
- [ ] Week 8 Day 4-5: Multi-scale hypothesis test (2μm+8μm)
- **Target**: +5% SSIM over Week 6 baseline
- **Deliverable**: Best encoder + decoder combination

### Week 9-10: Generative Models (Days 21-30) [OPTIONAL]
**Goal**: Test if generative models improve over ZINB
**ONLY proceed if Week 6 shows evidence of multimodality**
- [ ] Week 9 Day 1-3: Implement Stem diffusion baseline
- [ ] Week 9 Day 4-5 + Week 10 Day 1-2: ZINB + Diffusion hybrid
- [ ] Week 10 Day 3-5: Synthesis, comparison to ZINB
- **Decision Criteria**: Only proceed if ZINB predictions look "blurry"

### Week 11: Integration + Optimization (Days 31-35)
**Goal**: Combine best components, optimize hyperparameters
- [ ] Day 1-2: Integrate best encoder + decoder + loss + (optional) generative
- [ ] Day 3-5: Bayesian hyperparameter optimization (~15 trials)
- **Target**: +2% SSIM over default hyperparameters
- **Deliverable**: Final optimized model

### Week 12: Final Evaluation (Days 36-40)
**Goal**: Rigorous 3-fold CV, statistical testing, ablations
- [ ] Day 1-2: 3-fold LOOCV (train P1+P2→test P5, etc.)
- [ ] Day 3-4: Component ablation study
- [ ] Day 5: Robustness + negative controls
- **Target**: Mean SSIM >0.60, 95% CI excludes 0.5699
- **Deliverable**: Final comprehensive evaluation report

### Critical Checkpoints (Go/No-Go Decisions)
- **Week 5 Day 5**: Proceed to ZINB? (All infrastructure checkpoints pass)
- **Week 6 Day 10**: Proceed to encoders or generative? (ZINB performance)
- **Week 8 Day 5**: Skip or run generative? (Evidence of multimodality)
- **Week 12 Day 5**: Success or pivot? (Final SSIM vs baseline)

---

## Month 4: Novel Architecture (Weeks 13-16)

### Week 13: Design Workshop
- [ ] Design modular architecture with swappable components
- [ ] Create ablation plan for each component
- [ ] Output: `docs/architecture/novel_system_design.md`

### Week 14-15: Implementation
- [ ] Build modular system at `code/models/novel_predictor.py`
- [ ] Implement multi-encoder fusion (if H001 validated)
- [ ] Implement feature-similarity graph (if H002 promising)
- [ ] Implement gene-category decoders (if H003 validated)
- [ ] Unit tests pass

### Week 16: Initial Validation
- [ ] Quick test on P5 (each component individually)
- [ ] If any component >2% SSIM gain → proceed to Month 5
- [ ] If no gains → pivot to next hypotheses
- [ ] Output: Quick validation report

---

## Month 5: Optimization + Validation (Weeks 17-20)

### Week 17-18: Hyperparameter Optimization
- [ ] Nested CV tuning per outer fold (no test leakage)
- [ ] Grid search: learning rate, decoder depth, graph k-neighbors
- [ ] Budget: ~15 configurations
- [ ] Output: `configs/best_config.yaml`

### Week 19: Full Cross-Validation
- [ ] 3-fold LOOCV with nested-tuned hyperparameters
- [ ] SSIM, Pearson/Spearman per gene, per-category metrics
- [ ] Statistical tests (hierarchical bootstrap, report CI)
- [ ] Output: `results/month5/full_validation_results.csv`

### Week 20: Robustness Testing
- [ ] Per-patient analysis
- [ ] Per-gene-category analysis
- [ ] Failure mode analysis
- [ ] Component ablation study
- [ ] External CRC validation (Ken Lau cohort) if resolution compatible
- [ ] Output: `docs/analysis/robustness_analysis.md`

---

## Month 6: Publications (Weeks 21-24)

### Week 21: Generate Figures
- [ ] Figure 1: Architecture diagram
- [ ] Figure 2: Main results (3-fold CV)
- [ ] Figure 3: Ablation study
- [ ] Figure 4: Biological validation (WSI)
- [ ] Figure 5: Generalization analysis
- [ ] Use LaTeX Architect MCP for professional typesetting

### Week 22: Write Manuscripts
- [ ] Methods paper draft (`manuscripts/methods_paper_2um_st/`)
- [ ] Benchmark paper draft (`manuscripts/benchmark_paper_2um/`)

### Week 23: Compile Benchmark
- [ ] Complete results table (all methods)
- [ ] Encoder/decoder/loss comparison tables
- [ ] Prepare benchmark dataset release

### Week 24: Code Release
- [ ] Clean code, add documentation
- [ ] README, installation guide, usage examples
- [ ] Public GitHub repo
- [ ] HuggingFace weights upload
- [ ] Zenodo dataset upload
- [ ] Apply for HEST-1K access (note limitations if pending)

---

## Pivot Log

*(Document any major plan changes here)*

### 2025-12-26: Strategic Planning Complete
**Decision**: Conservative, test-driven approach for Month 2-3
- **Change from original plan**: Merged Month 2 (implementation) + Month 3 (ablations) into single 8-week block
- **Rationale**: Weekly checkpoints prevent wasted effort, simple-before-complex philosophy
- **Key additions**:
  - ENACT infrastructure (Week 5) - was missing from original plan
  - ZINB loss (Week 6) - prioritized over diffusion based on hypothesis ranking
  - Robustness early (Week 6) - moved from Month 5 to address n=3 fragility
  - Optional generative (Week 9-10) - only if ZINB shows evidence of underfitting
- **Impact**: More conservative timeline, but higher probability of success (80% vs 60% in original plan)

---

## Key Decision Points

- ✅ **Month 0 → Month 1**: Proceed (design complete)
- ✅ **Month 1 Week 1 → Week 2**: Proceed (70+ methods cataloged)
- ✅ **Month 1 Week 2 Day 0-1 → Month 2**: Proceed (evaluation protocol locked)
- ⏸️ **Week 5 → Week 6**: Proceed if ENACT validated + baseline reproduced
- ⏸️ **Week 6 → Week 7**: Proceed if ZINB >0.60 OR shows robustness
- ⏸️ **Week 8 → Week 9**: Skip generative if ZINB >0.65, run if evidence of multimodality
- ⏸️ **Week 12 → Month 4**: Proceed if SSIM >0.60, pivot if <0.58
- ⏸️ **Month 4 → 5**: Proceed if at least 1 component shows >2% SSIM gain
- ⏸️ **Month 5 → 6**: Proceed if novel method beats baseline with significance

---

## Notes

- Update this roadmap weekly
- Check boxes as tasks complete
- Document pivots immediately
- Keep aligned with session summaries
