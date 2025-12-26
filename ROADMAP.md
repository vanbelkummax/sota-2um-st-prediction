# SOTA 2μm ST Prediction - Project Roadmap

**Last Updated**: 2025-12-26
**Current Phase**: Month 0 - Design Complete ✅

---

## Overview Timeline

- ✅ **Month 0** (Dec 2025): Design phase complete
- ⏸️ **Month 1** (Jan 2026): Literature synthesis
- ⏸️ **Month 2** (Feb 2026): Implementation of SOTA methods
- ⏸️ **Month 3** (Mar 2026): Systematic ablations + hypothesis generation
- ⏸️ **Month 4** (Apr 2026): Novel architecture design
- ⏸️ **Month 5** (May 2026): Optimization + full validation
- ⏸️ **Month 6** (Jun 2026): Publication preparation

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

### Week 5-6: Implement Priority Methods
- [ ] GHIST (multi-task CNN with cell segmentation)
- [ ] THItoGene (improved Hist2ST)
- [ ] sCellST (subcellular UNet)
- [ ] CONCHv1.5 + Hist2ST (frozen encoder test)

### Week 7: Code Archaeology
- [ ] Compare paper claims vs code reality
- [ ] Extract undocumented training tricks
- [ ] Document in `docs/architecture_reviews/`

### Week 8: Initial Comparisons
- [ ] Run all implementations on P5 (held-out)
- [ ] Compare to baseline (Prov-GigaPath + Hist2ST, SSIM 0.5699)
- [ ] Update hypothesis backlog based on findings

---

## Month 3: Ablations + Hypotheses (Weeks 9-12)

### Week 9: Encoder Ablation @ 2μm
- [ ] Test Virchow2, Prov-GigaPath, CONCHv1.5, UNI2-h, H-optimus-1, GigaPath
- [ ] Fixed: Hist2ST decoder, MSE loss, frozen encoders
- [ ] Output: `results/month3_ablations/encoder_ablation_2um.csv`

### Week 10: Decoder Ablation @ 2μm
- [ ] Test Hist2ST, MiniUNet, THItoGene, sCellST, custom variants
- [ ] Fixed: Prov-GigaPath encoder, MSE loss
- [ ] Output: `results/month3_ablations/decoder_ablation_2um.csv`

### Week 11: Loss Function Ablation
- [ ] Test MSE, log-MSE, log1p-MSE, Huber, quantile losses
- [ ] Test with/without activations (None, ReLU, GELU)
- [ ] Output: `results/month3_ablations/loss_function_ablation.csv`

### Week 12: Hypothesis Prioritization
- [ ] Review all findings from Month 2-3
- [ ] Prioritize hypotheses (Tier 1: Hot, Tier 2: Warm, Tier 3: Cold)
- [ ] Select top 5-10 for Month 4 architecture design
- [ ] Output: Updated `docs/hypotheses/BACKLOG.md`

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
- [ ] Grid search: learning rate, decoder depth, graph k-neighbors
- [ ] Train on P1+P2, test on P5
- [ ] Budget: ~15 configurations
- [ ] Output: `configs/best_config.yaml`

### Week 19: Full Cross-Validation
- [ ] 3-fold LOOCV (all patient combinations)
- [ ] SSIM, PCC @ 2μm, per-gene, per-category metrics
- [ ] Statistical tests (paired t-test)
- [ ] Output: `results/month5/full_validation_results.csv`

### Week 20: Robustness Testing
- [ ] Per-patient analysis
- [ ] Per-gene-category analysis
- [ ] Failure mode analysis
- [ ] Component ablation study
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

---

## Pivot Log

*(Document any major plan changes here)*

### 2025-12-26: Initial Design
- No pivots yet - starting from design phase

---

## Key Decision Points

- **Month 1 → 2**: Proceed if at least 5 methods fully documented
- **Month 3 → 4**: Proceed if at least 3 promising hypotheses identified
- **Month 4 → 5**: Proceed if at least 1 component shows >2% SSIM gain
- **Month 5 → 6**: Proceed if novel method beats baseline (>0.5699 SSIM)

---

## Notes

- Update this roadmap weekly
- Check boxes as tasks complete
- Document pivots immediately
- Keep aligned with session summaries
