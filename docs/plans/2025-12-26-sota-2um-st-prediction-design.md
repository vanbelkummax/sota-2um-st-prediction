# SOTA 2μm H&E → Spatial Transcriptomics Prediction System
**PhD-Scale Research Design (4-6 Month Timeline)**

**Date**: 2025-12-26
**Principal Investigator**: Max Van Belkum (MD-PhD, Vanderbilt)
**Collaborators**: Yuankai Huo Lab (Computational Pathology)
**Status**: Design Phase Complete → Ready for Implementation

---

## Executive Summary

**Goal**: Design and build a state-of-the-art system for predicting spatial gene expression from H&E histology images at 2μm resolution (Visium HD), surpassing current methods through systematic literature synthesis, critical hypothesis generation, and novel architectural innovations.

**Current Baseline**:
- Best model: Prov-GigaPath (frozen) + Hist2ST decoder + MSE loss
- Performance: SSIM 0.5699 at 2μm resolution
- Dataset: 3 CRC patients (P1, P2, P5), 50 genes, Visium HD

**Target Outcome**:
- Novel architecture achieving >0.60 SSIM at 2μm
- Two publications: (1) Methods paper, (2) Benchmark paper
- Open-source code, pretrained weights, benchmark dataset
- Complete understanding of what drives 2μm performance

**Timeline**: 6 months (January - June 2026)
- Month 1: Literature synthesis (27+ methods)
- Month 2-3: Implementation + systematic ablations + hypothesis generation
- Month 4: Novel architecture design
- Month 5: Optimization + validation
- Month 6: Publication preparation

---

## Table of Contents

1. [Knowledge Persistence Infrastructure](#1-knowledge-persistence-infrastructure)
2. [Session Continuity Protocols](#2-session-continuity-protocols)
3. [Month 1: Literature Synthesis](#3-month-1-literature-synthesis)
4. [Month 2-3: Implementation + Hypothesis Generation](#4-month-2-3-implementation--hypothesis-generation)
5. [Adaptive Planning Strategy](#5-adaptive-planning-strategy)
6. [Code & Weight Archaeology](#6-code--weight-archaeology)
7. [Resource Acquisition Pipeline](#7-resource-acquisition-pipeline)
8. [Month 4-6: Novel Architecture + Publications](#8-month-4-6-novel-architecture--publications)
9. [Success Metrics](#9-success-metrics)
10. [Risk Mitigation](#10-risk-mitigation)

---

## 1. Knowledge Persistence Infrastructure

**Challenge**: 6-month project spanning 100+ Claude sessions requires persistent knowledge that accumulates across conversations.

### Core Knowledge Systems

**1.1 PolyMaX Synthesizer MCP**
- Location: `/home/user/mcp_servers/polymax-synthesizer/`
- Purpose: Hierarchical paper synthesis with persistent SQLite database
- Usage: Extract architectural details from GHIST, TRIPLEX, sCellST, etc.
- Output: `papers.db` with structured paper annotations

**1.2 Memory MCP (Knowledge Graph)**
- Purpose: Store high-level insights as entities + relations
- Entity types: Encoder, Decoder, Method, Finding, Hypothesis
- Relation types: outperforms, implements, supports, invalidates
- **Discipline**: Keep graph <100 entities (only critical insights)

**1.3 Vanderbilt Professors MCP**
- Purpose: Search Huo, Lau, Landman papers for spatial transcriptomics methods
- Usage: Cross-reference findings with local Vanderbilt expertise

**1.4 Filesystem Organization**
```
/home/user/sota-2um-st-prediction/          # Main repo (git tracked)
/home/user/work/sota-2um-st-prediction/     # Results (persistent, ext4)
/home/user/Desktop/SESSION_*.md             # Session summaries (quick ref)
/home/user/docs/plans/                      # Design docs (this file)
```

### Cross-Session Workflow
1. **Session START**: Read plan + latest summary + memory graph
2. **During**: Update memory with discoveries, log to session summary
3. **Session END**: Write summary, commit to GitHub, update memory graph

---

## 2. Session Continuity Protocols

### Every Session START (Mandatory First 3 Commands)

```bash
# 1. Read master plan
cat /home/user/docs/plans/2025-12-26-sota-2um-st-prediction-design.md

# 2. Read latest session summary
cat $(ls -t /home/user/Desktop/SESSION_*.md | head -1)

# 3. Check current phase
cat /home/user/sota-2um-st-prediction/ROADMAP.md
```

Then: Review memory graph, sync with GitHub, identify active experiments.

### Every Session END (Mandatory Before Exit)

**1. Update Memory Graph**
```
Use mcp__memory__add_observations for discoveries:
- "Prov-GigaPath outperforms Virchow2 by 3.7% at 2μm"
- "GHIST uses multi-task supervision with cell segmentation"
- "Frozen encoders > fine-tuned on 3-patient dataset"
```

**2. Push Results to GitHub**
```bash
cd /home/user/sota-2um-st-prediction
git add results/ablations/encoder_comparison_2um.csv
git commit -m "Month 2: CONCHv1.5 encoder results (SSIM 0.571)

- Tested CONCHv1.5 (ViT-L, frozen) vs Prov-GigaPath
- SSIM: 0.571 vs 0.5699 (marginal improvement)
- Per-gene analysis shows CONCHv1.5 better for immune genes

🤖 Generated with Claude Code"
git push origin main
```

**3. Write Session Summary**
```bash
# Template: SESSION_YYYY-MM-DD_MonthX_WeekY.md
# Include: What tested, key findings, updated hypotheses, next goals
```

**4. Update Quick Reference**
```bash
# Update /home/user/Desktop/QUICKREF_SOTA2UM.md with:
# - Current best performance
# - Current phase (Month X, Week Y)
# - Active hypotheses
# - Next session goals
```

---

## 3. Month 1: Literature Synthesis

**Goal**: Build complete knowledge base of all H&E → ST methods (2020-2025), with deep architectural extraction from 2024-2025 SOTA methods.

### Week 1: Broad Discovery

**Tasks**:
- Use `polymax-synthesizer` to search 27+ methods from MASTER_REVIEW
- Search `vanderbilt-professors` MCP for Huo lab ST papers
- Use `mcp__github__search_code` to find implementation repos
- Acquire papers via arXiv API, Europe PMC, bioRxiv
- Add paywalled papers to USER_ACCESS_NEEDED.md

**Output**: Complete method inventory with GitHub links, PDFs, claimed performance

**Methods to Inventory**:
- Tier 1 (Essential): ST-Net, HisToGene, Hist2ST, BLEEP, HGGEP
- Tier 2 (SOTA 2024-25): GHIST, THItoGene, TRIPLEX, PixNet, sCellST
- Tier 3 (Additional): STMCL, MISO, SpaDiT, DANet, ResSAT, HistoSPACE

### Week 2: Deep Extraction (Priority Methods)

**Focus on 2024-2025 methods claiming 2μm SOTA**:

**GHIST** (Nature Methods 2025) - ACQUIRED ✅
- Paper: `/mnt/c/Users/User/Downloads/41592_2025_Article_2795.pdf`
- Extract: Multi-task architecture, cell segmentation auxiliary task, claimed metrics
- Questions: What resolution tested? What dataset? Reproducible?

**TRIPLEX** (2024)
- Extract: CLIP+GNN+Transformer fusion architecture
- Questions: Does vision-language help at 2μm?

**THItoGene** (2024)
- Extract: Differences from Hist2ST (attention mechanism, graph construction)
- Questions: What specifically "improved"?

**sCellST** (Nature Biotech 2024)
- Extract: Subcellular UNet upsampling strategy
- Questions: Does their upsampling preserve spatial coherence?

**Method**: Use parallel agents (Task tool) to extract from each paper simultaneously:
1. Architecture diagram
2. Loss functions
3. Training strategies (learning rate, augmentation, regularization)
4. Reported metrics (resolution, dataset, performance)
5. Code availability

**Output**: Detailed architectural specs in polymax database + memory graph

### Week 3-4: Reverse Engineering + Pattern Analysis

**Tasks**:
1. For methods without code: Reconstruct architecture from paper descriptions
2. Identify common patterns across SOTA methods:
   - Multi-scale features? (Yes/No counts)
   - Transformer attention? (Which type?)
   - Graph networks? (Spatial vs feature-based?)
   - Loss functions? (MSE, Poisson, custom?)
3. Statistical analysis: Plot performance vs architectural complexity
4. Cross-disciplinary literature mining:
   - Super-resolution imaging (CARE, CSBDeep)
   - Weather forecasting (GraphCast multi-scale)
   - Protein structure (AlphaFold MSA+structure fusion)
   - Neural rendering (NeRF continuous fields)

**Output**: `/home/user/sota-2um-st-prediction/docs/synthesis/`
- `literature_synthesis_summary.json` (structured data)
- `architectural_patterns.md` (critical analysis)
- `critical_ingredients_2um.md` (hypothesis: what's needed for 2μm SOTA?)

---

## 4. Month 2-3: Implementation + Hypothesis Generation

**Goal**: Implement SOTA methods, run systematic ablations, **actively question field assumptions** to generate novel hypotheses.

### Month 2: Implementation + Critical Analysis (Weeks 5-8)

**Tier 1: Must Implement**

**1. GHIST** (if GPU memory allows)
- Use gradient checkpointing + FP16 for 24GB RTX 5090
- **Question**: Does multi-task learning actually help at 2μm?
- **Critical analysis**: Is multi-task overfitting to their dataset?

**2. THItoGene** (Improved Hist2ST)
- **Question**: What specifically improved?
- **Method**: Strip to minimal changes from Hist2ST, isolate contributions
- **Expected**: Likely just better hyperparameters, not architecture

**3. sCellST** (Subcellular UNet)
- **Question**: Does upsampling strategy beat Hist2ST multi-pathway fusion?
- **Test hypothesis**: Simple is better for sparse 2μm data

**4. CONCHv1.5 (ViT-L, frozen)** - ACQUIRED ✅
- HuggingFace access: MahmoodLab/UNI2-h (1536-dim)
- **Test**: CONCHv1.5 frozen + Hist2ST vs Prov-GigaPath
- **Hypothesis**: Vision-language pretraining gives richer features

**Critical Analysis Activity**: For each implementation, document:
- What assumptions does this method make?
- Do those assumptions hold at 2μm CRC?
- What if we invert the assumption?

**Output**: `/home/user/sota-2um-st-prediction/docs/architecture_reviews/METHOD_analysis.md`

### Month 3: Systematic Ablations (Weeks 9-12)

**Ablation Matrix** (all with frozen encoders):

| Factor | Options | Current Best | Novel Hypothesis |
|--------|---------|--------------|------------------|
| **Encoder** | Virchow2, Prov-GigaPath, CONCHv1.5, UNI2-h, H-optimus-1, GigaPath | Prov-GigaPath (0.5699) | **Ensemble of frozen encoders** |
| **Decoder** | Hist2ST, MiniUNet, THItoGene, sCellST, Custom | Hist2ST | **Gene-category-specific decoders** |
| **Loss** | MSE, log-MSE, log1p-MSE, Huber, Quantile | MSE (no softplus) | **Per-gene adaptive loss** |
| **Activation** | None, ReLU, GELU, Sigmoid, Tanh | None (linear) | **Learned activation per gene** |
| **Multi-scale** | 2μm only, 2μm+8μm joint, 256px+512px features | Single scale | **Hierarchical feature fusion** |

**Key Findings to Carry Forward**:
- ✅ Frozen encoders > fine-tuned (3-patient dataset too small)
- ✅ MSE works fine without softplus (Poisson unnecessary)
- ✅ Prov-GigaPath > Virchow2 at 2μm (+3.7%)

### Field Assumptions to Question

**1. "One decoder for all genes"**
- Current: Single model predicts all 18K genes
- Question: Do epithelial markers need different architecture than immune genes?
- **Hypothesis H001**: Gene-category-specific decoders (epithelial, immune, stromal)
- Cross-discipline: NLP uses task-specific heads on shared backbone

**2. "Spatial neighbors are most important"**
- Current: GNNs use k-nearest spatial neighbors
- Question: Are histologically-similar patches more informative?
- **Hypothesis H002**: Feature similarity graph > spatial graph
- Cross-discipline: Computer vision retrieval uses feature similarity

**3. "Predict all genes simultaneously"**
- Current: Parallel prediction of all genes
- Question: Do correlated genes benefit from sequential prediction?
- **Hypothesis H003**: Autoregressive gene prediction (easy → hard)
- Cross-discipline: GPT (language), AlphaFold (structure)

**4. "2μm is information-limited"**
- Current: Finer resolution = harder
- Question: Is 2μm actually richer (cellular detail)?
- **Hypothesis H004**: Current architectures don't exploit 2μm signal
- Cross-discipline: Super-resolution uses cellular priors

**5. "H&E alone is sufficient"**
- Current: Only morphology
- Question: What if we add tissue structure priors?
- **Hypothesis H005**: Nuclei/gland segmentation as auxiliary input
- Cross-discipline: Object detection + regression

**6. "Transformers need position embeddings"**
- Current: Absolute/relative position
- Question: Do we need position, or just tissue context?
- **Hypothesis H006**: Rotation-invariant features
- Cross-discipline: Equivariant neural networks

### Hypothesis Tracking (Three-Tier System)

**Tier 1: Hot Ideas** (memory graph, max 20)
- Immediately testable, clear implementation path
- Contradict current assumptions
- Example: "Use feature similarity for graph edges"

**Tier 2: Warm Ideas** (`docs/hypotheses/BACKLOG.md`, unlimited)
- One-line descriptions from literature review
- Example:
  ```markdown
  ## Warm Hypotheses
  - [ ] H_arch_001: Rotation-equivariant convolutions
  - [ ] H_arch_002: Nuclei segmentation auxiliary task
  - [ ] H_train_001: Curriculum learning (easy → hard genes)
  ```

**Tier 3: Cold Ideas** (paper notes in PolyMaX database)
- Raw notes, searchable but not actively tracked

**Promotion Rules**:
- Tier 3 → 2: Insight becomes concrete hypothesis
- Tier 2 → 1: Ready to implement (move to memory)
- Tier 1 → Archive: Tested (move result to memory, remove hypothesis)

### Cross-Disciplinary Literature Mining (Week 11-12)

**Domains to explore** (use polymax + WebSearch):
1. **Super-resolution microscopy**: CARE, CSBDeep learned upsampling
2. **Weather forecasting**: GraphCast multi-scale graph networks
3. **Protein structure**: AlphaFold MSA+structure fusion
4. **Neural rendering**: NeRF continuous field prediction
5. **Time-series**: Transformers for irregular sampling

**Output**: `docs/hypotheses/novel_architecture_ideas.md` (10-15 ranked ideas)

---

## 5. Adaptive Planning Strategy

**Philosophy**: The plan is a hypothesis, not a law. Update when evidence demands.

### When to Pivot

✅ **Pivot if**:
- New SOTA discovered (>0.60 SSIM at 2μm)
- Dead end confirmed (week of work, fundamentally flawed)
- Unexpected breakthrough (novel hypothesis works spectacularly)
- Resource constraints (GHIST needs 48GB, can't run)

❌ **Don't pivot after**:
- Single failed experiment
- One bad paper
- Minor setbacks

### Pivot Workflow

1. **Document**: Write `docs/pivots/PIVOT_YYYY-MM-DD.md`
2. **Update ROADMAP.md**: Strikethrough old plan, add new priority
3. **Update memory**: Add entity + relation explaining pivot
4. **Session summary**: Note pivot for future sessions

**Example Pivot**:
```markdown
# Pivot: Dropping Poisson Loss Focus

## Original Plan
Month 3 Week 2: Deep dive Poisson vs NB losses

## Why Pivoting
MSE without softplus works perfectly (SSIM 0.5699)
No evidence Poisson adds value at 2μm

## New Plan
Month 3 Week 2: Decoder architecture ablations instead

## Impact
Saves 1 week, reallocates to more promising direction
```

---

## 6. Code & Weight Archaeology

**Philosophy**: Code reveals implementation truth. Weights reveal learned patterns.

### Code Archaeology Protocol (Month 1-2)

**For each method**:

**1. Clone & Explore**
```bash
cd /home/user/work/code_archaeology/
git clone https://github.com/sotiraslab/GHIST
cd GHIST

# Map repo
tree -L 2 -I '__pycache__|*.pyc'

# Find key files
find . -name "*train*.py" -o -name "*model*.py"
```

**2. Extract Architecture**
```bash
# Find model class
grep -r "class.*Model\|class.*Net" --include="*.py"

# Read forward pass (use LSP tool)
# Extract hyperparameters
grep -r "learning_rate\|epochs\|batch_size" --include="*.py" --include="*.yaml"
```

**3. Compare Paper vs Code**
- Paper claims: "8-layer transformer"
- Code reality: Check `config.yaml:num_layers`
- Document discrepancies in `docs/architecture_reviews/METHOD_code_review.md`

**4. Extract Training Tricks** (undocumented)
```bash
# Data augmentation
grep -r "augment\|flip\|rotate" --include="*.py"

# Regularization
grep -r "dropout\|weight_decay" --include="*.py"

# Learning rate schedules
grep -r "scheduler\|warmup\|cosine" --include="*.py"
```

**5. Check Reproducibility**
```bash
# Pretrained weights available?
find . -name "*.pth" -o -name "*.pt"

# GitHub issues for reproduction problems
gh issue list --repo sotiraslab/GHIST --search "reproduce"
```

**Output**: `docs/architecture_reviews/METHOD_code_archaeology.md`

### Weight Archaeology Protocol (Month 2-3)

**When pretrained weights available**:

**1. Download & Inspect**
```python
import torch
checkpoint = torch.load('ghist_pretrained.pth', map_location='cpu')

print("Keys:", checkpoint.keys())
if 'model_state_dict' in checkpoint:
    for k, v in checkpoint['model_state_dict'].items():
        print(f"{k}: {v.shape}")
```

**2. Extract Learned Patterns**

a) **Attention patterns** (if Transformer)
```python
model = TheirModel()
model.load_state_dict(checkpoint['model_state_dict'])
attention_layer = model.decoder.transformer.layers[0].self_attn
# Visualize: Which positions attend to each other?
```

b) **Output bias patterns**
```python
final_bias = model.output_head.bias.cpu().numpy()
plt.hist(final_bias, bins=50)
# Large negative bias = gene predicted low by default
```

c) **Compare to baseline**
```python
their_attn_norm = attention_layer.in_proj_weight.norm().item()
# If norms very different → different learning dynamics
```

**3. Transfer Learning Test**
```python
# Load their pretrained decoder
their_model = load_pretrained_ghist()

# Replace output layer for your 50 genes
their_model.output_head = nn.Linear(512, 50)

# Fine-tune on your 3 CRC patients
# If transfers well → architecture captures generalizable patterns
# If not → overfit to their dataset
```

**Output**: `docs/weight_archaeology/METHOD_weights.md`

### Code Archaeology Database

Track in `CODE_ARCHAEOLOGY.md`:

```markdown
## GHIST
- **Repo**: https://github.com/sotiraslab/GHIST
- **Cloned**: 2025-01-10
- **Status**: ✅ Inspected
- **Key Findings**:
  - Claims "8-layer transformer" → Actually 4 layers
  - Undocumented: StochasticDepth p=0.2
  - No pretrained weights → Must retrain
- **Reproduction**: Medium (2-3 days)

## THItoGene
- **Repo**: https://github.com/liyichen1998/THItoGene
- **Status**: ✅ Inspected
- **Key Findings**:
  - "Improved attention" = standard MultiheadAttention
  - Main diff: Learning rate schedule (cosine vs step)
- **Reproduction**: Easy (1 day)
```

---

## 7. Resource Acquisition Pipeline

### User Access Request Queue

**File**: `/home/user/sota-2um-st-prediction/USER_ACCESS_NEEDED.md`

Priority levels:
- 🔴 HIGH: Blocks current work
- 🟡 MEDIUM: Needed soon
- 🟢 LOW: Nice to have

**Resources Acquired (Session 2025-12-26)**:
- ✅ GHIST paper: `/mnt/c/Users/User/Downloads/41592_2025_Article_2795.pdf`
- ✅ UNI2-h weights: HuggingFace MahmoodLab/UNI2-h
- ✅ bioRxiv paper: `/mnt/c/Users/User/Downloads/2024.11.07.622225v1.full.pdf`
- ✅ arXiv paper: `/mnt/c/Users/User/Downloads/2406.16192v2.pdf`

### Paper Acquisition Priority

**Tier 1: Direct API Access** (fast, unlimited)
1. **arXiv** via arXiv API + `export.arxiv.org/pdf/`
   ```bash
   # 3 second rate limit
   curl -A "sota-2um-agent/0.1 (max@vanderbilt.edu)" \
     "https://export.arxiv.org/pdf/2312.12345.pdf" -o paper.pdf
   sleep 3
   ```

2. **Europe PMC** via MCP (configured)

3. **bioRxiv/medRxiv** via API

**Tier 2: HuggingFace ML Papers**
```bash
curl "https://huggingface.co/api/papers/search?q=spatial+transcriptomics"
# Extract arxiv ID, use Tier 1
```

**Tier 3: WebSearch + Publisher Sites** (rate limited)

**Tier 4: User Manual Access** (paywalled, gated models)

### Local Paper Cache

```
papers/
├── arxiv/
│   ├── 2312.12345_GHIST.pdf
│   └── metadata.json
├── biorxiv/
├── europe_pmc/
└── paywalled/  # User-provided
    └── 41592_2025_Article_2795.pdf  # GHIST
```

### Model Weight Repository

```
/home/user/work/model_weights/
├── ghist/
│   ├── pretrained.pth
│   └── INSPECTION_NOTES.md
├── conch_v15/
│   ├── pytorch_model.bin
│   └── INSPECTION_NOTES.md
└── INVENTORY.md
```

---

## 8. Month 4-6: Novel Architecture + Publications

### Month 4: Architecture Design (Weeks 13-16)

**Week 13: Design Workshop**

Take top 5-10 hypotheses from Month 3, design **modular architecture**:

```python
class NovelST2umPredictor:
    def __init__(self, config):
        # Component 1: Multi-encoder fusion (H001)
        if config.multi_encoder:
            self.encoder = EnsembleEncoder(
                encoders=['prov-gigapath', 'conch_v15', 'uni2-h'],
                fusion='learned_attention'
            )

        # Component 2: Graph construction (H002)
        if config.graph_type == 'feature_similarity':
            self.graph = FeatureSimilarityGraph(k=8)
        elif config.graph_type == 'spatial':
            self.graph = SpatialKNNGraph(k=8)

        # Component 3: Gene-specific decoders (H003)
        if config.gene_specific:
            self.decoder = CategorySpecificDecoder({
                'epithelial': Hist2STDecoder(...),
                'immune': Hist2STDecoder(...),
                'stromal': Hist2STDecoder(...),
            })

        # Component 4: Adaptive loss (H004)
        if config.adaptive_loss:
            self.loss = PerGeneLoss(base='mse')

        # Component 5: Multi-scale (H005)
        if config.multiscale:
            self.fuser = MultiScaleFuser(scales=[256, 512])
```

**Output**: `docs/architecture/novel_system_design.md`

**Week 14-15: Implementation**

Build at `code/models/novel_predictor.py` with:
- Modular components (easy to ablate)
- Clean interfaces
- Unit tests

**Week 16: Initial Validation**

Quick test on P5 (held-out):
- Baseline: Prov-GigaPath + Hist2ST (0.5699)
- Test each component individually
- If any >2% gain → proceed to Month 5
- If no gains → pivot to next hypotheses

### Month 5: Optimization + Validation (Weeks 17-20)

**Week 17-18: Hyperparameter Optimization**

Grid search on P1+P2, test P5:
- Learning rate (1e-4, 5e-5, 1e-3)
- Decoder depth (1, 2, 4 layers)
- Graph k-neighbors (4, 8, 16)
- Multi-scale weights

Budget: ~10-15 configs × 2 hours = 1.5 days GPU time

**Week 19: Full Cross-Validation**

3-fold LOOCV with best config:
- Fold 1: Train P1+P2, test P5
- Fold 2: Train P1+P5, test P2
- Fold 3: Train P2+P5, test P1

**Metrics**:
- SSIM @ 2μm (primary)
- PCC @ 2μm (secondary)
- Per-gene, per-category analysis

**Comparisons**:
- Your previous best (0.5699)
- Month 2 implementations (GHIST, THItoGene, sCellST)
- Literature claims

**Week 20: Robustness Testing**

- Per-patient analysis
- Per-gene-category analysis
- Failure mode analysis (which genes fail?)
- Component ablation study

**Output**: `results/month5/full_validation_results.csv`

### Month 6: Publications (Weeks 21-24)

**Track 1: Methods Paper** ("Novel Architecture for 2μm ST Prediction")
- Target: Nature Methods, Nature Machine Intelligence, MICCAI 2026

**Week 21: Generate Figures**
- Figure 1: Architecture diagram
- Figure 2: Main results (3-fold CV with stats)
- Figure 3: Ablation study
- Figure 4: Biological validation (WSI visualizations)
- Figure 5: Generalization analysis

**Use LaTeX Architect MCP**:
```python
mcp__latex-architect__generate_figure_block(
    filename='figures/fig1_architecture.pdf',
    caption='Proposed modular architecture...',
    label='fig:architecture',
    wide=True,
    placement='t!'  # Huo lab standard
)
```

**Week 22: Write Manuscript**
- Abstract, Introduction, Methods, Results, Discussion, Conclusion
- Use `manuscripts/methods_paper_2um_st/main.tex`

**Track 2: Benchmark Paper** ("Comprehensive Comparison at 2μm")
- Target: Nature Communications, Scientific Data

**Week 23: Compile Benchmark**
- All Month 2-3 implementations
- Complete results table (8-10 methods head-to-head)
- Encoder/decoder/loss comparison tables
- Benchmark dataset release package

**Week 24: Code Release**

GitHub repo finalization:
```
sota-2um-st-prediction/
├── README.md (installation, quickstart, results)
├── docs/ (INSTALLATION.md, USAGE.md, BENCHMARK.md)
├── code/ (models, training, evaluation)
├── pretrained_weights/ (best model)
├── results/ (benchmark_results.csv)
└── figures/ (publication quality)
```

**Submissions**:
- Methods paper → Nature Methods
- Benchmark paper → Nature Communications
- Code → GitHub (public)
- Weights → HuggingFace
- Dataset → Zenodo

---

## 9. Success Metrics

### Technical Metrics

**Primary**:
- SSIM @ 2μm > 0.60 (beat current 0.5699)
- Statistical significance (p < 0.05 vs baseline, paired t-test)
- Generalizes across all 3 patients

**Secondary**:
- PCC @ 2μm > 0.40
- Per-category improvements (epithelial, immune, stromal)
- Genes with r > 0.8 count increases

### Research Metrics

**Knowledge Accumulation**:
- ✅ 27+ methods catalogued and analyzed
- ✅ 5-7 methods implemented and benchmarked
- ✅ 10-15 novel hypotheses generated and tested
- ✅ Complete code archaeology for top methods

**Publication Metrics**:
- ✅ 2 manuscripts submitted (methods + benchmark)
- ✅ Open-source code released (>100 GitHub stars target)
- ✅ Benchmark dataset used by other groups
- ✅ Pretrained weights downloaded >1000 times

### Process Metrics

**Session Continuity**:
- Memory graph maintained <100 entities
- 100% of sessions have session summary
- 100% of key results pushed to GitHub
- No lost context between sessions

**Adaptive Planning**:
- <5 pivots over 6 months (stable plan)
- Pivots documented within 24 hours
- Roadmap always reflects current state

---

## 10. Risk Mitigation

### Technical Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| GHIST too memory-intensive | Medium | Use gradient checkpointing, FP16, or skip |
| Poor generalization | Low | LOOCV ensures unbiased evaluation |
| Encoder access issues | Medium | Apply early, have backup (ImageNet) |
| Novel architecture fails | Medium | Have fallback (ensemble of best existing) |

### Timeline Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Implementation harder than expected | High | Start with simple methods first |
| Encoder download slow | Medium | Cache features locally |
| Code bugs | High | Budget extra debugging time |
| Hypothesis generation slow | Medium | Parallel literature mining |

### Knowledge Continuity Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory graph grows too large | High | Enforce <100 entity limit, archive old |
| Session summaries not written | Critical | Make it mandatory end protocol |
| GitHub not updated | High | Daily commit habit |
| Context lost between sessions | Critical | Session START protocol mandatory |

---

## Appendix A: Quick Reference Commands

### Session Start
```bash
# 1. Load plan
cat /home/user/docs/plans/2025-12-26-sota-2um-st-prediction-design.md

# 2. Load latest summary
cat $(ls -t /home/user/Desktop/SESSION_*.md | head -1)

# 3. Check phase
cat /home/user/sota-2um-st-prediction/ROADMAP.md

# 4. Sync GitHub
cd /home/user/sota-2um-st-prediction && git pull && git log -5 --oneline

# 5. Check memory graph
# Use mcp__memory__read_graph
```

### Session End
```bash
# 1. Update memory graph (use MCP tools)
# 2. Push to GitHub
cd /home/user/sota-2um-st-prediction
git add -A
git commit -m "Month X Week Y: <description>"
git push origin main

# 3. Write session summary
# Create /home/user/Desktop/SESSION_YYYY-MM-DD_MonthX_WeekY.md

# 4. Update quick reference
# Edit /home/user/Desktop/QUICKREF_SOTA2UM.md
```

---

## Appendix B: Resource Inventory (Session 2025-12-26)

**Papers Acquired**:
1. GHIST (Nature Methods 2025): `41592_2025_Article_2795.pdf`
2. bioRxiv preprint: `2024.11.07.622225v1.full.pdf`
3. arXiv preprint: `2406.16192v2.pdf`

**Model Access Granted**:
1. UNI2-h (HuggingFace): MahmoodLab/UNI2-h, ViT-H/14, 1536-dim
2. CONCH (HuggingFace): MahmoodLab/CONCH, ViT-B/16, 512-dim
3. CONCHv1.5 (HuggingFace): MahmoodLab/CONCHv1_5, ViT-L, mentioned in model card

**Encoder Infrastructure**:
- ✅ Virchow2 (tested 8μm)
- ✅ UNI2-h (tested 8μm)
- ✅ H-optimus-1 (tested 8μm)
- ✅ Prov-GigaPath (tested 2μm, current best)
- ✅ CONCH (tested 8μm Ridge only)
- ⏸️ CONCHv1.5 (not tested yet, priority for Month 2)
- ✅ GigaPath (tested 8μm)
- ✅ Phikon (tested 8μm)
- ✅ DenseNet-ImageNet (baseline)

**Decoder Infrastructure**:
- ✅ Hist2ST (current best at 2μm)
- ✅ MiniUNet (baseline)
- ⏸️ THItoGene (plan to implement Month 2)
- ⏸️ sCellST (plan to implement Month 2)
- ⏸️ GHIST decoder (plan to implement Month 2)

---

## Document History

- **2025-12-26**: Initial design complete (brainstorming session)
- **Status**: Ready for Month 1 implementation
- **Next Review**: After Month 1 synthesis complete

---

**END OF DESIGN DOCUMENT**
