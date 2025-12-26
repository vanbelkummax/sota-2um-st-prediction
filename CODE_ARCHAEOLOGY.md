# Code Archaeology Log

**Purpose**: Track inspected repositories, key findings, and reproduction feasibility.

**Last Updated**: 2025-12-26

---

## Overview

| Method | Repo Status | Inspected | Reproduction | Priority |
|--------|-------------|-----------|--------------|----------|
| GHIST | TBD | ⏸️ Pending | TBD | HIGH |
| TRIPLEX | TBD | ⏸️ Pending | TBD | MEDIUM |
| THItoGene | TBD | ⏸️ Pending | TBD | HIGH |
| sCellST | TBD | ⏸️ Pending | TBD | MEDIUM |

---

## Inspection Template

### METHOD_NAME
- **Repo**: [GitHub URL]
- **Cloned**: YYYY-MM-DD
- **Status**: ⏸️ Pending / 🔍 Inspecting / ✅ Complete
- **Key Findings**:
  - Paper claim vs code reality
  - Undocumented tricks (augmentation, regularization, LR schedule)
  - Hyperparameter discrepancies
- **Pretrained Weights**: Available / Not available / Location
- **Reproduction**: Easy / Medium / Hard / Impossible
- **Time Estimate**: X days to implement + train
- **Code Review**: [Link to docs/architecture_reviews/METHOD_code_archaeology.md]

---

## GHIST
- **Repo**: https://github.com/sotiraslab/GHIST (assumed, verify)
- **Cloned**: Not yet
- **Status**: ⏸️ Pending
- **Plan**: Clone during Month 1 Week 2
- **Paper**: Acquired ✅ `/mnt/c/Users/User/Downloads/41592_2025_Article_2795.pdf`
- **Priority**: HIGH (claims 2μm SOTA)

## TRIPLEX
- **Repo**: https://github.com/NEXGEM/TRIPLEX (assumed, verify)
- **Cloned**: Not yet
- **Status**: ⏸️ Pending
- **Plan**: Search during Month 1 Week 2
- **Priority**: MEDIUM (multimodal fusion interesting)

## THItoGene
- **Repo**: https://github.com/liyichen1998/THItoGene (assumed, verify)
- **Cloned**: Not yet
- **Status**: ⏸️ Pending
- **Plan**: Clone during Month 2 Week 1
- **Priority**: HIGH (claims "improved" Hist2ST)

## sCellST
- **Repo**: Unknown (search needed)
- **Cloned**: Not yet
- **Status**: ⏸️ Pending
- **Plan**: Find repo during Month 1 Week 2
- **Priority**: MEDIUM (subcellular upsampling novel)

---

## Inspection Workflow

1. **Clone repository**
   ```bash
   cd /home/user/work/code_archaeology/
   git clone [REPO_URL]
   cd [METHOD_NAME]
   ```

2. **Map structure**
   ```bash
   tree -L 2 -I '__pycache__|*.pyc'
   find . -name "*train*.py" -o -name "*model*.py"
   ```

3. **Extract architecture**
   ```bash
   grep -r "class.*Model\|class.*Net" --include="*.py"
   # Use Read tool + LSP for detailed inspection
   ```

4. **Compare paper vs code**
   - Check claimed architecture vs actual implementation
   - Document discrepancies

5. **Extract training tricks**
   ```bash
   grep -r "augment\|flip\|rotate" --include="*.py"
   grep -r "dropout\|weight_decay" --include="*.py"
   grep -r "scheduler\|warmup\|cosine" --include="*.py"
   ```

6. **Check reproducibility**
   ```bash
   find . -name "*.pth" -o -name "*.pt" -o -name "*.ckpt"
   gh issue list --repo [OWNER]/[REPO] --search "reproduce"
   ```

7. **Document findings**
   - Write `docs/architecture_reviews/[METHOD]_code_archaeology.md`
   - Update this log

---

## Notes

- Priority based on: (1) Claims at 2μm, (2) Code availability, (3) Novelty
- "Easy" reproduction = working code + clear docs + pretrained weights
- "Medium" = working code but missing pieces (no weights, unclear docs)
- "Hard" = buggy code or major dependencies missing
- "Impossible" = no public code available
