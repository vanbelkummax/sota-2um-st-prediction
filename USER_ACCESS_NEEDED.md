# User Access Request Queue

**Instructions for Max**:
- Check this file at start of each session
- Items marked 🔴 HIGH block current work
- Items marked 🟡 MEDIUM needed soon
- Items marked 🟢 LOW nice-to-have

**Last Updated**: 2025-12-26

---

## ✅ COMPLETED (User Provided)

### Papers
- [x] **GHIST (Nature Methods 2025)** (2025-12-26)
  - File: `/mnt/c/Users/User/Downloads/41592_2025_Article_2795.pdf`
  - Status: Acquired, ready for Month 1 synthesis

- [x] **bioRxiv preprint** (2025-12-26)
  - File: `/mnt/c/Users/User/Downloads/2024.11.07.622225v1.full.pdf`
  - Status: Acquired

- [x] **arXiv preprint** (2025-12-26)
  - File: `/mnt/c/Users/User/Downloads/2406.16192v2.pdf`
  - Status: Acquired

### Models
- [x] **UNI2-h pretrained weights** (HuggingFace gated) (2025-12-26)
  - Access: https://huggingface.co/MahmoodLab/UNI2-h
  - Status: Access granted, ViT-H/14, 1536-dim

- [x] **CONCH pretrained weights** (HuggingFace gated)
  - Access: https://huggingface.co/MahmoodLab/CONCH
  - Status: Access granted, ViT-B/16, 512-dim

---

## 🔴 HIGH Priority (Blocking Work)

*(None currently - all Month 1 Week 1 resources acquired)*

---

## 🟡 MEDIUM Priority (Needed Soon)

### Papers
- [ ] **TRIPLEX supplementary materials** (ICLR 2024)
  - **Why**: Architecture details sparse in main paper
  - **Access**: Check OpenReview for appendix
  - **Alternative tried**: Will search via polymax-synthesizer first
  - **Needed by**: Month 1 Week 2 (2026-01-15)

- [ ] **sCellST supplementary methods** (Nature Biotech 2024)
  - **Why**: Upsampling strategy details needed
  - **Access**: Supplement PDF from journal
  - **Alternative tried**: Will try Europe PMC first
  - **Needed by**: Month 1 Week 2

### Datasets
- [ ] **HEST-1K benchmark dataset**
  - **Why**: Validate methods on external benchmark
  - **Access**: Dataset request form at hest-benchmark.org
  - **Alternative**: Using our Visium HD only (acceptable for now)
  - **Needed by**: Month 5 Week 3 (for generalization testing)

---

## 🟢 LOW Priority (Nice to Have)

### Papers
- [ ] **THItoGene appendix**
  - **Why**: Additional ablation studies
  - **Access**: Check Bioinformatics journal supplement
  - **Alternative**: Have main paper, can proceed without
  - **Needed by**: Month 2 Week 2

### Models
- [ ] **GHIST pretrained weights** (if available)
  - **Why**: Weight archaeology + transfer learning tests
  - **Access**: Check GitHub releases
  - **Alternative**: Will train from scratch if needed
  - **Needed by**: Month 2 Week 3

---

## 📋 Acquisition Log

### Session 2025-12-26
- ✅ Acquired GHIST paper (Nature Methods 2025)
- ✅ Confirmed UNI2-h HuggingFace access
- ✅ Acquired 2 additional papers (bioRxiv, arXiv)
- Action: Copy PDFs to `papers/paywalled/` at Month 1 start

### Future Sessions
*(Track acquisition events here)*

---

## Notes

- **How to request**: User downloads → provides file path to Claude
- **Gated models**: User applies via HuggingFace form using institutional email
- **Priority escalation**: If HIGH priority item blocks >1 week → escalate to user
- **Alternatives**: Always try public APIs first (arXiv, Europe PMC, bioRxiv)
