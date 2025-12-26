# SOTA 2μm H&E → Spatial Transcriptomics Prediction

**Building state-of-the-art system for predicting spatial gene expression from H&E histology at 2μm resolution**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Current Status

**Phase**: Month 0 - Design Complete ✅
**Next Milestone**: Month 1 - Literature Synthesis (Jan 2026)

**Current Best Performance**:
- Model: Prov-GigaPath (frozen) + Hist2ST decoder + MSE loss
- SSIM @ 2μm: **0.5699**
- Dataset: 3 CRC patients (P1, P2, P5), 50 genes, Visium HD

**Target**: SSIM > 0.60 at 2μm resolution

## Project Timeline

- **Month 1** (Jan 2026): Literature synthesis (27+ methods)
- **Month 2-3** (Feb-Mar 2026): Implementation + ablations + hypothesis generation
- **Month 4** (Apr 2026): Novel architecture design
- **Month 5** (May 2026): Optimization + validation
- **Month 6** (Jun 2026): Publication preparation

## Quick Start

```bash
# Clone repository
git clone https://github.com/vanbelkummax/sota-2um-st-prediction
cd sota-2um-st-prediction

# See full design document
cat docs/plans/2025-12-26-sota-2um-st-prediction-design.md

# Check current roadmap
cat ROADMAP.md
```

## Repository Structure

```
sota-2um-st-prediction/
├── README.md                    # This file
├── ROADMAP.md                   # Month-by-month timeline
├── docs/
│   ├── plans/                   # Design documents
│   ├── hypotheses/              # Research hypotheses
│   ├── architecture_reviews/    # Method analysis
│   └── synthesis/               # Literature synthesis
├── code/
│   ├── models/                  # Model implementations
│   ├── training/                # Training scripts
│   └── evaluation/              # Evaluation scripts
├── results/                     # Experimental results
├── figures/                     # Publication figures
└── papers/                      # Literature cache
```

## Key Findings (Updated Regularly)

1. ✅ **Frozen encoders > fine-tuned** (3-patient dataset too small)
2. ✅ **MSE works without softplus** (Poisson unnecessary at 2μm)
3. ✅ **Prov-GigaPath > Virchow2** at 2μm (+3.7% SSIM)
4. ✅ **Hist2ST decoder** outperforms simpler architectures
5. ⏸️ CONCH v1.5 untested at 2μm with neural decoder

## Resources

- **Design Document**: [docs/plans/2025-12-26-sota-2um-st-prediction-design.md](docs/plans/2025-12-26-sota-2um-st-prediction-design.md)
- **Quick Reference**: [/home/user/Desktop/QUICKREF_SOTA2UM.md](/home/user/Desktop/QUICKREF_SOTA2UM.md)
- **Access Requests**: [USER_ACCESS_NEEDED.md](USER_ACCESS_NEEDED.md)

## Contributing

This is a PhD research project by Max Van Belkum (Vanderbilt MD-PhD).
Collaborations welcome - contact: max.vanbelkum@vanderbilt.edu

## License

MIT License - See [LICENSE](LICENSE) file

## Citation

```bibtex
@misc{vanbelkum2026sota2um,
  title={SOTA 2μm Spatial Transcriptomics Prediction from H&E},
  author={Van Belkum, Max and Huo, Yuankai},
  year={2026},
  institution={Vanderbilt University}
}
```

---

**Last Updated**: 2025-12-26
**Status**: Design phase complete, ready for Month 1
