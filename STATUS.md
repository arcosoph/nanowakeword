# Nanowakeword: Project Status & Release Notes

*Last Updated: November 17, 2025*

This document provides timely updates, performance notes, stability reports, and important recommendations regarding the Nanowakeword framework. We advise reviewing this file for the latest information on specific versions and features.

---

### Current Release: `v1.3.3`

#### Architecture Stability Report

*   **✅ Production-Ready:** `DNN`, `CNN`, `LSTM`, `GRU`, `RNN`,  `Transformer`, `TCN`. These architectures are stable and recommended for production use.

*   **⚠️ Experimental:** `QuartzNet`, `Conformer`, `E-Branchformer`, `CRNN`.
    *   **Recommendation:** These state-of-the-art architectures are provided for advanced users and researchers. In the current version, they have shown to be highly sensitive to hyperparameter configurations and may not converge to a useful solution with the default settings ("model collapse"). We are making it our highest priority to develop robust intelligent defaults for these models in a future release. **We strongly advise against using them for production purposes at this time.**

---
### Known Issues & Bug Fixes

*   This section will be updated if any critical bugs are discovered in the current PyPI release and when workarounds are available.

### Future Development Notes

*   Our immediate priority is improving the out-of-the-box stability for all experimental architectures.
