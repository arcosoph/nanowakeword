# Nanowakeword: Project Status & Release Notes

This document provides timely updates, performance notes, stability reports, and important recommendations regarding the Nanowakeword framework. We advise reviewing this file for the latest information on specific versions and features.

---

# Current Release: `v2.0.1`


## ⚠️ Known Issue: `ModuleNotFoundError: No module named 'pkg_resources'`

If you encounter the following error while running training:

```bash
ModuleNotFoundError: No module named 'pkg_resources'
```

### 🧩 Cause

This issue occurs with newer versions of `setuptools` (70+), where `pkg_resources` is no longer included by default.  
Some dependencies (such as `pronouncing`) still rely on `pkg_resources`, which causes the training process to fail.

### ✅ Temporary Fix

Downgrade `setuptools` to a compatible version:

```bash
pip install setuptools==69.5.1
```

Then verify:

```bash
python -c "import pkg_resources; print('OK')"
```

If it prints `OK`, the issue is resolved.

---

🚧 This is a temporary workaround.  
A proper fix will be provided as soon as possible once dependency compatibility is addressed upstream.


# Release: `v2.0.0`
*   **✅ Production-Ready:** `DNN`, `CNN`, `LSTM`, `GRU`, `RNN`, `QuartzNet`, `Transformer`, `TCN`. These architectures are stable and recommended for production use.

*    **Untested:** `Conformer`, `E-Branchformer`, `CRNN` are not tested

### Future Development Notes

*   Currently, work is being done to increase the accuracy of the model.
*   It may be possible to create an E2E model very soon.


### Release: `v1.3.3`

#### Architecture Stability Report

*   **✅ Production-Ready:** `DNN`, `CNN`, `LSTM`, `GRU`, `RNN`,  `Transformer`, `TCN`. These architectures are stable and recommended for production use.

*   **⚠️ Experimental:** `QuartzNet`, `Conformer`, `E-Branchformer`, `CRNN`.
    *   **Recommendation:** These state-of-the-art architectures are provided for advanced users and researchers. In the current version, they have shown to be highly sensitive to hyperparameter configurations and may not converge to a useful solution with the default settings ("model collapse"). We are making it our highest priority to develop robust intelligent defaults for these models in a future release. **We strongly advise against using them for production purposes at this time.**

---
### Known Issues & Bug Fixes

*   This section will be updated if any critical bugs are discovered in the current PyPI release and when workarounds are available.

### Future Development Notes

*   Our immediate priority is improving the out-of-the-box stability for all experimental architectures.
