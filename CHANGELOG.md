# Changelog

All notable changes to this project will be documented in this file.

---

## [3.0.0] - 2026-08-19

### Added

- **End-to-End (E2E) Training** - train wake word models directly on raw PCM audio waveforms. The model learns its own feature extraction internally, eliminating the need for pre-computed mel-spectrogram embeddings or a separate embedding model.
  - Three E2E architectures available via `model_type`:
    - `e2e_dnn` - Lightweight DNN with a `RawAudioFrontend` (fastest, smallest)
    - `e2e_cnn` - CNN with a `RawAudioFrontend` and `RawAudioBackbone` (balanced accuracy/size)
    - `e2e_quartznet` - QuartzNet-style blocks with configurable `e2e_quartznet_config` (most expressive)
  - E2E-specific config parameters: `mode: "e2e"`, `clip_samples` (fixed audio length in samples), `e2e_frontend_channels`, `e2e_frontend_depth`, `e2e_quartznet_config`
  - E2E data pipeline uses `data_generation_manifest` (augmented WAV clips) and `data_manifest` (WAV directories) instead of `feature_generation_manifest` / `.npy` files
  - E2E ONNX export includes a `mode: "e2e"` metadata tag and patches `MelSpectrogram` layers for ONNX-safe STFT
  - **Colab notebook**: `notebooks/Train_Your_First_E2E_Wake_Word_Model.ipynb`
  - **Example config**: `examples/e2e_training_config.yaml`

- **E2E Inference in `NanoInterpreter`** - E2E models are auto-detected via the `mode` metadata tag in the ONNX file. When an E2E model is loaded, the interpreter sets `preprocessor = None` (no mel-spectrogram extraction needed), accumulates raw audio in an internal buffer, and runs inference directly on raw PCM clips of `clip_samples` length once enough audio is collected. This means E2E models require **zero external dependencies** - no embedding model, no mel-spectrogram model, nothing. A single `.onnx` file is fully self-contained.

- **Custom E2E Architectures** - `model_type: "custom"` now works in E2E mode. Provide a `custom_model_config` with `module_path` and `class_name` pointing to any PyTorch `nn.Module` that accepts raw waveform input and returns an embedding. The custom class receives `input_shape`, `embedding_dim`, `dropout_prob`, `activation_fn`, `config`, and `frontend_channels` kwargs (with signature-based filtering).

- **E2E Cascade Support** - E2E models can be used in 2-stage cascade mode just like embedding models. The gate model can be an E2E model or a lightweight embedding model; the interpreter automatically detects each model's type.

- **E2E Remote Verifier** - the WebSocket server supports a new `e2e` pipeline mode (`--pipeline e2e`). The edge device sends raw PCM audio and the server runs the full E2E model directly. Also added `_TAG_AUDIO = 0x03` wire protocol tag for raw audio transmission in both `full` and `e2e` pipeline modes.

- **E2E Distillation** - `distill_model_e2e()` trains a tiny `e2e_dnn` student model from an E2E teacher, producing a compact `_lite.onnx` gate model. Configurable via `distillation:` block (`student_layer_size`, `student_n_blocks`, `student_embedding_dim`, `student_dropout_prob`).

### Changed

- `NanoInterpreter` now accepts `vad_threshold` and `enable_noise_reduction` as kwargs on `load_model()`, in addition to the existing constructor parameters.
- `predict_clip()` auto-detects E2E models via `self.preprocessor is None` and routes raw audio directly to `predict()` without chunking for feature extraction.
- ONNX export now writes `mode` metadata (`"e2e"` or `"embedding"`) to the model file for reliable detection at inference time.

---

## [2.1.0] - 2026-05-13

### Added

- **Unified CLI (`nanowakeword`)** - single entry point for the entire pipeline. No more separate commands for different tasks. Context is inferred from the flags you provide:
  ```bash
  nanowakeword -c config.yaml -T          # train
  nanowakeword -c config.yaml -d          # distill standalone
  nanowakeword --model my_model.onnx      # start server
  nanowakeword --info my_model.onnx       # inspect model
  ```
  The old `nanowakeword-train` command is kept as a backward-compatible alias.

- **Knowledge Distillation (`--distill` / `-d`)** - automatically generates a lightweight `_lite.onnx` gate model from any trained teacher using temperature-scaled KL divergence. Two modes:
  - Post-training: add `-d` alongside `-T` and the lite model is built right after training
  - Standalone: run `nanowakeword -c config.yaml -d` on an already-trained model, no retraining needed
  - Default student: ~12K parameters, ~50KB ONNX (~5–10x smaller than a typical teacher)
  - Fully configurable via `distillation:` config block (`steps`, `temperature`, `alpha`, `student_layer_size`, `student_embedding_dim`, etc.)

- **2-Stage Cascade Inference** - the lite model acts as a lightweight gatekeeper (Stage 1). The full model only runs when the gate fires, saving CPU on always-on systems:
  ```python
  # Auto-discovers my_model_lite.onnx in the same folder
  interpreter = NanoInterpreter.load_model("my_model.onnx", cascade=True)

  # Explicit gate path
  interpreter = NanoInterpreter.load_model(
      model="my_model.onnx",
      gate_model="my_model_lite.onnx",
      gate_threshold=0.25,
  )
  ```

- **RemoteVerifier WebSocket server** - host the full model (or the entire pipeline) on any machine and have edge devices connect to it. Three pipeline modes:
  - `verifier_only` (default) - edge sends pre-computed features, server runs only the wake word model
  - `full` - edge sends raw audio, server runs mel + embedding + wake word model
  ```bash
  nanowakeword --model my_model.onnx --pipeline full --port 8765
  ```

- **Distributed inference in `NanoInterpreter.load_model()`** - new `remote_verifier` and `remote_pipeline` parameters:
  ```python
  # Gate local, verifier remote
  NanoInterpreter.load_model(
      model="my_model_lite.onnx",
      remote_verifier="ws://192.168.1.100:8765",
      gate_threshold=0.25,
  )

  # Gate local, full pipeline remote (edge runs only the gate)
  NanoInterpreter.load_model(
      model="my_model_lite.onnx",
      remote_verifier="ws://server:8765",
      remote_pipeline="full",
  )

  # No local model - server handles everything
  NanoInterpreter.load_model(remote_verifier="ws://server:8765", remote_pipeline="full")
  ```

- **`NanoInterpreter` API improvements:**
  - `model` parameter replaces `model_path` (backward compatible - positional usage still works)
  - New properties: `score`, `verifier_score`, `gate_score`, `model_name`, `gate_name`, `is_cascade`, `info`
  - `detected(threshold)` method - clean boolean check
  - `listen()` now supports `blocking=False` (background thread), `on_audio` callback, and `on_score` callback
  - `stop()` - terminates a non-blocking `listen()` loop
  - `__repr__` - `print(interpreter)` shows useful state

- **`DetectionResult` object** - `predict()` now returns a rich result object instead of a plain dict. Supports both attribute access (`.score`, `.gate_score`, `.detected`) and dict-compatible access (`.get()`, `result["name"]`) for full backward compatibility.

- **`--info` flag** - inspect any `.onnx` model without loading the interpreter:
  ```bash
  nanowakeword --info my_model.onnx
  # Shows: name, type (lite/full), file size, parameter count, architecture type, input/output shapes
  ```

- **Collate function robustness** - training no longer crashes when `.npy` feature files have slightly different frame counts. The collate function now pads/truncates to the most common length in each batch.

- **Buffer warmup guard** - the interpreter no longer crashes on the first few audio chunks when a model requires more feature frames than the buffer currently holds (e.g., a 45-frame model on startup).

### Changed

- `nanowakeword-train` is now a backward-compatible alias. The primary command is `nanowakeword`.
- Distillation is enabled by default (`distillation.enabled: true`). Set to `false` to skip it.
- `predict()` return type changed from `dict` to `DetectionResult`. Existing code using `.get()` or `["key"]` access continues to work unchanged.

---

## [2.0.1–2.0.6] - 2026-02-02 to 2026-05-09

- Preprocessing stability improvements for edge-case audio formats
- Training loop stability fixes under low-data conditions
- Validation dataloader robustness improvements
- Documentation and configuration guide updates

---

## [2.0.0] - 2026-01-12

### Breaking Changes

- All import paths reorganized. Scripts from any **1.x** version will **not work** without modification.
- Update imports to match the new v2.0.0 package structure.

---

## [1.4.0–1.4.3] - 2025-12-13 to 2026-01-09

- Preprocessing pipeline improvements
- Training stability enhancements for recurrent architectures
- Documentation corrections

---

## [1.3.3] - 2025-11-14

### Added

- **6 new architectures:** CRNN, TCN, QuartzNet, Transformer, Conformer, E-Branchformer

### Fixed

- ONNX export failure for modern architectures - upgraded default opset to 17
- `average_models` crash with BatchNorm layers (`num_batches_tracked` type mismatch)
- TCN initialization `TypeError` with `nn.Sequential` and lambda functions

### Improved

- `ConfigGenerator` refined for more realistic augmentation round defaults
- All architecture definitions moved to dedicated `architectures.py` module

---

## [1.3.2] - 2025-11-08

### Added

- **Training resumption** - `--resume <path>` CLI flag + `checkpointing:` config block. Saves full training state (model, optimizer, scheduler, step, loss history).

### Fixed

- `AttributeError: 'LSTMModel' object has no attribute 'layer1'` crash during LSTM/GRU/CNN training

---

## [1.3.0] - 2025-11-05

Major re-architecture of the training framework.

### Added

- `auto_train` - autonomous training with EMA-based stability tracking and checkpoint ensembling (SWA)
- `ConfigProxy` - every training parameter controllable from a single YAML file
- Memory-mapped training - stream terabyte-scale feature sets from disk
- Live terminal training dashboard
- Strategic batch composition engine (`batch_composition` config)
- Standardized ONNX export with `InferenceWrapper` (output shape `[B, 1, 1]`)
- Configurable optimizer suite (AdamW, Adam, SGD) and LR schedulers (OneCycle, Cyclic, Cosine)
- Hybrid loss architecture (Triplet + classification)

### Removed

- All TensorFlow and TFLite dependencies
- Requirement for a separate validation set
