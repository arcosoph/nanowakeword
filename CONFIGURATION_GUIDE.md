# Nanowakeword Configuration Guide

Complete documentation of all configurable parameters in the **Nanowakeword** package, including descriptions, default values, meanings, and usage examples.

> **Latest release:** [![GitHub release](https://img.shields.io/github/v/release/arcosoph/nanowakeword?color=6C63FF&logo=github&logoColor=white)](https://github.com/arcosoph/nanowakeword/releases/latest)  
> **Guide updated for:** `v3.0.0`  
> **Documentation updated:** August 19, 2026
---

## Table of Contents

1.  [Workflow Overview](#workflow-overview)
2.  [Project & Data Paths](#project--data-paths)
3.  [Pipeline Control](#pipeline-control)
4.  [Mode Selection: Embedding vs E2E](#mode-selection-embedding-vs-e2e)
5.  [Model Architecture](#model-architecture)
6.  [Feature Manifest](#feature-manifest)
7.  [Batch Composition & ISBL Sampling](#batch-composition--isbl-sampling)
8.  [Training & Optimization](#training--optimization)
9.  [Loss Functions](#loss-functions)
10. [Checkpointing & Early Stopping](#checkpointing--early-stopping)
11. [Validation](#validation)
12. [Data Generation (TTS)](#data-generation-tts)
13. [Feature Generation Manifest](#feature-generation-manifest)
14. [Audio Processing Settings](#audio-processing-settings)
15. [Augmentation Settings](#augmentation-settings)
16. [Distillation](#distillation)
17. [ONNX Export Settings](#onnx-export-settings)
18. [Custom Export](#custom-export)
19. [Intelligent Auto-Configuration](#intelligent-auto-configuration)
20. [Inference Parameters](#inference-parameters)
21. [Server Configuration](#server-configuration)
22. [Command-Line Arguments](#command-line-arguments)

---

## Workflow Overview

NanoWakeWord operates in three sequential stages, each toggleable via
configuration keys or CLI flags:

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│    Generate Clips    │ --> │   Transform Clips    │ --> │     Train Model      │
│    (TTS synthesis)   │     │   (augment + feats)  │     │   (train + distill)  │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

| Stage | Config key | CLI flag | What it does |
|-------|-----------|----------|-------------|
| Generate clips | `generate_clips` | `-G` | Synthesizes audio from text using Piper TTS |
| Transform clips | `transform_clips` | `-t` | Augments audio and extracts embeddings (or saves raw WAVs in E2E mode) |
| Train model | `train_model` | `-T` | Trains the neural network and exports ONNX |
| Distill | `distill` (or `distillation.enabled`) | `-d` | Builds a lightweight "lite" student model |

---

## Project & Data Paths

### `output_dir`
- **Type:** `string`
- **Default:** `"./trained_models"`
- **Description:** Base directory where all trained models and artifacts are stored.
  A subdirectory named after `model_name` is created inside this directory.

### `model_name`
- **Type:** `string`
- **Default:** Auto-generated via `auto_gen_name()` - format `nww_<model_type>_model_v<N>`
- **Description:** Name of the trained model. Used as the subdirectory name
  under `output_dir` and as the filename for exported model files.
- **Example:**
  ```yaml
  model_name: "my_wakeword_v1"
  ```

### `target_phrase`
- **Type:** `string`
- **Default:** None
- **Description:** The wake word text. Used as the default phrase for TTS
  generation tasks that do not specify an explicit phrase. Should be set
  if using `data_generation_tasks` with `fixed_phrase` or `auto_adversarial`
  text sources.

### `positive_data_path`
- **Type:** `string` (file path)
- **Default:** None
- **Description:** Directory containing positive audio samples (actual wake word
  utterances). Required if using real (non-synthetic) positive data.
- **Requirements:**
  - Supports `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`,
    `.alac`, `.opus`, `.pcm`
  - Audio is converted to 16 kHz / 16-bit mono if `convert_audio: true`

### `negative_data_path`
- **Type:** `string` (file path)
- **Default:** None
- **Description:** Directory containing negative audio samples (non-wake-word
  utterances, e.g. general speech).

### `background_paths`
- **Type:** `list` of strings
- **Default:** `[]` (empty)
- **Description:** Directories containing background noise audio files for
  augmentation. Multiple paths are supported. The intelligent config engine
  computes a duplication rate for each path based on available noise duration.
- **Example:**
  ```yaml
  background_paths:
    - "./data/office_noise"
    - "./data/street_noise"
  ```

### `rir_paths`
- **Type:** `list` of strings
- **Default:** `[]` (empty)
- **Description:** Directories containing Room Impulse Response (RIR) WAV files.
  Each directory is scanned for all files (not just `.wav`).

### `convert_audio`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** When `true`, all audio directories (`positive_data_path`,
  `negative_data_path`, `background_paths`, `rir_paths`) are verified and
  converted to 16 kHz, 16-bit, mono WAV before the pipeline begins. A
  per-directory verification receipt is cached in
  `output_dir/.cache/verification_receipts/` to skip re-processing on
  subsequent runs.
- **CLI flag:** `-f` / `--force-verify` re-verifies all directories,
  ignoring the cache.

---

## Pipeline Control

These boolean switches enable or disable pipeline stages. CLI flags take
precedence over config file values when both are set.

### `generate_clips`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** Enables the TTS clip generation stage. Requires
  `data_generation_tasks` to be defined.

### `transform_clips`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** Enables the feature extraction / augmentation stage. Requires
  `data_generation_manifest` (or `feature_generation_manifest`) to be defined.
  In E2E mode, this stage produces augmented raw audio WAV files instead of
  `.npy` feature arrays.

### `train_model`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** Enables the model training stage.

### `overwrite`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** When `true`, forces regeneration of existing feature files,
  overwriting them. When `false`, existing `.npy` files are skipped.
- **CLI flag:** `--overwrite`

---

## Mode Selection: Embedding vs E2E

### `mode`
- **Type:** `string`
- **Default:** `"embedding"`
- **Valid Options:** `"embedding"`, `"e2e"`
- **Description:** Selects the training paradigm.

In **embedding mode** (default), the framework uses a pre-trained on-device
mel-spectrogram model and a Google speech-embedding model to convert raw audio
into 96-dimensional feature embeddings. The wake word model trains on these
pre-computed embeddings. Feature extraction happens once during the
`transform_clips` stage and is stored as `.npy` files.

In **E2E mode**, the model trains directly on raw PCM audio waveforms.
Feature extraction (mel-spectrogram) is part of the neural network itself. The `transform_clips` stage produces augmented raw WAV
files. The `feature_manifest` must point to directories of `.wav` files (not
`.npy` arrays).

- **Example:**
  ```yaml
  mode: "e2e"        # Train end-to-end on raw audio
  model_type: "e2e_dnn"
  ```

---

## Model Architecture

### `model_type`
- **Type:** `string`
- **Default:** `"dnn"`
- **Valid Options (Embedding mode):**
  `"dnn"`, `"cnn"`, `"lstm"`, `"gru"`, `"rnn"`, `"crnn"`, `"tcn"`,
  `"quartznet"`, `"transformer"`, `"conformer"`, `"e_branchformer"`,
  `"bcresnet"`, `"custom"`

- **Valid Options (E2E mode):**
  `"e2e_dnn"`, `"e2e_cnn"`, `"e2e_quartznet"`, `"custom"`

- **Description:** Selects the neural network architecture. Architectures are
  categorized by complexity and use case:

| Architecture | Category | Description |
|:---|:---:|:---|
| `dnn` | Feedforward | Dense layers with LayerNorm; smallest and fastest |
| `cnn` | Convolutional | 2D convolutions over spectrogram-like inputs |
| `rnn` | Recurrent | Bidirectional LSTM (fixed 64-dim hidden) |
| `lstm` | Recurrent | Configurable bidirectional LSTM |
| `gru` | Recurrent | Configurable bidirectional GRU |
| `crnn` | Hybrid | CNN frontend + configurable LSTM/GRU backend |
| `tcn` | Convolutional | Temporal Convolutional Network with dilated convolutions |
| `quartznet` | Convolutional | 1D time-channel separable convolutions (parameter-efficient) |
| `transformer` | Attention | Standard Transformer encoder with positional encoding |
| `conformer` | Hybrid | Conformer blocks (conv + multi-head attention) |
| `e_branchformer` | Hybrid | E-Branchformer with parallel attention/conv branches |
| `bcresnet` | Convolutional | Broadcasting-residual CNN with depthwise separable convolutions |
| `e2e_dnn` | E2E | Raw audio → Conv frontend → Dense embedding head |
| `e2e_cnn` | E2E | Raw audio → Conv frontend + CNN backbone |
| `e2e_quartznet` | E2E | Raw audio → Conv frontend + QuartzNet backbone |
| `custom` | User-defined | Loads a user-provided `torch.nn.Module` |

### `layer_size`
- **Type:** `integer`
- **Default:** `128` (auto-adjusted by intelligent config if not specified)
- **Applies to:** `dnn`, `lstm`, `gru`, `crnn`
- **Description:** Number of neurons in each hidden layer. Controls model
  capacity - larger values increase parameters and training time.

### `n_blocks`
- **Type:** `integer`
- **Default:** `1` (passed explicitly in `trainer.py` when not in config);
  auto-adjusted by the intelligent config engine when not specified
- **Applies to:** `dnn`, `lstm`, `gru`, `rnn`, `transformer`, `crnn` (RNN depth),
  `conformer`, `e_branchformer`, `tcn`
- **Description:** Number of stacked blocks/layers in the network. The meaning
  varies by architecture:
  - `dnn`: Number of `FCNBlock` layers after the input projection
  - `lstm`/`gru`/`rnn`: Number of recurrent layers
  - `transformer`/`conformer`/`e_branchformer`: Number of encoder blocks
  - `crnn`: Number of RNN layers (CNN portion is fixed)
  - `tcn`: Derived from the length of `tcn_channels`

### `dropout_prob`
- **Type:** `float`
- **Default:** `0.5` (auto-adjusted by intelligent config)
- **Valid Range:** `0.0` to `0.8`
- **Description:** Dropout probability applied in hidden layers. Higher values
  increase regularization. Applied within recurrent layers (when `n_blocks > 1`),
  after pooling in CNNs, and in the classifier head.

### `activation_function`
- **Type:** `string`
- **Default:** `"relu"`
- **Valid Options:** `"relu"`, `"gelu"`, `"silu"`
- **Description:** Activation function used in hidden layers. `gelu` and `silu`
  are smooth alternatives to `relu` that can improve convergence at a small
  computational cost.

### `embedding_dim`
- **Type:** `integer`
- **Default:** `64`
- **Description:** Dimensionality of the final embedding vector produced by the
  backbone network, before the classifier head. The classifier projects from
  `embedding_dim` → `embedding_dim // 2` → `n_classes`.

### Architecture-Specific Parameters

#### Transformer (`model_type: "transformer"`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `transformer_d_model` | `128` | Model dimension (input to attention) |
| `transformer_n_head` | `4` | Number of attention heads |

```yaml
model_type: "transformer"
transformer_d_model: 192
transformer_n_head: 6
```

#### CRNN (`model_type: "crnn"`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `crnn_cnn_channels` | `[16, 32, 32]` | CNN channel progression |
| `crnn_rnn_type` | `"lstm"` | `"lstm"` or `"gru"` |

```yaml
model_type: "crnn"
crnn_cnn_channels: [32, 64, 64]
crnn_rnn_type: "gru"
```

#### TCN (`model_type: "tcn"`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `tcn_channels` | `[64, 64, 128]` | Channel width per TemporalBlock |
| `tcn_kernel_size` | `3` | Convolution kernel size |

```yaml
model_type: "tcn"
tcn_channels: [128, 128, 256, 256]
tcn_kernel_size: 4
```

#### Conformer (`model_type: "conformer"`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `conformer_d_model` | `144` | Model dimension |
| `conformer_n_head` | `4` | Number of attention heads |

```yaml
model_type: "conformer"
conformer_d_model: 192
conformer_n_head: 6
```

#### E-Branchformer (`model_type: "e_branchformer"`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `branchformer_d_model` | `144` | Model dimension |
| `branchformer_n_head` | `4` | Number of attention heads |

```yaml
model_type: "e_branchformer"
branchformer_d_model: 256
branchformer_n_head: 8
```

#### QuartzNet (`model_type: "quartznet"`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `quartznet_config` | `[[256, 33, 1], [256, 33, 1], [512, 39, 1]]` | List of `[channels, kernel_size, repetitions]` triples |

```yaml
model_type: "quartznet"
quartznet_config:
  - [256, 33, 1]
  - [256, 33, 1]
  - [512, 39, 1]
  - [512, 39, 1]
```

#### BcResNet (`model_type: "bcresnet"`)
No architecture-specific parameters beyond `embedding_dim` and `dropout_prob`.
The network uses a fixed structure of 3 `BcResNetBlock` layers (32→64→128→256
channels) with adaptive global average pooling.

#### E2E Models (`model_type: "e2e_dnn"`, `"e2e_cnn"`, `"e2e_quartznet"`)
All E2E models use a `RawAudioFrontend` followed by a backbone network.
Additional parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `e2e_frontend_channels` | `32` | Output channels of the first conv layer in the frontend |
| `e2e_frontend_depth` | `2` for `e2e_dnn`/`e2e_cnn`, `3` for `e2e_quartznet` | Number of conv blocks in the frontend |
| `e2e_quartznet_config` | `[[64, 11, 1], [64, 13, 1], [64, 17, 1]]` | QuartzNet block config (only for `e2e_quartznet`) |

```yaml
mode: "e2e"
model_type: "e2e_quartznet"
e2e_frontend_channels: 64
e2e_frontend_depth: 3
sample_rate: 16000
clip_samples: 16000
```

### <a id="custom-custom"></a>Custom Architecture (`model_type: "custom"`)


Loads a user-defined `torch.nn.Module` class from a Python file or importable
module.

Required settings under `custom_model_config`:

| Key | Type | Description |
|-----|------|-------------|
| `module_path` | `string` | Path to a `.py` file or an importable module name |
| `class_name` | `string` | Name of the class to instantiate |

Optional settings:

| Key | Type | Description |
|-----|------|-------------|
| `params` | `dict` | Extra keyword arguments passed to the constructor |

The custom class constructor receives (based on what it accepts via signature
inspection):

| Argument | Type | Description |
|----------|------|-------------|
| `input_shape` | `tuple` | Shape of a single input sample |
| `embedding_dim` | `int` | Number of embedding output dimensions |
| `dropout_prob` | `float` | Dropout probability |
| `activation_fn` | `nn.Module` | Activation function instance |
| `config` | `ConfigProxy` | Full configuration proxy |
| `*params` | - | Any additional keys from `params` |

The forward method must return a tensor of shape `[batch_size, embedding_dim]`.

```python
# my_model.py
import torch
from torch import nn

class MyCustomModel(nn.Module):
    def __init__(self, input_shape, embedding_dim=64, dropout_prob=0.5,
                 activation_fn=None, config=None, hidden_channels=32):
        super().__init__()
        self.activation_fn = activation_fn if activation_fn is not None else nn.ReLU()
        # ... build layers ...

    def forward(self, x):
        # x: [batch, time, features] or [batch, 1, time, features]
        # return: [batch, embedding_dim]
        ...
```

```yaml
model_type: "custom"
custom_model_config:
  module_path: "my_model.py"
  class_name: "MyCustomModel"
  params:
    hidden_channels: 64
```

---

## Feature Manifest

Defines paths to pre-computed feature files (`.npy` format) used for training.
In E2E mode, paths point to directories of `.wav` files instead.

```yaml
feature_manifest:
  targets:            # Positive samples (wake word)
    t: "./trained_models/model_v1/features/positive.npy"
  negatives:          # Negative samples
    n: "./features/negative.npy"
    hn: "./features/hard_negatives.npy"
    b: "./features/noise.npy"          # Background noise

  targets_val:        # Optional validation data (suffix `_val`)
    t: "./features/val_positive.npy"
  negatives_val:
    n: "./features/val_negatives.npy"
```

### `data_manifest`
- **Type:** `dict`
- **Default:** None
- **Description:** Alias for `feature_manifest`. The trainer checks for
  `feature_manifest` first, then falls back to `data_manifest`. Both use the
  same structure.

### Categories
Each top-level key in the manifest is a **category**. The category name
determines the label assigned to samples:

| Category | Label | Description |
|----------|-------|-------------|
| `targets` | `1.0` | Wake word samples (positive class) |
| `negatives` | `0.0` | Non-wake-word samples (negative class) |
| Any `targets_*_val` | `1.0` | Validation wake word samples |
| Any `negatives_*_val` | `0.0` | Validation negative samples |

Categories ending in `_val` are automatically routed to the validation
dataloader. Non-`_val` categories are used for training.

---

## Batch Composition & ISBL Sampling

### `batch_composition`
- **Type:** `dict` mapping `string` → `integer`
- **Default:** `{'targets': 30, 'negatives': 230}` (when not specified)
- **Description:** Defines how many samples are drawn per batch from each
  dataset or dataset group in the `feature_manifest`. The sum of all values
  equals the total batch size.

Keys in `batch_composition` can reference either:
- A **specific dataset key** (e.g., `t`, `hn`, `b`) - samples are drawn only
  from that dataset.
- A **category name** (e.g., `targets`, `negatives`) - samples are drawn from
  all datasets under that category, proportionally to their hardness scores.

```yaml
batch_composition:
  targets: 20         # 20 samples drawn from all datasets under 'targets'
  n: 50              # 50 samples drawn only from 'negatives.n'
  hn: 30             # 30 samples drawn only from 'negatives.hn'
  b: 150             # 150 samples drawn only from 'negatives.b'
  # Total batch = 250
```

### ISBL (Importance Sampling based on Loss) Parameters

The `DynamicClassAwareSampler` uses a hardness score per sample, updated
dynamically during training using the ISBL algorithm. Higher loss = higher
hardness = higher sampling probability.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hardness_ema_alpha` | `float` | `0.05` | EMA smoothing factor for updating sample hardness scores |
| `hardness_floor` | `float` | `0.05` | Minimum hardness score (prevents scores from collapsing to zero) |
| `hardness_reset_interval` | `int` | `5000` | Reset hardness scores periodically (0 = disabled) |
| `hardness_reset_decay` | `float` | `0.5` | Decay factor when resetting (0.5 = 50% old, 50% reset to 1.0) |

These parameters control how aggressively the sampler prioritizes hard
examples. The sampler applies the formula:

```
P(sample) ∝ (hardness ^ 0.75) + epsilon
```

where `0.75` is the fixed `hardness_smoothing_factor` in the sampler.

---

## Training & Optimization

### `steps`
- **Type:** `integer`
- **Default:** Auto-calculated by the intelligent config engine (range 10000–40000)
- **Description:** Total number of training iterations. The intelligent engine
  estimates this from the effective data volume (base hours × augmentation
  rounds) multiplied by 1000 steps per effective hour, adjusted for data
  quality.

### `batch_size`
- **Type:** `integer`
- **Default:** Auto-calculated (the intelligent config engine was working on
  this but the logic is currently commented out; users should specify
  explicitly if needed)
- **Description:** Number of training samples per gradient step. The total
  batch size is determined by `batch_composition` (sum of its values). This
  parameter is informational; the actual batch size comes from the sampler.

### `optimizer_type`
- **Type:** `string`
- **Default:** `"adamw"`
- **Valid Options:** `"adamw"`, `"adam"`, `"sgd"`
- **Description:** Optimization algorithm used during training.

### `learning_rate_max`
- **Type:** `float`
- **Default:** Auto-calculated by the intelligent config engine (base 5e-5)
- **Description:** Maximum learning rate. Used with cyclical schedulers
  (`onecycle`, `cyclic`) or as the starting learning rate (`cosine`).

### `learning_rate_base`
- **Type:** `float`
- **Default:** `learning_rate_max / 10` (auto-calculated)
- **Description:** Base/minimum learning rate. Used as the minimum in cyclic
  schedulers, the floor in cosine annealing, or the starting point for
  `onecycle`.

### `lr_scheduler_type`
- **Type:** `string`
- **Default:** `"onecycle"`
- **Valid Options:** `"onecycle"`, `"cyclic"`, `"cosine"`
- **Description:** Learning rate schedule strategy.
  - `onecycle`: One-cycle schedule from base to max LR and back to base
  - `cyclic`: Triangular2 cyclic schedule with configurable up/down phases
  - `cosine`: Cosine annealing from max LR to base LR

### `clr_step_size_up`
- **Type:** `integer`
- **Default:** Auto-calculated by intelligent config (based on `steps / num_cycles`)
- **Description:** Number of steps to increase LR in each cycle. Only used
  when `lr_scheduler_type: "cyclic"`.

### `clr_step_size_down`
- **Type:** `integer`
- **Default:** Same as `clr_step_size_up`
- **Description:** Number of steps to decrease LR in each cycle. Only used
  when `lr_scheduler_type: "cyclic"`.

### `weight_decay`
- **Type:** `float`
- **Default:** `0.01`
- **Description:** L2 regularization coefficient applied to all parameters.

### `momentum`
- **Type:** `float`
- **Default:** `0.9`
- **Valid Range:** `0.0` to `1.0`
- **Description:** Momentum factor. Only used when `optimizer_type: "sgd"`.

### `num_workers`
- **Type:** `integer`
- **Default:** `2`
- **Description:** Number of worker processes for PyTorch `DataLoader`.
  Increasing this can speed up data loading on multi-core systems. When set
  to `0`, data loading is single-threaded.

---

## Loss Functions

The training loop supports two loss function variants, selected via
`loss_function`.

### `loss_function`
- **Type:** `string`
- **Default:** `"bias_weighted"`
- **Valid Options:** `"bias_weighted"`, `"asymmetric_focal"`
- **Description:** Loss function strategy. `asymmetric_focal` has its
  implementation commented out in the current code and falls back to
  `bias_weighted` behavior. Use `"bias_weighted"` for full functionality.

### `LOSS_BIAS`
- **Type:** `float`
- **Default:** `0.75`
- **Description:** Weight given to the **negative** class in
  `BiasWeightedLoss`. A value of `0.75` means 75% of the loss comes from
  negative samples and 25% from positive samples. This upweights hard
  negatives, which is critical for wake word detection where the vast
  majority of real-world audio is non-wake-word.

  The combined loss is:
  `total_loss = LOSS_BIAS * mean(neg_loss) + (1 - LOSS_BIAS) * mean(pos_loss)`

  Label smoothing of `0.05` is applied to the soft targets used in the
  cross-entropy calculation, but masks (pos/neg split) are computed from
  the original hard labels before smoothing. Per-example losses (used for
  ISBL hardness tracking) are computed with the same class weighting.

### `afl_gamma_pos` (Asymmetric Focal Loss)
- **Type:** `float`
- **Default:** `0.0`
- **Description:** Focusing parameter for positive samples. Currently
  unused because the asymmetric focal loss implementation is commented out.

### `afl_gamma_neg` (Asymmetric Focal Loss)
- **Type:** `float`
- **Default:** `4.0`
- **Description:** Focusing parameter for negative samples. Currently
  unused (see `afl_gamma_pos`).

### Logit Regularization Parameters

The training loop applies an L2 penalty on extreme logit magnitudes to
prevent the model from becoming overconfident:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logit_reg_weight` | `float` | `2e-4` | Weight of the logit regularization term |
| `logit_reg_margin` | `float` | `6.0` | Target logit magnitude; logits exceeding ±margin are penalized |

Set `logit_reg_weight` to `0` to disable logit regularization.

---

## Checkpointing & Early Stopping

### `checkpointing`
- **Type:** `dict`
- **Description:** Controls periodic checkpoint saving during training.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `boolean` | `false` | Enable/disable checkpointing |
| `interval_steps` | `integer` | `1000` | Save a checkpoint every N steps |
| `limit` | `integer` | `3` | Maximum number of checkpoint files to retain (oldest are deleted) |

Each checkpoint stores: model state dict, optimizer state, scheduler state,
EMA loss, best error score, best training checkpoints (SWA pool), RNG states
for deterministic resume, and training history.

```yaml
checkpointing:
  enabled: true
  interval_steps: 500
  limit: 5
```

### `checkpoint_averaging_top_k`
- **Type:** `integer`
- **Default:** `5`
- **Description:** Number of best checkpoints (by stable loss) to average
  via Stochastic Weight Averaging (SWA) when building the final model.
  Only used as a fallback when no validation data is available or when the
  validation set overlaps with training data (detected when val error = 0
  and the best checkpoint appears very late in training).

### `early_stopping_patience`
- **Type:** `integer`
- **Default:** `10%` of `steps` (or `0` if `steps < 3000`)
- **Description:** Stop training if the stable (EMA) loss has not improved by
  at least `min_delta` for this many steps. Set to `0` to disable. This is
  a fallback - when validation data is present, validation-based early
  stopping takes priority.

### `val_early_stopping_patience`
- **Type:** `integer`
- **Default:** `15%` of `steps`
- **Description:** Stop training if the validation error score has not
  improved for this many steps. Only active when validation data is provided.

### `min_delta`
- **Type:** `float`
- **Default:** `0.0001`
- **Description:** Minimum improvement in EMA loss required to reset the
  early stopping counter.

### `checkpoint_pool_interval`
- **Type:** `integer`
- **Default:** `500`
- **Description:** Interval (in steps) at which the current model state is
  considered for the training-loss checkpoint pool. This pool is used as a
  fallback for final model averaging when no validation data is available.
  The pool tracks the top-K checkpoints with the lowest EMA loss.

### `enable_journaling`
- **Type:** `boolean`
- **Default:** `true`
- **Description:** When `true`, after training completes, a `training_journal.md`
  file is written to `output_dir` summarizing the run. The journal uses a
  "show on change" layout: only configuration parameters that changed since
  the previous run are displayed in each row. The journal history is cached
  in `output_dir/.cache/journal_cache/training_history.json`.

### `show_training_summary`
- **Type:** `boolean`
- **Default:** `true`
- **Description:** When `true`, prints an ASCII table showing the effective
  training configuration at startup, and updates it dynamically during
  training with live metrics (loss, learning rate, accuracy, etc.).

### `debug_mode`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** When `true`, enables verbose debug logging. A rotating log
  file (5 MB, 30 backups) is written to
  `output_dir/model_name/training_artifacts/training_debug/training_debug.log`.
  Debug logs include per-step metrics, validation results, hardness score
  updates, and checkpoint saves.

### `stabilization_steps`
- **Type:** `integer`
- **Default:** `5%` of `max_steps` (auto-calculated; was `1500` in older versions)
- **Description:** Number of initial warmup steps before metrics like
  validation and early stopping are evaluated. Prevents premature decisions
  during initial convergence.

### `ema_alpha`
- **Type:** `float`
- **Default:** `0.01`
- **Description:** Exponential moving average factor for tracking the
  smoothed (stable) training loss. The smoothed loss is:
  `ema = alpha * current_loss + (1 - alpha) * ema`. Lower values produce
  smoother loss curves.

---

## Validation

Validation runs periodically during training to track generalization. It
requires `feature_manifest` (or `data_manifest`) entries with the `_val`
suffix (e.g., `targets_val`, `negatives_val`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validation_batch_size` | `integer` | `256` | Batch size for validation passes |
| `val_interval` | `integer` | `500` | Run validation every N training steps |
| `val_stabilization_steps` | `integer` | Same as `stabilization_steps` | Delay before starting validation |
| `val_subsample_batches` | `integer` | `0` (use all) | Subsample N batches during training for speed; 0 uses the full set |
| `val_miss_weight` | `float` | `4.0` | Penalty weight for missed detections (false negatives) in the operating-point search |
| `val_fp_weight` | `float` | `1.0` | Penalty weight for false positives in the operating-point search |
| `validation_smoothing_window` | `integer` | `3` | Number of recent validation runs to average for smoothing |

The validation routine sweeps thresholds from 0.2 to 0.8 (13 steps) and
selects the one that minimizes the weighted error score:
`miss_weight * FN + fp_weight * FP`.

---

## Data Generation (TTS)

Synthetic audio clips are generated using Piper TTS. This stage is controlled
by the `data_generation_tasks` key - a list of task dictionaries, each
defining an independent synthesis job.

### `data_generation_tasks`
- **Type:** `list` of `dict`
- **Description:** List of TTS generation tasks. Each task is a dictionary
  with the following keys:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | `string` | `"Unnamed Task N"` | Descriptive name for the task |
| `enabled` | `boolean` | `true` | Skip this task if `false` |
| `output_dir` | `string` | (required) | Directory to save generated `.wav` files |
| `num_samples` | `integer` | (required) | Number of audio clips to generate |
| `file_prefix` | `string` | `"sample"` | Prefix for output filenames |
| `tts_settings` | `dict` | `{}` | Task-specific TTS settings (overrides global) |
| `text_source` | `dict` | (required) | Defines how text is generated (see below) |

### `text_source`

Each task must specify a `text_source` dictionary with a `type` key:

#### `type: "fixed_phrase"`
Generates audio for a single phrase, repeated `num_samples` times.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `phrase` | `string` | Falls back to global `target_phrase` | The exact text to synthesize |

#### `type: "from_list"`
Generates audio from a user-provided list of phrases.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `phrases` | `list[str]` | (required) | List of text phrases |
| `repeat_each` | `integer` | `1` | How many times to repeat each phrase |

#### `type: "auto_adversarial"`
Generates phonetically similar English words/phrases using the CMU Pronouncing
Dictionary. Creates challenging negative samples that sound similar to the
wake word.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `base_phrase` | `string` | Falls back to `target_phrase` / `phrase` | Phrase to generate variations from |
| `include_partial_phrase` | `float` | `0.0` | Probability of creating shorter adversarial phrases |
| `include_input_words` | `float` | `0.0` | Probability of preserving original wake-word words |
| `multi_word_prob` | `float` | `0.4` | Probability of embedding the adversarial phrase in a longer phrase |
| `max_multi_word_len` | `integer` | `3` | Maximum total words for multi-word expansions |

#### `type: "phoneme_adversarial"`
Generates adversarial text by substituting phonemes in the wake word using
a trained phonemizer model. Creates extremely phonetically confusable
negatives.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `base_phrase` | `string` | Falls back to `target_phrase` / `phrase` | Phrase to generate phoneme variants from |
| `min_distance` | `float` | `0.35` | Minimum phonetic distance (0.0 = identical, 1.0 = completely different) |

The phonemizer model is automatically downloaded from
`https://github.com/arcosoph/phonemize/releases/download/v0.2.0/phonemize_m1.pt`
on first use, or loaded from `NwwResourcesModel/phonemize_model/phonemize_m1.pt`.

### `tts_settings`
- **Type:** `dict`
- **Default:** `{}` (uses Piper TTS defaults)
- **Description:** Global or task-specific TTS synthesis parameters. These
  are passed directly to `generate_samples()` as keyword arguments.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `models` | `str` / `list[str]` / `dict` | None (auto-load) | Piper model name, file path, directory, or `{'onnx': url, 'json': url}` dict |
| `models_dir` | `string` | `"NwwResourcesModel/tts_models"` | Directory to load or download TTS models |
| `speaker_ids` | `int` / `list[int]` | None | Speaker ID(s) for multi-speaker models |
| `length_scales` | `list[float]` | `[1.0]` | Controls speech speed (higher = slower) |
| `noise_scales` | `list[float]` | `[0.667]` | Amount of randomness in speech sound |
| `noise_w_scales` | `list[float]` | `[0.8]` | Waveform variation for naturalness |

```yaml
target_phrase: "hey nano"

tts_settings:
  models: "en_US-libritts_r-medium"
  models_dir: "./tts_models"
  length_scales: [0.9, 1.0, 1.1]
  noise_scales: [0.667, 0.8]
  noise_w_scales: [0.8]
  speaker_ids: [0, 1, 2]

data_generation_tasks:
  - name: "Positive Wake Words"
    enabled: true
    output_dir: "dataset/positive"
    num_samples: 1000
    text_source:
      type: "fixed_phrase"

  - name: "Phoneme-Based Hard Negatives"
    enabled: true
    output_dir: "dataset/negative"
    num_samples: 1500
    file_prefix: "neg_phoneme"
    text_source:
      type: "phoneme_adversarial"
      min_distance: 0.4
```

---

## Feature Generation Manifest

When `transform_clips` is enabled, the `data_generation_manifest` (or
`feature_generation_manifest`) key defines how raw audio is transformed into
training-ready features (embedding mode) or augmented WAV files (E2E mode).

```yaml
data_generation_manifest:
  positive_features:
    input_audio_dirs: ["./dataset/positive"]
    output_filename: "positive_features.npy"
    use_background_noise: true
    use_rir: true
    augmentation_rounds: 10
    augmentation_settings:
      min_snr_in_db: 5.0
      pitch_prob: 0.5

  negative_features:
    input_audio_dirs: ["./dataset/negative", "./dataset/external_neg"]
    output_filename: "negative_features.npy"
    use_background_noise: true
    use_rir: false
    augmentation_rounds: 5
```

### Recipe Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_audio_dirs` | `list[string]` | (required) | Source audio directories; `.wav` files are discovered recursively |
| `output_filename` | `string` | (required for embedding) | Name of the output `.npy` file (saved in `features/` dir) |
| `num_samples` | `integer` | `0` | Number of clips to generate (only for E2E mode) |
| `use_background_noise` | `boolean` | `true` | Mix with background noise from `background_paths` |
| `use_rir` | `boolean` | `false` | Apply room impulse response convolution from `rir_paths` |
| `augmentation_rounds` | `integer` | `1` | How many times to augment each input clip |
| `augmentation_settings` | `dict` | Inherits global | Override augmentation params for this recipe |

In **E2E mode**, instead of `.npy` output, each recipe saves `.wav` files
to its `output_dir` (not `output_filename`). The `num_samples` key controls
how many total clips are produced (per augmentation round). Files are named
`{file_prefix}_{index:06d}.wav`.

In **embedding mode**, the output is a memory-mapped `.npy` file of shape
`(num_clips, frames, 96)` stored in `output_dir/model_name/features/`.

### `background_paths_duplication_rate`
- **Type:** `list[integer]`
- **Default:** Auto-calculated by intelligent config engine
- **Description:** Duplication rate for background noise paths. Each background
  path is repeated this many times during augmentation to balance noise
  availability across datasets of different sizes. The intelligent config
  computes this as `ceil(max_noise_hours / path_hours)` for each path.

---

## Audio Processing Settings

### `sample_rate`
- **Type:** `integer`
- **Default:** `16000`
- **Description:** Target sample rate for all audio processing (both training
  and inference). All audio is resampled to this rate.

### `clip_samples`
- **Type:** `integer`
- **Default:** `16000`
- **Description:** Fixed clip length in samples for E2E mode training. Each
  audio file is either cropped (random position) or zero-padded to this
  length. Required for E2E training and inference.

### `audio_processing`
- **Type:** `dict`
- **Description:** Controls how audio clip duration is determined during the
  transform stage.

```yaml
audio_processing:
  clip_length_samples: 16000          # Explicit clip length (all modes)

  # OR use autotune (embedding mode only, ignored in E2E):
  autotune_length:
    enabled: true                        # Default: true
    num_samples_to_inspect: 50           # Number of positive clips to inspect
    duration_buffer_ms: 750             # Buffer added to median duration
    min_allowable_length: 32000         # Minimum clip length (2.0s @ 16kHz)
    snap_to_min_tolerance: 4000         # Snap to minimum if within this tolerance
```

#### `audio_processing.clip_length_samples`
- **Type:** `integer`
- **Default:** None (falls through to autotune)
- **Description:** If set, all audio clips are exactly this many samples long.

#### `audio_processing.autotune_length`
- **Type:** `dict`
- **Description:** When `enabled` is `true` and `clip_length_samples` is not
  set, the framework inspects a random sample of positive audio clips,
  computes the median duration, adds a buffer, and snaps to a round number
  of milliseconds.

### `augmentation_batch_size`
- **Type:** `integer`
- **Default:** Auto-calculated (based on system RAM and CPU cores):
  - Systems with `psutil`: computed from available RAM and CPU count,
    rounded to nearest power of 2 in range [16, 128]
  - Without `psutil`: `32`
- **Description:** Batch size for the audio augmentation processing stage
  (separate from training batch size).

### `feature_gen_num_workers`
- **Type:** `integer`
- **Default:** `num_workers` (from training config) if not set
- **Description:** Number of worker processes for multiprocessing during
  feature generation. Only used in embedding mode.

### `feature_gen_cpu_ratio`
- **Type:** `float`
- **Default:** `0.6`
- **Valid Range:** `0.0` to `1.0`
- **Description:** Fraction of available CPU cores used for mel-spectrogram
  computation during embedding extraction. The actual count is
  `max(1, int(cpu_count * ratio))`.

---

## Augmentation Settings

Controls stochastic audio augmentation applied during the `transform_clips`
stage. Parameters can be set globally (top-level `augmentation_settings`) or
per-recipe in `data_generation_manifest`.

```yaml
augmentation_settings:
  gain_prob: 1.0              # Probability of applying gain adjustment
  min_gain_in_db: -3.0        # Minimum gain in dB
  max_gain_in_db: 3.0         # Maximum gain in dB
  pitch_prob: 0.5             # Probability of pitch shifting
  min_pitch_semitones: -2.0   # Minimum pitch shift
  max_pitch_semitones: 2.0    # Maximum pitch shift
  min_snr_in_db: 5.0          # Minimum signal-to-noise ratio
  max_snr_in_db: 30.0         # Maximum signal-to-noise ratio
  rir_prob: 0.5               # Probability of RIR convolution
  min_volume_augmentation: 0.5  # Minimum target volume level
  max_volume_augmentation: 1.0  # Maximum target volume level
```

### `augmentation_settings.rir_prob`
- **Type:** `float`
- **Default:** `0.5`
- **Valid Range:** `0.0` to `1.0`
- **Description:** Probability of applying room impulse response convolution.
  Requires `rir_paths` to be configured and `use_rir: true` in the recipe.

### `augmentation_settings.gain_prob` / `min_gain_in_db` / `max_gain_in_db`
- **Type:** `float`
- **Defaults:** `1.0`, `-3.0`, `3.0`
- **Description:** Apply random gain adjustment. `gain_prob` is the
  probability per clip. Gain is uniformly sampled from
  `[min_gain_in_db, max_gain_in_db]`.

### `augmentation_settings.pitch_prob` / `min_pitch_semitones` / `max_pitch_semitones`
- **Type:** `float`
- **Defaults:** `0.5`, `-2.0`, `2.0`
- **Description:** Apply random pitch shifting without changing duration.
  `pitch_prob` is the probability per clip. Shift is uniformly sampled from
  `[min, max]` semitones.

### `augmentation_settings.min_snr_in_db` / `max_snr_in_db`
- **Type:** `float`
- **Defaults:** `5.0`, `30.0`
- **Description:** Signal-to-noise ratio range (in dB) for background noise
  mixing. Lower values = more noise. Background noise is randomly selected
  from `background_paths`.

### `augmentation_settings.min_volume_augmentation` / `max_volume_augmentation`
- **Type:** `float`
- **Defaults:** `0.5`, `1.0`
- **Description:** Target peak volume level after augmentation. Each clip is
  scaled so its peak amplitude falls within this range (relative to full
  scale of 1.0). This replaces peak normalization with a stochastic volume
  target for robustness to varying microphone gains.

---

## Distillation

Knowledge distillation builds a tiny "student" model that mimics the
output of the trained "teacher" model. The student is always a stripped-down
DNN (regardless of teacher architecture) to maximize speed.

### `distillation`
- **Type:** `dict`
- **Description:** Controls knowledge distillation. The student loss is:

  ```
  loss = alpha * KL(student_soft || teacher_soft) * T²  +  (1 - alpha) * BCE(student_logit, hard_label)
  ```

  where `T` is the temperature and `alpha` is the soft-loss weight.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `boolean` | `true` | Enable distillation after training |
| `steps` | `integer` | `8000` | Number of distillation training steps |
| `temperature` | `float` | `4.0` | Temperature scaling for soft labels |
| `alpha` | `float` | `0.7` | Weight on soft loss (1-alpha = hard loss) |
| `learning_rate` | `float` | `5e-4` | Student learning rate |
| `log_interval` | `integer` | `500` | Logging interval for distillation |
| `student_layer_size` | `integer` | `8` | Hidden layer size of student DNN |
| `student_n_blocks` | `integer` | `1` | Number of FCN blocks in student |
| `student_embedding_dim` | `integer` | `8` | Embedding dim of student |
| `student_dropout_prob` | `float` | `0.1` | Dropout in student model |

```yaml
distillation:
  enabled: true
  steps: 10000
  temperature: 3.0
  alpha: 0.6
  learning_rate: 3e-4
  student_layer_size: 16
  student_embedding_dim: 16
```

The distilled model is exported as `<model_name>_lite.onnx` and can be used
as a gatekeeper in cascade mode during inference.

---

## ONNX Export Settings

### `onnx_opset_version`
- **Type:** `integer`
- **Default:** `17`
- **Description:** ONNX opset version used by the built-in ONNX exporter. Higher
  versions support more operators but may reduce compatibility with older
  runtimes. Can be overridden by placing `onnx_opset_version` under the
  `custom_export` section.

---

## Custom Export

After training (and after distillation), the framework exports the model to
ONNX and PyTorch formats. A user-provided custom export hook can be
configured to produce additional formats (e.g., CoreML, TFLite).

### `custom_export` (or `export_model`)
- **Type:** `dict`
- **Description:** Configuration for the custom export hook. Supports two
  modes: Python script or shell command.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `script` | `string` | None | Path to a Python file exposing a callable |
| `function` | `string` | `"export_model"` | Name of the callable in the script |
| `command` | `string` | None | Shell command with `{model_path}`, `{model_name}`, `{output_dir}` placeholders |
| `onnx_opset_version` | `integer` | Inherits top-level value (default `17`) | Overrides the top-level `onnx_opset_version` for this export |

When using `script` mode, the callable receives these keyword arguments:
`model`, `input_shape`, `config`, `model_name`, `output_dir`. The callable
may use the in-memory PyTorch model directly, or operate on the ONNX file
produced by the built-in exporter.

```yaml
custom_export:
  script: "/path/to/my_coreml_export.py"
  function: "export_to_coreml"
  onnx_opset_version: 17
```

Or using shell command mode:

```yaml
custom_export:
  command: "python /scripts/onnx_to_coreml.py --onnx {model_path} --out {output_dir}"
```

---

## Intelligent Auto-Configuration

The `ConfigGenerator` class analyzes dataset statistics and computes
appropriate hyperparameters when they are not explicitly set in the config.
The generated values always **merge with and are overridden by** user
settings in the config file.

### Input Statistics

The config generator uses dataset statistics computed by
`DatasetAnalyzer`. These are passed as the `stats` argument to
`ConfigGenerator(stats)`. In the trainer, if `DatasetAnalyzer` is not
explicitly called, `ConfigGenerator` runs with empty stats and produces
minimal defaults.

| Stat | Description |
|------|-------------|
| `H_pos` | Duration of positive data in hours |
| `H_neg` | Duration of negative data in hours |
| `H_noise` | Total duration of background noise in hours |
| `H_noise_paths` | Dict mapping each noise path to its duration in hours |
| `A_noise` | Average RMS amplitude of noise (0–1) |
| `N_rir` | Count of RIR files |

### Generated Parameters

| Parameter | How it's computed |
|-----------|-------------------|
| `augmentation_rounds` | `clamp(required_multiplier, 2, 5)` where `required_multiplier = dynamic_target_hours / base_hours` |
| `steps` | `effective_data_volume * 1000`, adjusted by data quality (`1.1 - 0.2 * normalized_quality`), clamped to [10000, 40000] |
| `n_blocks` | `clamp(log10(effective_data_volume + 1) * 2.0, 1.0, 4.0)`, rounded |
| `layer_size` | `64 * 2^(n_blocks - 1)`, clamped to [64, 512] |
| `learning_rate_max` | `5e-5 * clamp(size_factor, 0.8, 2.0) * clamp(noise_factor, 0.5, 1.0)` |
| `learning_rate_base` | `learning_rate_max / 10` |
| `dropout_prob` | `clamp(0.6 + overfitting_risk * 0.75, 0.4, 0.8)` |
| `clr_step_size_up` | `steps / num_cycles * 0.4` |
| `clr_step_size_down` | `steps / num_cycles * 0.6` |
| `background_paths_duplication_rate` | `ceil(max_noise_hours / path_hours)` per path |
| `augmentation_batch_size` | `min([16, 32, 64, 128], key=lambda x: abs(x - calculated))` where `calculated = 16 * (safe_ram_gb / 6.0) * sqrt(cores / 4)`; `32` if `psutil` unavailable |
| `tts_batch_size` | GPU: `512` (>=12 GB), `256` (>=8 GB), `128` (>=4 GB), `32` (else); CPU: from `cpu_cores` and `total_ram_gb` weighted metric, nearest power of 2 in [16, 256]; `32` if `psutil` unavailable |

---

## Inference Parameters

The `NanoInterpreter` class (vendored from `nanowakeword.interpreter.NanoInterpreter`)
accepts the following keyword arguments via `load_model()`:

```python
from nanowakeword import NanoInterpreter

interpreter = NanoInterpreter.load_model(
    model="my_model.onnx",
    vad_threshold=0.5,            # Enable VAD filtering
    enable_noise_reduction=True,  # Enable noise reduction
)
```

### `enable_noise_reduction`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** When `true`, applies stationary noise reduction (via
  `noisereduce`) to each audio chunk before prediction. Requires the
  `noisereduce` package (`pip install noisereduce`).

### `vad_threshold`
- **Type:** `float`
- **Default:** `0`
- **Description:** When greater than `0`, enables voice activity detection
  using Silero's VAD model. Audio chunks with no detected speech are
  suppressed (scores zeroed) before the wake word model runs, conserving
  CPU. The specific threshold behavior depends on the inference pipeline
  (local or remote). On the local `NanoInterpreter`, VAD scores are
  accumulated in an internal buffer; when the average VAD confidence is
  below this threshold, the wake word model is bypassed for that chunk.

### Cascade (Gate + Verifier) Mode

When `cascade=True` is passed to `load_model()`, the interpreter
automatically looks for `<model_name>_lite.onnx` in the same directory and
uses it as a lightweight gatekeeper (Stage 1). The main model (Stage 2)
only runs when the gate score exceeds `gate_threshold`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cascade` | `boolean` | `false` | Enable 2-stage cascade mode |
| `gate_model` | `string` | Auto-discovered | Explicit path to a custom gate model |
| `gate_threshold` | `float` | `0.3` | Gate score threshold to activate the verifier |

```python
# Auto-discover lite model in same directory
interpreter = NanoInterpreter.load_model("my_model.onnx", cascade=True)

# Explicit gate model
interpreter = NanoInterpreter.load_model(
    "my_model.onnx",
    gate_model="custom_lite.onnx",
    gate_threshold=0.25,
)
```

### Remote Verifier Mode

The interpreter can offload the verifier (Stage 2) to a remote WebSocket
server running the `remote_verifier` module.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remote_verifier` | `string` | None | WebSocket URI (e.g. `ws://192.168.1.100:8765`) |
| `remote_pipeline` | `string` | `"verifier_only"` | `"verifier_only"` or `"full"` |
| `remote_timeout` | `float` | `2.0` | Seconds to wait for server response |
| `remote_api_key` | `string` | None | API key for server authentication |
| `remote_token` | `string` | None | Token for server authentication |
| `remote_ssl_certfile` | `string` | None | Client certificate for mTLS |
| `remote_ssl_keyfile` | `string` | None | Client private key for mTLS |
| `remote_ssl_ca_certs` | `string` | None | CA bundle for server TLS verification |

```python
# Gate local, verifier remote (edge runs mel+embedding+gate)
interpreter = NanoInterpreter.load_model(
    model="my_model_lite.onnx",
    remote_verifier="ws://192.168.1.100:8765",
    gate_threshold=0.25,
)

# Edge sends raw audio; server runs full pipeline
interpreter = NanoInterpreter.load_model(
    model="my_model_lite.onnx",
    remote_verifier="ws://192.168.1.100:8765",
    remote_pipeline="full",
    gate_threshold=0.25,
)
```

### `predict()` Parameters

```python
result = interpreter.predict(
    audio_chunk,             # np.ndarray, int16 PCM
    patience={},             # {model_name: n_frames}
    threshold={},            # {model_name: value}
    debounce_time=0.0,       # seconds between detections
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `np.ndarray` | (required) | 16-bit PCM audio chunk |
| `patience` | `dict` | `{}` | Require N consecutive frames above threshold before firing |
| `threshold` | `dict` | `{}` | Per-model detection thresholds |
| `debounce_time` | `float` | `0.0` | Suppress repeated detections within this many seconds |

`patience` and `debounce_time` cannot be used simultaneously. When using
either, `threshold` must also be provided.

### `DetectionResult`

The `predict()` method returns a `DetectionResult` object with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `scores` | `dict` | Raw `{model_name: score}` mapping |
| `model_name` | `string` | Primary model name |
| `score` | `float` | Primary model's score |
| `gate_name` | `string\|None` | Gate model name (cascade mode) |
| `gate_score` | `float` | Gate model score (0.0 if not in cascade) |
| `detected` | `bool` | True if `score >= threshold` |
| `threshold` | `float` | Threshold used for `detected` |

### `listen()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `on_detection` | `callable` | prints to console | Callback when detection fires: `(model_name, score)` |
| `threshold` | `float` | `0.5` | Detection threshold (0–1) |
| `cooldown` | `float` | `1.0` | Minimum seconds between detections |
| `chunk_size` | `integer` | `1280` | Audio frames per inference chunk (80ms @ 16kHz) |
| `on_score` | `callable` | None | Per-chunk callback: `(verifier_score, gate_score)` |
| `on_audio` | `callable` | None | Per-chunk callback: `(audio_array)` |
| `blocking` | `boolean` | `true` | If `true`, blocks until Ctrl+C |

---

## Server Configuration

The RemoteVerifier is a WebSocket server that hosts wake word model inference.
It is started via `nanowakeword --model <path>` or
`python -m nanowakeword.interpreter.remote_verifier --model <path>`.

### Pipeline Modes

| Mode | Edge runs | Server runs | Wire format tag |
|------|----------|-------------|-----------------|
| `verifier_only` | mel + embedding + gate | verifier only | `0x01` (features) |
| `embedding` | mel | embedding + verifier | `0x02` (mel frames) |
| `full` | gate only | mel + embedding + verifier | `0x03` (raw audio) |
| `e2e` | gate only | E2E model | `0x03` (raw audio) |

### Server CLI Arguments

| Flag | Description |
|------|-------------|
| `--model` | Path to `.onnx` model (required) |
| `--pipeline` | `verifier_only`, `embedding`, `full`, or `e2e` |
| `--host` | Bind address (default `0.0.0.0`) |
| `--port` | Port (default `8765`) |
| `--log` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--api-key` | API key for auth (repeat for multiple) |
| `--enable-tokens` | Allow API key → token exchange |
| `--token-ttl` | Token lifetime in seconds (default `3600`) |
| `--token-secret` | HMAC secret for token signing |
| `--rate-limit` | Max messages per window per IP (0 = disabled) |
| `--rate-window` | Rate limit window in seconds (default `60`) |
| `--ip-allowlist` | Allowed IP or CIDR (repeat for multiple) |
| `--ssl-certfile` / `--ssl-keyfile` | TLS certificate and key |
| `--ssl-ca-certs` | CA bundle for mutual TLS |
| `--max-connections` | Max simultaneous clients (0 = unlimited) |
| `--ban-duration` | Ban seconds after rate-limit breach (default `300`) |

### Programmatic Server Usage

```python
from nanowakeword.interpreter.remote_verifier import serve
from nanowakeword.interpreter import build_security

security = build_security(
    api_keys=["my-secret-key"],
    enable_tokens=True,
    token_ttl=1800,
    rate_limit=200,
    rate_window=60,
    ip_allowlist=["192.168.1.0/24"],
    ssl_certfile="server.crt",
    ssl_keyfile="server.key",
)

serve(
    model_path="my_model.onnx",
    pipeline="full",
    host="0.0.0.0",
    port=8765,
    security=security,
)
```

### SecurityConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_keys` | `list[str]` | `[]` | Plaintext API keys (hashed in memory at startup) |
| `enable_tokens` | `boolean` | `false` | Enable API key → token exchange |
| `token_ttl` | `integer` | `3600` | Token lifetime in seconds |
| `token_secret` | `string` | auto-generated | HMAC secret for token signing |
| `rate_limit` | `integer` | `0` | Max messages/window/IP (0 = disabled) |
| `rate_window` | `integer` | `60` | Rate limit window in seconds |
| `ip_allowlist` | `list[str]` | `[]` | Allowed IPs/CIDRs (empty = allow all) |
| `ssl_certfile` | `string` | None | PEM certificate for WSS/TLS |
| `ssl_keyfile` | `string` | None | PEM private key for WSS/TLS |
| `ssl_ca_certs` | `string` | None | CA bundle for mutual TLS |
| `max_connections` | `integer` | `0` | Max simultaneous clients (0 = unlimited) |
| `ban_duration` | `integer` | `300` | Ban duration after rate-limit breach |

---

## Command-Line Arguments

The `nanowakeword` CLI entry point (defined in `cli.py`) supports:

```bash
# Full pipeline: generate + transform + train + distill
nanowakeword -c config.yaml -G -t -T -d

# Training only (uses pipeline flags from config file)
nanowakeword -c config.yaml -T

# Inspect a model file and exit
nanowakeword --info my_model.onnx

# Start the RemoteVerifier server
nanowakeword --model my_model.onnx --pipeline full --port 8765

# Resume from checkpoint
nanowakeword -c config.yaml --resume ./trained_models/my_model

# Force regenerate features
nanowakeword -c config.yaml --overwrite
```

| Flag | Shorthand | Description |
|------|-----------|-------------|
| `--config` | `-c` | Path to YAML config (required for training) |
| `--generate_clips` | `-G` | Enable TTS clip generation stage |
| `--transform_clips` | `-t` | Enable augmentation + feature extraction |
| `--train` | `-T` | Enable model training |
| `--distill` | `-d` | Generate lite model via distillation |
| `--force-verify` | `-f` | Re-verify all data directories |
| `--overwrite` | | Force regeneration of feature files |
| `--resume` | | Resume from a model directory checkpoint |
| `--model` | | Start RemoteVerifier server with this model |
| `--pipeline` | | Server pipeline mode: `verifier_only` or `full` |
| `--host` | | Server bind address |
| `--port` | | Server port |
| `--info` | | Inspect an ONNX model file and exit |

### CLI vs Config Precedence

When no pipeline flags (`-G`, `-t`, `-T`, `-d`) are provided on the command
line, the framework reads `generate_clips`, `transform_clips`, `train_model`,
and `distill` from the YAML config file. CLI flags always override
config file values when both are set.

---

## Complete Example Configuration

```yaml
# === Paths ===
output_dir: "./trained_models"
model_name: "my_hotword_v1"
positive_data_path: "./training_data/positive"
negative_data_path: "./training_data/negative"
background_paths:
  - "./training_data/noise"
rir_paths:
  - "./training_data/rir"

# === Mode & Architecture ===
mode: "embedding"
model_type: "conformer"
conformer_d_model: 144
conformer_n_head: 4
embedding_dim: 64
n_blocks: 4
layer_size: 192
dropout_prob: 0.3
activation_function: "gelu"

# === Data Generation (TTS) ===
generate_clips: true
target_phrase: "hey nano"
tts_settings:
  models: "en_US-libritts_r-medium"
  length_scales: [0.9, 1.0, 1.1]
  noise_scales: [0.667, 0.8]
  speaker_ids: [0, 1, 2, 3]

data_generation_tasks:
  - name: "Positive Wake Words"
    enabled: true
    output_dir: "dataset/positive"
    num_samples: 2000
    text_source:
      type: "fixed_phrase"

  - name: "Hard Negatives"
    enabled: true
    output_dir: "dataset/negative"
    num_samples: 2000
    text_source:
      type: "phoneme_adversarial"
      min_distance: 0.4

# === Feature Generation ===
transform_clips: true
convert_audio: true
sample_rate: 16000
audio_processing:
  autotune_length:
    enabled: true
    num_samples_to_inspect: 50
    duration_buffer_ms: 750
    min_allowable_length: 32000
    snap_to_min_tolerance: 4000

augmentation_batch_size: 64
feature_gen_cpu_ratio: 0.6

augmentation_settings:
  gain_prob: 1.0
  min_gain_in_db: -2.0
  max_gain_in_db: 2.0
  pitch_prob: 0.3
  min_pitch_semitones: -1.0
  max_pitch_semitones: 1.0
  min_snr_in_db: 5.0
  max_snr_in_db: 30.0
  rir_prob: 0.3

data_generation_manifest:
  positive_features:
    input_audio_dirs: ["./dataset/positive"]
    output_filename: "positive_features.npy"
    augmentation_rounds: 5

  negative_features:
    input_audio_dirs: ["./dataset/negative"]
    output_filename: "negative_features.npy"
    augmentation_rounds: 3

  noise_features:
    input_audio_dirs: ["./dataset/positive"]
    output_filename: "noise_features.npy"
    use_background_noise: true
    augmentation_rounds: 3

# === Feature Manifest (for training) ===
feature_manifest:
  targets:
    p: "./trained_models/my_hotword_v1/features/positive_features.npy"
  negatives:
    n: "./trained_models/my_hotword_v1/features/negative_features.npy"
    b: "./trained_models/my_hotword_v1/features/noise_features.npy"
  targets_val:
    p: "./trained_models/my_hotword_v1/features/val_positive.npy"
  negatives_val:
    n: "./trained_models/my_hotword_v1/features/val_negative.npy"

# === Training ===
train_model: true
batch_composition:
  targets: 30
  negatives: 230
num_workers: 4

optimizer_type: "adamw"
lr_scheduler_type: "onecycle"
weight_decay: 0.01
steps: 30000

LOSS_BIAS: 0.75
loss_function: "bias_weighted"
logit_reg_weight: 2e-4
logit_reg_margin: 6.0

# === Validation ===
val_interval: 500
val_subsample_batches: 0
val_miss_weight: 4.0
val_fp_weight: 1.0

# === Checkpointing ===
checkpointing:
  enabled: true
  interval_steps: 1000
  limit: 3

early_stopping_patience: 0
val_early_stopping_patience: 5000
checkpoint_averaging_top_k: 5

# === Distillation ===
distillation:
  enabled: true
  steps: 8000
  temperature: 4.0
  alpha: 0.7
  learning_rate: 5e-4

# === ONNX Export ===
onnx_opset_version: 17

# === Custom Export (optional) ===
# custom_export:
#   script: "/path/to/export_to_tflite.py"
#   onnx_opset_version: 17

# === Display & Logging ===
show_training_summary: true
debug_mode: false
enable_journaling: true
force_verify: false

```
