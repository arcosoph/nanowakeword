# NanoWakeWord Configuration Guide

Complete documentation of all configurable parameters in the **NanoWakeWord** package, including descriptions, default values, meanings, and usage examples.

## Table of Contents

1. [Project & Data Paths](#project--data-paths)
2. [Model Architecture](#model-architecture)
3. [Training & Optimization](#training--optimization)
4. [Feature Manifest](#feature-manifest)
5. [Batch Composition](#batch-composition)
6. [Data Generation](#data-generation)
7. [Augmentation Settings](#augmentation-settings)
8. [Feature Generation Manifest](#feature-generation-manifest)
9. [Advanced Settings](#advanced-settings)
10. [Pipeline Control](#pipeline-control)
11. [Intelligent Auto-Configuration](#intelligent-auto-configuration)
12. [Inference Parameters](#inference-parameters)

---

## Project & Data Paths

Configuration parameters for project organization and data source locations.

### `model_name`
- **Type:** `string`
- **Default:** Auto-generated based on model type (e.g., `XXX_dnn_v1`)
- **Description:** Name of the trained model. Used for creating directories and organizing outputs.
- **Example:**
  ```yaml
  model_name: "my_wakeword_A_v1"
  ```

### `output_dir`
- **Type:** `string`
- **Default:** `"./trained_models"`
- **Description:** Base directory where all trained models and artifacts will be stored.
- **Example:**
  ```yaml
  output_dir: "./trained_models"
  # Creates: ./trained_models/my_wakeword_v1/model/, ./trained_models/my_wakeword_v1/features/
  ```

### `positive_data_path`
- **Type:** `string` (file path)
- **Mandatory**: Yes
- **Default**: None
- **Description:** Directory containing positive audio samples (actual wake word utterances).
- **Requirements:**
  - Must contain `.wav` files at 16 kHz sample rate
  - Mono or stereo audio (will be converted to mono)
  - Can be empty if using only generated synthetic samples

### `negative_data_path`
- **Type:** `string` (file path)
- **Mandatory**: Yes
- **Default**: None
- **Description:** Directory containing negative audio samples (non-wake-word utterances).
- **Example:**
  ```yaml
  negative_data_path: "./data/common_words"
  ```

### `background_paths`
- **Type:** `list` of strings
- **Default:** Optional
- **Description:** Directories containing background noise audio files for augmentation. Multiple paths supported.
- **Example:**
  ```yaml
  background_paths: # You can add multiple path or only one
    - "./data/office_noise"
    - "./data/street_noise"
    - "./data/home_noise"
  ```

### `rir_paths`
- **Type:** `list` of strings
- **Default:** Optional
- **Description:** Directories containing Room Impulse Response (RIR) files for acoustic augmentation.
- **Note:** At least one RIR path is required for intelligent configuration.

---

## Model Architecture

Parameters controlling the neural network structure and behavior.

### `model_type`
- **Type:** `string`
- **Default:** `"dnn"`
- **Valid Options:** `"dnn"`, `"lstm"`, `"gru"`, `"rnn"`, `"cnn"`, `"transformer"`, `"crnn"`, `"tcn"`, `"quartznet"`, `"conformer"`, `"e_branchformer"`
- **Description:** The neural network architecture to use for wake word detection.
- **Complexity Levels (from simplest to most complex):**
  - `dnn` - Dense feedforward network (lightweight, fast)
  - `cnn` - Convolutional Neural Network (good for spectrograms)
  - `lstm`, `gru`, `rnn` - Recurrent networks (excellent for sequences)
  - `crnn` - Hybrid CNN-RNN (combines both strengths)
  - `transformer`, `conformer`, `e_branchformer` - Advanced attention-based (most powerful, most complex)

- **Examples by use case:**
  ```yaml
  # Embedded/Edge device (minimal resources)
  model_type: "dnn"
  
  # Edge device with more resources
  model_type: "lstm"
  
  # Desktop/cloud with ample resources
  model_type: "conformer"
  ```

### `layer_size` (DNN/RNN-based architectures)
- **Type:** `integer`
- **Default:** `128`
- **Valid Range:** `64` to `512`
- **Description:** Number of neurons in each hidden layer for feedforward and recurrent layers.
- **Relationship to model capacity:** Larger values = more parameters = longer training, better performance (up to a point)
- **Example:**
  ```yaml
  layer_size: 256  # Larger model, slower but potentially better
  ```

### `n_blocks`
- **Type:** `integer`
- **Default:** `3`
- **Valid Range:** `1` to `10`
- **Description:** Number of stacked blocks/layers in the model.
  - For `dnn`: Number of fully connected layers
  - For `lstm`/`gru`: Number of recurrent layers
  - For `transformer`: Number of encoder layers
  - For `crnn`: Number of RNN layers (CNN part is fixed)

- **Example:**
  ```yaml
  n_blocks: 5  # Deeper network
  ```

### `dropout_prob`
- **Type:** `float`
- **Default:** `0.5` (intelligently adjusted)
- **Valid Range:** `0.0` to `0.8`
- **Description:** Dropout probability per layer to prevent overfitting.
  - Higher values = more regularization = potential underfitting
  - Lower values = less regularization = potential overfitting
  - Typically 0.2-0.5 for most models

- **Example:**
  ```yaml
  dropout_prob: 0.3
  ```

### `activation_function` (Advanced)
- **Type:** `string`
- **Default:** `"relu"`
- **Valid Options:** `"relu"`, `"gelu"`, `"silu"`
- **Description:** Activation function used in hidden layers.
  - `relu` - Traditional, fast, widely supported
  - `gelu` - Smooth, often better convergence
  - `silu` - Modern alternative (Swish activation)

- **Example:**
  ```yaml
  activation_function: "gelu"
  ```

### `embedding_dim` (Advanced)
- **Type:** `integer`
- **Default:** `64`
- **Valid Range:** `32` to `256`
- **Description:** Dimensionality of the final embedding before classification.

### Architecture-Specific Parameters

#### Transformer Architecture
```yaml
model_type: "transformer"
transformer_d_model: 128        # Model dimension, default: 128
transformer_n_head: 4           # Number of attention heads, default: 4
```

#### CRNN Architecture
```yaml
model_type: "crnn"
crnn_cnn_channels: [16, 32, 32]  # CNN channel progression, default: [16, 32, 32]
crnn_rnn_type: "lstm"             # "lstm" or "gru", default: "lstm"
```

#### TCN Architecture
```yaml
model_type: "tcn"
tcn_channels: [64, 64, 128]      # Channel progression, default: [64, 64, 128]
tcn_kernel_size: 3                # Convolution kernel size, default: 3
```

#### Conformer Architecture
```yaml
model_type: "conformer"
conformer_d_model: 144            # Model dimension, default: 144
conformer_n_head: 4               # Attention heads, default: 4
```

#### E-Branchformer Architecture
```yaml
model_type: "e_branchformer"
branchformer_d_model: 144         # Model dimension, default: 144
branchformer_n_head: 4            # Attention heads, default: 4
```

#### QuartzNet Architecture
```yaml
model_type: "quartznet"
quartznet_config:                 # Channel, kernel, repeat config
  - [256, 33, 1]
  - [256, 33, 1]
  - [512, 39, 1]
```

---

## Training & Optimization

Parameters governing the training loop, optimization, and learning rate scheduling.

### `steps`
- **Type:** `integer`
- **Default:** `20000` (intelligently adjusted based on data volume)
- **Valid Range:** `1000` to `100000`
- **Description:** Total number of training iterations/steps.
- **Calculation Logic:**
  - `base_steps = effective_data_volume * 1000` steps per hour
  - Adjusted based on data quality and model complexity
  - Typically 10,000-40,000 for most scenarios

- **Example:**
  ```yaml
  steps: 50000  # For very large/complex datasets
  ```

### `batch_size`
- **Type:** `integer`
- **Default:** `128`
- **Valid Range:**
  - **Minimum**: 1 (at least 1 sample per batch required)
  - **Maximum**: Limited by GPU/CPU memory
  - CPU training → 16–128+ typical
  - single GPU → 32–256+ typical
  - multi-GPU → 512+ possible

- **Description:** Number of training samples per batch.
  - Larger batches = faster training, more stable gradients, more memory
  - Smaller batches = slower training, noisier gradients, less memory

- **Example:**
  ```yaml
  batch_size: 128
  ```

### `optimizer_type`
- **Type:** `string`
- **Default:** `"adamw"`
- **Valid Options:** `"adamw"`, `"adam"`, `"sgd"`
- **Description:** Optimization algorithm.
  - `adamw` - Adaptive Moment Estimation with Weight decay (recommended)
  - `adam` - Original adaptive optimizer
  - `sgd` - Stochastic Gradient Descent (simple, slower convergence)

- **Example:**
  ```yaml
  optimizer_type: "adamw"
  ```

### `learning_rate_max`
- **Type:** `float`
- **Default:** Auto-calculated
- **Description:** Maximum learning rate during training (used with cycle schedulers).
- **Intelligently Adjusted Based On:**
  - Dataset size (larger datasets → higher LR)
  - Data noise levels (cleaner data → higher LR)
  - Model complexity

- **Example:**
  ```yaml
  learning_rate_max: 0.001
  ```

### `learning_rate_base`
- **Type:** `float`
- **Default:** `learning_rate_max / 10`
- **Description:** Minimum/base learning rate during cyclical scheduling.
- **Note:** Automatically calculated if not specified.

### `lr_scheduler_type`
- **Type:** `string`
- **Default:** `"onecycle"`
- **Valid Options:** `"onecycle"`, `"cyclic"`, `"cosine"`
- **Description:** Learning rate schedule strategy.
  - `onecycle` - One cycle from base to max LR and back (good for fast convergence)
  - `cyclic` - Multiple triangular cycles (good for exploration)
  - `cosine` - Cosine annealing (smooth, gradual decrease)

- **Example:**
  ```yaml
  lr_scheduler_type: "onecycle"
  ```

### `clr_step_size_up` (Cyclic LR)
- **Type:** `integer`
- **Default:** Auto-calculated based on total steps
- **Description:** Number of steps to increase LR in each cycle.

### `clr_step_size_down` (Cyclic LR)
- **Type:** `integer`
- **Default:** Auto-calculated based on total steps
- **Description:** Number of steps to decrease LR in each cycle.

### `weight_decay`
- **Type:** `float`
- **Default:** `0.01`
- **Description:** L2 regularization coefficient to prevent overfitting.

### `momentum` (SGD optimizer)
- **Type:** `float`
- **Default:** `0.9`
- **Valid Range:** `0.0` to `1.0`
- **Description:** Momentum factor for SGD optimizer.

### `num_workers`
- **Type:** `integer`
- **Default:** `2`
- **Valid Range:** `0` to `CPU_count`
- **Description:** Number of worker threads for data loading.
  - 0 = single thread (slower, no multiprocessing)
  - 2-4 = typical for most systems
  - Increase for large datasets and fast GPUs

---

## Feature Manifest

Defines paths to pre-computed audio feature files (.npy format) used for training.

### Structure
```yaml
feature_manifest: # You can add Multiple Sources
  targets:           # Positive samples (wake word)
    key1: "path/to/features.npy"
    # others.. 
  negatives:         # Negative samples (non-wake-words)
    key1: "path/to/negatives.npy"
    
  backgrounds:       # Background noise samples
    key1: "path/to/noise.npy"
    # others..
  # Optional: Validation data (if _val key suffix used)
  targets_val:
    key1: "path/to/val_positive.npy"
  negatives_val:
    key1: "path/to/val_negatives.npy"
  backgrounds_val:
    key1: "path/to/val_noise.npy"
```

### Key Naming Convention (It will use `batch_composition`)
- Keys within each category can be arbitrary unique identifiers
- Short keys preferred for readability (e.g., `t`, `n`, `b`)
- Multiple feature sources can be specified with different keys (e.g., `real_pos`, `bg2`, `hard_neg`)

### Example with Multiple Sources
```yaml
feature_manifest:
  targets:
    t: "./trained_models/model_v1/features/positive.npy"
    my_voice: "./voice/muhammad_abid/muhammad_abid_data.npy"
    
  negatives:
    common_words: "./features/common_words.npy"
    hard_negatives: "./features/similar_words.npy"
    external_dataset: "./external/negatives_1m.npy"
    
  backgrounds:
    office: "./features/office_noise.npy"
    street: "./features/street_noise.npy"
    home: "./features/home_noise.npy"
```

---

## Batch Composition

Controls the ratio of different sample types in each training batch.

### Structure
```yaml
batch_composition:
  targets: 32          # Number of positive samples per batch
  negatives: 64        # Number of negative samples per batch
  backgrounds: 32      # Number of background noise samples per batch
```

### Example: Balanced Curriculum
```yaml
batch_size: 128        # Total samples per batch

batch_composition:
  targets: 32          # 25% positive samples
  negatives: 64        # 50% negative/hard samples
  backgrounds: 32      # 25% background noise
```






Here is a **clear and professional English explanation** that makes it obvious to the user that `batch_composition` works **based on `feature_manifest` datasets**:

---

### Batch Composition

`batch_composition` defines **how many feature samples are taken per training batch from the datasets specified in `feature_manifest`.**

Each entry in `batch_composition` corresponds to a dataset or dataset group defined in `feature_manifest`.

```yaml
batch_composition:
  target: 10
  n: 68
  hn: 10
  b: 40
  # others..
```

This means that each training batch will contain:

* **10 samples** from the `targets` datasets (all datasets inside the `targets`)
* **68 samples** from the `negatives.n` dataset
* **10 samples** from the `negatives.hn` dataset
* **40 samples** from the `backgrounds.b` dataset

---

### Relationship with `feature_manifest`

`batch_composition` always uses the datasets defined in `feature_manifest`.

For example:

```yaml
feature_manifest:
  targets:
    t: positive_features.npy

  negatives:
    n: negative_features.npy
    hn: hard_negative_features.npy

  backgrounds:
    b: noise_features.npy
```

The keys used in `batch_composition` must match the dataset keys or dataset groups defined in `feature_manifest`.

---

### How Samples Are Selected

When a group name is used:

```yaml
batch_composition:
  target: 10
```

the samples are **randomly selected from all datasets inside the `targets` group.**

For example:

```yaml
targets:
  t1: dataset1.npy
  t2: dataset2.npy
  t3: dataset3.npy
```

Then:

```
target: 10
```

means:

* A total of **10 samples will be taken from the targets group**
* Samples are selected **randomly across all target datasets**
* Not exactly 10 from each dataset

Example distribution:

* t1 → 3 samples
* t2 → 4 samples
* t3 → 3 samples

---

### Selecting From a Specific Dataset

To select samples from a specific dataset, use its dataset key:

```yaml
batch_composition:
  t: 10
```

This means:

* **10 samples will be taken only from `targets.t`**

because:

```yaml
targets:
  t: positive_features.npy
```

---

### Summary

* `feature_manifest` defines **where the datasets are located**
* `batch_composition` defines **how many samples are taken from those datasets per batch**
* Keys in `batch_composition` must match keys or groups in `feature_manifest`









### Auto-Composition (if not specified)
If `batch_composition` is not provided, NanoWakeWord automatically generates balanced defaults:
- All three categories present: 25% targets, 50% negatives, 25% backgrounds
- Two categories: 33% / 67% split
- One category: 100% to that category

---

## Data Generation

Parameters for synthetic audio generation using Text-to-Speech (TTS).

*It will be updated later*

---

## Augmentation Settings

Audio augmentation parameters for training robustness.

### Structure
```yaml
augmentation_settings:
  min_snr_in_db: 3.0           # Minimum signal-to-noise ratio
  max_snr_in_db: 30.0          # Maximum signal-to-noise ratio
  rir_prob: 0.2                # Probability of applying RIR
  pitch_prob: 0.3              # Probability of pitch shift
  min_pitch_semitones: -2.0    # Minimum pitch shift
  max_pitch_semitones: 2.0     # Maximum pitch shift
  gain_prob: 1.0               # Probability of gain adjustment
  min_gain_in_db: -6.0         # Minimum gain in dB
  max_gain_in_db: 6.0          # Maximum gain in dB
  ColoredNoise: 0.30           # Probability of adding colored noise
```

### Parameter Descriptions

#### `min_snr_in_db` / `max_snr_in_db`
- **Type:** `float`
- **Range:** Typically -10 to +40 dB
- **Description:** Signal-to-Noise ratio range when mixing audio with background noise.
  - Lower SNR = harder augmentation (more noise, harder training)
  - Higher SNR = easier augmentation (less noise, cleaner audio)

#### `rir_prob`
- **Type:** `float` (0.0-1.0)
- **Default:** `0.2`
- **Description:** Probability of applying room impulse response convolution.
- **Effect:** Simulates acoustic room effects for robustness.

#### `pitch_prob` / `min_pitch_semitones` / `max_pitch_semitones`
- **Type:** `float`
- **Pitch Range:** Typically ±2 to ±5 semitones
- **Description:** Pitch shifting for voice variation without changing content.

#### `gain_prob` / `min_gain_in_db` / `max_gain_in_db`
- **Type:** `float`
- **Gain Range:** Typically -6 to +6 dB
- **Description:** Volume adjustment for robustness to different microphone levels.

#### `ColoredNoise`
- **Type:** `float` (0.0-1.0)
- **Default:** `0.30`
- **Description:** Probability of adding colored noise (pink/brown noise).

### Example: Aggressive Augmentation
```yaml
augmentation_settings:
  min_snr_in_db: -5.0          # Very noisy (challenging)
  max_snr_in_db: 20.0
  rir_prob: 0.5                # Frequent RIR
  pitch_prob: 0.6              # Frequent pitch shift
  min_pitch_semitones: -4.0    # Wider pitch range
  max_pitch_semitones: 4.0
  gain_prob: 1.0
  min_gain_in_db: -12.0        # Wider gain range
  max_gain_in_db: 12.0
```

---

## Feature Generation Manifest

Defines how to generate and process feature files from raw audio.

### Structure
```yaml
feature_generation_manifest:
  feature_key_name1:
    input_audio_dirs: ["path/to/audio"]  # Source audio directories
    output_filename: "output_features.npy" # Output file name
    use_background_noise: true            # Mix with background noise
    use_rir: true                         # Apply RIR augmentation
    augmentation_rounds: 10               # Number of augmentation iterations
    augmentation_settings:                # Optional: override global settings
      min_snr_in_db: 5.0
      pitch_prob: 0.5
```

### Parameters

#### `input_audio_dirs`
- **Type:** `list` of strings
- **Description:** Directories containing raw audio files to process.

#### `output_filename`
- **Type:** `string`
- **Description:** Name of the output .npy feature file (without `.npy` extension).

#### `use_background_noise`
- **Type:** `boolean`
- **Default:** `true`
- **Description:** Mix samples with background noise from `background_paths`.

#### `use_rir`
- **Type:** `boolean`
- **Default:** `true`
- **Description:** Apply room impulse response convolution.

#### `augmentation_rounds`
- **Type:** `integer`
- **Default:** `10`
- **Valid Range:** `1` to `50`
- **Description:** How many times to augment each audio sample.
  - Higher rounds = more training data, slower generation
  - Examples: 1-3 rounds for large datasets, 10-20 for small datasets

#### `augmentation_settings`
- **Type:** `dict` (optional)
- **Description:** Feature-specific augmentation overrides (if not using global settings).

### Example: Multiple Feature Generations
```yaml
feature_generation_manifest:
  positive_features:
    input_audio_dirs: ["./data/positive"]
    output_filename: "positive_features.npy"
    use_background_noise: true
    use_rir: true
    augmentation_rounds: 15
    
  hard_negative_features:
    input_audio_dirs: ["./data/negative"]
    output_filename: "hard_negative_features.npy"
    use_background_noise: true
    use_rir: true
    augmentation_rounds: 20
    
  pure_noise_features:
    input_audio_dirs: ["./data/background_noise"]
    output_filename: "noise_features.npy"
    use_background_noise: false
    use_rir: false
    augmentation_rounds: 5
    augmentation_settings:
      gain_prob: 0.5
      pitch_prob: 0.3

  others_features:
    # your paramiters...
```

---

## Advanced Settings

Fine-tuning parameters for specialized scenarios.

### `augmentation_batch_size`
- **Type:** `integer`
- **Default:** Auto-calculated (16-128 based on system resources)
- **Description:** Batch size for audio augmentation (separate from training batch size).
- **Note:** Intelligently calculated based on available RAM and CPU cores.

### `feature_gen_cpu_ratio`
- **Type:** `float`
- **Default:** `1.0`
- **Valid Range:** `0.0` to `1.0`
- **Description:** CPU utilization ratio for feature generation (0.0=GPU only, 1.0=CPU ratio).

### Checkpointing & Early Stopping

#### `checkpoint_averaging_top_k`
- **Type:** `integer`
- **Default:** `5`
- **Description:** Number of best checkpoints to average for final model.

#### `checkpointing.enabled`
- **Type:** `boolean`
- **Default:** `true`
- **Description:** Enable periodic model checkpointing during training.

#### `checkpointing.interval_steps`
- **Type:** `integer`
- **Default:** `1000`
- **Description:** Save checkpoint every N training steps.

#### `checkpointing.limit`
- **Type:** `integer`
- **Default:** `2`
- **Description:** Maximum checkpoint files to keep (oldest are deleted).

#### `early_stopping_patience`
- **Type:** `integer`
- **Default:** `0`
- **Valid Range:** `0` to `100`
- **Description:** Stop training if no improvement for N validation checks.
- **0 = disabled**

#### `main_delta`
- **Type:** `float`
- **Default:** `0.0001`
- **Description:** Minimum improvement threshold for early stopping.

### Loss & Training Dynamics

#### `stabilization_steps`
- **Type:** `integer`
- **Default:** `1500`
- **Description:** Number of gradual warmup steps at training start.
- **Effect:** Prevents instability in initial iterations.

#### `ema_alpha`
- **Type:** `float`
- **Default:** `0.01`
- **Valid Range:** `0.0` to `1.0`
- **Description:** Exponential moving average smoothing factor for loss tracking.
- **Higher values**: Faster response to recent changes
- **Lower values**: Smoother, more stable trend

### Validation Settings

#### `validation_batch_size`
- **Type:** `integer`
- **Default:** `256`
- **Description:** Batch size for validation pass.

### Export Settings

#### `onnx_opset_version`
- **Type:** `integer`
- **Default:** `17`
- **Valid Range:** `11` to `20`
- **Description:** ONNX opset version for model export compatibility.
- **Note:** Lower versions = broader compatibility, higher versions = latest features.

---

## Pipeline Control

Master switches to enable/disable major processing stages.

### `generate_clips`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** Enable/disable the clip generation stage (TTS synthesis).
- **Example:**
  ```yaml
  generate_clips: true
  ```

### `transform_clips`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** Enable/disable feature extraction and augmentation stage.
- **⚠️ Important:** Set to `false` when not actively generating features to avoid infinite loops.

### `train_model`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** Enable/disable the training stage.

### `overwrite`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** Force regeneration of feature files, overwriting existing files.
- **⚠️ Warning:** Use with caution as it will delete existing computed features.

### `force_verify`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** Force re-verification of all data directories, ignoring cache.

### `show_training_summary`
- **Type:** `boolean`
- **Default:** `true`
- **Description:** Display effective training configuration in tabular format.

### `debug_mode`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** Enable debug logging and visualization outputs.

### `enable_journaling`
- **Type:** `boolean`
- **Default:** `true`
- **Description:** Log training metrics and model information to journal.

---

## Command-Line Arguments

Running training with configuration overrides:

```bash
# Basic training
python -m nanowakeword.trainer -c your_config_path.yaml 

# Generate + Transform + Train
python -m nanowakeword.trainer -c config.yaml -G -t -T

# Force re-verification of data
python -m nanowakeword.trainer -c config.yaml --force-verify

# Force regeneration of features
python -m nanowakeword.trainer -c config.yaml --overwrite

# Resume from previous training
python -m nanowakeword.trainer -c config.yaml --resume ./trained_models/my_model_v1

# Only transform (no generation, no training)
python -m nanowakeword.trainer -c config.yaml -t
```

### Arguments Explanation
- `-c, --config_path` - Path to YAML config file (required)
- `-G, --generate_clips` - Enable synthetic data generation stage
- `-t, --transform_clips` - Enable feature generation and augmentation
- `-T, --train_model` - Enable model training
- `-f, --force-verify` - Ignore cache and re-verify all data
- `--overwrite` - Regenerate all feature files (destructive)
- `--resume` - Resume training from specific model directory

---