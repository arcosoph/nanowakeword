# Nanowakeword: The Complete Configuration Guide

Welcome to the comprehensive guide for all configurable parameters in Nanowakeword. This document serves as the ultimate reference for advanced users, researchers, and developers who wish to fine-tune every aspect of the training pipeline.

**A Note on the Intelligent Engine:** You do not need to manually set most of these parameters. The cornerstone of Nanowakeword is its **Intelligent Configuration Engine**, which automatically analyzes your data and hardware to generate an optimized configuration. This guide is for those who wish to override those intelligent defaults for specific experiments or advanced control.

Any parameter listed here can be added to your `config.yaml` file to override the default or auto-generated value.

---

## Table of Contents

1.  [Core Project & Model Selection](#1-core-project--model-selection)
2.  [Data Sources & Pipeline Control](#2-data-sources--pipeline-control)
3.  [Synthetic Data Generation (TTS)](#3-synthetic-data-generation-tts)
4.  [Audio Processing & Feature Engineering](#4-audio-processing--feature-engineering)
5.  [Data Augmentation](#5-data-augmentation)
6.  [Model Architecture Specifics](#6-model-architecture-specifics)
7.  [Training, Optimization & Batching](#7-training-optimization--batching)
8.  [Loss Function Configuration](#8-loss-function-configuration)
9.  [Training Stability & Fault Tolerance](#9-training-stability--fault-tolerance)
10. [Export & Debugging](#10-export--debugging)

---

### 1. Core Project & Model Selection

These are the fundamental settings that define your project and the primary model architecture.

*   `model_type`
    *   **Description:** Specifies the neural network architecture to be trained. This is one of the most important choices you will make.
    *   **Type:** `string`
    *   **Default:** `"dnn"`
    *   <details>
        <summary><strong>(👉ﾟヮﾟ)👉 Click to see Guidance & Architecture Trade-offs</strong></summary>
        
        Choosing the right architecture depends on your specific goals for accuracy, speed, and noise robustness. Here is a guide to help you decide:

        *   **For Maximum Speed & Simplicity (`dnn`, `cnn`):**
            *   **`dnn`:** The fastest and most lightweight option. Its simple, fully-connected structure makes it ideal for microcontrollers and highly resource-constrained devices where every millisecond and kilobyte counts.
            *   **`cnn`:** Highly efficient at extracting local, time-invariant features. It excels with short, sharp, and explosive wake words (e.g., "Snap!").
        
        *   **For Superior Noise Robustness (`lstm`, `gru`, `crnn`):**
            *   **`lstm`:** The gold standard for sequential data. Its gating mechanism allows it to remember long-term dependencies, making it exceptionally good at filtering out background noise and understanding complex, multi-syllable phrases.
            *   **`gru`:** A modern, slightly faster alternative to LSTM with a simpler architecture. It offers a great balance between high noise robustness and computational efficiency.
            *   **`crnn`:** A powerful hybrid model. It first uses CNN layers to extract robust features and then feeds them into an RNN (LSTM/GRU) to model temporal context. This combination is extremely effective in very challenging audio environments.

        *   **For Modern Performance & Efficiency (`tcn`, `quartznet`):**
            *   **`tcn`:** A high-speed, parallelizable alternative to RNNs. It uses causal, dilated convolutions to capture long-range dependencies, often resulting in faster training and inference than LSTMs with competitive accuracy.
            *   **`quartznet`:** A highly parameter-efficient architecture from NVIDIA. It uses depthwise-separable convolutions to achieve top-tier accuracy with a very small model footprint, making it a perfect choice for powerful but lightweight edge deployment.

        *   **For State-of-the-Art Accuracy (`transformer`, `conformer`, `e_branchformer`):**
            *   **`transformer`:** The foundational attention-based model. Its self-attention mechanism allows it to weigh the importance of different parts of the audio simultaneously, giving it a deep "global" understanding of the entire utterance.
            *   **`conformer`:** The current state-of-the-art for many speech tasks. It brilliantly fuses the global context understanding of a Transformer with the local feature extraction of a CNN, achieving unmatched performance.
            *   **`e_branchformer`:** The bleeding-edge evolution of the Conformer. It processes attention and convolution in parallel branches, offering the highest potential accuracy for complex speech patterns.

        </details>
    *   **Example:** `model_type: "lstm"`

*   `model_name`
    *   **Description:** A unique name for your trained model. This name will be used for the project directory and the final exported files.
    *   **Type:** `string`
    *   **Default:** `Intelligently Generated` (e.g., `nww_lstm_model_v1`)
    *   **Example:** `model_name: "jarvis_v2"`

*   `output_dir`
    *   **Description:** The root directory where all trained models and their associated assets (features, checkpoints, graphs) will be saved.
    *   **Type:** `string` (path)
    *   **Default:** `"./trained_models"`

### 2. Data Sources & Pipeline Control

These parameters define where your data is located and which stages of the Nanowakeword pipeline should be executed.

*   `positive_data_path`, `negative_data_path`
    *   **Description:** Paths to the folders containing your positive (wake word) and negative (non-wake word) audio samples.
    *   **Type:** `string` (path)
    *   **Default:** `null` (Required)

*   `background_paths`, `rir_paths`
    *   **Description:** Lists of paths to folders containing background noise and Room Impulse Response (RIR) audio files, respectively. These are crucial for robust data augmentation.
    *   **Type:** `list` of `string` (paths)
    *   **Default:** `[]`

*   `generate_clips`, `transform_clips`, `train_model`
    *   **Description:** Boolean flags that control which major stages of the pipeline are executed. For a full run from scratch, all should be `true`.
    *   **Type:** `boolean`
    *   **Default:** `false`

### 3. Synthetic Data Generation (TTS)

Configure the built-in TTS engine to generate synthetic training data.

*   `target_phrase`
    *   **Description:** The wake word or phrase you want the TTS engine to generate.
    *   **Type:** `list` of `string`
    *   **Default:** `null`

*   `generate_positive_samples`, `generate_negative_samples`
    *   **Description:** The number of positive and negative audio samples to generate.
    *   **Type:** `integer`
    *   **Default:** `0`

*   `custom_negative_phrases`
    *   **Description:** A list of specific phrases to add to the negative set. This is highly effective for handling specific false-positive cases (e.g., words that sound similar to your wake word).
    *   **Type:** `list` of `string`
    *   **Default:** `[]`

*   `tts_batch_size`
    *   **Description:** The number of audio clips to generate in parallel during the TTS process.
    *   **Type:** `integer`
    *   **Default:** `Intelligently Generated` based on hardware.

### 4. Audio Processing & Feature Engineering

Control how raw audio is processed into numerical features.

*   `audio_processing.clip_length_samples`
    *   **Description:** Manually forces all training clips to a fixed length in samples (at 16kHz). If this is set, the autotune feature will be skipped.
    *   **Type:** `integer`
    *   **Default:** `null` (Autotune is used)

*   `audio_processing.autotune_length.*`
    *   **Description:** A group of parameters to control the automatic clip length detection. It is generally recommended to keep this enabled.
    *   `enabled`: `boolean`, default `true`.
    *   `duration_buffer_ms`: `integer`, default `750`. Extra padding added to the median clip length.
    *   `min_allowable_length`: `integer`, default `16000`. The absolute minimum clip length.

*   `overwrite`
    *   **Description:** If `true`, forces the regeneration of feature files, overwriting any existing ones. Use with caution as this can be time-consuming.
    *   **Type:** `boolean`
    *   **Default:** `false`

### 5. Data Augmentation

Fine-tune the on-the-fly data augmentation pipeline.

*   `augmentation_rounds`
    *   **Description:** The number of times the entire dataset is passed through the augmentation engine.
    *   **Type:** `integer`
    *   **Default:** `Intelligently Generated` (typically between 2 and 5).

*   `min_snr_in_db`, `max_snr_in_db`
    *   **Description:** The minimum and maximum Signal-to-Noise Ratio (in decibels) for mixing background noise. Lower values mean more challenging (noisier) audio.
    *   **Type:** `integer` or `float`
    *   **Default:** `Intelligently Generated`

*   `augmentation_settings.*`
    *   **Description:** A group of parameters to set the probability (from `0.0` to `1.0`) for applying specific augmentations.
    *   **Options:** `BackgroundNoise`, `RIR`, `PitchShift`, `BandStopFilter`, `ColoredNoise`, etc.
    *   **Default:** `Intelligently Generated`

### 6. Model Architecture Specifics

These parameters allow you to customize the internal structure of your chosen `model_type`.

*   `n_blocks`
    *   **Description:** A general-purpose parameter used to define the depth of many architectures (e.g., number of LSTM layers, Transformer layers, etc.).
    *   **Type:** `integer`
    *   **Default:** `Intelligently Generated`

*   `layer_size`
    *   **Description:** A general-purpose parameter for the width of many architectures (e.g., number of hidden units in an LSTM/GRU layer).
    *   **Type:** `integer`
    *   **Default:** `Intelligently Generated`

*   `embedding_dim`
    *   **Description:** The final dimension of the output embedding vector from the core model.
    *   **Type:** `integer`
    *   **Default:** `64`
    
*   `activation_function`
    *   **Description:** The activation function used in some architectures.
    *   **Type:** `string`
    *   **Default:** `"relu"`
    *   <details>
        <summary><strong>(👉ﾟヮﾟ)👉 Click to see Guidance & Trade-offs</strong></summary>

        *   `"relu"`: The standard, fast, and reliable choice. It's computationally cheap and works well in most cases.
        *   `"gelu"` and `"silu"` (also known as Swish): More modern, smoother functions that can sometimes lead to slightly better accuracy and faster convergence, especially in deeper, Transformer-based models. They come at a very minor computational cost.
        </details>
    *   **Example:** `activation_function: "silu"`

*   **Architecture-Specific Parameters:**
    *   `crnn_cnn_channels`: `list`, e.g., `[16, 32, 64]`
    *   `crnn_rnn_type`: `string`, e.g., `"lstm"` or `"gru"`
    *   `tcn_channels`: `list`, e.g., `[64, 128]`
    *   `tcn_kernel_size`: `integer`, e.g., `3`
    *   `quartznet_config`: `list` of `list`, e.g., `[[256, 33, 1], [512, 39, 1]]`
    *   `transformer_d_model`, `transformer_n_head`: `integer`
    *   `conformer_d_model`, `conformer_n_head`: `integer`
    *   `branchformer_d_model`, `branchformer_n_head`: `integer`

### 7. Training, Optimization & Batching

Configure the core training loop, optimizer, and how data is batched.

*   `steps`
    *   **Description:** The total number of training steps to perform.
    *   **Type:** `integer`
    *   **Default:** `Intelligently Generated`

*   `batch_composition.batch_size`
    *   **Description:** The number of samples in each training batch.
    *   **Type:** `integer`
    *   **Default:** `Intelligently Generated` based on hardware.
    
*   `batch_composition.source_distribution`
    *   **Description:** Defines the percentage of positive, negative speech, and pure noise samples in each batch. The sum must be 100.
    *   **Type:** `dict`
    *   **Default:** `Intelligently Generated`

*   `optimizer_type`
    *   **Description:** The optimization algorithm to use.
    *   **Type:** `string`
    *   **Default:** `"adamw"`
    *   <details>
        <summary><strong>(👉ﾟヮﾟ)👉 Click to see Guidance & Trade-offs</strong></summary>

        *   `"adamw"`: An improved version of the Adam optimizer with decoupled weight decay. It often leads to better model generalization and is the **recommended default** for most modern deep learning tasks.
        *   `"adam"`: The classic and highly effective adaptive optimizer. It's a very strong and reliable choice.
        *   `"sgd"`: Stochastic Gradient Descent. A foundational optimizer. While often slower to converge than adaptive optimizers, it can sometimes find better, more generalizable minima with careful tuning of the learning rate and momentum.
        </details>
    *   **Example:** `optimizer_type: "adamw"`

*   `learning_rate_max`, `learning_rate_base`
    *   **Description:** The maximum and base learning rates for schedulers like CyclicLR.
    *   **Type:** `float`
    *   **Default:** `Intelligently Generated`

*   `lr_scheduler_type`
    *   **Description:** The learning rate scheduler, which dynamically adjusts the learning rate during training.
    *   **Type:** `string`
    *   **Default:** `"cyclic"`
    *   <details>
        <summary><strong>(👉ﾟヮﾟ)👉 Click to see Guidance & Trade-offs</strong></summary>

        *   `"cyclic"` (CyclicLR): A powerful scheduler that cycles the learning rate between a base and max value. It's excellent for exploring the loss landscape and can help the model escape from local minima.
        *   `"onecycle"` (OneCycleLR): Another very powerful scheduler that is known for enabling "super-convergence" (achieving good results with much faster training). It's a great choice for many tasks.
        *   `"cosine"` (CosineAnnealingLR): A simple and effective scheduler that smoothly decreases the learning rate in a cosine curve. It is very predictable, robust, and a very popular choice in recent research.
        </details>
    *   **Example:** `lr_scheduler_type: "onecycle"`

### 8. Loss Function Configuration

Fine-tune the loss functions that guide the model's learning.

*   `classification_loss`
    *   **Description:** The primary classification loss function.
    *   **Type:** `string`
    *   **Default:** `"labelsmoothing"`
    *   <details>
        <summary><strong>(👉ﾟヮﾟ)👉 Click to see Guidance & Trade-offs</strong></summary>

        *   `"labelsmoothing"`: A robust default that improves generalization. It prevents the model from becoming overconfident by slightly "blurring" the hard 0 and 1 labels (e.g., to 0.1 and 0.9). This encourages the model to learn less extreme weights.
        *   `"focalloss"`: Specifically designed to handle class imbalance, which is very common in wake word datasets where negative samples vastly outnumber positive ones. It automatically down-weights easy-to-classify examples, forcing the model to focus its efforts on the harder, more ambiguous samples.
        *   `"bce"` (Binary Cross-Entropy): The standard, fundamental loss for binary classification tasks. It's a good baseline but can be sensitive to class imbalance and may not perform as well as the other two options without careful tuning.
        </details>
    *   **Example:** `classification_loss: "focalloss"`

*   `focal_loss_alpha`, `focal_loss_gamma`
    *   **Description:** Tuning parameters for Focal Loss. Only used if `classification_loss` is `"focalloss"`.
    *   **Type:** `float`
    *   **Default:** `alpha: 0.25`, `gamma: 2.0`

*   `label_smoothing`
    *   **Description:** The smoothing factor for Label Smoothing BCE loss.
    *   **Type:** `float`
    *   **Default:** `0.1`

*   `triplet_loss_margin`
    *   **Description:** The desired margin between positive and negative samples in the Triplet Loss embedding space. A larger margin forces a more distinct separation.
    *   **Type:** `float`
    *   **Default:** `0.2`

*   `loss_weight_triplet`, `loss_weight_class`
    *   **Description:** The weights to apply to the triplet and classification components of the final hybrid loss.
    *   **Type:** `float`
    *   **Default:** `triplet: 0.5`, `class: 1.0`

### 9. Training Stability & Fault Tolerance

Control mechanisms like early stopping and checkpointing.

*   `early_stopping_patience`
    *   **Description:** The number of steps without improvement in the stable (EMA) loss before training is stopped. Set to `0` or a negative value to disable.
    *   **Type:** `integer`
    *   **Default:** `Intelligently Generated` (or disabled for very short trainings).

*   `checkpointing.*`
    *   **Description:** A group of parameters to control the automatic checkpointing and resumption system.
    *   `enabled`: `boolean`, default `false`.
    *   `interval_steps`: `integer`, default `1000`.
    *   `limit`: `integer`, default `3`. (Number of recent checkpoints to keep).

*   `checkpoint_averaging_top_k`
    *   **Description:** The number of best-performing checkpoints to average together to create the final model.
    *   **Type:** `integer`
    *   **Default:** `5`

### 10. Export & Debugging

Settings related to the final model export and debugging during training.

*   `onnx_opset_version`
    *   **Description:** The ONNX opset version to use for exporting the model. Modern architectures (Transformer, Conformer) require a higher version (>=14).
    *   **Type:** `integer`
    *   **Default:** `17`

*   `debug_mode`
    *   **Description:** If `true`, enables verbose logging to a file during training, which is useful for debugging gradients and data shapes.
    *   **Type:** `boolean`
    *   **Default:** `false`
