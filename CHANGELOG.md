# Changelog

All notable changes to this project will be documented in this file.

## 1.3.4 (unreleased)

- **Full information coming soon**


## [1.3.3] - 2025-11-14

### Added

-   **Massive Expansion of the Model Architecture Library:** Nanowakeword now supports a comprehensive suite of state-of-the-art neural network architectures, transforming it into a versatile and powerful speech modeling toolkit. The new additions include:
    -   **CRNN (Convolutional Recurrent Neural Network):** A robust hybrid model combining the strengths of CNNs and RNNs.
    -   **TCN (Temporal Convolutional Network):** A modern, high-speed alternative to RNNs for sequence modeling.
    -   **QuartzNet:** A highly parameter-efficient architecture for achieving top performance with a small model footprint.
    -   **Transformer:** The foundational attention-based model for deep contextual understanding.
    -   **Conformer & E-Branchformer:** State-of-the-art and bleeding-edge architectures that fuse Transformers and CNNs for ultimate performance.

### Fixed

-   **ONNX Export Failure for Modern Architectures:**
    -   Resolved a critical `RuntimeError` that prevented modern architectures (Transformer, Conformer, etc.) from being exported to the ONNX format.
    -   The issue was traced to an outdated default ONNX opset version. The framework now defaults to a modern, backward-compatible opset version (17), ensuring that all models, from classic to state-of-the-art, can be exported successfully.
-   **Model Averaging Error with BatchNorm:**
    -   Fixed a `RuntimeError` (`result type Float can't be cast to... Long`) that occurred during the checkpoint averaging process for models containing `BatchNorm` layers (e.g., CRNN).
    -   The `average_models` function has been improved to intelligently handle and average only floating-point parameters, ignoring integer-based counters like `num_batches_tracked`.
-   **TCN Model Initialization Bug:**
    -   Corrected a `TypeError` that prevented the TCN architecture from being initialized correctly due to an issue with `nn.Sequential` handling non-module `lambda` functions. The TCN block has been refactored for robust performance.

### Improved

-   **Intelligent Configuration Engine:** The `ConfigGenerator` has been refined for more realistic and efficient hyperparameter selection. The default maximum `augmentation_rounds` has been lowered to reduce feature computation time while maintaining high model quality.
-   **Codebase Structure & Maintainability:** In a major refactoring effort, all model architecture definitions have been moved from the monolithic `trainer.py` file into a dedicated, well-organized `architectures.py` file. This significantly improves code clarity, modularity, and future maintainability.

## [1.3.2] - 2025-11-08

### Added

-   **Powerful Training Resumption Capability**:
    -   Introduced a robust checkpointing system that allows users to resume interrupted training sessions seamlessly.
    -   Added a new `--resume <path_to_project>` command-line argument to load the latest checkpoint and continue training.
    -   New `checkpointing` section in `config.yaml` gives users full control over the feature, including enabling/disabling it, setting the save interval (`interval_steps`), and limiting the number of saved checkpoints (`limit`).
    -   Checkpoints now save the complete training state, including the model, optimizer, scheduler, step number, and loss history, ensuring a flawless recovery.

### Fixed

-   **Critical Bug in LSTM/GRU Training**:
    -   Resolved a critical `AttributeError` (`'LSTMModel' object has no attribute 'layer1'`) that caused the training process to crash when using `LSTM`, `GRU`, or `CNN` model architectures.
    -   The error was triggered by a debug code block that was not model-agnostic. The logic has been refactored to work reliably with all model types, making the training process stable across all architectures.


## [1.3.0] - 2025-11-5

This release marks a fundamental re-architecture of the NanoWakeWord trainer, transforming it from a static script into a transparent, scalable, and highly sophisticated training framework. The focus is on engineering excellence, providing unparalleled control, and producing state-of-the-art models ready for production environments.

### Key Features & Architectural Innovations

-   **Autonomous Training & Optimization Engine (`auto_train`)**
    We have pioneered a fully autonomous training process that goes beyond simple step counting. The engine leverages an Exponential Moving Average (EMA) of the loss to understand model stability in real-time. It intelligently identifies and saves only the best-performing model checkpoints, then averages their weights to produce a final model with superior generalization and robustness against overfitting. This is a self-reliant system that delivers peak performance with minimal supervision.

-   **Hyper-Flexible, Configuration-Driven Core (`ConfigProxy`)**
    At the heart of the new trainer is a revolutionary `ConfigProxy` engine. This system automatically discovers and makes every single training parameter—from nested optimizer settings and loss function weights to augmentation probabilities—controllable via a single `config.yaml` file. This transforms the trainer into a true framework, eliminating all "magic numbers" and providing you with absolute control over the entire architecture and training loop.

-   **Train on Virtually Unlimited Data with Memory-Mapping**
    We have shattered the limitations of system RAM. The new `mmap_batch_generator` allows you to train on massive, terabyte-scale feature sets as easily as you would with small datasets. Data is streamed efficiently from disk directly to the model, enabling you to build models on a scale that was previously impossible without enterprise-grade hardware.

-   **Live In-Terminal Training Dashboard**
    Experience a new level of insight and clarity. We have replaced static console outputs with a sophisticated, live-updating dashboard that renders directly in your terminal. It provides a real-time, comprehensive view of every active training parameter, while progress bars and logs flow cleanly beneath it. This professional UI keeps you fully informed without clutter.

-   **Strategic Batch Composition Engine**
    Gain expert-level control over what your model learns. The `batch_composition` feature allows you to strategically define the precise ratio of positive, negative speech, and pure noise samples in every single batch. This powerful tool enables you to directly address challenges like class imbalance or difficult false-positive cases by fine-tuning the data the model sees at each step.

-   **Production-Ready, Standardized ONNX Export**
    Deploy with absolute confidence. The export process now features a robust `InferenceWrapper` that guarantees the final ONNX model has a standardized, industry-compatible output shape of `[batch, 1, 1]`. This eliminates a common and frustrating source of downstream integration errors, ensuring your model works seamlessly in any production environment.

-   **Fully Configurable Optimizer and Scheduler Suite**:
    The choice of optimizer (`adamw`, `adam`, `sgd`) and all its related hyperparameters (e.g., `weight_decay`, `momentum`) are now exposed in the configuration, providing complete control over the model's optimization process.

-   **Advanced Hybrid Loss & Pluggable Architecture**:
    The model now learns highly discriminative features by optimizing a hybrid loss function, combining the strengths of Triplet Loss for embedding separation and a dedicated classification loss. The architecture is now pluggable, allowing users to select the optimal classification loss (`labelsmoothing`, `focalloss`, `bce`) and learning rate scheduler (`cyclic`, `onecycle`, `cosine`) directly from the configuration file.

### Changed / Improved

-   **Professional API & CLI**: Command-line arguments and configuration parameters have been standardized for maximum clarity and ease of use (e.g., `--config_path`, `transform_clips`).
-   **Organized Project Structure**: All generated assets are now saved to a clean, numbered directory structure (`1_features/`, `2_training_artifacts/`, `3_model/`) for intuitive project management.
-   **Intelligent Data Verification**: The pre-processing step now intelligently caches the state of your data directories, skipping redundant verification for unchanged files to save you valuable time.

### Removed

-   **Brittle TFLite & TensorFlow Dependencies**: We have removed all dependencies on TensorFlow and TFLite to create a lightweight, stable framework focused entirely on the universal and high-performance **ONNX Runtime**.
-   **Requirement for External Validation Sets**: The new autonomous training engine is entirely self-reliant and no longer requires a separate validation set to achieve optimal results, simplifying the workflow.
