
<p align="center">
  <img src="assets/logo/logo_0.png" alt="Logo" width="690">
</p>


# NanoWakeWord

### The Intelligent, One-Command Wake Word Model Trainer

**NanoWakeWord is a next-generation, fully automated framework for creating high-performance, custom wake word models. It's not just a tool; it's an intelligent engine that analyzes your data and crafts the perfect training strategy for you.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## Key Features

*   **Intelligent Auto-Configuration:** NanoWakeWord analyzes your dataset's size, quality, and balance, then automatically generates the optimal model architecture and hyperparameters. No more guesswork!
*   **One-Command Training:** Go from raw audio files (in any format) to a fully trained, production-ready model with a single command.
*   **Pro-active Data Harmonizer:** Automatically detects and fixes imbalances in your dataset by synthesizing high-quality positive and negative samples as needed.
*   **Automatic Pre-processing:** Just drop your raw audio files (MP3, M4A, FLAC, etc.) into the data folders. NanoWakeWord handles resampling, channel conversion, and format conversion automatically.
*   **Professional Terminal UI:** A clean, elegant, and informative command-line interface that makes the training process a pleasure to watch.
*   **Flexible & Controllable:** While highly automated, it provides full control to expert users through a clean `training_config.yaml` file.

## Getting Started

### Prerequisites

*   Python 3.8 or higher
*   Git
*   `ffmpeg` (for audio processing)

### Installation

Nanowakeword will be available on PyPI soon!

```bash
# Coming soon to PyPI!
pip install nanowakeword
```

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arcosoph/nanowakeword.git
    cd nanowakeword
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements_lock_3_13.txt
    ```
    
4.   **FFmpeg:** You must have FFmpeg installed on your system and available in your system's PATH. This is required for automatic audio preprocessing.
*  **On Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and follow their instructions to add it to your PATH.
*  **On macOS (using Homebrew):** `brew install ffmpeg`
*  **On Debian/Ubuntu:** `sudo apt update && sudo apt install ffmpeg`

## ⚙️ Usage

### Quick Start: The One-Command Magic

This is the recommended way for most users.

1.  **Prepare Your Data:** Place your raw audio files (in any format) in the respective subfolders inside `./training_data/` (`positive/`, `negative/`, `noise/`, `rir/`).

```
training_data/
├── positive/         # Contains examples of your wake word (e.g., "hey_nano.wav")
│   ├── sample1.wav
│   └── user_01.mp3
├── negative/         # Contains other speech/sounds that are NOT the wake word
│   ├── not_wakeword1.m4a
│   └── random_speech.wav
├── noise/            # Contains background noise files (e.g., fan, traffic sounds)
│   ├── cafe.flac
│   └── office_noise.aac
├── rir/              # (Optional but recommended) Contains Room Impulse Response files
│   ├── small_room.ogg
│   └── hall.wav
└── fp_val_data.npy   # (Optional) False positive validation data = long audio without wake words. Used to measure FP/hour.
```

2.  **Run the Trainer:** Execute the following command. The engine will handle everything else.

    ```bash
    python -m nanowakeword.train --training_config ./training_config.yaml --auto-config --generate_clips --augment_clips --train_model --overwrite
    ```

### Detailed Workflow

The command above performs the following steps automatically:

1.  **Data Pre-processing:** Converts all audio files in your data directories to the required format (16kHz, mono, WAV).
2.  **Intelligent Configuration (`--auto-config`):** Analyzes your dataset and generates an optimal training plan and hyperparameters.
3.  **Synthetic Data Generation (`--generate_clips`):** If the intelligent engine determines a data imbalance, it synthesizes new audio samples to create a robust dataset.
4.  **Augmentation & Feature Extraction (`--augment_clips`):** Creates thousands of augmented audio variations and extracts numerical features for training.
5.  **Model Training (`--train_model`):** Trains the model using the intelligently generated configuration on the prepared features.

### Command-Line Arguments

| Argument            | Description                                                                          |
| ------------------- | ------------------------------------------------------------------------------------ |
| `--training_config` | **Required.** Path to the base `.yaml` configuration file.                           |
| `--auto-config`     | Enables the intelligent engine to automatically determine the best hyperparameters.  |
| `--generate_clips`  | Activates the synthetic data generation step.                                        |
| `--augment_clips`   | Activates the data augmentation and feature extraction step.                         |
| `--train_model`     | Activates the final model training step.                                             |
| `--overwrite`       | If present, overwrites existing feature files during the augmentation step.          |

## Configuration (`training_config.yaml`)

The `training_config.yaml` file is the central control center. While `--auto-config` handles most settings, you must specify the essential paths.

```yaml
# Section 1: Essential Paths (User must fill this)
model_name: "my_wakeword_v1" #(REQUIRED)
output_dir: "./trained_models" #(REQUIRED)
wakeword_data_path: "./training_data/positive" #(REQUIRED)
# ... and other paths ...

# Section 2: Manual Training Configuration (Used when --auto-config is NOT present)
model_type: "lstm"     # Or other architectures such as `DNN` #(REQUIRED)
total_length: 32000
layer_size: 128
# ... and other manual settings ...
```
*For a full explanation of all parameters, please see the `training_config.yaml` file in the `examples` folder.*

## Performance and Evaluation

Nanowakeword is designed to produce high-accuracy models with excellent real-world performance. The models are trained to achieve a high recall rate while maintaining an extremely low number of false positives, making them reliable for always-on applications.

Below is a typical training performance graph for a model trained on a standard dataset using our `--auto-config` engine.

<!-- ![Training Performance Graph](assets/Graphs/training_performance_graph.png) -->
<h3>📈 Training Performance Graph</h3>

<p align="center">
  <img src="assets/Graphs/training_performance_graph.png" width="720">
</p>

### Key Performance Insights:

*   **Fast Convergence**: As shown in the "Validation Recall" graph, the model learns to detect the wake word very quickly, typically achieving over **80% recall within the first 15 validation steps**. This demonstrates the efficiency of the chosen model architecture and learning strategy.
*   **Low False Positive Rate**: Our training methodology heavily penalizes false positives. In a typical evaluation, a Nanowakeword model achieves an extremely low rate of false activations, often as low as **one false positive every 5-10 hours** on average (`False Positives per Hour: < 0.2`). This is crucial for a smooth user experience.
*   **High Accuracy and Recall**: While performance varies depending on the quality and quantity of the training data, a well-trained model consistently achieves:
    *   **Accuracy > 90%**: The model is correct in its predictions most of the time.
    *   **Recall > 70%**: The model is effective at detecting the wake word when it is spoken.

### The Role of the Intelligent Engine

The performance shown above is a direct result of the **Intelligent Configuration Engine**. For the dataset used in this example, the engine made the following key decisions:

*   **Adaptive Model Complexity**: It analyzed the dataset size and chose an appropriate 3-layer , complex enough to learn the patterns but not so large as to overfit.
*   **Optimized Training Duration**: Instead of a fixed number of steps, it calculated that ~18,000 steps would be optimal for this dataset's quality, saving training time.
*   **Balanced Batching**: It adjusted the training batch composition to include 18% `pure_noise`, as it detected sufficient background noise in the user-provided data, focusing more on differentiating the wake word from other speech.

This intelligent, data-driven approach is what allows Nanowakeword to consistently produce robust and reliable models.

## 📥 Pre-trained Models

To help you get started immediately, Nanowakeword provides a pre-trained, high-performance model ready for use. More community-requested models are also on the way!

### Available Now: "Arcosoph"
This is the official flagship model, developed and trained using Nanowakeword itself. It is highly accurate and serves as a perfect example of the quality you can achieve with this engine.

*   **Wake Word:** "Arcosoph" (pronounced *Ar-co-soph*)
*   **Performance:** Achieves a very low false-positive rate (less than one per 10 hours) while maintaining high accuracy.
*   **How to Use:** Download the model files from the [Hugging Face](https://huggingface.co/arcosoph/nanowakeword-lstm-base/tree/main).

### Coming Soon!
We are planning to release more pre-trained models for common wake words based on community feedback. Some of the planned models include:
*   "Hey Computer"
*   "Okay Nano"
*   "Jarvis"

Stay tuned for updates!

## ⚖️ Our Philosophy

In a world of complex machine learning tools, Nanowakeword is built on a simple philosophy:

1.  **Simplicity First**: You shouldn't need a Ph.D. in machine learning to train a high-quality wake word model. We believe in abstracting away the complexity.
2.  **Intelligence over Manual Labor**: The best hyperparameters are data-driven. Our goal is to replace hours of manual tuning with intelligent, automated analysis.
3.  **Performance on the Edge**: Wake word detection should be fast, efficient, and run anywhere. We focus on creating models that are small and optimized for devices like the Raspberry Pi.
4.  **Empowerment Through Open Source**: Everyone should have access to powerful voice technology. By being fully open-source, we empower developers and hobbyists to build the next generation of voice-enabled applications.

## FAQ

**1. Which Python version should I use?**

> The recommended Python version depends on your preferred output format for the trained model:

> * **For `.onnx` models:** You can use **Python 3.8 to 3.13**. This setup has been tested and is fully supported. A lock file for Python 3.13 (`requirements_lock_3_13.txt`) is provided for reference.

> * **For `.tflite` models:** Due to TensorFlow's dependency limitations, it is highly recommended to use versions below **Python 3.11>**. TensorFlow does not yet officially support Python versions newer than 3.11, so conversion to `.tflite` will fail.

**2. What kind of hardware do I need for training?**
> Training is best done on a machine with a dedicated `GPU`, as it can be computationally intensive. However, training on a `CPU` is also possible, although it will be slower. Inference (running the model) is very lightweight and can be run on almost any device, including a Raspberry Pi 3 or 4.

**3. How much data do I need to train a good model?**
> For a good starting point, we recommend at least 400+ clean recordings of your wake words from a few different voices. You can also create synthetic words using NanoWakeWord. The more data you have, the better your model will be. Our intelligent engine is designed to work well even with small datasets.

**4. Can I train a model for a language other than English?**
> Yes! NanoWakeWord is language-agnostic. As long as you can provide audio samples for your wake words, you can train a model for any language.

## Contributing

Contributions are welcome! If you have ideas for new features, bug fixes, or improvements to the "formula engine," please open an issue or submit a pull request.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

* This project stands on the shoulders of giants. It was initially inspired by the architecture and concepts of the `OpenWakeWord` project.
