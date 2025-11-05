
<p align="center">
  <img src="assets/logo/logo_0.png" alt="Logo" width="290">
</p>

# NanoWakeWord

### Next-Generation Wake Word Framework

**NanoWakeWord is a next-generation, adaptive framework designed to build high-performance, custom wake word models. More than just a tool, it’s an intelligent engine that understands your data and optimizes the entire training process to deliver exceptional accuracy and efficiency.**

[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-FFB000?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb)
[![Join the Discord](https://img.shields.io/badge/Join%20the%20Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/rYfShVvacB)
[![PyPI](https://img.shields.io/pypi/v/nanowakeword.svg?color=6C63FF&logo=pypi&logoColor=white)](https://pypi.org/project/nanowakeword/)
[![Python](https://img.shields.io/pypi/pyversions/nanowakeword.svg?color=3776AB&logo=python&logoColor=white)](https://pypi.org/project/nanowakeword/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/nanowakeword?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLACK&left_text=downloads)](https://pepy.tech/projects/nanowakeword)
[![License](https://img.shields.io/pypi/l/nanowakeword?color=white&logo=apache&logoColor=black)](https://pypi.org/project/nanowakeword/)
</div>

---

**Quick Access**
- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance-and-evaluation)
- [Using model](#using-your-trained-model-inference)
- [FAQ](#faq)

## ✨ **Build Your Pro Model**
Learn by doing. This Colab notebook walks you through each step to create and export your own optimized `.onnx` model—perfect for beginners and developers alike.

| Architecture | Recommended Use Case | Performance Profile | Start Training |
| :--- | :--- | :--- | :--- |
| **DNN** | Excellent for general use. Fast, lightweight, and reliable. | Fastest training, lowest resource usage. | [▶️ **With DNN**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=dnn) |
| **LSTM** | Ideal for noisy environments or complex, multi-syllable wake words. | Best-in-class noise robustness. | [▶️ **With LSTM**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=lstm) |
| **GRU** | A faster, lighter alternative to LSTM with similar high performance. | Great balance between robustness and speed. | [▶️ **With GRU**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=gru) |
| **CNN** | Effective for short, sharp, and explosive wake words. | Highly efficient at feature extraction. | [▶️ **With CNN**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=cnn) |
| **RNN** | A classic architecture, good for baseline experiments. | Simple recurrent structure. | [▶️ **With RNN**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=rnn) |

## Key Features

*   **Intelligent, Data-Driven Architecture with Full Override Capability:**
    The framework's core is an intelligent engine that analyzes your dataset to generate an optimal baseline architecture and a complete set of training hyperparameters. This provides a state-of-the-art starting point out-of-the-box. For advanced users and researchers, **every single parameter**—from model depth and optimizer choice to loss function weights and learning rate schedules—is exposed in the `.yaml` configuration, offering granular control for experimentation and fine-tuning.

*   **State-of-the-Art Hybrid Training Strategy:**
    We've implemented a powerful training methodology that combines **Triplet Loss** with a pluggable **Classification Loss** (`labelsmoothing`, `focalloss`, `bce`). This hybrid approach forces the model to learn highly discriminative and robust acoustic embeddings, which is critical for minimizing false activations in real-world scenarios. The entire training stack, including schedulers and optimizers, is modular, allowing for advanced customization.

*   **Train on Virtually Unlimited Data with Memory-Mapping:**
    Break the RAM barrier. NanoWakeWord's data pipeline is built on a memory-mapped generator that streams features directly from disk. This allows you to train on massive, terabyte-scale datasets as seamlessly as you would with smaller ones, enabling a level of scale typically reserved for enterprise-grade systems.

*   **Autonomous Optimization with Checkpoint Averaging:**
    The training process is more than a simple loop; it's an autonomous optimization system. It uses an Exponential Moving Average (EMA) of the loss to analyze model stability, intelligently triggers early stopping, and automatically averages the weights of the best-performing checkpoints. This produces a final model with superior generalization and resilience against overfitting.

*   **Transparent, Live Training Dashboard:**
    Gain complete transparency into the training process. The framework features a sophisticated, non-destructive terminal dashboard that provides a real-time, comprehensive view of every active parameter. This allows engineers and researchers to monitor the exact state of the system, from learning rates to batch composition, without interrupting the workflow.

*   **Strategic Batch Composition Engine:**
    Go beyond simple data loading. This powerful feature gives you expert-level control over the makeup of every single training batch. You can strategically define the precise ratio of positive, negative speech, and pure noise samples, allowing you to directly address specific training challenges like class imbalance or difficult false-positive cases.

*   **Automatic Pre-processing:**
     Just drop your raw audio files (MP3, M4A, FLAC, etc.) into the data folders. NanoWakeWord handles resampling, channel conversion, and format conversion automatically.


## Getting Started

### Prerequisites

*   Python 3.9 or higher
*   `ffmpeg` (for audio processing)

### Installation

Install the lightweight core package for **inference**:
```bash
pip install nanowakeword
```

To **train your own models**, install the full package with all training dependencies:
```bash
pip install "nanowakeword[train]"
```

**FFmpeg:** If you want to train your model you must have FFmpeg installed on your system and available in your system's PATH. This is required for automatic audio preprocessing.
*  **On Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and follow their instructions to add it to your PATH.
*  **On macOS (using Homebrew):** `brew install ffmpeg`
*  **On Debian/Ubuntu:** `sudo apt update && sudo apt install ffmpeg`


## Usage

The primary method for controlling the NanoWakeWord framework is through a `config.yaml` file. This file acts as the central hub for your entire project, defining data paths and controlling which pipeline stages are active.

### Simple Example Workflow

1.  **Prepare Your Data Structure:**
    Organize your raw audio files (`.mp3`, `.wav`, etc.) into their respective subfolders.
    ```
    training_data/
    ├── positive/         # Your wake word samples ("hey_nano.wav")
    │   ├── sample1.wav
    │   └── user_01.mp3
    ├── negative/         # Speech/sounds that are NOT the wake word
    │   ├── not_wakeword1.m4a
    │   └── random_speech.wav
    ├── noise/            # Background noises (fan, traffic, crowd)
    │   ├── cafe.wav
    │   └── office_noise.flac
    └── rir/              # Room Impulse Response files
        ├── small_room.wav
        └── hall.wav
    ```

2.  **Define Your Configuration:**
    Create a `config.yaml` file to manage your training pipeline. This approach ensures your experiments are repeatable and well-documented.
    ```yaml
    # In your config.yaml
    # Essential Paths (Required)
    model_type: dnn # Or other architectures such as `LSTM`, `GRU`, `RNN` etc
    model_name: "my_wakeword_v1"
    output_dir: "./trained_models"
    positive_data_path: "./training_data/positive"
    negative_data_path: "./training_data/negative"
    background_paths:
    - "./training_data/noise"
    rir_paths:
    - "./training_data/rir"
    
    # Enable the stages for a full run
    generate_clips: true
    transform_clips: true
    train_model: true

    # Add more setting (Optional)
    # For example, to apply a specific set of parameters:
    n_blocks: 3      
    layer_size: 256
    classification_loss: labelsmoothing
    ```
*For a full explanation of all parameters, please see the [`training_config.yaml`](https://github.com/arcosoph/nanowakeword/blob/main/examples/training_config.yaml) or [`train_config_full.yaml`](https://github.com/arcosoph/nanowakeword/blob/main/examples/train_config_full.yaml) file in the `examples` folder.*


3.  **Execute the Pipeline:**
    Launch the trainer by pointing it to your configuration file. The stages enabled in your config will run automatically.
    ```bash
    nanowakeword-train -c ./path/to/config.yaml
    ```

### Command-Line Arguments (Overrides)

For on-the-fly experiments or to temporarily modify your pipeline without editing your configuration file, you can use the following command-line arguments. **Any flag used will take precedence over the corresponding setting in your `config.yaml` file.**

| Argument            | Shorthand                 | Description                                                                                             |
| ------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------- |
| `--config_path`     | `-c`                      | **Required.** Path to the base `.yaml` configuration file.                                              |
| `--generate_clips`  | `-G`                      | Activates the 'Generation' stage.                                                                       |
| `--transform_clips` | `-t`                      | Activates the preparatory 'transform' stage (augmentation and feature extraction).                      |
| `--train_model`     | `-T`                      | Activates the final 'Training' stage to build the model.                                                |
| `--force-verify`    | `-f`                      | Forces re-verification of all data directories, ignoring the cache.                                     |
| `--overwrite`       | *(none by design)*       | Forces regeneration of feature files. **Use with caution as this deletes existing data.**                 |

### The Intelligent Workflow

The command above automates a sophisticated, multi-stage pipeline:

1.  **Data Verification & Pre-processing:** Scans and converts all audio to a standardized format (16kHz, mono, WAV).
2.  **Intelligent Configuration:** Analyzes the dataset to generate an optimal model architecture and training hyperparameters.
3.  **Synthetic Data Generation:** If the engine detects a data imbalance, it synthesizes new audio samples to create a robust dataset.
4.  **Augmentation & Feature Extraction:** Creates thousands of augmented audio variations and extracts numerical features, saving them in a memory-efficient format.
5.  **Autonomous Model Training:** Trains the model using the intelligently generated configuration, automatically stopping when peak performance is reached.
6.  **Checkpoint Averaging & Export:** Averages the weights of the most stable models found during training and exports a final, production-ready `.onnx` file.

## Performance and Evaluation

Nanowakeword is engineered to produce state-of-the-art, highly accurate models with exceptional real-world performance. The new dual-loss training architecture, combined with our powerful Intelligent Configuration Engine, ensures models achieve a very low stable loss while maintaining a clear separation between positive and negative predictions. This makes them extremely reliable for always-on, resource-constrained applications.

Below is a typical training performance graph for a model trained on a standard dataset. This entire process, from hyperparameter selection to training duration, is managed automatically by Nanowakeword's core engine.

### 📈 Training Performance Graph

<p align="center">
  <img src="assets/Graphs/training_performance_graph.png" width="600">
</p>

### Key Performance Insights:

*   **Stable and Efficient Learning:** The "Training Loss (Stable/EMA)" curve demonstrates the model's rapid and stable convergence. The loss consistently decreases and flattens, indicating that the model has effectively learned the underlying patterns of the wake word without overfitting. The raw loss (light blue) shows the natural variance between batches, while the stable loss (dark blue) confirms a solid and reliable learning trend.

*   **Intelligent Early Stopping:** The training process is not just powerful but also efficient. In this example, the process was scheduled for **18,109 steps** but was intelligently halted at **11,799 steps** by the early stopping mechanism. This feature saved significant time and computational resources by automatically detecting the point of maximum learning, preventing hours of unnecessary training.

*   **Exceptional Confidence and Separation:** The final report card is a testament to the model's quality. With an **Average Stable Loss of just 0.2065**, the model is highly accurate. More importantly, the high margin between the positive and negative confidence scores highlights its decision-making power:
    *   **Avg. Positive Confidence (Logit): `3.166`** (Extremely confident when the wake word is spoken)
    *   **Avg. Negative Confidence (Logit): `-3.137`** (Equally confident in rejecting incorrect words and noise)
    This large separation is crucial for minimizing false activations and ensuring the model responds only when it should.

*   **Extremely Low False Positive Rate:** While real-world performance depends on the environment, our new training methodology, which heavily penalizes misclassifications, produces models with an exceptionally low rate of false activations. A well-trained model often achieves **less than one false positive every 8-12 hours** on average, making it ideal for a seamless user experience.

### The Role of the Intelligent Configuration Engine

The outstanding performance shown above is a direct result of the data-driven decisions made automatically by the Intelligent Configuration Engine. For the dataset used in this example, the engine made the following critical choices:

*   **Adaptive Model Complexity:** It analyzed the 2.6 hours of effective data volume (after augmentation) and determined that an **3 blocks and a layer size of 256** (`model_complexity_score: 2.64`) would be optimal. This provided enough capacity to learn complex temporal patterns without being excessive for the dataset size.
*   **Data-Driven Augmentation Strategy:** Based on the high amount of noise and reverberation data provided (`H_noise: 5.06`, `N_rir: 1668`), it set aggressive augmentation probabilities (`RIR: 0.8`, `background_noise_probability: 0.9`) to ensure the model would be robust in challenging real-world environments.
*   **Balanced Batch Composition:** It intelligently adjusted the training batch to include **27% `pure_noise`**. This decision was based on its analysis of the user-provided data, allowing the model to focus more on differentiating the wake word from both ambient noise and other human speech (`negative_speech: 44%`).

This intelligent, automated, and data-centric approach is the core of Nanowakeword, enabling it to consistently produce robust, efficient, and highly reliable wake-word detection models without requiring manual tuning from the user.

## Using Your Trained Model (Inference)

Once you have your `model.onnx` file, using it for inference is simple. Here’s a minimal example in Python using the `nanowakeword` librarie.

```python
import pyaudio
import numpy as np
import os
import sys
from nanowakeword.nanointerpreter import NanoInterpreter 
import time
# --- Simple Configuration ---
MODEL_PATH = r"trained_models\hey_computer_v1\3_model\hey_computer_v1.onnx"
THRESHOLD = 0.9  # A simple threshold for detection
COOLDOWN = 2     # A simple cooldown managed outside the interpreter
# If you want, you can use more advanced methods like VAD or PATIENCE_FRAMES.

# --- Initialization ---
if not os.path.exists(MODEL_PATH):
    sys.exit(f"Error: Model not found at '{MODEL_PATH}'")

try:
    print(" Initializing NanoInterpreter (Simple Mode)...")
    
    # Load the model with NO advanced features.
    interpreter = NanoInterpreter.load_model(MODEL_PATH)
    
    key = list(interpreter.models.keys())[0]
    print(f" Interpreter ready. Listening for '{key}'...")

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)

    last_detection_time = 0
    
    # --- Main Loop ---
    while True:
        audio_chunk = np.frombuffer(stream.read(1280, exception_on_overflow=False), dtype=np.int16)
        
        # Call predict with NO advanced parameters.
        score = interpreter.predict(audio_chunk).get(key, 0.0)

        # The detection logic is simple and external.
        current_time = time.time()
        if score > THRESHOLD and (current_time - last_detection_time > COOLDOWN):
            print(f"🎯 Detected '{key}'! (Score: {score:.2f})")
            last_detection_time = current_time
            interpreter.reset()
        else:
            print(f"Score: {score:.3f}", end='\r', flush=True)

except KeyboardInterrupt:
    print("")
```


## 🎙️ Pre-trained Models

To help you get started quickly, `nanowakeword` comes with a rich collection of pre-trained models. These pre-trained models are ready to use and support a wide variety of wake words, eliminating the need to spend time training your own model from scratch.

Because our library of models is constantly evolving with new additions and improvements, we maintain a live, up-to-date list directly on our GitHub project page. This ensures you always have access to the latest information.

For a comprehensive list of all available models and their descriptions, please visit the official model registry:

**[View the Official List of Pre-trained Models (✿◕‿◕✿)](https://huggingface.co/arcosoph/nanowakeword-models#pre-trained-models)**


## ⚖️ Our Philosophy

In a world of complex machine learning tools, Nanowakeword is built on a simple philosophy:

1.  **Simplicity First**: You shouldn't need a Ph.D. in machine learning to train a high-quality wake word model. We believe in abstracting away the complexity.
2.  **Intelligence over Manual Labor**: The best hyperparameters are data-driven. Our goal is to replace hours of manual tuning with intelligent, automated analysis.
3.  **Performance on the Edge**: Wake word detection should be fast, efficient, and run anywhere. We focus on creating models that are small and optimized for devices like the Raspberry Pi.
4.  **Empowerment Through Open Source**: Everyone should have access to powerful voice technology. By being fully open-source, we empower developers and hobbyists to build the next generation of voice-enabled applications.

## FAQ

**1. Which Python version should I use?**

>  You can use **Python 3.8 to 3.13**. This setup has been tested and is fully supported.

**2. What kind of hardware do I need for training?**
> Training is best done on a machine with a dedicated `GPU`, as it can be computationally intensive. However, training on a `CPU` is also possible, although it will be slower. Inference (running the model) is very lightweight and can be run on almost any device, including a Raspberry Pi 3 or 4, etc.

**3. How much data do I need to train a good model?**
> For a good starting point, we recommend at least 400+ clean recordings of your wake words from a few different voices. The total duration of negative audio should be at least 3 times longer than positive audio. You can also create synthetic words using NanoWakeWord. The more data you have, the better your model will be. Our intelligent engine is designed to work well even with small datasets.

**4. Can I train a model for a language other than English?**
> Yes! NanoWakeWord is language-agnostic. As long as you can provide audio samples for your wake words, you can train a model for any language.

**5. Which version of Nanowakeword should I use?**
> Always use the latest version of Nanowakeword. Version v1.3.0 is the minimum supported, but using the latest ensures full compatibility and best performance.

## Roadmap

NanoWakeWord is an actively developed project. Here are some of the features and improvements we are planning for the future:

-   **Model Quantization:** Tools to automatically quantize the final `.onnx` model for even better performance on edge devices.
-   **Advanced Augmentation:** Adding more audio augmentation techniques like SpecAugment.
-   **Model Zoo Expansion:** Adding more pre-trained models for different languages and phrases.
-   **Performance Benchmarks:** A dedicated section with benchmarks on popular hardware like Raspberry Pi.

We welcome feedback and contributions to help shape the future of this project!

## Contributing

Contributions are the lifeblood of open source. We welcome contributions of all forms, from bug reports and documentation improvements to new features.

To get started, please see our **[Contribution Guide](CONTRIBUTING.md)**, which includes information on setting up a development environment, running tests, and our code of conduct.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
