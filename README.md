
<p align="center">
  <img src="https://raw.githubusercontent.com/arcosoph/nanowakeword/main/assets/logo/logo_0.png" alt="Logo" width="290">
</p>

<p align="center">
    <a href="https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb"><img alt="Open In Colab" src="https://img.shields.io/badge/Open%20in%20Colab-FFB000?logo=googlecolab&logoColor=white"></a>
    <a href="https://discord.gg/rYfShVvacB"><img alt="Join the Discord" src="https://img.shields.io/badge/Join%20the%20Discord-5865F2?logo=discord&logoColor=white"></a>
    <a href="https://pypi.org/project/nanowakeword/"><img alt="PyPI" src="https://img.shields.io/pypi/v/nanowakeword.svg?color=6C63FF&logo=pypi&logoColor=white"></a>
    <a href="https://pypi.org/project/nanowakeword/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/nanowakeword.svg?color=3776AB&logo=python&logoColor=white"></a>
    <a href="https://pepy.tech/projects/nanowakeword"><img alt="PyPI Downloads" src="https://static.pepy.tech/personalized-badge/nanowakeword?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLACK&left_text=downloads"></a>
    <a href="https://pypi.org/project/nanowakeword/"><img alt="License" src="https://img.shields.io/pypi/l/nanowakeword?color=white&logo=apache&logoColor=black"></a>
</p>

**NanoWakeWord is a next-generation, adaptive framework designed to build high-performance, custom wake word models. More than just a tool, it’s an intelligent engine that understands your data and optimizes the entire training process to deliver exceptional accuracy and efficiency.**

**Quick Access**
- [Installation](https://github.com/arcosoph/nanowakeword?tab=readme-ov-file#installation)
- [Usage](https://github.com/arcosoph/nanowakeword?tab=readme-ov-file#usage)
- [Performance](https://github.com/arcosoph/nanowakeword?tab=readme-ov-file#performance-and-evaluation)
- [Using model](https://github.com/arcosoph/nanowakeword?tab=readme-ov-file#using-your-trained-model-inference)
- [Features](https://github.com/arcosoph/nanowakeword?tab=readme-ov-file#state-of-the-art-features-and-architecture)
- [FAQ](https://github.com/arcosoph/nanowakeword?tab=readme-ov-file#faq)

## ✨ **Choose Your Architecture, Build Your Pro Model**
NanoWakeWord is a versatile framework offering a rich library of neural network architectures. Each is optimized for different scenarios, allowing you to build the perfect model for your specific needs. This Colab notebook lets you experiment with any of them.

| Architecture | Recommended Use Case | Performance Profile | Start Training |
| :--- | :--- | :--- | :--- |
| **DNN** | General use on resource-constrained devices (e.g., MCUs). | **Fastest Training, Low Memory** | [▶️ **Launch**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=dnn) |
| **CNN** | Short, sharp, and explosive wake words. | Efficient Feature Extraction | [▶️ **Launch**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=cnn) |
| **LSTM** | Noisy environments or complex, multi-syllable phrases. | **Best-in-Class Noise Robustness** | [▶️ **Launch**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=lstm) |
| **GRU** | A faster, lighter alternative to LSTM with similar high performance. | Balanced: Speed & Robustness | [▶️ **Launch**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=gru) |
| **CRNN** | Challenging audio requiring both feature and context analysis. | Hybrid Power: CNN + RNN | [▶️ **Launch**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=crnn) |
| **TCN** | Modern, high-speed sequential processing. | **Faster than RNNs** (Parallel) | [▶️ **Launch**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=tcn) |
| **QuartzNet**| Top accuracy with a small footprint on edge devices. | **Parameter-Efficient & Accurate** | [▶️ **Launch**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=quartznet) |
| **Transformer**| **Deep Contextual Understanding** via Self-Attention mechanism. | **SOTA Performance & Flexibility** | [▶️ **Launch**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=transformer) |
| **Conformer** | State-of-the-art hybrid for ultimate real-world performance. | **SOTA: Global + Local Features** | [▶️ **Launch**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=conformer) |
| **E-Branchformer**| Bleeding-edge research for potentially the highest accuracy. | Peak Accuracy Potential | [▶️ **Launch**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=e_branchformer) |
| **RNN** | Baseline experiments or educational purposes. | Simple & Foundational | [▶️ **Launch**](https://colab.research.google.com/github/arcosoph/nanowakeword/blob/main/notebooks/Train_Your_First_Wake_Word_Model.ipynb?model_type=rnn) |

## State-of-the-Art Features and Architecture

Nanowakeword is not merely a tool; it's a holistic, end-to-end ecosystem engineered to democratize the creation of state-of-the-art, custom wake word models. It moves beyond simple scripting by integrating a series of automated, production-grade systems that orchestrate the entire lifecycle—from data analysis and feature engineering to advanced training and deployment-optimized inference.

<details>
<summary><strong>1. Automated ML Engineering for Peak Performance</strong></summary>

The cornerstone and "brain" of the framework is its data-driven configuration engine. This system performs a holistic analysis of your unique dataset and hardware environment to replace hours of manual, error-prone hyper-parameter tuning with a single, intelligent process. It crafts a powerful, optimized training baseline by synergistically determining:

*   **Adaptive Architectural Scaling:** It doesn't just use a fixed architecture; it sculpts one for you. The engine dynamically scales the model's complexity—tuning its depth, width, and regularization (e.g., layers, neurons, dropout) to perfectly match the volume and complexity of your training data. This core function is critical for preventing both underfitting on small datasets and overfitting on large ones.

*   **Optimized Training & Convergence Strategy:** Based on data characteristics, it formulates a multi-stage, dynamic learning rate schedule and determines the precise training duration required to reach optimal convergence. This ensures the model is trained to its full potential without wasting computational resources on diminishing returns.

*   **Hardware-Aware Performance Tuning:** The engine profiles your entire hardware stack (CPU cores, system RAM, and GPU VRAM) to maximize throughput at every stage. It calculates the maximum efficient batch sizes for data generation, augmentation, and model training, ensuring that your hardware's full potential is unlocked.

*   **Automatic Pre-processing:** Just drop your raw audio files (`.mp3`, `.m4a`, `.flac`, etc.) into the data folders — NanoWakeWord automatically handles resampling, channel conversion, and format standardization.

*   **Data-Driven Augmentation Policy:** Rather than applying a generic augmentation strategy, the engine crafts a custom augmentation policy. It analyzes the statistical properties of your provided noise and reverberation files to tailor the intensity, probability, and type of on-the-fly augmentations, creating a training environment that mirrors real-world challenges.

While this engine provides a state-of-the-art baseline, it does not sacrifice flexibility. **Advanced users retain full, granular control and can override any of the dozens of automatically generated parameters by simply specifying their desired value in the `.yaml` file.**

</details>

<details>
<summary><strong>2. The Production-Grade Data Pipeline: From Raw Audio to Optimized Features</strong></summary>

Recognizing that data is the bedrock of any great model, Nanowakeword automates the entire data engineering lifecycle with a pipeline designed for scale and quality:

*   **Phonetic Adversarial Negative Generation:** This is a key differentiator. The system moves beyond generic noise and random words by performing a phonetic analysis of your wake word. It then synthesizes acoustically confusing counter-examples—phrases that sound similar but are semantically different. This forces the model to learn fine-grained phonetic boundaries, dramatically reducing the false positive rate in real-world use.

*   **Dynamic On-the-Fly Augmentation:** During training, a powerful augmentation engine injects a rich tapestry of real-world acoustic scenarios in real-time. This includes applying background noise at varying SNR levels, convolving clips with room impulse responses (RIR) for realistic reverberation, and applying a suite of other transformations like pitch shifting and filtering.

*   **Seamless Large-Scale Data Handling (`mmap`):** The framework shatters the memory ceiling of conventional training scripts. By utilizing memory-mapped files, it streams features directly from disk, enabling seamless training on datasets that can be hundreds of gigabytes or even terabytes in size, all on standard consumer hardware.

</details>

<details>
<summary><strong>3. A Modern Training Paradigm: State-of-the-Art Optimization Techniques</strong></summary>

The training process itself is infused with cutting-edge techniques to ensure the final model is not just accurate, but exceptionally robust and reliable:

*   **Hybrid Loss Architecture:** The model's learning is guided by a sophisticated, dual-objective loss function. **Triplet Loss** sculpts a highly discriminative embedding space, pushing dissimilar sounds far apart. Simultaneously, a **Classification Loss** (such as Focal Loss or Label Smoothing) fine-tunes the final decision boundary for raw accuracy. These two losses work in concert to produce models with superior discrimination capabilities.

*   **Checkpoint Ensembling / Stochastic Weight Averaging (SWA):** Instead of relying on a single "best" checkpoint, the framework identifies and averages the weights of the most stable and high-performing models from the training run. This powerful ensembling technique finds a flatter, more robust minimum in the loss landscape, leading to a final model with provably better generalization to unseen data.

*   **Resilient, Fault-Tolerant Workflow:** Long training sessions are protected. The framework automatically saves the entire training state—model weights, optimizer progress, scheduler state, and even the precise position of the data generator. This allows you to resume an interrupted session from the exact point you left off, ensuring zero progress is lost.

*   **Transparent Live Dashboard:** A clean, dynamic terminal table provides a real-time, transparent view of all effective training parameters as they are being used, offering complete insight into the automated process.

</details>

<details>
<summary><strong>4. The Deployment-Optimized Inference Engine: High Performance on the Edge</strong></summary>

A model's true value is in its deployment. Nanowakeword's inference engine is designed from the ground up for efficiency, low latency, and the challenges of real-world deployment:

*   **Stateful Streaming Architecture:** It processes continuous audio streams incrementally, maintaining temporal context via hidden states for recurrent models (like LSTMs/GRUs). This is essential for delivering instant, low-latency predictions in real-time applications.

*   **Universal ONNX Export:** The final trained model is exported to the industry-standard ONNX format. This guarantees maximum hardware acceleration and platform-agnostic deployment across a vast range of environments, from powerful servers to resource-constrained edge devices.

*   **Integrated On-Device Post-Processing Stack:** The engine is a complete, production-ready solution. It incorporates an on-device stack that includes optional **Voice Activity Detection (VAD)** to conserve power, **Noise Reduction** to enhance clarity, and intelligent **Debouncing/Patience Filters**. This stack transforms the raw model output into a reliable, robust trigger, ready for integration out of the box.

</details>



## Getting Started

### Prerequisites

*   Python 3.9 or higher
*   `ffmpeg` (for audio processing)

### Installation

Install the latest stable version from PyPI for **inference**:
```bash
pip install nanowakeword
```

To **train your own models**, install the full package with all training dependencies:
```bash
pip install "nanowakeword[train]"
```
**Pro-Tip: Bleeding-Edge Updates**  
While the PyPI package offers the latest stable release, you can install the most up-to-the-minute version directly from GitHub to get access to new features and fixes before they are officially released:
```bash
pip install git+https://github.com/arcosoph/nanowakeword.git
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
    model_type: dnn # Or other architectures such as `LSTM`, `GRU`, `RNN`, `Transformer` etc..
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
    # ...
    classification_loss: labelsmoothing
    # ...
    checkpointing:
      enabled: true
      interval_steps: 500
      limit: 3
    # Other...
    ```
*For a full explanation of all parameters, please see the [`training_config`](https://github.com/arcosoph/nanowakeword/blob/main/examples/training_config.yaml) or [`CONFIGURATION_GUIDE`](https://github.com/arcosoph/nanowakeword/blob/main/CONFIGURATION_GUIDE.md).*


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
| `--resume`          | *(none)*                  | Resumes training from the latest checkpoint in the specified project directory.                         |
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
  <img src="https://raw.githubusercontent.com/arcosoph/nanowakeword/main/assets/Graphs/training_performance_graph.png" width="600">
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

Your trained `.onnx` model is ready for action! The easiest and most powerful way to run inference is with our lightweight `NanoInterpreter` class. It's designed for high performance and requires minimal code to get started.

Here’s a practical example of how to use it:

```python
import pyaudio
import numpy as np
import os
import sys
import time
# Import the interpreter class from the library
from nanowakeword.nanointerpreter import NanoInterpreter 
# --- Simple Configuration ---
MODEL_PATH = r"model/path/your.onnx"
THRESHOLD = 0.9  # A simple threshold for detection
COOLDOWN = 2     # A simple cooldown managed outside the interpreter
# If you want, you can use more advanced methods like VAD or PATIENCE_FRAMES.

# Initialization 
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
    
    # Main Loop 
    while True:
        audio_chunk = np.frombuffer(stream.read(1280, exception_on_overflow=False), dtype=np.int16)
        
        # Call predict with NO advanced parameters.
        score = interpreter.predict(audio_chunk).get(key, 0.0)

        # The detection logic is simple and external.
        current_time = time.time()
        if score > THRESHOLD and (current_time - last_detection_time > COOLDOWN):
            print(f"Detected '{key}'! (Score: {score:.2f})")
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

To get started, please see our **[Contribution Guide](https://github.com/arcosoph/nanowakeword/blob/main/CONTRIBUTING.md)**, which includes information on setting up a development environment, running tests, and our code of conduct.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/arcosoph/nanowakeword/blob/main/LICENSE) file for details.
