# Contributing to NanoWakeWord (✿◕‿◕✿)

First off, thank you for considering contributing to NanoWakeWord! We are thrilled that you are interested in helping us build the most powerful, intelligent, and lightweight wake word engine available. Every contribution, from a small typo fix in the documentation to a major new feature, is deeply valued.

This document provides a clear roadmap for making the contribution process smooth and effective for everyone.

## How Can I Contribute?

There are many ways to make a meaningful impact on the project:

*   **🐛 Reporting Bugs:** If you encounter unexpected behavior or an error, please [open an issue](https://github.com/arcosoph/nanowakeword/issues). A detailed bug report is one of the most valuable contributions.
*   **✨ Proposing Enhancements:** Have a great idea for a new feature or an improvement to an existing one? We'd love to hear it. Open an issue to start a discussion.
*   **📄 Improving Documentation:** Clear documentation is crucial. If you find parts of our guides or docstrings confusing, please submit a pull request with your improvements.
*   **💻 Writing Code:** If you're ready to fix a bug or implement a new feature, we welcome pull requests with open arms.

## Ground Rules & Philosophy

Our goal is to maintain a high-quality, robust, and maintainable codebase. To that end:

*   **Code Style:** We use **Ruff** for lightning-fast code formatting and linting. Adhering to this style keeps the codebase clean and consistent.
*   **Commit Messages:** Write clear, concise, and professional commit messages. We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification (e.g., `feat:`, `fix:`, `docs:`).
*   **Focused Pull Requests:** Keep your pull requests focused on a single issue or feature. Small, atomic PRs are significantly easier and faster to review and merge.

## Your First Code Contribution

Ready to start writing code? Here’s how to set up your environment and submit your first pull request.

### Step 1: Fork and Clone the Repository

1.  **Fork** the `arcosoph/nanowakeword` repository on GitHub.
2.  **Clone** your personal fork to your local machine:
    ```bash
    git clone https://github.com/YOUR_USERNAME/nanowakeword.git
    cd nanowakeword
    ```

### Step 2: Set Up the Development Environment

We use a virtual environment to isolate project dependencies.

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

2.  **Activate the environment:**
    *   On Windows (PowerShell):
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install in Editable Mode with Training Dependencies:** This is the most important step. Installing in "editable" mode (`-e`) means that any changes you make to the local source code will be immediately reflected when you run the scripts. We also install the `[train]` extras, which include all dependencies needed for development and training.
    ```bash
    pip install -e ".[train]"
    ```
    *This command installs all the necessary libraries like PyTorch, `torch_audiomentations`, `rich`, etc., which are required to run the training engine.*

### Step 3: Make Your Changes

1.  **Create a new branch** for your work. Use a descriptive prefix like `feature/` or `fix/`.
    ```bash
    git checkout -b feature/my-awesome-new-feature
    ```
    or
    ```bash
    git checkout -b fix/resolve-config-proxy-bug
    ```

2.  **Write your code!** This is the creative part. Implement your feature or fix the bug.

### Step 4: Format and Lint Your Code

Before committing, please run our automated code quality tools. This ensures consistency across the entire project.

```bash
# This single command will format your code and fix many common linting issues.
ruff format . && ruff check . --fix```
```
### Step 5: Submit a Pull Request (PR)

1.  **Commit** your changes using the Conventional Commits format.
    ```bash
    git add .
    git commit -m "feat(training): Add support for FocalLoss in the training engine"
    ```

2.  **Push** your branch to your fork on GitHub.
    ```bash
    git push origin feature/my-awesome-new-feature
    ```

3.  **Open a Pull Request** from your fork to the `main` branch of the `arcosoph/nanowakeword` repository. In the description, clearly explain the "what" and "why" of your changes and link any relevant issues.

## Code of Conduct

All participants in this project are expected to adhere to the [NanoWakeWord Code of Conduct](CODE_OF_CONDUCT.md). Please ensure you are familiar with it.

Thank you again for your interest in making NanoWakeWord better. We are excited to see your contributions!