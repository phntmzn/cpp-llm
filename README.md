# C++ LLM

**A GPT-style language model built with PyTorch to learn and generate C++ code.**

`cpp-llm` is an experimental project focused on building a language model for C++ source code. It provides a modular foundation for dataset preparation, tokenization, Transformer architecture, model training, inference, and evaluation.

The goal is to explore how large language models learn programming-language patterns and generate code using PyTorch.

> **Status:** Experimental development. The project structure is designed for training a small language model from scratch. Training and inference capabilities depend on the implementation of the corresponding modules.

---

## Features

- **PyTorch architecture** — Build a GPT-style decoder-only Transformer.
- **C++ datasets** — Organize `.cpp`, `.hpp`, `.h`, and other source files for training.
- **Tokenization** — Convert source code into token IDs.
- **Causal self-attention** — Learn relationships between tokens in source code.
- **Training pipeline** — Develop custom training loops and optimization routines.
- **Checkpoint management** — Save and load model weights.
- **Code generation** — Generate C++ text from a prompt.
- **Evaluation tools** — Measure model performance and track training progress.
- **Modular design** — Keep data processing, modeling, training, and inference separate.

---

## Project Structure

```text
cpp-llm/
│
├── data/
│   ├── raw/
│   │   ├── example.cpp
│   │   ├── example.hpp
│   │   └── sample.txt
│   │
│   └── processed/
│       ├── train.bin
│       └── val.bin
│
├── configs/
│   └── model.yaml
│
├── cpp_llm/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── prepare_data.py
│   │
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   └── tokenizer.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   ├── attention.py
│   │   ├── transformer_block.py
│   │   └── gpt.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── trainer.py
│   │   └── checkpoint.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── generate.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── evaluate.py
│
├── checkpoints/
│   └── .gitkeep
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_attention.py
│   ├── test_model.py
│   └── test_dataset.py
│
├── scripts/
│   ├── train.sh
│   └── generate.sh
│
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Requirements

- Python 3.10 or newer
- PyTorch
- NumPy
- pytest
- tiktoken
- PyYAML

Training performance depends on the model configuration, dataset size, and available hardware.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cpp-llm.git
cd cpp-llm
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your project includes a configured `pyproject.toml`, install it in editable mode:

```bash
pip install -e .
```

### 4. Verify PyTorch

```bash
python -c "import torch; print(torch.__version__)"
```

To check available hardware acceleration:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

CUDA is used on supported NVIDIA GPUs. MPS may be available on supported Apple Silicon Macs.

---

## Dataset Preparation

Place your C++ source files inside `data/raw/`.

Supported extensions may include:

- `.cpp`
- `.cc`
- `.cxx`
- `.c`
- `.h`
- `.hpp`
- `.hh`
- `.hxx`

Example:

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
```

Save the example as `data/raw/example.cpp`.

### Prepare the dataset

If the data-preparation module has been implemented, run:

```bash
python -m cpp_llm.data.prepare_data
```

The preprocessing pipeline should:

1. Discover C++ source files.
2. Read and normalize source text where appropriate.
3. Tokenize the source code.
4. Split data into training and validation sets.
5. Save processed data for model training.

The expected processed-data directory is:

```text
data/processed/
├── train.bin
└── val.bin
```

**Note:** Empty `.bin` files created as project placeholders are not valid training datasets. The preprocessing implementation must write actual token data before training can begin.

For meaningful evaluation, split datasets by repository or source file before creating training sequences to reduce data leakage.

---

## Model Architecture

The project uses a GPT-style, decoder-only Transformer architecture.

```text
C++ Source Code
       |
       v
    Tokenizer
       |
       v
    Token IDs
       |
       v
 Token Embeddings
       |
       v
Positional Embeddings
       |
       v
 Transformer Blocks
       |
       v
  Layer Normalization
       |
       v
 Output Projection
       |
       v
 Next-Token Prediction
       |
       v
  Generated C++ Code
```

### Core components

| Component | Responsibility |
|---|---|
| Tokenizer | Converts source code into token IDs |
| Embeddings | Represents tokens and their positions |
| Self-attention | Learns relationships between tokens |
| Transformer block | Combines attention and feed-forward layers |
| GPT model | Predicts the next token |
| Trainer | Optimizes model parameters |
| Checkpoint manager | Saves and restores model state |
| Generator | Produces text autoregressively |
| Evaluator | Measures model performance |

The model learns by predicting the next token in a sequence of C++ source code.

---

## Configuration

Model and training parameters are defined in:

```text
configs/model.yaml
```

Example configuration:

```yaml
model:
  block_size: 256
  n_embd: 256
  n_heads: 4
  n_layers: 4
  dropout: 0.1

training:
  batch_size: 8
  learning_rate: 0.0003
  max_steps: 5000
  eval_interval: 250
  seed: 42

data:
  corpus_path: data/processed/cpp_corpus.txt
  validation_fraction: 0.1

output:
  checkpoint_dir: checkpoints
```

### Configuration parameters

| Parameter | Description |
|---|---|
| `block_size` | Maximum input sequence length |
| `n_embd` | Embedding dimension |
| `n_heads` | Number of attention heads |
| `n_layers` | Number of Transformer blocks |
| `dropout` | Dropout probability |
| `batch_size` | Number of sequences per training batch |
| `learning_rate` | Optimizer learning rate |
| `max_steps` | Maximum training steps |
| `eval_interval` | Frequency of validation |

These are initial experimental values, not performance guarantees.

---

## Training

Once the training loop and preprocessing pipeline are implemented, the intended workflow is:

```bash
python -m cpp_llm.data.prepare_data
python -m cpp_llm.training.train
```

Alternatively, use the training shell script if it has been configured:

```bash
bash scripts/train.sh
```

The training process should:

1. Load the processed dataset.
2. Initialize the tokenizer and model.
3. Create training and validation data loaders.
4. Calculate next-token prediction loss.
5. Backpropagate gradients.
6. Update model parameters.
7. Evaluate validation loss.
8. Save model checkpoints.

Training from scratch requires sufficient source-code data and compute resources. A small experimental model is primarily intended for learning and architectural experimentation.

---

## Code Generation

After training and saving a compatible checkpoint, generate code using:

```bash
python -m cpp_llm.inference.generate \
  --checkpoint checkpoints/final.pt \
  --prompt "#include <iostream>\nint main() {\n" \
  --max-new-tokens 200
```

The model generates tokens autoregressively, using previously generated tokens as context.

Generated code may contain syntax errors, incomplete functions, or incorrect logic. Compile and test generated C++ before using it in a real project.

---

## Evaluation

Run the test suite:

```bash
pytest -q
```

The `tests/` directory contains tests for the tokenizer, attention mechanism, model, and dataset.

Model evaluation can also be extended to measure:

- Training and validation loss
- Perplexity
- C++ compilation success rate
- Unit-test pass rate
- Code completion quality
- Functional correctness

Lower validation loss indicates improved next-token prediction on the evaluation dataset, but does not guarantee correct or secure code generation.

---

## Checkpoints

Model checkpoints are stored in:

```text
checkpoints/
```

Example:

```text
checkpoints/
├── latest.pt
└── final.pt
```

Checkpoint files can become large. Avoid committing large model weights or datasets to Git unless that is intentional.

---

## Roadmap

- [x] Establish modular project structure
- [x] Implement initial Transformer components
- [x] Add tokenizer and dataset interfaces
- [ ] Complete binary dataset preprocessing
- [ ] Implement the full training loop
- [ ] Add training-resume support
- [ ] Implement robust inference and sampling controls
- [ ] Add repository-level dataset splitting
- [ ] Evaluate generated code with a C++ compiler
- [ ] Benchmark model performance
- [ ] Explore fine-tuning pretrained coding models

---

## Contributing

Contributions, bug reports, and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Submit a pull request describing the changes.

Please ensure that any training data you contribute can legally be redistributed and used for model training.

---

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Disclaimer

`cpp-llm` is an experimental machine-learning project intended for educational and research purposes. Generated code should be reviewed, compiled, and tested before use.

**Built with Python and PyTorch.**

**Before publishing:** Your current structure generator creates empty files. The README describes the intended workflow, but the training script and binary dataset pipeline must be implemented before the complete training process will work.
