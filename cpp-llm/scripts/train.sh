#!/usr/bin/env bash

set -Eeuo pipefail

# Move to the project root, regardless of the current directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration.
VENV_DIR="${PROJECT_ROOT}/.venv"
CONFIG_FILE="${PROJECT_ROOT}/configs/model.yaml"
CHECKPOINT_FILE="${PROJECT_ROOT}/checkpoints/latest.pt"

echo "======================================"
echo "          cpp-llm Training"
echo "======================================"
echo "Project: ${PROJECT_ROOT}"

# Check Python.
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

# Create the virtual environment if needed.
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Python: $(python --version)"

# Verify the configuration exists.
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: ${CONFIG_FILE}"
    exit 1
fi

# Install dependencies if PyTorch is unavailable.
if ! python -c "import torch, yaml, numpy, tiktoken" >/dev/null 2>&1; then
    echo "Installing project dependencies..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi

# Ensure training data exists and is not empty.
for data_file in \
    "${PROJECT_ROOT}/data/processed/train.bin" \
    "${PROJECT_ROOT}/data/processed/val.bin"
do
    if [[ ! -s "$data_file" ]]; then
        echo "Error: Training data is missing or empty:"
        echo "  $data_file"
        echo "Prepare the tokenized datasets before training."
        exit 1
    fi
done

# Create the checkpoint directory.
mkdir -p "${PROJECT_ROOT}/checkpoints"

# Parse command-line arguments.
RESUME=false

for arg in "$@"; do
    case "$arg" in
        --resume)
            RESUME=true
            ;;
    esac
done

# Start training.
echo ""
echo "Starting cpp-llm training..."

if [[ "$RESUME" == true ]]; then
    if [[ ! -s "$CHECKPOINT_FILE" ]]; then
        echo "Error: Cannot resume; checkpoint is missing or empty:"
        echo "  $CHECKPOINT_FILE"
        exit 1
    fi

    python -m cpp_llm.training.train \
        --config "$CONFIG_FILE" \
        --resume
else
    python -m cpp_llm.training.train \
        --config "$CONFIG_FILE"
fi

echo ""
echo "Training finished."
echo "Checkpoint: ${CHECKPOINT_FILE}"
