#!/usr/bin/env bash

set -Eeuo pipefail

# Resolve the project root regardless of the current directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${PROJECT_ROOT}/.venv"
CONFIG_FILE="${PROJECT_ROOT}/configs/model.yaml"
CHECKPOINT_FILE="${PROJECT_ROOT}/checkpoints/latest.pt"

echo "======================================"
echo "         cpp-llm Generation"
echo "======================================"

# Check for the virtual environment.
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: Virtual environment not found."
    echo "Create it with: python3 -m venv .venv"
    exit 1
fi

# Activate the environment.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Check required files.
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Model configuration not found: $CONFIG_FILE"
    exit 1
fi

if [[ ! -s "$CHECKPOINT_FILE" ]]; then
    echo "Error: Model checkpoint is missing or empty:"
    echo "  $CHECKPOINT_FILE"
    echo "Train the model before generating code."
    exit 1
fi

# Check Python dependencies.
if ! python -c "import torch, yaml, tiktoken" >/dev/null 2>&1; then
    echo "Error: Required dependencies are missing."
    echo "Run: python -m pip install -r requirements.txt"
    exit 1
fi

# Default generation settings.
PROMPT="#include <iostream>

int main() {
"
MAX_NEW_TOKENS=200
TEMPERATURE=0.8
TOP_K=50
DEVICE="auto"
OUTPUT_FILE=""

# Parse command-line arguments.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt)
            PROMPT="${2:?Missing value for --prompt}"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="${2:?Missing value for --max-new-tokens}"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="${2:?Missing value for --temperature}"
            shift 2
            ;;
        --top-k)
            TOP_K="${2:?Missing value for --top-k}"
            shift 2
            ;;
        --device)
            DEVICE="${2:?Missing value for --device}"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="${2:?Missing value for --output}"
            shift 2
            ;;
        --help|-h)
            cat <<'HELP'
Usage:
  ./scripts/generate.sh [options]

Options:
  --prompt TEXT          C++ prompt to continue
  --max-new-tokens N     Maximum tokens to generate (default: 200)
  --temperature VALUE    Sampling temperature (default: 0.8)
  --top-k N              Top-k sampling (default: 50; use 0 to disable)
  --device DEVICE        auto, cpu, mps, or cuda
  --output FILE          Save generated code to a file
  --help                 Show this help message

Example:
  ./scripts/generate.sh \
    --prompt "Write a C++ factorial function" \
    --max-new-tokens 150 \
    --output generated.cpp
HELP
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1"
            echo "Use --help to see available options."
            exit 1
            ;;
    esac
done

# Construct the generation command safely.
CMD=(
    python -m cpp_llm.inference.generate
    --config "$CONFIG_FILE"
    --checkpoint "$CHECKPOINT_FILE"
    --prompt "$PROMPT"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top-k "$TOP_K"
    --device "$DEVICE"
)

if [[ -n "$OUTPUT_FILE" ]]; then
    CMD+=(--output "$OUTPUT_FILE")
fi

echo "Checkpoint: $CHECKPOINT_FILE"
echo "Device: $DEVICE"
echo "Generating C++ code..."
echo ""

"${CMD[@]}"

echo ""
echo "Generation complete."
