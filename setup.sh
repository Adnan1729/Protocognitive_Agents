#!/usr/bin/env bash
# One-time environment setup. Run from the repository root:
#     bash setup.sh
set -e

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Error: $PYTHON_BIN not found."
    echo "TensorFlow requires Python 3.9-3.12. Install one of those, or set"
    echo "PYTHON_BIN to point to it, e.g.:  PYTHON_BIN=python3.10 bash setup.sh"
    exit 1
fi

echo "Creating virtual environment in .venv/ using $PYTHON_BIN..."
"$PYTHON_BIN" -m venv .venv

# Activation path differs between Unix and Git Bash on Windows
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    source .venv/Scripts/activate
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing proteinoid_spikes (editable) and dependencies..."
pip install -e .

echo ""
echo "Setup complete. Activate the environment with:"
if [ -f ".venv/bin/activate" ]; then
    echo "    source .venv/bin/activate"
else
    echo "    source .venv/Scripts/activate"
fi