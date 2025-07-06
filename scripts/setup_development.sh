#!/bin/bash
#
# setup_development.sh
#
# This script sets up the development environment for the Adaptive Prompt Enhancement System.
# It creates a virtual environment and installs all required dependencies, including
# development tools, using the pyproject.toml file as the source of truth.
#
# It will prefer using 'uv' for speed if it is installed, otherwise it falls back
# to standard 'python -m venv' and 'pip'.

set -e

echo "--- Setting up development environment ---"

# Check for Python 3.11+
if ! command -v python3 &> /dev/null || ! python3 -c 'import sys; assert sys.version_info >= (3, 11)' &> /dev/null; then
    echo "❌ Error: Python 3.11 or higher is required. Please install it and ensure 'python3' is in your PATH."
    exit 1
fi

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "✅ Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    echo "🐍 Creating virtual environment in '$VENV_DIR'..."
    if command -v uv &> /dev/null; then
        echo "(using uv)"
        uv venv
    else
        echo "(using python3 -m venv)"
        python3 -m venv $VENV_DIR
    fi
fi

# Activate the virtual environment
echo "엑 Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "📦 Installing dependencies from pyproject.toml..."

if command -v uv &> /dev/null; then
    echo "(using uv pip install)"
    # Install the project in editable mode with all optional dependencies (including 'dev')
    uv pip install -e ".[dev]"
else
    echo "(using pip)"
    # Ensure pip is up-to-date
    pip install --upgrade pip
    # Install the project in editable mode with all optional dependencies (including 'dev')
    pip install -e ".[dev]"
fi

echo ""
echo "✅ Development environment setup complete!"
echo "To activate the environment in your shell, run:"
echo "source $VENV_DIR/bin/activate"
echo "---" 