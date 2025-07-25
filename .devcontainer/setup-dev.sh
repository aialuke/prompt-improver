#!/bin/bash
set -e

echo "Setting up development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "/workspace/.venv" ]; then
    echo "Creating virtual environment..."
    python -m venv /workspace/.venv
fi

# Activate virtual environment
source /workspace/.venv/bin/activate

# Upgrade pip and core tools
echo "Upgrading pip and core tools..."
pip install --upgrade pip setuptools wheel pip-tools

# Install dependencies
echo "Installing project dependencies..."
if [ -f "/workspace/requirements.txt" ]; then
    pip install -r /workspace/requirements.txt
fi

if [ -f "/workspace/requirements-dev.txt" ]; then
    pip install -r /workspace/requirements-dev.txt
fi

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
if [ -f "/workspace/.pre-commit-config.yaml" ]; then
    pre-commit install
    pre-commit install --hook-type commit-msg
    pre-commit install --hook-type pre-push
fi

# Set up Git configuration
echo "Configuring Git..."
git config --global --add safe.directory /workspace
git config --global core.editor "code --wait"

# Create necessary directories
echo "Creating project directories..."
mkdir -p /workspace/logs
mkdir -p /workspace/data/ml_studies
mkdir -p /workspace/.mypy_cache
mkdir -p /workspace/.ruff_cache
mkdir -p /workspace/.pytest_cache

# Set permissions
echo "Setting permissions..."
sudo chown -R vscode:vscode /workspace/.venv
sudo chown -R vscode:vscode /workspace/logs
sudo chown -R vscode:vscode /workspace/data

# Install the project in editable mode
echo "Installing project in editable mode..."
pip install -e /workspace

# Download NLTK data if needed
echo "Setting up NLTK resources..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Verify installation
echo "Verifying installation..."
python -c "import prompt_improver; print(f'Project installed successfully! Version: {prompt_improver.__version__ if hasattr(prompt_improver, \"__version__\") else \"development\"}')"

echo "Development environment setup complete!"