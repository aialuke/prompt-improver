#!/bin/bash
# Installation script for naming convention enforcement

echo "Installing naming convention enforcement tools..."

# Install pre-commit
pip install pre-commit

# Install linting tools
pip install flake8 pep8-naming pylint mypy black isort

# Install pre-commit hooks
pre-commit install

# Copy configuration files to project root
cp linting_configs/setup.cfg .
cp linting_configs/pylintrc .
cp linting_configs/mypy.ini .
cp linting_configs/.pre-commit-config.yaml .

# Create .github/workflows directory if it doesn't exist
mkdir -p .github/workflows
cp linting_configs/.github/workflows/code-quality.yml .github/workflows/

echo "✓ Naming convention enforcement installed successfully!"
echo ""
echo "Usage:"
echo "  pre-commit run --all-files    # Run all checks"
echo "  flake8 src/                   # Run flake8 linting"
echo "  mypy src/                     # Run type checking"
echo "  python naming_convention_analyzer.py  # Run naming analysis"
