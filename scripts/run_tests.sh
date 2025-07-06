#!/bin/bash
#
# run_tests.sh
#
# This script runs the entire Python test suite using pytest.
# It ensures that the script is run from the project root and that
# the virtual environment is activated.

set -e

echo "--- Running Test Suite ---"

VENV_DIR=".venv"

# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "⚠️ Virtual environment not found. Please run 'scripts/setup_development.sh' first."
    exit 1
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Run linting and formatting checks first
echo " linting with ruff..."
ruff check src tests

echo "🎨 Checking formatting with ruff..."
ruff format --check src tests

echo "🔬 Running pytest..."
pytest tests/

echo ""
echo "✅ All tests passed successfully!"
echo "---" 