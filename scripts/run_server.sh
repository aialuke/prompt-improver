#!/bin/bash
#
# run_server.sh
#
# This script starts the FastAPI server for local development.
# It ensures that the script is run from the project root and that
# the virtual environment is activated.

set -e

echo "--- Starting FastAPI Development Server ---"

VENV_DIR=".venv"

# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "⚠️ Virtual environment not found. Please run 'scripts/setup_development.sh' first."
    exit 1
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Start the server with uvicorn, enabling auto-reload
echo "🚀 Launching Uvicorn server..."
echo "Access the API at http://127.0.0.1:8000"
echo "API docs available at http://127.0.0.1:8000/docs"

uvicorn prompt_improver.main:app --host 127.0.0.1 --port 8000 --reload 