#!/bin/bash
# Generate embeddings for the entire project

cd "$(dirname "$0")/.."
echo "🔍 Generating embeddings for the project..."

if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Copy .env.example and configure VOYAGE_API_KEY"
    exit 1
fi

source .env
export VOYAGE_API_KEY

python src/setup.py --generate-embeddings
echo "✅ Embeddings generation complete!"
