#!/bin/bash
# Update embeddings for changed files

cd "$(dirname "$0")/.."
echo "🔄 Updating embeddings for changed files..."

if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Copy .env.example and configure VOYAGE_API_KEY"
    exit 1
fi

source .env
export VOYAGE_API_KEY

python src/embedding_updater.py --update
echo "✅ Embeddings update complete!"
