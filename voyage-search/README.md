# Voyage AI Semantic Search Tool

A powerful semantic code search system using Voyage AI embeddings with Claude Code CLI integration.

## Features

- 🧠 **Semantic Understanding**: AI-powered code comprehension
- ⚡ **High Performance**: Binary rescoring + Voyage AI reranking  
- 🔍 **Hybrid Search**: Semantic + lexical search combined
- 🤖 **Claude Code Integration**: Direct integration with Claude Code CLI
- 📊 **Incremental Updates**: Smart embedding updates for changed files
- 🎯 **Context Aware**: Contextualized embeddings for better relevance

## Quick Start

1. **Install dependencies:**
   ```bash
   cd voyage-search
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.voyager .env
   # Edit .env and add your VOYAGE_API_KEY
   ```

3. **Generate embeddings:**
   ```bash
   python src/setup.py --generate-embeddings
   ```

4. **Search your code:**
   ```bash
   python src/search_integration.py "your search query" --format-claude
   ```

## Directory Structure

- `src/` - Source code
- `config/` - Claude Code integration configs
- `data/` - Generated embeddings and metadata
- `docs/` - Documentation
- `scripts/` - Convenience scripts

## Documentation

See `docs/USAGE.md` for detailed usage instructions and `docs/INTEGRATION.md` for Claude Code CLI integration.
