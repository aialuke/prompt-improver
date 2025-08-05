# Claude Code CLI with Enhanced Semantic Search

## 🚀 **THREE WAYS TO USE WITH CLAUDE CODE**

### **Method 1: Direct MCP Integration (RECOMMENDED)**
Claude Code can use the search system directly through MCP tools:

```
# In Claude Code CLI, use these tools directly:
semantic_search(query="quicksort algorithm", analysis_type="explain")
search_explain(query="database connection management")
search_optimize(query="ML training performance")
```

### **Method 2: Bash Commands**
Claude Code can execute the Python scripts directly:

```bash
# Search for code patterns
python claude_code_search_integration.py "quicksort algorithm" --format-claude
python claude_code_search_integration.py "database connection" --analysis explain --format-claude
python claude_code_search_integration.py "error handling" --analysis optimize --format-claude
```

### **Method 3: Custom Commands (if configured)**
If custom commands are set up in `.claude/claude_code_commands.json`:

```bash
# Search commands
search "quicksort algorithm implementation"
explain "machine learning training loop"
optimize "database query performance"
review "authentication and authorization"
refactor "configuration management patterns"
```

## 🔧 **MCP SETUP FOR CLAUDE CODE**

### **1. Configure MCP Server**
Add to your Claude Code configuration:

```json
{
  "mcpServers": {
    "voyage-ai-search": {
      "command": "python",
      "args": ["claude_code_mcp_server.py"],
      "env": {
        "VOYAGE_API_KEY": "${VOYAGE_API_KEY}"
      }
    }
  }
}
```

### **2. Available MCP Tools**
- **`semantic_search`**: General semantic search with all options
- **`search_explain`**: Search and generate explanation prompts
- **`search_optimize`**: Search and generate optimization suggestions

### **3. Example MCP Usage in Claude Code**
```
# Search for quicksort and explain it
semantic_search(query="quicksort algorithm", analysis_type="explain", max_results=2)

# Find database code and optimize it
search_optimize(query="database connection management", max_results=3)

# Search with custom parameters
semantic_search(query="async patterns", min_similarity=0.2, search_method="hybrid")
```

## 🎯 **INTEGRATION FEATURES**

✅ **Direct Claude Code Access**: MCP tools available in Claude Code CLI
✅ **Contextualized Embeddings**: Using voyage-context-3 for better code understanding
✅ **Binary Rescoring**: Fast initial retrieval with full-precision reranking
✅ **Voyage AI Reranking**: Official rerank-2.5-lite for optimal relevance
✅ **Hybrid Search**: Semantic + BM25 lexical search
✅ **Multiple Analysis Types**: explain, optimize, review, refactor
✅ **Formatted Output**: Ready for Claude Code analysis

## 📊 **PERFORMANCE**

- **Search Time**: ~1 second for comprehensive search
- **Coverage**: 2,103 code chunks from 393 files
- **Quality**: High relevance scores with context awareness
- **Methods**: Binary rescoring + Voyage AI reranking

## ⚙️ **SETUP**

### **Prerequisites**
```bash
# Set API key
export VOYAGE_API_KEY="your-api-key-here"

# Generate embeddings (one-time)
python src/generate_embeddings.py

# Install MCP library (if using MCP server)
pip install mcp
```

### **Test the Integration**
```bash
# Test direct script
python claude_code_search_integration.py "test query" --format-claude

# Test MCP server (if configured)
python claude_code_mcp_server.py
```
