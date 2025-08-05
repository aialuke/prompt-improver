# Claude Code CLI Integration Guide

## Setup Instructions

### 1. Copy Configuration Files

Copy the configuration files to your Claude Code CLI configuration directory:

```bash
# Copy command definitions
cp voyage-search/config/claude_code_commands.json .claude/

# Copy MCP server configuration  
cp voyage-search/config/mcp_servers.json .claude/

# Copy aliases
cp voyage-search/config/aliases.json .claude/
```

### 2. Update .claude/settings.local.json

Add permissions for the search tool:

```json
{
  "permissions": {
    "allow": [
      "Bash(python voyage-search/src/search_integration.py:*)",
      "Bash(python voyage-search/src/mcp_server.py:*)",
      "mcp__voyage-ai-search__semantic_search",
      "mcp__voyage-ai-search__search_explain",
      "mcp__voyage-ai-search__search_optimize"
    ]
  }
}
```

### 3. Environment Setup

Ensure your VOYAGE_API_KEY is available:

```bash
export VOYAGE_API_KEY="your-api-key-here"
```

## Usage in Claude Code CLI

### Direct Commands
```bash
python voyage-search/src/search_integration.py "your query" --format-claude
```

### MCP Tools (if configured)
```
semantic_search(query="your query", analysis_type="explain")
search_explain(query="your query")
search_optimize(query="your query")
```

### Custom Commands (if configured)
```bash
search "your query"
explain "your query"  
optimize "your query"
```

## Troubleshooting

1. **Import errors**: Ensure the parent project's `src/` directory is accessible
2. **Embeddings not found**: Run `python voyage-search/src/setup.py --generate-embeddings`
3. **API key issues**: Check that VOYAGE_API_KEY is set in your environment
