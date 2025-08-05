# External Tools Configuration

## Voyage AI Semantic Search Tool

The Voyage AI semantic search tool has been extracted to a standalone repository.

### Installation

1. Clone the repository:
   ```bash
   git clone /path/to/voyage-ai-semantic-search
   cd voyage-ai-semantic-search
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env and add your VOYAGE_API_KEY
   ```

### Usage

Basic search:
```bash
python /path/to/voyage-ai-semantic-search/src/search_integration.py "your query" --format-claude
```

### Claude Code Integration

To re-enable voyage-search commands, add to your configurations:

**claude_code_commands.json:**
```json
{
  "commands": {
    "search": {
      "description": "Semantic code search with Voyage AI embeddings",
      "script": "python /path/to/voyage-ai-semantic-search/src/search_integration.py"
    }
  }
}
```

**mcp_servers.json:**
```json
{
  "mcpServers": {
    "voyage-ai-search": {
      "command": "python",
      "args": ["/path/to/voyage-ai-semantic-search/src/mcp_server.py"],
      "env": {
        "VOYAGE_API_KEY": "${VOYAGE_API_KEY}"
      }
    }
  }
}
```
