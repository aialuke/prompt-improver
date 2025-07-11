# MCP Server Setup Instructions

This document provides setup instructions for connecting Claude Code and Cursor IDE to the unified MCP prompt evaluation server.

## Prerequisites

1. Node.js 14+ installed
2. MCP dependencies installed (`npm install`)
3. Claude Code or Cursor IDE with MCP support

## Claude Code Setup

1. **Copy Configuration File**
   ```bash
   cp claude-code-config.json ~/.claude/mcp_config.json
   ```

2. **Alternative: Manual Configuration**
   Add to your Claude Code configuration file (`~/.claude/config.json`):
   ```json
   {
     "mcpServers": {
       "prompt-evaluation": {
         "command": "node",
         "args": ["path/to/your/project/src/mcp-server/prompt-evaluation-server.js"],
         "env": {
           "NODE_ENV": "development"
         }
       }
     }
   }
   ```

3. **Restart Claude Code**

4. **Test Connection**
   In Claude Code, you should now have access to:
   - `evaluate_prompt_improvement` tool
   - `analyze_prompt_structure` tool

## Cursor IDE Setup

1. **Create Workspace Configuration**
   Add to your workspace's `.vscode/settings.json` or project root:
   ```json
   {
     "mcpServers": {
       "prompt-evaluation": {
         "command": "node",
         "args": ["./src/mcp-server/prompt-evaluation-server.js"],
         "cwd": "${workspaceFolder}",
         "env": {
           "PROMPT_EVALUATION_MODE": "ide"
         }
       }
     }
   }
   ```

2. **Alternative: Copy Configuration**
   ```bash
   cp cursor-ide-config.json .vscode/mcp_config.json
   ```

3. **Restart Cursor IDE**

4. **Test Connection**
   In Cursor, the MCP tools should be available in the AI assistant.

## Testing the Setup

### Command Line Test
```bash
# Test MCP server standalone
npm run mcp:server &
echo '{"method": "tools/list"}' | node -e "process.stdin.pipe(process.stdout)"

# Validate server
npm run mcp:validate
```

### Framework Integration Test
```bash
# Test framework integration with MCP
node -e "
import MCPLLMJudge from './src/evaluation/mcp-llm-judge.js';
const judge = new MCPLLMJudge();
judge.testConnection().then(result => {
  console.log('MCP Test Result:', result);
  process.exit(result.success ? 0 : 1);
});
"
```

### IDE Integration Test

#### In Claude Code:
1. Open a prompt engineering task
2. Use the prompt evaluation tools:
   ```
   Please evaluate the improvement between these prompts:
   Original: "Create a button"
   Improved: "Create a responsive, accessible button component with hover states and proper ARIA labels"
   ```

#### In Cursor:
1. Open the project with the MCP configuration
2. Ask the AI assistant to evaluate a prompt
3. The assistant should use the MCP tools automatically

## Available Tools

### `evaluate_prompt_improvement`
Compares two prompts and provides improvement scores.

**Parameters:**
- `original` (string): Original prompt text
- `improved` (string): Improved prompt text
- `context` (object): Project context
- `metrics` (array): Evaluation metrics (default: clarity, completeness, specificity, structure)

### `analyze_prompt_structure`
Analyzes the structural quality of a single prompt.

**Parameters:**
- `prompt` (string): Prompt text to analyze
- `context` (object): Project context

## Troubleshooting

### MCP Server Won't Start
1. Check Node.js version: `node --version` (requires 14+)
2. Install dependencies: `npm install`
3. Check server logs: `npm run mcp:server-dev`

### IDE Can't Connect
1. Verify configuration file path and format
2. Check file permissions
3. Ensure server executable: `chmod +x src/mcp-server/prompt-evaluation-server.js`
4. Test server manually: `npm run mcp:server`

### Evaluation Fails
1. Check server is running
2. Verify input format
3. Check logs for error details
4. Test with simple inputs first

### Common Issues

#### "Cannot find module" Error
```bash
# Ensure MCP SDK is installed
npm install @modelcontextprotocol/sdk
```

#### "Connection timeout" Error
- Increase timeout in configuration
- Check system resources
- Test server startup manually

#### "Tool not found" Error
- Verify server configuration
- Check tool names match exactly
- Restart IDE after configuration changes

## Configuration Options

### Server Configuration (`mcp-server-config.json`)
- Timeout settings
- Logging levels
- Tool enablement
- Default evaluation metrics

### Client Configuration
- Connection timeouts
- Retry settings
- Environment variables
- Working directory

## Development Mode

For development and debugging:

```bash
# Start server with debugging
npm run mcp:server-dev

# Start server with verbose logging
NODE_ENV=development npm run mcp:server

# Test server with manual inputs
echo '{"method": "tools/call", "params": {"name": "analyze_prompt_structure", "arguments": {"prompt": "test"}}}' | npm run mcp:server
```

## Support

For issues or questions:
1. Check server logs: `npm run mcp:server-dev`
2. Validate configuration: `npm run mcp:validate`
3. Test connection: Run framework integration test
4. Review this setup guide

The MCP server provides consistent prompt evaluation across all IDE clients while maintaining full framework integration.