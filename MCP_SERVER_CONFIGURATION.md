# MCP Server Configuration Management

## Problem Solved
Multiple MCP server instances were running because three different applications each had their own configuration files launching the same servers:
- 3x firecrawl-mcp
- 3x server-memory 
- 2x sequential-thinking
- 2x context7

## Root Cause
Each application maintained its own MCP configuration:
1. **Claude Desktop App**: `/Users/lukemckenzie/Library/Application Support/Claude/claude_desktop_config.json`
2. **Claude CLI**: `/Users/lukemckenzie/.config/claude/claude_config.json`
3. **VS Code Kilo Code**: `/Users/lukemckenzie/Library/Application Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json`

## Solution Implemented

### Configuration Distribution
We've centralized MCP servers based on their primary use:

#### Claude CLI (Primary Development Hub)
Location: `~/.config/claude/claude_config.json`
Servers:
- `context7` - Documentation lookup
- `sequential-thinking` - Complex reasoning
- `memory` - Persistent memory
- `postgres-apes` - Database access  
- `firecrawl` - Web scraping

#### Claude Desktop App (Desktop Tools Only)
Location: `~/Library/Application Support/Claude/claude_desktop_config.json`
Servers:
- `filesystem` - Local file access
- `desktop-commander` - Desktop automation
- `composio` - GitHub integration

#### VS Code Extension (Disabled)
Location: `~/Library/Application Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json`
Servers: None (empty configuration)

## Why This Approach

1. **No Duplicates**: Each MCP server runs in only one place
2. **Clear Ownership**: Development tools in CLI, desktop tools in Desktop app
3. **Reduced Resource Usage**: Single instance per server instead of 2-3x
4. **Easier Maintenance**: Know exactly where each server is configured

## Future: Universal MCP Access

The MCP specification currently uses a 1:1 client-server model with stdio transport, preventing true server sharing. Future solutions include:

1. **Remote MCP Servers**: Run servers as HTTP services instead of local processes
2. **MCP Proxy**: Create a local proxy that multiple apps can connect to
3. **Built-in Sharing**: Wait for MCP spec updates to support shared instances

For now, our centralized configuration approach eliminates duplicates while maintaining full functionality.

## Verification Commands

Check for MCP processes:
```bash
ps aux | grep -E "(mcp|server-memory|firecrawl|sequential)" | grep -v grep
```

View configurations:
```bash
# Claude CLI config
cat ~/.config/claude/claude_config.json

# Claude Desktop config  
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json

# VS Code config
cat ~/Library/Application\ Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json
```