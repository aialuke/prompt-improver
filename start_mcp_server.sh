#!/bin/bash
# Start MCP server with environment variables loaded

echo "🔧 Loading environment variables from .env file..."

# Load environment variables from .env file
set -a  # automatically export all variables
source .env
set +a  # stop automatically exporting

echo "🚀 Starting MCP server..."
python -m prompt_improver.mcp_server.mcp_server
