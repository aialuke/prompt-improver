#!/bin/bash

# Test script for Firecrawl MCP Server
echo "Testing Firecrawl MCP Server..."

# Set the API key
export FIRECRAWL_API_KEY=fc-ee493e4221094b2d804b5b6f858412c1

# Test basic functionality by running the server briefly
echo "Starting Firecrawl MCP Server..."
echo "Use Ctrl+C to stop the server after testing"

# Run the server
npx -y firecrawl-mcp
