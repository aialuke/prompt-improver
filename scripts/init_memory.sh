#!/bin/bash

# Initialize MCP memory for any project directory
# Usage: ./scripts/init_memory.sh [project_name]

set -e

PROJECT_DIR="${PWD}"
PROJECT_NAME="${1:-$(basename "$PROJECT_DIR")}"
MEMORY_DIR="${PROJECT_DIR}/.mcp-memory"
MEMORY_FILE="${MEMORY_DIR}/memory.json"

echo "🧠 Initializing MCP memory for project: ${PROJECT_NAME}"
echo "📍 Project directory: ${PROJECT_DIR}"

# Create memory directory if it doesn't exist
if [ ! -d "$MEMORY_DIR" ]; then
    echo "📁 Creating memory directory: ${MEMORY_DIR}"
    mkdir -p "$MEMORY_DIR"
fi

# Create memory file if it doesn't exist
if [ ! -f "$MEMORY_FILE" ]; then
    echo "💾 Creating memory file: ${MEMORY_FILE}"
    echo '{"entities": [], "relations": []}' > "$MEMORY_FILE"
else
    echo "✅ Memory file already exists: ${MEMORY_FILE}"
fi

# Show memory file info
echo "📊 Memory file status:"
echo "   Path: ${MEMORY_FILE}"
echo "   Size: $(stat -f%z "$MEMORY_FILE" 2>/dev/null || stat -c%s "$MEMORY_FILE" 2>/dev/null || echo "unknown") bytes"

# Check if memory file is valid JSON
if python -m json.tool "$MEMORY_FILE" > /dev/null 2>&1; then
    echo "✅ Memory file is valid JSON"
    
    # Show current memory stats
    ENTITY_COUNT=$(python -c "import json; data=json.load(open('$MEMORY_FILE')); print(len(data.get('entities', [])))" 2>/dev/null || echo "0")
    RELATION_COUNT=$(python -c "import json; data=json.load(open('$MEMORY_FILE')); print(len(data.get('relations', [])))" 2>/dev/null || echo "0")
    
    echo "📈 Current memory contents:"
    echo "   Entities: ${ENTITY_COUNT}"
    echo "   Relations: ${RELATION_COUNT}"
else
    echo "❌ Memory file is invalid JSON - reinitializing"
    echo '{"entities": [], "relations": []}' > "$MEMORY_FILE"
fi

echo ""
echo "🎯 Memory initialization complete!"
echo "💡 Usage tips:"
echo "   - CLI: Run claude-code from this directory (${PROJECT_DIR})"
echo "   - IDE: Open this directory as a workspace in Claude Code"
echo "   - Memory will be project-specific and isolated"