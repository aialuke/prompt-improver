#!/bin/bash
# Setup optimized VS Code settings for better performance

echo "🚀 Setting up optimized VS Code configuration..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Backup current settings
if [ -f ".vscode/settings.json" ]; then
    echo "📦 Backing up current settings to .vscode/settings.json.backup"
    cp .vscode/settings.json .vscode/settings.json.backup
fi

# Apply optimized settings
echo "⚡ Applying optimized settings..."
cp .vscode/settings-optimized.json .vscode/settings.json

echo ""
echo "✅ Optimized settings applied!"
echo ""
echo "🎯 Performance improvements made:"
echo "  • Disabled Python strict type checking (use 'basic' mode)"
echo "  • Disabled all inlay hints and semantic highlighting"
echo "  • Disabled automatic indexing and imports"
echo "  • Added aggressive file exclusions and watcher ignores"
echo "  • Disabled resource-intensive language features"
echo "  • Prioritized Ruff for fast linting/formatting"
echo ""
echo "💡 Next steps:"
echo "  1. Run './scripts/cleanup_ide_lag.sh' to clean caches"
echo "  2. Restart VS Code"
echo "  3. To restore original settings: cp .vscode/settings.json.backup .vscode/settings.json"