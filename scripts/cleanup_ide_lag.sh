#!/bin/bash
# IDE Performance Cleanup Script
# Removes unnecessary caches and temporary files to improve IDE performance

echo "🧹 Starting IDE performance cleanup..."

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Function to safely remove directories/files
safe_remove() {
    if [ -e "$1" ]; then
        echo "  Removing: $1"
        rm -rf "$1"
    fi
}

echo "📁 Cleaning Python caches..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

echo "🔧 Cleaning tool caches..."
safe_remove ".ruff_cache"
safe_remove ".mypy_cache"
safe_remove ".pytest_cache"
safe_remove ".hypothesis"

echo "📊 Cleaning coverage and test artifacts..."
safe_remove "htmlcov"
safe_remove ".coverage"
safe_remove "coverage.xml"
safe_remove "*.coverage"

echo "🗑️ Cleaning build artifacts..."
safe_remove "build"
safe_remove "dist"
safe_remove "*.egg-info"
find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true

echo "🔍 Cleaning IDE specific files..."
safe_remove ".DS_Store"
find . -name ".DS_Store" -delete 2>/dev/null || true

echo "📝 Cleaning backup files..."
find . -type f \( -name "*.bak" -o -name "*.backup" -o -name "*.old" -o -name "*~" -o -name "*.swp" -o -name "*.swo" \) -delete 2>/dev/null || true

echo "🎯 Analyzing remaining large directories..."
echo "Top 10 largest directories:"
du -sh */ 2>/dev/null | sort -rh | head -10

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "💡 Additional recommendations:"
echo "  1. Restart VS Code to apply changes"
echo "  2. Consider using 'scripts/setup_optimized_vscode.sh' for optimized settings"
echo "  3. Run 'git clean -fdx' (⚠️  CAUTION: removes all untracked files) for deeper clean"