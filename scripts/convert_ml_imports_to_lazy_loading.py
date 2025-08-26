#!/usr/bin/env python3
"""Script to systematically convert ML imports to lazy loading patterns.

This script processes all Python files with direct ML imports and converts them
to use the centralized lazy loading utilities to eliminate dependency contamination.

Usage:
    python scripts/convert_ml_imports_to_lazy_loading.py [--dry-run] [--file FILE]
"""

import argparse
import logging
import os
import re
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Base directory (assumes script is in scripts/ folder)
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"

# Import patterns to convert
IMPORT_PATTERNS = {
    # NumPy patterns
    r"^import numpy as np$": "from prompt_improver.core.utils.lazy_ml_loader import get_numpy",
    r"^import numpy$": "from prompt_improver.core.utils.lazy_ml_loader import get_numpy",
    r"^from numpy import (.+)$": "from prompt_improver.core.utils.lazy_ml_loader import get_numpy",

    # PyTorch patterns
    r"^import torch$": "from prompt_improver.core.utils.lazy_ml_loader import get_torch",
    r"^from torch import (.+)$": "from prompt_improver.core.utils.lazy_ml_loader import get_torch",

    # SciPy patterns
    r"^from scipy import stats$": "from prompt_improver.core.utils.lazy_ml_loader import get_scipy_stats",
    r"^from scipy\.stats import (.+)$": "from prompt_improver.core.utils.lazy_ml_loader import get_scipy_stats",
    r"^import scipy$": "from prompt_improver.core.utils.lazy_ml_loader import get_scipy",

    # Scikit-learn patterns
    r"^from sklearn\.utils import (.+)$": "from prompt_improver.core.utils.lazy_ml_loader import get_sklearn_utils, get_sklearn",
    r"^from sklearn\.metrics import (.+)$": "from prompt_improver.core.utils.lazy_ml_loader import get_sklearn_metrics, get_sklearn",
    r"^from sklearn import (.+)$": "from prompt_improver.core.utils.lazy_ml_loader import get_sklearn",
    r"^import sklearn$": "from prompt_improver.core.utils.lazy_ml_loader import get_sklearn",

    # Transformers patterns
    r"^from transformers import (.+)$": "from prompt_improver.core.utils.lazy_ml_loader import get_transformers_components",
    r"^import transformers$": "from prompt_improver.core.utils.lazy_ml_loader import get_transformers",
}

# Usage patterns to convert
USAGE_REPLACEMENTS = {
    # NumPy replacements
    r"\bnp\.": "get_numpy().",
    r"\bnumpy\.": "get_numpy().",

    # PyTorch replacements
    r"\btorch\.": "get_torch().",

    # SciPy replacements
    r"\bstats\.": "get_scipy_stats().",
    r"\bscipy\.": "get_scipy().",

    # Scikit-learn replacements (more specific patterns)
    r"\bresample\(": "get_sklearn_utils().resample(",
    r"\baccuracy_score\(": "get_sklearn_metrics().accuracy_score(",
    r"\bprecision_score\(": "get_sklearn_metrics().precision_score(",
    r"\brecall_score\(": "get_sklearn_metrics().recall_score(",
    r"\bf1_score\(": "get_sklearn_metrics().f1_score(",
    r"\bclassification_report\(": "get_sklearn_metrics().classification_report(",
    r"\bconfusion_matrix\(": "get_sklearn_metrics().confusion_matrix(",
    r"\broc_auc_score\(": "get_sklearn_metrics().roc_auc_score(",
    r"\bsklearn\.": "get_sklearn().",
}

# Files to skip (already converted or special cases)
SKIP_FILES = {
    "lazy_ml_loader.py",  # Our own loader
    "protocols/ml.py",    # Already has proper TYPE_CHECKING
}


def find_ml_files() -> list[Path]:
    """Find all Python files with ML imports."""
    try:
        # Use ripgrep if available for faster search
        result = subprocess.run([
            "rg", "-l", "--type", "py",
            "import numpy|from numpy|import torch|from torch|import scipy|from scipy|import sklearn|from sklearn|import transformers|from transformers"
        ], check=False, cwd=BASE_DIR, capture_output=True, text=True)

        if result.returncode == 0:
            files = [Path(line.strip()) for line in result.stdout.splitlines()]
            return [f for f in files if f.exists() and not any(skip in str(f) for skip in SKIP_FILES)]
    except FileNotFoundError:
        logger.warning("ripgrep not found, falling back to find")

    # Fallback to find + grep
    files = []
    for root, _, filenames in os.walk(SRC_DIR):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = Path(root) / filename
                if any(skip in str(filepath) for skip in SKIP_FILES):
                    continue

                try:
                    with open(filepath, encoding='utf-8') as f:
                        content = f.read()
                        if any(pattern in content for pattern in [
                            "import numpy", "import torch", "import scipy",
                            "import sklearn", "import transformers", "from numpy",
                            "from torch", "from scipy", "from sklearn", "from transformers"
                        ]):
                            files.append(filepath)
                except Exception as e:
                    logger.warning(f"Could not read {filepath}: {e}")

    return files


def analyze_file_imports(filepath: Path) -> tuple[list[str], bool]:
    """Analyze a file's imports and determine conversion needs."""
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()

        # Check for existing lazy loading imports
        has_lazy_imports = "from prompt_improver.core.utils.lazy_ml_loader" in content

        # Find ML import lines
        ml_imports = []
        lines = content.splitlines()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(pattern in stripped for pattern in [
                "import numpy", "import torch", "import scipy",
                "import sklearn", "import transformers", "from numpy",
                "from torch", "from scipy", "from sklearn", "from transformers"
            ]):
                ml_imports.append(f"Line {i + 1}: {line}")

        return ml_imports, has_lazy_imports

    except Exception as e:
        logger.exception(f"Error analyzing {filepath}: {e}")
        return [], False


def convert_file_imports(filepath: Path, dry_run: bool = False) -> dict[str, any]:
    """Convert a single file's imports to lazy loading."""
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes_made = []

        # Skip if already has lazy imports
        if "from prompt_improver.core.utils.lazy_ml_loader" in content:
            return {
                "status": "skipped",
                "reason": "already_converted",
                "changes": []
            }

        # Convert import statements
        lines = content.splitlines()
        new_lines = []
        lazy_imports_to_add = set()

        for line in lines:
            stripped = line.strip()
            converted = False

            # Check each import pattern
            for pattern, replacement in IMPORT_PATTERNS.items():
                if re.match(pattern, stripped):
                    # Extract the indentation
                    indentation = line[:len(line) - len(line.lstrip())]

                    # Add the lazy import (will be deduped later)
                    lazy_imports_to_add.add(replacement)

                    # Comment out the original import
                    new_lines.append(f"{indentation}# {stripped}  # Converted to lazy loading")
                    changes_made.append(f"Converted import: {stripped}")
                    converted = True
                    break

            if not converted:
                new_lines.append(line)

        # Add lazy imports after the existing imports
        if lazy_imports_to_add:
            # Find insertion point (after last import but before first non-import)
            insertion_point = 0
            for i, line in enumerate(new_lines):
                if (line.strip().startswith(('import ', 'from ')) and
                    not line.strip().startswith('#') and
                    'typing' not in line):
                    insertion_point = i + 1

            # Insert lazy imports
            for lazy_import in sorted(lazy_imports_to_add):
                new_lines.insert(insertion_point, lazy_import)
                insertion_point += 1
                changes_made.append(f"Added lazy import: {lazy_import}")

        content = '\n'.join(new_lines)

        # Convert usage patterns
        for pattern, replacement in USAGE_REPLACEMENTS.items():
            new_content, count = re.subn(pattern, replacement, content)
            if count > 0:
                content = new_content
                changes_made.append(f"Replaced {count} usage(s) of pattern: {pattern}")

        # Only write if changes were made and not dry run
        if changes_made and not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        return {
            "status": "converted" if changes_made else "no_changes",
            "changes": changes_made,
            "changes_count": len(changes_made)
        }

    except Exception as e:
        logger.exception(f"Error converting {filepath}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "changes": []
        }


def main():
    parser = argparse.ArgumentParser(description="Convert ML imports to lazy loading")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--file", help="Convert specific file only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting ML import to lazy loading conversion")

    files_to_process = [Path(args.file)] if args.file else find_ml_files()

    logger.info(f"Found {len(files_to_process)} files with ML imports")

    results = {
        "converted": 0,
        "skipped": 0,
        "no_changes": 0,
        "errors": 0,
        "total_changes": 0
    }

    for filepath in files_to_process:
        logger.info(f"Processing: {filepath.relative_to(BASE_DIR)}")

        # Analyze first
        imports, _has_lazy = analyze_file_imports(filepath)
        if args.verbose and imports:
            logger.debug(f"  ML imports found: {len(imports)}")
            for imp in imports[:3]:  # Show first 3
                logger.debug(f"    {imp}")

        # Convert
        result = convert_file_imports(filepath, dry_run=args.dry_run)
        results[result["status"]] += 1
        results["total_changes"] += result.get("changes_count", 0)

        if result["changes"]:
            logger.info(f"  {'Would make' if args.dry_run else 'Made'} {len(result['changes'])} changes")
            if args.verbose:
                for change in result["changes"][:3]:  # Show first 3 changes
                    logger.debug(f"    {change}")
        elif result["status"] == "skipped":
            logger.info(f"  Skipped: {result.get('reason', 'unknown')}")
        elif result["status"] == "error":
            logger.error(f"  Error: {result.get('error', 'unknown')}")

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Files processed: {len(files_to_process)}")
    logger.info(f"Converted: {results['converted']}")
    logger.info(f"Skipped (already converted): {results['skipped']}")
    logger.info(f"No changes needed: {results['no_changes']}")
    logger.info(f"Errors: {results['errors']}")
    logger.info(f"Total changes made: {results['total_changes']}")

    if args.dry_run:
        logger.info("\nThis was a DRY RUN. No files were modified.")
        logger.info("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
