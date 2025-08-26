#!/usr/bin/env python3
"""Trace import contamination to find exact source.

This script uses sys.settrace to track exactly what imports are causing
the NumPy contamination during lazy loader import.
"""

import sys
import time
from pathlib import Path


def trace_imports(frame, event, arg):
    """Trace function to track imports."""
    if event == 'call':
        filename = frame.f_code.co_filename
        func_name = frame.f_code.co_name

        # Track coredis/beartype/numpy related imports
        if any(lib in filename.lower() for lib in ['coredis', 'beartype', 'numpy']):
            print(f"IMPORT: {filename}:{func_name}")

        # Track our package imports
        if 'prompt_improver' in filename and 'site-packages' not in filename:
            if any(pattern in filename for pattern in ['__init__.py', 'cache', 'services']):
                print(f"PACKAGE: {Path(filename).relative_to(Path.cwd())}:{func_name}")

    return trace_imports


def main():
    """Run import tracing."""
    print("Starting import trace...")

    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))

    # Enable tracing
    sys.settrace(trace_imports)

    try:
        print("\n=== TRACING LAZY LOADER IMPORT ===")
        start = time.time()
        elapsed = (time.time() - start) * 1000
        print(f"\nImport completed in {elapsed:.1f}ms")

        # Check what got loaded
        ml_modules = [name for name in sys.modules if any(lib in name.lower() for lib in ['numpy', 'beartype', 'coredis'])]
        print(f"\nML modules loaded: {len(ml_modules)}")
        for mod in sorted(ml_modules)[:10]:
            print(f"  - {mod}")

    finally:
        sys.settrace(None)


if __name__ == "__main__":
    main()
