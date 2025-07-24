#!/usr/bin/env python3
"""
Comprehensive Validation Script for Naming Convention Changes
"""

import subprocess
import sys
from pathlib import Path


def run_validation_suite():
    """Run comprehensive validation suite."""
    print("=== Comprehensive Validation Suite ===")

    validations = [
        ("Syntax Check", "python -m py_compile src/prompt_improver/__init__.py"),
        ("Import Check", "python -c 'import sys; sys.path.insert(0, "src"); import prompt_improver'"),
        ("Type Check", "mypy src/prompt_improver --ignore-missing-imports --no-error-summary"),
        ("Naming Convention Check", "python naming_convention_analyzer.py"),
    ]

    results = []
    for name, command in validations:
        print(f"\nRunning {name}...")
        result = subprocess.run(command.split(), capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ {name} passed")
            results.append((name, True, ""))
        else:
            print(f"✗ {name} failed")
            print(f"Error: {result.stderr}")
            results.append((name, False, result.stderr))

    # Summary
    print("\n=== Validation Summary ===")
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    for name, success, error in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")

    return passed == total


if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)
