#!/usr/bin/env python3
"""Monitor for unwanted dependencies."""

import subprocess
import sys


def check_unwanted_dependencies():
    """Check for unwanted dependencies."""
    unwanted = ["aiosqlite", "sqlite3", "psycopg2"]
    
    result = subprocess.run(["pip", "list"], capture_output=True, text=True)
    installed = result.stdout.lower()
    
    found_unwanted = []
    for dep in unwanted:
        if dep in installed:
            found_unwanted.append(dep)
    
    if found_unwanted:
        print(f"✗ Found unwanted dependencies: {found_unwanted}")
        return False
    else:
        print("✓ No unwanted dependencies found")
        return True


if __name__ == "__main__":
    success = check_unwanted_dependencies()
    sys.exit(0 if success else 1)