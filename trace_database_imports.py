#!/usr/bin/env python3
"""
Import tracer to identify the exact import chain causing ML dependencies 
to be loaded when importing database.models.
"""
import sys
import time
import traceback
from types import ModuleType
from typing import Set, List

# Track imports as they happen
imported_modules: List[str] = []
ml_modules: Set[str] = set()

# Original import function
import builtins
original_import = builtins.__import__

def trace_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Custom import function that traces all imports."""
    # Track the import
    imported_modules.append(name)
    
    # Check if this is an ML-related import
    if any(ml_keyword in name.lower() for ml_keyword in [
        'torch', 'sklearn', 'scipy', 'numpy', 'optuna', 
        'ml.', 'machine_learning', 'optimization.validation'
    ]):
        ml_modules.add(name)
        print(f"🔴 ML IMPORT DETECTED: {name}")
        print(f"   Import stack depth: {len(imported_modules)}")
        print(f"   Last 5 imports: {imported_modules[-5:]}")
        print("   Stack trace:")
        for line in traceback.format_stack()[-8:-1]:  # Show relevant stack
            if 'site-packages' not in line and '__pycache__' not in line:
                print(f"     {line.strip()}")
        print("")
    
    # Call original import
    try:
        return original_import(name, globals, locals, fromlist, level)
    except Exception as e:
        print(f"❌ Import failed: {name} - {e}")
        raise

# Monkey patch the import function
builtins.__import__ = trace_import

print("🔍 Starting import trace for database.models...")
print("="*60)

start_time = time.time()

try:
    # This is the problematic import
    from prompt_improver.database.models import PromptSession
    
    end_time = time.time()
    import_time = (end_time - start_time) * 1000
    
    print(f"✅ Import completed in {import_time:.0f}ms")
    print(f"📊 Total modules imported: {len(imported_modules)}")
    print(f"🔴 ML modules detected: {len(ml_modules)}")
    
    if ml_modules:
        print("\n🔴 ML MODULES THAT WERE IMPORTED:")
        for module in sorted(ml_modules):
            print(f"   - {module}")
    
    print(f"\n📋 ALL IMPORTS (first 50):")
    for i, module in enumerate(imported_modules[:50]):
        marker = "🔴" if any(ml_keyword in module.lower() for ml_keyword in [
            'torch', 'sklearn', 'scipy', 'numpy', 'optuna', 'ml.', 'optimization'
        ]) else "  "
        print(f"   {i+1:2d}. {marker} {module}")
    
    if len(imported_modules) > 50:
        print(f"   ... and {len(imported_modules) - 50} more modules")

except Exception as e:
    end_time = time.time()
    import_time = (end_time - start_time) * 1000
    print(f"❌ Import failed after {import_time:.0f}ms: {e}")
    print(f"📊 Modules imported before failure: {len(imported_modules)}")
    
    if ml_modules:
        print("\n🔴 ML MODULES IMPORTED BEFORE FAILURE:")
        for module in sorted(ml_modules):
            print(f"   - {module}")

finally:
    # Restore original import
    builtins.__import__ = original_import

print("\n" + "="*60)
print("🏁 Import trace completed")