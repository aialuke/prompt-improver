#!/usr/bin/env python3
"""
Debug pickle file structure to identify deserialization issues.
"""

import pickle
import pickletools
import sys
from pathlib import Path
from typing import Any, Dict


def analyze_pickle_structure(pickle_path: Path) -> None:
    """Analyze the structure of the pickle file to identify class dependencies."""
    print(f"🔍 Analyzing pickle file structure: {pickle_path}")
    print("=" * 70)
    
    if not pickle_path.exists():
        print(f"❌ Pickle file not found: {pickle_path}")
        return
    
    # Step 1: Use pickletools to analyze the pickle opcodes
    print("📋 Pickle opcodes analysis:")
    try:
        with open(pickle_path, 'rb') as f:
            print("   Analyzing pickle opcodes...")
            pickletools.dis(f)
    except Exception as e:
        print(f"   ❌ Pickletools analysis failed: {e}")
    
    print("\n" + "=" * 70)
    
    # Step 2: Try to load with custom unpickler to catch class references
    print("🔍 Custom unpickler analysis:")
    
    class DebugUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            print(f"   📦 Loading class: {module}.{name}")
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError) as e:
                print(f"   ❌ Failed to load {module}.{name}: {e}")
                raise
    
    try:
        with open(pickle_path, 'rb') as f:
            unpickler = DebugUnpickler(f)
            data = unpickler.load()
        print(f"   ✅ Successfully loaded pickle data")
        print(f"   📊 Data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        
    except Exception as e:
        print(f"   ❌ Custom unpickler failed: {e}")
        print(f"   Error type: {type(e).__name__}")
    
    print("\n" + "=" * 70)
    
    # Step 3: Try loading individual components
    print("🧩 Component-by-component analysis:")
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"   🔑 Key '{key}': {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
                
                # Analyze first few items if it's a list
                if isinstance(value, list) and len(value) > 0:
                    print(f"      First item type: {type(value[0])}")
                    print(f"      First item module: {getattr(type(value[0]), '__module__', 'Unknown')}")
                    print(f"      First item class: {type(value[0]).__name__}")
                    
                    # Try to access attributes of first item
                    if hasattr(value[0], '__dict__'):
                        attrs = list(value[0].__dict__.keys())[:5]  # First 5 attributes
                        print(f"      First item attributes: {attrs}")
        
    except Exception as e:
        print(f"   ❌ Component analysis failed: {e}")
        print(f"   Error type: {type(e).__name__}")


def investigate_class_imports() -> None:
    """Investigate how classes are defined in the generation script."""
    print("\n🔍 Investigating class definitions in generate_embeddings.py")
    print("=" * 70)
    
    generate_embeddings_path = Path("src/generate_embeddings.py")
    if not generate_embeddings_path.exists():
        print(f"❌ generate_embeddings.py not found")
        return
    
    # Search for class definitions
    with open(generate_embeddings_path, 'r') as f:
        content = f.read()
    
    # Find class definitions
    import re
    class_pattern = r'^class\s+(\w+).*?:'
    classes = re.findall(class_pattern, content, re.MULTILINE)
    
    print(f"📋 Classes defined in generate_embeddings.py:")
    for cls in classes:
        print(f"   • {cls}")
    
    # Find dataclass definitions
    dataclass_pattern = r'@dataclass.*?\nclass\s+(\w+)'
    dataclasses = re.findall(dataclass_pattern, content, re.DOTALL)
    
    print(f"\n📋 Dataclasses defined:")
    for cls in dataclasses:
        print(f"   • {cls}")
    
    # Find enum definitions
    enum_pattern = r'class\s+(\w+)\(Enum\)'
    enums = re.findall(enum_pattern, content)
    
    print(f"\n📋 Enums defined:")
    for cls in enums:
        print(f"   • {cls}")


def test_manual_class_import() -> None:
    """Test importing classes manually to identify import issues."""
    print("\n🧪 Testing manual class imports")
    print("=" * 70)
    
    # Add src to path
    sys.path.insert(0, str(Path("src")))
    
    classes_to_test = [
        ("generate_embeddings", "CodeChunk"),
        ("generate_embeddings", "ChunkType"), 
        ("generate_embeddings", "EmbeddingMetadata"),
    ]
    
    for module_name, class_name in classes_to_test:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            print(f"   ✅ Successfully imported {module_name}.{class_name}")
            print(f"      Class: {cls}")
            print(f"      Module: {cls.__module__}")
            
        except ImportError as e:
            print(f"   ❌ Import error for {module_name}.{class_name}: {e}")
        except AttributeError as e:
            print(f"   ❌ Attribute error for {module_name}.{class_name}: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected error for {module_name}.{class_name}: {e}")


def test_incremental_loading() -> None:
    """Test loading pickle file with different strategies."""
    print("\n🔬 Testing incremental loading strategies")
    print("=" * 70)
    
    pickle_path = Path("src/embeddings.pkl")
    
    # Strategy 1: Load with __main__ module manipulation
    print("📋 Strategy 1: __main__ module manipulation")
    try:
        # Add src to path and import classes into __main__
        sys.path.insert(0, str(Path("src")))
        import generate_embeddings
        
        # Add classes to __main__ namespace
        import __main__
        __main__.CodeChunk = generate_embeddings.CodeChunk
        __main__.ChunkType = generate_embeddings.ChunkType
        if hasattr(generate_embeddings, 'EmbeddingMetadata'):
            __main__.EmbeddingMetadata = generate_embeddings.EmbeddingMetadata
        
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"   ✅ Strategy 1 successful!")
        print(f"   📊 Loaded {len(data.get('chunks', []))} chunks")
        return data
        
    except Exception as e:
        print(f"   ❌ Strategy 1 failed: {e}")
    
    # Strategy 2: Custom find_class function
    print("\n📋 Strategy 2: Custom find_class function")
    try:
        sys.path.insert(0, str(Path("src")))
        import generate_embeddings
        
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "__main__":
                    # Redirect __main__ classes to generate_embeddings module
                    if hasattr(generate_embeddings, name):
                        return getattr(generate_embeddings, name)
                return super().find_class(module, name)
        
        with open(pickle_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            data = unpickler.load()
        
        print(f"   ✅ Strategy 2 successful!")
        print(f"   📊 Loaded {len(data.get('chunks', []))} chunks")
        return data
        
    except Exception as e:
        print(f"   ❌ Strategy 2 failed: {e}")
    
    return None


def main():
    """Main investigation function."""
    print("🔍 SYSTEMATIC PICKLE LOADING INVESTIGATION")
    print("=" * 70)
    
    pickle_path = Path("src/embeddings.pkl")
    
    # Step 1: Analyze pickle structure
    analyze_pickle_structure(pickle_path)
    
    # Step 2: Investigate class imports
    investigate_class_imports()
    
    # Step 3: Test manual class import
    test_manual_class_import()
    
    # Step 4: Test incremental loading
    data = test_incremental_loading()
    
    if data:
        print(f"\n🎉 SUCCESS: Loaded real embeddings with {len(data.get('chunks', []))} chunks!")
    else:
        print(f"\n❌ FAILURE: Could not load real embeddings")


if __name__ == "__main__":
    main()
