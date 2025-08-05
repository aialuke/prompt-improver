#!/usr/bin/env python3
"""
Debug embeddings format to understand the structure.
"""

import pickle
import sys
from pathlib import Path
import numpy as np

# Add src to path and import classes
sys.path.insert(0, str(Path("src")))
from generate_embeddings import CodeChunk, ChunkType, EmbeddingMetadata

# Add classes to __main__ namespace for pickle loading
import __main__
__main__.CodeChunk = CodeChunk
__main__.ChunkType = ChunkType
__main__.EmbeddingMetadata = EmbeddingMetadata

def debug_embeddings_format():
    """Debug the embeddings format to understand the structure."""
    print("🔍 Debugging Embeddings Format")
    print("=" * 50)
    
    pickle_path = Path("src/embeddings.pkl")
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"📊 Data keys: {list(data.keys())}")
        
        for key, value in data.items():
            print(f"\n🔑 Key '{key}':")
            print(f"   Type: {type(value)}")
            
            if hasattr(value, '__len__'):
                print(f"   Length: {len(value)}")
            
            if isinstance(value, list) and len(value) > 0:
                print(f"   First item type: {type(value[0])}")
                if hasattr(value[0], '__dict__'):
                    attrs = list(value[0].__dict__.keys())[:5]
                    print(f"   First item attributes: {attrs}")
            
            if isinstance(value, np.ndarray):
                print(f"   Shape: {value.shape}")
                print(f"   Dtype: {value.dtype}")
            
            # Check if it's a list of arrays
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                print(f"   List of arrays - first array shape: {value[0].shape}")
                print(f"   List of arrays - first array dtype: {value[0].dtype}")
                
                # Convert to single array
                try:
                    combined = np.array(value)
                    print(f"   Combined array shape: {combined.shape}")
                    print(f"   Combined array dtype: {combined.dtype}")
                except Exception as e:
                    print(f"   ❌ Cannot combine arrays: {e}")
                    
                    # Try stacking
                    try:
                        stacked = np.vstack(value)
                        print(f"   Stacked array shape: {stacked.shape}")
                        print(f"   Stacked array dtype: {stacked.dtype}")
                    except Exception as e2:
                        print(f"   ❌ Cannot stack arrays: {e2}")
        
        # Test loading with proper conversion
        print(f"\n🔧 Testing proper conversion:")
        embeddings = data['embeddings']
        chunks = data['chunks']
        
        if isinstance(embeddings, list):
            print(f"   Converting list of {len(embeddings)} embeddings to array...")
            try:
                embeddings_array = np.vstack(embeddings)
                print(f"   ✅ Converted to array shape: {embeddings_array.shape}")
                
                # Test with search system
                sys.path.insert(0, str(Path("src")))
                from search_code import HybridCodeSearch
                
                search_system = HybridCodeSearch(
                    embeddings=embeddings_array,
                    chunks=chunks,
                    metadata={"test": True}
                )
                
                # Test a simple search
                results = search_system.hybrid_search("calculate similarity", top_k=3)
                print(f"   ✅ Search test successful: {len(results)} results")
                
                if results:
                    print(f"   🥇 Top result: {results[0].chunk_name}")
                
            except Exception as e:
                print(f"   ❌ Conversion failed: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_embeddings_format()
