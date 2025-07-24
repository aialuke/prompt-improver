#!/usr/bin/env python3
"""
Update Type Imports Script
Updates all imports to use new snake_case type aliases.
"""

import os
import re
from pathlib import Path


def update_type_imports_in_file(file_path: Path) -> bool:
    """Update type imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Type alias mappings (PascalCase -> snake_case)
        type_mappings = {
            'FloatArray': 'float_array',
            'IntArray': 'int_array',
            'BoolArray': 'bool_array',
            'FeatureArray': 'feature_array',
            'LabelArray': 'label_array',
            'GenericArray': 'generic_array',
            'Features': 'features',
            'Labels': 'labels',
            'Predictions': 'predictions',
            'Probabilities': 'probabilities',
            'Embeddings': 'embeddings',
            'Weights': 'weights',
            'ClusterLabels': 'cluster_labels',
            'ClusterCenters': 'cluster_centers',
            'DistanceMatrix': 'distance_matrix',
            'ReducedFeatures': 'reduced_features',
            'TransformMatrix': 'transform_matrix',
            'ModelConfig': 'model_config',
            'HyperParameters': 'hyper_parameters',
            'MetricsDict': 'metrics_dict',
            'ProgressCallback': 'progress_callback',
            'MetricsCallback': 'metrics_callback',
            'PipelineStep': 'pipeline_step',
            'PipelineConfig': 'pipeline_config',
        }
        
        # Update import statements
        for old_type, new_type in type_mappings.items():
            # Update from imports
            patterns = [
                # from ...types import OldType
                rf'from\s+([^\s]+\.types)\s+import\s+([^,\n]*\b{re.escape(old_type)}\b[^,\n]*)',
                # from ...types import OldType, OtherType
                rf'(\bfrom\s+[^\s]+\.types\s+import\s+[^,\n]*)\b{re.escape(old_type)}\b([^,\n]*)',
            ]
            
            for pattern in patterns:
                def replace_import(match):
                    if len(match.groups()) == 2:
                        module = match.group(1)
                        imports = match.group(2)
                        updated_imports = imports.replace(old_type, new_type)
                        return f'from {module} import {updated_imports}'
                    else:
                        prefix = match.group(1)
                        suffix = match.group(2)
                        return f'{prefix}{new_type}{suffix}'
                
                content = re.sub(pattern, replace_import, content)
            
            # Update type annotations and usage
            # Be careful with word boundaries to avoid partial matches
            content = re.sub(rf'\b{re.escape(old_type)}\b', new_type, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated type imports in {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False


def update_all_type_imports():
    """Update type imports in all Python files."""
    src_dir = Path("src/prompt_improver")
    updated_files = []
    
    for py_file in src_dir.rglob("*.py"):
        # Skip the types.py file itself
        if py_file.name == "types.py":
            continue
            
        if update_type_imports_in_file(py_file):
            updated_files.append(str(py_file))
    
    print(f"\n📊 Updated {len(updated_files)} files")
    return updated_files


if __name__ == "__main__":
    print("🔄 Updating type imports to snake_case...")
    updated_files = update_all_type_imports()
    
    if updated_files:
        print("\n📝 Files updated:")
        for file_path in updated_files[:10]:  # Show first 10
            print(f"  - {file_path}")
        if len(updated_files) > 10:
            print(f"  ... and {len(updated_files) - 10} more files")
    
    print("\n✅ Type import update complete!")
