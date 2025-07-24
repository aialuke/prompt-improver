#!/usr/bin/env python3
"""Script to add proper type annotations to ML module files."""

import os
import re
from pathlib import Path
from typing import List, Tuple

def add_type_imports(content: str) -> str:
    """Add ML type imports if they're not already present."""
    # Check if types are already imported
    if "from ...types import" in content or "from ..types import" in content:
        return content
    
    # Find the right place to add imports (after other imports)
    lines = content.split('\n')
    import_end_idx = 0
    
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_end_idx = i + 1
        elif import_end_idx > 0 and line and not line.startswith(' '):
            # Found non-import line after imports
            break
    
    # Determine relative import level
    if '/optimization/algorithms/' in str(Path(__file__)):
        import_line = "\nfrom ...types import features, labels, cluster_labels, weights, float_array, int_array, metrics_dict"
    elif '/learning/algorithms/' in str(Path(__file__)):
        import_line = "\nfrom ...types import features, labels, cluster_labels, weights, float_array, int_array, metrics_dict"
    else:
        import_line = "\nfrom ..types import features, labels, cluster_labels, weights, float_array, int_array, metrics_dict"
    
    lines.insert(import_end_idx, import_line)
    return '\n'.join(lines)

def fix_numpy_annotations(content: str) -> str:
    """Replace np.ndarray with proper type annotations."""
    replacements = [
        # Function parameter annotations
        (r'features:\s*np\.ndarray', 'features: features'),
        (r'labels:\s*np\.ndarray\s*\|', 'labels: Optional[labels] |'),
        (r'labels:\s*Optional\[np\.ndarray\]', 'labels: Optional[labels]'),
        (r'sample_weights:\s*np\.ndarray\s*\|', 'sample_weights: Optional[weights] |'),
        (r'sample_weights:\s*Optional\[np\.ndarray\]', 'sample_weights: Optional[weights]'),
        (r'cluster_labels:\s*np\.ndarray', 'cluster_labels: cluster_labels'),
        (r'cluster_centers:\s*Optional\[np\.ndarray\]', 'cluster_centers: Optional[cluster_centers]'),
        
        # Return type annotations
        (r'->\s*np\.ndarray:', '-> features:'),
        (r'->\s*Optional\[np\.ndarray\]:', '-> Optional[features]:'),
        (r'->\s*dict\[str,\s*Any\]:', '-> Dict[str, Any]:'),
        
        # Type hints in dataclasses
        (r':\s*np\.ndarray\s*\n', ': float_array\n'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    return content

def process_file(filepath: Path) -> bool:
    """Process a single Python file to fix type annotations."""
    try:
        content = filepath.read_text()
        original_content = content
        
        # Add imports if needed
        content = add_type_imports(content)
        
        # Fix numpy annotations
        content = fix_numpy_annotations(content)
        
        # Add Optional import if needed
        if 'Optional[' in content and 'from typing import' in content:
            typing_import_pattern = r'from typing import ([^)]+)'
            match = re.search(typing_import_pattern, content)
            if match and 'Optional' not in match.group(1):
                imports = match.group(1).strip()
                new_imports = f"{imports}, Optional"
                content = re.sub(typing_import_pattern, f'from typing import {new_imports}', content)
        
        if content != original_content:
            filepath.write_text(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main function to process all ML files."""
    ml_dir = Path(__file__).parent
    files_to_process = [
        'learning/algorithms/context_aware_weighter.py',
        'learning/algorithms/failure_analyzer.py',
        'optimization/algorithms/dimensionality_reducer.py',
        'optimization/algorithms/rule_optimizer.py',
        'optimization/algorithms/clustering_optimizer.py',
        'optimization/algorithms/multi_armed_bandit.py',
        'optimization/validation/optimization_validator.py',
        'evaluation/pattern_significance_analyzer.py',
        'evaluation/advanced_statistical_validator.py',
    ]
    
    processed_count = 0
    for file_path in files_to_process:
        full_path = ml_dir / file_path
        if full_path.exists():
            if process_file(full_path):
                print(f"✓ Updated {file_path}")
                processed_count += 1
            else:
                print(f"- No changes needed for {file_path}")
        else:
            print(f"✗ File not found: {file_path}")
    
    print(f"\nProcessed {processed_count} files")

if __name__ == "__main__":
    main()