#!/usr/bin/env python3
"""
Verification script to detect unused dependencies in the codebase.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set


def extract_imports_from_file(file_path: Path) -> Set[str]:
    """Extract all import statements from a Python file."""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    
    except (UnicodeDecodeError, SyntaxError):
        # Skip files that can't be parsed
        pass
    
    return imports


def find_all_imports(root_dir: Path) -> Set[str]:
    """Find all imports used in the codebase."""
    all_imports = set()
    
    for py_file in root_dir.rglob('*.py'):
        if 'venv' in str(py_file) or '.venv' in str(py_file):
            continue
        all_imports.update(extract_imports_from_file(py_file))
    
    return all_imports


def parse_requirements(req_file: Path) -> List[str]:
    """Parse requirements file and extract package names."""
    packages = []
    
    if not req_file.exists():
        return packages
    
    with open(req_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-e'):
                # Extract package name (before ==, >=, etc.)
                pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0]
                packages.append(pkg_name)
    
    return packages


def normalize_package_name(pkg_name: str) -> str:
    """Normalize package name for comparison."""
    # Convert dashes to underscores and lowercase
    return pkg_name.replace('-', '_').lower()


def main():
    """Main verification function."""
    root_dir = Path(__file__).parent.parent
    
    print("🔍 Analyzing dependencies usage...")
    
    # Find all imports in the codebase
    used_imports = find_all_imports(root_dir)
    used_normalized = {normalize_package_name(imp) for imp in used_imports}
    
    # Parse requirements files
    requirements_files = [
        root_dir / 'requirements.lock',
        root_dir / 'requirements-dev.txt',
        root_dir / 'requirements-test-real.txt'
    ]
    
    all_packages = set()
    for req_file in requirements_files:
        packages = parse_requirements(req_file)
        all_packages.update(packages)
    
    # Known unused packages (confirmed by analysis)
    known_unused = {
        'adal', 'azure-common', 'azure-core', 'azure-graphrbac', 'azure-mgmt-authorization',
        'azure-mgmt-containerregistry', 'azure-mgmt-core', 'azure-mgmt-keyvault',
        'azure-mgmt-network', 'azure-mgmt-resource', 'azure-mgmt-storage',
        'azureml-core', 'boto3', 'botocore', 'google-api-core', 'google-auth',
        'google-cloud-core', 'google-cloud-storage', 'google-crc32c',
        'google-resumable-media', 'googleapis-common-protos',
        'pip-audit', 'pipdeptree', 'pipreqs', 'radon',
        'knack', 'mando', 'docopt', 'yarg'
    }
    
    # Check for unused dependencies
    unused_packages = []
    potentially_unused = []
    
    for pkg in all_packages:
        pkg_normalized = normalize_package_name(pkg)
        
        if pkg in known_unused:
            unused_packages.append(pkg)
        elif pkg_normalized not in used_normalized:
            # Check if it's a known indirect dependency or system package
            if pkg_normalized in {'setuptools', 'wheel', 'pip', 'packaging', 'certifi',
                                 'urllib3', 'charset_normalizer', 'idna', 'requests',
                                 'typing_extensions', 'zipp', 'importlib_metadata'}:
                continue  # Skip system/build packages
            potentially_unused.append(pkg)
    
    # Report results
    print(f"\n✅ Found {len(used_imports)} unique imports in codebase")
    print(f"📦 Found {len(all_packages)} packages in requirements files")
    
    if unused_packages:
        print(f"\n❌ Confirmed unused packages ({len(unused_packages)}):")
        for pkg in sorted(unused_packages):
            print(f"  - {pkg}")
    
    if potentially_unused:
        print(f"\n⚠️  Potentially unused packages ({len(potentially_unused)}):")
        for pkg in sorted(potentially_unused):
            print(f"  - {pkg}")
    
    if not unused_packages and not potentially_unused:
        print("\n🎉 No unused dependencies detected!")
    
    return len(unused_packages) + len(potentially_unused)


if __name__ == '__main__':
    sys.exit(main())