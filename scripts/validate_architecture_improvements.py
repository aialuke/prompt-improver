#!/usr/bin/env python3
"""
Architecture Validation Script
Validates that the proposed architecture improvements resolve circular dependencies
and follow clean architecture principles.
"""

import ast
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple


def extract_imports_from_file(file_path: Path) -> Dict[str, List[str]]:
    """Extract imports from a Python file using AST parsing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = {
            'absolute': [],
            'relative': [],
            'from_imports': [],
            'relative_from': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports['absolute'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                level = node.level
                
                if level > 0:  # Relative import
                    for alias in node.names:
                        imports['relative_from'].append({
                            'module': module,
                            'name': alias.name,
                            'level': level
                        })
                else:  # Absolute import
                    for alias in node.names:
                        imports['from_imports'].append({
                            'module': module,
                            'name': alias.name
                        })
        
        return imports
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {'absolute': [], 'relative': [], 'from_imports': [], 'relative_from': []}


def build_dependency_graph(src_dir: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Dict]]:
    """Build a comprehensive dependency graph from source code."""
    graph = defaultdict(set)
    file_imports = {}
    
    # Walk through all Python files
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
            
        rel_path = py_file.relative_to(src_dir)
        module_name = str(rel_path).replace('/', '.').replace('.py', '')
        
        imports = extract_imports_from_file(py_file)
        file_imports[module_name] = {
            'path': str(py_file),
            'imports': imports
        }
        
        # Process imports to build graph
        for imp in imports.get('absolute', []):
            if imp.startswith('prompt_improver'):
                target = imp.replace('prompt_improver.', '')
                graph[module_name].add(target)
        
        for imp in imports.get('from_imports', []):
            if imp['module'].startswith('prompt_improver'):
                target = imp['module'].replace('prompt_improver.', '')
                graph[module_name].add(target)
        
        # Handle relative imports
        for imp in imports.get('relative_from', []):
            current_parts = module_name.split('.')
            level = imp['level']
            if level <= len(current_parts):
                target_parts = current_parts[:-level] if level > 0 else current_parts
                if imp['module']:
                    target_parts.extend(imp['module'].split('.'))
                target = '.'.join(target_parts)
                graph[module_name].add(target)
    
    return dict(graph), file_imports


def find_circular_dependencies(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Find circular dependencies using DFS."""
    cycles = []
    visited = set()
    rec_stack = set()
    path = []
    
    def dfs(node: str):
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return
        
        if node in visited:
            return
            
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            dfs(neighbor)
        
        path.pop()
        rec_stack.remove(node)
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return cycles


def validate_clean_architecture_layers(graph: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Validate that dependencies follow clean architecture layer rules."""
    violations = {
        'domain_violations': [],
        'application_violations': [],
        'infrastructure_violations': [],
        'presentation_violations': []
    }
    
    # Define layer patterns
    layer_patterns = {
        'domain': ['domain.', 'shared.interfaces.', 'shared.types.'],
        'application': ['application.', 'domain.', 'shared.'],
        'infrastructure': ['infrastructure.', 'application.', 'domain.', 'shared.'],
        'presentation': ['presentation.', 'application.', 'domain.', 'shared.']
    }
    
    for module, dependencies in graph.items():
        # Determine module layer
        module_layer = None
        for layer, patterns in layer_patterns.items():
            if any(pattern in module for pattern in patterns):
                module_layer = layer
                break
        
        if not module_layer:
            continue
            
        # Check if dependencies violate layer rules
        for dep in dependencies:
            dep_layer = None
            for layer, patterns in layer_patterns.items():
                if any(pattern in dep for pattern in patterns):
                    dep_layer = layer
                    break
            
            if not dep_layer:
                continue
            
            # Check for violations
            if module_layer == 'domain' and dep_layer in ['application', 'infrastructure', 'presentation']:
                violations['domain_violations'].append(f"{module} -> {dep}")
            elif module_layer == 'application' and dep_layer in ['infrastructure', 'presentation']:
                violations['application_violations'].append(f"{module} -> {dep}")
    
    return violations


def check_interface_usage(file_imports: Dict[str, Dict]) -> Dict[str, int]:
    """Check how well interfaces are being used instead of concrete classes."""
    interface_usage = {
        'interface_imports': 0,
        'concrete_imports': 0,
        'total_imports': 0
    }
    
    for module, data in file_imports.items():
        imports = data['imports']
        
        for imp in imports.get('from_imports', []):
            if 'prompt_improver' in imp['module']:
                interface_usage['total_imports'] += 1
                
                if 'interfaces' in imp['module'] or imp['name'].startswith('I'):
                    interface_usage['interface_imports'] += 1
                else:
                    interface_usage['concrete_imports'] += 1
    
    return interface_usage


def validate_dependency_injection_usage(file_imports: Dict[str, Dict]) -> Dict[str, int]:
    """Check usage of dependency injection patterns."""
    di_usage = {
        'files_with_inject_decorator': 0,
        'files_with_provide_imports': 0,
        'files_with_container_imports': 0,
        'total_files': len(file_imports)
    }
    
    for module, data in file_imports.items():
        imports = data['imports']
        
        # Check for dependency injection imports
        for imp in imports.get('from_imports', []):
            if 'dependency_injector' in imp['module']:
                if imp['name'] in ['inject', 'Provide']:
                    di_usage['files_with_provide_imports'] += 1
                if 'container' in imp['module'].lower():
                    di_usage['files_with_container_imports'] += 1
        
        # Check file content for @inject decorator (simplified check)
        try:
            with open(data['path'], 'r') as f:
                content = f.read()
                if '@inject' in content:
                    di_usage['files_with_inject_decorator'] += 1
        except:
            pass
    
    return di_usage


def main():
    """Main validation function."""
    print("🔍 Architecture Validation Report")
    print("=" * 50)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src" / "prompt_improver"
    
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        sys.exit(1)
    
    # Build dependency graph
    print("📊 Building dependency graph...")
    graph, file_imports = build_dependency_graph(src_dir)
    
    print(f"✅ Analyzed {len(file_imports)} Python files")
    print(f"✅ Found {len(graph)} modules with dependencies")
    
    # Check for circular dependencies
    print("\n🔄 Checking for circular dependencies...")
    cycles = find_circular_dependencies(graph)
    
    if cycles:
        print(f"❌ Found {len(cycles)} circular dependencies:")
        for i, cycle in enumerate(cycles[:5], 1):  # Show first 5
            print(f"   {i}. {' -> '.join(cycle)}")
        if len(cycles) > 5:
            print(f"   ... and {len(cycles) - 5} more")
    else:
        print("✅ No circular dependencies found!")
    
    # Validate clean architecture
    print("\n🏗️  Validating clean architecture layers...")
    violations = validate_clean_architecture_layers(graph)
    
    total_violations = sum(len(v) for v in violations.values())
    if total_violations == 0:
        print("✅ No clean architecture violations found!")
    else:
        print(f"❌ Found {total_violations} layer violations:")
        for violation_type, violation_list in violations.items():
            if violation_list:
                print(f"   {violation_type}: {len(violation_list)} violations")
                for violation in violation_list[:3]:  # Show first 3
                    print(f"     - {violation}")
                if len(violation_list) > 3:
                    print(f"     ... and {len(violation_list) - 3} more")
    
    # Check interface usage
    print("\n🔌 Checking interface usage...")
    interface_stats = check_interface_usage(file_imports)
    
    if interface_stats['total_imports'] > 0:
        interface_percentage = (interface_stats['interface_imports'] / 
                             interface_stats['total_imports']) * 100
        print(f"📈 Interface usage: {interface_percentage:.1f}% "
              f"({interface_stats['interface_imports']}/{interface_stats['total_imports']})")
        
        if interface_percentage >= 70:
            print("✅ Good interface usage!")
        elif interface_percentage >= 40:
            print("⚠️  Moderate interface usage - room for improvement")
        else:
            print("❌ Low interface usage - needs improvement")
    else:
        print("ℹ️  No internal imports found")
    
    # Check dependency injection usage
    print("\n💉 Checking dependency injection usage...")
    di_stats = validate_dependency_injection_usage(file_imports)
    
    di_percentage = (di_stats['files_with_inject_decorator'] / 
                    di_stats['total_files']) * 100
    print(f"📈 Dependency injection usage: {di_percentage:.1f}% "
          f"({di_stats['files_with_inject_decorator']}/{di_stats['total_files']} files)")
    
    if di_percentage >= 50:
        print("✅ Good dependency injection adoption!")
    elif di_percentage >= 20:
        print("⚠️  Moderate DI usage - continue migration")
    else:
        print("❌ Low DI usage - needs implementation")
    
    # Top 10 most coupled modules
    print("\n📊 Most coupled modules:")
    sorted_deps = sorted(graph.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (module, deps) in enumerate(sorted_deps[:5], 1):
        print(f"   {i}. {module}: {len(deps)} dependencies")
    
    # Overall assessment
    print("\n📋 Overall Assessment:")
    
    score = 0
    max_score = 4
    
    if len(cycles) == 0:
        score += 1
        print("✅ No circular dependencies")
    else:
        print("❌ Circular dependencies present")
    
    if total_violations == 0:
        score += 1
        print("✅ Clean architecture layers respected")
    else:
        print("❌ Clean architecture violations present")
    
    if interface_stats['total_imports'] > 0 and interface_percentage >= 50:
        score += 1
        print("✅ Good interface usage")
    else:
        print("❌ Needs better interface usage")
    
    if di_percentage >= 30:
        score += 1
        print("✅ Dependency injection being adopted")
    else:
        print("❌ Needs dependency injection implementation")
    
    print(f"\n🎯 Architecture Score: {score}/{max_score} ({score/max_score*100:.0f}%)")
    
    if score == max_score:
        print("🎉 Excellent! Architecture follows best practices.")
    elif score >= max_score * 0.75:
        print("👍 Good architecture with minor improvements needed.")
    elif score >= max_score * 0.5:
        print("⚠️  Architecture needs significant improvements.")
    else:
        print("🚨 Architecture needs major refactoring.")
    
    # Save detailed results
    results = {
        'circular_dependencies': cycles,
        'architecture_violations': violations,
        'interface_usage': interface_stats,
        'dependency_injection_usage': di_stats,
        'most_coupled_modules': [(mod, len(deps)) for mod, deps in sorted_deps[:10]],
        'overall_score': f"{score}/{max_score}"
    }
    
    results_file = project_root / "architecture_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return score == max_score


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)