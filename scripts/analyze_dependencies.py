#!/usr/bin/env python3
"""
Comprehensive Dependency Analysis Tool for Prompt Improver

Analyzes circular dependencies, architectural violations, and coupling issues
in the codebase to enable systematic refactoring.

Features:
- Circular dependency detection with cycle visualization
- Architectural layer violation analysis  
- Module coupling metrics and recommendations
- Interface abstraction opportunities identification
- Automated compliance testing framework
"""

import ast
import os
import json
import sys
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any

# Optional dependencies for visualization
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Warning: networkx and matplotlib not available. Graph generation disabled.")

from datetime import datetime

@dataclass
class DependencyMetrics:
    """Metrics for dependency analysis"""
    total_modules: int
    total_dependencies: int
    circular_dependencies: List[List[str]]
    high_coupling_modules: List[Tuple[str, int]]
    architectural_violations: List[Dict[str, Any]]
    interface_opportunities: List[Dict[str, Any]]
    
class ArchitecturalLayers:
    """Define the expected architectural layers for compliance checking"""
    
    LAYERS = {
        'infrastructure': ['database', 'cache', 'security', 'utils'],
        'core': ['core', 'shared'],
        'domain': ['rule_engine', 'ml'],
        'application': ['performance', 'monitoring', 'metrics'],
        'interface': ['api', 'cli', 'tui', 'mcp_server'],
        'external': ['feedback', 'dashboard']
    }
    
    # Dependencies should only flow downward or stay within layer
    ALLOWED_DEPENDENCIES = {
        'external': ['interface', 'application', 'domain', 'core', 'infrastructure'],
        'interface': ['application', 'domain', 'core', 'infrastructure'],
        'application': ['domain', 'core', 'infrastructure'],
        'domain': ['core', 'infrastructure'],
        'core': ['infrastructure'],
        'infrastructure': []
    }

class DependencyAnalyzer:
    """Main dependency analysis engine"""
    
    def __init__(self, src_path: Path):
        self.src_path = src_path
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.module_to_layer: Dict[str, str] = {}
        self._build_dependency_graph()
        self._classify_modules()
    
    def _extract_imports(self, file_path: Path) -> List[str]:
        """Extract all prompt_improver imports from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith('prompt_improver'):
                            imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith('prompt_improver'):
                        imports.append(node.module)
            
            return imports
        except Exception as e:
            print(f'Warning: Error parsing {file_path}: {e}')
            return []
    
    def _build_dependency_graph(self):
        """Build comprehensive dependency graph"""
        for py_file in self.src_path.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
                
            # Convert file path to module name
            try:
                relative_path = py_file.relative_to(Path('src'))
                module_name = str(relative_path.with_suffix('').as_posix()).replace('/', '.')
                
                imports = self._extract_imports(py_file)
                for imp in imports:
                    self.dependencies[module_name].add(imp)
                    self.reverse_dependencies[imp].add(module_name)
            except ValueError:
                # Handle files outside src directory
                continue
    
    def _classify_modules(self):
        """Classify modules into architectural layers"""
        for module in self.dependencies.keys():
            parts = module.split('.')
            if len(parts) >= 2:
                # Extract the second level (e.g., 'database' from 'prompt_improver.database.models')
                module_category = parts[1] if len(parts) > 1 else parts[0]
                
                # Find which layer this module belongs to
                for layer, categories in ArchitecturalLayers.LAYERS.items():
                    if module_category in categories:
                        self.module_to_layer[module] = layer
                        break
                else:
                    self.module_to_layer[module] = 'unknown'
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect all circular dependencies using strongly connected components"""
        # Create NetworkX graph
        G = nx.DiGraph()
        for module, deps in self.dependencies.items():
            for dep in deps:
                G.add_edge(module, dep)
        
        # Find strongly connected components with more than 1 node
        cycles = []
        for component in nx.strongly_connected_components(G):
            if len(component) > 1:
                # Find the actual cycle within this component
                subgraph = G.subgraph(component)
                try:
                    cycle = nx.find_cycle(subgraph)
                    cycle_nodes = [edge[0] for edge in cycle] + [cycle[-1][1]]
                    cycles.append(cycle_nodes)
                except nx.NetworkXNoCycle:
                    continue
        
        return cycles
    
    def find_architectural_violations(self) -> List[Dict[str, Any]]:
        """Find dependencies that violate architectural layers"""
        violations = []
        
        for module, deps in self.dependencies.items():
            module_layer = self.module_to_layer.get(module, 'unknown')
            
            for dep in deps:
                dep_layer = self.module_to_layer.get(dep, 'unknown')
                
                # Check if this dependency is allowed
                allowed_layers = ArchitecturalLayers.ALLOWED_DEPENDENCIES.get(module_layer, [])
                
                if dep_layer not in allowed_layers and dep_layer != module_layer:
                    violations.append({
                        'from_module': module,
                        'from_layer': module_layer,
                        'to_module': dep,
                        'to_layer': dep_layer,
                        'violation_type': 'layer_breach',
                        'severity': self._calculate_violation_severity(module_layer, dep_layer)
                    })
        
        return violations
    
    def _calculate_violation_severity(self, from_layer: str, to_layer: str) -> str:
        """Calculate severity of architectural violation"""
        layer_order = ['infrastructure', 'core', 'domain', 'application', 'interface', 'external']
        
        try:
            from_idx = layer_order.index(from_layer)
            to_idx = layer_order.index(to_layer)
            
            if from_idx < to_idx:
                return 'critical'  # Lower layer depending on higher layer
            elif from_idx == to_idx:
                return 'low'  # Same layer dependency
            else:
                return 'medium'  # Normal downward dependency but might be too deep
        except ValueError:
            return 'unknown'
    
    def identify_high_coupling_modules(self, threshold: int = 10) -> List[Tuple[str, int]]:
        """Identify modules with high coupling (too many dependencies)"""
        high_coupling = []
        for module, deps in self.dependencies.items():
            dep_count = len(deps)
            if dep_count >= threshold:
                high_coupling.append((module, dep_count))
        
        return sorted(high_coupling, key=lambda x: x[1], reverse=True)
    
    def identify_interface_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for interface abstraction"""
        opportunities = []
        
        # Find modules that are heavily depended upon (potential interface candidates)
        for module, dependents in self.reverse_dependencies.items():
            if len(dependents) >= 3:  # Arbitrary threshold
                opportunities.append({
                    'module': module,
                    'dependent_count': len(dependents),
                    'dependents': list(dependents),
                    'recommendation': 'Consider creating interface/protocol for this module',
                    'potential_interface_name': f"I{module.split('.')[-1].title()}"
                })
        
        return sorted(opportunities, key=lambda x: x['dependent_count'], reverse=True)
    
    def analyze(self) -> DependencyMetrics:
        """Perform comprehensive dependency analysis"""
        circular_deps = self.detect_circular_dependencies()
        architectural_violations = self.find_architectural_violations()
        high_coupling = self.identify_high_coupling_modules()
        interface_opportunities = self.identify_interface_opportunities()
        
        return DependencyMetrics(
            total_modules=len(self.dependencies),
            total_dependencies=sum(len(deps) for deps in self.dependencies.values()),
            circular_dependencies=circular_deps,
            high_coupling_modules=high_coupling,
            architectural_violations=architectural_violations,
            interface_opportunities=interface_opportunities
        )
    
    def generate_dependency_graph(self, output_path: Path, max_nodes: int = 50):
        """Generate visual dependency graph"""
        if not HAS_VISUALIZATION:
            print("Skipping graph generation: visualization libraries not available")
            return
            
        G = nx.DiGraph()
        
        # Add only the most connected nodes to avoid clutter
        top_modules = sorted(
            self.dependencies.items(), 
            key=lambda x: len(x[1]) + len(self.reverse_dependencies[x[0]]),
            reverse=True
        )[:max_nodes]
        
        for module, deps in top_modules:
            for dep in deps:
                if dep in dict(top_modules):
                    G.add_edge(module.split('.')[-1], dep.split('.')[-1])
        
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Color nodes by layer
        colors = {
            'infrastructure': 'lightblue',
            'core': 'lightgreen', 
            'domain': 'yellow',
            'application': 'orange',
            'interface': 'red',
            'external': 'purple',
            'unknown': 'gray'
        }
        
        node_colors = [colors.get(self.module_to_layer.get(f"prompt_improver.{node}", 'unknown'), 'gray') for node in G.nodes()]
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=1000, font_size=8, arrows=True)
        
        plt.title("Module Dependency Graph (Top Connected Modules)")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

class RefactoringRecommendations:
    """Generate specific refactoring recommendations"""
    
    @staticmethod
    def generate_recommendations(metrics: DependencyMetrics) -> Dict[str, List[str]]:
        """Generate actionable refactoring recommendations"""
        recommendations = {
            'immediate': [],
            'short_term': [],
            'long_term': []
        }
        
        # Immediate actions for circular dependencies
        if metrics.circular_dependencies:
            recommendations['immediate'].extend([
                f"CRITICAL: Break circular dependency: {' -> '.join(cycle)}"
                for cycle in metrics.circular_dependencies
            ])
        
        # Short-term actions for high coupling
        for module, count in metrics.high_coupling_modules[:5]:
            recommendations['short_term'].append(
                f"Reduce dependencies in {module} (currently {count}): "
                f"Extract service interfaces and use dependency injection"
            )
        
        # Long-term architectural improvements
        critical_violations = [v for v in metrics.architectural_violations if v['severity'] == 'critical']
        if critical_violations:
            recommendations['long_term'].extend([
                f"Architectural violation: {v['from_module']} ({v['from_layer']}) "
                f"should not depend on {v['to_module']} ({v['to_layer']})"
                for v in critical_violations[:3]
            ])
        
        # Interface creation opportunities
        for opportunity in metrics.interface_opportunities[:3]:
            recommendations['long_term'].append(
                f"Create interface {opportunity['potential_interface_name']} "
                f"for {opportunity['module']} (used by {opportunity['dependent_count']} modules)"
            )
        
        return recommendations

def main():
    """Main entry point for dependency analysis"""
    parser = argparse.ArgumentParser(description='Analyze codebase dependencies')
    parser.add_argument('--src-path', type=Path, default=Path('src/prompt_improver'),
                        help='Path to source code directory')
    parser.add_argument('--output-dir', type=Path, default=Path('.'),
                        help='Output directory for reports')
    parser.add_argument('--generate-graph', action='store_true',
                        help='Generate dependency visualization')
    parser.add_argument('--format', choices=['json', 'text'], default='text',
                        help='Output format')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(exist_ok=True)
    
    # Run analysis
    analyzer = DependencyAnalyzer(args.src_path)
    metrics = analyzer.analyze()
    recommendations = RefactoringRecommendations.generate_recommendations(metrics)
    
    # Generate reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.format == 'json':
        # JSON report
        report = {
            'timestamp': timestamp,
            'metrics': asdict(metrics),
            'recommendations': recommendations
        }
        
        output_file = args.output_dir / f'dependency_analysis_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"JSON report saved to: {output_file}")
    
    else:
        # Text report
        output_file = args.output_dir / f'dependency_analysis_{timestamp}.txt'
        with open(output_file, 'w') as f:
            f.write(f"DEPENDENCY ANALYSIS REPORT\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"=" * 50 + "\n\n")
            
            f.write(f"OVERVIEW:\n")
            f.write(f"- Total modules: {metrics.total_modules}\n")
            f.write(f"- Total dependencies: {metrics.total_dependencies}\n")
            f.write(f"- Circular dependencies: {len(metrics.circular_dependencies)}\n")
            f.write(f"- High coupling modules: {len(metrics.high_coupling_modules)}\n")
            f.write(f"- Architectural violations: {len(metrics.architectural_violations)}\n\n")
            
            if metrics.circular_dependencies:
                f.write(f"CIRCULAR DEPENDENCIES:\n")
                for i, cycle in enumerate(metrics.circular_dependencies, 1):
                    f.write(f"  {i}. {' -> '.join(cycle)}\n")
                f.write("\n")
            
            if metrics.high_coupling_modules:
                f.write(f"HIGH COUPLING MODULES:\n")
                for module, count in metrics.high_coupling_modules[:10]:
                    f.write(f"  {module}: {count} dependencies\n")
                f.write("\n")
            
            if metrics.architectural_violations:
                f.write(f"ARCHITECTURAL VIOLATIONS:\n")
                for violation in metrics.architectural_violations[:10]:
                    f.write(f"  {violation['severity'].upper()}: "
                           f"{violation['from_module']} -> {violation['to_module']}\n")
                f.write("\n")
            
            f.write(f"REFACTORING RECOMMENDATIONS:\n\n")
            f.write(f"Immediate Actions:\n")
            for rec in recommendations['immediate']:
                f.write(f"  - {rec}\n")
            
            f.write(f"\nShort-term Actions:\n")
            for rec in recommendations['short_term']:
                f.write(f"  - {rec}\n")
            
            f.write(f"\nLong-term Actions:\n")
            for rec in recommendations['long_term']:
                f.write(f"  - {rec}\n")
        
        print(f"Text report saved to: {output_file}")
    
    # Generate dependency graph if requested
    if args.generate_graph:
        graph_file = args.output_dir / f'dependency_graph_{timestamp}.png'
        analyzer.generate_dependency_graph(graph_file)
        print(f"Dependency graph saved to: {graph_file}")
    
    # Print summary to console
    print(f"\nANALYSIS SUMMARY:")
    print(f"- Found {len(metrics.circular_dependencies)} circular dependencies")
    print(f"- Identified {len(metrics.high_coupling_modules)} high-coupling modules")
    print(f"- Detected {len(metrics.architectural_violations)} architectural violations")
    print(f"- Suggested {len(metrics.interface_opportunities)} interface opportunities")
    
    return len(metrics.circular_dependencies) == 0  # Return success if no circular dependencies

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)