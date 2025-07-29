#!/usr/bin/env python3
"""
Comprehensive Circular Import Analyzer for APES Codebase
========================================================

Modern 2025 Python tool for detecting and analyzing circular imports using both
static analysis and runtime detection. Provides detailed mapping of dependency
relationships and remediation recommendations.
"""

import ast
import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional, NamedTuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json


class ImportType(Enum):
    """Types of imports for categorization."""
    DIRECT = "direct"           # from module import item
    MODULE = "module"           # import module
    RELATIVE = "relative"       # from .module import item
    TYPE_ONLY = "type_only"     # TYPE_CHECKING imports


class CircularImportSeverity(Enum):
    """Severity levels for circular imports."""
    CRITICAL = "critical"       # Blocks runtime execution
    HIGH = "high"              # Causes import errors in some contexts
    MEDIUM = "medium"          # Potential issues, warnings
    LOW = "low"                # Minor architectural concerns


@dataclass
class ImportInfo:
    """Information about a single import statement."""
    source_module: str
    target_module: str
    import_type: ImportType
    line_number: int
    import_statement: str
    is_conditional: bool = False
    is_in_function: bool = False


@dataclass
class CircularChain:
    """Represents a circular import chain."""
    modules: List[str]
    severity: CircularImportSeverity
    import_details: List[ImportInfo]
    root_cause: str
    remediation_strategy: str


class CircularImportAnalyzer:
    """Comprehensive circular import analyzer using modern Python patterns."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_root = project_root / "src"
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.import_details: Dict[Tuple[str, str], ImportInfo] = {}
        self.circular_chains: List[CircularChain] = []
        self.module_files: Dict[str, Path] = {}
        
    def analyze_codebase(self) -> Dict[str, any]:
        """Perform comprehensive circular import analysis."""
        print("🔍 Starting comprehensive circular import analysis...")
        
        # Step 1: Discover all Python modules
        self._discover_modules()
        print(f"📁 Discovered {len(self.module_files)} Python modules")
        
        # Step 2: Parse imports from all modules
        self._parse_all_imports()
        print(f"📊 Parsed imports from {len(self.import_graph)} modules")
        
        # Step 3: Detect circular import chains
        self._detect_circular_chains()
        print(f"🔄 Found {len(self.circular_chains)} circular import chains")
        
        # Step 4: Categorize and analyze severity
        self._categorize_circular_imports()
        
        # Step 5: Generate comprehensive report
        return self._generate_analysis_report()
    
    def _discover_modules(self):
        """Discover all Python modules in the project."""
        for py_file in self.src_root.rglob("*.py"):
            if py_file.name == "__init__.py":
                # Handle package __init__.py files
                module_path = py_file.parent.relative_to(self.src_root)
                module_name = ".".join(module_path.parts) if module_path.parts else ""
            else:
                # Handle regular .py files
                module_path = py_file.relative_to(self.src_root)
                module_parts = list(module_path.parts[:-1]) + [py_file.stem]
                module_name = ".".join(module_parts)
            
            if module_name:  # Skip empty module names
                self.module_files[module_name] = py_file
    
    def _parse_all_imports(self):
        """Parse imports from all discovered modules."""
        for module_name, file_path in self.module_files.items():
            try:
                self._parse_module_imports(module_name, file_path)
            except Exception as e:
                print(f"⚠️  Error parsing {module_name}: {e}")
    
    def _parse_module_imports(self, module_name: str, file_path: Path):
        """Parse imports from a single module using AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            # Track context for conditional/function imports
            in_function = False
            in_type_checking = False
            
            for node in ast.walk(tree):
                # Track if we're in a function
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    in_function = True
                
                # Check for TYPE_CHECKING blocks
                if isinstance(node, ast.If):
                    if self._is_type_checking_block(node):
                        in_type_checking = True
                
                # Parse import statements
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._process_import_node(
                        node, module_name, in_function, in_type_checking
                    )
                    
        except SyntaxError as e:
            print(f"⚠️  Syntax error in {module_name}: {e}")
        except Exception as e:
            print(f"⚠️  Error parsing {module_name}: {e}")
    
    def _is_type_checking_block(self, node: ast.If) -> bool:
        """Check if an if statement is a TYPE_CHECKING block."""
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            return True
        if isinstance(node.test, ast.Attribute):
            if (isinstance(node.test.value, ast.Name) and 
                node.test.value.id == "typing" and 
                node.test.attr == "TYPE_CHECKING"):
                return True
        return False
    
    def _process_import_node(self, node, module_name: str, in_function: bool, in_type_checking: bool):
        """Process an import node and extract import information."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                target_module = alias.name
                self._add_import(
                    module_name, target_module, ImportType.MODULE,
                    node.lineno, f"import {alias.name}",
                    in_function, in_type_checking
                )
        
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                target_module = self._resolve_relative_import(node.module, module_name, node.level)
                import_type = ImportType.RELATIVE if node.level > 0 else ImportType.DIRECT
                
                # Create import statement string
                if node.level > 0:
                    dots = "." * node.level
                    stmt = f"from {dots}{node.module or ''} import {', '.join(alias.name for alias in node.names)}"
                else:
                    stmt = f"from {node.module} import {', '.join(alias.name for alias in node.names)}"
                
                self._add_import(
                    module_name, target_module, import_type,
                    node.lineno, stmt, in_function, in_type_checking
                )
    
    def _resolve_relative_import(self, module: str, current_module: str, level: int) -> str:
        """Resolve relative imports to absolute module names."""
        if level == 0:
            return module
        
        current_parts = current_module.split('.')
        if level > len(current_parts):
            return module or ""
        
        base_parts = current_parts[:-level] if level < len(current_parts) else []
        if module:
            return '.'.join(base_parts + [module])
        else:
            return '.'.join(base_parts)
    
    def _add_import(self, source: str, target: str, import_type: ImportType, 
                   line_number: int, statement: str, in_function: bool, in_type_checking: bool):
        """Add an import to the graph and details."""
        if target and target in self.module_files:
            self.import_graph[source].add(target)
            
            import_info = ImportInfo(
                source_module=source,
                target_module=target,
                import_type=ImportType.TYPE_ONLY if in_type_checking else import_type,
                line_number=line_number,
                import_statement=statement,
                is_conditional=in_type_checking,
                is_in_function=in_function
            )
            
            self.import_details[(source, target)] = import_info
    
    def _detect_circular_chains(self):
        """Detect all circular import chains using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                self._create_circular_chain(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.import_graph.get(node, set()):
                if neighbor in self.module_files:
                    dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for module in self.import_graph:
            if module not in visited:
                dfs(module, [])
    
    def _create_circular_chain(self, cycle: List[str]):
        """Create a CircularChain object from a detected cycle."""
        # Get import details for the cycle
        import_details = []
        for i in range(len(cycle) - 1):
            source, target = cycle[i], cycle[i + 1]
            if (source, target) in self.import_details:
                import_details.append(self.import_details[(source, target)])
        
        # Determine severity
        severity = self._determine_severity(import_details)
        
        # Identify root cause
        root_cause = self._identify_root_cause(cycle, import_details)
        
        # Suggest remediation strategy
        remediation = self._suggest_remediation(cycle, import_details)
        
        circular_chain = CircularChain(
            modules=cycle[:-1],  # Remove duplicate last module
            severity=severity,
            import_details=import_details,
            root_cause=root_cause,
            remediation_strategy=remediation
        )
        
        self.circular_chains.append(circular_chain)
    
    def _determine_severity(self, import_details: List[ImportInfo]) -> CircularImportSeverity:
        """Determine the severity of a circular import chain."""
        # Critical: Direct imports at module level
        if any(not detail.is_in_function and not detail.is_conditional 
               and detail.import_type in [ImportType.DIRECT, ImportType.MODULE] 
               for detail in import_details):
            return CircularImportSeverity.CRITICAL
        
        # High: Some conditional imports but still problematic
        if any(not detail.is_conditional for detail in import_details):
            return CircularImportSeverity.HIGH
        
        # Medium: Mostly conditional but architectural concern
        if any(detail.import_type != ImportType.TYPE_ONLY for detail in import_details):
            return CircularImportSeverity.MEDIUM
        
        # Low: Only TYPE_CHECKING imports
        return CircularImportSeverity.LOW
    
    def _identify_root_cause(self, cycle: List[str], import_details: List[ImportInfo]) -> str:
        """Identify the root cause of the circular import."""
        # Analyze the modules and import patterns
        if any("database" in module and "core" in module for module in cycle):
            return "Database-Core circular dependency"
        elif any("metrics" in module and "performance" in module for module in cycle):
            return "Metrics-Performance circular dependency"
        elif any("ml" in module and "database" in module for module in cycle):
            return "ML-Database circular dependency"
        elif len(cycle) > 4:
            return "Complex multi-module circular dependency"
        else:
            return "Simple circular dependency between related modules"
    
    def _suggest_remediation(self, cycle: List[str], import_details: List[ImportInfo]) -> str:
        """Suggest remediation strategy for the circular import."""
        strategies = []
        
        # Check for TYPE_CHECKING opportunities
        if any(not detail.is_conditional for detail in import_details):
            strategies.append("Convert type-only imports to TYPE_CHECKING blocks")
        
        # Check for lazy import opportunities
        if any(not detail.is_in_function for detail in import_details):
            strategies.append("Move runtime imports to lazy loading functions")
        
        # Check for dependency injection opportunities
        if any("__init__" in detail.import_statement for detail in import_details):
            strategies.append("Implement dependency injection pattern")
        
        # Check for module restructuring needs
        if len(cycle) > 3:
            strategies.append("Extract shared dependencies to separate module")
        
        return "; ".join(strategies) if strategies else "Manual analysis required"
    
    def _categorize_circular_imports(self):
        """Categorize circular imports by severity and impact."""
        severity_counts = defaultdict(int)
        for chain in self.circular_chains:
            severity_counts[chain.severity] += 1
        
        print(f"\n📊 Circular Import Summary:")
        for severity in CircularImportSeverity:
            count = severity_counts[severity]
            if count > 0:
                print(f"   {severity.value.upper()}: {count} chains")
    
    def _generate_analysis_report(self) -> Dict[str, any]:
        """Generate comprehensive analysis report."""
        return {
            "summary": {
                "total_modules": len(self.module_files),
                "total_imports": sum(len(imports) for imports in self.import_graph.values()),
                "circular_chains": len(self.circular_chains),
                "severity_breakdown": {
                    severity.value: sum(1 for chain in self.circular_chains 
                                      if chain.severity == severity)
                    for severity in CircularImportSeverity
                }
            },
            "circular_chains": [
                {
                    "modules": chain.modules,
                    "severity": chain.severity.value,
                    "root_cause": chain.root_cause,
                    "remediation_strategy": chain.remediation_strategy,
                    "import_details": [
                        {
                            "source": detail.source_module,
                            "target": detail.target_module,
                            "type": detail.import_type.value,
                            "line": detail.line_number,
                            "statement": detail.import_statement,
                            "conditional": detail.is_conditional,
                            "in_function": detail.is_in_function
                        }
                        for detail in chain.import_details
                    ]
                }
                for chain in self.circular_chains
            ],
            "import_graph": {
                module: list(imports) 
                for module, imports in self.import_graph.items()
            }
        }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    analyzer = CircularImportAnalyzer(project_root)
    
    try:
        report = analyzer.analyze_codebase()
        
        # Save detailed report
        report_file = project_root / "circular_import_analysis.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        print(f"🎯 Found {len(analyzer.circular_chains)} circular import chains")
        
        # Print critical issues
        critical_chains = [c for c in analyzer.circular_chains 
                          if c.severity == CircularImportSeverity.CRITICAL]
        if critical_chains:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_chains)}):")
            for chain in critical_chains[:5]:  # Show first 5
                print(f"   {' → '.join(chain.modules)}")
                print(f"   Cause: {chain.root_cause}")
                print(f"   Fix: {chain.remediation_strategy}\n")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)
