"""Performance Analysis and Code Quality Metrics Collection for 2025.

This module provides comprehensive performance analysis, code quality metrics,
and regression detection tools following 2025 best practices for Python
applications with ML workloads.

Features:
- Real-time performance monitoring
- Code quality metrics collection
- Performance regression detection
- Dependency analysis and optimization
- Memory usage tracking
- ML-specific performance metrics
- Automated baseline establishment
- Performance trend analysis
"""

import ast
import logging
import psutil
import time
import tracemalloc
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union,
    Protocol, runtime_checkable, TypeVar, Generic
)
from contextlib import contextmanager
import asyncio

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from .types import (
        PerformanceMetrics, MetricPoint, MetricType, 
        PositiveFloat, NonNegativeInt, PromptImproverError
    )
    TYPES_AVAILABLE = True
except ImportError:
    TYPES_AVAILABLE = False
    PerformanceMetrics = dict
    MetricPoint = dict
    MetricType = str
    PositiveFloat = float
    NonNegativeInt = int
    PromptImproverError = Exception

F = TypeVar('F', bound=Callable[..., Any])

@dataclass
class CodeQualityMetrics:
    """Code quality metrics for a module or function."""
    
    module_name: str
    function_name: Optional[str] = None
    
    # Complexity metrics
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    lines_of_code: int = 0
    maintainability_index: float = 0.0
    
    # Code metrics
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    docstring_coverage: float = 0.0
    
    # Quality scores
    code_quality_score: float = 0.0
    test_coverage: float = 0.0
    type_coverage: float = 0.0
    
    # Performance indicators
    memory_footprint_kb: int = 0
    import_time_ms: float = 0.0
    
    # Timestamp
    measured_at: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    
    operation_name: str
    component: str
    
    # Timing baselines
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    
    # Resource baselines
    avg_memory_mb: float
    peak_memory_mb: float
    avg_cpu_percent: float
    
    # Throughput baselines
    requests_per_second: float
    success_rate: float
    
    # Quality baselines
    code_quality_score: float
    test_coverage: float
    
    # Metadata
    sample_count: int
    established_at: datetime
    confidence_level: float = 0.95

@dataclass
class RegressionAlert:
    """Performance regression alert."""
    
    operation_name: str
    metric_name: str
    baseline_value: float
    current_value: float
    regression_percent: float
    severity: str  # 'minor', 'major', 'critical'
    detected_at: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

class PerformanceTimer:
    """High-precision performance timer with context management."""
    
    def __init__(self, operation_name: str, component: str = "unknown"):
        self.operation_name = operation_name
        self.component = component
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_memory: Optional[int] = None
        self.peak_memory: Optional[int] = None
        self.cpu_start: Optional[float] = None
        
    def __enter__(self) -> 'PerformanceTimer':
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def start(self) -> None:
        """Start timing the operation."""
        # Enable memory tracing if not already enabled
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        self.start_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        self.start_memory = current
        
        # Get CPU time
        process = psutil.Process()
        self.cpu_start = process.cpu_percent()
    
    def stop(self) -> PerformanceMetrics:
        """Stop timing and return metrics."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.perf_counter()
        
        # Get memory usage
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            self.peak_memory = peak
        
        # Calculate metrics
        response_time_ms = (self.end_time - self.start_time) * 1000
        memory_mb = (self.peak_memory - self.start_memory) / 1024 / 1024 if self.peak_memory and self.start_memory else 0
        
        if TYPES_AVAILABLE:
            return PerformanceMetrics(
                component=self.component,
                operation=self.operation_name,
                response_time_ms=int(response_time_ms),
                cpu_time_ms=int(response_time_ms),  # Approximation
                memory_peak_mb=int(abs(memory_mb)),
                requests_per_second=1000.0 / response_time_ms if response_time_ms > 0 else 0.0,
                success_rate=100.0,  # Assume success unless exception
                error_rate=0.0
            )
        else:
            return {
                "component": self.component,
                "operation": self.operation_name,
                "response_time_ms": int(response_time_ms),
                "memory_peak_mb": int(abs(memory_mb))
            }
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        current_time = self.end_time or time.perf_counter()
        return (current_time - self.start_time) * 1000

class CodeAnalyzer:
    """Analyzes code quality and complexity metrics."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_file(self, file_path: Path) -> CodeQualityMetrics:
        """Analyze a Python file for quality metrics.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Code quality metrics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(file_path))
            
            # Calculate metrics
            metrics = CodeQualityMetrics(
                module_name=file_path.stem,
                lines_of_code=len(source.splitlines()),
                cyclomatic_complexity=self._calculate_cyclomatic_complexity(tree),
                cognitive_complexity=self._calculate_cognitive_complexity(tree),
                function_count=self._count_functions(tree),
                class_count=self._count_classes(tree),
                import_count=self._count_imports(tree),
                docstring_coverage=self._calculate_docstring_coverage(tree),
                memory_footprint_kb=self._estimate_memory_footprint(source)
            )
            
            # Calculate composite scores
            metrics.code_quality_score = self._calculate_quality_score(metrics)
            metrics.type_coverage = self._calculate_type_coverage(tree)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return CodeQualityMetrics(module_name=file_path.stem)
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try,
                               ast.With, ast.Assert, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity (simplified)."""
        complexity = 0
        nesting_level = 0
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.nesting = 0
            
            def visit_If(self, node):
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1
            
            def visit_While(self, node):
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1
            
            def visit_For(self, node):
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        return visitor.complexity
    
    def _count_functions(self, tree: ast.AST) -> int:
        """Count function definitions."""
        return len([node for node in ast.walk(tree) 
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))])
    
    def _count_classes(self, tree: ast.AST) -> int:
        """Count class definitions."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
    
    def _count_imports(self, tree: ast.AST) -> int:
        """Count import statements."""
        return len([node for node in ast.walk(tree) 
                   if isinstance(node, (ast.Import, ast.ImportFrom))])
    
    def _calculate_docstring_coverage(self, tree: ast.AST) -> float:
        """Calculate docstring coverage percentage."""
        functions = [node for node in ast.walk(tree) 
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        
        if not functions:
            return 100.0
        
        documented = 0
        for func in functions:
            if (func.body and isinstance(func.body[0], ast.Expr) and 
                isinstance(func.body[0].value, ast.Constant) and 
                isinstance(func.body[0].value.value, str)):
                documented += 1
        
        return (documented / len(functions)) * 100
    
    def _calculate_type_coverage(self, tree: ast.AST) -> float:
        """Calculate type annotation coverage."""
        functions = [node for node in ast.walk(tree) 
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        
        if not functions:
            return 100.0
        
        typed = 0
        for func in functions:
            has_return_type = func.returns is not None
            has_arg_types = all(arg.annotation is not None for arg in func.args.args)
            
            if has_return_type and has_arg_types:
                typed += 1
        
        return (typed / len(functions)) * 100
    
    def _estimate_memory_footprint(self, source: str) -> int:
        """Estimate memory footprint in KB."""
        # Simple estimation based on source size and complexity
        base_size = len(source.encode('utf-8'))
        return base_size // 1024 + 1
    
    def _calculate_quality_score(self, metrics: CodeQualityMetrics) -> float:
        """Calculate overall code quality score (0-100)."""
        # Weighted scoring formula
        complexity_score = max(0, 100 - metrics.cyclomatic_complexity * 2)
        cognitive_score = max(0, 100 - metrics.cognitive_complexity * 3)
        docstring_score = metrics.docstring_coverage
        type_score = metrics.type_coverage
        
        # Weighted average
        weights = [0.25, 0.25, 0.25, 0.25]
        scores = [complexity_score, cognitive_score, docstring_score, type_score]
        
        return sum(w * s for w, s in zip(weights, scores))

class DependencyAnalyzer:
    """Analyzes project dependencies for optimization opportunities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_dependencies(self, root_path: Path) -> Dict[str, Any]:
        """Analyze project dependencies.
        
        Args:
            root_path: Root project path
            
        Returns:
            Dependency analysis results
        """
        results = {
            "import_graph": {},
            "circular_dependencies": [],
            "unused_imports": [],
            "heavy_imports": [],
            "optimization_suggestions": []
        }
        
        # Build import graph
        import_graph = self._build_import_graph(root_path)
        results["import_graph"] = import_graph
        
        # Detect circular dependencies
        circular_deps = self._detect_circular_dependencies(import_graph)
        results["circular_dependencies"] = circular_deps
        
        # Find unused imports
        unused = self._find_unused_imports(root_path)
        results["unused_imports"] = unused
        
        # Identify heavy imports
        heavy = self._identify_heavy_imports(import_graph)
        results["heavy_imports"] = heavy
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(results)
        results["optimization_suggestions"] = suggestions
        
        return results
    
    def _build_import_graph(self, root_path: Path) -> Dict[str, Set[str]]:
        """Build import dependency graph."""
        import_graph = defaultdict(set)
        
        for py_file in root_path.rglob("*.py"):
            if py_file.name.startswith("test_"):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=str(py_file))
                
                module_name = self._path_to_module(py_file, root_path)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            import_graph[module_name].add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            import_graph[module_name].add(node.module)
                            
            except Exception as e:
                self.logger.warning(f"Could not analyze {py_file}: {e}")
        
        return dict(import_graph)
    
    def _detect_circular_dependencies(self, import_graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Detect circular dependencies in import graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in import_graph.get(node, set()):
                if neighbor in import_graph:  # Only internal modules
                    dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for module in import_graph:
            if module not in visited:
                dfs(module, [])
        
        return cycles
    
    def _find_unused_imports(self, root_path: Path) -> List[Tuple[str, str]]:
        """Find unused import statements."""
        unused = []
        
        for py_file in root_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source, filename=str(py_file))
                
                # Get all imports
                imports = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            name = alias.asname or alias.name
                            imports[name] = alias.name
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            name = alias.asname or alias.name
                            imports[name] = f"{node.module}.{alias.name}" if node.module else alias.name
                
                # Check usage
                for imported_name, full_name in imports.items():
                    # Simple check: is the name used in the source?
                    if imported_name not in source.replace(f"import {imported_name}", ""):
                        unused.append((str(py_file), full_name))
                        
            except Exception as e:
                self.logger.warning(f"Could not analyze imports in {py_file}: {e}")
        
        return unused
    
    def _identify_heavy_imports(self, import_graph: Dict[str, Set[str]]) -> List[Tuple[str, int]]:
        """Identify modules with heavy import footprints."""
        heavy_imports = []
        
        # Known heavy libraries
        heavy_libs = {
            'numpy', 'pandas', 'tensorflow', 'torch', 'sklearn', 
            'matplotlib', 'seaborn', 'scipy', 'cv2'
        }
        
        for module, imports in import_graph.items():
            heavy_count = sum(1 for imp in imports if any(heavy in imp for heavy in heavy_libs))
            if heavy_count > 0:
                heavy_imports.append((module, heavy_count))
        
        return sorted(heavy_imports, key=lambda x: x[1], reverse=True)
    
    def _generate_optimization_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []
        
        # Circular dependency suggestions
        if analysis["circular_dependencies"]:
            suggestions.append(
                f"Break {len(analysis['circular_dependencies'])} circular dependencies "
                "using dependency injection or interface segregation"
            )
        
        # Unused import suggestions
        if analysis["unused_imports"]:
            suggestions.append(
                f"Remove {len(analysis['unused_imports'])} unused imports "
                "to reduce memory footprint"
            )
        
        # Heavy import suggestions
        if analysis["heavy_imports"]:
            suggestions.append(
                "Consider lazy loading for heavy imports like numpy, pandas, or ML libraries"
            )
        
        return suggestions
    
    def _path_to_module(self, file_path: Path, root_path: Path) -> str:
        """Convert file path to module name."""
        relative_path = file_path.relative_to(root_path)
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        return ".".join(module_parts)

class PerformanceRegressor:
    """Detects performance regressions by comparing against baselines."""
    
    def __init__(self, 
                 baseline_storage: Optional[Dict[str, PerformanceBaseline]] = None,
                 logger: Optional[logging.Logger] = None):
        self.baselines = baseline_storage or {}
        self.logger = logger or logging.getLogger(__name__)
        self.recent_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def establish_baseline(self, 
                          operation_name: str, 
                          component: str,
                          metrics: List[PerformanceMetrics]) -> PerformanceBaseline:
        """Establish performance baseline from metrics.
        
        Args:
            operation_name: Name of the operation
            component: Component name
            metrics: List of performance metrics
            
        Returns:
            Established baseline
        """
        if not metrics:
            raise ValueError("Cannot establish baseline with no metrics")
        
        # Extract timing data
        response_times = [m.response_time_ms for m in metrics]
        memory_usage = [m.memory_peak_mb for m in metrics]
        
        if NUMPY_AVAILABLE:
            avg_response = float(np.mean(response_times))
            p95_response = float(np.percentile(response_times, 95))
            p99_response = float(np.percentile(response_times, 99))
            avg_memory = float(np.mean(memory_usage))
            peak_memory = float(np.max(memory_usage))
        else:
            avg_response = sum(response_times) / len(response_times)
            p95_response = sorted(response_times)[int(len(response_times) * 0.95)]
            p99_response = sorted(response_times)[int(len(response_times) * 0.99)]
            avg_memory = sum(memory_usage) / len(memory_usage)
            peak_memory = max(memory_usage)
        
        baseline = PerformanceBaseline(
            operation_name=operation_name,
            component=component,
            avg_response_time_ms=avg_response,
            p95_response_time_ms=p95_response,
            p99_response_time_ms=p99_response,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            avg_cpu_percent=50.0,  # Default
            requests_per_second=1000.0 / avg_response if avg_response > 0 else 0.0,
            success_rate=100.0,  # Default
            code_quality_score=85.0,  # Default
            test_coverage=80.0,  # Default
            sample_count=len(metrics),
            established_at=datetime.now()
        )
        
        self.baselines[f"{component}:{operation_name}"] = baseline
        self.logger.info(f"Established baseline for {component}:{operation_name}")
        
        return baseline
    
    def check_regression(self, 
                        current_metrics: PerformanceMetrics,
                        threshold_percent: float = 20.0) -> Optional[RegressionAlert]:
        """Check for performance regression.
        
        Args:
            current_metrics: Current performance metrics
            threshold_percent: Regression threshold percentage
            
        Returns:
            RegressionAlert if regression detected, None otherwise
        """
        key = f"{current_metrics.component}:{current_metrics.operation}"
        baseline = self.baselines.get(key)
        
        if not baseline:
            self.logger.debug(f"No baseline found for {key}")
            return None
        
        # Check response time regression
        current_response = current_metrics.response_time_ms
        baseline_response = baseline.avg_response_time_ms
        
        if baseline_response > 0:
            response_regression = ((current_response - baseline_response) / baseline_response) * 100
            
            if response_regression > threshold_percent:
                severity = self._determine_severity(response_regression)
                
                return RegressionAlert(
                    operation_name=current_metrics.operation,
                    metric_name="response_time_ms",
                    baseline_value=baseline_response,
                    current_value=current_response,
                    regression_percent=response_regression,
                    severity=severity,
                    details={
                        "component": current_metrics.component,
                        "threshold_percent": threshold_percent
                    }
                )
        
        return None
    
    def _determine_severity(self, regression_percent: float) -> str:
        """Determine regression severity."""
        if regression_percent > 100:
            return "critical"
        elif regression_percent > 50:
            return "major"
        else:
            return "minor"

class PerformanceAnalyzer:
    """Main performance analysis orchestrator."""
    
    def __init__(self, 
                 root_path: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None):
        self.root_path = root_path or Path("src")
        self.logger = logger or logging.getLogger(__name__)
        
        self.code_analyzer = CodeAnalyzer(logger)
        self.dependency_analyzer = DependencyAnalyzer(logger)
        self.regressor = PerformanceRegressor(logger=logger)
        
        self.active_timers: Dict[str, PerformanceTimer] = {}
        self.metrics_history: deque = deque(maxlen=1000)
    
    @contextmanager
    def measure_performance(self, operation: str, component: str = "unknown"):
        """Context manager for measuring operation performance."""
        timer = PerformanceTimer(operation, component)
        try:
            yield timer
        finally:
            if timer.start_time:
                metrics = timer.stop()
                self.record_metrics(metrics)
    
    def measure_function(self, operation_name: Optional[str] = None, 
                        component: Optional[str] = None):
        """Decorator for measuring function performance."""
        def decorator(func: F) -> F:
            nonlocal operation_name, component
            operation_name = operation_name or func.__name__
            component = component or func.__module__
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with self.measure_performance(operation_name, component):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.measure_performance(operation_name, component):
                        return func(*args, **kwargs)
                return sync_wrapper
        
        return decorator
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        self.metrics_history.append(metrics)
        
        # Check for regressions
        regression = self.regressor.check_regression(metrics)
        if regression:
            self.logger.warning(f"Performance regression detected: {regression}")
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Perform comprehensive codebase analysis."""
        self.logger.info("Starting comprehensive codebase analysis")
        
        results = {
            "code_quality": {},
            "dependencies": {},
            "performance_summary": {},
            "recommendations": []
        }
        
        # Analyze code quality
        quality_metrics = []
        for py_file in self.root_path.rglob("*.py"):
            if not py_file.name.startswith("test_"):
                metrics = self.code_analyzer.analyze_file(py_file)
                quality_metrics.append(metrics)
        
        results["code_quality"] = {
            "files_analyzed": len(quality_metrics),
            "average_quality_score": sum(m.code_quality_score for m in quality_metrics) / len(quality_metrics) if quality_metrics else 0,
            "total_lines_of_code": sum(m.lines_of_code for m in quality_metrics),
            "average_complexity": sum(m.cyclomatic_complexity for m in quality_metrics) / len(quality_metrics) if quality_metrics else 0,
            "docstring_coverage": sum(m.docstring_coverage for m in quality_metrics) / len(quality_metrics) if quality_metrics else 0
        }
        
        # Analyze dependencies
        results["dependencies"] = self.dependency_analyzer.analyze_dependencies(self.root_path)
        
        # Performance summary
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            results["performance_summary"] = {
                "recent_measurements": len(recent_metrics),
                "avg_response_time_ms": sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics),
                "avg_memory_mb": sum(m.memory_peak_mb for m in recent_metrics) / len(recent_metrics)
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results["recommendations"] = recommendations
        
        self.logger.info("Codebase analysis complete")
        return results
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Code quality recommendations
        quality = analysis.get("code_quality", {})
        if quality.get("average_quality_score", 0) < 70:
            recommendations.append("Improve code quality: focus on reducing complexity and improving documentation")
        
        if quality.get("docstring_coverage", 0) < 80:
            recommendations.append("Increase docstring coverage to improve maintainability")
        
        # Dependency recommendations
        deps = analysis.get("dependencies", {})
        if deps.get("circular_dependencies"):
            recommendations.append("Break circular dependencies to improve modularity")
        
        if deps.get("unused_imports"):
            recommendations.append("Remove unused imports to reduce memory footprint")
        
        # Performance recommendations
        perf = analysis.get("performance_summary", {})
        if perf.get("avg_response_time_ms", 0) > 1000:
            recommendations.append("Optimize performance: average response time exceeds 1 second")
        
        return recommendations
    
    def generate_report(self, format: str = "text") -> str:
        """Generate performance analysis report."""
        analysis = self.analyze_codebase()
        
        if format == "text":
            return self._generate_text_report(analysis)
        elif format == "json":
            import json
            return json.dumps(analysis, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_text_report(self, analysis: Dict[str, Any]) -> str:
        """Generate text report."""
        lines = [
            "Performance Analysis Report",
            "=" * 30,
            ""
        ]
        
        # Code Quality Section
        quality = analysis.get("code_quality", {})
        lines.extend([
            "Code Quality Metrics:",
            f"  Files analyzed: {quality.get('files_analyzed', 0)}",
            f"  Average quality score: {quality.get('average_quality_score', 0):.1f}/100",
            f"  Total lines of code: {quality.get('total_lines_of_code', 0)}",
            f"  Average complexity: {quality.get('average_complexity', 0):.1f}",
            f"  Docstring coverage: {quality.get('docstring_coverage', 0):.1f}%",
            ""
        ])
        
        # Dependencies Section
        deps = analysis.get("dependencies", {})
        lines.extend([
            "Dependency Analysis:",
            f"  Circular dependencies: {len(deps.get('circular_dependencies', []))}",
            f"  Unused imports: {len(deps.get('unused_imports', []))}",
            f"  Heavy imports: {len(deps.get('heavy_imports', []))}",
            ""
        ])
        
        # Performance Section
        perf = analysis.get("performance_summary", {})
        lines.extend([
            "Performance Summary:",
            f"  Recent measurements: {perf.get('recent_measurements', 0)}",
            f"  Average response time: {perf.get('avg_response_time_ms', 0):.1f}ms",
            f"  Average memory usage: {perf.get('avg_memory_mb', 0):.1f}MB",
            ""
        ])
        
        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            lines.extend([
                "Recommendations:",
                *[f"  • {rec}" for rec in recommendations],
                ""
            ])
        
        return "\n".join(lines)

# Factory function for easy usage
def create_performance_analyzer(root_path: Optional[Path] = None) -> PerformanceAnalyzer:
    """Create a performance analyzer instance.
    
    Args:
        root_path: Root path for analysis
        
    Returns:
        Configured PerformanceAnalyzer instance
    """
    return PerformanceAnalyzer(root_path)

# Export main classes and functions
__all__ = [
    "PerformanceAnalyzer",
    "PerformanceTimer", 
    "CodeAnalyzer",
    "DependencyAnalyzer",
    "PerformanceRegressor",
    "CodeQualityMetrics",
    "PerformanceBaseline",
    "RegressionAlert",
    "create_performance_analyzer"
]