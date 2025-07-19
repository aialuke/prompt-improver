#!/usr/bin/env python3
"""ML Contract Validation Script for Pre-commit Hook"""

import ast
import os
import sys

# Modern Python type hints - using built-in types


class MLContractValidator(ast.NodeVisitor):
    """Validates ML-related contracts in Python files."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

        # Required contracts for ML components
        self.required_ml_methods = {"fit", "predict", "score", "transform"}

        # Performance requirements
        self.performance_keywords = {"response_time", "timeout", "cache", "async"}

        # ML monitoring keywords
        self.monitoring_keywords = {"drift", "metrics", "performance", "logging"}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Validate ML-related classes."""
        class_name = node.name.lower()

        # Check ML model classes
        if any(
            keyword in class_name
            for keyword in ["model", "estimator", "classifier", "regressor"]
        ):
            self._validate_ml_model_class(node)

        # Check evaluation classes
        if any(keyword in class_name for keyword in ["evaluator", "judge", "analyzer"]):
            self._validate_evaluation_class(node)

        # Check monitoring classes
        if any(keyword in class_name for keyword in ["monitor", "tracker", "logger"]):
            self._validate_monitoring_class(node)

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Validate ML-related functions."""
        func_name = node.name.lower()

        # Check async functions for timeout handling
        if node.decorator_list:
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "asyncio":
                    self._validate_async_function(node)

        # Check performance-critical functions
        if any(keyword in func_name for keyword in ["evaluate", "improve", "analyze"]):
            self._validate_performance_function(node)

        self.generic_visit(node)

    def _validate_ml_model_class(self, node: ast.ClassDef) -> None:
        """Validate ML model class contracts."""
        method_names = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}

        # Check for required ML interface methods
        missing_methods = self.required_ml_methods - method_names
        if missing_methods and len(missing_methods) > 2:  # Allow some flexibility
            self.warnings.append(
                f"ML model class '{node.name}' missing methods: {missing_methods}"
            )

        # Check for performance monitoring
        has_monitoring = any(
            keyword in str(ast.dump(node)).lower()
            for keyword in self.monitoring_keywords
        )
        if not has_monitoring:
            self.warnings.append(
                f"ML model class '{node.name}' lacks performance monitoring"
            )

    def _validate_evaluation_class(self, node: ast.ClassDef) -> None:
        """Validate evaluation class contracts."""
        class_source = ast.dump(node).lower()

        # Check for response time handling
        if "response_time" not in class_source and "timeout" not in class_source:
            self.warnings.append(
                f"Evaluation class '{node.name}' should handle response time requirements"
            )

        # Check for fallback mechanisms
        if "fallback" not in class_source and "backup" not in class_source:
            self.warnings.append(
                f"Evaluation class '{node.name}' should implement fallback mechanisms"
            )

    def _validate_monitoring_class(self, node: ast.ClassDef) -> None:
        """Validate monitoring class contracts."""
        class_source = ast.dump(node).lower()

        # Check for prometheus metrics
        if "prometheus" not in class_source and "metrics" not in class_source:
            self.warnings.append(
                f"Monitoring class '{node.name}' should integrate with Prometheus metrics"
            )

    def _validate_async_function(self, node: ast.FunctionDef) -> None:
        """Validate async function contracts."""
        func_source = ast.dump(node).lower()

        # Check for timeout handling
        if "timeout" not in func_source and "asyncio.wait_for" not in func_source:
            self.warnings.append(
                f"Async function '{node.name}' should implement timeout handling"
            )

    def _validate_performance_function(self, node: ast.FunctionDef) -> None:
        """Validate performance-critical function contracts."""
        func_source = ast.dump(node).lower()

        # Check for performance monitoring
        has_timing = any(
            keyword in func_source
            for keyword in ["time", "duration", "performance", "metrics"]
        )
        if not has_timing:
            self.warnings.append(
                f"Performance function '{node.name}' should include timing/metrics"
            )


def validate_file(filepath: str) -> bool:
    """Validate a single Python file for ML contracts."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        # Parse the AST
        tree = ast.parse(content, filename=filepath)

        # Run validation
        validator = MLContractValidator()
        validator.visit(tree)

        # Report results
        success = len(validator.errors) == 0

        if validator.errors:
            print(f"❌ {filepath}: ML contract validation failed")
            for error in validator.errors:
                print(f"   ERROR: {error}")

        if validator.warnings:
            print(f"⚠️  {filepath}: ML contract warnings")
            for warning in validator.warnings:
                print(f"   WARNING: {warning}")

        if not validator.errors and not validator.warnings:
            print(f"✅ {filepath}: ML contracts valid")

        return success

    except SyntaxError as e:
        print(f"❌ {filepath}: Syntax error - {e}")
        return False
    except OSError as e:
        print(f"❌ {filepath}: File I/O error - {e}")
        return False
    except (UnicodeDecodeError, ValueError) as e:
        print(f"❌ {filepath}: File encoding/content error - {e}")
        return False
    except RecursionError as e:
        print(f"❌ {filepath}: AST parsing recursion error (file too complex) - {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Validation cancelled by user")
        raise
    except Exception as e:
        print(f"❌ {filepath}: Unexpected validation error - {e}")
        import logging

        logging.exception(f"Unexpected error validating {filepath}")
        return False


def main():
    """Main validation script."""
    if len(sys.argv) < 2:
        print("Usage: validate_ml_contracts.py <file1.py> [file2.py] ...")
        sys.exit(1)

    files = sys.argv[1:]
    all_success = True

    for filepath in files:
        if not os.path.exists(filepath):
            print(f"❌ {filepath}: File not found")
            all_success = False
            continue

        if not filepath.endswith(".py"):
            continue  # Skip non-Python files

        success = validate_file(filepath)
        all_success = all_success and success

    if not all_success:
        print("\n❌ ML contract validation failed for some files")
        sys.exit(1)
    else:
        print("\n✅ All ML contracts validated successfully")


if __name__ == "__main__":
    main()
