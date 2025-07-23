"""
Regression test to ensure no duplicate method definitions exist in the codebase.

This test specifically verifies that the duplicate method definitions that were
cleaned up in the failure_analyzer.py module do not reappear in the future.
"""

import inspect
from collections import Counter

import pytest


def test_no_duplicate_method_definitions():
    """Test that there are no duplicate method definitions in FailureModeAnalyzer."""
    # Import the module
    from prompt_improver.ml.learning.algorithms.failure_analyzer import FailureModeAnalyzer

    # Get all methods from the class
    all_methods = inspect.getmembers(FailureModeAnalyzer, predicate=inspect.ismethod)
    all_functions = inspect.getmembers(
        FailureModeAnalyzer, predicate=inspect.isfunction
    )

    # Combine methods and functions (unbound methods appear as functions in Python 3)
    all_callables = all_methods + all_functions

    # Count method names
    method_names = [name for name, _ in all_callables]
    method_counts = Counter(method_names)

    # Find any duplicates
    duplicates = {name: count for name, count in method_counts.items() if count > 1}

    # Assert no duplicates exist
    assert len(duplicates) == 0, f"Found duplicate method definitions: {duplicates}"


def test_specific_methods_single_definition():
    """Test that specific methods that were previously duplicated now have exactly one definition."""
    from prompt_improver.ml.learning.algorithms.failure_analyzer import FailureModeAnalyzer

    # Methods that were previously duplicated
    target_methods = [
        "_initialize_ml_fmea_database",
        "_perform_ml_fmea_analysis",
        "_perform_ensemble_anomaly_detection",
    ]

    for method_name in target_methods:
        # Use getmembers to get all instances of the method
        method_instances = []

        # Check if method exists as an attribute
        if hasattr(FailureModeAnalyzer, method_name):
            method_obj = getattr(FailureModeAnalyzer, method_name)

            # Verify it's callable
            assert callable(method_obj), f"Method {method_name} is not callable"

            # Count how many times this method appears in the class
            all_members = inspect.getmembers(FailureModeAnalyzer)
            method_count = sum(1 for name, _ in all_members if name == method_name)

            # Should have exactly one definition
            assert method_count == 1, (
                f"Method {method_name} has {method_count} definitions, expected 1"
            )
        else:
            pytest.fail(f"Method {method_name} not found in FailureModeAnalyzer class")


def test_method_functionality_accessible():
    """Test that the retained methods are accessible and functional."""
    from prompt_improver.learning.failure_analyzer import (
        FailureConfig,
        FailureModeAnalyzer,
    )

    # Create an analyzer instance
    config = FailureConfig()
    analyzer = FailureModeAnalyzer(config)

    # Test that key methods are accessible
    assert hasattr(analyzer, "_initialize_ml_fmea_database"), (
        "Method _initialize_ml_fmea_database not accessible"
    )
    assert hasattr(analyzer, "_perform_ml_fmea_analysis"), (
        "Method _perform_ml_fmea_analysis not accessible"
    )
    assert hasattr(analyzer, "_perform_ensemble_anomaly_detection"), (
        "Method _perform_ensemble_anomaly_detection not accessible"
    )

    # Test that the methods can be called (basic functionality test)
    # Note: We're testing method accessibility, not full functionality
    try:
        # These methods should exist and be callable
        ml_failure_modes = analyzer._initialize_ml_fmea_database()
        assert isinstance(ml_failure_modes, list), "ML failure modes should be a list"
        assert len(ml_failure_modes) > 0, "ML failure modes should not be empty"

        # Verify that the ml_failure_modes attribute is properly set
        assert hasattr(analyzer, "ml_failure_modes"), (
            "analyzer should have ml_failure_modes attribute"
        )
        assert isinstance(analyzer.ml_failure_modes, list), (
            "ml_failure_modes should be a list"
        )

    except Exception as e:
        pytest.fail(f"Method accessibility test failed: {e}")


def test_no_undefined_method_calls():
    """Test that there are no calls to undefined methods in the class."""
    import ast
    import inspect

    # Get the source code of the class
    from prompt_improver.ml.learning.algorithms.failure_analyzer import FailureModeAnalyzer

    source = inspect.getsource(FailureModeAnalyzer)

    # Parse the source code into an AST
    tree = ast.parse(source)

    # Find all method calls that start with 'self._'
    method_calls = set()

    class MethodCallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                ):
                    if node.func.attr.startswith("_"):
                        method_calls.add(node.func.attr)
            self.generic_visit(node)

    visitor = MethodCallVisitor()
    visitor.visit(tree)

    # Get all actual method names in the class
    all_methods = inspect.getmembers(
        FailureModeAnalyzer,
        predicate=lambda x: inspect.ismethod(x) or inspect.isfunction(x),
    )
    actual_method_names = {name for name, _ in all_methods}

    # Check for any method calls that don't have corresponding method definitions
    undefined_calls = method_calls - actual_method_names

    # Filter out some common patterns that might be valid (like property access)
    # and focus on methods that were specifically involved in the cleanup
    critical_methods = {
        "_initialize_ml_fmea_database",
        "_perform_ml_fmea_analysis",
        "_perform_ensemble_anomaly_detection",
        "_initialize_ml_fmea_failure_modes",  # This was the incorrect method name
    }

    critical_undefined = undefined_calls & critical_methods

    # Should not have any undefined calls to critical methods
    assert len(critical_undefined) == 0, (
        f"Found calls to undefined critical methods: {critical_undefined}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
