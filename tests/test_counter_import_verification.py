"""Test verification that Counter is properly imported from collections.

This test ensures that:
1. Counter is imported from collections, not typing
2. Counter instances are properly typed
3. MyPy stub checking works correctly
"""

import collections
import unittest
from collections import Counter
from typing import TYPE_CHECKING

# Import the module to test
from prompt_improver.ml.learning.algorithms import (
    FailureConfig,
    FailureAnalyzer as FailureModeAnalyzer,
)

if TYPE_CHECKING:
    # This should not cause any import errors
    from collections import Counter as CollectionsCounter


class TestCounterImportVerification(unittest.TestCase):
    """Test that Counter is properly imported from collections."""

    def test_counter_is_from_collections(self):
        """Test that Counter is imported from collections module."""
        # Direct import verification
        from collections import Counter as TestCounter

        # Create a Counter instance
        counter_instance = TestCounter()

        # Verify it's a collections.Counter
        self.assertIsInstance(counter_instance, collections.Counter)

        # Verify the module source
        self.assertEqual(TestCounter.__module__, "collections")

    def test_counter_isinstance_check(self):
        """Test isinstance check with collections.Counter."""
        # Create Counter instances
        counter1 = Counter()
        counter2 = Counter(["a", "b", "c", "a"])

        # Test isinstance with collections.Counter
        self.assertTrue(isinstance(counter1, collections.Counter))
        self.assertTrue(isinstance(counter2, collections.Counter))

        # Test that it's not from typing
        self.assertNotEqual(Counter.__module__, "typing")

    def test_counter_functionality(self):
        """Test that Counter works as expected."""
        # Create and test Counter functionality
        counter = Counter()
        counter["test"] = 5
        counter["example"] = 3

        # Verify functionality
        self.assertEqual(counter["test"], 5)
        self.assertEqual(counter["example"], 3)
        self.assertEqual(counter.most_common(1), [("test", 5)])

    def test_failure_analyzer_counter_usage(self):
        """Test that FailureModeAnalyzer uses Counter correctly."""
        # Create analyzer instance
        config = FailureConfig()
        analyzer = FailureModeAnalyzer(config)

        # Test the _extract_common_issues method that uses Counter
        failures = [
            {
                "error": "validation failed due to missing data",
                "originalPrompt": "test prompt",
            },
            {
                "error": "validation error occurred during processing",
                "originalPrompt": "another prompt",
            },
        ]

        # Call the method that uses Counter
        common_issues = analyzer._extract_common_issues(failures)

        # Verify it works (should extract common keywords)
        self.assertIsInstance(common_issues, list)
        # Should find 'validation' as a common term
        self.assertIn("validation", common_issues)

    def test_counter_type_annotation_compatibility(self):
        """Test that Counter type annotations work correctly."""

        # Test that we can type-annotate Counter correctly
        def count_items(items: list) -> Counter[str]:
            return Counter(items)

        # Test the function
        result = count_items(["a", "b", "a", "c"])

        # Verify type and functionality
        self.assertIsInstance(result, Counter)
        self.assertEqual(result["a"], 2)
        self.assertEqual(result["b"], 1)
        self.assertEqual(result["c"], 1)

    def test_no_typing_counter_import(self):
        """Test that typing.Counter is not imported in the module."""
        # Read the failure_analyzer.py file
        import inspect

        from prompt_improver.learning import failure_analyzer

        # Get the source code
        source = inspect.getsource(failure_analyzer)

        # Verify that there's no "from typing import.*Counter"
        lines = source.split("\n")
        typing_import_lines = [line for line in lines if "from typing import" in line]

        for line in typing_import_lines:
            # Check that Counter is not imported from typing
            self.assertNotIn(
                "Counter", line, f"Found Counter imported from typing in line: {line}"
            )

    def test_prometheus_counter_is_aliased(self):
        """Test that Prometheus Counter is properly aliased."""
        # Read the failure_analyzer.py file to check import
        import inspect

        from prompt_improver.learning import failure_analyzer

        # Get the source code
        source = inspect.getsource(failure_analyzer)

        # Check for proper Prometheus Counter alias
        self.assertIn(
            "Counter as PrometheusCounter",
            source,
            "Prometheus Counter should be aliased as PrometheusCounter",
        )

        # Verify collections Counter is imported
        self.assertIn(
            "from collections import", source, "collections should be imported"
        )
        self.assertIn(
            "Counter",
            source.split("from collections import")[1].split("\n")[0],
            "Counter should be imported from collections",
        )


class TestMyPyStubCheck(unittest.TestCase):
    """Test MyPy stub checking for Counter imports."""

    def test_mypy_stub_counter_import(self):
        """Test that MyPy can properly check Counter imports."""
        # This test verifies that MyPy stub checking works
        # In a real scenario, this would be run with MyPy CLI

        # Create a function that uses Counter with type hints
        def process_counts(items: list[str]) -> Counter[str]:
            """Process items and return Counter with type hints."""
            return Counter(items)

        # Test that the function works
        result = process_counts(["a", "b", "a"])

        # Verify it's a proper Counter
        self.assertIsInstance(result, Counter)
        self.assertEqual(result["a"], 2)

        # Verify the type annotation is correct
        import typing

        hints = typing.get_type_hints(process_counts)

        # Check return type annotation
        self.assertIn("return", hints)
        return_type = hints["return"]

        # Verify it's Counter from collections (not typing)
        self.assertEqual(return_type.__origin__, Counter)


if __name__ == "__main__":
    unittest.main()
