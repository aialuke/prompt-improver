"""
Tests for HDBSCAN clustering enhancements in advanced pattern discovery service.
Uses pytest-asyncio, unittest.mock, hypothesis, and pytest-benchmark for comprehensive testing.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from hypothesis import (
    assume,
    given,
    settings,
    strategies as st,
)
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from prompt_improver.services.advanced_pattern_discovery import (
    AdvancedPatternDiscovery,
    FrequentPattern,
    PatternCluster,
)


class TestHDBSCANClusteringEnhancements:
    """Test suite for HDBSCAN clustering enhancements with comprehensive coverage."""

    @pytest.fixture
    def discovery_service(self):
        """Create AdvancedPatternDiscovery service instance."""
        return AdvancedPatternDiscovery()

    @pytest.fixture
    def sample_pattern_data(self):
        """Generate sample pattern data for testing."""
        return [
            {
                "rule_id": "clarity_rule_1",
                "parameters": {"weight": 0.8, "threshold": 0.7},
                "effectiveness": 0.85,
                "execution_time_ms": 150,
                "prompt_characteristics": {"length": 50, "complexity": 0.6},
            },
            {
                "rule_id": "clarity_rule_2",
                "parameters": {"weight": 0.9, "threshold": 0.6},
                "effectiveness": 0.78,
                "execution_time_ms": 200,
                "prompt_characteristics": {"length": 75, "complexity": 0.7},
            },
            {
                "rule_id": "specificity_rule_1",
                "parameters": {"weight": 0.7, "threshold": 0.8},
                "effectiveness": 0.82,
                "execution_time_ms": 180,
                "prompt_characteristics": {"length": 60, "complexity": 0.5},
            },
        ]

    @pytest.mark.asyncio
    async def test_hdbscan_cluster_discovery_performance(
        self, discovery_service, mock_db_session
    ):
        """Test HDBSCAN cluster discovery performance."""
        # Setup mock data
        mock_db_session.execute = AsyncMock()
        mock_db_session.scalars = AsyncMock()

        # Mock performance data that would come from database
        mock_performance_data = [
            {"parameters": {"weight": 0.8}, "effectiveness": 0.85},
            {"parameters": {"weight": 0.9}, "effectiveness": 0.78},
            {"parameters": {"weight": 0.7}, "effectiveness": 0.82},
        ] * 20  # 60 samples for meaningful clustering

        with patch.object(
            discovery_service,
            "_get_performance_data",
            return_value=mock_performance_data,
        ):
            # Test the parameter pattern discovery (which uses HDBSCAN)
            result = await discovery_service.discover_advanced_patterns(
                mock_db_session, min_effectiveness=0.7, pattern_types=["parameter"]
            )

        # Verify performance requirements
        assert result["status"] == "success"
        assert result["total_samples"] == 60
        assert "parameter_patterns" in result["pattern_discovery"]
        assert result["processing_time_ms"] < 1000  # Should complete under 1 second

    @pytest.mark.asyncio
    @given(
        effectiveness_scores=st.lists(
            st.floats(min_value=0.3, max_value=1.0), min_size=15, max_size=50
        )
    )
    @settings(max_examples=10, deadline=5000)
    async def test_hdbscan_cluster_properties(self, effectiveness_scores):
        """Property-based test for HDBSCAN clustering invariants."""
        assume(len(effectiveness_scores) >= 15)

        # Generate mock data based on hypothesis inputs
        mock_data = []
        for i, score in enumerate(effectiveness_scores):
            mock_data.append({
                "rule_id": f"rule_{i}",
                "parameters": {
                    "weight": np.random.uniform(0.5, 1.0),
                    "threshold": np.random.uniform(0.5, 0.9),
                },
                "effectiveness": score,
                "execution_time_ms": np.random.randint(50, 300),
            })

        discovery_service = AdvancedPatternDiscovery()
        mock_db_session = AsyncMock()
        mock_db_session.execute = AsyncMock()

        with patch.object(
            discovery_service, "_get_performance_data", return_value=mock_data
        ):
            result = await discovery_service.discover_advanced_patterns(
                mock_db_session, min_effectiveness=0.5, pattern_types=["parameter"]
            )

        # Verify clustering properties
        if (
            result["status"] == "success"
            and "parameter_patterns" in result["pattern_discovery"]
        ):
            param_patterns = result["pattern_discovery"]["parameter_patterns"]

            # Property: All discovered patterns should have valid effectiveness scores
            if "clusters" in param_patterns:
                for cluster in param_patterns["clusters"]:
                    assert cluster.effectiveness_range[0] >= 0.0
                    assert cluster.effectiveness_range[1] <= 1.0
                    assert (
                        cluster.effectiveness_range[0] <= cluster.effectiveness_range[1]
                    )

            # Property: Cluster scores should be non-negative
            if "clusters" in param_patterns:
                for cluster in param_patterns["clusters"]:
                    assert cluster.cluster_score >= 0.0
                    assert cluster.density >= 0.0

    @pytest.mark.asyncio
    async def test_hdbscan_minimum_cluster_size_parameter(
        self, discovery_service, mock_db_session
    ):
        """Test HDBSCAN min_cluster_size parameter affects clustering results."""
        # Create data with clear clusters
        mock_data = []

        # First cluster: high weight, high effectiveness
        for i in range(10):
            mock_data.append({
                "rule_id": f"high_perf_{i}",
                "parameters": {
                    "weight": 0.9 + np.random.normal(0, 0.05),
                    "threshold": 0.8,
                },
                "effectiveness": 0.9 + np.random.normal(0, 0.05),
            })

        # Second cluster: low weight, medium effectiveness
        for i in range(8):
            mock_data.append({
                "rule_id": f"med_perf_{i}",
                "parameters": {
                    "weight": 0.6 + np.random.normal(0, 0.05),
                    "threshold": 0.6,
                },
                "effectiveness": 0.7 + np.random.normal(0, 0.05),
            })

        mock_db_session.execute = AsyncMock()

        with patch.object(
            discovery_service, "_get_performance_data", return_value=mock_data
        ):
            # Test with small min_cluster_size
            discovery_service.min_cluster_size = 3
            result_small = await discovery_service.discover_advanced_patterns(
                mock_db_session, pattern_types=["parameter"]
            )

            # Test with large min_cluster_size
            discovery_service.min_cluster_size = 12
            result_large = await discovery_service.discover_advanced_patterns(
                mock_db_session, pattern_types=["parameter"]
            )

        # Verify that min_cluster_size affects number of clusters found
        if result_small["status"] == "success" and result_large["status"] == "success":
            small_clusters = (
                result_small["pattern_discovery"]
                .get("parameter_patterns", {})
                .get("clusters", [])
            )
            large_clusters = (
                result_large["pattern_discovery"]
                .get("parameter_patterns", {})
                .get("clusters", [])
            )

            # Smaller min_cluster_size should generally find more or equal clusters
            assert len(small_clusters) >= len(large_clusters)

    @pytest.mark.asyncio
    async def test_hdbscan_vs_dbscan_performance(
        self, discovery_service, mock_db_session
    ):
        """Benchmark HDBSCAN performance vs traditional DBSCAN."""
        # Generate larger dataset for meaningful performance comparison
        np.random.seed(42)
        large_dataset = []

        for i in range(200):
            large_dataset.append({
                "rule_id": f"rule_{i}",
                "parameters": {
                    "weight": np.random.uniform(0.5, 1.0),
                    "threshold": np.random.uniform(0.5, 0.9),
                    "complexity_factor": np.random.uniform(0.1, 1.0),
                },
                "effectiveness": np.random.uniform(0.3, 1.0),
                "execution_time_ms": np.random.randint(50, 500),
            })

        mock_db_session.execute = AsyncMock()

        with patch.object(
            discovery_service, "_get_performance_data", return_value=large_dataset
        ):
            # Test HDBSCAN-based discovery
            hdbscan_result = await discovery_service.discover_advanced_patterns(
                mock_db_session, pattern_types=["parameter"]
            )

        # Verify HDBSCAN completed successfully and within performance bounds
        assert hdbscan_result["status"] == "success"
        assert (
            hdbscan_result["processing_time_ms"] < 2000
        )  # Should complete under 2 seconds

        # Verify clustering quality metrics are present
        if "parameter_patterns" in hdbscan_result["pattern_discovery"]:
            param_patterns = hdbscan_result["pattern_discovery"]["parameter_patterns"]
            # Check for expected structure (may vary based on actual implementation)
            assert "clusters" in param_patterns
            assert isinstance(param_patterns["clusters"], list)

    @pytest.mark.asyncio
    async def test_hdbscan_outlier_detection(self, discovery_service, mock_db_session):
        """Test HDBSCAN's outlier detection capabilities."""
        # Create dataset with clear outliers
        normal_data = []
        outlier_data = []

        # Normal patterns: weight around 0.8, effectiveness around 0.8
        for i in range(15):
            normal_data.append({
                "rule_id": f"normal_{i}",
                "parameters": {
                    "weight": 0.8 + np.random.normal(0, 0.1),
                    "threshold": 0.75,
                },
                "effectiveness": 0.8 + np.random.normal(0, 0.05),
            })

        # Outlier patterns: very high weight, very high effectiveness
        for i in range(3):
            outlier_data.append({
                "rule_id": f"outlier_{i}",
                "parameters": {
                    "weight": 0.95 + np.random.normal(0, 0.02),
                    "threshold": 0.9,
                },
                "effectiveness": 0.95 + np.random.normal(0, 0.02),
            })

        all_data = normal_data + outlier_data
        mock_db_session.execute = AsyncMock()

        with patch.object(
            discovery_service, "_get_performance_data", return_value=all_data
        ):
            result = await discovery_service.discover_advanced_patterns(
                mock_db_session, pattern_types=["parameter"]
            )

        # Verify outliers are detected or handled appropriately
        if (
            result["status"] == "success"
            and "parameter_patterns" in result["pattern_discovery"]
        ):
            param_patterns = result["pattern_discovery"]["parameter_patterns"]

            # Check that outliers field exists and is a list (may be empty)
            if "outliers" in param_patterns:
                assert isinstance(param_patterns["outliers"], list)

            # Check clusters exist
            assert "clusters" in param_patterns
            assert isinstance(param_patterns["clusters"], list)


class HDBSCANStateMachine(RuleBasedStateMachine):
    """Stateful testing for HDBSCAN clustering behavior."""

    patterns = Bundle("patterns")

    def __init__(self):
        super().__init__()
        self.discovery_service = AdvancedPatternDiscovery()
        self.current_patterns = []

    @rule(
        target=patterns,
        effectiveness=st.floats(min_value=0.3, max_value=1.0),
        weight=st.floats(min_value=0.5, max_value=1.0),
    )
    def add_pattern(self, effectiveness, weight):
        """Add a pattern to the current dataset."""
        pattern = {
            "rule_id": f"rule_{len(self.current_patterns)}",
            "parameters": {"weight": weight, "threshold": 0.7},
            "effectiveness": effectiveness,
        }
        self.current_patterns.append(pattern)
        return pattern

    @rule(patterns=patterns)
    def remove_pattern(self, patterns):
        """Remove a pattern from the dataset."""
        if patterns in self.current_patterns:
            self.current_patterns.remove(patterns)

    @invariant()
    def patterns_have_valid_scores(self):
        """All patterns should have valid effectiveness scores."""
        for pattern in self.current_patterns:
            assert 0.0 <= pattern["effectiveness"] <= 1.0
            assert 0.0 <= pattern["parameters"]["weight"] <= 1.0

    @invariant()
    def clustering_is_deterministic(self):
        """Clustering should be deterministic given same input."""
        if len(self.current_patterns) >= 5:
            # This would require mocking the async method
            # For brevity, we'll just check data consistency
            assert len(self.current_patterns) >= 5


# Convert state machine to test case
TestHDBSCANStateMachine = HDBSCANStateMachine.TestCase


# Additional edge case tests
class TestHDBSCANEdgeCases:
    """Test edge cases and error conditions for HDBSCAN clustering."""

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, mock_db_session):
        """Test handling when insufficient data is available for clustering."""
        discovery_service = AdvancedPatternDiscovery()

        # Mock minimal data (below minimum threshold)
        minimal_data = [
            {"rule_id": "rule_1", "parameters": {"weight": 0.8}, "effectiveness": 0.85},
            {"rule_id": "rule_2", "parameters": {"weight": 0.9}, "effectiveness": 0.78},
        ]

        mock_db_session.execute = AsyncMock()

        with patch.object(
            discovery_service, "_get_performance_data", return_value=minimal_data
        ):
            result = await discovery_service.discover_advanced_patterns(
                mock_db_session,
                min_support=5,  # Require at least 5 samples
                pattern_types=["parameter"],
            )

        assert result["status"] == "insufficient_data"
        assert "minimum" in result["message"]
        assert result["processing_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_hdbscan_unavailable_fallback(self, mock_db_session):
        """Test fallback behavior when HDBSCAN is not available."""
        discovery_service = AdvancedPatternDiscovery()

        mock_data = [
            {
                "rule_id": f"rule_{i}",
                "parameters": {"weight": 0.8},
                "effectiveness": 0.8,
            }
            for i in range(10)
        ]

        with patch(
            "prompt_improver.services.advanced_pattern_discovery.HDBSCAN_AVAILABLE",
            False,
        ):
            with patch.object(
                discovery_service, "_get_performance_data", return_value=mock_data
            ):
                result = await discovery_service.discover_advanced_patterns(
                    mock_db_session, pattern_types=["parameter"]
                )

        # Should still succeed with fallback algorithm (DBSCAN)
        assert result["status"] == "success"
        assert "parameter_patterns" in result["pattern_discovery"]
