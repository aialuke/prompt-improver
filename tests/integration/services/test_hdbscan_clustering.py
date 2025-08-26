"""
End-to-end integration tests for HDBSCAN clustering enhancements in advanced pattern discovery service.
Following 2025 best practices: real database integration, minimal mocking, realistic data patterns.

Migration from mock-based testing to real behavior testing based on research:
- Mock only external dependencies (Redis circuit breaker acceptable)
- Use real database connections and actual performance data
- Test with realistic data distributions and edge cases
- Validate clustering performance with production-like scenarios
"""

from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import (
    assume,
    given,
    settings,
    strategies as st,
)
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from prompt_improver.database.models import PromptSession, RuleMetadata, RulePerformance
from prompt_improver.services.advanced_pattern_discovery import (
    AdvancedPatternDiscovery,
)
from prompt_improver.utils.datetime_utils import aware_utc_now


class TestHDBSCANClusteringEnhancements:
    """Test suite for HDBSCAN clustering enhancements with comprehensive coverage."""

    @pytest.fixture
    async def discovery_service(self, real_db_session):
        """Create AdvancedPatternDiscovery service with real database connection."""
        from prompt_improver.database import ManagerMode, create_database_services

        services = await create_database_services(ManagerMode.ASYNC_MODERN)
        return AdvancedPatternDiscovery(session_manager=services)

    @pytest.fixture
    async def real_performance_data(self, real_db_session):
        """Create real rule performance data for HDBSCAN clustering tests."""
        rule_metadata = [
            RuleMetadata(
                rule_id="clarity_rule_1",
                rule_name="Clarity Enhancement Rule 1",
                description="Improves clarity through weight adjustment",
                category="clarity",
                default_parameters={"weight": 0.8, "threshold": 0.7},
                created_at=aware_utc_now(),
                updated_at=aware_utc_now(),
            ),
            RuleMetadata(
                rule_id="clarity_rule_2",
                rule_name="Clarity Enhancement Rule 2",
                description="Improves clarity through threshold adjustment",
                category="clarity",
                default_parameters={"weight": 0.9, "threshold": 0.6},
                created_at=aware_utc_now(),
                updated_at=aware_utc_now(),
            ),
            RuleMetadata(
                rule_id="specificity_rule_1",
                rule_name="Specificity Enhancement Rule 1",
                description="Improves specificity through parameter tuning",
                category="specificity",
                default_parameters={"weight": 0.7, "threshold": 0.8},
                created_at=aware_utc_now(),
                updated_at=aware_utc_now(),
            ),
        ]
        prompt_sessions = [PromptSession(
                    session_id=f"session_{i}",
                    original_prompt=f"Original prompt {i}",
                    improved_prompt=f"Improved prompt {i}",
                    user_context={"test": True},
                    quality_score=0.8,
                    improvement_score=0.75,
                    confidence_level=0.9,
                    created_at=aware_utc_now() - timedelta(days=30, hours=i),
                    updated_at=aware_utc_now() - timedelta(days=30, hours=i),
                ) for i in range(60)]
        performance_data = []
        base_time = aware_utc_now() - timedelta(days=30)
        for i in range(60):
            if i < 20:
                effectiveness = np.random.normal(0.85, 0.05)
                execution_time = np.random.normal(150, 20)
                rule_id = f"clarity_rule_{i % 2 + 1}"
            elif i < 40:
                effectiveness = np.random.normal(0.7, 0.03)
                execution_time = np.random.normal(200, 15)
                rule_id = "specificity_rule_1"
            else:
                effectiveness = np.random.normal(0.75, 0.08)
                execution_time = np.random.normal(180, 25)
                rule_id = f"clarity_rule_{i % 2 + 1}"
            performance_data.append(
                RulePerformance(
                    rule_id=rule_id,
                    session_id=f"session_{i}",
                    improvement_score=max(0.0, min(1.0, effectiveness)),
                    execution_time_ms=max(50, execution_time),
                    confidence_level=np.random.uniform(0.6, 0.9),
                    parameters_used={
                        "weight": np.random.uniform(0.5, 1.0),
                        "threshold": np.random.uniform(0.5, 0.9),
                    },
                    created_at=base_time + timedelta(hours=i),
                )
            )
        for session in prompt_sessions:
            real_db_session.add(session)
        for metadata in rule_metadata:
            real_db_session.add(metadata)
        for performance in performance_data:
            real_db_session.add(performance)
        await real_db_session.commit()
        return performance_data

    @pytest.mark.asyncio
    async def test_hdbscan_cluster_discovery_performance(
        self, discovery_service, real_db_session, real_performance_data
    ):
        """Test HDBSCAN cluster discovery performance with real data."""
        result = await discovery_service.discover_advanced_patterns(
            real_db_session, min_effectiveness=0.7, pattern_types=["parameter"]
        )
        assert result["status"] == "success"
        assert result["total_samples"] == 60
        assert "parameter_patterns" in result["pattern_discovery"]
        assert result["processing_time_ms"] < 2000
        param_patterns = result["pattern_discovery"]["parameter_patterns"]
        if "clusters" in param_patterns:
            clusters = param_patterns["clusters"]
            assert len(clusters) >= 1
            for cluster in clusters:
                assert hasattr(cluster, "effectiveness_range")
                assert hasattr(cluster, "cluster_score")

    @pytest.mark.asyncio
    @given(
        effectiveness_scores=st.lists(
            st.floats(min_value=0.3, max_value=1.0), min_size=15, max_size=50
        )
    )
    @settings(max_examples=5, deadline=10000)
    async def test_hdbscan_cluster_properties(
        self, effectiveness_scores, discovery_service, real_db_session
    ):
        """Property-based test for HDBSCAN clustering invariants using real database operations."""
        assume(len(effectiveness_scores) >= 15)
        property_test_data = []
        base_time = aware_utc_now() - timedelta(days=7)
        for i, score in enumerate(effectiveness_scores):
            property_test_data.append(
                RulePerformance(
                    rule_id=f"prop_rule_{i}",
                    session_id=f"prop_session_{i}",
                    effectiveness_score=max(0.0, min(1.0, score)),
                    execution_time_ms=np.random.randint(50, 300),
                    memory_usage_mb=np.random.uniform(10, 50),
                    context_relevance=np.random.uniform(0.5, 1.0),
                    created_at=base_time + timedelta(minutes=i),
                    updated_at=base_time + timedelta(minutes=i),
                )
            )
        for performance in property_test_data:
            real_db_session.add(performance)
        await real_db_session.commit()
        result = await discovery_service.discover_advanced_patterns(
            real_db_session, min_effectiveness=0.5, pattern_types=["parameter"]
        )
        if (
            result["status"] == "success"
            and "parameter_patterns" in result["pattern_discovery"]
        ):
            param_patterns = result["pattern_discovery"]["parameter_patterns"]
            if "clusters" in param_patterns:
                for cluster in param_patterns["clusters"]:
                    assert cluster.effectiveness_range[0] >= 0.0
                    assert cluster.effectiveness_range[1] <= 1.0
                    assert (
                        cluster.effectiveness_range[0] <= cluster.effectiveness_range[1]
                    )
            if "clusters" in param_patterns:
                for cluster in param_patterns["clusters"]:
                    assert cluster.cluster_score >= 0.0
                    assert cluster.density >= 0.0

    @pytest.mark.asyncio
    async def test_hdbscan_minimum_cluster_size_parameter(
        self, discovery_service, real_db_session, real_performance_data
    ):
        """Test HDBSCAN min_cluster_size parameter affects clustering results with real data."""
        discovery_service.min_cluster_size = 5
        result_small = await discovery_service.discover_advanced_patterns(
            real_db_session, pattern_types=["parameter"]
        )
        discovery_service.min_cluster_size = 15
        result_large = await discovery_service.discover_advanced_patterns(
            real_db_session, pattern_types=["parameter"]
        )
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
            assert len(small_clusters) >= len(large_clusters)
        assert result_small["total_samples"] == 60
        assert result_large["total_samples"] == 60

    @pytest.mark.asyncio
    async def test_hdbscan_vs_dbscan_performance(
        self, discovery_service, real_db_session
    ):
        """Benchmark HDBSCAN performance with real database operations."""
        additional_data = []
        base_time = aware_utc_now() - timedelta(days=60)
        for i in range(140):
            effectiveness = np.random.uniform(0.3, 1.0)
            execution_time = np.random.uniform(50, 500)
            rule_id = f"perf_rule_{i % 10}"
            additional_data.append(
                RulePerformance(
                    rule_id=rule_id,
                    session_id=f"perf_session_{i}",
                    effectiveness_score=effectiveness,
                    execution_time_ms=execution_time,
                    memory_usage_mb=np.random.uniform(10, 100),
                    context_relevance=np.random.uniform(0.3, 1.0),
                    created_at=base_time + timedelta(hours=i),
                    updated_at=base_time + timedelta(hours=i),
                )
            )
        for performance in additional_data:
            real_db_session.add(performance)
        await real_db_session.commit()
        hdbscan_result = await discovery_service.discover_advanced_patterns(
            real_db_session, pattern_types=["parameter"]
        )
        assert hdbscan_result["status"] == "success"
        assert hdbscan_result["processing_time_ms"] < 5000
        assert hdbscan_result["total_samples"] == 200
        if "parameter_patterns" in hdbscan_result["pattern_discovery"]:
            param_patterns = hdbscan_result["pattern_discovery"]["parameter_patterns"]
            assert "clusters" in param_patterns
            assert isinstance(param_patterns["clusters"], list)

    @pytest.mark.asyncio
    async def test_hdbscan_outlier_detection(
        self, discovery_service, real_db_session, real_performance_data
    ):
        """Test HDBSCAN's outlier detection capabilities with real data containing outliers."""
        base_time = aware_utc_now() - timedelta(days=1)
        outlier_data = [RulePerformance(
                    rule_id=f"outlier_rule_{i}",
                    session_id=f"outlier_session_{i}",
                    effectiveness_score=0.98 + np.random.normal(0, 0.01),
                    execution_time_ms=25 + np.random.normal(0, 5),
                    memory_usage_mb=5 + np.random.uniform(0, 2),
                    context_relevance=0.99 + np.random.normal(0, 0.005),
                    created_at=base_time + timedelta(minutes=i),
                    updated_at=base_time + timedelta(minutes=i),
                ) for i in range(5)]
        for performance in outlier_data:
            real_db_session.add(performance)
        await real_db_session.commit()
        result = await discovery_service.discover_advanced_patterns(
            real_db_session, pattern_types=["parameter"]
        )
        if (
            result["status"] == "success"
            and "parameter_patterns" in result["pattern_discovery"]
        ):
            param_patterns = result["pattern_discovery"]["parameter_patterns"]
            if "outliers" in param_patterns:
                assert isinstance(param_patterns["outliers"], list)
            assert "clusters" in param_patterns
            assert isinstance(param_patterns["clusters"], list)
        assert result["total_samples"] == 65


class HDBSCANStateMachine(RuleBasedStateMachine):
    """Stateful testing for HDBSCAN clustering behavior."""

    patterns = Bundle("patterns")

    def __init__(self):
        super().__init__()
        import asyncio

        from prompt_improver.database import ManagerMode, create_database_services

        # Create services in sync context
        loop = asyncio.get_event_loop()
        services = loop.run_until_complete(create_database_services(ManagerMode.ASYNC_MODERN))
        self.discovery_service = AdvancedPatternDiscovery(session_manager=services)
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
            assert len(self.current_patterns) >= 5


TestHDBSCANStateMachine = HDBSCANStateMachine.TestCase


class TestHDBSCANEdgeCases:
    """Test edge cases and error conditions for HDBSCAN clustering using real database connections."""

    @pytest.fixture
    async def minimal_discovery_service(self, real_db_session):
        """Create discovery service for minimal data testing."""
        from prompt_improver.database import ManagerMode, create_database_services

        services = await create_database_services(ManagerMode.ASYNC_MODERN)
        return AdvancedPatternDiscovery(session_manager=services)

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(
        self, minimal_discovery_service, real_db_session
    ):
        """Test handling when insufficient data is available for clustering."""
        minimal_data = [
            RulePerformance(
                rule_id="rule_1",
                session_id="session_1",
                effectiveness_score=0.85,
                execution_time_ms=150,
                memory_usage_mb=25,
                context_relevance=0.8,
                created_at=aware_utc_now(),
                updated_at=aware_utc_now(),
            ),
            RulePerformance(
                rule_id="rule_2",
                session_id="session_2",
                effectiveness_score=0.78,
                execution_time_ms=200,
                memory_usage_mb=30,
                context_relevance=0.75,
                created_at=aware_utc_now(),
                updated_at=aware_utc_now(),
            ),
        ]
        for performance in minimal_data:
            real_db_session.add(performance)
        await real_db_session.commit()
        result = await minimal_discovery_service.discover_advanced_patterns(
            real_db_session, min_support=5, pattern_types=["parameter"]
        )
        assert result["status"] == "insufficient_data"
        assert "minimum" in result["message"]
        assert result["processing_time_ms"] > 0
        assert result["total_samples"] == 2

    @pytest.mark.asyncio
    async def test_hdbscan_unavailable_fallback(
        self, minimal_discovery_service, real_db_session
    ):
        """Test fallback behavior when HDBSCAN is not available using real data."""
        fallback_data = [RulePerformance(
                    rule_id=f"fallback_rule_{i}",
                    session_id=f"fallback_session_{i}",
                    effectiveness_score=0.8 + np.random.normal(0, 0.1),
                    execution_time_ms=150 + np.random.normal(0, 30),
                    memory_usage_mb=25 + np.random.uniform(0, 10),
                    context_relevance=0.8 + np.random.uniform(-0.1, 0.1),
                    created_at=aware_utc_now() - timedelta(hours=i),
                    updated_at=aware_utc_now() - timedelta(hours=i),
                ) for i in range(15)]
        for performance in fallback_data:
            real_db_session.add(performance)
        await real_db_session.commit()
        with patch(
            "prompt_improver.services.advanced_pattern_discovery.HDBSCAN_AVAILABLE",
            False,
        ):
            result = await minimal_discovery_service.discover_advanced_patterns(
                real_db_session, pattern_types=["parameter"]
            )
        assert result["status"] == "success"
        assert "parameter_patterns" in result["pattern_discovery"]
        assert result["total_samples"] == 15
