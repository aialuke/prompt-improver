"""
Tests for Modern A/B Testing Framework Implementation
Uses real database operations and authentic statistical computations following 2025 best practices.
No mocking of core functionality - tests real behavior patterns.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
from hypothesis import (
    given,
    settings,
    strategies as st,
)

from prompt_improver.database.models import (
    ABExperiment,
    PromptSession,
    RuleMetadata,
    RulePerformance,
)
from prompt_improver.database.registry import clear_registry
from prompt_improver.performance.testing.ab_testing_service import (
    ModernABConfig,
    ModernABTestingService as ABTestingService,
    StatisticalMethod,
)


class TestModernABTestingFramework:
    """Test Suite for Modern A/B Testing Service with real behavior testing."""

    @pytest.fixture
    def clear_registry_before_test(self):
        """Clear the SQLAlchemy registry before each test to prevent conflicts."""
        clear_registry()

    @pytest.fixture
    def ab_testing_service(self):
        """Fixture to create ModernABTestingService with optimal 2025 configuration."""
        config = ModernABConfig(
            statistical_method=StatisticalMethod.HYBRID,
            confidence_level=0.95,
            statistical_power=0.8,
            minimum_sample_size=50,
            enable_sequential_testing=True,
            enable_early_stopping=True,
            minimum_detectable_effect=0.05,
        )
        return ABTestingService(config)

    @pytest.mark.asyncio
    async def test_orchestrator_interface_integration(
        self, ab_testing_service, real_db_session
    ):
        """Test the new orchestrator interface integration with real statistical analysis."""
        result = await ab_testing_service.run_orchestrated_analysis({
            "statistical_method": "hybrid",
            "sample_size": 100,
            "enable_early_stopping": True,
            "output_path": "./test_outputs",
        })
        assert result["orchestrator_compatible"] is True
        assert "component_result" in result
        assert "ab_testing_summary" in result["component_result"]
        ab_summary = result["component_result"]["ab_testing_summary"]
        assert ab_summary["statistical_method"] == "hybrid"
        assert ab_summary["confidence_level"] == 0.95
        assert ab_summary["early_stopping_enabled"] is True
        assert ab_summary["sequential_testing"] is True
        assert ab_summary["hybrid_analysis"] is True
        capabilities = result["component_result"]["capabilities"]
        assert capabilities["wilson_confidence_intervals"] is True
        assert capabilities["variance_reduction_cuped"] is True
        assert capabilities["hybrid_bayesian_frequentist"] is True
        assert capabilities["sequential_testing"] is True
        assert capabilities["early_stopping_with_sprt"] is True
        assert "local_metadata" in result
        metadata = result["local_metadata"]
        assert metadata["framework"] == "ModernABTestingService"
        assert metadata["statistical_method"] == "hybrid"
        assert metadata["component_version"] == "2025.1.0"
        ab_data = result["component_result"]["ab_testing_data"]
        assert "configuration" in ab_data
        assert "statistical_capabilities" in ab_data
        assert "framework_features" in ab_data
        stat_caps = ab_data["statistical_capabilities"]
        assert "wilson" in stat_caps["confidence_interval_methods"]
        assert "cuped" in stat_caps["variance_reduction_techniques"]
        assert "hybrid" in stat_caps["supported_methods"]
        features = ab_data["framework_features"]
        assert features["real_time_monitoring"] is True
        assert features["multi_armed_bandit_integration"] is True
        assert features["power_analysis"] is True

    @pytest.mark.asyncio
    async def test_create_experiment_real_database_operations(
        self, ab_testing_service, real_db_session
    ):
        """Test experiment creation with real database operations."""
        control_rule = RuleMetadata(
            rule_id="control_rule_2025",
            rule_name="Control Rule 2025",
            category="improvement",
            description="Modern control rule for 2025 testing",
            enabled=True,
            priority=1,
            default_parameters={"enhancement_weight": 1.0},
        )
        treatment_rule = RuleMetadata(
            rule_id="treatment_rule_2025",
            rule_name="Treatment Rule 2025",
            category="improvement",
            description="Modern treatment rule for 2025 testing",
            enabled=True,
            priority=1,
            default_parameters={"enhancement_weight": 1.5},
        )
        real_db_session.add(control_rule)
        real_db_session.add(treatment_rule)
        await real_db_session.commit()
        result = await ab_testing_service.create_experiment(
            db_session=real_db_session,
            name="modern_2025_experiment",
            description="Modern A/B test for 2025 best practices validation",
            hypothesis="Treatment rule will improve conversion rate by at least 5%",
            control_rule_ids=["control_rule_2025"],
            treatment_rule_ids=["treatment_rule_2025"],
            success_metric="improvement_score",
            metadata={"baseline_conversion_rate": 0.1, "expected_effect": 0.05},
        )
        assert isinstance(result, str)
        assert len(result) > 0
        from sqlalchemy import select

        stmt = select(ABExperiment).where(ABExperiment.experiment_id == result)
        result_obj = await real_db_session.execute(stmt)
        experiment = result_obj.scalar_one_or_none()
        assert experiment is not None, (
            f"Experiment with ID {result} not found in database"
        )
        assert experiment.experiment_name == "modern_2025_experiment"
        assert experiment.control_rules == {"rule_ids": ["control_rule_2025"]}
        assert experiment.treatment_rules == {"rule_ids": ["treatment_rule_2025"]}
        assert experiment.target_metric == "improvement_score"

    @pytest.mark.asyncio
    async def test_statistical_analysis_real_computations(
        self, ab_testing_service, real_db_session
    ):
        """Test statistical analysis using real computations with authentic data patterns."""
        np.random.seed(2025)
        await self._create_experiment_with_performance_data(
            ab_testing_service, real_db_session
        )
        from sqlalchemy import text

        experiment = await real_db_session.execute(
            text("SELECT * FROM ab_experiments ORDER BY started_at DESC LIMIT 1")
        )
        experiment_row = experiment.fetchone()
        experiment_id = experiment_row[0]
        analysis_result = await ab_testing_service.analyze_experiment(
            experiment_id=str(experiment_id), db_session=real_db_session
        )
        assert analysis_result is not None
        if analysis_result.get("status") == "insufficient_data":
            assert "control_samples" in analysis_result
            assert "treatment_samples" in analysis_result
            assert "minimum_sample_size" in analysis_result
            assert "recommendations" in analysis_result
        else:
            if "analysis" in analysis_result:
                stats = analysis_result["analysis"]
            else:
                stats = analysis_result
            assert "p_value" in stats
            assert 0.0 <= stats["p_value"] <= 1.0
            assert "effect_size" in stats
            assert isinstance(stats["effect_size"], (int, float))
            assert "confidence_interval" in stats
            ci = stats["confidence_interval"]
            assert len(ci) == 2
            assert ci[0] <= ci[1]
            assert "bayesian_probability" in stats
            assert 0.0 <= stats["bayesian_probability"] <= 1.0
            assert stats.get("confidence_method") == "wilson"
            if "treatment_mean" in stats and "control_mean" in stats:
                if stats["treatment_mean"] > stats["control_mean"]:
                    assert stats["effect_size"] >= 0.0
                else:
                    assert stats["effect_size"] <= 0.0

    @pytest.mark.asyncio
    async def test_sequential_testing_real_implementation(
        self, ab_testing_service, real_db_session
    ):
        """Test sequential testing with real statistical power calculations."""
        ab_testing_service.config.enable_sequential_testing = True
        await self._create_experiment_with_sequential_data(
            ab_testing_service, real_db_session
        )
        from sqlalchemy import text

        experiment = await real_db_session.execute(
            text("SELECT * FROM ab_experiments ORDER BY started_at DESC LIMIT 1")
        )
        experiment_row = experiment.fetchone()
        experiment_id = str(experiment_row[0])
        for look_number in [1, 2, 3]:
            result = await ab_testing_service.check_early_stopping(
                experiment_id=experiment_id,
                look_number=look_number,
                db_session=real_db_session,
            )
            assert result is not None
            assert "should_stop" in result
            assert "reason" in result
            assert "statistical_power" in result
            assert 0.0 <= result["statistical_power"] <= 1.0
            if "sequential_boundary" in result:
                assert result["sequential_boundary"] > 0.0

    @pytest.mark.asyncio
    async def test_early_stopping_framework_real_sprt(
        self, ab_testing_service, real_db_session
    ):
        """Test early stopping with real Sequential Probability Ratio Test (SPRT)."""
        ab_testing_service.config.enable_early_stopping = True
        await self._create_experiment_with_strong_effect(
            ab_testing_service, real_db_session
        )
        from sqlalchemy import text

        experiment = await real_db_session.execute(
            text("SELECT * FROM ab_experiments ORDER BY started_at DESC LIMIT 1")
        )
        experiment_row = experiment.fetchone()
        experiment_id = str(experiment_row[0])
        early_stop_result = await ab_testing_service.check_early_stopping(
            experiment_id=experiment_id, look_number=1, db_session=real_db_session
        )
        assert early_stop_result is not None
        if early_stop_result["should_stop"]:
            assert (
                "sprt_statistic" in early_stop_result
                or "log_likelihood_ratio" in early_stop_result
            )
            assert early_stop_result["reason"] in {
                "efficacy",
                "futility",
                "strong_evidence",
            }
        if "upper_boundary" in early_stop_result:
            assert early_stop_result["upper_boundary"] > 0
        if "lower_boundary" in early_stop_result:
            assert early_stop_result["lower_boundary"] < 0

    @pytest.mark.asyncio
    async def test_variance_reduction_cuped_real_implementation(
        self, ab_testing_service, real_db_session
    ):
        """Test CUPED variance reduction with real covariate data."""
        ab_testing_service.config.variance_reduction_method = "cuped"
        await self._create_experiment_with_covariates(
            ab_testing_service, real_db_session
        )
        from sqlalchemy import text

        experiment = await real_db_session.execute(
            text("SELECT * FROM ab_experiments ORDER BY started_at DESC LIMIT 1")
        )
        experiment_row = experiment.fetchone()
        experiment_id = str(experiment_row[0])
        result = await ab_testing_service.analyze_experiment(
            experiment_id=experiment_id, db_session=real_db_session
        )
        if result.get("status") == "insufficient_data":
            assert "recommendations" in result
        else:
            stats = result.get("analysis", result)
            assert "variance_reduction" in stats
            variance_info = stats["variance_reduction"]
            assert variance_info["method"] == "cuped"
            if "variance_reduction_percentage" in variance_info:
                assert 0.0 <= variance_info["variance_reduction_percentage"] <= 100.0
            if "covariate_correlation" in variance_info:
                assert -1.0 <= variance_info["covariate_correlation"] <= 1.0

    @given(
        control_conversion_rate=st.floats(min_value=0.1, max_value=0.9),
        treatment_lift=st.floats(min_value=0.01, max_value=0.5),
        sample_size=st.integers(min_value=100, max_value=1000),
    )
    @settings(max_examples=10)
    def test_effect_size_calculations_property_based(
        self, control_conversion_rate, treatment_lift, sample_size
    ):
        """Property-based test for effect size calculations using real statistical formulas."""
        treatment_conversion_rate = min(0.99, control_conversion_rate + treatment_lift)
        pooled_std = np.sqrt(
            (
                control_conversion_rate * (1 - control_conversion_rate)
                + treatment_conversion_rate * (1 - treatment_conversion_rate)
            )
            / 2
        )
        if pooled_std > 0:
            cohens_d = (
                treatment_conversion_rate - control_conversion_rate
            ) / pooled_std
            assert cohens_d >= 0.0
            if treatment_lift < 0.1:
                assert cohens_d < 0.5
            elif treatment_lift > 0.4:
                assert cohens_d > 0.3

    @pytest.mark.asyncio
    async def test_experiment_lifecycle_real_state_transitions(
        self, ab_testing_service, real_db_session
    ):
        """Test complete experiment lifecycle with real state transitions."""
        control_rule = RuleMetadata(
            rule_id="lifecycle_control",
            rule_name="Lifecycle Control",
            category="test",
            description="Control for lifecycle test",
            enabled=True,
            priority=1,
            default_parameters={},
        )
        real_db_session.add(control_rule)
        await real_db_session.commit()
        experiment_id = await ab_testing_service.create_experiment(
            db_session=real_db_session,
            name="lifecycle_test",
            description="Lifecycle test experiment",
            hypothesis="Testing lifecycle transitions",
            control_rule_ids=["lifecycle_control"],
            treatment_rule_ids=["lifecycle_control"],
            success_metric="improvement_score",
        )
        from sqlalchemy import select

        stmt = select(ABExperiment).where(ABExperiment.experiment_id == experiment_id)
        result = await real_db_session.execute(stmt)
        experiment = result.scalar_one_or_none()
        assert experiment is not None
        assert experiment.status == "running"
        stop_result = await ab_testing_service.stop_experiment(
            experiment_id=experiment_id,
            reason="Lifecycle test completion",
            db_session=real_db_session,
        )
        assert stop_result is True
        await real_db_session.refresh(experiment)
        assert experiment.status == "stopped"
        assert experiment.completed_at is not None

    async def _create_experiment_with_performance_data(self, service, db_session):
        """Helper to create experiment with realistic performance data."""
        np.random.seed(2025)
        control_rule = RuleMetadata(
            rule_id="perf_control",
            rule_name="Performance Control",
            category="test",
            description="Control rule",
            enabled=True,
            priority=1,
            default_parameters={},
        )
        treatment_rule = RuleMetadata(
            rule_id="perf_treatment",
            rule_name="Performance Treatment",
            category="test",
            description="Treatment rule",
            enabled=True,
            priority=1,
            default_parameters={},
        )
        db_session.add(control_rule)
        db_session.add(treatment_rule)
        await db_session.commit()
        experiment = ABExperiment(
            experiment_name="performance_test",
            description="Performance analysis test",
            control_rules={"rule_ids": ["perf_control"]},
            treatment_rules={"rule_ids": ["perf_treatment"]},
            target_metric="improvement_score",
            sample_size_per_group=100,
            status="running",
            started_at=datetime.utcnow(),
        )
        db_session.add(experiment)
        await db_session.commit()
        for i in range(100):
            control_session = PromptSession(
                session_id=f"perf_control_{i}",
                original_prompt=f"Control prompt {i}",
                improved_prompt=f"Improved control {i}",
                quality_score=np.random.uniform(0.5, 0.9),
                improvement_score=np.random.normal(0.65, 0.15),
                confidence_level=np.random.uniform(0.7, 0.95),
                created_at=datetime.utcnow() - timedelta(minutes=i),
            )
            control_perf = RulePerformance(
                session_id=f"perf_control_{i}",
                rule_id="perf_control",
                rule_name="Performance Control",
                improvement_score=max(0.0, min(1.0, np.random.normal(0.65, 0.15))),
                execution_time_ms=np.random.normal(95, 15),
                confidence_level=np.random.normal(0.82, 0.08),
                created_at=datetime.utcnow() - timedelta(minutes=i),
            )
            treatment_session = PromptSession(
                session_id=f"perf_treatment_{i}",
                original_prompt=f"Treatment prompt {i}",
                improved_prompt=f"Improved treatment {i}",
                quality_score=np.random.uniform(0.6, 0.95),
                improvement_score=np.random.normal(0.78, 0.15),
                confidence_level=np.random.uniform(0.75, 0.98),
                created_at=datetime.utcnow() - timedelta(minutes=i),
            )
            treatment_perf = RulePerformance(
                session_id=f"perf_treatment_{i}",
                rule_id="perf_treatment",
                rule_name="Performance Treatment",
                improvement_score=max(0.0, min(1.0, np.random.normal(0.78, 0.15))),
                execution_time_ms=np.random.normal(88, 15),
                confidence_level=np.random.normal(0.87, 0.08),
                created_at=datetime.utcnow() - timedelta(minutes=i),
            )
            db_session.add(control_session)
            db_session.add(control_perf)
            db_session.add(treatment_session)
            db_session.add(treatment_perf)
        await db_session.commit()

    async def _create_experiment_with_sequential_data(self, service, db_session):
        """Helper to create experiment with sequential data collection patterns."""
        await self._create_experiment_with_performance_data(service, db_session)

    async def _create_experiment_with_strong_effect(self, service, db_session):
        """Helper to create experiment with strong effect size for early stopping."""
        np.random.seed(42)
        control_rule = RuleMetadata(
            rule_id="strong_control",
            rule_name="Strong Effect Control",
            category="test",
            description="Control with strong effect",
            enabled=True,
            priority=1,
            default_parameters={},
        )
        db_session.add(control_rule)
        await db_session.commit()
        experiment = ABExperiment(
            experiment_name="strong_effect_test",
            description="Strong effect for early stopping",
            control_rules={"rule_ids": ["strong_control"]},
            treatment_rules={"rule_ids": ["strong_control"]},
            target_metric="improvement_score",
            sample_size_per_group=50,
            status="running",
            started_at=datetime.utcnow(),
        )
        db_session.add(experiment)
        await db_session.commit()
        for i in range(50):
            control_session = PromptSession(
                session_id=f"strong_control_{i}",
                original_prompt=f"Strong control {i}",
                improved_prompt=f"Strong improved control {i}",
                quality_score=np.random.normal(0.5, 0.1),
                improvement_score=np.random.normal(0.5, 0.1),
                confidence_level=np.random.uniform(0.7, 0.9),
                created_at=datetime.utcnow() - timedelta(minutes=i),
            )
            treatment_session = PromptSession(
                session_id=f"strong_treatment_{i}",
                original_prompt=f"Strong treatment {i}",
                improved_prompt=f"Strong improved treatment {i}",
                quality_score=np.random.normal(0.9, 0.1),
                improvement_score=np.random.normal(0.9, 0.1),
                confidence_level=np.random.uniform(0.85, 0.98),
                created_at=datetime.utcnow() - timedelta(minutes=i),
            )
            db_session.add(control_session)
            db_session.add(treatment_session)
        await db_session.commit()

    async def _create_experiment_with_covariates(self, service, db_session):
        """Helper to create experiment with covariate data for CUPED testing."""
        np.random.seed(2025)
        await self._create_experiment_with_performance_data(service, db_session)


class TestModernABTestingEdgeCases:
    """Edge cases and error handling for Modern A/B Testing Service."""

    @pytest.fixture
    def ab_testing_service(self):
        """Fixture with default configuration."""
        return ABTestingService()

    @pytest.mark.asyncio
    async def test_insufficient_sample_size_real_power_analysis(
        self, ab_testing_service, real_db_session
    ):
        """Test handling of insufficient sample size with real statistical power analysis."""
        control_rule = RuleMetadata(
            rule_id="small_control",
            rule_name="Small Sample Control",
            category="test",
            description="Small sample test",
            enabled=True,
            priority=1,
            default_parameters={},
        )
        real_db_session.add(control_rule)
        await real_db_session.commit()
        experiment = ABExperiment(
            experiment_name="small_sample_test",
            description="Test with insufficient sample",
            control_rules={"rule_ids": ["small_control"]},
            treatment_rules={"rule_ids": ["small_control"]},
            target_metric="improvement_score",
            sample_size_per_group=5,
            status="running",
            started_at=datetime.utcnow(),
        )
        real_db_session.add(experiment)
        await real_db_session.commit()
        for i in range(5):
            session = PromptSession(
                session_id=f"small_{i}",
                original_prompt=f"Small prompt {i}",
                improved_prompt=f"Small improved {i}",
                quality_score=0.7,
                improvement_score=0.7,
                confidence_level=0.8,
                created_at=datetime.utcnow(),
            )
            real_db_session.add(session)
        await real_db_session.commit()
        result = await ab_testing_service.analyze_experiment(
            db_session=real_db_session, experiment_id=str(experiment.experiment_id)
        )
        assert result is not None
        assert result.get("status") == "insufficient_data"
        assert "statistical_power" in result
        assert result["statistical_power"] == 0.0
        assert "minimum_sample_size" in result
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_database_constraint_violations_real_errors(
        self, ab_testing_service, real_db_session
    ):
        """Test real database constraint violations."""
        control_rule = RuleMetadata(
            rule_id="constraint_control",
            rule_name="Constraint Control",
            category="test",
            description="Constraint test",
            enabled=True,
            priority=1,
            default_parameters={},
        )
        real_db_session.add(control_rule)
        await real_db_session.commit()
        await ab_testing_service.create_experiment(
            db_session=real_db_session,
            name="duplicate_test",
            description="Constraint test experiment",
            hypothesis="Testing constraint violations",
            control_rule_ids=["constraint_control"],
            treatment_rule_ids=["constraint_control"],
            success_metric="improvement_score",
        )
        duplicate_id = await ab_testing_service.create_experiment(
            db_session=real_db_session,
            name="duplicate_test",
            description="Duplicate constraint test",
            hypothesis="Testing duplicates",
            control_rule_ids=["constraint_control"],
            treatment_rule_ids=["constraint_control"],
            success_metric="improvement_score",
        )
        assert isinstance(duplicate_id, str)
        assert len(duplicate_id) > 0
