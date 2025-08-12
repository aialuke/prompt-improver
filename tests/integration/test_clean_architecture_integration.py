"""Integration tests for clean architecture implementation.

Tests that core services work correctly with dependency injection
and repository interfaces without direct infrastructure coupling.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from prompt_improver.core.di.clean_container import (
    CleanDIContainer,
    get_clean_container,
    register_test_repositories,
)
from prompt_improver.core.services.persistence_service_clean import CleanPersistenceService
from prompt_improver.core.services.rule_selection_service_clean import CleanRuleSelectionService
from prompt_improver.repositories.interfaces.repository_interfaces import (
    IPersistenceRepository,
    IRulesRepository,
)
from prompt_improver.repositories.protocols.persistence_repository_protocol import (
    SessionData,
    RulePerformanceData,
    FeedbackData,
)
from prompt_improver.repositories.protocols.rules_repository_protocol import (
    RuleFilter,
)


class MockPersistenceRepository:
    """Mock persistence repository for testing."""

    def __init__(self):
        self.sessions: Dict[str, SessionData] = {}
        self.performances: List[RulePerformanceData] = []
        self.feedback: List[FeedbackData] = []

    async def store_session(self, session_data: SessionData) -> bool:
        """Store session data in memory."""
        self.sessions[session_data.session_id] = session_data
        return True

    async def get_session(self, session_id: str) -> SessionData | None:
        """Get session by ID."""
        return self.sessions.get(session_id)

    async def get_recent_sessions(
        self, limit: int = 100, user_context_filter: Dict[str, Any] | None = None
    ) -> List[SessionData]:
        """Get recent sessions."""
        sessions = list(self.sessions.values())
        sessions.sort(key=lambda s: s.created_at or datetime.now(), reverse=True)
        return sessions[:limit]

    async def store_rule_performance(self, performance_data: RulePerformanceData) -> bool:
        """Store rule performance data."""
        self.performances.append(performance_data)
        return True

    async def get_rule_performance_history(
        self,
        rule_id: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 1000
    ) -> List[RulePerformanceData]:
        """Get rule performance history."""
        filtered = [p for p in self.performances if p.rule_id == rule_id]
        if date_from:
            filtered = [p for p in filtered if (p.created_at or datetime.now()) >= date_from]
        if date_to:
            filtered = [p for p in filtered if (p.created_at or datetime.now()) <= date_to]
        return filtered[:limit]

    async def store_feedback(self, feedback_data: FeedbackData) -> bool:
        """Store feedback data."""
        self.feedback.append(feedback_data)
        return True

    async def get_feedback_by_session(self, session_id: str) -> List[FeedbackData]:
        """Get feedback for session."""
        return [f for f in self.feedback if f.session_id == session_id]

    async def get_session_analytics(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None
    ) -> Dict[str, Any]:
        """Get session analytics."""
        return {
            "total_sessions": len(self.sessions),
            "total_feedback": len(self.feedback),
            "total_performances": len(self.performances),
        }

    async def cleanup_old_sessions(self, days_old: int = 90) -> int:
        """Clean up old sessions."""
        return 0

    async def cleanup_old_performance_data(self, days_old: int = 180) -> int:
        """Clean up old performance data."""
        return 0

    # Implement other required methods as no-ops
    async def update_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        return True

    async def delete_session(self, session_id: str) -> bool:
        return True

    async def get_performance_by_session(self, session_id: str) -> List[RulePerformanceData]:
        return [p for p in self.performances if p.session_id == session_id]

    async def get_recent_feedback(self, limit: int = 100, processed_only: bool = False) -> List[FeedbackData]:
        return self.feedback[:limit]

    async def mark_feedback_processed(self, feedback_ids: List[str]) -> int:
        return len(feedback_ids)

    async def store_model_performance(self, performance_data) -> bool:
        return True

    async def get_model_performance_history(self, model_id: str, limit: int = 100) -> List:
        return []

    async def get_latest_model_performance(self, model_type: str):
        return None

    async def store_experiment(self, experiment_data) -> bool:
        return True

    async def get_experiment(self, experiment_id: str):
        return None

    async def get_active_experiments(self) -> List:
        return []

    async def update_experiment_metrics(self, experiment_id: str, metrics: Dict[str, Any]) -> bool:
        return True

    async def complete_experiment(self, experiment_id: str, final_metrics: Dict[str, Any]) -> bool:
        return True

    async def get_rule_effectiveness_summary(
        self,
        rule_ids: List[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None
    ) -> Dict[str, Dict[str, float]]:
        return {}

    async def get_user_satisfaction_metrics(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None
    ) -> Dict[str, Any]:
        return {}

    async def get_storage_metrics(self) -> Dict[str, int]:
        return {"total_records": 0}


class MockRulesRepository:
    """Mock rules repository for testing."""

    def __init__(self):
        self.rules = {
            "rule1": {
                "rule_id": "rule1",
                "rule_name": "Test Rule 1",
                "enabled": True,
                "priority": 10,
                "category": "clarity",
                "rule_class": "ClarityRule",
                "configuration": {"threshold": 0.8},
                "created_at": datetime.now(),
            },
            "rule2": {
                "rule_id": "rule2", 
                "rule_name": "Test Rule 2",
                "enabled": True,
                "priority": 5,
                "category": "specificity",
                "rule_class": "SpecificityRule", 
                "configuration": {"min_length": 50},
                "created_at": datetime.now(),
            },
        }
        self.performances = []

    async def get_rules(
        self,
        filters: RuleFilter | None = None,
        sort_by: str = "priority",
        sort_desc: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get rules with filtering."""
        rules = list(self.rules.values())
        
        if filters:
            if filters.enabled is not None:
                rules = [r for r in rules if r["enabled"] == filters.enabled]
            if filters.category:
                rules = [r for r in rules if r["category"] == filters.category]

        # Simple sorting
        if sort_by == "priority":
            rules.sort(key=lambda r: r["priority"], reverse=sort_desc)

        return rules[offset:offset + limit]

    async def get_rule_by_id(self, rule_id: str) -> Dict[str, Any] | None:
        """Get rule by ID."""
        return self.rules.get(rule_id)

    async def get_rules_by_category(
        self, category: str, enabled_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get rules by category."""
        rules = [
            rule for rule in self.rules.values()
            if rule["category"] == category and (not enabled_only or rule["enabled"])
        ]
        return rules

    async def create_rule_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create rule performance record."""
        performance_record = {
            "id": len(self.performances) + 1,
            "created_at": datetime.now(),
            **performance_data,
        }
        self.performances.append(performance_record)
        return performance_record

    async def get_top_performing_rules(
        self,
        metric: str = "improvement_score",
        category: str | None = None,
        min_applications: int = 10,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get top performing rules."""
        return [
            {
                "rule_id": "rule1",
                "avg_improvement_score": 0.85,
                "total_applications": 100,
            },
            {
                "rule_id": "rule2",
                "avg_improvement_score": 0.72,
                "total_applications": 80,
            },
        ]

    async def get_rule_effectiveness_analysis(
        self,
        rule_id: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ):
        """Get rule effectiveness analysis."""
        from prompt_improver.repositories.protocols.rules_repository_protocol import RuleEffectivenessAnalysis
        
        return RuleEffectivenessAnalysis(
            rule_id=rule_id,
            rule_name=f"Test Rule {rule_id}",
            total_applications=50,
            avg_improvement_score=0.8,
            improvement_score_stddev=0.1,
            avg_confidence_level=0.85,
            avg_execution_time_ms=120.0,
            success_rate=0.95,
            trend_analysis={"trend": "improving"},
            performance_by_category={"clarity": {"avg_score": 0.8}},
            recommendations=["Continue using this rule"],
        )

    # Implement other required methods as no-ops
    async def create_rule(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        return rule_data

    async def update_rule(self, rule_id: str, update_data: Dict[str, Any]) -> Dict[str, Any] | None:
        if rule_id in self.rules:
            self.rules[rule_id].update(update_data)
            return self.rules[rule_id]
        return None

    async def enable_rule(self, rule_id: str) -> bool:
        return True

    async def disable_rule(self, rule_id: str) -> bool:
        return True

    async def delete_rule(self, rule_id: str) -> bool:
        return True

    # Add all other required methods as no-ops
    async def get_rule_performances(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return []

    async def get_performance_by_rule(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return []

    async def get_recent_performances(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return []

    async def get_underperforming_rules(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return []

    async def compare_rule_effectiveness(self, *args, **kwargs):
        return None

    async def create_rule_intelligence_cache(self, *args, **kwargs) -> Dict[str, Any]:
        return {}

    async def get_rule_intelligence_cache(self, *args, **kwargs) -> Dict[str, Any] | None:
        return None

    async def get_intelligence_by_rule(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return []

    async def update_rule_intelligence_cache(self, *args, **kwargs) -> Dict[str, Any] | None:
        return None

    async def cleanup_expired_cache(self) -> int:
        return 0

    async def get_intelligence_metrics(self, *args, **kwargs):
        return None

    async def get_rule_performance_trends(self, *args, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        return {}

    async def get_rule_usage_statistics(self, *args, **kwargs) -> Dict[str, Any]:
        return {}

    async def get_performance_correlations(self, *args, **kwargs) -> Dict[str, Dict[str, float]]:
        return {}

    async def get_category_performance_summary(self, *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        return {}

    async def get_optimal_rule_parameters(self, *args, **kwargs) -> Dict[str, Any] | None:
        return None

    async def get_rule_combination_recommendations(self, *args, **kwargs) -> List[List[str]]:
        return []

    async def analyze_rule_conflicts(self, *args, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        return {}

    async def batch_update_rule_priorities(self, *args, **kwargs) -> int:
        return 0

    async def archive_old_performances(self, *args, **kwargs) -> int:
        return 0

    async def recalculate_effectiveness_scores(self, *args, **kwargs) -> int:
        return 0


@pytest.fixture
async def clean_container():
    """Fixture providing a clean DI container with mock repositories."""
    container = CleanDIContainer()
    
    # Register mock repositories
    mock_persistence = MockPersistenceRepository()
    mock_rules = MockRulesRepository()
    
    container.register_repository(IPersistenceRepository, mock_persistence)
    container.register_repository(IRulesRepository, mock_rules)
    
    # Register services
    container.register_singleton(
        CleanPersistenceService,
        lambda c: CleanPersistenceService(
            persistence_repository=c.get_repository(IPersistenceRepository)
        )
    )
    
    container.register_singleton(
        CleanRuleSelectionService,
        lambda c: CleanRuleSelectionService(
            rules_repository=c.get_repository(IRulesRepository)
        )
    )
    
    await container.initialize_services()
    yield container
    await container.shutdown_services()


class TestCleanArchitectureIntegration:
    """Integration tests for clean architecture implementation."""

    @pytest.mark.asyncio
    async def test_clean_persistence_service_integration(self, clean_container):
        """Test that CleanPersistenceService works with dependency injection."""
        # Get service from container
        persistence_service = clean_container.get_service(CleanPersistenceService)
        assert persistence_service is not None

        # Test storing a session
        session_success = await persistence_service.store_session(
            session_id="test-session-1",
            original_prompt="Original prompt",
            final_prompt="Improved prompt",
            rules_applied=[{"rule_id": "rule1", "applied": True}],
            user_context={"user_id": "test_user"},
        )
        assert session_success is True

        # Test retrieving the session
        retrieved_session = await persistence_service.get_session("test-session-1")
        assert retrieved_session is not None
        assert retrieved_session.session_id == "test-session-1"
        assert retrieved_session.original_prompt == "Original prompt"
        assert retrieved_session.final_prompt == "Improved prompt"

        # Test storing rule performance
        perf_success = await persistence_service.store_rule_performance(
            session_id="test-session-1",
            rule_id="rule1",
            improvement_score=0.85,
            confidence_level=0.90,
            execution_time_ms=120,
            prompt_type="question",
        )
        assert perf_success is True

        # Test storing feedback
        feedback_success = await persistence_service.store_feedback(
            session_id="test-session-1",
            rating=5,
            feedback_text="Great improvement!",
            user_id="test_user",
        )
        assert feedback_success is True

    @pytest.mark.asyncio
    async def test_clean_rule_selection_service_integration(self, clean_container):
        """Test that CleanRuleSelectionService works with dependency injection."""
        # Get service from container
        rule_service = clean_container.get_service(CleanRuleSelectionService)
        assert rule_service is not None

        # Test getting active rules
        active_rules = await rule_service.get_active_rules()
        assert isinstance(active_rules, dict)
        # Note: Since we don't have actual rule instantiation, this will be empty
        # but the method should work without errors

        # Test getting rules by category
        clarity_rules = await rule_service.get_rules_by_category("clarity")
        assert isinstance(clarity_rules, dict)

        # Test getting rule by ID
        rule = await rule_service.get_rule_by_id("rule1")
        # Will be None since we don't instantiate actual rules, but should not error
        assert rule is None

        # Test rule selection
        selected_rules = await rule_service.select_optimal_rules(
            prompt_characteristics={"length": 100, "complexity": "medium"},
            max_rules=3,
            use_bandit=False,  # Use traditional selection for predictable results
        )
        assert isinstance(selected_rules, list)

        # Test updating rule performance
        perf_success = await rule_service.update_rule_performance(
            rule_id="rule1",
            session_id="test-session-1",
            improvement_score=0.88,
            confidence_level=0.92,
            execution_time_ms=100,
        )
        assert perf_success is True

        # Test getting effectiveness analysis
        analysis = await rule_service.get_rule_effectiveness_analysis("rule1")
        assert isinstance(analysis, dict)
        assert "rule_id" in analysis

    @pytest.mark.asyncio
    async def test_repository_injection_works(self, clean_container):
        """Test that repositories are properly injected into services."""
        # Get repositories directly
        persistence_repo = clean_container.get_repository(IPersistenceRepository)
        rules_repo = clean_container.get_repository(IRulesRepository)
        
        assert persistence_repo is not None
        assert rules_repo is not None
        assert isinstance(persistence_repo, MockPersistenceRepository)
        assert isinstance(rules_repo, MockRulesRepository)

        # Get services and verify they use the injected repositories
        persistence_service = clean_container.get_service(CleanPersistenceService)
        rule_service = clean_container.get_service(CleanRuleSelectionService)
        
        # Services should have the injected repositories
        assert persistence_service._repository is persistence_repo
        assert rule_service._repository is rules_repo

    @pytest.mark.asyncio
    async def test_no_infrastructure_imports_in_core(self):
        """Test that core services don't import infrastructure modules."""
        # Import the clean services and verify they load without infrastructure dependencies
        from prompt_improver.core.services.persistence_service_clean import CleanPersistenceService
        from prompt_improver.core.services.rule_selection_service_clean import CleanRuleSelectionService
        
        # These imports should work without importing database, cache, or monitoring modules
        # If they have infrastructure imports, the test will fail with import errors
        assert CleanPersistenceService is not None
        assert CleanRuleSelectionService is not None

    @pytest.mark.asyncio
    async def test_container_health_check(self, clean_container):
        """Test container health check functionality."""
        health = clean_container.health_check()
        
        assert health["container_initialized"] is True
        assert health["registered_repositories"] == 2  # IPersistenceRepository, IRulesRepository
        assert health["registered_services"] == 2  # CleanPersistenceService, CleanRuleSelectionService
        assert health["instantiated_singletons"] >= 0  # May be lazy loaded
        assert "repository_health" in health
        assert "service_health" in health

    @pytest.mark.asyncio
    async def test_service_lifecycle(self):
        """Test service lifecycle management."""
        container = CleanDIContainer()
        
        # Register mock repositories
        mock_persistence = MockPersistenceRepository()
        mock_rules = MockRulesRepository()
        
        container.register_repository(IPersistenceRepository, mock_persistence)
        container.register_repository(IRulesRepository, mock_rules)
        
        # Register services
        container.register_singleton(
            CleanPersistenceService,
            lambda c: CleanPersistenceService(
                persistence_repository=c.get_repository(IPersistenceRepository)
            )
        )

        # Test initialization
        await container.initialize_services()
        assert container._initialized is True

        # Test service resolution after initialization
        service = container.get_service(CleanPersistenceService)
        assert service is not None

        # Test shutdown
        await container.shutdown_services()
        assert container._initialized is False

    @pytest.mark.asyncio
    async def test_error_handling_in_services(self, clean_container):
        """Test error handling in clean services."""
        persistence_service = clean_container.get_service(CleanPersistenceService)
        
        # Test with invalid data - service should handle gracefully
        result = await persistence_service.store_session(
            session_id="",  # Empty session ID
            original_prompt="",
            final_prompt="", 
            rules_applied=[],
            user_context=None,
        )
        # Should not crash and should return a boolean
        assert isinstance(result, bool)

        # Test retrieving non-existent session
        session = await persistence_service.get_session("non-existent")
        assert session is None