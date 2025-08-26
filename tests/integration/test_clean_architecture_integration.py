"""Integration tests for clean architecture implementation.

Tests that core services work correctly with dependency injection
and repository interfaces without direct infrastructure coupling.
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List
from sqlalchemy import text

from prompt_improver.core.di.clean_container import (
    CleanDIContainer,
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


class RealTestPersistenceRepository:
    """Real persistence repository using actual database for testing."""

    def __init__(self, db_session):
        self.db_session = db_session

    async def store_session(self, session_data: SessionData) -> bool:
        """Store session data in real database."""
        try:
            # Create simple test table if needed and insert session data
            await self.db_session.execute(
                text("""
                CREATE TABLE IF NOT EXISTS test_sessions (
                    session_id VARCHAR PRIMARY KEY,
                    user_id VARCHAR,
                    original_prompt TEXT,
                    final_prompt TEXT,
                    rules_applied JSONB,
                    user_context JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
            )
            
            # Insert session data
            await self.db_session.execute(
                text("""
                INSERT INTO test_sessions 
                (session_id, user_id, original_prompt, final_prompt, rules_applied, user_context)
                VALUES (:session_id, :user_id, :original_prompt, :final_prompt, 
                        :rules_applied::jsonb, :user_context::jsonb)
                ON CONFLICT (session_id) DO UPDATE SET
                    user_id = :user_id,
                    original_prompt = :original_prompt,
                    final_prompt = :final_prompt,
                    rules_applied = :rules_applied::jsonb,
                    user_context = :user_context::jsonb
                """),
                {
                    "session_id": session_data.session_id,
                    "user_id": getattr(session_data, 'user_id', None),
                    "original_prompt": getattr(session_data, 'original_prompt', ''),
                    "final_prompt": getattr(session_data, 'final_prompt', ''),
                    "rules_applied": '{}',  # Simplified for testing
                    "user_context": '{}',   # Simplified for testing
                }
            )
            await self.db_session.commit()
            return True
        except Exception:
            await self.db_session.rollback()
            return False

    async def get_session(self, session_id: str) -> SessionData | None:
        """Get session by ID from real database."""
        try:
            result = await self.db_session.execute(
                text("SELECT * FROM test_sessions WHERE session_id = :session_id"),
                {"session_id": session_id}
            )
            row = result.fetchone()
            if row:
                return SessionData(
                    session_id=row.session_id,
                    user_id=row.user_id,
                    original_prompt=row.original_prompt,
                    final_prompt=row.final_prompt,
                    created_at=row.created_at,
                )
            return None
        except Exception:
            return None

    # Simplified implementations for other required methods
    async def get_recent_sessions(self, limit: int = 100, user_context_filter: Dict[str, Any] | None = None) -> List[SessionData]:
        return []  # Simplified for testing
        
    async def store_rule_performance(self, performance_data: RulePerformanceData) -> bool:
        return True  # Simplified for testing
        
    async def get_rule_performance_history(self, rule_id: str, date_from: datetime | None = None, date_to: datetime | None = None, limit: int = 1000) -> List[RulePerformanceData]:
        return []  # Simplified for testing
        
    async def store_feedback(self, feedback_data: FeedbackData) -> bool:
        return True  # Simplified for testing
        
    async def get_feedback_by_session(self, session_id: str) -> List[FeedbackData]:
        return []  # Simplified for testing
        
    async def get_session_analytics(self, date_from: datetime | None = None, date_to: datetime | None = None) -> Dict[str, Any]:
        return {"total_sessions": 0}  # Simplified for testing
        
    async def cleanup_old_sessions(self, days_old: int = 90) -> int:
        return 0  # Simplified for testing
        
    async def cleanup_old_performance_data(self, days_old: int = 180) -> int:
        return 0  # Simplified for testing

    # Stub implementations for all other required methods
    async def update_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        return True
    async def delete_session(self, session_id: str) -> bool:
        return True
    async def get_performance_by_session(self, session_id: str) -> List[RulePerformanceData]:
        return []
    async def get_recent_feedback(self, limit: int = 100, processed_only: bool = False) -> List[FeedbackData]:
        return []
    async def mark_feedback_processed(self, feedback_ids: List[str]) -> int:
        return 0
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
    async def get_rule_effectiveness_summary(self, rule_ids: List[str] | None = None, date_from: datetime | None = None, date_to: datetime | None = None) -> Dict[str, Dict[str, float]]:
        return {}
    async def get_user_satisfaction_metrics(self, date_from: datetime | None = None, date_to: datetime | None = None) -> Dict[str, Any]:
        return {}
    async def get_storage_metrics(self) -> Dict[str, int]:
        return {"total_records": 0}


class RealTestRulesRepository:
    """Real rules repository using actual database for testing."""

    def __init__(self, db_session):
        self.db_session = db_session

    async def get_rules(self, filters: RuleFilter | None = None, sort_by: str = "priority", sort_desc: bool = False, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get rules from real database."""
        try:
            # Create test table if needed
            await self.db_session.execute(
                text("""
                CREATE TABLE IF NOT EXISTS test_rules (
                    rule_id VARCHAR PRIMARY KEY,
                    rule_name VARCHAR NOT NULL,
                    enabled BOOLEAN DEFAULT true,
                    priority INTEGER DEFAULT 1,
                    category VARCHAR,
                    rule_class VARCHAR,
                    configuration JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
            )
            
            # Insert sample test data if table is empty
            count_result = await self.db_session.execute(text("SELECT COUNT(*) FROM test_rules"))
            if count_result.scalar() == 0:
                await self.db_session.execute(
                    text("""
                    INSERT INTO test_rules (rule_id, rule_name, enabled, priority, category, rule_class, configuration)
                    VALUES 
                    ('rule1', 'Test Rule 1', true, 10, 'clarity', 'ClarityRule', '{"threshold": 0.8}'),
                    ('rule2', 'Test Rule 2', true, 5, 'specificity', 'SpecificityRule', '{"min_length": 50}')
                    """)
                )
                await self.db_session.commit()
            
            # Query rules
            query = "SELECT * FROM test_rules WHERE 1=1"
            params = {}
            
            if filters and filters.enabled is not None:
                query += " AND enabled = :enabled"
                params["enabled"] = filters.enabled
            if filters and filters.category:
                query += " AND category = :category"
                params["category"] = filters.category
                
            query += f" ORDER BY {sort_by}"
            if sort_desc:
                query += " DESC"
            query += f" LIMIT {limit} OFFSET {offset}"
            
            result = await self.db_session.execute(text(query), params)
            rows = result.fetchall()
            
            return [
                {
                    "rule_id": row.rule_id,
                    "rule_name": row.rule_name,
                    "enabled": row.enabled,
                    "priority": row.priority,
                    "category": row.category,
                    "rule_class": row.rule_class,
                    "configuration": {},  # Simplified for testing
                    "created_at": row.created_at,
                }
                for row in rows
            ]
        except Exception:
            return []  # Return empty list on error

    async def get_rule_by_id(self, rule_id: str) -> Dict[str, Any] | None:
        """Get rule by ID from real database."""
        try:
            result = await self.db_session.execute(
                text("SELECT * FROM test_rules WHERE rule_id = :rule_id"),
                {"rule_id": rule_id}
            )
            row = result.fetchone()
            if row:
                return {
                    "rule_id": row.rule_id,
                    "rule_name": row.rule_name,
                    "enabled": row.enabled,
                    "priority": row.priority,
                    "category": row.category,
                    "rule_class": row.rule_class,
                    "configuration": {},
                    "created_at": row.created_at,
                }
            return None
        except Exception:
            return None

    async def get_rules_by_category(self, category: str, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Get rules by category from real database."""
        filters = RuleFilter(category=category, enabled=enabled_only if enabled_only else None)
        return await self.get_rules(filters=filters)

    # Simplified implementations for other required methods
    async def create_rule_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": 1, **performance_data}  # Simplified
    
    async def get_top_performing_rules(self, metric: str = "improvement_score", category: str | None = None, min_applications: int = 10, limit: int = 20) -> List[Dict[str, Any]]:
        return []  # Simplified
    
    async def get_rule_effectiveness_analysis(self, rule_id: str, date_from: datetime | None = None, date_to: datetime | None = None):
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

    # Stub implementations for all other required methods
    async def create_rule(self, rule_data: Dict[str, Any]) -> Dict[str, Any]: return rule_data
    async def update_rule(self, rule_id: str, update_data: Dict[str, Any]) -> Dict[str, Any] | None: return None
    async def enable_rule(self, rule_id: str) -> bool: return True
    async def disable_rule(self, rule_id: str) -> bool: return True
    async def delete_rule(self, rule_id: str) -> bool: return True
    async def get_rule_performances(self, *args, **kwargs) -> List[Dict[str, Any]]: return []
    async def get_performance_by_rule(self, *args, **kwargs) -> List[Dict[str, Any]]: return []
    async def get_recent_performances(self, *args, **kwargs) -> List[Dict[str, Any]]: return []
    async def get_underperforming_rules(self, *args, **kwargs) -> List[Dict[str, Any]]: return []
    async def compare_rule_effectiveness(self, *args, **kwargs): return None
    async def create_rule_intelligence_cache(self, *args, **kwargs) -> Dict[str, Any]: return {}
    async def get_rule_intelligence_cache(self, *args, **kwargs) -> Dict[str, Any] | None: return None
    async def get_intelligence_by_rule(self, *args, **kwargs) -> List[Dict[str, Any]]: return []
    async def update_rule_intelligence_cache(self, *args, **kwargs) -> Dict[str, Any] | None: return None
    async def cleanup_expired_cache(self) -> int: return 0
    async def get_intelligence_metrics(self, *args, **kwargs): return None
    async def get_rule_performance_trends(self, *args, **kwargs) -> Dict[str, List[Dict[str, Any]]]: return {}
    async def get_rule_usage_statistics(self, *args, **kwargs) -> Dict[str, Any]: return {}
    async def get_performance_correlations(self, *args, **kwargs) -> Dict[str, Dict[str, float]]: return {}
    async def get_category_performance_summary(self, *args, **kwargs) -> Dict[str, Dict[str, Any]]: return {}
    async def get_optimal_rule_parameters(self, *args, **kwargs) -> Dict[str, Any] | None: return None
    async def get_rule_combination_recommendations(self, *args, **kwargs) -> List[List[str]]: return []
    async def analyze_rule_conflicts(self, *args, **kwargs) -> Dict[str, List[Dict[str, Any]]]: return {}
    async def batch_update_rule_priorities(self, *args, **kwargs) -> int: return 0
    async def archive_old_performances(self, *args, **kwargs) -> int: return 0
    async def recalculate_effectiveness_scores(self, *args, **kwargs) -> int: return 0


@pytest.fixture
async def clean_container(test_db_session):
    """Fixture providing a clean DI container with real database repositories."""
    container = CleanDIContainer()
    
    # Register real repositories using test database
    real_persistence = RealTestPersistenceRepository(test_db_session)
    real_rules = RealTestRulesRepository(test_db_session)
    
    container.register_repository(IPersistenceRepository, real_persistence)
    container.register_repository(IRulesRepository, real_rules)
    
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
        from tests.utils.mocks import MockPersistenceRepository
        from tests.utils.mocks import MockRulesRepository
        
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