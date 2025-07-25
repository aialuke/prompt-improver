"""
Comprehensive Pydantic validation test suite to ensure all models work correctly.
Tests real behavior (not mocks) across all Pydantic models in the codebase.
"""

import json
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Import Pydantic models from database
from prompt_improver.database.models import (
    PromptSession,
    ABExperiment,
    RuleMetadata,
    RulePerformance,
    DiscoveredPattern,
    UserFeedback,
    MLModelPerformance,
    ImprovementSession,
    TrainingSession,
    TrainingIteration,
    AprioriAssociationRule,
    AprioriPatternDiscovery,
    FrequentItemset,
    PatternEvaluation,
    AdvancedPatternResults,
    ImprovementSessionCreate,
    ABExperimentCreate,
    RulePerformanceCreate,
    UserFeedbackCreate,
    TrainingSessionCreate,
    TrainingSessionUpdate,
    AprioriAnalysisRequest,
    AprioriAnalysisResponse,
    PatternDiscoveryRequest,
    PatternDiscoveryResponse,
    RuleEffectivenessStats,
    UserSatisfactionStats,
    TrainingPrompt,
    GenerationSession,
    GenerationBatch,
    GenerationMethodPerformance,
    SyntheticDataSample,
    GenerationQualityAssessment,
    GenerationAnalytics
)

# Import other Pydantic models
from prompt_improver.utils.redis_cache import RedisConfig
from prompt_improver.database.config import DatabaseConfig


class TestPydanticModelImports:
    """Test that all Pydantic models can be imported without errors"""
    
    def test_database_models_import(self):
        """Test that all database models import correctly"""
        models = [
            PromptSession, ABExperiment, RuleMetadata, RulePerformance,
            DiscoveredPattern, UserFeedback, MLModelPerformance,
            ImprovementSession, TrainingSession, TrainingIteration,
            AprioriAssociationRule, AprioriPatternDiscovery, FrequentItemset,
            PatternEvaluation, AdvancedPatternResults, TrainingPrompt,
            GenerationSession, GenerationBatch, GenerationMethodPerformance,
            SyntheticDataSample, GenerationQualityAssessment, GenerationAnalytics
        ]
        
        for model in models:
            assert hasattr(model, '__annotations__'), f"{model.__name__} should have annotations"
            assert hasattr(model, 'model_validate'), f"{model.__name__} should have model_validate method"
    
    def test_api_models_import(self):
        """Test that all API models import correctly"""
        api_models = [
            ImprovementSessionCreate, ABExperimentCreate, RulePerformanceCreate,
            UserFeedbackCreate, TrainingSessionCreate, TrainingSessionUpdate,
            AprioriAnalysisRequest, AprioriAnalysisResponse,
            PatternDiscoveryRequest, PatternDiscoveryResponse,
            RuleEffectivenessStats, UserSatisfactionStats
        ]
        
        for model in api_models:
            assert hasattr(model, '__annotations__'), f"{model.__name__} should have annotations"
            assert hasattr(model, 'model_validate'), f"{model.__name__} should have model_validate method"


class TestBaseModelSerialization:
    """Test BaseModel serialization methods work correctly"""
    
    def test_model_dump_basic(self):
        """Test basic model_dump functionality"""
        session = ImprovementSessionCreate(
            session_id="test-123",
            original_prompt="Original prompt",
            final_prompt="Improved prompt",
            rules_applied=["rule1", "rule2"],
            user_context={"domain": "business"},
            improvement_metrics={"score": 0.85}
        )
        
        # Test model_dump
        data = session.model_dump()
        assert isinstance(data, dict)
        assert data["session_id"] == "test-123"
        assert data["original_prompt"] == "Original prompt"
        assert data["rules_applied"] == ["rule1", "rule2"]
        assert data["user_context"]["domain"] == "business"
        assert data["improvement_metrics"]["score"] == 0.85
    
    def test_model_dump_json(self):
        """Test model_dump_json functionality"""
        experiment = ABExperimentCreate(
            experiment_name="Test Experiment",
            description="A test experiment",
            control_rules={"rule1": {"param": "value1"}},
            treatment_rules={"rule1": {"param": "value2"}},
            target_metric="improvement_score",
            sample_size_per_group=100,
            status="running"
        )
        
        # Test model_dump_json
        json_str = experiment.model_dump_json()
        assert isinstance(json_str, str)
        
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["experiment_name"] == "Test Experiment"
        assert parsed["control_rules"]["rule1"]["param"] == "value1"
        assert parsed["treatment_rules"]["rule1"]["param"] == "value2"
    
    def test_model_dump_exclude(self):
        """Test model_dump with exclude parameter"""
        feedback = UserFeedbackCreate(
            session_id="session-123",
            rating=4,
            feedback_text="Good improvement",
            improvement_areas=["clarity", "tone"]
        )
        
        # Test exclude parameter
        data = feedback.model_dump(exclude={"feedback_text"})
        assert "session_id" in data
        assert "rating" in data
        assert "improvement_areas" in data
        assert "feedback_text" not in data
    
    def test_model_dump_include(self):
        """Test model_dump with include parameter"""
        rule_perf = RulePerformanceCreate(
            session_id="session-123",
            rule_id="rule-456",
            improvement_score=0.75,
            execution_time_ms=250.0,
            confidence_level=0.85,
            parameters_used={"param1": "value1"}
        )
        
        # Test include parameter
        data = rule_perf.model_dump(include={"rule_id", "improvement_score"})
        assert len(data) == 2
        assert data["rule_id"] == "rule-456"
        assert data["improvement_score"] == 0.75


class TestBaseModelValidation:
    """Test BaseModel validation methods work correctly"""
    
    def test_model_validate_success(self):
        """Test successful model_validate"""
        data = {
            "window_days": 30,
            "min_support": 0.1,
            "min_confidence": 0.6,
            "min_lift": 1.0,
            "max_itemset_length": 5,
            "save_to_database": True
        }
        
        # Test model_validate
        request = AprioriAnalysisRequest.model_validate(data)
        assert request.window_days == 30
        assert request.min_support == 0.1
        assert request.min_confidence == 0.6
        assert request.min_lift == 1.0
        assert request.max_itemset_length == 5
        assert request.save_to_database is True
    
    def test_model_validate_with_defaults(self):
        """Test model_validate with default values"""
        minimal_data = {}
        
        # Test with defaults
        request = AprioriAnalysisRequest.model_validate(minimal_data)
        assert request.window_days == 30  # default
        assert request.min_support == 0.1  # default
        assert request.min_confidence == 0.6  # default
        assert request.min_lift == 1.0  # default
        assert request.max_itemset_length == 5  # default
        assert request.save_to_database is True  # default
    
    def test_model_validate_from_json(self):
        """Test model validation from JSON string"""
        json_data = """
        {
            "session_id": "training-789",
            "continuous_mode": false,
            "max_iterations": 100,
            "improvement_threshold": 0.05,
            "timeout_seconds": 1800,
            "auto_init_enabled": false
        }
        """
        
        # Test model_validate from JSON
        training_session = TrainingSessionCreate.model_validate_json(json_data)
        assert training_session.session_id == "training-789"
        assert training_session.continuous_mode is False
        assert training_session.max_iterations == 100
        assert training_session.improvement_threshold == 0.05
        assert training_session.timeout_seconds == 1800
        assert training_session.auto_init_enabled is False
    
    def test_model_validate_error_handling(self):
        """Test model validation error handling"""
        # Test with invalid data
        invalid_data = {
            "rating": 10,  # should be between 1-5
            "session_id": "test"
        }
        
        with pytest.raises(Exception):  # Should raise validation error
            UserFeedbackCreate.model_validate(invalid_data)
    
    def test_field_constraints_validation(self):
        """Test that Field constraints are properly validated"""
        # Test ge/le constraints
        with pytest.raises(Exception):
            AprioriAnalysisRequest.model_validate({
                "window_days": 0,  # should be >= 1
                "min_support": 0.1
            })
        
        with pytest.raises(Exception):
            AprioriAnalysisRequest.model_validate({
                "window_days": 400,  # should be <= 365
                "min_support": 0.1
            })
        
        # Test valid constraints
        valid_request = AprioriAnalysisRequest.model_validate({
            "window_days": 30,
            "min_support": 0.1,
            "min_confidence": 0.6,
            "min_lift": 1.0,
            "max_itemset_length": 5
        })
        assert valid_request.window_days == 30


class TestComplexFieldValidation:
    """Test validation of complex fields like JSON, lists, etc."""
    
    def test_json_field_validation(self):
        """Test validation of JSON fields"""
        data = {
            "discovery_run_id": "test-run-123",
            "transaction_count": 150,
            "frequent_itemsets_count": 25,
            "association_rules_count": 10,
            "execution_time_seconds": 45.5,
            "top_itemsets": [
                {"itemset": ["rule1", "rule2"], "support": 0.8},
                {"itemset": ["rule3"], "support": 0.6}
            ],
            "top_rules": [
                {"antecedent": ["rule1"], "consequent": ["rule2"], "confidence": 0.9}
            ],
            "pattern_insights": {
                "most_effective_combination": ["rule1", "rule2"],
                "improvement_correlation": 0.85
            },
            "config": {
                "min_support": 0.1,
                "min_confidence": 0.6
            },
            "status": "completed",
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
        # Test complex JSON validation
        response = AprioriAnalysisResponse.model_validate(data)
        assert response.discovery_run_id == "test-run-123"
        assert len(response.top_itemsets) == 2
        assert response.top_itemsets[0]["support"] == 0.8
        assert response.pattern_insights["improvement_correlation"] == 0.85
    
    def test_list_field_validation(self):
        """Test validation of list fields"""
        data = {
            "min_effectiveness": 0.7,
            "min_support": 5,
            "use_advanced_discovery": True,
            "include_apriori": True,
            "pattern_types": ["association", "sequence", "parameter"],
            "use_ensemble": True
        }
        
        discovery_request = PatternDiscoveryRequest.model_validate(data)
        assert discovery_request.pattern_types == ["association", "sequence", "parameter"]
        assert discovery_request.use_advanced_discovery is True
    
    def test_optional_field_validation(self):
        """Test validation of optional fields"""
        # Test with minimal required data
        minimal_data = {
            "rule_id": "test-rule",
            "rule_name": "Test Rule",
            "usage_count": 10,
            "avg_improvement": 0.75,
            "score_stddev": 0.15,
            "min_improvement": 0.5,
            "max_improvement": 0.9,
            "avg_confidence": 0.8,
            "avg_execution_time": 150.0,
            "prompt_types_count": 5
        }
        
        stats = RuleEffectivenessStats.model_validate(minimal_data)
        assert stats.rule_id == "test-rule"
        assert stats.avg_improvement == 0.75


class TestDatabaseModelValidation:
    """Test validation of database models (SQLModel)"""
    
    def test_table_model_creation(self):
        """Test creation of table models"""
        from prompt_improver.utils.datetime_utils import naive_utc_now
        
        # Test PromptSession creation
        session_data = {
            "session_id": "test-session-123",
            "original_prompt": "Original prompt text",
            "improved_prompt": "Improved prompt text",
            "user_context": {"domain": "technical", "complexity": "high"},
            "quality_score": 0.85,
            "improvement_score": 0.75,
            "confidence_level": 0.9
        }
        
        # This should work without database connection
        session = PromptSession(**session_data)
        assert session.session_id == "test-session-123"
        assert session.user_context["domain"] == "technical"
        assert session.quality_score == 0.85
    
    def test_foreign_key_field_validation(self):
        """Test validation of foreign key fields"""
        rule_perf_data = {
            "rule_id": "rule-123",
            "rule_name": "Test Rule",
            "prompt_id": "prompt-456",
            "prompt_type": "business",
            "improvement_score": 0.8,
            "confidence_level": 0.85,
            "execution_time_ms": 200,
            "rule_parameters": {"param1": "value1"},
            "prompt_characteristics": {"length": 100, "complexity": "medium"}
        }
        
        rule_perf = RulePerformance(**rule_perf_data)
        assert rule_perf.rule_id == "rule-123"
        assert rule_perf.prompt_id == "prompt-456"
        assert rule_perf.rule_parameters["param1"] == "value1"
    
    def test_protected_namespace_config(self):
        """Test that model_config for protected namespaces works"""
        # Test UserFeedback which has model_config
        feedback_data = {
            "session_id": "session-123",
            "rating": 4,
            "feedback_text": "Great improvement!",
            "improvement_areas": ["clarity", "conciseness"],
            "is_processed": False,
            "ml_optimized": True,
            "model_id": "model-789"
        }
        
        feedback = UserFeedback(**feedback_data)
        assert feedback.model_id == "model-789"  # This should work with protected namespace fix
        assert feedback.rating == 4


class TestConfigurationModels:
    """Test validation of configuration models from various modules"""
    
    def test_redis_config_validation(self):
        """Test RedisConfig validation"""
        config_data = {
            "host": "localhost",
            "port": 6379,
            "cache_db": 0,
            "max_connections": 10,
            "socket_timeout": 30,
            "connect_timeout": 5,
            "use_ssl": False
        }
        
        config = RedisConfig.model_validate(config_data)
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.max_connections == 10
        assert config.socket_timeout == 30
        assert config.use_ssl is False


class TestRealBehaviorValidation:
    """Test real behavior validation with actual data scenarios"""
    
    def test_end_to_end_model_workflow(self):
        """Test complete workflow using multiple models"""
        # 1. Create training session
        training_data = {
            "session_id": "training-real-test",
            "continuous_mode": True,
            "max_iterations": 50,
            "improvement_threshold": 0.03,
            "timeout_seconds": 1800
        }
        training_session = TrainingSessionCreate.model_validate(training_data)
        
        # 2. Create training session update
        update_data = {
            "status": "running",
            "current_iteration": 10,
            "current_performance": 0.75,
            "best_performance": 0.78,
            "performance_history": [0.65, 0.70, 0.75, 0.78],
            "data_points_processed": 1000,
            "models_trained": 5
        }
        session_update = TrainingSessionUpdate.model_validate(update_data)
        
        # 3. Create apriori analysis request
        apriori_data = {
            "window_days": 14,
            "min_support": 0.15,
            "min_confidence": 0.7,
            "min_lift": 1.2,
            "max_itemset_length": 4
        }
        apriori_request = AprioriAnalysisRequest.model_validate(apriori_data)
        
        # Verify all models work together
        assert training_session.session_id == "training-real-test"
        assert session_update.current_iteration == 10
        assert apriori_request.window_days == 14
        
        # Test serialization round-trip
        training_json = training_session.model_dump_json()
        training_restored = TrainingSessionCreate.model_validate_json(training_json)
        assert training_restored.session_id == training_session.session_id
    
    def test_complex_nested_validation(self):
        """Test validation of complex nested data structures"""
        complex_data = {
            "status": "completed",
            "discovery_run_id": "advanced-run-123",
            "traditional_patterns": {
                "rule_combinations": [
                    {"rules": ["rule1", "rule2"], "effectiveness": 0.85},
                    {"rules": ["rule3", "rule4"], "effectiveness": 0.78}
                ],
                "parameter_patterns": {
                    "optimal_ranges": {"param1": [0.5, 0.8], "param2": [10, 50]}
                }
            },
            "advanced_patterns": {
                "clustering_results": {
                    "clusters": [
                        {"id": 1, "size": 45, "centroid": [0.75, 0.82]},
                        {"id": 2, "size": 32, "centroid": [0.65, 0.71]}
                    ]
                }
            },
            "apriori_patterns": {
                "frequent_itemsets": [
                    {"itemset": ["rule1", "rule2"], "support": 0.75},
                    {"itemset": ["rule1", "rule3"], "support": 0.68}
                ],
                "association_rules": [
                    {
                        "antecedent": ["rule1"],
                        "consequent": ["rule2"],
                        "confidence": 0.85,
                        "lift": 1.25
                    }
                ]
            },
            "cross_validation": {
                "fold_results": [0.85, 0.82, 0.88, 0.79, 0.86],
                "mean_score": 0.84,
                "std_score": 0.033
            },
            "unified_recommendations": [
                {
                    "recommendation": "Use rule1 and rule2 together",
                    "confidence": 0.9,
                    "expected_improvement": 0.15
                }
            ],
            "business_insights": {
                "top_performing_combinations": ["rule1+rule2", "rule3+rule4"],
                "improvement_potential": 0.25
            },
            "discovery_metadata": {
                "algorithms_used": ["apriori", "hdbscan", "fp_growth"],
                "execution_time": 125.5,
                "data_points_analyzed": 5000
            }
        }
        
        response = PatternDiscoveryResponse.model_validate(complex_data)
        assert response.status == "completed"
        assert len(response.unified_recommendations) == 1
        assert response.unified_recommendations[0]["confidence"] == 0.9
        assert "rule1+rule2" in response.business_insights["top_performing_combinations"]


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in validation"""
    
    def test_validation_error_messages(self):
        """Test that validation errors provide useful messages"""
        # Test invalid rating
        with pytest.raises(Exception) as exc_info:
            UserFeedbackCreate.model_validate({
                "session_id": "test",
                "rating": 0  # Invalid: should be 1-5
            })
        
        # The error should contain information about the constraint
        error_str = str(exc_info.value)
        assert "rating" in error_str or "greater" in error_str
    
    def test_none_vs_optional_handling(self):
        """Test proper handling of None vs Optional fields"""
        # Test with explicit None values
        data = {
            "session_id": "test-session",
            "original_prompt": "Test prompt",
            "improved_prompt": "Better test prompt",
            "user_context": None,  # Optional field
            "quality_score": None,  # Optional field
            "improvement_score": 0.5,
            "confidence_level": None  # Optional field
        }
        
        # This should work without errors
        session = PromptSession(**data)
        assert session.user_context is None
        assert session.quality_score is None
        assert session.improvement_score == 0.5
    
    def test_empty_collections_validation(self):
        """Test validation with empty collections"""
        # Test with empty lists and dicts
        data = {
            "session_id": "test-empty",
            "original_prompt": "Test",
            "final_prompt": "Better test",
            "rules_applied": [],  # Empty list
            "user_context": {},  # Empty dict
            "improvement_metrics": {}  # Empty dict
        }
        
        session = ImprovementSessionCreate.model_validate(data)
        assert session.rules_applied == []
        assert session.user_context == {}
        assert session.improvement_metrics == {}


def test_import_all_models():
    """Integration test: Import and validate all Pydantic models can be used"""
    # This test ensures we can import everything without circular imports or other issues
    
    # Test database models
    from prompt_improver.database.models import (
        PromptSession, TrainingPrompt, GenerationSession
    )
    
    # Test configuration models
    from prompt_improver.database.config import DatabaseConfig
    from prompt_improver.utils.redis_cache import RedisConfig
    
    # Test that all models have the expected Pydantic methods
    models_to_test = [
        PromptSession, TrainingPrompt, GenerationSession,
        DatabaseConfig, RedisConfig
    ]
    
    for model in models_to_test:
        # Check for Pydantic methods
        assert hasattr(model, 'model_validate'), f"{model.__name__} missing model_validate"
        if not hasattr(model, '__tablename__'):  # Non-SQLModel classes
            assert hasattr(model, 'model_dump'), f"{model.__name__} missing model_dump"
            assert hasattr(model, 'model_dump_json'), f"{model.__name__} missing model_dump_json"


def test_redis_config_real_validation():
    """Test RedisConfig validation with real scenarios"""
    # Test successful redis config with actual field names
    redis_data = {
        "host": "localhost",
        "port": 6379,
        "cache_db": 0,
        "max_connections": 10,
        "socket_timeout": 30,
        "connect_timeout": 5,
        "use_ssl": False
    }
    
    config = RedisConfig.model_validate(redis_data)
    assert config.host == "localhost"
    assert config.port == 6379
    assert config.max_connections == 10
    assert config.socket_timeout == 30
    assert config.use_ssl is False


if __name__ == "__main__":
    # Run tests directly if called as script
    pytest.main([__file__, "-v"])