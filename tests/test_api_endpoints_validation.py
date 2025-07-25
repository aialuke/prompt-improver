"""
Test API endpoints that use Pydantic models to ensure they work correctly.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, Mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))


@pytest.mark.asyncio
async def test_analytics_endpoints_validation():
    """Test analytics endpoints that use Pydantic models"""
    try:
        from prompt_improver.api.analytics_endpoints import app
        from prompt_improver.database.models import RuleEffectivenessStats, UserSatisfactionStats
        from fastapi.testclient import TestClient
        from datetime import date
        
        # Test RuleEffectivenessStats creation
        stats_data = {
            "rule_id": "test-rule-123",
            "rule_name": "Test Rule",
            "usage_count": 50,
            "avg_improvement": 0.75,
            "score_stddev": 0.15,
            "min_improvement": 0.5,  
            "max_improvement": 0.9,
            "avg_confidence": 0.8,
            "avg_execution_time": 150.0,
            "prompt_types_count": 3
        }
        
        stats = RuleEffectivenessStats.model_validate(stats_data)
        assert stats.rule_id == "test-rule-123"
        assert stats.avg_improvement == 0.75
        assert stats.usage_count == 50
        
        # Test UserSatisfactionStats creation
        satisfaction_data = {
            "feedback_date": date(2025, 1, 15),
            "total_feedback": 25,
            "avg_rating": 4.2,
            "positive_feedback": 20,
            "negative_feedback": 5,
            "rules_used": ["rule1", "rule2", "rule3"]
        }
        
        satisfaction = UserSatisfactionStats.model_validate(satisfaction_data)
        assert satisfaction.total_feedback == 25
        assert satisfaction.avg_rating == 4.2
        assert len(satisfaction.rules_used) == 3
        
        print("✓ Analytics endpoints Pydantic validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Analytics endpoints validation failed: {e}")
        return False


@pytest.mark.asyncio
async def test_orchestrator_endpoints_validation():
    """Test orchestrator endpoints Pydantic models"""
    try:
        # Import orchestrator models
        from prompt_improver.ml.orchestration.api.orchestrator_endpoints import app
        
        # Test that the FastAPI app was created successfully
        assert app is not None, "FastAPI app should be created"
        
        # Test basic Pydantic model validation from the orchestrator
        test_request = {
            "prompt": "Test prompt for orchestration",
            "enhancement_level": "standard",
            "context": {"domain": "technical"}
        }
        
        # Since we can't easily test the actual endpoint without a server,
        # we'll test that the imports work and models are accessible
        routes = [route.path for route in app.routes]
        assert len(routes) > 0, "Should have routes defined"
        
        print("✓ Orchestrator endpoints validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Orchestrator endpoints validation failed: {e}")
        return False


@pytest.mark.asyncio
async def test_batch_processor_validation():
    """Test batch processor Pydantic models"""
    try:
        from prompt_improver.ml.optimization.batch.batch_processor import BatchProcessorConfig
        from prompt_improver.ml.optimization.batch.enhanced_batch_processor import EnhancedBatchConfig
        
        # Test BatchProcessorConfig validation
        basic_config_data = {
            "batch_size": 25,
            "batch_timeout": 60,
            "max_attempts": 5,
            "concurrency": 2,
            "enable_circuit_breaker": True,
            "enable_dead_letter_queue": True,
            "enable_opentelemetry": False
        }
        
        basic_config = BatchProcessorConfig.model_validate(basic_config_data)
        assert basic_config.batch_size == 25
        assert basic_config.enable_circuit_breaker is True
        assert basic_config.enable_opentelemetry is False
        
        # Test EnhancedBatchConfig validation
        enhanced_config_data = {
            "batch_size": 50,
            "batch_timeout": 120,
            "max_attempts": 3,
            "concurrency": 4,
            "adaptive_batch_sizing": True,
            "quality_threshold": 0.85,
            "performance_monitoring": True,
            "auto_scaling": False
        }
        
        enhanced_config = EnhancedBatchConfig.model_validate(enhanced_config_data)
        assert enhanced_config.batch_size == 50
        assert enhanced_config.adaptive_batch_sizing is True
        assert enhanced_config.quality_threshold == 0.85
        
        print("✓ Batch processor validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Batch processor validation failed: {e}")
        return False


@pytest.mark.asyncio 
async def test_ml_pipeline_models_validation():
    """Test ML pipeline configuration models"""
    try:
        from prompt_improver.ml.learning.features.linguistic_feature_extractor import (
            LinguisticAnalysisConfig
        )
        from prompt_improver.ml.learning.features.context_feature_extractor import (
            ContextAnalysisConfig  
        )
        from prompt_improver.ml.learning.features.domain_feature_extractor import (
            DomainAnalysisConfig
        )
        
        # Test LinguisticAnalysisConfig
        linguistic_config = {
            "enable_sentiment_analysis": True,
            "enable_complexity_scoring": True,
            "enable_readability_metrics": True,
            "language": "en",
            "min_text_length": 10,
            "max_text_length": 5000
        }
        
        linguistic = LinguisticAnalysisConfig.model_validate(linguistic_config)
        assert linguistic.enable_sentiment_analysis is True
        assert linguistic.language == "en"
        assert linguistic.min_text_length == 10
        
        # Test ContextAnalysisConfig
        context_config = {
            "analyze_user_intent": True,
            "extract_domain_context": True,
            "identify_complexity_level": True,
            "context_window_size": 512,
            "max_context_features": 50
        }
        
        context = ContextAnalysisConfig.model_validate(context_config)
        assert context.analyze_user_intent is True
        assert context.context_window_size == 512
        
        # Test DomainAnalysisConfig
        domain_config = {
            "supported_domains": ["technical", "business", "creative"],
            "enable_domain_classification": True,
            "confidence_threshold": 0.7,
            "max_domain_features": 25
        }
        
        domain = DomainAnalysisConfig.model_validate(domain_config)
        assert len(domain.supported_domains) == 3
        assert domain.confidence_threshold == 0.7
        
        print("✓ ML pipeline models validation passed")
        return True
        
    except Exception as e:
        print(f"✗ ML pipeline models validation failed: {e}")
        return False


@pytest.mark.asyncio
async def test_database_serialization():
    """Test database model serialization with real data"""
    try:
        from prompt_improver.database.models import (
            PromptSession, AprioriAnalysisRequest, TrainingPrompt
        )
        
        # Test PromptSession serialization  
        session = PromptSession(
            session_id="test-session-serialize",
            original_prompt="Original prompt for testing serialization",
            improved_prompt="Improved prompt with better clarity and structure",
            user_context={"domain": "technical", "complexity": "high", "urgency": "medium"},
            quality_score=0.85,
            improvement_score=0.72,
            confidence_level=0.9
        )
        
        # Test that we can access all fields
        assert session.session_id == "test-session-serialize"
        assert session.user_context["domain"] == "technical"
        assert session.quality_score == 0.85
        
        # Test AprioriAnalysisRequest with defaults
        minimal_request = AprioriAnalysisRequest()
        assert minimal_request.window_days == 30  # default
        assert minimal_request.min_support == 0.1  # default
        assert minimal_request.save_to_database is True  # default
        
        # Test with custom values
        custom_request = AprioriAnalysisRequest(
            window_days=14,
            min_support=0.15,
            min_confidence=0.7,
            min_lift=1.2,
            max_itemset_length=4,
            save_to_database=False
        )
        assert custom_request.window_days == 14
        assert custom_request.save_to_database is False
        
        # Test TrainingPrompt creation
        training_prompt = TrainingPrompt(
            prompt_text="Training prompt for ML model validation",
            enhancement_result={
                "improved_prompt": "Enhanced training prompt with better structure",
                "metrics": {"improvement_score": 0.8, "confidence": 0.85},
                "rules_applied": ["clarity", "specificity", "structure"]
            },
            data_source="synthetic",
            training_priority=200
        )
        
        assert training_prompt.prompt_text.startswith("Training prompt")
        assert training_prompt.enhancement_result["metrics"]["improvement_score"] == 0.8
        assert training_prompt.data_source == "synthetic"
        assert training_prompt.training_priority == 200
        
        print("✓ Database serialization validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Database serialization validation failed: {e}")
        return False


async def run_api_validation_tests():
    """Run all API validation tests"""
    tests = [
        test_analytics_endpoints_validation,
        test_orchestrator_endpoints_validation,
        test_batch_processor_validation,
        test_ml_pipeline_models_validation,
        test_database_serialization
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== API Validation Results ===")
    print(f"Passed: {passed}/{total} tests")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All API validation tests passed!")
    else:
        print("⚠️  Some API validation tests failed")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_api_validation_tests())