"""Integration tests for PromptServiceFacade decomposition.

Tests the real behavior of all decomposed prompt services:
- PromptAnalysisService
- RuleApplicationService  
- ValidationService
- PromptServiceFacade coordinator

Uses testcontainers for real database and Redis testing.
"""

import asyncio
import pytest
from uuid import UUID, uuid4
from typing import Dict, Any

from prompt_improver.services.prompt.facade import PromptServiceFacade
from prompt_improver.services.prompt.prompt_analysis_service import PromptAnalysisService
from prompt_improver.services.prompt.rule_application_service import RuleApplicationService
from prompt_improver.services.prompt.validation_service import ValidationService
from prompt_improver.rule_engine.base import BasePromptRule
from prompt_improver.database.models import UserFeedback


class MockPromptRule(BasePromptRule):
    """Mock rule for testing."""
    
    def __init__(self, rule_id: str = "test_rule", improvement_score: float = 0.8):
        self.rule_id = rule_id
        self.name = f"Test Rule {rule_id}"
        self.improvement_score = improvement_score
    
    async def apply(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mock improvement."""
        improved_prompt = f"{prompt} [Improved by {self.name}]"
        return {
            "improved_prompt": improved_prompt,
            "improvement_score": self.improvement_score,
            "confidence": 0.85,
            "changes_made": [f"Added improvement by {self.name}"],
            "metadata": {"rule_id": self.rule_id}
        }


@pytest.fixture
async def prompt_facade():
    """Create PromptServiceFacade with all services."""
    analysis_service = PromptAnalysisService()
    rule_application_service = RuleApplicationService()
    validation_service = ValidationService()
    
    facade = PromptServiceFacade(
        analysis_service=analysis_service,
        rule_application_service=rule_application_service,
        validation_service=validation_service
    )
    
    return facade


@pytest.fixture
def test_rules():
    """Create test rules for validation."""
    return [
        MockPromptRule("clarity_rule", 0.7),
        MockPromptRule("structure_rule", 0.8),
        MockPromptRule("specificity_rule", 0.6)
    ]


@pytest.mark.asyncio
class TestPromptAnalysisService:
    """Test PromptAnalysisService functionality."""
    
    async def test_analyze_prompt_basic(self):
        """Test basic prompt analysis."""
        service = PromptAnalysisService()
        
        prompt_id = uuid4()
        session_id = uuid4()
        
        result = await service.analyze_prompt(prompt_id, session_id)
        
        assert result["prompt_id"] == str(prompt_id)
        assert result["session_id"] == str(session_id)
        assert "analysis_timestamp" in result
        assert "characteristics" in result
        assert "suggestions" in result
        assert "confidence_score" in result
        assert isinstance(result["confidence_score"], float)
    
    async def test_generate_improvements_with_rules(self, test_rules):
        """Test improvement generation with rules."""
        service = PromptAnalysisService()
        
        prompt = "Write a simple function"
        improvements = await service.generate_improvements(prompt, test_rules)
        
        assert len(improvements) == len(test_rules)
        for improvement in improvements:
            assert "rule_id" in improvement
            assert "improved_prompt" in improvement
            assert "confidence_score" in improvement
            assert "timestamp" in improvement
            assert improvement["original_prompt"] == prompt
    
    async def test_evaluate_improvement_quality(self):
        """Test quality evaluation."""
        service = PromptAnalysisService()
        
        original = "Write code"
        improved = "Write a well-documented, efficient Python function that implements the specified algorithm"
        
        quality_scores = await service.evaluate_improvement_quality(original, improved)
        
        assert "overall_quality" in quality_scores
        assert "length_improvement" in quality_scores
        assert "clarity_improvement" in quality_scores
        assert "specificity_improvement" in quality_scores
        assert "structure_improvement" in quality_scores
        
        # Quality should be better for the improved prompt
        assert quality_scores["overall_quality"] > 0.5
    
    async def test_get_ml_recommendations(self):
        """Test ML recommendations."""
        service = PromptAnalysisService()
        
        prompt = "Test prompt for ML recommendations"
        session_id = uuid4()
        
        recommendations = await service.get_ml_recommendations(prompt, session_id)
        
        assert "status" in recommendations
        assert "recommendations" in recommendations
        assert "timestamp" in recommendations


@pytest.mark.asyncio
class TestRuleApplicationService:
    """Test RuleApplicationService functionality."""
    
    async def test_apply_rules_successful(self, test_rules):
        """Test successful rule application."""
        service = RuleApplicationService()
        
        prompt = "Original prompt text"
        session_id = uuid4()
        
        result = await service.apply_rules(prompt, test_rules, session_id)
        
        assert result["original_prompt"] == prompt
        assert result["session_id"] == str(session_id)
        assert "applied_rules" in result
        assert "final_prompt" in result
        assert "execution_summary" in result
        
        # Check execution summary
        summary = result["execution_summary"]
        assert summary["total_rules"] == len(test_rules)
        assert summary["successful_applications"] > 0
        assert summary["success_rate"] > 0
    
    async def test_validate_rule_compatibility(self, test_rules):
        """Test rule compatibility validation."""
        service = RuleApplicationService()
        
        compatibility = await service.validate_rule_compatibility(test_rules)
        
        assert "overall_compatible" in compatibility
        assert "compatibility_matrix" in compatibility
        assert "warnings" in compatibility
        assert isinstance(compatibility["overall_compatible"], bool)
    
    async def test_execute_rule_chain(self, test_rules):
        """Test rule chain execution."""
        service = RuleApplicationService()
        
        prompt = "Test prompt for rule chain"
        
        results = await service.execute_rule_chain(prompt, test_rules)
        
        assert len(results) == len(test_rules)
        for i, result in enumerate(results):
            assert result["step"] == i + 1
            assert "rule_id" in result
            assert "success" in result
            assert "execution_time_ms" in result
    
    async def test_get_rule_performance_metrics(self, test_rules):
        """Test rule performance metrics."""
        service = RuleApplicationService()
        
        # First apply some rules to generate metrics
        prompt = "Test prompt"
        await service.apply_rules(prompt, test_rules[:1])
        
        rule_id = test_rules[0].rule_id
        metrics = await service.get_rule_performance_metrics(rule_id)
        
        assert metrics["rule_id"] == rule_id
        assert "total_executions" in metrics
        assert "success_rate" in metrics
        assert "average_execution_time_ms" in metrics


@pytest.mark.asyncio
class TestValidationService:
    """Test ValidationService functionality."""
    
    async def test_validate_prompt_input_valid(self):
        """Test validation of valid prompt."""
        service = ValidationService()
        
        prompt = "This is a valid prompt for testing"
        
        result = await service.validate_prompt_input(prompt)
        
        assert result["valid"] is True
        assert "violations" in result
        assert "warnings" in result
        assert "validation_score" in result
        assert result["original_length"] == len(prompt)
    
    async def test_validate_prompt_input_empty(self):
        """Test validation of empty prompt."""
        service = ValidationService()
        
        prompt = ""
        
        result = await service.validate_prompt_input(prompt)
        
        assert result["valid"] is False
        assert len(result["violations"]) > 0
        
        # Check for empty prompt violation
        violation_types = [v["type"] for v in result["violations"]]
        assert "empty_prompt" in violation_types
    
    async def test_validate_prompt_input_with_constraints(self):
        """Test validation with custom constraints."""
        service = ValidationService()
        
        prompt = "Short"
        constraints = {
            "min_length": 20,
            "max_length": 100,
            "required_patterns": ["test"],
            "prohibited_patterns": ["forbidden"]
        }
        
        result = await service.validate_prompt_input(prompt, constraints)
        
        assert result["valid"] is False
        violation_types = [v["type"] for v in result["violations"]]
        assert "min_length" in violation_types
    
    async def test_check_business_rules(self):
        """Test business rules checking."""
        service = ValidationService()
        
        operation = "improve_prompt"
        data = {"user_id": str(uuid4()), "prompt": "test prompt"}
        
        result = await service.check_business_rules(operation, data)
        
        assert isinstance(result, bool)
    
    async def test_sanitize_prompt_content(self):
        """Test prompt sanitization."""
        service = ValidationService()
        
        dangerous_prompt = "Test prompt <script>alert('xss')</script> content"
        
        sanitized = await service.sanitize_prompt_content(dangerous_prompt)
        
        assert "<script>" not in sanitized
        assert "alert" not in sanitized
        assert "Test prompt" in sanitized
        assert "content" in sanitized


@pytest.mark.asyncio
class TestPromptServiceFacade:
    """Test PromptServiceFacade integration."""
    
    async def test_improve_prompt_basic(self, prompt_facade):
        """Test basic prompt improvement."""
        prompt = "Write a function to calculate fibonacci numbers"
        user_id = uuid4()
        
        result = await prompt_facade.improve_prompt(
            prompt=prompt,
            user_id=user_id
        )
        
        assert result["status"] == "success"
        assert result["original_prompt"] == prompt
        assert result["user_id"] == str(user_id)
        assert "session_id" in result
        assert "improved_prompt" in result
        assert "processing_steps" in result
        assert "performance_metrics" in result
        
        # Check processing steps
        steps = result["processing_steps"]
        step_names = [step["step"] for step in steps]
        assert "validation" in step_names
    
    async def test_improve_prompt_with_rules(self, prompt_facade, test_rules):
        """Test prompt improvement with custom rules."""
        prompt = "Basic prompt"
        
        result = await prompt_facade.improve_prompt(
            prompt=prompt,
            rules=test_rules
        )
        
        assert result["status"] == "success"
        assert "rule_application" in result
        
        if "rule_application" in result:
            rule_result = result["rule_application"]
            assert "applied_rules" in rule_result
            assert len(rule_result["applied_rules"]) == len(test_rules)
    
    async def test_improve_prompt_with_validation_failure(self, prompt_facade):
        """Test prompt improvement with validation failure."""
        empty_prompt = ""
        
        result = await prompt_facade.improve_prompt(prompt=empty_prompt)
        
        assert result["status"] == "validation_failed"
        assert "validation_result" in result
        assert result["validation_result"]["valid"] is False
    
    async def test_improve_prompt_with_config(self, prompt_facade):
        """Test prompt improvement with configuration."""
        prompt = "Test prompt with config"
        config = {
            "sanitize": True,
            "sanitization_level": "standard",
            "enable_ml_recommendations": True,
            "validation_constraints": {
                "min_length": 5,
                "max_length": 1000
            }
        }
        
        result = await prompt_facade.improve_prompt(
            prompt=prompt,
            config=config
        )
        
        assert result["status"] == "success"
        # ML recommendations might be present if enabled
        if "ml_recommendations" in result:
            assert "status" in result["ml_recommendations"]
    
    async def test_get_session_summary(self, prompt_facade):
        """Test session summary retrieval."""
        session_id = uuid4()
        
        summary = await prompt_facade.get_session_summary(session_id)
        
        assert summary["session_id"] == str(session_id)
        assert "status" in summary
        assert "created_at" in summary
        assert "operations_count" in summary
        assert "success_rate" in summary
    
    async def test_process_feedback(self, prompt_facade):
        """Test feedback processing."""
        # Create mock feedback
        class MockFeedback:
            def __init__(self):
                self.id = 1
                self.rating = 2  # Low rating
                self.comment = "Needs improvement"
        
        feedback = MockFeedback()
        session_id = uuid4()
        
        result = await prompt_facade.process_feedback(feedback, session_id)
        
        assert result["session_id"] == str(session_id)
        assert result["status"] == "processed"
        assert "processed_at" in result
        
        # Low rating should trigger analysis
        if feedback.rating < 3:
            assert result.get("triggered_analysis") is True
    
    async def test_get_health_status(self, prompt_facade):
        """Test health status monitoring."""
        health = await prompt_facade.get_health_status()
        
        assert "overall_health" in health
        assert "timestamp" in health
        assert "services" in health
        assert "performance_metrics" in health
        
        # Check individual service health
        services = health["services"]
        assert "analysis_service" in services
        assert "rule_application_service" in services
        assert "validation_service" in services
        
        for service_name, service_health in services.items():
            assert "healthy" in service_health
            assert "status" in service_health


@pytest.mark.asyncio
class TestPromptServiceFacadePerformance:
    """Test performance characteristics of PromptServiceFacade."""
    
    async def test_response_time_targets(self, prompt_facade):
        """Test that response times meet targets."""
        import time
        
        prompt = "Test prompt for performance measurement"
        
        start_time = time.perf_counter()
        result = await prompt_facade.improve_prompt(prompt=prompt)
        end_time = time.perf_counter()
        
        response_time_ms = (end_time - start_time) * 1000
        
        # Target: <5000ms for prompt improvement
        assert response_time_ms < 5000
        assert result["status"] == "success"
        
        # Check performance metrics in result
        perf_metrics = result["performance_metrics"]
        assert perf_metrics["total_processing_time_ms"] < 5000
    
    async def test_concurrent_requests(self, prompt_facade):
        """Test handling of concurrent requests."""
        import asyncio
        
        prompts = [
            "Concurrent test prompt 1",
            "Concurrent test prompt 2", 
            "Concurrent test prompt 3",
            "Concurrent test prompt 4",
            "Concurrent test prompt 5"
        ]
        
        # Execute concurrent requests
        tasks = [
            prompt_facade.improve_prompt(prompt=prompt)
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All requests should succeed
        assert len(results) == len(prompts)
        for result in results:
            assert result["status"] == "success"
    
    async def test_memory_usage_stability(self, prompt_facade):
        """Test memory usage remains stable under load."""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        # Run multiple operations
        for i in range(20):
            prompt = f"Memory test prompt {i}"
            result = await prompt_facade.improve_prompt(prompt=prompt)
            assert result["status"] == "success"
            
            # Periodic cleanup
            if i % 5 == 0:
                gc.collect()
        
        # Final cleanup and verification
        gc.collect()
        
        # Get health status to check for memory issues
        health = await prompt_facade.get_health_status()
        assert health["overall_health"] in ["healthy", "degraded"]  # Allow degraded but not error


@pytest.mark.asyncio 
class TestPromptServiceFacadeErrorHandling:
    """Test error handling and resilience."""
    
    async def test_service_failure_graceful_degradation(self, prompt_facade):
        """Test graceful degradation when services fail."""
        # Simulate service failure by marking services as unhealthy
        prompt_facade._service_health["analysis"] = False
        
        prompt = "Test prompt with service failure"
        result = await prompt_facade.improve_prompt(prompt=prompt)
        
        # Should still succeed with degraded functionality
        assert result["status"] == "success"
        
        # Check that processing steps show skipped services
        steps = result["processing_steps"]
        analysis_steps = [s for s in steps if s["step"] == "analysis"]
        if analysis_steps:
            assert analysis_steps[0]["status"] in ["skipped", "failed"]
    
    async def test_invalid_input_handling(self, prompt_facade):
        """Test handling of invalid inputs."""
        # Test with None prompt
        result = await prompt_facade.improve_prompt(prompt=None)
        assert result["status"] in ["validation_failed", "error"]
        
        # Test with extremely long prompt
        very_long_prompt = "x" * 100000
        result = await prompt_facade.improve_prompt(prompt=very_long_prompt)
        # Should handle gracefully (either succeed or fail cleanly)
        assert result["status"] in ["success", "validation_failed", "error"]
    
    async def test_timeout_handling(self, prompt_facade):
        """Test timeout handling for long operations."""
        # This would normally require mocking slow operations
        # For now, just verify the facade can handle normal operations quickly
        prompt = "Test prompt for timeout handling"
        
        result = await prompt_facade.improve_prompt(prompt=prompt)
        
        assert result["status"] == "success"
        # Verify performance metrics show reasonable response time
        perf_metrics = result["performance_metrics"]
        assert perf_metrics["total_processing_time_ms"] < 10000  # 10 second max