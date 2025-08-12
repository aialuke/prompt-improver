"""
Integration tests for PromptImprovementService.

Tests service interactions with real databases and internal services.
External APIs are mocked but internal components use real implementations.
"""

import pytest
import time
from uuid import uuid4

@pytest.mark.integration
class TestPromptImprovementServiceIntegration:
    """Integration tests for PromptImprovementService with real dependencies."""
    
    @pytest.mark.asyncio
    async def test_improve_prompt_with_database(self, prompt_improvement_service, integration_test_data, setup_test_database_schema):
        """Test prompt improvement with real database operations."""
        # Arrange
        session_id = integration_test_data["session_id"]
        prompt = "Fix this bug"
        
        # Act
        start_time = time.time()
        result = await prompt_improvement_service.improve_prompt(prompt, session_id)
        duration = (time.time() - start_time) * 1000
        
        # Assert
        assert "improved_prompt" in result
        assert result["improved_prompt"] != prompt
        assert len(result["improved_prompt"]) > len(prompt)
        assert result["confidence"] > 0
        assert result["processing_time_ms"] > 0
        assert duration < 1000, f"Integration test took {duration:.2f}ms (should be <1000ms)"
    
    @pytest.mark.asyncio
    async def test_session_persistence(self, prompt_improvement_service, test_db_session, integration_test_data):
        """Test session data persistence in database."""
        # Arrange
        session_id = integration_test_data["session_id"]
        user_id = integration_test_data["user_id"]
        context = integration_test_data["context"]
        
        # Act - Create session
        await prompt_improvement_service.create_session(session_id, user_id, context)
        
        # Improve multiple prompts
        prompts = ["Fix bug", "Make better", "Add tests"]
        results = []
        for prompt in prompts:
            result = await prompt_improvement_service.improve_prompt(prompt, session_id)
            results.append(result)
        
        # Assert - Verify session history
        history = await prompt_improvement_service.get_session_history(session_id)
        assert len(history) == 3
        assert all(item["session_id"] == session_id for item in history)
        assert all(item["confidence"] > 0 for item in history)
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, prompt_improvement_service, test_redis_client, integration_test_data):
        """Test cache integration with real Redis."""
        # Arrange
        session_id = integration_test_data["session_id"]
        prompt = "Optimize this function"
        
        # Act - First call (should cache result)
        result1 = await prompt_improvement_service.improve_prompt(prompt, session_id)
        
        # Second call (should use cache)
        start_time = time.time()
        result2 = await prompt_improvement_service.improve_prompt(prompt, session_id)
        cache_duration = (time.time() - start_time) * 1000
        
        # Assert
        assert result1["improved_prompt"] == result2["improved_prompt"]
        assert result1["confidence"] == result2["confidence"]
        assert cache_duration < 50, f"Cached response took {cache_duration:.2f}ms (should be <50ms)"
        
        # Verify cache key exists in Redis
        cache_key = f"prompt_improvement:{session_id}:{abs(hash(prompt))}"
        cached_data = await test_redis_client.get(cache_key)
        assert cached_data is not None
    
    @pytest.mark.asyncio
    async def test_rule_engine_integration(self, prompt_improvement_service, integration_test_data):
        """Test integration with rule engine components."""
        # Arrange
        session_id = integration_test_data["session_id"]
        test_cases = [
            ("Fix bug", ["clarity", "specificity"]),
            ("make better", ["clarity", "specificity", "structure"]),
            ("Help", ["clarity", "specificity", "context"])
        ]
        
        # Act & Assert
        for prompt, expected_rule_types in test_cases:
            result = await prompt_improvement_service.improve_prompt(prompt, session_id)
            
            assert "rules_applied" in result
            assert len(result["rules_applied"]) > 0
            
            # Verify expected rule types were applied
            applied_rules = result["rules_applied"]
            for expected_rule in expected_rule_types:
                assert any(expected_rule in rule for rule in applied_rules), \
                    f"Expected rule type '{expected_rule}' not found in {applied_rules}"
    
    @pytest.mark.asyncio
    async def test_analytics_integration(self, prompt_improvement_service, integration_test_data):
        """Test analytics data collection and retrieval."""
        # Arrange
        session_id = integration_test_data["session_id"]
        prompts = ["Debug issue", "Refactor code", "Write documentation"]
        
        # Act - Generate analytics data
        for prompt in prompts:
            await prompt_improvement_service.improve_prompt(prompt, session_id)
        
        # Get analytics
        analytics = await prompt_improvement_service.get_session_analytics(session_id)
        
        # Assert
        assert "total_improvements" in analytics
        assert "average_confidence" in analytics
        assert "processing_time_stats" in analytics
        assert "rules_usage" in analytics
        
        assert analytics["total_improvements"] == 3
        assert 0 < analytics["average_confidence"] <= 1.0
        assert analytics["processing_time_stats"]["count"] == 3
        assert len(analytics["rules_usage"]) > 0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, prompt_improvement_service, integration_test_data):
        """Test batch prompt processing with real services."""
        # Arrange
        session_id = integration_test_data["session_id"]
        prompts = [
            "Fix authentication bug",
            "Optimize database query",
            "Add input validation",
            "Refactor API endpoint",
            "Write unit tests"
        ]
        
        # Act
        start_time = time.time()
        results = await prompt_improvement_service.batch_improve_prompts(prompts, session_id)
        duration = (time.time() - start_time) * 1000
        
        # Assert
        assert len(results) == 5
        assert all("improved_prompt" in result for result in results)
        assert all(result["confidence"] > 0 for result in results)
        assert all(len(result["improved_prompt"]) > len(original) 
                  for result, original in zip(results, prompts))
        
        # Performance check for batch processing
        assert duration < 2000, f"Batch processing took {duration:.2f}ms (should be <2000ms)"
        
        # Verify all results were persisted
        history = await prompt_improvement_service.get_session_history(session_id)
        assert len(history) == 5
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, prompt_improvement_service, test_db_session, integration_test_data):
        """Test error recovery and transaction rollback."""
        # Arrange
        session_id = integration_test_data["session_id"]
        
        # Act & Assert - Test with invalid input
        with pytest.raises(ValueError):
            await prompt_improvement_service.improve_prompt("", session_id)
        
        # Verify database state is clean after error
        history = await prompt_improvement_service.get_session_history(session_id)
        assert len(history) == 0
        
        # Verify service still works after error
        result = await prompt_improvement_service.improve_prompt("Valid prompt", session_id)
        assert result["improved_prompt"] != "Valid prompt"
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, prompt_improvement_service, integration_test_data):
        """Test handling multiple concurrent sessions."""
        import asyncio
        
        # Arrange
        session_ids = [str(uuid4()) for _ in range(5)]
        prompt = "Process this concurrently"
        
        # Act - Process concurrent sessions
        tasks = []
        for session_id in session_ids:
            task = prompt_improvement_service.improve_prompt(prompt, session_id)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        duration = (time.time() - start_time) * 1000
        
        # Assert
        assert len(results) == 5
        assert all("improved_prompt" in result for result in results)
        assert duration < 3000, f"Concurrent processing took {duration:.2f}ms (should be <3000ms)"
        
        # Verify each session has its own history
        for session_id in session_ids:
            history = await prompt_improvement_service.get_session_history(session_id)
            assert len(history) == 1
            assert history[0]["session_id"] == session_id
    
    @pytest.mark.asyncio
    async def test_external_service_mocking(self, prompt_improvement_service, mock_external_services, integration_test_data):
        """Test that external services are properly mocked."""
        # Arrange
        session_id = integration_test_data["session_id"]
        prompt = "Test external service mocking"
        
        # Act
        result = await prompt_improvement_service.improve_prompt(prompt, session_id)
        
        # Assert
        assert "improved_prompt" in result
        
        # Verify external services were called (mocked)
        mock_external_services["http_post"].assert_not_called()  # Should not make real HTTP calls
        mock_external_services["openai"].assert_not_called()     # Should not call real OpenAI API
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, prompt_improvement_service, integration_test_data):
        """Test performance monitoring and metrics collection."""
        # Arrange
        session_id = integration_test_data["session_id"]
        prompt = "Monitor this performance"
        
        # Act
        result = await prompt_improvement_service.improve_prompt(prompt, session_id)
        
        # Assert performance metrics are captured
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] > 0
        assert result["processing_time_ms"] < 500  # Should be under 500ms
        
        # Verify monitoring data
        metrics = await prompt_improvement_service.get_performance_metrics(session_id)
        assert "response_times" in metrics
        assert "cache_hit_rate" in metrics
        assert "error_rate" in metrics