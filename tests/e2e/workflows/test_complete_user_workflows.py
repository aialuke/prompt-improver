"""
Complete user workflow end-to-end tests.

Tests full user scenarios from start to finish with real system deployment.
Validates complete business workflows and user experience.
"""

import pytest
import time
import asyncio
from uuid import uuid4

@pytest.mark.e2e
@pytest.mark.workflow
class TestCompleteUserWorkflows:
    """End-to-end tests for complete user workflows."""
    
    def test_new_user_onboarding_workflow(self, e2e_client, e2e_performance_monitor, scenario_validator):
        """Test complete new user onboarding workflow."""
        e2e_performance_monitor.start_measurement("onboarding_workflow")
        
        # Step 1: User creates first session
        user_id = f"new_user_{uuid4()}"
        session_id = e2e_client.create_session(user_id)
        assert session_id is not None
        
        # Step 2: User makes first prompt improvement
        result = e2e_client.improve_prompt(
            session_id=session_id,
            prompt="Help me debug my code",
            context={"domain": "software_development", "experience": "beginner"}
        )
        
        # Validate improvement quality
        scenario_validator.validate_improvement_quality(
            "Help me debug my code",
            result["improved_prompt"],
            result["confidence"]
        )
        
        # Step 3: User views their history
        history = e2e_client.get_session_history(session_id)
        assert len(history["history"]) == 1
        assert history["history"][0]["original_prompt"] == "Help me debug my code"
        
        # Step 4: User makes several more improvements
        prompts = [
            "Fix this error",
            "Make code faster",
            "Add tests"
        ]
        
        for prompt in prompts:
            result = e2e_client.improve_prompt(session_id, prompt)
            scenario_validator.validate_improvement_quality(
                prompt, result["improved_prompt"], result["confidence"]
            )
        
        # Step 5: User views analytics
        analytics = e2e_client.get_analytics(session_id)
        assert analytics["total_improvements"] == 4
        assert 0.5 <= analytics["average_confidence"] <= 1.0
        assert len(analytics["improvement_trends"]) > 0
        
        e2e_performance_monitor.end_measurement("onboarding_workflow")
        e2e_performance_monitor.assert_performance("onboarding_workflow", 10.0)  # Should complete in <10s
    
    def test_software_developer_daily_workflow(self, e2e_client, e2e_test_scenarios, scenario_validator):
        """Test typical software developer daily workflow."""
        # Arrange
        scenario = e2e_test_scenarios["software_development"]
        user_id = "developer_user"
        session_id = e2e_client.create_session(user_id)
        
        # Simulate daily workflow
        daily_prompts = [
            ("Debug authentication issue", "Fix login bug that's causing 401 errors"),
            ("Optimize database query", "Make the user search query run faster"),
            ("Write API documentation", "Document the new REST endpoints for user management"),
            ("Add error handling", "Add proper error handling to the payment processing function"),
            ("Refactor legacy code", "Clean up the old user authentication system")
        ]
        
        improvements = []
        for original, expected_theme in daily_prompts:
            # Act
            result = e2e_client.improve_prompt(
                session_id=session_id,
                prompt=original,
                context=scenario["context"]
            )
            
            # Assert
            scenario_validator.validate_improvement_quality(original, result["improved_prompt"], result["confidence"])
            
            # Verify improvement contains expected themes
            improved = result["improved_prompt"].lower()
            assert any(theme in improved for theme in expected_theme.lower().split()), \
                f"Improvement '{result['improved_prompt']}' doesn't contain expected themes from '{expected_theme}'"
            
            improvements.append(result)
        
        # Verify session analytics show learning patterns
        analytics = e2e_client.get_analytics(session_id)
        assert analytics["total_improvements"] == 5
        assert analytics["domain_focus"]["software_development"] > 0.8  # Focused on software development
        assert "clarity" in analytics["top_rules_applied"]
        assert "specificity" in analytics["top_rules_applied"]
    
    @pytest.mark.asyncio
    async def test_mcp_integration_workflow(self, e2e_client, scenario_validator):
        """Test MCP protocol integration workflow."""
        # Step 1: Connect via MCP and improve prompt
        result = await e2e_client.mcp_improve_prompt(
            prompt="Make this API faster",
            context={"domain": "performance", "urgency": "high"}
        )
        
        # Parse MCP response
        import json
        content = result["content"][0]["text"]
        mcp_result = json.loads(content)
        
        # Validate MCP improvement
        scenario_validator.validate_improvement_quality(
            "Make this API faster",
            mcp_result["improved_prompt"],
            mcp_result["confidence"]
        )
        
        # Step 2: Verify session was created via MCP
        session_id = mcp_result.get("session_id")
        if session_id:
            history = e2e_client.get_session_history(session_id)
            assert len(history["history"]) == 1
    
    def test_batch_processing_workflow(self, e2e_client, e2e_performance_monitor, scenario_validator):
        """Test batch processing workflow for power users."""
        e2e_performance_monitor.start_measurement("batch_processing")
        
        # Arrange
        user_id = "power_user"
        session_id = e2e_client.create_session(user_id)
        
        # Large batch of prompts to process
        batch_prompts = [
            "Fix bug",
            "Add feature",
            "Update docs",
            "Write tests",
            "Deploy changes",
            "Monitor performance", 
            "Review code",
            "Optimize queries",
            "Add logging",
            "Handle errors"
        ]
        
        # Act
        start_time = time.time()
        results = []
        
        # Process batch (simulating rapid user input)
        for prompt in batch_prompts:
            result = e2e_client.improve_prompt(session_id, prompt)
            results.append(result)
        
        batch_duration = time.time() - start_time
        
        # Assert
        assert len(results) == 10
        for i, result in enumerate(results):
            scenario_validator.validate_improvement_quality(
                batch_prompts[i], result["improved_prompt"], result["confidence"]
            )
        
        # Performance requirements for batch processing
        assert batch_duration < 15.0, f"Batch processing took {batch_duration:.2f}s (should be <15s)"
        
        # Verify all results were persisted
        history = e2e_client.get_session_history(session_id)
        assert len(history["history"]) == 10
        
        e2e_performance_monitor.end_measurement("batch_processing")
        scenario_validator.validate_session_consistency(history)
    
    def test_cross_domain_learning_workflow(self, e2e_client, e2e_test_scenarios):
        """Test user working across multiple domains."""
        # Arrange
        user_id = "cross_domain_user"
        session_id = e2e_client.create_session(user_id)
        
        # Work across different domains in sequence
        domain_workflows = [
            ("software_development", "Debug authentication error in Express.js app"),
            ("data_science", "Analyze customer churn patterns in e-commerce data"),
            ("general", "Explain machine learning to non-technical stakeholders"),
            ("software_development", "Implement rate limiting in REST API"),
            ("data_science", "Create predictive model for sales forecasting")
        ]
        
        domain_results = {}
        for domain, prompt in domain_workflows:
            context = e2e_test_scenarios[domain]["context"]
            result = e2e_client.improve_prompt(session_id, prompt, context)
            
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(result)
        
        # Verify domain-specific improvements
        analytics = e2e_client.get_analytics(session_id)
        assert len(analytics["domains_used"]) == 3  # Used 3 different domains
        assert analytics["domain_transitions"] > 0   # Switched between domains
        
        # Verify learning across domains
        for domain, results in domain_results.items():
            # Later results in same domain should have higher confidence
            if len(results) > 1:
                first_confidence = results[0]["confidence"]
                last_confidence = results[-1]["confidence"]
                # Allow for some variance, but general trend should be improvement
                assert last_confidence >= first_confidence - 0.1
    
    @pytest.mark.performance
    def test_high_load_user_workflow(self, e2e_client, load_test_config, e2e_performance_monitor):
        """Test user workflow under high load conditions."""
        import threading
        import concurrent.futures
        
        # Arrange
        concurrent_users = load_test_config["concurrent_users"]
        test_duration = load_test_config["test_duration_seconds"]
        success_threshold = load_test_config["success_rate_threshold"]
        
        results = []
        errors = []
        response_times = []
        
        def simulate_user_session(user_index):
            """Simulate single user session under load."""
            try:
                user_id = f"load_test_user_{user_index}"
                session_id = e2e_client.create_session(user_id)
                
                # Simulate user behavior over test duration
                start_time = time.time()
                while (time.time() - start_time) < test_duration:
                    request_start = time.time()
                    
                    result = e2e_client.improve_prompt(
                        session_id=session_id,
                        prompt=f"Test prompt from user {user_index}",
                        context={"domain": "testing", "load_test": True}
                    )
                    
                    request_duration = (time.time() - request_start) * 1000
                    response_times.append(request_duration)
                    results.append(result)
                    
                    # Brief pause between requests
                    time.sleep(0.5)
                    
            except Exception as e:
                errors.append(f"User {user_index}: {str(e)}")
        
        # Act
        e2e_performance_monitor.start_measurement("high_load_test")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(simulate_user_session, i) 
                for i in range(concurrent_users)
            ]
            concurrent.futures.wait(futures)
        
        e2e_performance_monitor.end_measurement("high_load_test")
        
        # Assert
        total_requests = len(results) + len(errors)
        success_rate = len(results) / total_requests if total_requests > 0 else 0
        
        assert success_rate >= success_threshold, \
            f"Success rate {success_rate:.2%} below threshold {success_threshold:.2%}"
        
        # Performance assertions
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
            p99_response_time = sorted(response_times)[int(len(response_times) * 0.99)]
            
            assert p95_response_time <= load_test_config["response_time_p95_ms"], \
                f"P95 response time {p95_response_time:.2f}ms exceeds limit"
            assert p99_response_time <= load_test_config["response_time_p99_ms"], \
                f"P99 response time {p99_response_time:.2f}ms exceeds limit"
    
    def test_session_recovery_workflow(self, e2e_client, scenario_validator):
        """Test session recovery after interruption."""
        # Step 1: Create session and make improvements
        user_id = "recovery_test_user"
        session_id = e2e_client.create_session(user_id)
        
        initial_prompts = ["First prompt", "Second prompt"]
        for prompt in initial_prompts:
            e2e_client.improve_prompt(session_id, prompt)
        
        # Step 2: Simulate session interruption (get history to verify state)
        history_before = e2e_client.get_session_history(session_id)
        assert len(history_before["history"]) == 2
        
        # Step 3: Resume session with same session_id
        recovery_prompt = "Resumed after interruption"
        result = e2e_client.improve_prompt(session_id, recovery_prompt)
        
        scenario_validator.validate_improvement_quality(
            recovery_prompt, result["improved_prompt"], result["confidence"]
        )
        
        # Step 4: Verify session continuity
        history_after = e2e_client.get_session_history(session_id)
        assert len(history_after["history"]) == 3
        
        # Verify chronological order is maintained
        scenario_validator.validate_session_consistency(history_after)
        
        # Verify all prompts are present
        original_prompts = [item["original_prompt"] for item in history_after["history"]]
        assert "First prompt" in original_prompts
        assert "Second prompt" in original_prompts
        assert "Resumed after interruption" in original_prompts