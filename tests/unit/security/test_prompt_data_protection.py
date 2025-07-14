"""
Comprehensive unit tests for PromptDataProtection security framework.

Tests all security functionality including sensitive data detection, sanitization,
audit logging, compliance reporting, and safety validation.

Follows pytest best practices:
- Comprehensive fixture-based test setup
- Parametrized tests for multiple security scenarios
- Mock-based isolation for database operations
- Statistical validation with proper error handling
- Performance benchmarking for security operations
- Integration testing patterns
"""

import pytest
import asyncio
import json
import re
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.prompt_improver.service.security import PromptDataProtection


class TestPromptDataProtection:
    """Comprehensive test suite for PromptDataProtection security framework"""
    
    @pytest.fixture(scope="class")
    def data_protection(self):
        """Create a PromptDataProtection instance for testing"""
        console = MagicMock()
        return PromptDataProtection(console=console)
    
    @pytest.fixture(scope="class")
    def sample_sensitive_prompts(self):
        """Sample prompts containing various types of sensitive data"""
        return {
            "openai_key": "Please analyze this data using sk-1234567890123456789012345678901234567890",
            "github_token": "Upload this to GitHub using token ghp_1234567890123456789012345678901234567890123456",
            "email": "Send results to user@company.com and admin@example.org",
            "ssn": "Process user data for SSN 123-45-6789 and 987-65-4321",
            "credit_card": "Payment info: 1234 5678 9012 3456 and 9876-5432-1098-7654",
            "password": "Connect with password=secretpass123 and api_key=myapikey456",
            "bearer_token": "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "mixed_sensitive": "User sk-test123 with email test@example.com and SSN 111-22-3333",
            "clean_prompt": "Analyze the weather patterns and provide a summary report",
            "false_positive": "Task ID: sk-task-abc123 and reference email-format-validation"
        }
    
    @pytest.fixture(scope="class")
    def expected_redactions(self):
        """Expected redaction results for sample prompts"""
        return {
            "openai_key": {"count": 1, "types": ["openai_api_key"]},
            "github_token": {"count": 1, "types": ["github_token"]},
            "email": {"count": 2, "types": ["email_address"]},
            "ssn": {"count": 2, "types": ["ssn_pattern"]},
            "credit_card": {"count": 2, "types": ["credit_card"]},
            "password": {"count": 2, "types": ["password_field", "api_key_field"]},
            "bearer_token": {"count": 1, "types": ["bearer_token"]},
            "mixed_sensitive": {"count": 3, "types": ["openai_api_key", "email_address", "ssn_pattern"]},
            "clean_prompt": {"count": 0, "types": []},
            "false_positive": {"count": 0, "types": []}
        }

    # ==================== Sensitive Data Detection Tests ====================
    
    @pytest.mark.asyncio
    async def test_sanitize_prompt_basic_functionality(self, data_protection, sample_sensitive_prompts):
        """Test basic prompt sanitization functionality"""
        
        # Test with OpenAI API key
        prompt = sample_sensitive_prompts["openai_key"]
        session_id = "test_session_001"
        
        with patch.object(data_protection, 'audit_redaction', new_callable=AsyncMock) as mock_audit:
            sanitized, summary = await data_protection.sanitize_prompt_before_storage(prompt, session_id)
            
            # Verify sanitization occurred
            assert "[REDACTED_OPENAI_API_KEY]" in sanitized
            assert "sk-1234567890123456789012345678901234567890" not in sanitized
            
            # Verify summary structure
            assert summary["redactions_made"] == 1
            assert "openai_api_key" in summary["redaction_types"]
            assert summary["sanitized_length"] < summary["original_length"]
            
            # Verify audit was called
            mock_audit.assert_called_once_with(session_id, 1, {
                "openai_api_key": {"count": 1, "placeholder": "[REDACTED_OPENAI_API_KEY]"}
            })
    
    @pytest.mark.parametrize("prompt_key,expected", [
        ("openai_key", {"redactions": 1, "types": ["openai_api_key"]}),
        ("github_token", {"redactions": 1, "types": ["github_token"]}),
        ("email", {"redactions": 2, "types": ["email_address"]}),
        ("ssn", {"redactions": 2, "types": ["ssn_pattern"]}),
        ("credit_card", {"redactions": 2, "types": ["credit_card"]}),
        ("password", {"redactions": 2, "types": ["password_field", "api_key_field"]}),
        ("bearer_token", {"redactions": 1, "types": ["bearer_token"]}),
        ("mixed_sensitive", {"redactions": 3, "types": ["openai_api_key", "email_address", "ssn_pattern"]}),
        ("clean_prompt", {"redactions": 0, "types": []}),
    ])
    @pytest.mark.asyncio
    async def test_detection_accuracy_comprehensive(self, data_protection, sample_sensitive_prompts, prompt_key, expected):
        """Test detection accuracy across all sensitive data types"""
        
        prompt = sample_sensitive_prompts[prompt_key]
        session_id = f"test_session_{prompt_key}"
        
        with patch.object(data_protection, 'audit_redaction', new_callable=AsyncMock):
            sanitized, summary = await data_protection.sanitize_prompt_before_storage(prompt, session_id)
            
            # Verify redaction count
            assert summary["redactions_made"] == expected["redactions"], \
                f"Expected {expected['redactions']} redactions for {prompt_key}, got {summary['redactions_made']}"
            
            # Verify redaction types
            actual_types = set(summary["redaction_types"])
            expected_types = set(expected["types"])
            assert actual_types == expected_types, \
                f"Expected types {expected_types} for {prompt_key}, got {actual_types}"
    
    @pytest.mark.asyncio
    async def test_false_positive_prevention(self, data_protection, sample_sensitive_prompts):
        """Test that legitimate content is not incorrectly flagged"""
        
        prompt = sample_sensitive_prompts["false_positive"]
        session_id = "test_false_positive"
        
        with patch.object(data_protection, 'audit_redaction', new_callable=AsyncMock) as mock_audit:
            sanitized, summary = await data_protection.sanitize_prompt_before_storage(prompt, session_id)
            
            # Should not detect false positives
            assert summary["redactions_made"] == 0
            assert len(summary["redaction_types"]) == 0
            assert sanitized == prompt  # No changes should be made
            
            # Audit should not be called for clean prompts
            mock_audit.assert_not_called()

    # ==================== Audit Logging Tests ====================
    
    @pytest.mark.asyncio
    async def test_audit_logging_integration(self, data_protection):
        """Test audit logging integration with database"""
        
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None  # Simulate no existing session
        mock_session.execute.return_value = mock_result
        
        with patch('src.prompt_improver.service.security.sessionmanager.session') as mock_session_manager:
            mock_session_manager.return_value.__aenter__.return_value = mock_session
            
            # Test audit logging
            session_id = "test_audit_session"
            redaction_count = 2
            redaction_details = {
                "openai_api_key": {"count": 1, "placeholder": "[REDACTED_OPENAI_API_KEY]"},
                "email_address": {"count": 1, "placeholder": "[REDACTED_EMAIL_ADDRESS]"}
            }
            
            await data_protection.audit_redaction(session_id, redaction_count, redaction_details)
            
            # Verify database calls were made
            assert mock_session.execute.call_count >= 2  # At least check and insert/update
            
            # Verify audit data structure
            calls = mock_session.execute.call_args_list
            audit_call = None
            for call in calls:
                args, kwargs = call
                if 'INSERT INTO improvement_sessions' in str(args[0]):
                    audit_call = kwargs
                    break
            
            assert audit_call is not None, "Audit insert call not found"
            metadata_str = audit_call['metadata']
            metadata = json.loads(metadata_str)
            
            assert 'security_audit' in metadata
            security_audit = metadata['security_audit']
            assert security_audit['redactions'] == redaction_count
            assert security_audit['redaction_details'] == redaction_details
            assert security_audit['security_level'] == 'redacted'

    # ==================== Safety Validation Tests ====================
    
    @pytest.mark.asyncio
    async def test_prompt_safety_validation(self, data_protection, sample_sensitive_prompts):
        """Test prompt safety validation without modification"""
        
        # Test high-risk prompt
        high_risk_prompt = sample_sensitive_prompts["mixed_sensitive"]
        safety_report = await data_protection.validate_prompt_safety(high_risk_prompt)
        
        assert not safety_report["is_safe"]
        assert safety_report["risk_level"] in ["CRITICAL", "HIGH"]
        assert len(safety_report["issues_detected"]) == 3
        assert len(safety_report["recommendations"]) > 0
        
        # Verify issue details
        issue_types = [issue["type"] for issue in safety_report["issues_detected"]]
        assert "openai_api_key" in issue_types
        assert "email_address" in issue_types
        assert "ssn_pattern" in issue_types
        
        # Test clean prompt
        clean_prompt = sample_sensitive_prompts["clean_prompt"]
        clean_safety_report = await data_protection.validate_prompt_safety(clean_prompt)
        
        assert clean_safety_report["is_safe"]
        assert clean_safety_report["risk_level"] == "LOW"
        assert len(clean_safety_report["issues_detected"]) == 0
        assert len(clean_safety_report["recommendations"]) == 0

    @pytest.mark.parametrize("pattern_type,expected_risk", [
        ("openai_api_key", "CRITICAL"),
        ("github_token", "CRITICAL"),
        ("api_key_field", "CRITICAL"),
        ("secret_field", "CRITICAL"),
        ("bearer_token", "CRITICAL"),
        ("credit_card", "HIGH"),
        ("ssn_pattern", "HIGH"),
        ("password_field", "HIGH"),
        ("long_token", "MEDIUM"),
        ("email_address", "MEDIUM"),
    ])
    def test_risk_level_assessment(self, data_protection, pattern_type, expected_risk):
        """Test risk level assessment for different pattern types"""
        
        risk_level = data_protection._get_risk_level(pattern_type)
        assert risk_level == expected_risk, \
            f"Expected {expected_risk} risk for {pattern_type}, got {risk_level}"

    # ==================== Statistics and Reporting Tests ====================
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self, data_protection, sample_sensitive_prompts):
        """Test redaction statistics tracking"""
        
        # Reset statistics for clean test
        data_protection.reset_statistics()
        initial_stats = await data_protection.get_redaction_statistics()
        assert initial_stats["statistics"]["total_prompts_processed"] == 0
        
        # Process several prompts
        test_prompts = [
            ("session_1", sample_sensitive_prompts["openai_key"]),
            ("session_2", sample_sensitive_prompts["clean_prompt"]),
            ("session_3", sample_sensitive_prompts["mixed_sensitive"]),
        ]
        
        with patch.object(data_protection, 'audit_redaction', new_callable=AsyncMock):
            for session_id, prompt in test_prompts:
                await data_protection.sanitize_prompt_before_storage(prompt, session_id)
        
        # Verify statistics
        final_stats = await data_protection.get_redaction_statistics()
        stats = final_stats["statistics"]
        
        assert stats["total_prompts_processed"] == 3
        assert stats["prompts_with_sensitive_data"] == 2  # Only 2 had sensitive data
        assert stats["total_redactions"] == 4  # 1 + 0 + 3 redactions
        assert "openai_api_key" in stats["redactions_by_type"]
        assert "email_address" in stats["redactions_by_type"]

    @pytest.mark.asyncio
    async def test_security_audit_report_generation(self, data_protection):
        """Test security audit report generation"""
        
        # Mock database query results
        mock_session = AsyncMock()
        
        # Mock audit data query result
        mock_audit_result = MagicMock()
        mock_audit_result.fetchone.return_value = (100, 15, 25, 80, 85)  # total, with_redactions, total_redactions, recent, clean
        
        # Mock redaction types query result
        mock_types_result = MagicMock()
        mock_types_result.fetchall.return_value = [
            ("email_address", 10),
            ("openai_api_key", 8),
            ("credit_card", 5)
        ]
        
        mock_session.execute.side_effect = [mock_audit_result, mock_types_result]
        
        with patch('src.prompt_improver.service.security.sessionmanager.session') as mock_session_manager:
            mock_session_manager.return_value.__aenter__.return_value = mock_session
            
            report = await data_protection.get_security_audit_report(days=30)
            
            # Verify report structure
            assert "summary" in report
            assert "compliance" in report
            assert "redaction_patterns" in report
            assert "statistics" in report
            
            # Verify summary data
            summary = report["summary"]
            assert summary["total_sessions_audited"] == 100
            assert summary["sessions_with_redactions"] == 15
            assert summary["total_redactions_performed"] == 25
            assert summary["clean_sessions"] == 85
            
            # Verify compliance calculation
            compliance = report["compliance"]
            expected_compliance = round((85 / 100) * 100, 2)  # clean/total * 100
            assert compliance["compliance_score_percent"] == expected_compliance
            
            # Verify redaction patterns
            patterns = report["redaction_patterns"]
            assert len(patterns["types_detected"]) == 3
            assert patterns["most_common"] == "email_address"

    # ==================== Performance and Edge Case Tests ====================
    
    @pytest.mark.asyncio
    async def test_large_prompt_performance(self, data_protection):
        """Test performance with large prompts"""
        
        # Create a large prompt with mixed content
        large_prompt = ("This is a test prompt. " * 1000) + \
                      "API key: sk-1234567890123456789012345678901234567890 " + \
                      ("More content here. " * 1000) + \
                      "Email: test@example.com"
        
        session_id = "test_large_prompt"
        
        with patch.object(data_protection, 'audit_redaction', new_callable=AsyncMock):
            import time
            start_time = time.time()
            
            sanitized, summary = await data_protection.sanitize_prompt_before_storage(large_prompt, session_id)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Performance assertion - should process large prompts reasonably quickly
            assert processing_time < 1.0, f"Large prompt processing took {processing_time:.2f}s, expected < 1.0s"
            
            # Verify functionality still works
            assert summary["redactions_made"] == 2
            assert "openai_api_key" in summary["redaction_types"]
            assert "email_address" in summary["redaction_types"]

    @pytest.mark.asyncio
    async def test_empty_and_edge_case_prompts(self, data_protection):
        """Test handling of empty and edge case prompts"""
        
        edge_cases = [
            ("", "empty_prompt"),
            ("   ", "whitespace_only"),
            ("No sensitive data here", "normal_text"),
            ("sk-", "incomplete_key"),
            ("email@", "incomplete_email"),
            ("123-45-", "incomplete_ssn"),
        ]
        
        with patch.object(data_protection, 'audit_redaction', new_callable=AsyncMock) as mock_audit:
            for prompt, case_name in edge_cases:
                session_id = f"test_{case_name}"
                
                # Should not raise exceptions
                sanitized, summary = await data_protection.sanitize_prompt_before_storage(prompt, session_id)
                
                # Verify basic functionality
                assert isinstance(sanitized, str)
                assert isinstance(summary, dict)
                assert "redactions_made" in summary
                assert summary["redactions_made"] >= 0
                
                # Most edge cases should not trigger redactions
                if case_name in ["empty_prompt", "whitespace_only", "normal_text", "incomplete_key", "incomplete_email", "incomplete_ssn"]:
                    assert summary["redactions_made"] == 0

    @pytest.mark.asyncio
    async def test_audit_failure_handling(self, data_protection):
        """Test graceful handling of audit failures"""
        
        prompt = "Test with sk-1234567890123456789012345678901234567890"
        session_id = "test_audit_failure"
        
        # Mock audit failure
        with patch.object(data_protection, 'audit_redaction', new_callable=AsyncMock) as mock_audit:
            mock_audit.side_effect = Exception("Database connection failed")
            
            # Should not raise exception, should continue with sanitization
            sanitized, summary = await data_protection.sanitize_prompt_before_storage(prompt, session_id)
            
            # Verify sanitization still worked
            assert summary["redactions_made"] == 1
            assert "[REDACTED_OPENAI_API_KEY]" in sanitized
            assert "sk-1234567890123456789012345678901234567890" not in sanitized

    # ==================== Configuration and Security Tests ====================
    
    def test_sensitive_patterns_completeness(self, data_protection):
        """Test that all expected sensitive patterns are configured"""
        
        expected_pattern_types = {
            "openai_api_key", "github_token", "email_address", "ssn_pattern", 
            "credit_card", "long_token", "password_field", "api_key_field", 
            "secret_field", "bearer_token"
        }
        
        actual_pattern_types = {pattern_type for _, pattern_type in data_protection.sensitive_patterns}
        
        assert actual_pattern_types == expected_pattern_types, \
            f"Missing patterns: {expected_pattern_types - actual_pattern_types}, " \
            f"Unexpected patterns: {actual_pattern_types - expected_pattern_types}"

    def test_pattern_regex_validity(self, data_protection):
        """Test that all regex patterns are valid"""
        
        for pattern, pattern_type in data_protection.sensitive_patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"Invalid regex pattern for {pattern_type}: {pattern}. Error: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_sanitization(self, data_protection, sample_sensitive_prompts):
        """Test concurrent sanitization operations"""
        
        async def sanitize_prompt(session_id, prompt):
            with patch.object(data_protection, 'audit_redaction', new_callable=AsyncMock):
                return await data_protection.sanitize_prompt_before_storage(prompt, session_id)
        
        # Create concurrent tasks
        tasks = []
        for i, (prompt_key, prompt) in enumerate(sample_sensitive_prompts.items()):
            task = sanitize_prompt(f"concurrent_session_{i}", prompt)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all succeeded
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Task {i} failed with {result}"
            sanitized, summary = result
            assert isinstance(sanitized, str)
            assert isinstance(summary, dict)

    def test_recommendation_generation(self, data_protection):
        """Test security recommendation generation"""
        
        test_cases = [
            ([{"type": "openai_api_key", "count": 1, "risk_level": "CRITICAL"}], 
             "Remove API keys and tokens - use environment variables instead"),
            ([{"type": "credit_card", "count": 1, "risk_level": "HIGH"}], 
             "Remove credit card numbers - use masked or test data"),
            ([{"type": "ssn_pattern", "count": 1, "risk_level": "HIGH"}], 
             "Remove SSN patterns - use synthetic identifiers"),
            ([{"type": "email_address", "count": 1, "risk_level": "MEDIUM"}], 
             "Consider using example.com domains for email examples"),
        ]
        
        for detected_patterns, expected_recommendation in test_cases:
            recommendations = data_protection._generate_safety_recommendations(detected_patterns)
            
            assert expected_recommendation in recommendations, \
                f"Expected recommendation '{expected_recommendation}' not found in {recommendations}"
            
            # General recommendations should always be included
            assert "Consider using data masking techniques for sensitive information" in recommendations
            assert "Review prompt content before sharing or storing" in recommendations