"""
Integration tests for PromptDataProtection component with ML Pipeline Orchestrator.
Tests real behavior without mocks to ensure successful integration.
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

from src.prompt_improver.core.services.security import PromptDataProtection
from src.prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
from src.prompt_improver.ml.orchestration.core.component_registry import ComponentTier


class TestPromptDataProtectionIntegration:
    """Integration tests for PromptDataProtection with real behavior testing."""

    @pytest.fixture
    async def orchestrator(self):
        """Create and initialize ML Pipeline Orchestrator."""
        orchestrator = MLPipelineOrchestrator()
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.fixture
    async def prompt_data_protection(self):
        """Create PromptDataProtection component."""
        component = PromptDataProtection(enable_differential_privacy=True)
        await component.initialize()
        yield component
        await component.shutdown()

    @pytest.mark.asyncio
    async def test_component_initialization(self, prompt_data_protection):
        """Test component initializes successfully."""
        # Test initialization
        assert prompt_data_protection is not None
        
        # Test health check
        health = await prompt_data_protection.health_check()
        assert health["status"] == "healthy"
        assert health["component"] == "PromptDataProtection"
        assert health["version"] == "2025.1"
        assert health["gdpr_enabled"] is True
        assert health["differential_privacy_enabled"] is True

    @pytest.mark.asyncio
    async def test_orchestrator_component_loading(self, orchestrator):
        """Test component loads successfully through orchestrator."""
        # Load PromptDataProtection component
        loaded_component = await orchestrator.component_loader.load_component(
            "prompt_data_protection", ComponentTier.TIER_6_SECURITY
        )
        
        assert loaded_component is not None
        assert loaded_component.name == "prompt_data_protection"
        assert loaded_component.component_class.__name__ == "PromptDataProtection"
        
        # Initialize component
        success = await orchestrator.component_loader.initialize_component(
            "prompt_data_protection"
        )
        assert success is True

    @pytest.mark.asyncio
    async def test_sensitive_data_detection_real_behavior(self, prompt_data_protection):
        """Test real sensitive data detection without mocks."""
        test_cases = [
            {
                "prompt": "My API key is sk-1234567890abcdef1234567890abcdef12345678",
                "expected_redactions": 1,
                "expected_types": ["openai_api_key"]
            },
            {
                "prompt": "Contact me at john.doe@example.com for more info",
                "expected_redactions": 1,
                "expected_types": ["email_address"]
            },
            {
                "prompt": "My SSN is 123-45-6789 and credit card is 1234 5678 9012 3456",
                "expected_redactions": 2,
                "expected_types": ["ssn_pattern", "credit_card"]
            },
            {
                "prompt": "This is a clean prompt with no sensitive data",
                "expected_redactions": 0,
                "expected_types": []
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            session_id = f"test_session_{i}_{uuid.uuid4()}"
            
            sanitized, summary = await prompt_data_protection.sanitize_prompt_before_storage(
                test_case["prompt"], session_id, user_consent=True
            )
            
            # Verify redaction count
            assert summary["redactions_made"] == test_case["expected_redactions"]
            
            # Verify redaction types
            assert set(summary["redaction_types"]) == set(test_case["expected_types"])
            
            # Verify GDPR compliance
            assert summary["gdpr_compliant"] is True
            
            # Verify performance
            assert summary["processing_time_ms"] < 1000  # Should be under 1 second
            
            # Verify compliance score
            if test_case["expected_redactions"] == 0:
                assert summary["compliance_score"] == 100.0
            else:
                assert 0 <= summary["compliance_score"] <= 100

    @pytest.mark.asyncio
    async def test_orchestrator_component_invocation(self, orchestrator):
        """Test component invocation through orchestrator."""
        # Load and initialize component
        await orchestrator.component_loader.load_component(
            "prompt_data_protection", ComponentTier.TIER_6_SECURITY
        )
        await orchestrator.component_loader.initialize_component("prompt_data_protection")
        
        # Test health check through orchestrator
        health_result = await orchestrator.invoke_component(
            "prompt_data_protection", "health_check"
        )
        
        assert health_result["status"] == "healthy"
        assert health_result["component"] == "PromptDataProtection"

    @pytest.mark.asyncio
    async def test_batch_processing_real_behavior(self, prompt_data_protection):
        """Test batch processing with real data."""
        prompts = [
            "Clean prompt 1",
            "API key: sk-1234567890abcdef1234567890abcdef12345678",
            "Email: test@example.com",
            "Another clean prompt"
        ]
        session_ids = [f"batch_session_{i}_{uuid.uuid4()}" for i in range(len(prompts))]
        
        results = await prompt_data_protection.process_batch(prompts, session_ids)
        
        assert len(results) == len(prompts)
        
        # Check first result (clean)
        assert results[0]["protection_summary"]["redactions_made"] == 0
        
        # Check second result (API key)
        assert results[1]["protection_summary"]["redactions_made"] == 1
        assert "openai_api_key" in results[1]["protection_summary"]["redaction_types"]
        
        # Check third result (email)
        assert results[2]["protection_summary"]["redactions_made"] == 1
        assert "email_address" in results[2]["protection_summary"]["redaction_types"]
        
        # Check fourth result (clean)
        assert results[3]["protection_summary"]["redactions_made"] == 0

    @pytest.mark.asyncio
    async def test_performance_monitoring_real_behavior(self, prompt_data_protection):
        """Test performance monitoring with real operations."""
        # Process multiple prompts to generate performance data
        for i in range(10):
            session_id = f"perf_test_{i}_{uuid.uuid4()}"
            await prompt_data_protection.sanitize_prompt_before_storage(
                f"Test prompt {i} with no sensitive data", session_id, user_consent=True
            )
        
        # Get performance metrics
        metrics = await prompt_data_protection.get_performance_metrics()
        
        assert metrics["total_operations"] == 10
        assert metrics["avg_processing_time_ms"] > 0
        assert metrics["min_processing_time_ms"] >= 0
        assert metrics["max_processing_time_ms"] >= metrics["min_processing_time_ms"]
        assert 0 <= metrics["performance_score"] <= 100

    @pytest.mark.asyncio
    async def test_enhanced_statistics_real_behavior(self, prompt_data_protection):
        """Test enhanced statistics with real data processing."""
        # Process various types of prompts
        test_prompts = [
            ("Clean prompt", "clean_session"),
            ("API key: sk-test123456789012345678901234567890123456", "api_session"),
            ("Email: user@domain.com", "email_session")
        ]
        
        for prompt, session_id in test_prompts:
            await prompt_data_protection.sanitize_prompt_before_storage(
                prompt, f"{session_id}_{uuid.uuid4()}", user_consent=True
            )
        
        # Get enhanced statistics
        stats = await prompt_data_protection.get_enhanced_statistics()
        
        # Verify statistics structure
        assert "statistics" in stats
        assert "configuration" in stats
        assert "performance" in stats
        assert "compliance" in stats
        
        # Verify statistics content
        assert stats["statistics"]["total_prompts_processed"] == 3
        assert stats["statistics"]["prompts_with_sensitive_data"] >= 2
        assert stats["configuration"]["gdpr_enabled"] is True
        assert stats["compliance"]["framework_version"] == "2025.1"

    @pytest.mark.asyncio
    async def test_gdpr_compliance_real_behavior(self, prompt_data_protection):
        """Test GDPR compliance features with real behavior."""
        session_id = f"gdpr_test_{uuid.uuid4()}"
        test_prompt = "Email: gdpr.test@example.com"
        
        # Test with consent
        sanitized_with_consent, summary_with_consent = await prompt_data_protection.sanitize_prompt_before_storage(
            test_prompt, session_id, user_consent=True
        )
        
        assert summary_with_consent["gdpr_compliant"] is True
        assert summary_with_consent["redactions_made"] == 1
        
        # Test without consent
        sanitized_without_consent, summary_without_consent = await prompt_data_protection.sanitize_prompt_before_storage(
            test_prompt, f"{session_id}_no_consent", user_consent=False
        )
        
        assert summary_without_consent["error"] == "GDPR_CONSENT_REQUIRED"
        assert summary_without_consent["processed"] is False

    @pytest.mark.asyncio
    async def test_component_capabilities(self, prompt_data_protection):
        """Test component capabilities reporting."""
        capabilities = await prompt_data_protection.get_capabilities()
        
        expected_capabilities = [
            "data_protection",
            "sensitive_data_detection",
            "prompt_sanitization",
            "gdpr_compliance",
            "differential_privacy",
            "audit_logging",
            "performance_monitoring",
            "real_time_processing",
            "risk_assessment",
            "privacy_by_design",
            "orchestrator_compatible"
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities

    @pytest.mark.asyncio
    async def test_integration_end_to_end(self, orchestrator):
        """End-to-end integration test with orchestrator."""
        # Load component through orchestrator
        loaded_component = await orchestrator.component_loader.load_component(
            "prompt_data_protection", ComponentTier.TIER_6_SECURITY
        )
        assert loaded_component is not None
        
        # Initialize component
        init_success = await orchestrator.component_loader.initialize_component(
            "prompt_data_protection"
        )
        assert init_success is True
        
        # Test component invocation through orchestrator
        test_prompt = "API key: sk-test123456789012345678901234567890123456"
        session_id = f"e2e_test_{uuid.uuid4()}"
        
        # This would be the actual method call through orchestrator
        # For now, we test that the component is properly loaded and accessible
        component_instance = loaded_component.instance
        assert component_instance is not None
        
        # Test direct method call on loaded instance
        sanitized, summary = await component_instance.sanitize_prompt_before_storage(
            test_prompt, session_id, user_consent=True
        )
        
        assert summary["redactions_made"] == 1
        assert "openai_api_key" in summary["redaction_types"]
        assert summary["gdpr_compliant"] is True
        
        # Test health check through loaded instance
        health = await component_instance.health_check()
        assert health["status"] == "healthy"


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(pytest.main([__file__, "-v"]))
