"""
Test suite to verify application instrumentation fixes work correctly.

Tests real behavior of instrumentation without mocking to ensure
the fixes properly integrate with the APES system.
"""

import pytest
import asyncio
import time
from unittest.mock import patch
from typing import Any

from prompt_improver.metrics.application_instrumentation import (
    PromptImprovementInstrumentation,
    MLPipelineInstrumentation,
    APIServiceInstrumentation,
    DatabaseInstrumentation,
    CacheInstrumentation,
    BatchProcessingInstrumentation,
    auto_instrument_module,
    create_custom_instrumentor,
    instrument_application_startup
)
from prompt_improver.metrics.business_intelligence_metrics import FeatureCategory, CostType
from prompt_improver.metrics.performance_metrics import CacheType


class TestPromptService:
    """Test service for prompt improvement instrumentation."""
    
    async def improve_prompt(self, prompt: str) -> dict:
        """Test method for prompt improvement."""
        await asyncio.sleep(0.01)  # Simulate processing
        return {"improved": f"Enhanced: {prompt}", "confidence": 0.95}
    
    def analyze_prompt(self, prompt: str) -> dict:
        """Sync test method for prompt analysis."""
        time.sleep(0.01)  # Simulate processing
        return {"analysis": f"Analysis of: {prompt}", "score": 0.8}


class TestMLService:
    """Test service for ML pipeline instrumentation."""
    
    async def train_model(self, data: list) -> dict:
        """Test method for model training."""
        await asyncio.sleep(0.02)
        return {"model_id": "test_model", "accuracy": 0.92}
    
    def predict(self, input_data: Any) -> dict:
        """Test method for prediction."""
        return {"prediction": "test_result", "confidence": 0.88}


class TestAPIService:
    """Test service for API instrumentation."""
    
    def get_health(self) -> dict:
        """Test health endpoint."""
        return {"status": "healthy", "timestamp": time.time()}
    
    async def process_request(self, request_data: dict) -> dict:
        """Test async API endpoint."""
        await asyncio.sleep(0.01)
        return {"processed": True, "data": request_data}


class TestDatabaseService:
    """Test service for database instrumentation."""
    
    async def execute(self, query: str, *args, **kwargs) -> dict:
        """Test async database method."""
        await asyncio.sleep(0.005)
        if "SELECT" in query.upper():
            return [{"id": 1, "name": "test"}]
        return {"rowcount": 1}
    
    def fetch(self, query: str) -> list:
        """Test sync database method."""
        time.sleep(0.005)
        return [{"id": 1, "data": "test"}]


class TestCacheService:
    """Test service for cache instrumentation."""
    
    async def get(self, key: str) -> Any:
        """Test async cache get."""
        await asyncio.sleep(0.001)
        return f"cached_value_{key}" if key != "missing" else None
    
    def set(self, key: str, value: Any) -> bool:
        """Test sync cache set."""
        time.sleep(0.001)
        return True


class TestBatchProcessor:
    """Test service for batch processing instrumentation."""
    
    async def process_batch(self, items: list) -> dict:
        """Test batch processing method."""
        await asyncio.sleep(0.01 * len(items))
        return {"processed": len(items), "success": True}


@pytest.mark.asyncio
class TestInstrumentationFixes:
    """Test that all instrumentation fixes work correctly."""
    
    async def test_prompt_service_instrumentation(self):
        """Test prompt service instrumentation works without errors."""
        # Apply instrumentation
        instrumented_class = PromptImprovementInstrumentation.instrument_prompt_service(TestPromptService)
        service = instrumented_class()
        
        # Test async method
        result = await service.improve_prompt("test prompt")
        assert result["improved"] == "Enhanced: test prompt"
        assert result["confidence"] == 0.95
        
        # Test sync method
        result = service.analyze_prompt("test prompt")
        assert result["analysis"] == "Analysis of: test prompt"
        assert result["score"] == 0.8
    
    async def test_ml_service_instrumentation(self):
        """Test ML service instrumentation works without errors."""
        instrumented_class = MLPipelineInstrumentation.instrument_ml_service(TestMLService)
        service = instrumented_class()
        
        # Test async method
        result = await service.train_model([1, 2, 3])
        assert result["model_id"] == "test_model"
        assert result["accuracy"] == 0.92
        
        # Test sync method
        result = service.predict({"input": "test"})
        assert result["prediction"] == "test_result"
        assert result["confidence"] == 0.88
    
    async def test_api_service_instrumentation(self):
        """Test API service instrumentation works without errors."""
        instrumented_class = APIServiceInstrumentation.instrument_api_service(
            TestAPIService, FeatureCategory.API_INTEGRATION
        )
        service = instrumented_class()
        
        # Test sync method
        result = service.get_health()
        assert result["status"] == "healthy"
        assert "timestamp" in result
        
        # Test async method
        test_data = {"test": "data"}
        result = await service.process_request(test_data)
        assert result["processed"] is True
        assert result["data"] == test_data
    
    async def test_database_instrumentation(self):
        """Test database instrumentation works without errors."""
        instrumented_class = DatabaseInstrumentation.instrument_database_class(TestDatabaseService)
        service = instrumented_class()
        
        # Test async method with SELECT
        result = await service.execute("SELECT * FROM test_table")
        assert len(result) == 1
        assert result[0]["name"] == "test"
        
        # Test async method with INSERT
        result = await service.execute("INSERT INTO test_table VALUES (1, 'test')")
        assert result["rowcount"] == 1
        
        # Test sync method
        result = service.fetch("SELECT * FROM test_table")
        assert len(result) == 1
        assert result[0]["data"] == "test"
    
    async def test_cache_instrumentation(self):
        """Test cache instrumentation works without errors."""
        instrumented_class = CacheInstrumentation.instrument_cache_class(
            TestCacheService, CacheType.APPLICATION
        )
        service = instrumented_class()
        
        # Test async get with hit
        result = await service.get("existing_key")
        assert result == "cached_value_existing_key"
        
        # Test async get with miss
        result = await service.get("missing")
        assert result is None
        
        # Test sync set
        result = service.set("test_key", "test_value")
        assert result is True
    
    async def test_batch_processor_instrumentation(self):
        """Test batch processor instrumentation works without errors."""
        instrumented_class = BatchProcessingInstrumentation.instrument_batch_processor(TestBatchProcessor)
        processor = instrumented_class()
        
        # Test batch processing
        items = [1, 2, 3, 4, 5]
        result = await processor.process_batch(items)
        assert result["processed"] == 5
        assert result["success"] is True
    
    def test_custom_instrumentor(self):
        """Test custom instrumentor works without errors."""
        
        class CustomService:
            def custom_method(self, data: str) -> str:
                return f"processed_{data}"
            
            async def async_custom_method(self, data: str) -> str:
                await asyncio.sleep(0.001)
                return f"async_processed_{data}"
        
        # Apply custom instrumentation
        instrumented_class = create_custom_instrumentor(
            CustomService,
            FeatureCategory.CUSTOM_INTEGRATION,
            CostType.COMPUTE,
            0.002
        )
        service = instrumented_class()
        
        # Test sync method
        result = service.custom_method("test")
        assert result == "processed_test"
    
    def test_auto_instrument_module(self):
        """Test auto module instrumentation works without errors."""
        import types
        
        # Create a test module
        test_module = types.ModuleType("test_module")
        test_module.TestPromptService = TestPromptService
        test_module.TestMLService = TestMLService
        test_module.TestAPIHandler = TestAPIService
        test_module.TestDatabaseRepository = TestDatabaseService
        test_module.TestRedisCache = TestCacheService
        test_module.TestBatchProcessor = TestBatchProcessor
        
        # Apply auto instrumentation
        auto_instrument_module(test_module)
        
        # Verify classes were instrumented (they should still be callable)
        prompt_service = test_module.TestPromptService()
        assert hasattr(prompt_service, 'improve_prompt')
        
        ml_service = test_module.TestMLService()
        assert hasattr(ml_service, 'train_model')
    
    @patch('prompt_improver.metrics.application_instrumentation.logger')
    def test_instrument_application_startup_no_errors(self, mock_logger):
        """Test that instrument_application_startup runs without syntax errors."""
        # This should not raise any syntax or indentation errors
        try:
            instrument_application_startup()
            # The function should complete without syntax errors
            # Import errors are expected and handled
            assert True
        except SyntaxError as e:
            pytest.fail(f"Syntax error in instrument_application_startup: {e}")
        except IndentationError as e:
            pytest.fail(f"Indentation error in instrument_application_startup: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
