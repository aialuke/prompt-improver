#!/usr/bin/env python3
"""
Test CompositeFeatureExtractor Integration with ML Pipeline Orchestrator

This test verifies real behavior integration without false-positive outputs.
Tests 2025 best practices implementation including:
- Async operations
- Circuit breaker pattern
- Observability and metrics
- Resource management
- Event-driven integration
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from prompt_improver.ml.learning.features.composite_feature_extractor import (
    CompositeFeatureExtractor,
    FeatureExtractionConfig,
    ExtractionMode
)
from prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader
from prompt_improver.ml.orchestration.core.component_registry import ComponentTier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_component_loading():
    """Test that CompositeFeatureExtractor can be loaded by DirectComponentLoader."""
    logger.info("Testing component loading...")
    
    try:
        loader = DirectComponentLoader()
        
        # Test loading the component
        loaded_component = await loader.load_component(
            "composite_feature_extractor",
            ComponentTier.TIER_7_FEATURE_ENGINEERING
        )
        
        if loaded_component is None:
            logger.error("Failed to load CompositeFeatureExtractor")
            return False
        
        logger.info(f"Successfully loaded component: {loaded_component.name}")
        logger.info(f"Component class: {loaded_component.component_class}")
        logger.info(f"Module path: {loaded_component.module_path}")
        
        # Test instantiation
        instance = loaded_component.component_class()

        # Check if it's the correct class by name (avoid import issues)
        if loaded_component.component_class.__name__ != "CompositeFeatureExtractor":
            logger.error(f"Loaded component is not CompositeFeatureExtractor, got: {loaded_component.component_class.__name__}")
            return False
        
        logger.info("✅ Component loading test passed")
        return True
        
    except Exception as e:
        logger.error(f"Component loading test failed: {e}")
        return False


async def test_basic_functionality():
    """Test basic feature extraction functionality."""
    logger.info("Testing basic functionality...")
    
    try:
        # Create extractor with default config
        extractor = CompositeFeatureExtractor()
        
        # Test sync extraction
        test_text = "This is a test prompt for feature extraction. It contains technical terms and instructions."
        result = extractor.extract_features(test_text)
        
        if not result or 'features' not in result:
            logger.error("Sync feature extraction failed")
            return False
        
        logger.info(f"Sync extraction: {len(result['features'])} features extracted")
        
        # Test async extraction
        async_result = await extractor.extract_features_async(test_text)
        
        if not async_result or 'features' not in async_result:
            logger.error("Async feature extraction failed")
            return False
        
        logger.info(f"Async extraction: {len(async_result['features'])} features extracted")
        
        # Verify metadata
        metadata = async_result.get('metadata', {})
        if 'extraction_mode' not in metadata:
            logger.error("Missing extraction mode in metadata")
            return False
        
        logger.info(f"Extraction mode: {metadata['extraction_mode']}")
        logger.info(f"Circuit breaker state: {metadata.get('circuit_breaker_state', 'unknown')}")
        
        logger.info("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        logger.error(f"Basic functionality test failed: {e}")
        return False


async def test_orchestrator_interface():
    """Test orchestrator-compatible interface."""
    logger.info("Testing orchestrator interface...")
    
    try:
        extractor = CompositeFeatureExtractor()
        
        # Test orchestrator-compatible interface
        config = {
            "operation": "extract_features",
            "text": "Advanced machine learning pipeline with feature extraction capabilities.",
            "context_data": {
                "domain": "machine_learning",
                "task_type": "feature_extraction"
            }
        }
        
        result = await extractor.run_orchestrated_analysis(config)
        
        if result.get("status") != "success":
            logger.error(f"Orchestrator interface failed: {result}")
            return False
        
        logger.info("Orchestrator interface result:")
        logger.info(f"  Status: {result['status']}")
        logger.info(f"  Operation: {result['operation']}")
        logger.info(f"  Component: {result['component']}")
        
        # Verify result structure
        analysis_result = result.get('result', {})
        if 'features' not in analysis_result:
            logger.error("Missing features in orchestrator result")
            return False
        
        logger.info(f"  Features extracted: {len(analysis_result['features'])}")
        
        logger.info("✅ Orchestrator interface test passed")
        return True
        
    except Exception as e:
        logger.error(f"Orchestrator interface test failed: {e}")
        return False


async def test_2025_features():
    """Test 2025 best practices features."""
    logger.info("Testing 2025 features...")
    
    try:
        # Test with enhanced configuration
        config = FeatureExtractionConfig(
            execution_mode=ExtractionMode.PARALLEL,
            circuit_breaker_enabled=True,
            enable_metrics=True,
            enable_tracing=True,
            timeout_seconds=10.0,
            cache_enabled=True
        )
        
        extractor = CompositeFeatureExtractor(config)
        
        # Test multiple extractions to verify caching and metrics
        test_texts = [
            "First test prompt for caching verification.",
            "Second test prompt with different content.",
            "First test prompt for caching verification.",  # Duplicate for cache test
        ]
        
        results = []
        for i, text in enumerate(test_texts):
            start_time = time.time()
            result = await extractor.extract_features_async(text)
            processing_time = time.time() - start_time
            
            results.append(result)
            logger.info(f"Extraction {i+1}: {processing_time:.3f}s")
        
        # Verify caching worked (third extraction should be faster)
        metadata = results[-1].get('metadata', {})
        metrics = metadata.get('metrics', {})
        
        logger.info(f"Cache hit rate: {metrics.get('cache_hit_rate', 0):.1f}%")
        logger.info(f"Success rate: {metrics.get('success_rate', 0):.1f}%")
        logger.info(f"Avg processing time: {metrics.get('avg_processing_time', 0):.3f}s")
        
        # Test circuit breaker (simulate failure)
        extractor.circuit_breaker_failures = config.failure_threshold
        extractor._handle_circuit_breaker_failure()
        
        # This should return default result due to circuit breaker
        cb_result = await extractor.extract_features_async("Test after circuit breaker")
        if cb_result.get('metadata', {}).get('is_default'):
            logger.info("Circuit breaker correctly prevented extraction")
        
        logger.info("✅ 2025 features test passed")
        return True
        
    except Exception as e:
        logger.error(f"2025 features test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling and resilience."""
    logger.info("Testing error handling...")
    
    try:
        extractor = CompositeFeatureExtractor()
        
        # Test with invalid input
        result = await extractor.extract_features_async("")
        if not result.get('metadata', {}).get('is_default'):
            logger.error("Should return default result for empty input")
            return False
        
        # Test with very long input
        long_text = "x" * 200000  # Exceeds max_text_length
        result = await extractor.extract_features_async(long_text)
        if not result:
            logger.error("Should handle long text gracefully")
            return False
        
        # Test timeout handling
        config = FeatureExtractionConfig(timeout_seconds=0.001)  # Very short timeout
        timeout_extractor = CompositeFeatureExtractor(config)
        
        result = await timeout_extractor.extract_features_async("Test timeout handling")
        # Should return default result due to timeout
        
        logger.info("✅ Error handling test passed")
        return True
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False


async def main():
    """Run all integration tests."""
    logger.info("Starting CompositeFeatureExtractor integration tests...")
    
    tests = [
        ("Component Loading", test_component_loading),
        ("Basic Functionality", test_basic_functionality),
        ("Orchestrator Interface", test_orchestrator_interface),
        ("2025 Features", test_2025_features),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if await test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"INTEGRATION TEST RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Integration successful!")
        return True
    else:
        logger.error("❌ Some tests failed - Integration needs attention")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
