#!/usr/bin/env python3
"""
Test script for corrected Priority 5a component integration.

Tests the actual components (ProductionSyntheticDataGenerator and MLModelService)
after removing duplicate/non-existent components from the integration plan.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_production_synthetic_data_generator():
    """Test ProductionSyntheticDataGenerator orchestrator integration."""
    logger.info("Testing ProductionSyntheticDataGenerator orchestrator integration...")
    
    try:
        from prompt_improver.ml.preprocessing.synthetic_data_generator import ProductionSyntheticDataGenerator
        
        # Create generator instance
        generator = ProductionSyntheticDataGenerator(
            target_samples=50,
            generation_method="statistical",
            use_enhanced_scoring=True
        )
        
        # Test orchestrator interface
        config = {
            "target_samples": 25,
            "generation_method": "statistical",
            "output_path": "./outputs/test_synthetic",
            "quality_assessment": True
        }
        
        result = await generator.run_orchestrated_analysis(config)
        
        # Verify orchestrator compatibility
        assert result["orchestrator_compatible"] is True
        assert "component_result" in result
        assert "local_metadata" in result
        assert result["local_metadata"]["component_name"] == "ProductionSyntheticDataGenerator"
        
        # Verify synthetic data generation
        synthetic_data = result["component_result"]["synthetic_data"]
        assert len(synthetic_data["features"]) > 0
        assert len(synthetic_data["effectiveness_scores"]) > 0
        assert synthetic_data["metadata"]["total_samples"] > 0
        
        logger.info("✅ ProductionSyntheticDataGenerator orchestrator integration: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ ProductionSyntheticDataGenerator orchestrator integration: FAILED - {e}")
        return False


async def test_ml_model_service():
    """Test MLModelService orchestrator integration."""
    logger.info("Testing MLModelService orchestrator integration...")
    
    try:
        from prompt_improver.ml.core.ml_integration import MLModelService
        
        # Create service instance
        service = MLModelService()
        
        # Test training operation
        training_config = {
            "operation": "train",
            "model_config": {
                "model_type": "random_forest"
            },
            "training_data": {
                "rules": [
                    {"rule_id": "test_rule_1", "effectiveness": 0.85, "features": [1, 2, 3]},
                    {"rule_id": "test_rule_2", "effectiveness": 0.72, "features": [2, 3, 4]},
                ]
            },
            "output_path": "./outputs/test_models"
        }
        
        result = await service.run_orchestrated_analysis(training_config)
        
        # Verify orchestrator compatibility
        assert result["orchestrator_compatible"] is True
        assert "component_result" in result
        assert "local_metadata" in result
        assert result["local_metadata"]["component_name"] == "MLModelService"
        
        # Verify training operation
        ml_result = result["component_result"]["ml_operation_result"]
        operation_summary = result["component_result"]["operation_summary"]
        assert operation_summary["operation"] == "train"
        assert operation_summary["operation_type"] == "rule_training"
        
        logger.info("✅ MLModelService orchestrator integration: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ MLModelService orchestrator integration: FAILED - {e}")
        return False


async def test_component_registry_integration():
    """Test that components are properly registered in the orchestrator."""
    logger.info("Testing component registry integration...")
    
    try:
        from prompt_improver.ml.orchestration.core.component_registry import ComponentRegistry
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        
        # Initialize registry
        config = OrchestratorConfig()
        registry = ComponentRegistry(config)
        
        # Get predefined components
        predefined = registry._get_predefined_components()
        
        # Verify our components are registered
        assert "synthetic_data_generator" in predefined
        assert "ml_model_service" in predefined
        
        # Verify component details
        synthetic_component = predefined["synthetic_data_generator"]
        assert synthetic_component["class_name"] == "ProductionSyntheticDataGenerator"
        assert synthetic_component["module_path"] == "prompt_improver.ml.preprocessing.synthetic_data_generator"
        
        ml_component = predefined["ml_model_service"]
        assert ml_component["class_name"] == "MLModelService"
        assert ml_component["module_path"] == "prompt_improver.ml.core.ml_integration"
        
        logger.info("✅ Component registry integration: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component registry integration: FAILED - {e}")
        return False


async def test_duplicate_removal_verification():
    """Verify that duplicate components are no longer referenced."""
    logger.info("Testing duplicate component removal verification...")
    
    try:
        # Verify DiffusionSyntheticGenerator and NeuralSyntheticGenerator are internal classes
        from prompt_improver.ml.preprocessing.synthetic_data_generator import (
            DiffusionSyntheticGenerator, 
            NeuralSyntheticGenerator,
            ProductionSyntheticDataGenerator
        )
        
        # These should be importable as internal classes
        assert DiffusionSyntheticGenerator is not None
        assert NeuralSyntheticGenerator is not None
        
        # But they should be used within ProductionSyntheticDataGenerator
        generator = ProductionSyntheticDataGenerator(
            target_samples=10,
            generation_method="diffusion",
            neural_model_type="diffusion"
        )
        
        # Verify the internal generator is created
        assert generator.neural_generator is not None
        assert isinstance(generator.neural_generator, DiffusionSyntheticGenerator)
        
        logger.info("✅ Duplicate removal verification: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Duplicate removal verification: FAILED - {e}")
        return False


async def main():
    """Run all Priority 5a corrected integration tests."""
    logger.info("🔍 Starting Priority 5a Corrected Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        test_production_synthetic_data_generator,
        test_ml_model_service,
        test_component_registry_integration,
        test_duplicate_removal_verification
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info("=" * 60)
    logger.info(f"📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Priority 5a corrected integration tests PASSED!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests FAILED")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
