"""
Integration tests for modernized component registry.

Tests that AdvancedDimensionalityReducer and ProductionSyntheticDataGenerator
are properly registered and integrated with the orchestrator system.
"""

import pytest
import asyncio
import numpy as np
from typing import Dict, Any

from src.prompt_improver.ml.orchestration.core.component_registry import (
    ComponentRegistry, ComponentTier, ComponentInfo
)
from src.prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from src.prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader


class TestModernizedComponentRegistry:
    """Test suite for modernized component registry integration."""
    
    @pytest.fixture
    async def component_registry(self):
        """Create and initialize component registry."""
        config = OrchestratorConfig()
        registry = ComponentRegistry(config)
        await registry._load_component_definitions()
        return registry
    
    @pytest.fixture
    def component_loader(self):
        """Create component loader."""
        return DirectComponentLoader()
    
    @pytest.mark.asyncio
    async def test_modernized_components_registered(self, component_registry):
        """Test that modernized components are properly registered."""
        # Check dimensionality reducer registration
        dim_reducer_info = component_registry.components.get("dimensionality_reducer")
        assert dim_reducer_info is not None, "AdvancedDimensionalityReducer should be registered"
        assert dim_reducer_info.tier == ComponentTier.TIER_1_CORE, "Should be in Tier 1"
        
        # Check synthetic data generator registration
        synthetic_gen_info = component_registry.components.get("synthetic_data_generator")
        assert synthetic_gen_info is not None, "ProductionSyntheticDataGenerator should be registered"
        assert synthetic_gen_info.tier == ComponentTier.TIER_1_CORE, "Should be in Tier 1"
    
    @pytest.mark.asyncio
    async def test_neural_capabilities_registered(self, component_registry):
        """Test that neural network capabilities are properly registered."""
        dim_reducer_info = component_registry.components.get("dimensionality_reducer")
        
        # Check for neural capabilities
        capability_names = [cap.name for cap in dim_reducer_info.capabilities]
        
        # Should have neural network capabilities
        neural_capabilities = [
            "neural_autoencoder", "variational_autoencoder", 
            "transformer_attention", "diffusion_models"
        ]
        
        for neural_cap in neural_capabilities:
            assert any(neural_cap in cap for cap in capability_names), f"Should have {neural_cap} capability"
    
    @pytest.mark.asyncio
    async def test_component_loading_through_orchestrator(self, component_loader):
        """Test that components can be loaded through orchestrator."""
        # Test dimensionality reducer loading
        dim_reducer_component = await component_loader.load_component(
            "dimensionality_reducer", ComponentTier.TIER_1_CORE
        )
        assert dim_reducer_component is not None, "Should load dimensionality reducer"
        assert hasattr(dim_reducer_component, 'component_class'), "Should have component class"
        
        # Test synthetic data generator loading
        synthetic_gen_component = await component_loader.load_component(
            "synthetic_data_generator", ComponentTier.TIER_1_CORE
        )
        assert synthetic_gen_component is not None, "Should load synthetic data generator"
        assert hasattr(synthetic_gen_component, 'component_class'), "Should have component class"
    
    @pytest.mark.asyncio
    async def test_component_instantiation_through_orchestrator(self, component_loader):
        """Test that components can be instantiated through orchestrator."""
        # Load and instantiate dimensionality reducer
        dim_reducer_component = await component_loader.load_component(
            "dimensionality_reducer", ComponentTier.TIER_1_CORE
        )
        
        # Import the config class
        from src.prompt_improver.ml.optimization.algorithms.dimensionality_reducer import DimensionalityConfig
        
        config = DimensionalityConfig(target_dimensions=10, enable_neural_methods=True)
        dim_reducer_instance = dim_reducer_component.component_class(config=config)
        
        assert dim_reducer_instance is not None, "Should instantiate dimensionality reducer"
        assert hasattr(dim_reducer_instance, 'reduce_dimensions'), "Should have reduce_dimensions method"
        
        # Load and instantiate synthetic data generator
        synthetic_gen_component = await component_loader.load_component(
            "synthetic_data_generator", ComponentTier.TIER_1_CORE
        )
        
        synthetic_gen_instance = synthetic_gen_component.component_class(
            target_samples=100,
            generation_method="statistical"
        )
        
        assert synthetic_gen_instance is not None, "Should instantiate synthetic data generator"
        assert hasattr(synthetic_gen_instance, 'generate_data'), "Should have generate_data method"
    
    @pytest.mark.asyncio
    async def test_real_component_execution_through_orchestrator(self, component_loader):
        """Test real component execution through orchestrator (no mocks)."""
        # Test dimensionality reducer execution
        dim_reducer_component = await component_loader.load_component(
            "dimensionality_reducer", ComponentTier.TIER_1_CORE
        )
        
        from src.prompt_improver.ml.optimization.algorithms.dimensionality_reducer import DimensionalityConfig
        
        config = DimensionalityConfig(
            target_dimensions=5,
            enable_neural_methods=False,  # Use statistical methods for faster testing
            fast_mode=True
        )
        
        dim_reducer_instance = dim_reducer_component.component_class(config=config)
        
        # Create real test data
        np.random.seed(42)
        X = np.random.randn(50, 20)  # 50 samples, 20 features
        y = np.random.randint(0, 3, 50)  # 3 classes
        
        # Execute reduction
        result = await dim_reducer_instance.reduce_dimensions(X, y)
        
        # Validate real execution (not mock)
        assert result.original_dimensions == 20, "Should have correct original dimensions"
        assert result.reduced_dimensions == 5, "Should have correct reduced dimensions"
        assert result.transformed_data.shape == (50, 5), "Should have correct output shape"
        assert result.processing_time > 0, "Should have real processing time"
        assert 0.0 <= result.variance_preserved <= 1.0, "Should have valid variance preserved"
        
        # Test synthetic data generator execution
        synthetic_gen_component = await component_loader.load_component(
            "synthetic_data_generator", ComponentTier.TIER_1_CORE
        )
        
        synthetic_gen_instance = synthetic_gen_component.component_class(
            target_samples=30,
            generation_method="statistical",
            use_enhanced_scoring=True
        )
        
        # Execute generation
        synthetic_data = await synthetic_gen_instance.generate_data()
        
        # Validate real execution (not mock) - allow for quality filtering
        actual_samples = synthetic_data["metadata"]["total_samples"]
        assert 25 <= actual_samples <= 30, f"Should generate approximately correct number of samples, got {actual_samples}"
        assert len(synthetic_data["features"]) == actual_samples, "Features should match sample count"
        assert len(synthetic_data["effectiveness_scores"]) == actual_samples, "Scores should match sample count"
        assert "generation_timestamp" in synthetic_data["metadata"], "Should have real timestamp"
        
        # Verify data quality
        features_array = np.array(synthetic_data["features"])
        assert features_array.shape[0] == actual_samples, "Should have correct feature array shape"
        assert features_array.shape[1] > 0, "Should have feature dimensions"
        
        # Verify effectiveness scores are in valid range
        effectiveness_scores = synthetic_data["effectiveness_scores"]
        assert all(0.0 <= score <= 1.0 for score in effectiveness_scores), "Effectiveness scores should be valid"
    
    @pytest.mark.asyncio
    async def test_component_error_handling(self, component_loader):
        """Test component error handling through orchestrator."""
        # Test invalid component loading
        invalid_component = await component_loader.load_component(
            "nonexistent_component", ComponentTier.TIER_1_CORE
        )
        assert invalid_component is None, "Should return None for invalid component"
        
        # Test dimensionality reducer with invalid data
        dim_reducer_component = await component_loader.load_component(
            "dimensionality_reducer", ComponentTier.TIER_1_CORE
        )
        
        from src.prompt_improver.ml.optimization.algorithms.dimensionality_reducer import DimensionalityConfig
        
        config = DimensionalityConfig(target_dimensions=5)
        dim_reducer_instance = dim_reducer_component.component_class(config=config)
        
        # Test with invalid input (1D array instead of 2D)
        invalid_X = np.array([1, 2, 3, 4, 5])

        with pytest.raises((ValueError, IndexError)):  # Allow both ValueError and IndexError
            await dim_reducer_instance.reduce_dimensions(invalid_X)
    
    @pytest.mark.asyncio
    async def test_component_resource_requirements(self, component_registry):
        """Test that components have proper resource requirements."""
        # Check dimensionality reducer resource requirements
        dim_reducer_info = component_registry.components.get("dimensionality_reducer")
        resource_reqs = dim_reducer_info.resource_requirements
        
        assert "memory" in resource_reqs, "Should specify memory requirements"
        assert "cpu" in resource_reqs, "Should specify CPU requirements"
        assert "gpu" in resource_reqs, "Should specify GPU requirements"
        
        # Check synthetic data generator resource requirements
        synthetic_gen_info = component_registry.components.get("synthetic_data_generator")
        resource_reqs = synthetic_gen_info.resource_requirements
        
        assert "memory" in resource_reqs, "Should specify memory requirements"
        assert "cpu" in resource_reqs, "Should specify CPU requirements"
    
    @pytest.mark.asyncio
    async def test_neural_capabilities_metadata(self, component_registry):
        """Test that neural capabilities metadata is properly set."""
        # Check dimensionality reducer neural metadata
        dim_reducer_info = component_registry.components.get("dimensionality_reducer")
        metadata = dim_reducer_info.metadata
        
        if "neural_capabilities" in metadata:
            neural_caps = metadata["neural_capabilities"]
            assert neural_caps.get("pytorch_support") is True, "Should support PyTorch"
            assert "model_types" in neural_caps, "Should specify model types"
            
            model_types = neural_caps["model_types"]
            expected_types = ["autoencoder", "vae", "transformer", "diffusion"]
            for model_type in expected_types:
                assert model_type in model_types, f"Should support {model_type}"
        
        # Check synthetic data generator neural metadata
        synthetic_gen_info = component_registry.components.get("synthetic_data_generator")
        metadata = synthetic_gen_info.metadata
        
        if "neural_capabilities" in metadata:
            neural_caps = metadata["neural_capabilities"]
            assert neural_caps.get("pytorch_support") is True, "Should support PyTorch"
            assert "generation_methods" in neural_caps, "Should specify generation methods"
            
            gen_methods = neural_caps["generation_methods"]
            expected_methods = ["statistical", "neural", "hybrid", "diffusion"]
            for method in expected_methods:
                assert method in gen_methods, f"Should support {method}"
    
    @pytest.mark.asyncio
    async def test_component_dependencies(self, component_registry):
        """Test component dependencies are properly configured."""
        # Check that components have minimal dependencies
        dim_reducer_info = component_registry.components.get("dimensionality_reducer")
        assert len(dim_reducer_info.dependencies) == 0, "DimensionalityReducer should have no dependencies"
        
        synthetic_gen_info = component_registry.components.get("synthetic_data_generator")
        assert len(synthetic_gen_info.dependencies) == 0, "SyntheticDataGenerator should have no dependencies"
    
    @pytest.mark.asyncio
    async def test_component_version_info(self, component_registry):
        """Test component version information."""
        # Check version information
        dim_reducer_info = component_registry.components.get("dimensionality_reducer")
        assert dim_reducer_info.version is not None, "Should have version information"
        
        synthetic_gen_info = component_registry.components.get("synthetic_data_generator")
        assert synthetic_gen_info.version is not None, "Should have version information"
    
    def test_no_false_positives_in_registration(self, component_registry):
        """Test that component registration doesn't produce false positives."""
        # Verify that only real components are registered
        all_components = component_registry.components
        
        # Check that registered components actually exist
        for component_name, component_info in all_components.items():
            assert component_info.name == component_name, "Component name should match key"
            assert component_info.description is not None, "Should have description"
            assert len(component_info.capabilities) > 0, "Should have capabilities"
            
        # Verify tier assignments are correct
        tier1_components = component_registry.components_by_tier[ComponentTier.TIER_1_CORE]
        assert "dimensionality_reducer" in tier1_components, "DimensionalityReducer should be in Tier 1"
        assert "synthetic_data_generator" in tier1_components, "SyntheticDataGenerator should be in Tier 1"
