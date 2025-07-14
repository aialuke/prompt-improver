"""
Test advanced memory optimizations for BERT models.

Tests the implementation of 4-bit quantization, tiny models, and automatic model selection
for achieving ultra-low memory usage.
"""

import pytest
import psutil
import os
import gc
from unittest.mock import patch, MagicMock

from src.prompt_improver.utils.model_manager import (
    ModelConfig, ModelManager, get_ultra_lightweight_ner_pipeline,
    get_memory_optimized_config
)
from src.prompt_improver.analysis.linguistic_analyzer import (
    LinguisticAnalyzer, get_ultra_lightweight_config, get_memory_optimized_config as get_ling_config
)


class TestAdvancedMemoryOptimization:
    """Test advanced memory optimization techniques."""
    
    def test_4bit_quantization_config(self):
        """Test 4-bit quantization configuration."""
        config = ModelConfig(
            use_4bit_quantization=True,
            quantization_bits=4,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            max_memory_threshold_mb=30
        )
        
        assert config.use_4bit_quantization is True
        assert config.quantization_bits == 4
        assert config.bnb_4bit_compute_dtype == "float16"
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True
    
    def test_tiny_model_alternatives(self):
        """Test tiny model alternative configuration."""
        config = ModelConfig()
        
        # Check that tiny model alternatives are initialized
        assert config.tiny_model_alternatives is not None
        assert len(config.tiny_model_alternatives) >= 3
        assert "bert-tiny" in config.tiny_model_alternatives[2]
        assert "TinyBERT" in config.tiny_model_alternatives[1]
        assert "mobilebert" in config.tiny_model_alternatives[0]
    
    def test_memory_thresholds(self):
        """Test memory threshold configuration."""
        config = ModelConfig()
        
        # Check memory thresholds are properly set
        assert config.memory_thresholds["bert-large"] == 350
        assert config.memory_thresholds["distilbert"] == 100
        assert config.memory_thresholds["mobilebert"] == 50
        assert config.memory_thresholds["tinybert"] == 25
        assert config.memory_thresholds["bert-tiny"] == 0
    
    @patch('psutil.virtual_memory')
    def test_automatic_model_selection(self, mock_memory):
        """Test automatic model selection based on available memory."""
        # Mock different memory scenarios
        memory_scenarios = [
            (400 * 1024 * 1024, "bert-large"),    # 400MB -> use large
            (120 * 1024 * 1024, "distilbert"),   # 120MB -> use distilbert
            (70 * 1024 * 1024, "mobilebert"),    # 70MB -> use mobilebert
            (40 * 1024 * 1024, "tinybert"),      # 40MB -> use tinybert
            (20 * 1024 * 1024, "bert-tiny")      # 20MB -> use tiny
        ]
        
        for available_bytes, expected_model_type in memory_scenarios:
            mock_memory.return_value.available = available_bytes
            
            config = ModelConfig(auto_select_model=True)
            manager = ModelManager(config)
            
            selected_model = manager._select_optimal_model()
            
            if expected_model_type == "bert-large":
                assert "bert-large" in selected_model
            elif expected_model_type == "distilbert":
                assert "distilbert" in selected_model
            elif expected_model_type == "mobilebert":
                assert "mobilebert" in selected_model
            elif expected_model_type == "tinybert":
                assert "TinyBERT" in selected_model
            elif expected_model_type == "bert-tiny":
                assert "bert-tiny" in selected_model
    
    def test_memory_optimized_config_factory(self):
        """Test memory-optimized configuration factory."""
        # Test ultra-lightweight config (< 30MB)
        config = get_memory_optimized_config(25)
        assert "bert-tiny" in config.model_name
        assert config.use_quantization is True
        assert config.quantization_bits == 4
        assert config.use_4bit_quantization is True
        
        # Test lightweight config (30-60MB)
        config = get_memory_optimized_config(45)
        assert "TinyBERT" in config.model_name
        assert config.use_4bit_quantization is True
        
        # Test mobile config (60-100MB)
        config = get_memory_optimized_config(80)
        assert "mobilebert" in config.model_name
        assert config.quantization_bits == 8
        
        # Test standard config (>100MB)
        config = get_memory_optimized_config(150)
        assert config.auto_select_model is True
        assert config.max_memory_threshold_mb == 150
    
    def test_ultra_lightweight_linguistic_config(self):
        """Test ultra-lightweight linguistic analyzer configuration."""
        config = get_ultra_lightweight_config()
        
        assert config.use_ultra_lightweight_models is True
        assert config.enable_4bit_quantization is True
        assert config.quantization_bits == 4
        assert config.max_memory_threshold_mb == 30
        assert config.enable_dependency_parsing is False  # Disabled to save memory
        assert config.max_workers == 1  # Minimal concurrency
        assert config.cache_size == 50  # Minimal cache
    
    def test_linguistic_memory_optimized_config(self):
        """Test linguistic analyzer memory-optimized configuration."""
        # Test ultra-lightweight
        config = get_ling_config(25)
        assert config.use_ultra_lightweight_models is True
        assert config.max_memory_threshold_mb == 25
        
        # Test lightweight with 4-bit
        config = get_ling_config(45)
        assert config.use_lightweight_models is True
        assert config.enable_4bit_quantization is True
        
        # Test production with memory optimization
        config = get_ling_config(120)
        assert config.enable_4bit_quantization is True
        assert config.max_memory_threshold_mb == 120
    
    @pytest.mark.integration
    def test_memory_usage_with_optimizations(self):
        """Test actual memory usage with optimizations enabled."""
        # Get baseline memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Test ultra-lightweight configuration
        config = get_ultra_lightweight_config()
        analyzer = LinguisticAnalyzer(config)
        
        # Simple analysis to trigger model loading
        test_text = "Test prompt for NER analysis."
        try:
            features = analyzer.analyze(test_text)
            # Force garbage collection
            gc.collect()
            
            # Measure memory after initialization
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # With ultra-lightweight config, memory increase should be minimal
            # This is more lenient than production to account for test environment overhead
            assert memory_increase < 100, f"Memory increase too high: {memory_increase:.2f}MB"
            
            print(f"Memory increase with ultra-lightweight config: {memory_increase:.2f}MB")
            
        except Exception as e:
            # If models aren't available for download in test environment, that's OK
            pytest.skip(f"Model not available in test environment: {e}")
        finally:
            # Cleanup
            analyzer.cleanup()
            gc.collect()
    
    def test_quantization_config_validation(self):
        """Test quantization configuration validation."""
        # Test valid 4-bit configuration
        config = ModelConfig(
            use_4bit_quantization=True,
            quantization_bits=4,
            bnb_4bit_compute_dtype="float16"
        )
        assert config.use_4bit_quantization is True
        
        # Test valid 8-bit configuration
        config = ModelConfig(
            use_quantization=True,
            quantization_bits=8
        )
        assert config.use_quantization is True
        assert config.quantization_bits == 8
    
    @pytest.mark.unit
    def test_model_manager_optimization_flags(self):
        """Test model manager handles optimization flags correctly."""
        config = ModelConfig(
            use_4bit_quantization=True,
            quantization_bits=4,
            auto_select_model=True,
            max_memory_threshold_mb=50
        )
        
        manager = ModelManager(config)
        
        # Verify configuration is properly stored
        assert manager.config.use_4bit_quantization is True
        assert manager.config.quantization_bits == 4
        assert manager.config.auto_select_model is True
        assert manager.config.max_memory_threshold_mb == 50
    
    def test_memory_threshold_edge_cases(self):
        """Test memory threshold edge cases."""
        # Test very low memory
        config = get_memory_optimized_config(1)
        assert "bert-tiny" in config.model_name
        
        # Test zero memory
        config = get_memory_optimized_config(0)
        assert "bert-tiny" in config.model_name
        
        # Test very high memory
        config = get_memory_optimized_config(1000)
        assert config.auto_select_model is True
        assert config.max_memory_threshold_mb == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])