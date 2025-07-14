"""
Model Manager for Transformers and ML Models

This module provides centralized management for transformer models including
memory optimization, caching, and production-ready configurations.
"""

import logging
import gc
import threading
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import torch

# Handle transformers import gracefully
try:
    from transformers import (
        pipeline, AutoTokenizer, AutoModelForTokenClassification, 
        BitsAndBytesConfig, set_seed
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModelForTokenClassification = None
    BitsAndBytesConfig = None
    set_seed = None
    TRANSFORMERS_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuration for model loading and optimization."""
    
    # Model settings
    model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    task: str = "ner"
    aggregation_strategy: str = "simple"
    
    # Memory optimization
    use_quantization: bool = True
    quantization_bits: int = 8  # 4, 8, or 16
    torch_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    device_map: str = "auto"  # "auto", "cpu", "cuda"
    low_cpu_mem_usage: bool = True
    
    # Advanced quantization settings
    use_4bit_quantization: bool = False
    bnb_4bit_compute_dtype: str = "float16"  # Compute dtype for 4-bit
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True  # Double quantization
    
    # Performance settings
    max_memory: Optional[Dict[str, str]] = None
    offload_to_cpu: bool = False
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Fallback settings
    use_lightweight_fallback: bool = True
    fallback_model: str = "distilbert-base-uncased"
    tiny_model_alternatives: list = None  # Will be set in __post_init__
    max_memory_threshold_mb: int = 200
    
    # Auto model selection based on memory
    auto_select_model: bool = True
    memory_thresholds: Dict[str, int] = None  # Will be set in __post_init__
    
    def __post_init__(self):
        """Initialize default values for complex fields."""
        if self.tiny_model_alternatives is None:
            self.tiny_model_alternatives = [
                "google/mobilebert-uncased",  # ~100MB, mobile-optimized
                "huawei-noah/TinyBERT_General_4L_312D",  # ~60MB, 4 layers
                "prajjwal1/bert-tiny",  # ~17MB, extremely small
            ]
        
        if self.memory_thresholds is None:
            self.memory_thresholds = {
                "bert-large": 350,  # Use large models above 350MB
                "distilbert": 100,  # Use distilled models above 100MB
                "mobilebert": 50,   # Use mobile models above 50MB
                "tinybert": 25,     # Use tiny models above 25MB
                "bert-tiny": 0      # Fallback to tiny for minimal memory
            }


class ModelManager:
    """
    Centralized manager for transformer models with memory optimization.
    
    This class provides:
    - Model caching and singleton pattern
    - Memory optimization with quantization
    - Graceful fallback for resource-constrained environments
    - Production-ready error handling
    """
    
    _instances: Dict[str, 'ModelManager'] = {}
    _lock = threading.Lock()
    _model_cache: Dict[str, Any] = {}
    _cache_lock = threading.Lock()
    
    def __new__(cls, config: Optional[ModelConfig] = None):
        """Implement singleton pattern per model configuration."""
        if config is None:
            config = ModelConfig()
        
        # Create cache key from config
        cache_key = f"{config.model_name}_{config.task}_{config.quantization_bits}"
        
        with cls._lock:
            if cache_key not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[cache_key] = instance
            return cls._instances[cache_key]
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the model manager."""
        if hasattr(self, '_initialized'):
            return
        
        self.config = config or ModelConfig()
        self.logger = logging.getLogger(__name__)
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._memory_usage_mb = 0
        self._initialization_failed = False
        self._initialized = True
        
        self.logger.info(f"Initialized ModelManager for {self.config.model_name}")
    
    def get_pipeline(self, force_reload: bool = False) -> Optional[Any]:
        """
        Get the transformers pipeline with caching and optimization.
        
        Args:
            force_reload: Force reload the model even if cached
            
        Returns:
            Transformers pipeline or None if unavailable
        """
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers library not available")
            return None
        
        cache_key = f"{self.config.model_name}_{self.config.task}"
        
        with self._cache_lock:
            # Check cache first
            if not force_reload and cache_key in self._model_cache:
                cached_pipeline = self._model_cache[cache_key]
                if cached_pipeline is not None:
                    self.logger.debug(f"Using cached pipeline for {cache_key}")
                    return cached_pipeline
            
            # Initialize pipeline if not cached or force reload
            if self._pipeline is None or force_reload:
                self._pipeline = self._initialize_pipeline()
            
            # Cache the pipeline
            if self._pipeline is not None:
                self._model_cache[cache_key] = self._pipeline
                self.logger.info(f"Cached pipeline for {cache_key}")
            
            return self._pipeline
    
    def _select_optimal_model(self) -> str:
        """Select the optimal model based on available memory."""
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
        except:
            # If we can't determine memory, use default
            available_memory_mb = self.config.max_memory_threshold_mb
        
        self.logger.info(f"Available memory: {available_memory_mb:.1f}MB")
        
        # Select model based on memory thresholds
        if available_memory_mb >= self.config.memory_thresholds["bert-large"]:
            return self.config.model_name  # Use original large model
        elif available_memory_mb >= self.config.memory_thresholds["distilbert"]:
            return self.config.fallback_model
        elif available_memory_mb >= self.config.memory_thresholds["mobilebert"]:
            return self.config.tiny_model_alternatives[0]  # MobileBERT
        elif available_memory_mb >= self.config.memory_thresholds["tinybert"]:
            return self.config.tiny_model_alternatives[1]  # TinyBERT
        else:
            return self.config.tiny_model_alternatives[2]  # bert-tiny
    
    def _initialize_pipeline(self) -> Optional[Any]:
        """Initialize the transformers pipeline with optimization."""
        if self._initialization_failed:
            self.logger.debug("Skipping initialization due to previous failure")
            return None
        
        try:
            # Auto-select model if enabled
            if self.config.auto_select_model:
                selected_model = self._select_optimal_model()
                if selected_model != self.config.model_name:
                    self.logger.info(f"Auto-selected model: {selected_model}")
                    self.config.model_name = selected_model
                    # Adjust quantization based on model selection
                    if "tiny" in selected_model.lower():
                        self.config.use_4bit_quantization = True
                        self.config.quantization_bits = 4
            
            self.logger.info(f"Initializing pipeline for {self.config.model_name}")
            
            # Prepare model arguments
            model_kwargs = self._get_model_kwargs()
            
            # Try to initialize with optimizations
            try:
                pipeline_obj = self._create_optimized_pipeline(model_kwargs)
                if pipeline_obj is not None:
                    self._monitor_memory_usage()
                    return pipeline_obj
            except Exception as e:
                self.logger.warning(f"Optimized pipeline initialization failed: {e}")
            
            # Fallback to basic pipeline
            if self.config.use_lightweight_fallback:
                self.logger.info("Attempting fallback to basic pipeline")
                try:
                    pipeline_obj = self._create_basic_pipeline()
                    if pipeline_obj is not None:
                        self._monitor_memory_usage()
                        return pipeline_obj
                except Exception as e:
                    self.logger.error(f"Basic pipeline initialization failed: {e}")
            
            self._initialization_failed = True
            return None
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed completely: {e}")
            self._initialization_failed = True
            return None
    
    def _create_optimized_pipeline(self, model_kwargs: Dict[str, Any]) -> Optional[Any]:
        """Create pipeline with memory optimizations."""
        try:
            # Check if we need accelerate for device_map
            device_map_enabled = self.config.device_map != "cpu" and self.config.device_map is not None
            
            if device_map_enabled:
                try:
                    import accelerate
                except ImportError:
                    self.logger.warning("Accelerate not available, falling back to CPU-only mode")
                    device_map_enabled = False
            
            # Use quantization if enabled
            if self.config.use_quantization and self.config.quantization_bits in [4, 8]:
                if self.config.use_4bit_quantization or self.config.quantization_bits == 4:
                    # 4-bit quantization configuration
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype, torch.float16),
                        bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                        bnb_4bit_quant_type=self.config.bnb_4bit_quant_type
                    )
                    model_kwargs['quantization_config'] = quantization_config
                    self.logger.info("Using 4-bit quantization with double quantization")
                elif self.config.quantization_bits == 8 and device_map_enabled:
                    # 8-bit quantization configuration
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_4bit_compute_dtype=getattr(torch, self.config.torch_dtype, torch.bfloat16)
                    )
                    model_kwargs['quantization_config'] = quantization_config
                    self.logger.info("Using 8-bit quantization")
            
            # Set torch dtype
            if hasattr(torch, self.config.torch_dtype):
                model_kwargs['torch_dtype'] = getattr(torch, self.config.torch_dtype)
            
            # Create pipeline with or without device_map
            if device_map_enabled:
                pipeline_obj = pipeline(
                    self.config.task,
                    model=self.config.model_name,
                    tokenizer=self.config.model_name,
                    aggregation_strategy=self.config.aggregation_strategy,
                    device_map=self.config.device_map,
                    model_kwargs=model_kwargs
                )
                self.logger.info("Successfully created optimized pipeline with device mapping")
            else:
                # Simple pipeline without device mapping
                pipeline_obj = pipeline(
                    self.config.task,
                    model=self.config.model_name,
                    tokenizer=self.config.model_name,
                    aggregation_strategy=self.config.aggregation_strategy
                    # No device_map or model_kwargs to avoid dependency issues
                )
                self.logger.info("Successfully created optimized pipeline (CPU-only)")
            
            return pipeline_obj
            
        except Exception as e:
            self.logger.warning(f"Optimized pipeline creation failed: {e}")
            return None
    
    def _create_basic_pipeline(self) -> Optional[Any]:
        """Create basic pipeline without heavy optimizations."""
        try:
            # Use fallback model if configured
            model_name = self.config.fallback_model if self.config.use_lightweight_fallback else self.config.model_name
            
            # Create pipeline without device_map to avoid accelerate requirement
            pipeline_obj = pipeline(
                self.config.task,
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy=self.config.aggregation_strategy
                # No device_map to avoid accelerate dependency
            )
            
            self.logger.info(f"Successfully created basic pipeline with {model_name}")
            return pipeline_obj
            
        except Exception as e:
            self.logger.error(f"Basic pipeline creation failed: {e}")
            return None
    
    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get model kwargs for optimization."""
        kwargs = {
            'low_cpu_mem_usage': self.config.low_cpu_mem_usage,
        }
        
        if self.config.max_memory:
            kwargs['max_memory'] = self.config.max_memory
        
        if self.config.offload_to_cpu:
            kwargs['offload_folder'] = '/tmp/model_offload'
        
        return kwargs
    
    def _monitor_memory_usage(self):
        """Monitor memory usage after model loading."""
        try:
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                self._memory_usage_mb = memory_mb
                self.logger.info(f"GPU memory usage: {memory_mb:.1f} MB")
                
                if memory_mb > self.config.max_memory_threshold_mb * 5:  # 5x threshold for warning
                    self.logger.warning(f"High GPU memory usage detected: {memory_mb:.1f} MB")
            else:
                # Estimate CPU memory (basic approximation)
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self._memory_usage_mb = memory_mb
                self.logger.info(f"Process memory usage: {memory_mb:.1f} MB")
                
        except Exception as e:
            self.logger.debug(f"Memory monitoring failed: {e}")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self._memory_usage_mb
    
    def cleanup(self):
        """Clean up model resources."""
        try:
            if self._pipeline is not None:
                del self._pipeline
                self._pipeline = None
            
            if self._model is not None:
                del self._model
                self._model = None
            
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            
            # Clear from cache
            cache_key = f"{self.config.model_name}_{self.config.task}"
            with self._cache_lock:
                if cache_key in self._model_cache:
                    del self._model_cache[cache_key]
            
            # Force garbage collection
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Model resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    @classmethod
    def cleanup_all(cls):
        """Clean up all model manager instances."""
        with cls._lock:
            for instance in cls._instances.values():
                instance.cleanup()
            cls._instances.clear()
        
        with cls._cache_lock:
            cls._model_cache.clear()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @contextmanager
    def temporary_pipeline(self):
        """Context manager for temporary pipeline usage."""
        pipeline_obj = None
        try:
            pipeline_obj = self.get_pipeline()
            yield pipeline_obj
        finally:
            # Don't cleanup here as we want to keep cached pipelines
            pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.config.model_name,
            'task': self.config.task,
            'quantization_enabled': self.config.use_quantization,
            'quantization_bits': self.config.quantization_bits,
            'torch_dtype': self.config.torch_dtype,
            'memory_usage_mb': self._memory_usage_mb,
            'initialization_failed': self._initialization_failed,
            'pipeline_available': self._pipeline is not None
        }


# Convenience functions
def get_ner_pipeline(config: Optional[ModelConfig] = None) -> Optional[Any]:
    """
    Get a NER pipeline with default configuration.
    
    Args:
        config: Optional model configuration
        
    Returns:
        NER pipeline or None if unavailable
    """
    if config is None:
        config = ModelConfig(task="ner")
    
    manager = ModelManager(config)
    return manager.get_pipeline()


def get_lightweight_ner_pipeline() -> Optional[Any]:
    """
    Get a lightweight NER pipeline for testing/resource-constrained environments.
    
    Returns:
        Lightweight NER pipeline or None if unavailable
    """
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        # Use a very simple approach without device_map for testing
        pipeline_obj = pipeline(
            "ner",
            model="distilbert-base-uncased",
            tokenizer="distilbert-base-uncased",
            aggregation_strategy="simple"
            # No device_map to avoid accelerate requirement
        )
        return pipeline_obj
    except Exception as e:
        # If that fails, try with the basic BERT model on CPU
        try:
            pipeline_obj = pipeline(
                "ner",
                model="bert-base-uncased",
                tokenizer="bert-base-uncased", 
                aggregation_strategy="simple"
                # No device_map to avoid accelerate requirement
            )
            return pipeline_obj
        except Exception as e2:
            # Last resort: disable transformers pipeline
            return None


def get_ultra_lightweight_ner_pipeline() -> Optional[Any]:
    """
    Get an ultra-lightweight NER pipeline with aggressive memory optimization.
    
    Uses 4-bit quantization and tiny models to achieve <30MB memory usage.
    
    Returns:
        Ultra-lightweight NER pipeline or None if unavailable
    """
    config = ModelConfig(
        model_name="prajjwal1/bert-tiny",  # ~17MB base model
        task="ner",
        use_quantization=True,
        quantization_bits=4,
        use_4bit_quantization=True,
        torch_dtype="float16",
        low_cpu_mem_usage=True,
        auto_select_model=False,  # Don't auto-select, use tiny directly
        max_memory_threshold_mb=30
    )
    
    manager = ModelManager(config)
    return manager.get_pipeline()


def get_memory_optimized_config(target_memory_mb: int = 50) -> ModelConfig:
    """
    Get a model configuration optimized for a specific memory target.
    
    Args:
        target_memory_mb: Target memory usage in MB
        
    Returns:
        Optimized ModelConfig
    """
    if target_memory_mb < 30:
        # Ultra-lightweight: 4-bit tiny model
        return ModelConfig(
            model_name="prajjwal1/bert-tiny",
            use_quantization=True,
            quantization_bits=4,
            use_4bit_quantization=True,
            torch_dtype="float16",
            auto_select_model=False
        )
    elif target_memory_mb < 60:
        # Lightweight: 4-bit TinyBERT
        return ModelConfig(
            model_name="huawei-noah/TinyBERT_General_4L_312D",
            use_quantization=True,
            quantization_bits=4,
            use_4bit_quantization=True,
            torch_dtype="float16",
            auto_select_model=False
        )
    elif target_memory_mb < 100:
        # Mobile-optimized: 8-bit MobileBERT
        return ModelConfig(
            model_name="google/mobilebert-uncased",
            use_quantization=True,
            quantization_bits=8,
            torch_dtype="bfloat16",
            auto_select_model=False
        )
    else:
        # Standard: 8-bit DistilBERT with auto-selection
        return ModelConfig(
            use_quantization=True,
            quantization_bits=8,
            auto_select_model=True,
            max_memory_threshold_mb=target_memory_mb
        )


def cleanup_all_models():
    """Clean up all cached models and free memory."""
    ModelManager.cleanup_all()