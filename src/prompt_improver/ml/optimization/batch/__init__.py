"""Unified Batch Processing with 2025 Best Practices

Modern batch processing system with pluggable strategies, automatic optimization,
and comprehensive 2025 design patterns including Strategy, Factory, and Protocol-based design.

Features:
- Unified interface for all batch processing needs
- Automatic strategy selection based on data characteristics  
- Memory-efficient streaming for large datasets
- Dynamic batch size optimization
- Circuit breaker and retry patterns
- Modern asyncio with TaskGroup
- Clean, legacy-free implementation
"""
from .unified_batch_processor import ProcessingResult, ProcessingStrategy, UnifiedBatchConfig, UnifiedBatchProcessor, UnifiedMetrics, batch_processor, create_batch_processor, create_optimized_processor, create_streaming_processor
from .unified_batch_processor import DataCharacteristics
__all__ = ['UnifiedBatchProcessor', 'UnifiedBatchConfig', 'ProcessingStrategy', 'ProcessingResult', 'DataCharacteristics', 'UnifiedMetrics', 'create_batch_processor', 'create_streaming_processor', 'create_optimized_processor', 'batch_processor']
