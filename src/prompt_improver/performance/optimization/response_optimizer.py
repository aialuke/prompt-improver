"""Response optimization for minimal latency and payload size.

This module implements 2025 best practices for response optimization:
- High-performance JSON serialization with orjson
- Response compression with multiple algorithms
- Payload size reduction techniques
- Streaming responses for large data
"""

import asyncio
import gzip
import json
import logging
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import lz4.frame

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False
    orjson = None

# Enhanced 2025 compression support
try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False
    brotli = None

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    zstd = None

from .performance_optimizer import measure_mcp_operation

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of compression operation with 2025 enhancements."""
    compressed_data: bytes
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm: str
    compression_time_ms: float

    @property
    def size_reduction_percent(self) -> float:
        """Calculate size reduction percentage"""
        if self.original_size == 0:
            return 0.0
        return ((self.original_size - self.compressed_size) / self.original_size) * 100

    @property
    def size_reduction_bytes(self) -> int:
        """Calculate size reduction in bytes"""
        return self.original_size - self.compressed_size
    
    @property
    def size_reduction_percent(self) -> float:
        """Calculate size reduction percentage."""
        return (1 - self.compressed_size / self.original_size) * 100


class FastJSONSerializer:
    """High-performance JSON serialization with multiple backends."""
    
    def __init__(self, use_orjson: bool = True):
        self.use_orjson = use_orjson and HAS_ORJSON
        if self.use_orjson:
            logger.info("Using orjson for high-performance JSON serialization")
        else:
            logger.info("Using standard json library (consider installing orjson for better performance)")
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to JSON bytes with optimal performance."""
        if self.use_orjson:
            # orjson is significantly faster than standard json
            return orjson.dumps(
                data,
                option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS
            )
        else:
            # Fallback to standard json with optimizations
            return json.dumps(
                data,
                separators=(',', ':'),  # Compact format
                ensure_ascii=False,
                default=self._json_default
            ).encode('utf-8')
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to Python objects."""
        if self.use_orjson:
            return orjson.loads(data)
        else:
            return json.loads(data.decode('utf-8'))
    
    def _json_default(self, obj: Any) -> Any:
        """Default serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


class EnhancedResponseCompressor:
    """Enhanced multi-algorithm response compression with 2025 best practices."""

    def __init__(self):
        self.algorithms = {
            'gzip': self._compress_gzip,
            'deflate': self._compress_deflate,
            'lz4': self._compress_lz4,
            'brotli': self._compress_brotli,
            'zstd': self._compress_zstd,
            'none': self._compress_none
        }

        # Enhanced performance thresholds for algorithm selection
        self.size_thresholds = {
            'tiny': 256,        # < 256B - no compression
            'small': 1024,      # 256B-1KB - ultra-fast compression
            'medium': 10240,    # 1-10KB - fast compression
            'large': 102400,    # 10-100KB - balanced compression
            'huge': 1048576     # > 100KB - best compression
        }

        # Content-aware compression settings
        self.content_type_preferences = {
            'application/json': 'brotli',
            'text/html': 'brotli',
            'text/css': 'brotli',
            'text/javascript': 'brotli',
            'text/plain': 'gzip',
            'application/xml': 'gzip',
            'image/svg+xml': 'brotli',
            'binary': 'lz4',
            'default': 'gzip'
        }
    
    def compress(
        self,
        data: bytes,
        algorithm: Optional[str] = None,
        min_size: Optional[int] = None,
        content_type: Optional[str] = None
    ) -> CompressionResult:
        """Compress data with enhanced algorithm selection and content awareness.

        2025 Best Practice: Adaptive compression thresholds based on content type
        and intelligent algorithm selection for optimal performance.
        """
        # Adaptive minimum size based on content type (2025 best practice)
        if min_size is None:
            min_size = self._get_adaptive_min_size(content_type, len(data))

        # Check if compression should be skipped or forced (2025 enhancement)
        if len(data) < min_size and not self._should_force_compression(content_type, len(data)):
            # Don't compress very small payloads unless content type benefits
            return CompressionResult(
                compressed_data=data,
                original_size=len(data),
                compressed_size=len(data),
                compression_ratio=1.0,
                algorithm='none',
                compression_time_ms=0.0
            )

        # Auto-select algorithm if not specified
        if algorithm is None:
            algorithm = self._select_optimal_algorithm(data, content_type)

        # Fallback if algorithm not available
        if algorithm not in self.algorithms:
            algorithm = 'gzip'

        start_time = time.perf_counter()
        compressed_data = self.algorithms[algorithm](data)
        compression_time = (time.perf_counter() - start_time) * 1000

        return CompressionResult(
            compressed_data=compressed_data,
            original_size=len(data),
            compressed_size=len(compressed_data),
            compression_ratio=len(compressed_data) / len(data),
            algorithm=algorithm,
            compression_time_ms=compression_time
        )
    
    def _select_optimal_algorithm(self, data: bytes, content_type: Optional[str] = None) -> str:
        """Enhanced algorithm selection based on data characteristics and content type."""
        size = len(data)

        # Content-aware selection first
        if content_type:
            preferred = self.content_type_preferences.get(content_type, 'default')
            if preferred != 'default':
                # Check if preferred algorithm is available
                if preferred == 'brotli' and HAS_BROTLI:
                    return 'brotli'
                elif preferred == 'zstd' and HAS_ZSTD:
                    return 'zstd'
                elif preferred in self.algorithms:
                    return preferred

        # Size-based selection with enhanced thresholds
        if size < self.size_thresholds['tiny']:
            return 'none'
        elif size < self.size_thresholds['small']:
            # Ultra-fast compression for small payloads
            return 'lz4' if 'lz4' in self.algorithms else 'deflate'
        elif size < self.size_thresholds['medium']:
            # Fast compression with good ratio
            return 'brotli' if HAS_BROTLI else 'gzip'
        elif size < self.size_thresholds['large']:
            # Balanced compression
            return 'brotli' if HAS_BROTLI else 'gzip'
        else:
            # Best compression for large payloads
            return 'zstd' if HAS_ZSTD else ('brotli' if HAS_BROTLI else 'gzip')

    def _detect_content_type(self, data: bytes) -> str:
        """Detect content type from data for content-aware optimization."""
        try:
            # Try to decode as text
            text = data.decode('utf-8')

            # Simple heuristics for content type detection
            if text.strip().startswith('{') or text.strip().startswith('['):
                return 'application/json'
            elif '<html' in text.lower() or '<!doctype html' in text.lower():
                return 'text/html'
            elif text.strip().startswith('<?xml'):
                return 'application/xml'
            else:
                return 'text/plain'
        except UnicodeDecodeError:
            # Binary data
            return 'binary'

    def _get_adaptive_min_size(self, content_type: Optional[str], data_size: int) -> int:
        """Get adaptive minimum compression size based on content type - 2025 best practice"""
        if content_type is None:
            return 50  # Default smaller threshold for unknown content

        # Content-aware compression thresholds (2025 optimization)
        adaptive_thresholds = {
            'application/json': 30,    # JSON compresses very well, lower threshold
            'text/html': 40,           # HTML has repetitive tags, compresses well
            'text/css': 35,            # CSS has repetitive selectors
            'text/javascript': 40,     # JS has repetitive patterns
            'text/plain': 60,          # Plain text varies, moderate threshold
            'application/xml': 45,     # XML has repetitive tags
            'image/svg+xml': 35,       # SVG is XML-based, compresses well
            'binary': 200,             # Binary data less predictable
            'default': 50              # Conservative default
        }

        base_threshold = adaptive_thresholds.get(content_type, adaptive_thresholds['default'])

        # Dynamic adjustment based on data size (2025 enhancement)
        if data_size > 10000:  # Large data, always try compression
            return min(base_threshold, 20)
        elif data_size > 1000:  # Medium data, use base threshold
            return base_threshold
        else:  # Small data, be more selective
            return max(base_threshold, 30)

    def _should_force_compression(self, content_type: Optional[str], data_size: int) -> bool:
        """Determine if compression should be forced regardless of size - 2025 enhancement"""
        if content_type in ['application/json', 'text/html', 'text/css', 'application/xml']:
            # These content types compress very well even when small
            return data_size > 20
        return False
    
    def _compress_gzip(self, data: bytes) -> bytes:
        """Compress using gzip (best compression ratio)."""
        return gzip.compress(data, compresslevel=6)  # Balanced speed/compression
    
    def _compress_deflate(self, data: bytes) -> bytes:
        """Compress using deflate (good compression, fast)."""
        return zlib.compress(data, level=6)
    
    def _compress_lz4(self, data: bytes) -> bytes:
        """Compress using LZ4 (fastest compression)."""
        return lz4.frame.compress(data, compression_level=1)

    def _compress_brotli(self, data: bytes) -> bytes:
        """Compress using Brotli (excellent compression ratio for text)."""
        if not HAS_BROTLI:
            # Fallback to gzip
            return self._compress_gzip(data)

        # Brotli quality 6 provides good balance of speed and compression
        return brotli.compress(data, quality=6)

    def _compress_zstd(self, data: bytes) -> bytes:
        """Compress using Zstandard (fast with excellent compression)."""
        if not HAS_ZSTD:
            # Fallback to gzip
            return self._compress_gzip(data)

        # Zstd level 3 provides good balance
        compressor = zstd.ZstdCompressor(level=3)
        return compressor.compress(data)

    def _compress_none(self, data: bytes) -> bytes:
        """No compression (passthrough)."""
        return data


class EnhancedPayloadOptimizer:
    """Enhanced payload optimizer with content-aware optimization and 2025 best practices."""

    def __init__(self):
        self.json_serializer = FastJSONSerializer()
        self.compressor = EnhancedResponseCompressor()

        # Content optimization strategies
        self.content_optimizers = {
            'application/json': self._optimize_json_content,
            'text/html': self._optimize_html_content,
            'text/css': self._optimize_css_content,
            'text/javascript': self._optimize_js_content,
            'default': self._optimize_generic_content
        }
    
    def optimize_response(
        self,
        data: Any,
        compression_algorithm: Optional[str] = None,
        include_metadata: bool = True,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhanced response optimization with content-aware strategies."""
        start_time = time.perf_counter()

        # Step 1: Content-aware optimization
        if content_type is None:
            content_type = self._detect_content_type_from_data(data)

        content_optimizer = self.content_optimizers.get(content_type, self.content_optimizers['default'])
        optimized_data = content_optimizer(data)

        # Step 2: Minimize payload content
        minimized_data = self._minimize_payload(optimized_data)

        # Step 3: Serialize with high-performance JSON
        serialized_data = self.json_serializer.serialize(minimized_data)

        # Step 4: Enhanced compression with content awareness
        compression_result = self.compressor.compress(
            serialized_data,
            algorithm=compression_algorithm,
            content_type=content_type
        )
        
        optimization_time = (time.perf_counter() - start_time) * 1000
        
        result = {
            'data': compression_result.compressed_data,
            'compressed': compression_result.algorithm != 'none',
            'compression_algorithm': compression_result.algorithm
        }
        
        if include_metadata:
            result['optimization_metadata'] = {
                'original_size_bytes': compression_result.original_size,
                'compressed_size_bytes': compression_result.compressed_size,
                'compression_ratio': compression_result.compression_ratio,
                'size_reduction_percent': compression_result.size_reduction_percent,
                'optimization_time_ms': optimization_time,
                'serializer': 'orjson' if self.json_serializer.use_orjson else 'json'
            }
        
        return result
    
    def _minimize_payload(self, data: Any) -> Any:
        """Minimize payload size by removing unnecessary data."""
        if isinstance(data, dict):
            # Remove None values and empty collections
            minimized = {}
            for key, value in data.items():
                if value is not None:
                    if isinstance(value, (list, dict)) and len(value) == 0:
                        continue  # Skip empty collections
                    minimized[key] = self._minimize_payload(value)
            return minimized
        
        elif isinstance(data, list):
            return [self._minimize_payload(item) for item in data if item is not None]
        
        else:
            return data
    
    def create_streaming_response(
        self,
        data_generator,
        chunk_size: int = 8192
    ):
        """Create a streaming response for large datasets."""
        async def stream_chunks():
            async for chunk in data_generator:
                optimized_chunk = self.optimize_response(
                    chunk,
                    include_metadata=False
                )
                yield optimized_chunk['data']
        
        return stream_chunks()

    def _detect_content_type_from_data(self, data: Any) -> str:
        """Detect content type from data structure"""
        if isinstance(data, dict):
            return 'application/json'
        elif isinstance(data, (list, tuple)):
            return 'application/json'
        elif isinstance(data, str):
            if data.strip().startswith('<'):
                return 'text/html'
            else:
                return 'text/plain'
        else:
            return 'application/json'  # Default for serializable objects

    def _optimize_json_content(self, data: Any) -> Any:
        """Optimize JSON content for better compression"""
        if isinstance(data, dict):
            # Remove null values and empty collections
            optimized = {}
            for key, value in data.items():
                if value is not None and value != [] and value != {}:
                    if isinstance(value, (dict, list)):
                        optimized_value = self._optimize_json_content(value)
                        if optimized_value:  # Only include non-empty results
                            optimized[key] = optimized_value
                    else:
                        optimized[key] = value
            return optimized
        elif isinstance(data, list):
            # Remove null values from lists
            return [self._optimize_json_content(item) for item in data if item is not None]
        else:
            return data

    def _optimize_html_content(self, data: Any) -> Any:
        """Optimize HTML content (basic minification)"""
        if isinstance(data, str):
            # Basic HTML minification
            import re
            # Remove extra whitespace
            data = re.sub(r'\s+', ' ', data)
            # Remove whitespace around tags
            data = re.sub(r'>\s+<', '><', data)
            # Remove comments
            data = re.sub(r'<!--.*?-->', '', data, flags=re.DOTALL)
        return data

    def _optimize_css_content(self, data: Any) -> Any:
        """Optimize CSS content (basic minification)"""
        if isinstance(data, str):
            import re
            # Remove comments
            data = re.sub(r'/\*.*?\*/', '', data, flags=re.DOTALL)
            # Remove extra whitespace
            data = re.sub(r'\s+', ' ', data)
            # Remove whitespace around special characters
            data = re.sub(r'\s*([{}:;,])\s*', r'\1', data)
        return data

    def _optimize_js_content(self, data: Any) -> Any:
        """Optimize JavaScript content (basic minification)"""
        if isinstance(data, str):
            import re
            # Remove single-line comments
            data = re.sub(r'//.*$', '', data, flags=re.MULTILINE)
            # Remove multi-line comments
            data = re.sub(r'/\*.*?\*/', '', data, flags=re.DOTALL)
            # Remove extra whitespace
            data = re.sub(r'\s+', ' ', data)
        return data

    def _optimize_generic_content(self, data: Any) -> Any:
        """Generic content optimization"""
        return data


class ResponseOptimizer:
    """Enhanced response optimization coordinator with 2025 best practices."""

    def __init__(self):
        self.payload_optimizer = EnhancedPayloadOptimizer()
        self._optimization_stats = {
            'total_responses': 0,
            'total_bytes_saved': 0,
            'total_optimization_time_ms': 0,
            'compression_algorithm_usage': {}
        }
    
    async def optimize_mcp_response(
        self,
        response_data: Any,
        operation_name: str = "mcp_response",
        enable_compression: bool = True
    ) -> Dict[str, Any]:
        """Optimize MCP response with comprehensive optimization."""
        async with measure_mcp_operation(f"optimize_{operation_name}") as perf_metrics:
            start_time = time.perf_counter()
            
            try:
                # Optimize the response payload
                optimized = self.payload_optimizer.optimize_response(
                    response_data,
                    compression_algorithm='lz4' if enable_compression else 'none'
                )
                
                optimization_time = (time.perf_counter() - start_time) * 1000
                
                # Update statistics
                self._update_stats(optimized, optimization_time)
                
                # Add performance metadata
                perf_metrics.metadata.update({
                    'original_size': optimized.get('optimization_metadata', {}).get('original_size_bytes', 0),
                    'compressed_size': optimized.get('optimization_metadata', {}).get('compressed_size_bytes', 0),
                    'compression_algorithm': optimized.get('compression_algorithm', 'none'),
                    'optimization_time_ms': optimization_time
                })
                
                return optimized
                
            except Exception as e:
                logger.error(f"Response optimization failed: {e}")
                # Fallback to unoptimized response
                return {
                    'data': self.payload_optimizer.json_serializer.serialize(response_data),
                    'compressed': False,
                    'compression_algorithm': 'none',
                    'error': str(e)
                }
    
    def _update_stats(self, optimized_response: Dict[str, Any], optimization_time: float):
        """Update optimization statistics."""
        self._optimization_stats['total_responses'] += 1
        self._optimization_stats['total_optimization_time_ms'] += optimization_time
        
        metadata = optimized_response.get('optimization_metadata', {})
        if metadata:
            bytes_saved = metadata.get('original_size_bytes', 0) - metadata.get('compressed_size_bytes', 0)
            self._optimization_stats['total_bytes_saved'] += bytes_saved
            
            algorithm = optimized_response.get('compression_algorithm', 'none')
            self._optimization_stats['compression_algorithm_usage'][algorithm] = (
                self._optimization_stats['compression_algorithm_usage'].get(algorithm, 0) + 1
            )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        total_responses = self._optimization_stats['total_responses']
        
        if total_responses == 0:
            return {"message": "No responses optimized yet"}
        
        return {
            'total_responses_optimized': total_responses,
            'total_bytes_saved': self._optimization_stats['total_bytes_saved'],
            'avg_bytes_saved_per_response': self._optimization_stats['total_bytes_saved'] / total_responses,
            'avg_optimization_time_ms': self._optimization_stats['total_optimization_time_ms'] / total_responses,
            'compression_algorithm_usage': self._optimization_stats['compression_algorithm_usage'],
            'orjson_available': HAS_ORJSON
        }


# Global response optimizer instance
_global_optimizer: Optional[ResponseOptimizer] = None


def get_response_optimizer() -> ResponseOptimizer:
    """Get the global response optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = ResponseOptimizer()
    return _global_optimizer


# Convenience function
async def optimize_mcp_response(
    response_data: Any,
    operation_name: str = "mcp_response",
    enable_compression: bool = True
) -> Dict[str, Any]:
    """Optimize an MCP response for minimal latency."""
    optimizer = get_response_optimizer()
    return await optimizer.optimize_mcp_response(
        response_data,
        operation_name,
        enable_compression
    )
