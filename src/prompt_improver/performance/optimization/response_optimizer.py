"""Response optimization for minimal latency and payload size.

This module implements 2025 best practices for response optimization:
- High-performance JSON serialization with orjson
- Response compression with multiple algorithms
- Payload size reduction techniques
- Streaming responses for large data
"""
import gzip
import json
import logging
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
import lz4.frame
from prompt_improver.performance.optimization.performance_optimizer import measure_mcp_operation
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False
    orjson = None
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
        return (self.original_size - self.compressed_size) / self.original_size * 100

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

    def __init__(self, use_orjson: bool=True):
        self.use_orjson = use_orjson and HAS_ORJSON
        if self.use_orjson:
            logger.info('Using orjson for high-performance JSON serialization')
        else:
            logger.info('Using standard json library (consider installing orjson for better performance)')

    def serialize(self, data: Any) -> bytes:
        """Serialize data to JSON bytes with optimal performance."""
        if self.use_orjson:
            return orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS)
        return json.dumps(data, separators=(',', ':'), ensure_ascii=False, default=self._json_default).encode('utf-8')

    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to Python objects."""
        if self.use_orjson:
            return orjson.loads(data)
        return json.loads(data.decode('utf-8'))

    def _json_default(self, obj: Any) -> Any:
        """Default serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

class EnhancedResponseCompressor:
    """Enhanced multi-algorithm response compression with 2025 best practices."""

    def __init__(self):
        self.algorithms = {'gzip': self._compress_gzip, 'deflate': self._compress_deflate, 'lz4': self._compress_lz4, 'brotli': self._compress_brotli, 'zstd': self._compress_zstd, 'none': self._compress_none}
        self.size_thresholds = {'tiny': 256, 'small': 1024, 'medium': 10240, 'large': 102400, 'huge': 1048576}
        self.content_type_preferences = {'application/json': 'brotli', 'text/html': 'brotli', 'text/css': 'brotli', 'text/javascript': 'brotli', 'text/plain': 'gzip', 'application/xml': 'gzip', 'image/svg+xml': 'brotli', 'binary': 'lz4', 'default': 'gzip'}

    def compress(self, data: bytes, algorithm: str | None=None, min_size: int | None=None, content_type: str | None=None) -> CompressionResult:
        """Compress data with enhanced algorithm selection and content awareness.

        2025 Best Practice: Adaptive compression thresholds based on content type
        and intelligent algorithm selection for optimal performance.
        """
        if min_size is None:
            min_size = self._get_adaptive_min_size(content_type, len(data))
        if len(data) < min_size and (not self._should_force_compression(content_type, len(data))):
            return CompressionResult(compressed_data=data, original_size=len(data), compressed_size=len(data), compression_ratio=1.0, algorithm='none', compression_time_ms=0.0)
        if algorithm is None:
            algorithm = self._select_optimal_algorithm(data, content_type)
        if algorithm not in self.algorithms:
            algorithm = 'gzip'
        start_time = time.perf_counter()
        compressed_data = self.algorithms[algorithm](data)
        compression_time = (time.perf_counter() - start_time) * 1000
        return CompressionResult(compressed_data=compressed_data, original_size=len(data), compressed_size=len(compressed_data), compression_ratio=len(compressed_data) / len(data), algorithm=algorithm, compression_time_ms=compression_time)

    def _select_optimal_algorithm(self, data: bytes, content_type: str | None=None) -> str:
        """Enhanced algorithm selection based on data characteristics and content type."""
        size = len(data)
        if content_type:
            preferred = self.content_type_preferences.get(content_type, 'default')
            if preferred != 'default':
                if preferred == 'brotli' and HAS_BROTLI:
                    return 'brotli'
                if preferred == 'zstd' and HAS_ZSTD:
                    return 'zstd'
                if preferred in self.algorithms:
                    return preferred
        if size < self.size_thresholds['tiny']:
            return 'none'
        if size < self.size_thresholds['small']:
            return 'lz4' if 'lz4' in self.algorithms else 'deflate'
        if size < self.size_thresholds['medium']:
            return 'brotli' if HAS_BROTLI else 'gzip'
        if size < self.size_thresholds['large']:
            return 'brotli' if HAS_BROTLI else 'gzip'
        return 'zstd' if HAS_ZSTD else 'brotli' if HAS_BROTLI else 'gzip'

    def _detect_content_type(self, data: bytes) -> str:
        """Detect content type from data for content-aware optimization."""
        try:
            text = data.decode('utf-8')
            if text.strip().startswith('{') or text.strip().startswith('['):
                return 'application/json'
            if '<html' in text.lower() or '<!doctype html' in text.lower():
                return 'text/html'
            if text.strip().startswith('<?xml'):
                return 'application/xml'
            return 'text/plain'
        except UnicodeDecodeError:
            return 'binary'

    def _get_adaptive_min_size(self, content_type: str | None, data_size: int) -> int:
        """Get adaptive minimum compression size based on content type - 2025 best practice"""
        if content_type is None:
            return 50
        adaptive_thresholds = {'application/json': 30, 'text/html': 40, 'text/css': 35, 'text/javascript': 40, 'text/plain': 60, 'application/xml': 45, 'image/svg+xml': 35, 'binary': 200, 'default': 50}
        base_threshold = adaptive_thresholds.get(content_type, adaptive_thresholds['default'])
        if data_size > 10000:
            return min(base_threshold, 20)
        if data_size > 1000:
            return base_threshold
        return max(base_threshold, 30)

    def _should_force_compression(self, content_type: str | None, data_size: int) -> bool:
        """Determine if compression should be forced regardless of size - 2025 enhancement"""
        if content_type in ['application/json', 'text/html', 'text/css', 'application/xml']:
            return data_size > 20
        return False

    def _compress_gzip(self, data: bytes) -> bytes:
        """Compress using gzip (best compression ratio)."""
        return gzip.compress(data, compresslevel=6)

    def _compress_deflate(self, data: bytes) -> bytes:
        """Compress using deflate (good compression, fast)."""
        return zlib.compress(data, level=6)

    def _compress_lz4(self, data: bytes) -> bytes:
        """Compress using LZ4 (fastest compression)."""
        return lz4.frame.compress(data, compression_level=1)

    def _compress_brotli(self, data: bytes) -> bytes:
        """Compress using Brotli (excellent compression ratio for text)."""
        if not HAS_BROTLI:
            return self._compress_gzip(data)
        return brotli.compress(data, quality=6)

    def _compress_zstd(self, data: bytes) -> bytes:
        """Compress using Zstandard (fast with excellent compression)."""
        if not HAS_ZSTD:
            return self._compress_gzip(data)
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
        self.content_optimizers = {'application/json': self._optimize_json_content, 'text/html': self._optimize_html_content, 'text/css': self._optimize_css_content, 'text/javascript': self._optimize_js_content, 'default': self._optimize_generic_content}

    def optimize_response(self, data: Any, compression_algorithm: str | None=None, include_metadata: bool=True, content_type: str | None=None) -> dict[str, Any]:
        """Enhanced response optimization with content-aware strategies."""
        start_time = time.perf_counter()
        if content_type is None:
            content_type = self._detect_content_type_from_data(data)
        content_optimizer = self.content_optimizers.get(content_type, self.content_optimizers['default'])
        optimized_data = content_optimizer(data)
        minimized_data = self._minimize_payload(optimized_data)
        serialized_data = self.json_serializer.serialize(minimized_data)
        compression_result = self.compressor.compress(serialized_data, algorithm=compression_algorithm, content_type=content_type)
        optimization_time = (time.perf_counter() - start_time) * 1000
        result = {'data': compression_result.compressed_data, 'compressed': compression_result.algorithm != 'none', 'compression_algorithm': compression_result.algorithm}
        if include_metadata:
            result['optimization_metadata'] = {'original_size_bytes': compression_result.original_size, 'compressed_size_bytes': compression_result.compressed_size, 'compression_ratio': compression_result.compression_ratio, 'size_reduction_percent': compression_result.size_reduction_percent, 'optimization_time_ms': optimization_time, 'serializer': 'orjson' if self.json_serializer.use_orjson else 'json'}
        return result

    def _minimize_payload(self, data: Any) -> Any:
        """Minimize payload size by removing unnecessary data."""
        if isinstance(data, dict):
            minimized = {}
            for key, value in data.items():
                if value is not None:
                    if isinstance(value, (list, dict)) and len(value) == 0:
                        continue
                    minimized[key] = self._minimize_payload(value)
            return minimized
        if isinstance(data, list):
            return [self._minimize_payload(item) for item in data if item is not None]
        return data

    def create_streaming_response(self, data_generator, chunk_size: int=8192):
        """Create a streaming response for large datasets."""

        async def stream_chunks():
            async for chunk in data_generator:
                optimized_chunk = self.optimize_response(chunk, include_metadata=False)
                yield optimized_chunk['data']
        return stream_chunks()

    def _detect_content_type_from_data(self, data: Any) -> str:
        """Detect content type from data structure"""
        if isinstance(data, dict) or isinstance(data, (list, tuple)):
            return 'application/json'
        if isinstance(data, str):
            if data.strip().startswith('<'):
                return 'text/html'
            return 'text/plain'
        return 'application/json'

    def _optimize_json_content(self, data: Any) -> Any:
        """Optimize JSON content for better compression"""
        if isinstance(data, dict):
            optimized = {}
            for key, value in data.items():
                if value is not None and value != [] and (value != {}):
                    if isinstance(value, (dict, list)):
                        optimized_value = self._optimize_json_content(value)
                        if optimized_value:
                            optimized[key] = optimized_value
                    else:
                        optimized[key] = value
            return optimized
        if isinstance(data, list):
            return [self._optimize_json_content(item) for item in data if item is not None]
        return data

    def _optimize_html_content(self, data: Any) -> Any:
        """Optimize HTML content (basic minification)"""
        if isinstance(data, str):
            import re
            data = re.sub('\\s+', ' ', data)
            data = re.sub('>\\s+<', '><', data)
            data = re.sub('<!--.*?-->', '', data, flags=re.DOTALL)
        return data

    def _optimize_css_content(self, data: Any) -> Any:
        """Optimize CSS content (basic minification)"""
        if isinstance(data, str):
            import re
            data = re.sub('/\\*.*?\\*/', '', data, flags=re.DOTALL)
            data = re.sub('\\s+', ' ', data)
            data = re.sub('\\s*([{}:;,])\\s*', '\\1', data)
        return data

    def _optimize_js_content(self, data: Any) -> Any:
        """Optimize JavaScript content (basic minification)"""
        if isinstance(data, str):
            import re
            data = re.sub('//.*$', '', data, flags=re.MULTILINE)
            data = re.sub('/\\*.*?\\*/', '', data, flags=re.DOTALL)
            data = re.sub('\\s+', ' ', data)
        return data

    def _optimize_generic_content(self, data: Any) -> Any:
        """Generic content optimization"""
        return data

class ResponseOptimizer:
    """Enhanced response optimization coordinator with 2025 best practices."""

    def __init__(self):
        self.payload_optimizer = EnhancedPayloadOptimizer()
        self._optimization_stats = {'total_responses': 0, 'total_bytes_saved': 0, 'total_optimization_time_ms': 0, 'compression_algorithm_usage': {}}

    async def optimize_mcp_response(self, response_data: Any, operation_name: str='mcp_response', enable_compression: bool=True) -> dict[str, Any]:
        """Optimize MCP response with comprehensive optimization."""
        async with measure_mcp_operation(f'optimize_{operation_name}') as perf_metrics:
            start_time = time.perf_counter()
            try:
                optimized = self.payload_optimizer.optimize_response(response_data, compression_algorithm='lz4' if enable_compression else 'none')
                optimization_time = (time.perf_counter() - start_time) * 1000
                self._update_stats(optimized, optimization_time)
                perf_metrics.metadata.update({'original_size': optimized.get('optimization_metadata', {}).get('original_size_bytes', 0), 'compressed_size': optimized.get('optimization_metadata', {}).get('compressed_size_bytes', 0), 'compression_algorithm': optimized.get('compression_algorithm', 'none'), 'optimization_time_ms': optimization_time})
                return optimized
            except Exception as e:
                logger.error('Response optimization failed: %s', e)
                return {'data': self.payload_optimizer.json_serializer.serialize(response_data), 'compressed': False, 'compression_algorithm': 'none', 'error': str(e)}

    def _update_stats(self, optimized_response: dict[str, Any], optimization_time: float):
        """Update optimization statistics."""
        self._optimization_stats['total_responses'] += 1
        self._optimization_stats['total_optimization_time_ms'] += optimization_time
        metadata = optimized_response.get('optimization_metadata', {})
        if metadata:
            bytes_saved = metadata.get('original_size_bytes', 0) - metadata.get('compressed_size_bytes', 0)
            self._optimization_stats['total_bytes_saved'] += bytes_saved
            algorithm = optimized_response.get('compression_algorithm', 'none')
            self._optimization_stats['compression_algorithm_usage'][algorithm] = self._optimization_stats['compression_algorithm_usage'].get(algorithm, 0) + 1

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get comprehensive optimization statistics."""
        total_responses = self._optimization_stats['total_responses']
        if total_responses == 0:
            return {'message': 'No responses optimized yet'}
        return {'total_responses_optimized': total_responses, 'total_bytes_saved': self._optimization_stats['total_bytes_saved'], 'avg_bytes_saved_per_response': self._optimization_stats['total_bytes_saved'] / total_responses, 'avg_optimization_time_ms': self._optimization_stats['total_optimization_time_ms'] / total_responses, 'compression_algorithm_usage': self._optimization_stats['compression_algorithm_usage'], 'orjson_available': HAS_ORJSON}
_global_optimizer: ResponseOptimizer | None = None

def get_response_optimizer() -> ResponseOptimizer:
    """Get the global response optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = ResponseOptimizer()
    return _global_optimizer

async def optimize_mcp_response(response_data: Any, operation_name: str='mcp_response', enable_compression: bool=True) -> dict[str, Any]:
    """Optimize an MCP response for minimal latency."""
    optimizer = get_response_optimizer()
    return await optimizer.optimize_mcp_response(response_data, operation_name, enable_compression)
