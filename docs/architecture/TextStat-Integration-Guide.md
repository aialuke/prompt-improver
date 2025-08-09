# TextStat Integration Guide

**Version**: 1.0.0  
**Last Updated**: 2025-01-09  
**Related ADR**: [ADR-008: TextStat Wrapper Architecture](./ADR-008-TextStat-Wrapper-Architecture.md)

## Quick Start

### Simple Usage (Recommended)
```python
from prompt_improver.core.textstat_config import text_analysis

# Analyze any text with comprehensive metrics
result = text_analysis("Your prompt text here")
print(f"Readability: {result['flesch_reading_ease']}")
print(f"Grade Level: {result['flesch_kincaid_grade']}")
print(f"Syllables: {result['syllable_count']}")
```

### Advanced Configuration
```python
from prompt_improver.core.textstat_config import get_textstat_wrapper, TextStatConfig

# Custom configuration
config = TextStatConfig(
    language="en_US",           # CMUdict optimization
    enable_caching=True,        # 4958x speedup
    suppress_warnings=True,     # Clean logs
    enable_metrics=True         # Performance tracking
)

wrapper = get_textstat_wrapper(config)
score = wrapper.flesch_reading_ease("Your text")
```

## Integration Patterns

### In Async Contexts (LinguisticAnalyzer pattern)
```python
async def analyze_readability(self, text: str) -> dict:
    """Use in async contexts with executor for thread safety."""
    textstat_wrapper = get_textstat_wrapper()
    
    loop = asyncio.get_event_loop()
    analysis_result = await loop.run_in_executor(
        self.executor, 
        textstat_wrapper.comprehensive_analysis, 
        text
    )
    return analysis_result
```

### In Sync Contexts (PromptAnalyzer pattern)
```python
def calculate_readability_score(self, prompt: str) -> float:
    """Direct usage in synchronous contexts."""
    textstat_wrapper = get_textstat_wrapper()
    flesch_score = textstat_wrapper.flesch_reading_ease(prompt)
    return max(0.0, min(1.0, flesch_score / 100.0))
```

## Available Methods

### Core Readability Metrics
- `flesch_reading_ease(text)`: Flesch Reading Ease (0-100, higher = easier)
- `flesch_kincaid_grade(text)`: Flesch-Kincaid Grade Level
- `gunning_fog(text)`: Gunning Fog Index
- `smog_index(text)`: SMOG Index
- `coleman_liau_index(text)`: Coleman-Liau Index
- `automated_readability_index(text)`: ARI

### Text Analysis
- `syllable_count(text)`: Syllable count (100% accuracy with CMUdict)
- `sentence_count(text)`: Number of sentences
- `lexicon_count(text, removepunct=True)`: Word count

### Comprehensive Analysis
- `comprehensive_analysis(text)`: All metrics in single call (recommended)

### Management
- `health_check()`: System health and status
- `get_metrics()`: Performance statistics
- `clear_cache()`: Clear all caches

## Configuration Options

### TextStatConfig Parameters
```python
class TextStatConfig(BaseModel):
    language: str = "en_US"                    # CMUdict language
    enable_caching: bool = True                # LRU caching
    cache_size: int = 1024                     # Cache entries
    suppress_warnings: bool = True             # pkg_resources warnings
    enable_metrics: bool = True                # Performance tracking
```

### Performance Tuning
- **Cache Size**: Adjust based on text variety (default: 1024)
- **Language**: Use "en_US" for maximum syllable accuracy
- **Metrics**: Disable in high-performance scenarios if needed

## Monitoring and Health

### Health Check
```python
wrapper = get_textstat_wrapper()
health = wrapper.health_check()

print(f"Status: {health['status']}")           # healthy/unhealthy
print(f"Language: {health['language']}")       # en_US
print(f"Initialized: {health['initialized']}")  # True/False
```

### Performance Metrics
```python
metrics = wrapper.get_metrics()
print(f"Operations: {metrics['metrics']['total_operations']}")
print(f"Cache hit rate: {wrapper.metrics.cache_hit_rate:.1f}%")
print(f"Avg response: {wrapper.metrics.avg_response_time_ms:.2f}ms")
```

## Migration Guide

### From Direct TextStat Usage

**Before**:
```python
import textstat

def analyze_text(text):
    return {
        'flesch': textstat.flesch_reading_ease(text),
        'syllables': textstat.syllable_count(text)
    }
```

**After**:
```python
from prompt_improver.core.textstat_config import get_textstat_wrapper

def analyze_text(text):
    wrapper = get_textstat_wrapper()
    return {
        'flesch': wrapper.flesch_reading_ease(text),
        'syllables': wrapper.syllable_count(text)
    }
```

### Benefits of Migration
- ✅ **No pkg_resources warnings** during operations
- ✅ **4958x performance improvement** through caching
- ✅ **100% syllable accuracy** with CMUdict
- ✅ **Thread-safe operations** with proper warning suppression
- ✅ **Comprehensive monitoring** and health checks

## Error Handling

The wrapper provides graceful error handling with sensible defaults:

```python
# Handles empty text, errors, etc.
result = wrapper.comprehensive_analysis("")
# Returns safe defaults with error information

# Individual methods have fallbacks
score = wrapper.flesch_reading_ease("invalid")  # Returns 50.0
count = wrapper.syllable_count("error")         # Returns word count estimate
```

## Best Practices

### 1. Use Comprehensive Analysis
```python
# Preferred: Single call for multiple metrics
result = wrapper.comprehensive_analysis(text)

# Avoid: Multiple individual calls
flesch = wrapper.flesch_reading_ease(text)
grade = wrapper.flesch_kincaid_grade(text)
syllables = wrapper.syllable_count(text)
```

### 2. Configure Once, Use Everywhere
```python
# Set up configuration early in application lifecycle
config = TextStatConfig(language="en_US", enable_caching=True)
wrapper = get_textstat_wrapper(config)

# Reuse the configured instance throughout your application
```

### 3. Monitor Performance
```python
# Periodically check performance metrics
if wrapper.metrics.error_rate > 5.0:
    logger.warning(f"High TextStat error rate: {wrapper.metrics.error_rate:.1f}%")
```

### 4. Handle Async Contexts Properly
```python
# Use executor for async contexts to maintain thread safety
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, wrapper.comprehensive_analysis, text)
```

## Troubleshooting

### Common Issues

**Issue**: pkg_resources warnings still appearing  
**Solution**: Warnings on first import are expected; operation warnings should be suppressed

**Issue**: Poor cache performance  
**Solution**: Increase cache size or check for text variations

**Issue**: Inaccurate syllable counts  
**Solution**: Ensure language is set to "en_US" for CMUdict accuracy

### Debug Information
```python
# Check wrapper status
health = wrapper.health_check()
if health['status'] != 'healthy':
    print(f"Issue: {health.get('error', 'Unknown')}")

# Examine cache performance
cache_rate = wrapper.metrics.cache_hit_rate
if cache_rate < 50.0:
    print(f"Low cache hit rate: {cache_rate:.1f}%")
```

## Related Documentation

- [ADR-008: TextStat Wrapper Architecture](./ADR-008-TextStat-Wrapper-Architecture.md)
- [Performance Optimization Guide](../performance/README.md)
- [API Reference](../api/README.md)
- [Testing Guidelines](../testing/README.md)