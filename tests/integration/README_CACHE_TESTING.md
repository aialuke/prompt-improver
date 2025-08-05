# MultiLevelCache Real Behavior Testing Framework

This comprehensive testing framework provides 100% real behavior testing for the enhanced MultiLevelCache using TestContainers, covering all cache tiers, intelligent warming, and performance characteristics.

## Overview

The testing framework includes:

- **Real Redis Integration**: Uses TestContainers for actual Redis instances
- **Comprehensive Coverage**: Tests L1 (memory), L2 (Redis), and L3 (database) tiers
- **Intelligent Warming Validation**: Tests access pattern tracking and warming algorithms
- **Performance Benchmarking**: Validates SLA compliance and performance characteristics
- **Error Handling Testing**: Validates resilience and fault tolerance
- **Health Monitoring**: Tests health checks and monitoring integration

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install pytest pytest-asyncio testcontainers coredis

# Ensure Docker is running
docker --version
```

### Basic Test Execution

```bash
# Run all integration tests
python scripts/run_cache_tests.py --test-type integration

# Run specific test class
python scripts/run_cache_tests.py --test-class TestBasicCacheOperations

# Run performance tests
python scripts/run_cache_tests.py --test-type performance --verbose

# Run comprehensive benchmark
python scripts/run_cache_tests.py --benchmark --report-file benchmark_results.json
```

### Using pytest directly

```bash
# Run all cache tests
pytest tests/integration/test_multi_level_cache_real_behavior.py -v

# Run specific test class
pytest tests/integration/test_multi_level_cache_real_behavior.py::TestBasicCacheOperations -v

# Run performance tests only
pytest tests/integration/test_multi_level_cache_real_behavior.py -m performance -v

# Run with coverage
pytest tests/integration/test_multi_level_cache_real_behavior.py --cov=src/prompt_improver/utils/multi_level_cache
```

## Test Structure

### Test Classes

1. **TestBasicCacheOperations**: Basic CRUD operations with real Redis
2. **TestMultiLevelBehavior**: L1 → L2 → L3 fallback behavior
3. **TestIntelligentWarming**: Access pattern tracking and warming
4. **TestEnhancedStatistics**: Metrics collection and analysis
5. **TestErrorHandlingResilience**: Error scenarios and recovery
6. **TestPerformanceCharacteristics**: SLA compliance and performance
7. **TestHealthMonitoringIntegration**: Health checks and monitoring
8. **TestSystemIntegration**: End-to-end scenarios
9. **TestPerformanceBenchmarking**: Baseline performance metrics

### Test Infrastructure

- **Container Management**: Automated Redis container lifecycle
- **Test Data Generators**: Realistic test data with various patterns
- **Performance Profiling**: Detailed timing and throughput analysis
- **Mock Services**: L3 database fallback simulation
- **Assertion Helpers**: Domain-specific validation

## Configuration

### Test Scenarios

The framework supports multiple test configurations:

```python
# Default configuration
CacheTestConfig(
    l1_max_size=100,
    l2_default_ttl=300,
    enable_l2=True,
    enable_warming=True
)

# Performance testing
CacheTestConfig(
    l1_max_size=1000,
    warming_interval=30,
    max_warming_keys=50
)

# Memory constrained
CacheTestConfig(
    l1_max_size=10,
    warming_interval=300,
    max_warming_keys=5
)
```

### Performance Thresholds

```python
PerformanceThresholds(
    l1_max_avg_time=0.001,      # 1ms L1 operations
    full_cache_max_p95_time=0.05, # 50ms P95 response time
    min_ops_per_second=1000.0,   # 1000 ops/sec throughput
    min_overall_hit_rate=0.7,    # 70% hit rate
    max_error_rate=0.01          # 1% error rate
)
```

## Real Behavior Testing Features

### TestContainer Integration

```python
@pytest.fixture(scope="session")
def redis_test_container():
    """Real Redis container for behavior testing."""
    with RedisContainer(image="redis:7-alpine") as container:
        # Automatic container lifecycle management
        yield container
```

### Multi-Level Cache Validation

```python
async def test_l3_fallback_behavior(cache, mock_database):
    """Test complete L1 → L2 → L3 fallback chain."""
    # Miss L1 and L2, hit database, populate caches
    result = await cache.get('key', fallback_func=mock_database.get)
    
    # Verify data consistency and cache population
    assert result == expected_data
    assert cache._l1_cache.get('key') == expected_data
```

### Intelligent Warming Testing

```python
async def test_access_pattern_tracking(cache):
    """Test access pattern recording and warming candidates."""
    # Generate access patterns
    for i in range(10):
        await cache.get('hot_key', fallback_func=lambda: f'value_{i}')
    
    # Verify pattern tracking
    assert 'hot_key' in cache._access_patterns
    candidates = await cache.get_warming_candidates()
    assert any(c['key'] == 'hot_key' for c in candidates)
```

### Performance Benchmarking

```python
async def test_response_time_sla_compliance(cache):
    """Validate SLA compliance with real operations."""
    response_times = []
    for i in range(100):
        start = time.perf_counter()
        await cache.set(f'perf_{i}', f'value_{i}')
        await cache.get(f'perf_{i}')
        response_times.append(time.perf_counter() - start)
    
    p95_time = sorted(response_times)[95]
    assert p95_time < 0.05  # 50ms SLA
```

## Advanced Testing Scenarios

### Error Handling and Resilience

```python
async def test_redis_connection_failure_resilience(cache):
    """Test cache behavior when Redis fails."""
    # Force Redis connection failure
    await cache._l2_cache.close()
    
    # Cache should continue working with L1 only
    await cache.set('resilience_test', 'still_works')
    result = await cache.get('resilience_test')
    assert result == 'still_works'
```

### Concurrent Access Testing

```python
async def test_high_concurrency_scenario(cache):
    """Test cache under high concurrency."""
    async def user_simulation(user_id):
        for i in range(100):
            await cache.set(f'user_{user_id}_{i}', f'data_{i}')
            await cache.get(f'user_{user_id}_{i}')
    
    # Run 100 concurrent users
    await asyncio.gather(*[user_simulation(i) for i in range(100)])
    
    # Verify system integrity
    stats = cache.get_performance_stats()
    assert stats['performance_metrics']['error_rate']['overall_error_rate'] == 0
```

### Load Testing Integration

```python
class CacheLoadTester:
    """Structured load testing with performance analysis."""
    
    async def run_concurrent_load_test(self, concurrent_users=10, operations_per_user=100):
        results = await asyncio.gather(*[
            self.user_simulation(i, operations_per_user) 
            for i in range(concurrent_users)
        ])
        return self.analyze_results(results)
```

## Performance Analysis

### Profiling and Metrics

```python
class CachePerformanceProfiler:
    """Detailed performance profiling."""
    
    @asynccontextmanager
    async def profile_operation(self, operation_name):
        start_time = time.perf_counter()
        yield
        duration = time.perf_counter() - start_time
        self.record_measurement(operation_name, duration)
    
    def get_statistics(self, operation_name):
        return {
            'mean': ..., 'p95': ..., 'p99': ...,
            'ops_per_second': ..., 'std_dev': ...
        }
```

### Benchmark Results

The framework generates comprehensive benchmark reports:

```json
{
  "benchmark_suite": "MultiLevelCache Comprehensive Benchmark",
  "timestamp": "2025-01-01 12:00:00",
  "results": {
    "basic_operations": {
      "success": true,
      "execution_time_seconds": 15.42,
      "tests_passed": 12,
      "tests_failed": 0
    },
    "performance_characteristics": {
      "success": true,
      "l1_avg_response_time_ms": 0.5,
      "l2_avg_response_time_ms": 5.2,
      "throughput_ops_per_second": 1250.8
    }
  },
  "summary": {
    "overall_success": true,
    "total_execution_time_seconds": 120.5,
    "success_rate": 1.0
  }
}
```

## Troubleshooting

### Common Issues

1. **Docker not running**: Ensure Docker daemon is active
2. **Port conflicts**: TestContainers handles port allocation automatically
3. **Memory constraints**: Adjust L1 cache size for resource-limited environments
4. **Test timeouts**: Increase timeout for slower systems

### Debug Mode

```bash
# Enable verbose logging
python scripts/run_cache_tests.py --test-type integration --verbose --no-capture

# Run single test for debugging
python scripts/run_cache_tests.py --test-class TestBasicCacheOperations --test-method test_cache_set_and_get_simple_data --verbose
```

### Container Management

```bash
# Check running containers
docker ps

# Clean up test containers
docker container prune

# View container logs
docker logs <container_id>
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Cache Tests
on: [push, pull_request]

jobs:
  cache-tests:
    runs-on: ubuntu-latest
    services:
      docker:
        image: docker:dind
        
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt
          
      - name: Run cache integration tests
        run: |
          python scripts/run_cache_tests.py --test-type integration --report-file test-results.json
          
      - name: Run performance benchmarks
        run: |
          python scripts/run_cache_tests.py --benchmark --report-file benchmark-results.json
          
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: |
            test-results.json
            benchmark-results.json
```

## Contributing

### Adding New Tests

1. **Extend existing test classes** for related functionality
2. **Create new test classes** for distinct feature areas
3. **Use real behavior patterns** - avoid mocks when possible
4. **Include performance validation** for new features
5. **Add appropriate pytest markers** for test categorization

### Test Data Patterns

```python
# Use the test data generator for consistent patterns
from tests.utils.cache_test_helpers import CacheTestDataGenerator

generator = CacheTestDataGenerator()
test_data = generator.generate_test_dataset(size=100, data_type='users')
```

### Performance Baselines

When adding performance tests:

1. **Establish baselines** on reference hardware
2. **Use relative thresholds** for different environments
3. **Include regression detection** for performance changes
4. **Document expected performance characteristics**

## License

This testing framework is part of the prompt-improver project and follows the same licensing terms.