"""
Configuration and constants for MultiLevelCache testing.

Provides centralized configuration management, test scenarios,
performance thresholds, and validation criteria for comprehensive
cache testing with TestContainers and real behavior validation.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


# ============================================================================
# TEST CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class CacheTestConfig:
    """Configuration for cache testing scenarios."""
    l1_max_size: int = 100
    l2_default_ttl: int = 300
    enable_l2: bool = True
    enable_warming: bool = True
    warming_threshold: float = 2.0
    warming_interval: int = 60
    max_warming_keys: int = 10
    redis_image: str = "redis:7-alpine"
    redis_port: int = 6379


@dataclass
class PerformanceThresholds:
    """Performance requirements and SLA thresholds."""
    # Response time thresholds (in seconds)
    l1_max_avg_time: float = 0.001  # 1ms
    l1_max_p95_time: float = 0.002  # 2ms
    full_cache_max_avg_time: float = 0.01  # 10ms
    full_cache_max_p95_time: float = 0.05  # 50ms
    sla_target_ms: float = 200.0  # 200ms SLA
    
    # Throughput thresholds
    min_ops_per_second: float = 1000.0
    min_concurrent_throughput: float = 500.0
    min_sustained_throughput: float = 100.0
    
    # Hit rate thresholds
    min_overall_hit_rate: float = 0.7
    min_l1_hit_rate: float = 0.5
    min_warming_hit_rate: float = 0.5
    
    # Error rate thresholds
    max_error_rate: float = 0.01  # 1%
    max_consecutive_errors: int = 10
    
    # SLO compliance thresholds
    min_slo_compliance_rate: float = 0.95  # 95%
    
    # Memory efficiency thresholds
    max_memory_overhead_ratio: float = 5.0
    max_memory_per_item_kb: float = 50.0


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    concurrent_users: int = 10
    operations_per_user: int = 100
    sustained_duration_seconds: int = 60
    target_ops_per_second: int = 100
    operation_mix: Dict[str, float] = None
    
    def __post_init__(self):
        if self.operation_mix is None:
            self.operation_mix = {'get': 0.7, 'set': 0.25, 'delete': 0.05}


# ============================================================================
# TEST SCENARIO CONFIGURATIONS
# ============================================================================

class TestScenarios:
    """Predefined test scenario configurations."""
    
    @staticmethod
    def get_default_config() -> CacheTestConfig:
        """Default configuration for standard testing."""
        return CacheTestConfig()
    
    @staticmethod
    def get_performance_config() -> CacheTestConfig:
        """Configuration optimized for performance testing."""
        return CacheTestConfig(
            l1_max_size=1000,
            l2_default_ttl=600,
            warming_interval=30,
            max_warming_keys=50
        )
    
    @staticmethod
    def get_memory_constrained_config() -> CacheTestConfig:
        """Configuration for memory-constrained testing."""
        return CacheTestConfig(
            l1_max_size=10,
            warming_interval=300,
            max_warming_keys=5
        )
    
    @staticmethod
    def get_high_throughput_config() -> CacheTestConfig:
        """Configuration for high-throughput testing."""
        return CacheTestConfig(
            l1_max_size=5000,
            l2_default_ttl=600,
            warming_interval=10,
            max_warming_keys=100
        )
    
    @staticmethod
    def get_warming_disabled_config() -> CacheTestConfig:
        """Configuration with warming disabled."""
        return CacheTestConfig(
            enable_warming=False
        )
    
    @staticmethod
    def get_l2_disabled_config() -> CacheTestConfig:
        """Configuration with L2 cache disabled."""
        return CacheTestConfig(
            enable_l2=False,
            enable_warming=False
        )
    
    @staticmethod
    def get_resilience_test_config() -> CacheTestConfig:
        """Configuration for resilience testing."""
        return CacheTestConfig(
            l1_max_size=50,
            l2_default_ttl=120,
            warming_interval=300,
            max_warming_keys=20
        )


# ============================================================================
# PERFORMANCE AND VALIDATION CONSTANTS
# ============================================================================

class ValidationCriteria:
    """Validation criteria for different test aspects."""
    
    # Basic operation validation
    BASIC_OPERATIONS = {
        'data_consistency': True,
        'cache_isolation': True,
        'ttl_behavior': True
    }
    
    # Multi-level behavior validation
    MULTI_LEVEL_BEHAVIOR = {
        'l1_hit_priority': True,
        'l2_fallback': True,
        'l3_population': True,
        'cache_level_isolation': True
    }
    
    # Warming functionality validation
    WARMING_VALIDATION = {
        'access_pattern_tracking': True,
        'warming_candidate_identification': True,
        'background_warming_cycles': True,
        'manual_warming': True,
        'warming_error_handling': True
    }
    
    # Health and monitoring validation
    HEALTH_MONITORING = {
        'comprehensive_health_check': True,
        'degradation_detection': True,
        'opentelemetry_integration': True,
        'specialized_cache_integration': True
    }


class TestDataPatterns:
    """Test data patterns and sizes."""
    
    # Data size categories
    SMALL_DATA_SIZE = 100  # 100 bytes
    MEDIUM_DATA_SIZE = 10_000  # 10KB
    LARGE_DATA_SIZE = 100_000  # 100KB
    
    # Test dataset sizes
    SMALL_DATASET = 10
    MEDIUM_DATASET = 100
    LARGE_DATASET = 1000
    
    # Data complexity levels
    SIMPLE_DATA_TYPES = ['string', 'integer', 'float', 'boolean', 'null']
    COMPLEX_DATA_TYPES = ['nested_dict', 'mixed_array', 'unicode_text', 'date_strings']
    REALISTIC_DATA_TYPES = ['users', 'products', 'sessions']
    
    # Access patterns for warming tests
    HOT_ACCESS_PATTERN = {'frequency': 10, 'interval': 0.1}
    WARM_ACCESS_PATTERN = {'frequency': 5, 'interval': 1.0}
    COLD_ACCESS_PATTERN = {'frequency': 1, 'interval': 10.0}


# ============================================================================
# CONTAINER AND INFRASTRUCTURE SETTINGS
# ============================================================================

class ContainerSettings:
    """Container configuration and settings."""
    
    # Redis container settings
    REDIS_IMAGE = "redis:7-alpine"
    REDIS_PORT = 6379
    REDIS_STARTUP_TIMEOUT = 30  # seconds
    REDIS_HEALTH_CHECK_INTERVAL = 1.0  # seconds
    
    # Container resource limits
    MEMORY_LIMIT = "512m"
    CPU_LIMIT = "1.0"
    
    # Network settings
    NETWORK_TIMEOUT = 10.0  # seconds
    CONNECTION_POOL_SIZE = 20
    
    # Health check settings
    HEALTH_CHECK_RETRIES = 30
    HEALTH_CHECK_DELAY = 1.0


class TestExecutionSettings:
    """Test execution configuration."""
    
    # Timeout settings
    TEST_TIMEOUT_SECONDS = 30
    LONG_TEST_TIMEOUT_SECONDS = 300  # 5 minutes for performance tests
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    # Parallel execution settings
    MAX_CONCURRENT_TESTS = 4
    CONCURRENT_OPERATIONS_LIMIT = 100
    
    # Logging settings
    LOG_LEVEL = "INFO"
    ENABLE_PERFORMANCE_LOGGING = True
    ENABLE_CONTAINER_LOGGING = False


# ============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# ============================================================================

class EnvironmentConfigs:
    """Environment-specific configurations."""
    
    @staticmethod
    def get_ci_config() -> Dict[str, Any]:
        """Configuration for CI/CD environments."""
        return {
            'cache_config': CacheTestConfig(
                l1_max_size=50,
                l2_default_ttl=180,
                warming_interval=120
            ),
            'performance_thresholds': PerformanceThresholds(
                # Relaxed thresholds for CI
                full_cache_max_avg_time=0.02,
                full_cache_max_p95_time=0.1,
                min_ops_per_second=500.0
            ),
            'load_test_config': LoadTestConfig(
                concurrent_users=5,
                operations_per_user=50,
                sustained_duration_seconds=30
            ),
            'container_settings': {
                'startup_timeout': 60,
                'memory_limit': '256m'
            }
        }
    
    @staticmethod
    def get_development_config() -> Dict[str, Any]:
        """Configuration for development environments."""
        return {
            'cache_config': TestScenarios.get_default_config(),
            'performance_thresholds': PerformanceThresholds(),
            'load_test_config': LoadTestConfig(),
            'container_settings': ContainerSettings(),
            'enable_detailed_logging': True
        }
    
    @staticmethod
    def get_performance_test_config() -> Dict[str, Any]:
        """Configuration for dedicated performance testing."""
        return {
            'cache_config': TestScenarios.get_performance_config(),
            'performance_thresholds': PerformanceThresholds(
                # Stricter thresholds for performance testing
                l1_max_avg_time=0.0005,  # 0.5ms
                full_cache_max_avg_time=0.005,  # 5ms
                min_ops_per_second=2000.0
            ),
            'load_test_config': LoadTestConfig(
                concurrent_users=20,
                operations_per_user=200,
                sustained_duration_seconds=120,
                target_ops_per_second=500
            ),
            'container_settings': {
                'memory_limit': '1g',
                'cpu_limit': '2.0'
            }
        }


# ============================================================================
# TEST MATRIX CONFIGURATIONS
# ============================================================================

class TestMatrix:
    """Test matrix configurations for comprehensive coverage."""
    
    CACHE_CONFIGURATIONS = [
        ('default', TestScenarios.get_default_config()),
        ('performance', TestScenarios.get_performance_config()),
        ('memory_constrained', TestScenarios.get_memory_constrained_config()),
        ('warming_disabled', TestScenarios.get_warming_disabled_config()),
        ('l2_disabled', TestScenarios.get_l2_disabled_config())
    ]
    
    DATA_SCENARIOS = [
        ('simple_data', TestDataPatterns.SMALL_DATASET, 'simple'),
        ('complex_data', TestDataPatterns.SMALL_DATASET, 'complex'),
        ('user_data', TestDataPatterns.MEDIUM_DATASET, 'users'),
        ('product_data', TestDataPatterns.MEDIUM_DATASET, 'products'),
        ('session_data', TestDataPatterns.SMALL_DATASET, 'sessions'),
        ('mixed_data', TestDataPatterns.MEDIUM_DATASET, 'mixed')
    ]
    
    LOAD_TEST_SCENARIOS = [
        ('light_load', LoadTestConfig(concurrent_users=5, operations_per_user=50)),
        ('medium_load', LoadTestConfig(concurrent_users=10, operations_per_user=100)),
        ('heavy_load', LoadTestConfig(concurrent_users=20, operations_per_user=200)),
        ('sustained_load', LoadTestConfig(sustained_duration_seconds=60)),
        ('burst_load', LoadTestConfig(concurrent_users=50, operations_per_user=20))
    ]
    
    ERROR_SCENARIOS = [
        ('redis_disconnection', {'simulate': 'redis_failure', 'duration': 5}),
        ('high_latency', {'simulate': 'network_latency', 'latency_ms': 200}),
        ('memory_pressure', {'simulate': 'memory_pressure', 'fill_ratio': 1.5}),
        ('serialization_errors', {'simulate': 'serialization_failure', 'error_rate': 0.1}),
        ('concurrent_access', {'simulate': 'race_conditions', 'threads': 20})
    ]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_config_for_test_type(test_type: str) -> Dict[str, Any]:
    """
    Get configuration for specific test type.
    
    Args:
        test_type: Type of test ('unit', 'integration', 'performance', 'load', 'resilience')
        
    Returns:
        Complete configuration dictionary
    """
    configs = {
        'unit': {
            'cache_config': TestScenarios.get_default_config(),
            'performance_thresholds': PerformanceThresholds(),
            'enable_containers': False,
            'enable_performance_monitoring': False
        },
        'integration': {
            'cache_config': TestScenarios.get_default_config(),
            'performance_thresholds': PerformanceThresholds(),
            'load_test_config': LoadTestConfig(),
            'enable_containers': True,
            'enable_performance_monitoring': True
        },
        'performance': EnvironmentConfigs.get_performance_test_config(),
        'load': {
            'cache_config': TestScenarios.get_high_throughput_config(),
            'performance_thresholds': PerformanceThresholds(),
            'load_test_config': LoadTestConfig(
                concurrent_users=50,
                operations_per_user=500,
                sustained_duration_seconds=300
            ),
            'enable_containers': True,
            'enable_performance_monitoring': True
        },
        'resilience': {
            'cache_config': TestScenarios.get_resilience_test_config(),
            'performance_thresholds': PerformanceThresholds(
                # More lenient thresholds for resilience testing
                max_error_rate=0.05,  # 5%
                min_slo_compliance_rate=0.9  # 90%
            ),
            'enable_containers': True,
            'enable_error_simulation': True
        }
    }
    
    return configs.get(test_type, configs['integration'])


def get_thresholds_for_environment(environment: str = 'development') -> PerformanceThresholds:
    """
    Get performance thresholds for specific environment.
    
    Args:
        environment: Environment name ('development', 'ci', 'performance')
        
    Returns:
        Performance thresholds configuration
    """
    if environment == 'ci':
        return EnvironmentConfigs.get_ci_config()['performance_thresholds']
    elif environment == 'performance':
        return EnvironmentConfigs.get_performance_test_config()['performance_thresholds']
    else:
        return PerformanceThresholds()


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration for completeness and consistency.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if configuration is valid
    """
    required_keys = ['cache_config', 'performance_thresholds']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate cache configuration
    cache_config = config['cache_config']
    if cache_config.enable_warming and not cache_config.enable_l2:
        raise ValueError("Cache warming requires L2 cache to be enabled")
    
    if cache_config.l1_max_size <= 0:
        raise ValueError("L1 max size must be positive")
    
    if cache_config.l2_default_ttl <= 0:
        raise ValueError("L2 default TTL must be positive")
    
    # Validate performance thresholds
    thresholds = config['performance_thresholds']
    if thresholds.min_overall_hit_rate < 0 or thresholds.min_overall_hit_rate > 1:
        raise ValueError("Hit rate thresholds must be between 0 and 1")
    
    if thresholds.max_error_rate < 0 or thresholds.max_error_rate > 1:
        raise ValueError("Error rate thresholds must be between 0 and 1")
    
    return True


# ============================================================================
# DEFAULT CONFIGURATIONS
# ============================================================================

# Default configurations for different test types
DEFAULT_INTEGRATION_CONFIG = get_config_for_test_type('integration')
DEFAULT_PERFORMANCE_CONFIG = get_config_for_test_type('performance')
DEFAULT_LOAD_TEST_CONFIG = get_config_for_test_type('load')
DEFAULT_RESILIENCE_CONFIG = get_config_for_test_type('resilience')

# Validate all default configurations
for config_name, config in [
    ('integration', DEFAULT_INTEGRATION_CONFIG),
    ('performance', DEFAULT_PERFORMANCE_CONFIG),
    ('load', DEFAULT_LOAD_TEST_CONFIG),
    ('resilience', DEFAULT_RESILIENCE_CONFIG)
]:
    try:
        validate_config(config)
    except ValueError as e:
        raise ValueError(f"Invalid default {config_name} configuration: {e}")