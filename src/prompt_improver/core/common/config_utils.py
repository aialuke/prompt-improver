"""
Centralized configuration utilities to eliminate duplicate config loading patterns.

Consolidates the patterns:
- try: config = get_config() except: fallback
- Duplicate validation logic across config classes
- Repeated configuration initialization
"""

from typing import Optional, Any, TypeVar, Type, Union, Dict, cast
from functools import lru_cache
from dataclasses import dataclass
from .logging_utils import get_logger, LoggerMixin

logger = get_logger(__name__)

T = TypeVar('T')

@dataclass
class ConfigLoadResult:
    """Result of configuration loading operation."""
    config: Optional[Any]
    success: bool
    error: Optional[str]
    fallback_used: bool = False

@lru_cache(maxsize=1)
def get_config_safely(use_fallback: bool = True) -> ConfigLoadResult:
    """
    Safely load configuration with fallback handling.

    Consolidates the common pattern:
    try:
        config = get_config()
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        config = fallback_config

    Args:
        use_fallback: Whether to use fallback values on failure

    Returns:
        ConfigLoadResult with loaded config or fallback
    """
    try:
        from ..config import get_config
        config = get_config()
        return ConfigLoadResult(
            config=config,
            success=True,
            error=None,
            fallback_used=False
        )
    except Exception as e:
        error_msg = f"Failed to load centralized config: {e}"
        logger.warning(error_msg)

        if use_fallback:
            # Return minimal fallback config
            fallback_config = _create_fallback_config()
            return ConfigLoadResult(
                config=fallback_config,
                success=False,
                error=error_msg,
                fallback_used=True
            )
        else:
            return ConfigLoadResult(
                config=None,
                success=False,
                error=error_msg,
                fallback_used=False
            )

def _create_fallback_config():
    """Create minimal fallback configuration."""
    # Use argparse.Namespace to avoid types module shadowing warning
    from argparse import Namespace

    # Create nested namespace structure matching common config patterns
    database = Namespace(
        pool_min_size=5,
        pool_max_size=20,
        pool_target_utilization=0.7,
        pool_high_utilization_threshold=0.9,
        pool_low_utilization_threshold=0.3,
        pool_stress_threshold=0.8,
        pool_underutilized_threshold=0.4,
        timeout=30,
        retry_attempts=3
    )

    redis = Namespace(
        host="localhost",
        port=6379,
        db=0,
        timeout=5,
        retry_attempts=3,
        max_connections=10
    )

    api = Namespace(
        host="0.0.0.0",
        port=8000,
        timeout=30,
        max_request_size=10485760  # 10MB
    )

    ml = Namespace(
        batch_size=32,
        max_workers=4,
        timeout=300,
        model_cache_size=100
    )

    directory_paths = Namespace(
        prometheus_multiproc_dir="/tmp/prometheus_multiproc",
        logs_dir="./logs",
        data_dir="./data",
        models_dir="./models"
    )

    fallback = Namespace(
        database=database,
        redis=redis,
        api=api,
        ml=ml,
        directory_paths=directory_paths,
        environment="development",
        debug=True,
        log_level="INFO"
    )

    return fallback

class ConfigMixin(LoggerMixin):
    """
    Mixin class to provide consistent configuration access pattern.

    Eliminates duplicate config loading logic in classes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._config_result: Any = None
        self._config_cache: Any = None

    @property
    def config(self):
        """Get configuration with caching and fallback handling."""
        if self._config_cache is None:
            self._config_result = get_config_safely()
            self._config_cache = self._config_result.config

            if self._config_result.fallback_used:
                self.logger.warning("Using fallback configuration due to load failure")

        return self._config_cache

    @property
    def config_status(self) -> ConfigLoadResult:
        """Get detailed configuration loading status."""
        if self._config_result is None:
            # Trigger config loading
            _ = self.config
        return self._config_result

    def get_config_value(self, path: str, default: Any = None, required: bool = False):
        """
        Get nested configuration value by dot-separated path.

        Args:
            path: Dot-separated path (e.g., "database.pool_min_size")
            default: Default value if not found
            required: Raise error if not found and no default

        Returns:
            Configuration value or default

        Raises:
            ValueError: If required=True and value not found
        """
        config = self.config
        parts = path.split('.')

        current = config
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                if required and default is None:
                    raise ValueError(f"Required config value not found: {path}")
                return default

        return current

    def validate_config_requirements(self, required_paths: list[str]) -> bool:
        """
        Validate that all required configuration paths exist.

        Args:
            required_paths: List of required dot-separated paths

        Returns:
            True if all requirements met, False otherwise
        """
        missing_paths: list[str] = []

        for path in required_paths:
            try:
                self.get_config_value(path, required=True)
            except ValueError:
                missing_paths.append(path)

        if missing_paths:
            self.logger.error(f"Missing required config values: {missing_paths}")
            return False

        return True

def load_config_section(section_name: str, config_class: Type[T]) -> Optional[T]:
    """
    Load and validate a specific configuration section.

    Args:
        section_name: Name of config section (e.g., "database")
        config_class: Pydantic model class for validation

    Returns:
        Validated config instance or None on failure
    """
    config_result = get_config_safely()

    if not config_result.success:
        logger.error(f"Cannot load {section_name} config: {config_result.error}")
        return None

    try:
        section_data = getattr(config_result.config, section_name, None)
        if section_data is None:
            logger.error(f"Config section '{section_name}' not found")
            return None

        # If it's already the right type, return it
        if isinstance(section_data, config_class):
            return section_data

        # Try to convert namespace to dict for Pydantic
        if hasattr(section_data, '__dict__'):
            section_dict = cast(Dict[str, Any], vars(section_data))
        elif isinstance(section_data, dict):
            section_dict = cast(Dict[str, Any], section_data)
        else:
            logger.error(f"Cannot convert {section_name} config to required format")
            return None

        return config_class(**section_dict)

    except Exception as e:
        logger.error(f"Failed to load {section_name} config: {e}")
        return None

# Common validation patterns to reduce duplication
def validate_port(value: Any, name: str = "port") -> int:
    """Common port validation logic."""
    if not isinstance(value, int) or value < 1 or value > 65535:
        raise ValueError(f"{name} must be between 1 and 65535, got {value}")
    return value

def validate_timeout(value: Any, name: str = "timeout") -> Union[int, float]:
    """Common timeout validation logic."""
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value

def validate_percentage(value: Any, name: str = "percentage") -> float:
    """Common percentage validation logic."""
    if not isinstance(value, (int, float)) or value < 0 or value > 100:
        raise ValueError(f"{name} must be between 0 and 100, got {value}")
    return float(value)

def validate_ratio(value: Any, name: str = "ratio") -> float:
    """Common ratio validation logic."""
    if not isinstance(value, (int, float)) or value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return float(value)
