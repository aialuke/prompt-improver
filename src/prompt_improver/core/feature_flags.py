"""
Feature Flag Manager for Prompt Improver

Provides a comprehensive feature flag system with hot-reloading, percentage-based rollouts,
and sophisticated user bucketing for managing technical debt cleanup phases.

This implementation follows 2025 best practices:
- Type-safe configuration with Pydantic
- Hot-reload capability without service restart
- Percentage-based rollouts with consistent user bucketing
- Thread-safe operations
- Comprehensive monitoring and metrics
- Clean separation of concerns
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from pydantic import BaseModel, Field, validator
import yaml

logger = logging.getLogger(__name__)

class FlagState(str, Enum):
    """Feature flag states."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ROLLOUT = "rollout"

class RolloutStrategy(str, Enum):
    """Rollout strategies for feature flags."""
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    GRADUAL = "gradual"

class TargetingRule(BaseModel):
    """Individual targeting rule for feature flags."""
    name: str
    condition: str  # JsonLogic expression
    variant: str
    priority: int = 0

class RolloutConfig(BaseModel):
    """Configuration for percentage-based rollouts."""
    strategy: RolloutStrategy = RolloutStrategy.PERCENTAGE
    percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    user_list: List[str] = Field(default_factory=list)
    gradual_config: Optional[Dict[str, Any]] = None
    sticky: bool = True  # Consistent bucketing for same user
    
    @validator('percentage')
    def validate_percentage(cls, v):
        if not 0.0 <= v <= 100.0:
            raise ValueError("Percentage must be between 0.0 and 100.0")
        return v

class FeatureFlagDefinition(BaseModel):
    """Complete feature flag definition."""
    key: str
    state: FlagState
    default_variant: str = "off"
    variants: Dict[str, Any] = Field(default_factory=lambda: {"on": True, "off": False})
    rollout: Optional[RolloutConfig] = None
    targeting_rules: List[TargetingRule] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('variants')
    def validate_variants(cls, v, values):
        default_variant = values.get('default_variant', 'off')
        if default_variant not in v:
            raise ValueError(f"Default variant '{default_variant}' not found in variants")
        return v

class EvaluationContext(BaseModel):
    """Context for feature flag evaluation."""
    user_id: Optional[str] = None
    user_type: Optional[str] = None
    environment: str = "production"
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class EvaluationResult(BaseModel):
    """Result of feature flag evaluation."""
    flag_key: str
    variant: str
    value: Any
    reason: str
    rule_matched: Optional[str] = None
    evaluation_context: EvaluationContext
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)

class FeatureFlagMetrics(BaseModel):
    """Metrics for feature flag usage."""
    flag_key: str
    evaluations_count: int = 0
    variant_distribution: Dict[str, int] = Field(default_factory=dict)
    last_evaluated: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None

class FileWatcher(FileSystemEventHandler):
    """File system watcher for hot-reloading feature flags."""
    
    def __init__(self, callback):
        self.callback = callback
        self._last_modified = {}
        self._debounce_time = 0.5  # 500ms debounce
    
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = event.src_path
        now = time.time()
        
        # Debounce rapid file changes
        if file_path in self._last_modified:
            if now - self._last_modified[file_path] < self._debounce_time:
                return
        
        self._last_modified[file_path] = now
        
        try:
            self.callback(file_path)
        except Exception as e:
            logger.error(f"Error in file watcher callback: {e}")

class FeatureFlagManager:
    """
    Advanced feature flag manager with hot-reloading and percentage rollouts.
    
    Features:
    - Hot-reload configuration changes without restart
    - Percentage-based rollouts with consistent user bucketing
    - Sophisticated targeting rules using JsonLogic
    - Thread-safe operations
    - Comprehensive metrics and monitoring
    - Multiple configuration sources (YAML, JSON)
    """
    
    def __init__(self, config_path: Union[str, Path], watch_files: bool = True):
        self.config_path = Path(config_path)
        self.watch_files = watch_files
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Feature flag storage
        self._flags: Dict[str, FeatureFlagDefinition] = {}
        self._metrics: Dict[str, FeatureFlagMetrics] = {}
        
        # File watching
        self._observer: Optional[Observer] = None
        self._watcher_thread: Optional[threading.Thread] = None
        
        # Configuration metadata
        self._config_loaded_at: Optional[datetime] = None
        self._config_version: Optional[str] = None
        
        # Initialize
        self._load_configuration()
        if self.watch_files:
            self._start_file_watcher()
    
    def _load_configuration(self) -> None:
        """Load feature flag configuration from file."""
        try:
            with self._lock:
                if not self.config_path.exists():
                    logger.warning(f"Configuration file not found: {self.config_path}")
                    return
                
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix.lower() in ['.yml', '.yaml']:
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                # Load flags
                flags_data = config_data.get('flags', {})
                new_flags = {}
                
                for flag_key, flag_config in flags_data.items():
                    try:
                        flag_config['key'] = flag_key
                        flag_def = FeatureFlagDefinition(**flag_config)
                        new_flags[flag_key] = flag_def
                    except Exception as e:
                        logger.error(f"Error loading flag '{flag_key}': {e}")
                        continue
                
                # Update flags atomically
                self._flags = new_flags
                self._config_loaded_at = datetime.now(UTC)
                self._config_version = config_data.get('version', 'unknown')
                
                logger.info(f"Loaded {len(self._flags)} feature flags from {self.config_path}")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def _start_file_watcher(self) -> None:
        """Start file system watcher for hot-reloading."""
        if self._observer:
            return
        
        try:
            self._observer = Observer()
            watcher = FileWatcher(self._on_config_changed)
            
            # Watch the configuration directory
            watch_path = self.config_path.parent
            self._observer.schedule(watcher, str(watch_path), recursive=False)
            self._observer.start()
            
            logger.info(f"Started file watcher for {watch_path}")
            
        except Exception as e:
            logger.error(f"Error starting file watcher: {e}")
    
    def _on_config_changed(self, file_path: str) -> None:
        """Handle configuration file changes."""
        if Path(file_path).name == self.config_path.name:
            logger.info(f"Configuration file changed: {file_path}")
            self._load_configuration()
    
    def _hash_for_bucketing(self, flag_key: str, user_id: str) -> float:
        """Generate consistent hash for user bucketing."""
        # Combine flag key and user ID for consistent bucketing
        combined = f"{flag_key}:{user_id}"
        hash_bytes = hashlib.sha256(combined.encode()).digest()
        # Convert to float between 0 and 1
        hash_value = int.from_bytes(hash_bytes[:4], byteorder='big') / (2**32)
        return hash_value
    
    def _evaluate_rollout(self, flag: FeatureFlagDefinition, context: EvaluationContext) -> bool:
        """Evaluate if user should receive rollout variant."""
        if not flag.rollout or not context.user_id:
            return False
        
        rollout = flag.rollout
        
        if rollout.strategy == RolloutStrategy.USER_LIST:
            return context.user_id in rollout.user_list
        
        elif rollout.strategy == RolloutStrategy.PERCENTAGE:
            hash_value = self._hash_for_bucketing(flag.key, context.user_id)
            threshold = rollout.percentage / 100.0
            return hash_value < threshold
        
        elif rollout.strategy == RolloutStrategy.GRADUAL:
            # Implement gradual rollout logic here
            # This could involve time-based percentage increases
            return False
        
        return False
    
    def _evaluate_targeting_rules(
        self, 
        flag: FeatureFlagDefinition, 
        context: EvaluationContext
    ) -> Optional[str]:
        """Evaluate targeting rules against context."""
        # Sort rules by priority (higher first)
        sorted_rules = sorted(flag.targeting_rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            try:
                # Simple condition evaluation (could be enhanced with JsonLogic)
                if self._evaluate_condition(rule.condition, context):
                    return rule.variant
            except Exception as e:
                logger.error(f"Error evaluating rule '{rule.name}' for flag '{flag.key}': {e}")
                continue
        
        return None
    
    def _evaluate_condition(self, condition: str, context: EvaluationContext) -> bool:
        """Evaluate a simple condition against context."""
        # This is a simplified implementation
        # In production, you'd want to use a proper JsonLogic library or similar
        
        context_dict = {
            "user_id": context.user_id,
            "user_type": context.user_type,
            "environment": context.environment,
            **context.custom_attributes
        }
        
        try:
            # Very simple condition evaluation for demonstration
            # You should replace this with proper JsonLogic evaluation
            return eval(condition, {"__builtins__": {}}, context_dict)
        except:
            return False
    
    def _update_metrics(self, result: EvaluationResult) -> None:
        """Update metrics for flag evaluation."""
        with self._lock:
            if result.flag_key not in self._metrics:
                self._metrics[result.flag_key] = FeatureFlagMetrics(flag_key=result.flag_key)
            
            metrics = self._metrics[result.flag_key]
            metrics.evaluations_count += 1
            metrics.last_evaluated = result.evaluated_at
            
            # Update variant distribution
            if result.variant not in metrics.variant_distribution:
                metrics.variant_distribution[result.variant] = 0
            metrics.variant_distribution[result.variant] += 1
    
    def evaluate_flag(
        self, 
        flag_key: str, 
        context: EvaluationContext, 
        default_value: Any = False
    ) -> EvaluationResult:
        """
        Evaluate a feature flag for the given context.
        
        Args:
            flag_key: The feature flag key to evaluate
            context: The evaluation context (user, environment, etc.)
            default_value: Default value if flag not found or disabled
            
        Returns:
            EvaluationResult with the flag value and evaluation details
        """
        try:
            with self._lock:
                flag = self._flags.get(flag_key)
                
                if not flag:
                    result = EvaluationResult(
                        flag_key=flag_key,
                        variant="default",
                        value=default_value,
                        reason="FLAG_NOT_FOUND",
                        evaluation_context=context
                    )
                    self._update_metrics(result)
                    return result
                
                if flag.state == FlagState.DISABLED:
                    variant = flag.default_variant
                    value = flag.variants.get(variant, default_value)
                    result = EvaluationResult(
                        flag_key=flag_key,
                        variant=variant,
                        value=value,
                        reason="FLAG_DISABLED",
                        evaluation_context=context
                    )
                    self._update_metrics(result)
                    return result
                
                # Check targeting rules first
                targeted_variant = self._evaluate_targeting_rules(flag, context)
                if targeted_variant:
                    value = flag.variants.get(targeted_variant, default_value)
                    result = EvaluationResult(
                        flag_key=flag_key,
                        variant=targeted_variant,
                        value=value,
                        reason="TARGETING_MATCH",
                        evaluation_context=context
                    )
                    self._update_metrics(result)
                    return result
                
                # Check rollout configuration
                if flag.state == FlagState.ROLLOUT and self._evaluate_rollout(flag, context):
                    # User is in rollout, return "on" variant or first non-default variant
                    rollout_variant = "on" if "on" in flag.variants else next(
                        (v for v in flag.variants.keys() if v != flag.default_variant),
                        flag.default_variant
                    )
                    value = flag.variants.get(rollout_variant, default_value)
                    result = EvaluationResult(
                        flag_key=flag_key,
                        variant=rollout_variant,
                        value=value,
                        reason="ROLLOUT_MATCH",
                        evaluation_context=context
                    )
                    self._update_metrics(result)
                    return result
                
                # Default to default variant
                variant = flag.default_variant
                value = flag.variants.get(variant, default_value)
                result = EvaluationResult(
                    flag_key=flag_key,
                    variant=variant,
                    value=value,
                    reason="DEFAULT",
                    evaluation_context=context
                )
                self._update_metrics(result)
                return result
                
        except Exception as e:
            logger.error(f"Error evaluating flag '{flag_key}': {e}")
            
            # Update error metrics
            with self._lock:
                if flag_key not in self._metrics:
                    self._metrics[flag_key] = FeatureFlagMetrics(flag_key=flag_key)
                self._metrics[flag_key].error_count += 1
                self._metrics[flag_key].last_error = str(e)
            
            # Return safe default
            result = EvaluationResult(
                flag_key=flag_key,
                variant="error",
                value=default_value,
                reason="EVALUATION_ERROR",
                evaluation_context=context
            )
            return result
    
    def is_enabled(self, flag_key: str, context: EvaluationContext) -> bool:
        """Check if a feature flag is enabled for the given context."""
        result = self.evaluate_flag(flag_key, context, False)
        return bool(result.value)
    
    def get_variant(self, flag_key: str, context: EvaluationContext, default: str = "off") -> str:
        """Get the variant for a feature flag."""
        result = self.evaluate_flag(flag_key, context, default)
        return result.variant
    
    def get_all_flags(self) -> Dict[str, FeatureFlagDefinition]:
        """Get all currently loaded feature flags."""
        with self._lock:
            return self._flags.copy()
    
    def get_metrics(self, flag_key: Optional[str] = None) -> Union[FeatureFlagMetrics, Dict[str, FeatureFlagMetrics]]:
        """Get metrics for feature flags."""
        with self._lock:
            if flag_key:
                return self._metrics.get(flag_key, FeatureFlagMetrics(flag_key=flag_key))
            return self._metrics.copy()
    
    def reload_configuration(self) -> None:
        """Manually reload configuration."""
        logger.info("Manually reloading feature flag configuration")
        self._load_configuration()
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """Get information about the current configuration."""
        with self._lock:
            return {
                "config_path": str(self.config_path),
                "loaded_at": self._config_loaded_at.isoformat() if self._config_loaded_at else None,
                "version": self._config_version,
                "flags_count": len(self._flags),
                "watching_files": bool(self._observer and self._observer.is_alive())
            }
    
    def shutdown(self) -> None:
        """Shutdown the feature flag manager and cleanup resources."""
        logger.info("Shutting down feature flag manager")
        
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
        
        with self._lock:
            self._flags.clear()
            self._metrics.clear()

# Convenience functions for common use cases
_default_manager: Optional[FeatureFlagManager] = None

def initialize_feature_flags(config_path: Union[str, Path], watch_files: bool = True) -> FeatureFlagManager:
    """Initialize the global feature flag manager."""
    global _default_manager
    _default_manager = FeatureFlagManager(config_path, watch_files)
    return _default_manager

def get_feature_flag_manager() -> Optional[FeatureFlagManager]:
    """Get the global feature flag manager."""
    return _default_manager

def is_feature_enabled(flag_key: str, user_id: Optional[str] = None, **context_attrs) -> bool:
    """Check if a feature is enabled (convenience function)."""
    if not _default_manager:
        logger.warning("Feature flag manager not initialized")
        return False
    
    context = EvaluationContext(user_id=user_id, custom_attributes=context_attrs)
    return _default_manager.is_enabled(flag_key, context)

def get_feature_variant(flag_key: str, user_id: Optional[str] = None, default: str = "off", **context_attrs) -> str:
    """Get feature variant (convenience function)."""
    if not _default_manager:
        logger.warning("Feature flag manager not initialized")
        return default
    
    context = EvaluationContext(user_id=user_id, custom_attributes=context_attrs)
    return _default_manager.get_variant(flag_key, context, default)