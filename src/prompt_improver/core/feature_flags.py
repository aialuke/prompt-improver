"""Feature Flag Manager for Prompt Improver.

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

import asyncio
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from sqlmodel import Field
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class FlagState(StrEnum):
    """Feature flag states."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    ROLLOUT = "rollout"


class RolloutStrategy(StrEnum):
    """Rollout strategies for feature flags."""

    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    GRADUAL = "gradual"


class TargetingRule(BaseModel):
    """Individual targeting rule for feature flags."""

    name: str
    condition: str
    variant: str
    priority: int = 0


class RolloutConfig(BaseModel):
    """Configuration for percentage-based rollouts."""

    strategy: RolloutStrategy = RolloutStrategy.PERCENTAGE
    percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    user_list: list[str] = Field(default_factory=list)
    gradual_config: dict[str, Any] | None = None
    sticky: bool = True


class FeatureFlagDefinition(BaseModel):
    """Complete feature flag definition."""

    key: str
    state: FlagState
    default_variant: str = "off"
    variants: dict[str, Any] = Field(default_factory=lambda: {"on": True, "off": False})
    rollout: RolloutConfig | None = None
    targeting_rules: list[TargetingRule] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationContext(BaseModel):
    """Context for feature flag evaluation."""

    user_id: str | None = None
    user_type: str | None = None
    environment: str = "production"
    custom_attributes: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvaluationResult(BaseModel):
    """Result of feature flag evaluation."""

    flag_key: str
    variant: str
    value: Any
    reason: str
    rule_matched: str | None = None
    evaluation_context: EvaluationContext
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureFlagMetrics(BaseModel):
    """Metrics for feature flag usage."""

    flag_key: str
    evaluations_count: int = 0
    variant_distribution: dict[str, int] = Field(default_factory=dict)
    last_evaluated: datetime | None = None
    error_count: int = 0
    last_error: str | None = None


class FileWatcher(FileSystemEventHandler):
    """File system watcher for hot-reloading feature flags."""

    def __init__(self, callback) -> None:
        self.callback = callback
        self._last_modified = {}
        self._debounce_time = 0.5

    def on_modified(self, event):
        if event.is_directory:
            return
        file_path = event.src_path
        now = time.time()
        if file_path in self._last_modified:
            if now - self._last_modified[file_path] < self._debounce_time:
                return
        self._last_modified[file_path] = now
        try:
            self.callback(file_path)
        except Exception as e:
            logger.exception(f"Error in file watcher callback: {e}")


class FeatureFlagService:
    """Advanced feature flag service with hot-reloading and percentage rollouts implementing Clean Architecture patterns.

    Features:
    - Hot-reload configuration changes without restart
    - Percentage-based rollouts with consistent user bucketing
    - Sophisticated targeting rules using JsonLogic
    - Thread-safe operations
    - Comprehensive metrics and monitoring
    - Multiple configuration sources (YAML, JSON)
    """

    def __init__(self, config_path: str | Path, watch_files: bool = True) -> None:
        self.config_path = Path(config_path)
        self.watch_files = watch_files
        self._lock = asyncio.Lock()
        self._flags: dict[str, FeatureFlagDefinition] = {}
        self._metrics: dict[str, FeatureFlagMetrics] = {}
        self._observer: Observer | None = None
        self._config_loaded_at: datetime | None = None
        self._config_version: str | None = None
        # Configuration will be loaded via async_init()
        if self.watch_files:
            self._start_file_watcher()

    async def async_init(self) -> None:
        """Initialize the manager asynchronously."""
        await self._load_configuration()

    async def _load_configuration(self) -> None:
        """Load feature flag configuration from file."""
        try:
            async with self._lock:
                if not self.config_path.exists():
                    logger.warning(f"Configuration file not found: {self.config_path}")
                    return
                with open(self.config_path, encoding="utf-8") as f:
                    if self.config_path.suffix.lower() in {".yml", ".yaml"}:
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                flags_data = config_data.get("flags", {})
                new_flags = {}
                for flag_key, flag_config in flags_data.items():
                    try:
                        flag_config["key"] = flag_key
                        flag_def = FeatureFlagDefinition(**flag_config)
                        new_flags[flag_key] = flag_def
                    except Exception as e:
                        logger.exception(f"Error loading flag '{flag_key}': {e}")
                        continue
                self._flags = new_flags
                self._config_loaded_at = datetime.now(UTC)
                self._config_version = config_data.get("version", "unknown")
                logger.info(
                    f"Loaded {len(self._flags)} feature flags from {self.config_path}"
                )
        except Exception as e:
            logger.exception(f"Error loading configuration: {e}")

    def _start_file_watcher(self) -> None:
        """Start file system watcher for hot-reloading."""
        if self._observer:
            return
        try:
            self._observer = Observer()
            watcher = FileWatcher(self._on_config_changed)
            watch_path = self.config_path.parent
            self._observer.schedule(watcher, str(watch_path), recursive=False)
            self._observer.start()
            logger.info(f"Started file watcher for {watch_path}")
        except Exception as e:
            logger.exception(f"Error starting file watcher: {e}")

    def _on_config_changed(self, file_path: str) -> None:
        """Handle configuration file changes."""
        if Path(file_path).name == self.config_path.name:
            logger.info(f"Configuration file changed: {file_path}")
            # Schedule async reload
            asyncio.create_task(self._load_configuration())

    def _hash_for_bucketing(self, flag_key: str, user_id: str) -> float:
        """Generate consistent hash for user bucketing."""
        combined = f"{flag_key}:{user_id}"
        hash_bytes = hashlib.sha256(combined.encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big") / 2**32

    def _evaluate_rollout(
        self, flag: FeatureFlagDefinition, context: EvaluationContext
    ) -> bool:
        """Evaluate if user should receive rollout variant."""
        if not flag.rollout or not context.user_id:
            return False
        rollout = flag.rollout
        if rollout.strategy == RolloutStrategy.USER_LIST:
            return context.user_id in rollout.user_list
        if rollout.strategy == RolloutStrategy.PERCENTAGE:
            hash_value = self._hash_for_bucketing(flag.key, context.user_id)
            threshold = rollout.percentage / 100.0
            return hash_value < threshold
        if rollout.strategy == RolloutStrategy.GRADUAL:
            return False
        return False

    def _evaluate_targeting_rules(
        self, flag: FeatureFlagDefinition, context: EvaluationContext
    ) -> str | None:
        """Evaluate targeting rules against context."""
        sorted_rules = sorted(
            flag.targeting_rules, key=lambda r: r.priority, reverse=True
        )
        for rule in sorted_rules:
            try:
                if self._evaluate_condition(rule.condition, context):
                    return rule.variant
            except Exception as e:
                logger.exception(
                    f"Error evaluating rule '{rule.name}' for flag '{flag.key}': {e}"
                )
                continue
        return None

    def _evaluate_condition(self, condition: str, context: EvaluationContext) -> bool:
        """Evaluate a simple condition against context."""
        context_dict = {
            "user_id": context.user_id,
            "user_type": context.user_type,
            "environment": context.environment,
            **context.custom_attributes,
        }
        try:
            return eval(condition, {"__builtins__": {}}, context_dict)
        except Exception:
            return False

    async def _update_metrics(self, result: EvaluationResult) -> None:
        """Update metrics for flag evaluation."""
        async with self._lock:
            if result.flag_key not in self._metrics:
                self._metrics[result.flag_key] = FeatureFlagMetrics(
                    flag_key=result.flag_key
                )
            metrics = self._metrics[result.flag_key]
            metrics.evaluations_count += 1
            metrics.last_evaluated = result.evaluated_at
            if result.variant not in metrics.variant_distribution:
                metrics.variant_distribution[result.variant] = 0
            metrics.variant_distribution[result.variant] += 1

    async def evaluate_flag(
        self, flag_key: str, context: EvaluationContext, default_value: Any = False
    ) -> EvaluationResult:
        """Evaluate a feature flag for the given context.

        Args:
            flag_key: The feature flag key to evaluate
            context: The evaluation context (user, environment, etc.)
            default_value: Default value if flag not found or disabled

        Returns:
            EvaluationResult with the flag value and evaluation details
        """
        try:
            async with self._lock:
                flag = self._flags.get(flag_key)
                if not flag:
                    result = EvaluationResult(
                        flag_key=flag_key,
                        variant="default",
                        value=default_value,
                        reason="FLAG_NOT_FOUND",
                        evaluation_context=context,
                    )
                    await self._update_metrics(result)
                    return result
                if flag.state == FlagState.DISABLED:
                    variant = flag.default_variant
                    value = flag.variants.get(variant, default_value)
                    result = EvaluationResult(
                        flag_key=flag_key,
                        variant=variant,
                        value=value,
                        reason="FLAG_DISABLED",
                        evaluation_context=context,
                    )
                    await self._update_metrics(result)
                    return result
                targeted_variant = self._evaluate_targeting_rules(flag, context)
                if targeted_variant:
                    value = flag.variants.get(targeted_variant, default_value)
                    result = EvaluationResult(
                        flag_key=flag_key,
                        variant=targeted_variant,
                        value=value,
                        reason="TARGETING_MATCH",
                        evaluation_context=context,
                    )
                    await self._update_metrics(result)
                    return result
                if flag.state == FlagState.ROLLOUT and self._evaluate_rollout(
                    flag, context
                ):
                    rollout_variant = (
                        "on"
                        if "on" in flag.variants
                        else next(
                            (
                                v
                                for v in flag.variants
                                if v != flag.default_variant
                            ),
                            flag.default_variant,
                        )
                    )
                    value = flag.variants.get(rollout_variant, default_value)
                    result = EvaluationResult(
                        flag_key=flag_key,
                        variant=rollout_variant,
                        value=value,
                        reason="ROLLOUT_MATCH",
                        evaluation_context=context,
                    )
                    await self._update_metrics(result)
                    return result
                variant = flag.default_variant
                value = flag.variants.get(variant, default_value)
                result = EvaluationResult(
                    flag_key=flag_key,
                    variant=variant,
                    value=value,
                    reason="DEFAULT",
                    evaluation_context=context,
                )
                self._update_metrics(result)
                return result
        except Exception as e:
            logger.exception(f"Error evaluating flag '{flag_key}': {e}")
            async with self._lock:
                if flag_key not in self._metrics:
                    self._metrics[flag_key] = FeatureFlagMetrics(flag_key=flag_key)
                self._metrics[flag_key].error_count += 1
                self._metrics[flag_key].last_error = str(e)
            return EvaluationResult(
                flag_key=flag_key,
                variant="error",
                value=default_value,
                reason="EVALUATION_ERROR",
                evaluation_context=context,
            )

    async def is_enabled(self, flag_key: str, context: EvaluationContext) -> bool:
        """Check if a feature flag is enabled for the given context."""
        result = await self.evaluate_flag(flag_key, context, False)
        return bool(result.value)

    async def get_variant(
        self, flag_key: str, context: EvaluationContext, default: str = "off"
    ) -> str:
        """Get the variant for a feature flag."""
        result = await self.evaluate_flag(flag_key, context, default)
        return result.variant

    async def get_all_flags(self) -> dict[str, FeatureFlagDefinition]:
        """Get all currently loaded feature flags."""
        async with self._lock:
            return self._flags.copy()

    async def get_metrics(
        self, flag_key: str | None = None
    ) -> FeatureFlagMetrics | dict[str, FeatureFlagMetrics]:
        """Get metrics for feature flags."""
        async with self._lock:
            if flag_key:
                return self._metrics.get(
                    flag_key, FeatureFlagMetrics(flag_key=flag_key)
                )
            return self._metrics.copy()

    async def reload_configuration(self) -> None:
        """Manually reload configuration."""
        logger.info("Manually reloading feature flag configuration")
        await self._load_configuration()

    async def get_configuration_info(self) -> dict[str, Any]:
        """Get information about the current configuration."""
        async with self._lock:
            return {
                "config_path": str(self.config_path),
                "loaded_at": self._config_loaded_at.isoformat()
                if self._config_loaded_at
                else None,
                "version": self._config_version,
                "flags_count": len(self._flags),
                "watching_files": bool(self._observer and self._observer.is_alive()),
            }

    def shutdown(self) -> None:
        """Shutdown the feature flag manager and cleanup resources."""
        logger.info("Shutting down feature flag manager")
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
        # Note: shutdown method should be called from async context if needed
        self._flags.clear()
        self._metrics.clear()


_default_manager: FeatureFlagService | None = None


async def initialize_feature_flags(
    config_path: str | Path, watch_files: bool = True
) -> FeatureFlagService:
    """Initialize the global feature flag manager."""
    global _default_manager
    _default_manager = FeatureFlagService(config_path, watch_files)
    await _default_manager.async_init()
    return _default_manager


def get_feature_flag_manager() -> FeatureFlagService | None:
    """Get the global feature flag manager."""
    return _default_manager


async def is_feature_enabled(
    flag_key: str, user_id: str | None = None, **context_attrs
) -> bool:
    """Check if a feature is enabled (convenience function)."""
    if not _default_manager:
        logger.warning("Feature flag manager not initialized")
        return False
    context = EvaluationContext(user_id=user_id, custom_attributes=context_attrs)
    return await _default_manager.is_enabled(flag_key, context)


async def get_feature_variant(
    flag_key: str, user_id: str | None = None, default: str = "off", **context_attrs
) -> str:
    """Get feature variant (convenience function)."""
    if not _default_manager:
        logger.warning("Feature flag manager not initialized")
        return default
    context = EvaluationContext(user_id=user_id, custom_attributes=context_attrs)
    return await _default_manager.get_variant(flag_key, context, default)
