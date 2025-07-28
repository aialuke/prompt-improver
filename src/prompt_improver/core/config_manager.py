"""
Configuration Hot-Reload System for Prompt Improver

A comprehensive configuration management system with hot-reload capabilities,
atomic updates, thread-safe access, and multi-source support following SRE best practices.

Features:
- Hot-reload configuration changes without service restart
- Atomic configuration updates with rollback capability
- Thread-safe configuration access with async/await support
- Multiple configuration sources (files, environment, remote)
- Comprehensive error handling and graceful degradation
- Performance optimization for <100ms reload times
- Extensive monitoring and metrics collection
- Zero-downtime operation with proper event handling

Author: SRE System (Claude Code)
Date: 2025-07-25
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import weakref
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from urllib.parse import urlparse
import aiohttp
import yaml
from pydantic import BaseModel, Field
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ..security.config_validator import SecurityConfigValidator

logger = logging.getLogger(__name__)

class ConfigSourceType(str, Enum):
    """Configuration source types."""
    FILE = "file"
    ENVIRONMENT = "environment"
    REMOTE = "remote"
    MEMORY = "memory"

class ReloadStatus(str, Enum):
    """Configuration reload status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    ROLLED_BACK = "rolled_back"

class ConfigChangeType(str, Enum):
    """Types of configuration changes."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"

@dataclass
class ConfigChange:
    """Represents a single configuration change."""
    path: str
    change_type: ConfigChangeType
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ReloadResult:
    """Result of a configuration reload operation."""
    status: ReloadStatus
    reload_time_ms: float
    changes: List[ConfigChange]
    errors: List[str] = field(default_factory=list)
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ConfigMetrics:
    """Configuration management metrics."""
    total_reloads: int = 0
    successful_reloads: int = 0
    failed_reloads: int = 0
    average_reload_time_ms: float = 0.0
    last_reload_time: Optional[datetime] = None
    rollback_count: int = 0
    active_subscriptions: int = 0
    config_sources_count: int = 0
    current_config_version: str = ""

class ConfigVersion(BaseModel):
    """Configuration version tracking."""
    version_id: str = Field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:12])
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str
    checksum: str
    size_bytes: int = 0
    changes: List[ConfigChange] = Field(default_factory=list)

class ConfigSource(ABC):
    """Abstract base class for configuration sources."""
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority
        self.source_type = ConfigSourceType.MEMORY
        self._last_modified: Optional[datetime] = None
        self._checksum: Optional[str] = None
    
    @abstractmethod
    async def load_config(self) -> Dict[str, Any]:
        """Load configuration from this source."""
        pass
    
    @abstractmethod
    async def is_modified(self) -> bool:
        """Check if the configuration source has been modified."""
        pass
    
    async def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration data."""
        errors = []
        
        # Basic validation
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return False, errors
        
        # Security validation using existing patterns
        try:
            validator = SecurityConfigValidator()
            
            # Check for hardcoded secrets in config values
            config_str = json.dumps(config, default=str)
            for pattern, message in validator.INSECURE_PATTERNS:
                import re
                if re.search(pattern, config_str, re.IGNORECASE):
                    errors.append(f"Security issue: {message}")
        except Exception as e:
            logger.warning(f"Security validation failed: {e}")
        
        return len(errors) == 0, errors
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for configuration data."""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

class FileConfigSource(ConfigSource):
    """File-based configuration source."""
    
    def __init__(self, file_path: Union[str, Path], name: str = None, priority: int = 0):
        self.file_path = Path(file_path)
        super().__init__(name or f"file:{self.file_path.name}", priority)
        self.source_type = ConfigSourceType.FILE
        self._file_mtime: Optional[float] = None
    
    async def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.file_path.exists():
            logger.warning(f"Configuration file not found: {self.file_path}")
            return {}
        
        try:
            # Use async file reading for better performance
            import aiofiles
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Parse based on file extension
            if self.file_path.suffix.lower() in ['.yml', '.yaml']:
                config = yaml.safe_load(content)
            elif self.file_path.suffix.lower() == '.json':
                config = json.loads(content)
            else:
                raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
            
            # Environment variable substitution
            config = self._substitute_env_vars(config)
            
            # Update metadata
            self._file_mtime = self.file_path.stat().st_mtime
            self._checksum = self._calculate_checksum(config)
            self._last_modified = datetime.now(UTC)
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from {self.file_path}: {e}")
            raise
    
    async def is_modified(self) -> bool:
        """Check if file has been modified."""
        if not self.file_path.exists():
            return False
        
        current_mtime = self.file_path.stat().st_mtime
        return self._file_mtime is None or current_mtime != self._file_mtime
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # Extract environment variable name
            env_var = config[2:-1]
            default_value = None
            
            # Handle default values: ${VAR_NAME:default_value}
            if ':' in env_var:
                env_var, default_value = env_var.split(':', 1)
            
            return os.getenv(env_var, default_value)
        else:
            return config

class EnvironmentConfigSource(ConfigSource):
    """Environment variable configuration source."""
    
    def __init__(self, prefix: str = "", name: str = None, priority: int = 10):
        self.prefix = prefix
        super().__init__(name or f"env:{prefix or 'all'}", priority)
        self.source_type = ConfigSourceType.ENVIRONMENT
        self._env_snapshot: Dict[str, str] = {}
    
    async def load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        for key, value in os.environ.items():
            if self.prefix and not key.startswith(self.prefix):
                continue
            
            # Remove prefix if specified
            config_key = key[len(self.prefix):] if self.prefix else key
            config_key = config_key.lower()
            
            # Try to parse as JSON, fallback to string
            try:
                config[config_key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                config[config_key] = value
        
        self._env_snapshot = dict(os.environ)
        self._checksum = self._calculate_checksum(config)
        self._last_modified = datetime.now(UTC)
        
        return config
    
    async def is_modified(self) -> bool:
        """Check if environment variables have changed."""
        current_env = dict(os.environ)
        
        # Check relevant environment variables
        relevant_vars = {k: v for k, v in current_env.items() 
                        if not self.prefix or k.startswith(self.prefix)}
        old_relevant_vars = {k: v for k, v in self._env_snapshot.items() 
                           if not self.prefix or k.startswith(self.prefix)}
        
        return relevant_vars != old_relevant_vars

class RemoteConfigSource(ConfigSource):
    """Remote HTTP-based configuration source."""
    
    def __init__(self, url: str, headers: Dict[str, str] = None, 
                 timeout: int = 10, name: str = None, priority: int = 5):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        super().__init__(name or f"remote:{urlparse(url).netloc}", priority)
        self.source_type = ConfigSourceType.REMOTE
        self._last_etag: Optional[str] = None
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure: Optional[datetime] = None
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = timedelta(minutes=5)
    
    async def load_config(self) -> Dict[str, Any]:
        """Load configuration from remote source."""
        # Circuit breaker check
        if self._is_circuit_open():
            logger.warning(f"Circuit breaker open for {self.url}")
            return {}
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(self.timeout)) as session:
                headers = self.headers.copy()
                if self._last_etag:
                    headers['If-None-Match'] = self._last_etag
                
                async with session.get(self.url, headers=headers) as response:
                    if response.status == 304:  # Not Modified
                        return {}
                    
                    response.raise_for_status()
                    
                    # Update ETag for future requests
                    self._last_etag = response.headers.get('ETag')
                    
                    content_type = response.headers.get('Content-Type', '').lower()
                    
                    if 'application/json' in content_type:
                        config = await response.json()
                    elif 'application/yaml' in content_type or 'text/yaml' in content_type:
                        text = await response.text()
                        config = yaml.safe_load(text)
                    else:
                        text = await response.text()
                        try:
                            config = json.loads(text)
                        except json.JSONDecodeError:
                            config = yaml.safe_load(text)
                    
                    # Reset circuit breaker on success
                    self._circuit_breaker_failures = 0
                    self._circuit_breaker_last_failure = None
                    
                    self._checksum = self._calculate_checksum(config)
                    self._last_modified = datetime.now(UTC)
                    
                    return config
                    
        except Exception as e:
            # Circuit breaker logic
            self._circuit_breaker_failures += 1
            self._circuit_breaker_last_failure = datetime.now(UTC)
            
            logger.error(f"Error loading remote config from {self.url}: {e}")
            if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
                logger.warning(f"Circuit breaker opened for {self.url}")
            
            raise
    
    async def is_modified(self) -> bool:
        """Check if remote configuration has been modified."""
        if self._is_circuit_open():
            return False
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(5)) as session:
                headers = {'If-None-Match': self._last_etag} if self._last_etag else {}
                async with session.head(self.url, headers=headers) as response:
                    return response.status != 304
        except Exception:
            return False
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_breaker_failures < self._circuit_breaker_threshold:
            return False
        
        if not self._circuit_breaker_last_failure:
            return False
        
        time_since_failure = datetime.now(UTC) - self._circuit_breaker_last_failure
        return time_since_failure < self._circuit_breaker_timeout

class ConfigWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes."""
    
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self._last_modified: Dict[str, float] = {}
        self._debounce_time = 0.1  # 100ms debounce for performance
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = event.src_path
        now = time.time()
        
        # Debounce rapid changes
        if file_path in self._last_modified:
            if now - self._last_modified[file_path] < self._debounce_time:
                return
        
        self._last_modified[file_path] = now
        
        try:
            asyncio.create_task(self._async_callback(file_path))
        except Exception as e:
            logger.error(f"Error in config watcher callback: {e}")
    
    async def _async_callback(self, file_path: str):
        """Async wrapper for callback."""
        try:
            await self.callback(file_path)
        except Exception as e:
            logger.error(f"Error in async config callback: {e}")

class ConfigStore:
    """Thread-safe configuration store with atomic updates and rollback."""
    
    def __init__(self, max_versions: int = 10):
        self._current_config: Dict[str, Any] = {}
        self._config_history: List[Tuple[ConfigVersion, Dict[str, Any]]] = []
        self._max_versions = max_versions
        self._lock = asyncio.Lock()
        self._subscribers: Set[weakref.ReferenceType] = set()
    
    async def get_config(self, path: str = None, default: Any = None) -> Any:
        """Get configuration value by path."""
        async with self._lock:
            config = self._current_config
            
            if path is None:
                return config
            
            # Navigate nested dictionary using dot notation
            for key in path.split('.'):
                if isinstance(config, dict) and key in config:
                    config = config[key]
                else:
                    return default
            
            return config
    
    async def update_config(self, new_config: Dict[str, Any], version: ConfigVersion) -> List[ConfigChange]:
        """Atomically update configuration and return changes."""
        async with self._lock:
            # Calculate changes
            changes = self._calculate_changes(self._current_config, new_config)
            
            # Store current config in history
            if self._current_config:
                current_version = ConfigVersion(
                    source="previous",
                    checksum=self._calculate_checksum(self._current_config),
                    size_bytes=len(json.dumps(self._current_config, default=str))
                )
                self._config_history.append((current_version, deepcopy(self._current_config)))
            
            # Limit history size
            while len(self._config_history) > self._max_versions:
                self._config_history.pop(0)
            
            # Atomic update
            self._current_config = deepcopy(new_config)
            
            # Add version to history
            version.changes = changes
            version.size_bytes = len(json.dumps(new_config, default=str))
            self._config_history.append((version, deepcopy(new_config)))
            
            # Notify subscribers
            await self._notify_subscribers(changes)
            
            return changes
    
    async def rollback(self, steps: int = 1) -> bool:
        """Rollback to previous configuration version."""
        async with self._lock:
            if len(self._config_history) < steps + 1:
                return False
            
            # Get target configuration
            target_version, target_config = self._config_history[-(steps + 1)]
            
            # Calculate changes for rollback
            changes = self._calculate_changes(self._current_config, target_config)
            
            # Apply rollback
            self._current_config = deepcopy(target_config)
            
            # Remove rolled back versions from history
            self._config_history = self._config_history[:-(steps)]
            
            # Notify subscribers of rollback
            await self._notify_subscribers(changes)
            
            return True
    
    def get_history(self) -> List[ConfigVersion]:
        """Get configuration version history."""
        return [version for version, _ in self._config_history]
    
    def subscribe(self, callback: Callable[[List[ConfigChange]], None]) -> weakref.ReferenceType:
        """Subscribe to configuration changes."""
        weak_callback = weakref.WeakMethod(callback) if hasattr(callback, '__self__') else weakref.ref(callback)
        self._subscribers.add(weak_callback)
        return weak_callback
    
    def unsubscribe(self, weak_callback: weakref.ReferenceType):
        """Unsubscribe from configuration changes."""
        self._subscribers.discard(weak_callback)
    
    async def _notify_subscribers(self, changes: List[ConfigChange]):
        """Notify all subscribers of configuration changes."""
        dead_refs = set()
        
        for weak_ref in self._subscribers:
            callback = weak_ref()
            if callback is None:
                dead_refs.add(weak_ref)
                continue
            
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(changes)
                else:
                    callback(changes)
            except Exception as e:
                logger.error(f"Error in config change subscriber: {e}")
        
        # Clean up dead references
        self._subscribers -= dead_refs
    
    def _calculate_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[ConfigChange]:
        """Calculate differences between configurations."""
        changes = []
        
        def _compare_dicts(old_dict: Dict, new_dict: Dict, path: str = ""):
            # Find added and modified keys
            for key, new_value in new_dict.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in old_dict:
                    changes.append(ConfigChange(
                        path=current_path,
                        change_type=ConfigChangeType.ADDED,
                        new_value=new_value
                    ))
                elif old_dict[key] != new_value:
                    if isinstance(old_dict[key], dict) and isinstance(new_value, dict):
                        _compare_dicts(old_dict[key], new_value, current_path)
                    else:
                        changes.append(ConfigChange(
                            path=current_path,
                            change_type=ConfigChangeType.MODIFIED,
                            old_value=old_dict[key],
                            new_value=new_value
                        ))
            
            # Find deleted keys
            for key, old_value in old_dict.items():
                current_path = f"{path}.{key}" if path else key
                if key not in new_dict:
                    changes.append(ConfigChange(
                        path=current_path,
                        change_type=ConfigChangeType.DELETED,
                        old_value=old_value
                    ))
        
        _compare_dicts(old_config, new_config)
        return changes
    
    def _calculate_checksum(self, config: Dict[str, Any]) -> str:
        """Calculate checksum for configuration."""
        content = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

class ConfigManager:
    """
    Comprehensive configuration hot-reload system with multi-source support.
    
    Features:
    - Hot-reload configuration changes without service restart
    - Atomic configuration updates with rollback capability  
    - Thread-safe configuration access with async/await support
    - Multiple configuration sources (files, environment, remote)
    - Comprehensive error handling and graceful degradation
    - Performance optimization for <100ms reload times
    - Extensive monitoring and metrics collection
    - Zero-downtime operation
    """
    
    def __init__(self, watch_files: bool = True, max_config_versions: int = 10):
        self.watch_files = watch_files
        self._sources: List[ConfigSource] = []
        self._store = ConfigStore(max_config_versions)
        self._metrics = ConfigMetrics()
        self._observer: Optional[Observer] = None
        self._is_running = False
        self._startup_lock = asyncio.Lock()
        
        # Performance tracking
        self._reload_times: List[float] = []
        self._max_reload_time_samples = 100
        
        logger.info("ConfigManager initialized")
    
    async def add_source(self, source: ConfigSource):
        """Add a configuration source."""
        self._sources.append(source)
        self._sources.sort(key=lambda s: s.priority, reverse=True)
        self._metrics.config_sources_count = len(self._sources)
        
        logger.info(f"Added config source: {source.name} (priority: {source.priority})")
    
    async def start(self):
        """Start the configuration manager."""
        async with self._startup_lock:
            if self._is_running:
                return
            
            # Initial configuration load
            await self._reload_all_sources()
            
            # Start file watcher if enabled
            if self.watch_files:
                await self._start_file_watcher()
            
            self._is_running = True
            logger.info("ConfigManager started")
    
    async def stop(self):
        """Stop the configuration manager."""
        async with self._startup_lock:
            if not self._is_running:
                return
            
            # Stop file watcher
            if self._observer:
                self._observer.stop()
                self._observer.join(timeout=5.0)
                self._observer = None
            
            self._is_running = False
            logger.info("ConfigManager stopped")
    
    async def get_config(self, path: str = None, default: Any = None) -> Any:
        """Get configuration value by path."""
        return await self._store.get_config(path, default)
    
    async def get_section(self, section: str) -> Dict[str, Any]:
        """Get a configuration section."""
        section_config = await self._store.get_config(section, {})
        return section_config if isinstance(section_config, dict) else {}
    
    async def reload(self, source_name: str = None) -> ReloadResult:
        """Reload configuration from sources."""
        start_time = time.time()
        
        try:
            if source_name:
                # Reload specific source
                source = next((s for s in self._sources if s.name == source_name), None)
                if not source:
                    return ReloadResult(
                        status=ReloadStatus.FAILED,
                        reload_time_ms=0,
                        changes=[],
                        errors=[f"Source '{source_name}' not found"]
                    )
                result = await self._reload_source(source)
            else:
                # Reload all sources
                result = await self._reload_all_sources()
            
            # Update metrics
            reload_time_ms = (time.time() - start_time) * 1000
            self._update_reload_metrics(reload_time_ms, result.status == ReloadStatus.SUCCESS)
            result.reload_time_ms = reload_time_ms
            
            return result
            
        except Exception as e:
            logger.error(f"Error during config reload: {e}")
            reload_time_ms = (time.time() - start_time) * 1000
            self._update_reload_metrics(reload_time_ms, False)
            
            return ReloadResult(
                status=ReloadStatus.FAILED,
                reload_time_ms=reload_time_ms,
                changes=[],
                errors=[str(e)]
            )
    
    async def rollback(self, steps: int = 1) -> ReloadResult:
        """Rollback to previous configuration version."""
        start_time = time.time()
        
        try:
            success = await self._store.rollback(steps)
            reload_time_ms = (time.time() - start_time) * 1000
            
            if success:
                self._metrics.rollback_count += 1
                return ReloadResult(
                    status=ReloadStatus.ROLLED_BACK,
                    reload_time_ms=reload_time_ms,
                    changes=[],  # Changes would be calculated in rollback
                    source="rollback"
                )
            else:
                return ReloadResult(
                    status=ReloadStatus.FAILED,
                    reload_time_ms=reload_time_ms,
                    changes=[],
                    errors=["Insufficient configuration history for rollback"]
                )
                
        except Exception as e:
            logger.error(f"Error during config rollback: {e}")
            return ReloadResult(
                status=ReloadStatus.FAILED,
                reload_time_ms=(time.time() - start_time) * 1000,
                changes=[],
                errors=[str(e)]
            )
    
    def subscribe(self, callback: Callable[[List[ConfigChange]], None]) -> weakref.ReferenceType:
        """Subscribe to configuration changes."""
        subscription = self._store.subscribe(callback)
        self._metrics.active_subscriptions = len(self._store._subscribers)
        return subscription
    
    def unsubscribe(self, subscription: weakref.ReferenceType):
        """Unsubscribe from configuration changes."""
        self._store.unsubscribe(subscription)
        self._metrics.active_subscriptions = len(self._store._subscribers)
    
    def get_metrics(self) -> ConfigMetrics:
        """Get configuration management metrics."""
        return self._metrics
    
    def get_sources(self) -> List[ConfigSource]:
        """Get all configuration sources."""
        return self._sources.copy()
    
    def get_version_history(self) -> List[ConfigVersion]:
        """Get configuration version history."""
        return self._store.get_history()
    
    async def _reload_all_sources(self) -> ReloadResult:
        """Reload configuration from all sources."""
        merged_config = {}
        errors = []
        all_changes = []
        successful_sources = 0
        
        # Load from all sources in priority order
        for source in self._sources:
            try:
                if await source.is_modified() or not merged_config:
                    source_config = await source.load_config()
                    
                    # Validate configuration
                    is_valid, validation_errors = await source.validate_config(source_config)
                    if not is_valid:
                        errors.extend([f"{source.name}: {error}" for error in validation_errors])
                        continue
                    
                    # Merge configuration (higher priority sources override lower priority)
                    merged_config = self._merge_configs(source_config, merged_config)
                    successful_sources += 1
                    
                    logger.debug(f"Loaded config from {source.name}")
                    
            except Exception as e:
                error_msg = f"Error loading from {source.name}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        if not merged_config and errors:
            return ReloadResult(
                status=ReloadStatus.FAILED,
                reload_time_ms=0,
                changes=[],
                errors=errors
            )
        
        # Update configuration store
        version = ConfigVersion(
            source=f"merged:{successful_sources}_sources",
            checksum="",  # Will be calculated in store
        )
        
        try:
            changes = await self._store.update_config(merged_config, version)
            all_changes.extend(changes)
            
            # Update metrics
            self._metrics.current_config_version = version.version_id
            self._metrics.last_reload_time = datetime.now(UTC)
            
            if changes:
                logger.info(f"Configuration updated with {len(changes)} changes")
                for change in changes[:5]:  # Log first 5 changes
                    logger.debug(f"Config change: {change.change_type.value} {change.path}")
            
            status = ReloadStatus.PARTIAL if errors else ReloadStatus.SUCCESS
            return ReloadResult(
                status=status,
                reload_time_ms=0,  # Will be set by caller
                changes=all_changes,
                errors=errors,
                source="all_sources"
            )
            
        except Exception as e:
            logger.error(f"Error updating configuration store: {e}")
            return ReloadResult(
                status=ReloadStatus.FAILED,
                reload_time_ms=0,
                changes=[],
                errors=errors + [str(e)]
            )
    
    async def _reload_source(self, source: ConfigSource) -> ReloadResult:
        """Reload configuration from a specific source."""
        try:
            if not await source.is_modified():
                return ReloadResult(
                    status=ReloadStatus.SUCCESS,
                    reload_time_ms=0,
                    changes=[],
                    source=source.name
                )
            
            source_config = await source.load_config()
            
            # Validate configuration
            is_valid, validation_errors = await source.validate_config(source_config)
            if not is_valid:
                return ReloadResult(
                    status=ReloadStatus.FAILED,
                    reload_time_ms=0,
                    changes=[],
                    errors=validation_errors,
                    source=source.name
                )
            
            # For single source reload, we need to rebuild the entire merged config
            return await self._reload_all_sources()
            
        except Exception as e:
            logger.error(f"Error reloading source {source.name}: {e}")
            return ReloadResult(
                status=ReloadStatus.FAILED,
                reload_time_ms=0,
                changes=[],
                errors=[str(e)],
                source=source.name
            )
    
    async def _start_file_watcher(self):
        """Start file system watcher for configuration files."""
        if self._observer:
            return
        
        file_sources = [s for s in self._sources if isinstance(s, FileConfigSource)]
        if not file_sources:
            return
        
        try:
            self._observer = Observer()
            watcher = ConfigWatcher(self._on_config_file_changed)
            
            # Watch directories containing config files
            watched_dirs = set()
            for source in file_sources:
                watch_dir = source.file_path.parent
                if watch_dir not in watched_dirs:
                    self._observer.schedule(watcher, str(watch_dir), recursive=False)
                    watched_dirs.add(watch_dir)
                    logger.debug(f"Watching directory: {watch_dir}")
            
            self._observer.start()
            logger.info(f"Started file watcher for {len(watched_dirs)} directories")
            
        except Exception as e:
            logger.error(f"Error starting file watcher: {e}")
    
    async def _on_config_file_changed(self, file_path: str):
        """Handle configuration file changes."""
        logger.info(f"Configuration file changed: {file_path}")
        
        # Find matching source
        changed_source = None
        for source in self._sources:
            if isinstance(source, FileConfigSource) and str(source.file_path) == file_path:
                changed_source = source
                break
        
        if changed_source:
            try:
                result = await self.reload(changed_source.name)
                if result.status != ReloadStatus.SUCCESS:
                    logger.warning(f"Config reload failed: {result.errors}")
                else:
                    logger.info(f"Config reloaded successfully in {result.reload_time_ms:.1f}ms")
            except Exception as e:
                logger.error(f"Error handling config file change: {e}")
    
    def _merge_configs(self, high_priority: Dict[str, Any], low_priority: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configurations with high priority values overriding low priority."""
        result = deepcopy(low_priority)
        
        for key, value in high_priority.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                # Recursively merge nested dictionaries
                result[key] = self._merge_configs(value, result[key])
            else:
                # Override with high priority value
                result[key] = deepcopy(value)
        
        return result
    
    def _update_reload_metrics(self, reload_time_ms: float, success: bool):
        """Update reload performance metrics."""
        self._metrics.total_reloads += 1
        
        if success:
            self._metrics.successful_reloads += 1
        else:
            self._metrics.failed_reloads += 1
        
        # Update average reload time
        self._reload_times.append(reload_time_ms)
        if len(self._reload_times) > self._max_reload_time_samples:
            self._reload_times.pop(0)
        
        self._metrics.average_reload_time_ms = sum(self._reload_times) / len(self._reload_times)
        
        # Log performance warning if reload is slow
        if reload_time_ms > 100:
            logger.warning(f"Slow config reload: {reload_time_ms:.1f}ms (target: <100ms)")
    
    @asynccontextmanager
    async def managed_lifecycle(self):
        """Context manager for proper lifecycle management."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()

# Convenience functions for common usage patterns
_default_manager: Optional[ConfigManager] = None

async def initialize_config_manager(config_sources: List[ConfigSource] = None, 
                                  watch_files: bool = True) -> ConfigManager:
    """Initialize the global configuration manager."""
    global _default_manager
    
    _default_manager = ConfigManager(watch_files=watch_files)
    
    if config_sources:
        for source in config_sources:
            await _default_manager.add_source(source)
    
    await _default_manager.start()
    return _default_manager

def get_config_manager() -> Optional[ConfigManager]:
    """Get the global configuration manager."""
    return _default_manager

async def get_config(path: str = None, default: Any = None) -> Any:
    """Get configuration value (convenience function)."""
    if not _default_manager:
        logger.warning("Configuration manager not initialized")
        return default
    
    return await _default_manager.get_config(path, default)

async def get_config_section(section: str) -> Dict[str, Any]:
    """Get configuration section (convenience function)."""
    if not _default_manager:
        logger.warning("Configuration manager not initialized")
        return {}
    
    return await _default_manager.get_section(section)

# Type hints for better IDE support
__all__ = [
    'ConfigManager',
    'ConfigSource',
    'FileConfigSource', 
    'EnvironmentConfigSource',
    'RemoteConfigSource',
    'ConfigMetrics',
    'ReloadResult',
    'ConfigChange',
    'initialize_config_manager',
    'get_config_manager',
    'get_config',
    'get_config_section'
]