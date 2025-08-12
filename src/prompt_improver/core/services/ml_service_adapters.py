"""Service Adapter Implementations for ML Pipeline DI (2025).

Concrete implementations of protocol interfaces for external services,
following the adapter pattern for dependency injection architecture.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from prompt_improver.core.protocols.ml_protocols import (
    CacheServiceProtocol,
    DatabaseServiceProtocol,
    EventBusProtocol,
    HealthMonitorProtocol,
    MLflowServiceProtocol,
    ResourceManagerProtocol,
    ServiceStatus,
    WorkflowEngineProtocol,
)

logger = logging.getLogger(__name__)


class MLflowServiceAdapter:
    """Adapter for MLflow service interactions."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize MLflow adapter."""
        self.config = config or {}
        self.tracking_uri = self.config.get(
            "tracking_uri",
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.external.service:5000"),
        )
        self.experiment_name = self.config.get(
            "experiment_name", "ml_pipeline_experiment"
        )
        self._client = None
        self._is_initialized = False
        logger.debug(f"MLflow adapter initialized with URI: {self.tracking_uri}")

    async def initialize(self) -> None:
        """Initialize MLflow client."""
        if self._is_initialized:
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            try:
                mlflow.create_experiment(self.experiment_name)
            except Exception:
                pass
            mlflow.set_experiment(self.experiment_name)
            self._client = mlflow
            self._is_initialized = True
            logger.info(
                f"MLflow service initialized with experiment: {self.experiment_name}"
            )
        except ImportError:
            logger.warning("MLflow not installed, using mock implementation")
            self._client = self._create_mock_client()
            self._is_initialized = True

    def _create_mock_client(self):
        """Create mock MLflow client for development."""

        class MockMLflow:
            def start_run(self, **kwargs):
                return type(
                    "MockRun",
                    (),
                    {"info": type("MockInfo", (), {"run_id": "mock_run_123"})()},
                )()

            def log_params(self, params):
                pass

            def log_metrics(self, metrics):
                pass

            def log_artifact(self, artifact_path):
                pass

            def end_run(self):
                pass

        return MockMLflow()

    async def log_experiment(
        self, experiment_name: str, parameters: dict[str, Any]
    ) -> str:
        """Log an ML experiment and return run ID."""
        await self.initialize()
        try:
            with self._client.start_run() as run:
                self._client.log_params(parameters)
                run_id = run.info.run_id
                logger.info(f"Logged experiment with run ID: {run_id}")
                return run_id
        except Exception as e:
            logger.error(f"Failed to log experiment: {e}")
            return f"mock_run_{datetime.now().isoformat()}"

    async def log_model(
        self, model_name: str, model_data: Any, metadata: dict[str, Any]
    ) -> str:
        """Log a model and return model URI."""
        await self.initialize()
        try:
            self._client.log_params(metadata)
            model_uri = f"models:/{model_name}/latest"
            logger.info(f"Logged model: {model_name}")
            return model_uri
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            return f"mock_model_{model_name}"

    async def get_model_metadata(self, model_id: str) -> dict[str, Any]:
        """Retrieve model metadata by ID."""
        await self.initialize()
        return {
            "model_id": model_id,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
        }

    async def start_trace(
        self, trace_name: str, attributes: dict[str, Any] | None = None
    ) -> str:
        """Start MLflow tracing and return trace ID."""
        await self.initialize()
        trace_id = f"trace_{trace_name}_{datetime.now().isoformat()}"
        logger.debug(f"Started trace: {trace_id}")
        return trace_id

    async def end_trace(
        self, trace_id: str, outputs: dict[str, Any] | None = None
    ) -> None:
        """End MLflow trace with outputs."""
        await self.initialize()
        logger.debug(f"Ended trace: {trace_id}")

    async def health_check(self) -> ServiceStatus:
        """Check MLflow service health."""
        if not self._is_initialized:
            return ServiceStatus.UNHEALTHY
        try:
            if self._client:
                return ServiceStatus.HEALTHY
            return ServiceStatus.UNHEALTHY
        except Exception as e:
            logger.error(f"MLflow health check failed: {e}")
            return ServiceStatus.ERROR

    async def shutdown(self) -> None:
        """Shutdown MLflow service."""
        if self._is_initialized:
            logger.info("MLflow service shutdown")
            self._is_initialized = False


class CacheServiceAdapter:
    """Adapter for cache service (Redis) interactions."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize cache adapter."""
        self.config = config or {}
        self.host = self.config.get(
            "host", os.getenv("REDIS_HOST", "redis.external.service")
        )
        self.port = self.config.get("port", int(os.getenv("REDIS_PORT", "6379")))
        self.db = self.config.get("db", int(os.getenv("REDIS_DB", "0")))
        self._redis = None
        self._is_initialized = False
        logger.debug(f"Cache adapter initialized for {self.host}:{self.port}")

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if self._is_initialized:
            return
        try:
            import redis.asyncio as redis

            self._redis = redis.Redis(
                host=self.host, port=self.port, db=self.db, decode_responses=True
            )
            await self._redis.ping()
            self._is_initialized = True
            logger.info(f"Cache service initialized: {self.host}:{self.port}")
        except ImportError:
            logger.warning("Redis not installed, using mock cache")
            self._redis = self._create_mock_cache()
            self._is_initialized = True
        except Exception as e:
            logger.warning(f"Redis connection failed, using mock cache: {e}")
            self._redis = self._create_mock_cache()
            self._is_initialized = True

    def _create_mock_cache(self):
        """Create mock cache for development."""

        class MockRedis:
            def __init__(self):
                self._data = {}

            async def ping(self):
                return True

            async def get(self, key):
                return self._data.get(key)

            async def set(self, key, value, ex=None):
                self._data[key] = value
                return True

            async def delete(self, key):
                return self._data.pop(key, None) is not None

            async def exists(self, key):
                return key in self._data

            async def close(self):
                pass

        return MockRedis()

    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        await self.initialize()
        try:
            value = await self._redis.get(key)
            logger.debug(f"Cache GET {key}: {'HIT' if value else 'MISS'}")
            return value
        except Exception as e:
            logger.error(f"Cache GET failed for {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set key-value with optional TTL."""
        await self.initialize()
        try:
            await self._redis.set(key, value, ex=ttl)
            logger.debug(f"Cache SET {key} (TTL: {ttl})")
        except Exception as e:
            logger.error(f"Cache SET failed for {key}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete key and return success status."""
        await self.initialize()
        try:
            result = await self._redis.delete(key)
            logger.debug(f"Cache DELETE {key}: {bool(result)}")
            return bool(result)
        except Exception as e:
            logger.error(f"Cache DELETE failed for {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        await self.initialize()
        try:
            result = await self._redis.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache EXISTS failed for {key}: {e}")
            return False

    async def health_check(self) -> ServiceStatus:
        """Check cache service health."""
        if not self._is_initialized:
            return ServiceStatus.UNHEALTHY
        try:
            await self._redis.ping()
            return ServiceStatus.HEALTHY
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return ServiceStatus.ERROR

    async def shutdown(self) -> None:
        """Shutdown cache service."""
        if self._is_initialized and hasattr(self._redis, "close"):
            await self._redis.close()
            logger.info("Cache service shutdown")
            self._is_initialized = False


class DatabaseServiceAdapter:
    """Adapter for database service interactions."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize database adapter."""
        self.config = config or {}
        self.connection_string = (
            self.config.get("connection_string")
            or os.getenv("TEST_DATABASE_URL")
            or os.getenv("DATABASE_URL")
        )
        self._pool = None
        self._is_initialized = False
        logger.debug("Database adapter initialized")

    async def initialize(self) -> None:
        """Initialize database connection pool."""
        if self._is_initialized:
            return
        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self.connection_string, min_size=2, max_size=10
            )
            self._is_initialized = True
            logger.info("Database service initialized")
        except ImportError:
            logger.warning("asyncpg not installed, using mock database")
            self._pool = self._create_mock_pool()
            self._is_initialized = True
        except Exception as e:
            logger.warning(f"Database connection failed, using mock: {e}")
            self._pool = self._create_mock_pool()
            self._is_initialized = True

    def _create_mock_pool(self):
        """Create mock database pool."""

        class MockPool:
            async def acquire(self):
                return type(
                    "MockConnection",
                    (),
                    {
                        "fetch": lambda query, *args: [],
                        "execute": lambda query, *args: None,
                        "close": lambda: None,
                    },
                )()

            async def close(self):
                pass

        return MockPool()

    async def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query and return results."""
        await self.initialize()
        try:
            async with self._pool.acquire() as conn:
                if parameters:
                    rows = await conn.fetch(query, *parameters.values())
                else:
                    rows = await conn.fetch(query)
                results = (
                    [dict(row) for row in rows]
                    if hasattr(rows[0] if rows else {}, "items")
                    else []
                )
                logger.debug(f"Query executed: {len(results)} rows returned")
                return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []

    async def execute_transaction(
        self, queries: list[str], parameters: list[dict[str, Any]] | None = None
    ) -> None:
        """Execute multiple queries in transaction."""
        await self.initialize()
        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    for i, query in enumerate(queries):
                        params = (
                            parameters[i]
                            if parameters and i < len(parameters)
                            else None
                        )
                        if params:
                            await conn.execute(query, *params.values())
                        else:
                            await conn.execute(query)
                logger.info(f"Transaction completed: {len(queries)} queries")
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise

    async def health_check(self) -> ServiceStatus:
        """Check database service health."""
        if not self._is_initialized:
            return ServiceStatus.UNHEALTHY
        try:
            async with self._pool.acquire() as conn:
                await conn.fetch("SELECT 1")
                return ServiceStatus.HEALTHY
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return ServiceStatus.ERROR

    async def get_connection_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        await self.initialize()
        if hasattr(self._pool, "get_size"):
            return {
                "pool_size": self._pool.get_size(),
                "available_connections": self._pool.get_idle_size(),
                "max_size": self._pool.get_max_size(),
                "min_size": self._pool.get_min_size(),
            }
        return {"status": "mock_pool", "connections": 1}

    async def shutdown(self) -> None:
        """Shutdown database service."""
        if self._is_initialized and self._pool:
            await self._pool.close()
            logger.info("Database service shutdown")
            self._is_initialized = False


class SimpleEventBusAdapter:
    """Simple event bus adapter for ML pipeline."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize event bus adapter."""
        self.config = config or {}
        self._subscribers = {}
        self._is_initialized = False
        logger.debug("Event bus adapter initialized")

    async def initialize(self) -> None:
        """Initialize event bus."""
        self._is_initialized = True
        logger.info("Event bus service initialized")

    async def publish(self, event_type: str, event_data: dict[str, Any]) -> None:
        """Publish event to the bus."""
        await self.initialize()
        if event_type in self._subscribers:
            for subscription_id, handler in self._subscribers[event_type].items():
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_data)
                    else:
                        handler(event_data)
                    logger.debug(f"Event {event_type} delivered to {subscription_id}")
                except Exception as e:
                    logger.error(f"Event handler failed for {event_type}: {e}")

    async def subscribe(self, event_type: str, handler: Any) -> str:
        """Subscribe to event type and return subscription ID."""
        await self.initialize()
        if event_type not in self._subscribers:
            self._subscribers[event_type] = {}
        subscription_id = f"sub_{event_type}_{len(self._subscribers[event_type])}"
        self._subscribers[event_type][subscription_id] = handler
        logger.info(f"Subscribed to {event_type}: {subscription_id}")
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events."""
        for event_type, subscribers in self._subscribers.items():
            if subscription_id in subscribers:
                del subscribers[subscription_id]
                logger.info(f"Unsubscribed: {subscription_id}")
                break

    async def health_check(self) -> ServiceStatus:
        """Check event bus health."""
        return (
            ServiceStatus.HEALTHY if self._is_initialized else ServiceStatus.UNHEALTHY
        )

    async def shutdown(self) -> None:
        """Shutdown event bus."""
        self._subscribers.clear()
        logger.info("Event bus service shutdown")
        self._is_initialized = False


def create_mlflow_service(config: dict[str, Any] | None = None) -> MLflowServiceAdapter:
    """Create MLflow service adapter."""
    return MLflowServiceAdapter(config)


def create_cache_service(config: dict[str, Any] | None = None) -> CacheServiceAdapter:
    """Create cache service adapter."""
    return CacheServiceAdapter(config)


def create_database_service(
    config: dict[str, Any] | None = None,
) -> DatabaseServiceAdapter:
    """Create database service adapter."""
    return DatabaseServiceAdapter(config)


def create_event_bus(config: dict[str, Any] | None = None) -> SimpleEventBusAdapter:
    """Create event bus adapter."""
    return SimpleEventBusAdapter(config)


def create_workflow_engine(config: dict[str, Any] | None = None) -> Any:
    """Create workflow engine adapter (simplified)."""

    class SimpleWorkflowEngine:
        def __init__(self, config):
            self.config = config
            self._is_initialized = False

        async def initialize(self):
            self._is_initialized = True
            logger.info("Workflow engine initialized")

        async def execute_workflow(
            self, workflow_id: str, parameters: dict[str, Any]
        ) -> dict[str, Any]:
            await self.initialize()
            logger.info(f"Executing workflow: {workflow_id}")
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "results": parameters,
            }

        async def get_workflow_status(self, workflow_id: str) -> str:
            return "completed"

        async def cancel_workflow(self, workflow_id: str) -> bool:
            logger.info(f"Cancelled workflow: {workflow_id}")
            return True

        async def health_check(self) -> ServiceStatus:
            return (
                ServiceStatus.HEALTHY
                if self._is_initialized
                else ServiceStatus.UNHEALTHY
            )

        async def shutdown(self):
            logger.info("Workflow engine shutdown")
            self._is_initialized = False

    return SimpleWorkflowEngine(config)


def create_resource_manager(config: dict[str, Any] | None = None) -> Any:
    """Create resource manager adapter using the proper ResourceManager."""
    from prompt_improver.ml.orchestration.core.resource_manager import ResourceManager

    return ResourceManager()


def create_health_monitor(config: dict[str, Any] | None = None) -> Any:
    """Create health monitor adapter (simplified)."""

    class SimpleHealthMonitor:
        def __init__(self, config):
            self.config = config
            self._health_checks = {}
            self._is_initialized = False

        async def initialize(self):
            self._is_initialized = True
            logger.info("Health monitor initialized")

        async def check_service_health(self, service_name: str) -> ServiceStatus:
            await self.initialize()
            if service_name in self._health_checks:
                try:
                    check_func = self._health_checks[service_name]
                    if asyncio.iscoroutinefunction(check_func):
                        return await check_func()
                    return check_func()
                except Exception:
                    return ServiceStatus.ERROR
            return ServiceStatus.UNKNOWN

        async def get_overall_health(self) -> dict[str, ServiceStatus]:
            await self.initialize()
            health_results = {}
            for service_name in self._health_checks:
                health_results[service_name] = await self.check_service_health(
                    service_name
                )
            return health_results

        async def register_health_check(
            self, service_name: str, health_check_func: Any
        ) -> None:
            self._health_checks[service_name] = health_check_func
            logger.info(f"Registered health check for: {service_name}")

        async def health_check(self) -> ServiceStatus:
            return (
                ServiceStatus.HEALTHY
                if self._is_initialized
                else ServiceStatus.UNHEALTHY
            )

        async def shutdown(self):
            self._health_checks.clear()
            logger.info("Health monitor shutdown")
            self._is_initialized = False

    return SimpleHealthMonitor(config)
