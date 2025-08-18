"""Redis testcontainer infrastructure for real L2RedisService testing.

This module provides comprehensive Redis container integration for validating
L2RedisService simplifications with real behavior testing, ensuring connection
management, performance tracking, and error handling work correctly.

Features:
- Real Redis instances using testcontainers
- Connection management validation
- Performance tracking verification  
- Error handling and recovery testing
- Network timeout simulation
- Graceful shutdown validation
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

try:
    from testcontainers.redis import RedisContainer
except ImportError:
    # Fallback for environments without testcontainers
    RedisContainer = None

logger = logging.getLogger(__name__)


class RedisTestContainer:
    """Enhanced Redis testcontainer for real L2RedisService validation.
    
    Provides comprehensive Redis testing infrastructure with:
    - Real Redis instances via testcontainers
    - Connection lifecycle management
    - Performance monitoring validation
    - Error injection and recovery testing
    - Network simulation capabilities
    """

    def __init__(
        self,
        redis_version: str = "7-alpine",
        port: Optional[int] = None,
        password: Optional[str] = None,
    ):
        """Initialize Redis testcontainer.
        
        Args:
            redis_version: Redis version to use (default: 7-alpine)
            port: Container port (auto-assigned if None)
            password: Redis password (optional)
        """
        if RedisContainer is None:
            raise ImportError("testcontainers package is required for Redis testing")
            
        self.redis_version = redis_version
        self.port = port
        self.password = password
        self.container_id = f"redis_test_{uuid.uuid4().hex[:8]}"
        
        self._container: Optional[RedisContainer] = None
        self._connection_url: Optional[str] = None
        self._host: Optional[str] = None
        self._exposed_port: Optional[int] = None

    async def start(self) -> "RedisTestContainer":
        """Start Redis container and validate connectivity."""
        try:
            # Create and configure Redis container
            self._container = RedisContainer(
                image=f"redis:{self.redis_version}",
                port=6379
            )
            
            # Add password if specified
            if self.password:
                self._container = self._container.with_env("REDIS_PASSWORD", self.password)
                self._container = self._container.with_command(f"redis-server --requirepass {self.password}")
            
            # Bind to specific port if requested
            if self.port:
                self._container = self._container.with_bind_ports(6379, self.port)
            
            # Start container
            self._container.start()
            
            # Get connection details
            self._host = self._container.get_container_host_ip()
            self._exposed_port = self._container.get_exposed_port(6379)
            
            # Build connection details
            if self.password:
                self._connection_url = f"redis://:{self.password}@{self._host}:{self._exposed_port}/0"
            else:
                self._connection_url = f"redis://{self._host}:{self._exposed_port}/0"
            
            # Wait for Redis to be ready
            await self._wait_for_readiness()
            
            logger.info(
                f"Redis testcontainer started: {self.container_id} "
                f"(version {self.redis_version}, port {self._exposed_port})"
            )
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to start Redis testcontainer: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop Redis container and clean up resources."""
        try:
            if self._container:
                self._container.stop()
                self._container = None
                
            logger.info(f"Redis testcontainer stopped: {self.container_id}")
            
        except Exception as e:
            logger.warning(f"Error stopping Redis testcontainer: {e}")

    async def _wait_for_readiness(self, max_retries: int = 30, retry_delay: float = 0.5):
        """Wait for Redis to be ready for connections."""
        import coredis
        
        for attempt in range(max_retries):
            try:
                # Create basic Redis client for connectivity test
                client_config = {
                    "host": self._host,
                    "port": self._exposed_port,
                    "db": 0,
                    "socket_connect_timeout": 2,
                    "socket_timeout": 2,
                    "decode_responses": False,
                }
                
                if self.password:
                    client_config["password"] = self.password
                    
                client = coredis.Redis(**client_config)
                await client.ping()
                # coredis doesn't have close(), connection is managed internally
                
                logger.debug(f"Redis ready after {attempt + 1} attempts")
                return
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Redis not ready after {max_retries} attempts: {e}")
                await asyncio.sleep(retry_delay)

    def get_connection_info(self) -> Dict[str, Any]:
        """Get Redis connection information."""
        if not self._container or not self._host:
            raise RuntimeError("Container not started. Call start() first.")
            
        return {
            "host": self._host,
            "port": self._exposed_port,
            "password": self.password,
            "connection_url": self._connection_url,
            "database": 0,
        }

    def set_env_vars(self) -> None:
        """Set environment variables for L2RedisService to use this container."""
        connection_info = self.get_connection_info()
        
        os.environ["REDIS_HOST"] = connection_info["host"]
        os.environ["REDIS_PORT"] = str(connection_info["port"])
        os.environ["REDIS_DB"] = "0"
        
        if self.password:
            os.environ["REDIS_PASSWORD"] = self.password
        elif "REDIS_PASSWORD" in os.environ:
            del os.environ["REDIS_PASSWORD"]

    async def simulate_network_failure(self, duration_seconds: float = 2.0):
        """Simulate network failure by pausing container."""
        if not self._container:
            raise RuntimeError("Container not started")
            
        try:
            # Pause container to simulate network failure
            self._container.get_wrapped_container().pause()
            logger.debug(f"Simulated network failure for {duration_seconds}s")
            
            await asyncio.sleep(duration_seconds)
            
            # Resume container
            self._container.get_wrapped_container().unpause()
            logger.debug("Network failure simulation ended")
            
        except Exception as e:
            logger.error(f"Error simulating network failure: {e}")
            # Attempt to resume container
            try:
                self._container.get_wrapped_container().unpause()
            except:
                pass

    async def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        import coredis
        
        try:
            connection_info = self.get_connection_info()
            client = coredis.Redis(
                host=connection_info["host"],
                port=connection_info["port"],
                password=connection_info.get("password"),
                db=0,
            )
            
            info = await client.info()
            # coredis doesn't have close(), connection is managed internally  
            return info
            
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}

    async def flush_database(self):
        """Flush Redis database for clean tests."""
        import coredis
        
        try:
            connection_info = self.get_connection_info()
            client = coredis.Redis(
                host=connection_info["host"],
                port=connection_info["port"],
                password=connection_info.get("password"),
                db=0,
            )
            
            await client.flushdb()
            # coredis doesn't have close(), connection is managed internally
            logger.debug("Redis database flushed")
            
        except Exception as e:
            logger.error(f"Failed to flush Redis database: {e}")

    async def __aenter__(self) -> "RedisTestContainer":
        """Async context manager entry."""
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class RedisTestFixture:
    """Test fixture helper for Redis testcontainers.
    
    Provides convenience methods for L2RedisService validation and testing patterns.
    """

    def __init__(self, container: RedisTestContainer):
        self.container = container

    async def measure_operation_performance(
        self, 
        operation: str, 
        key: str = "test_key", 
        value: Any = None, 
        iterations: int = 100
    ) -> Dict[str, Any]:
        """Measure Redis operation performance."""
        import coredis
        
        connection_info = self.container.get_connection_info()
        client = coredis.Redis(
            host=connection_info["host"],
            port=connection_info["port"],
            password=connection_info.get("password"),
            db=0,
        )
        
        times = []
        successful = 0
        failed = 0
        
        try:
            for i in range(iterations):
                start_time = time.perf_counter()
                
                try:
                    if operation == "GET":
                        await client.get(f"{key}_{i}")
                    elif operation == "SET":
                        test_value = value or f"test_value_{i}"
                        # Serialize value like L2RedisService does
                        import json
                        serialized_value = json.dumps(test_value, default=str).encode("utf-8")
                        await client.set(f"{key}_{i}", serialized_value)
                    elif operation == "DELETE":
                        await client.delete([f"{key}_{i}"])
                    
                    successful += 1
                    
                except Exception:
                    failed += 1
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        finally:
            # coredis doesn't have close(), connection is managed internally
            pass
        
        return {
            "operation": operation,
            "iterations": iterations,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / iterations if iterations > 0 else 0,
            "avg_time_ms": sum(times) / len(times) if times else 0,
            "min_time_ms": min(times) if times else 0,
            "max_time_ms": max(times) if times else 0,
            "p95_time_ms": sorted(times)[int(len(times) * 0.95)] if times else 0,
            "times": times,
        }

    async def test_connection_recovery(self, failure_duration: float = 1.0) -> Dict[str, Any]:
        """Test connection recovery after network failure."""
        import coredis
        
        connection_info = self.container.get_connection_info()
        client = coredis.Redis(
            host=connection_info["host"],
            port=connection_info["port"],
            password=connection_info.get("password"),
            db=0,
        )
        
        recovery_results = {
            "pre_failure_success": False,
            "during_failure_success": False,
            "post_failure_success": False,
            "recovery_time_ms": None,
        }
        
        try:
            # Test before failure
            await client.set("test_recovery", "pre_failure")
            result = await client.get("test_recovery")
            recovery_results["pre_failure_success"] = (result == b"pre_failure")
            
            # Simulate network failure
            failure_task = asyncio.create_task(
                self.container.simulate_network_failure(failure_duration)
            )
            
            # Wait a bit then test during failure
            await asyncio.sleep(0.1)
            try:
                await client.set("test_recovery", "during_failure")
                recovery_results["during_failure_success"] = True
            except:
                recovery_results["during_failure_success"] = False
            
            # Wait for failure simulation to complete
            await failure_task
            
            # Test recovery
            recovery_start = time.perf_counter()
            max_recovery_attempts = 10
            
            for attempt in range(max_recovery_attempts):
                try:
                    await client.set("test_recovery", "post_failure")
                    result = await client.get("test_recovery")
                    if result == b"post_failure":
                        recovery_results["post_failure_success"] = True
                        recovery_results["recovery_time_ms"] = (
                            time.perf_counter() - recovery_start
                        ) * 1000
                        break
                except:
                    await asyncio.sleep(0.1)
            
        finally:
            # coredis doesn't have close(), connection is managed internally
            pass
        
        return recovery_results

    async def validate_performance_tracking(self, l2_service) -> Dict[str, Any]:
        """Validate L2RedisService performance tracking accuracy."""
        # Clear any existing stats
        initial_stats = l2_service.get_stats()
        
        # Perform operations
        test_operations = [
            ("get", "nonexistent_key", None),
            ("set", "test_key_1", {"data": "test1"}),
            ("get", "test_key_1", None),
            ("set", "test_key_2", {"data": "test2"}),
            ("delete", "test_key_1", None),
            ("exists", "test_key_2", None),
        ]
        
        operation_results = []
        for op, key, value in test_operations:
            start_time = time.perf_counter()
            
            if op == "get":
                result = await l2_service.get(key)
            elif op == "set":
                result = await l2_service.set(key, value)
            elif op == "delete":
                result = await l2_service.delete(key)
            elif op == "exists":
                result = await l2_service.exists(key)
            
            end_time = time.perf_counter()
            operation_results.append({
                "operation": op,
                "key": key,
                "result": result,
                "measured_time_ms": (end_time - start_time) * 1000,
            })
        
        # Get final stats
        final_stats = l2_service.get_stats()
        
        return {
            "initial_stats": initial_stats,
            "final_stats": final_stats,
            "operation_results": operation_results,
            "operations_tracked": final_stats["total_operations"] - initial_stats["total_operations"],
            "expected_operations": len(test_operations),
            "tracking_accurate": (
                final_stats["total_operations"] - initial_stats["total_operations"]
            ) == len(test_operations),
        }