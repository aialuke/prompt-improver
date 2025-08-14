"""Network Failure Simulator for Retry Testing.

Simulates various network failure conditions to test retry mechanisms,
circuit breakers, and error handling services with realistic failure scenarios.
"""

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of network failures to simulate."""
    
    CONNECTION_TIMEOUT = "connection_timeout"
    CONNECTION_REFUSED = "connection_refused"
    DNS_RESOLUTION_FAILURE = "dns_resolution_failure"
    INTERMITTENT_ERRORS = "intermittent_errors"
    SLOW_RESPONSE = "slow_response"
    PARTIAL_RESPONSE = "partial_response"
    HTTP_ERRORS = "http_errors"
    SSL_ERRORS = "ssl_errors"
    NETWORK_PARTITION = "network_partition"


@dataclass
class FailureScenario:
    """Configuration for a specific failure scenario."""
    
    failure_type: FailureType
    probability: float = 0.5  # 0.0 = never, 1.0 = always
    duration_ms: Optional[int] = None  # Duration for timeouts/delays
    error_message: str = ""
    custom_exception: Optional[Exception] = None
    
    # Pattern control
    pattern: str = "random"  # random, sequential, burst
    burst_count: int = 3  # For burst pattern
    recovery_time_ms: int = 1000  # Time before recovery


@dataclass
class NetworkOperation:
    """Represents a network operation for simulation."""
    
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operation_name: str = "unknown"
    target_host: str = "localhost"
    target_port: int = 80
    operation_type: str = "http_request"  # http_request, database_connection, api_call
    
    # Execution context
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    
    # Result
    success: bool = False
    failure_type: Optional[FailureType] = None
    error_message: str = ""
    retry_count: int = 0


class NetworkSimulator:
    """Network failure simulator for comprehensive retry testing.
    
    Simulates various network failure conditions to test:
    - Retry mechanisms and backoff strategies
    - Circuit breaker behavior under different failure modes
    - Error handling service responses to network issues
    - Timeout handling and connection pooling behavior
    """

    def __init__(self, simulator_id: Optional[str] = None):
        """Initialize network simulator.
        
        Args:
            simulator_id: Optional custom simulator identifier
        """
        self.simulator_id = simulator_id or str(uuid.uuid4())[:8]
        
        # Active failure scenarios
        self._active_scenarios: Dict[str, FailureScenario] = {}
        self._scenario_states: Dict[str, Dict[str, Any]] = {}
        
        # Operation tracking
        self._operations: Dict[str, NetworkOperation] = {}
        self._operation_stats: Dict[FailureType, int] = {ft: 0 for ft in FailureType}
        
        # Simulator state
        self._is_running = False
        self._background_task: Optional[asyncio.Task] = None
        
        logger.info(f"NetworkSimulator initialized: {self.simulator_id}")

    async def start(self):
        """Start the network simulator."""
        if self._is_running:
            logger.warning("Network simulator already running")
            return
        
        self._is_running = True
        self._background_task = asyncio.create_task(self._background_simulation_loop())
        
        logger.info(f"Network simulator started: {self.simulator_id}")

    async def stop(self):
        """Stop the network simulator and clean up resources."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        # Clear scenarios and operations
        self._active_scenarios.clear()
        self._scenario_states.clear()
        self._operations.clear()
        
        logger.info(f"Network simulator stopped: {self.simulator_id}")

    def add_failure_scenario(
        self,
        scenario_name: str,
        failure_type: FailureType,
        probability: float = 0.5,
        duration_ms: Optional[int] = None,
        pattern: str = "random",
        **kwargs
    ) -> None:
        """Add a failure scenario to the simulator.
        
        Args:
            scenario_name: Unique name for the scenario
            failure_type: Type of failure to simulate
            probability: Probability of failure (0.0 to 1.0)
            duration_ms: Duration for timeouts/delays
            pattern: Failure pattern (random, sequential, burst)
            **kwargs: Additional scenario configuration
        """
        scenario = FailureScenario(
            failure_type=failure_type,
            probability=probability,
            duration_ms=duration_ms,
            pattern=pattern,
            error_message=kwargs.get('error_message', f'{failure_type.value} simulated'),
            custom_exception=kwargs.get('custom_exception'),
            burst_count=kwargs.get('burst_count', 3),
            recovery_time_ms=kwargs.get('recovery_time_ms', 1000),
        )
        
        self._active_scenarios[scenario_name] = scenario
        self._scenario_states[scenario_name] = {
            "last_failure_time": 0,
            "burst_counter": 0,
            "in_burst": False,
            "operations_since_last": 0,
        }
        
        logger.info(f"Added failure scenario: {scenario_name} ({failure_type.value})")

    def remove_failure_scenario(self, scenario_name: str) -> bool:
        """Remove a failure scenario.
        
        Args:
            scenario_name: Name of scenario to remove
            
        Returns:
            True if scenario was removed, False if not found
        """
        if scenario_name in self._active_scenarios:
            del self._active_scenarios[scenario_name]
            del self._scenario_states[scenario_name]
            logger.info(f"Removed failure scenario: {scenario_name}")
            return True
        return False

    async def simulate_network_operation(
        self,
        operation_name: str,
        target_host: str = "localhost",
        target_port: int = 80,
        operation_type: str = "http_request"
    ) -> NetworkOperation:
        """Simulate a network operation with potential failures.
        
        Args:
            operation_name: Name of the operation
            target_host: Target host for the operation
            target_port: Target port
            operation_type: Type of network operation
            
        Returns:
            NetworkOperation result with success/failure information
        """
        operation = NetworkOperation(
            operation_name=operation_name,
            target_host=target_host,
            target_port=target_port,
            operation_type=operation_type,
        )
        
        self._operations[operation.operation_id] = operation
        
        try:
            # Check all active scenarios for potential failures
            for scenario_name, scenario in self._active_scenarios.items():
                should_fail, failure_details = await self._should_operation_fail(
                    scenario_name, scenario, operation
                )
                
                if should_fail:
                    # Apply failure
                    operation.success = False
                    operation.failure_type = scenario.failure_type
                    operation.error_message = failure_details.get('error_message', scenario.error_message)
                    
                    # Update statistics
                    self._operation_stats[scenario.failure_type] += 1
                    
                    # Simulate failure delay if specified
                    if scenario.duration_ms:
                        await asyncio.sleep(scenario.duration_ms / 1000.0)
                    
                    operation.end_time = time.perf_counter()
                    operation.duration_ms = (operation.end_time - operation.start_time) * 1000
                    
                    # Raise appropriate exception if specified
                    if scenario.custom_exception:
                        raise scenario.custom_exception
                    elif scenario.failure_type == FailureType.CONNECTION_TIMEOUT:
                        raise asyncio.TimeoutError(operation.error_message)
                    elif scenario.failure_type == FailureType.CONNECTION_REFUSED:
                        raise ConnectionRefusedError(operation.error_message)
                    elif scenario.failure_type == FailureType.DNS_RESOLUTION_FAILURE:
                        raise OSError(f"DNS resolution failed: {operation.error_message}")
                    else:
                        raise RuntimeError(operation.error_message)
            
            # If no failure occurred, simulate successful operation
            operation.success = True
            
            # Simulate successful operation delay (small random delay)
            success_delay = random.uniform(0.01, 0.05)  # 10-50ms
            await asyncio.sleep(success_delay)
            
            operation.end_time = time.perf_counter()
            operation.duration_ms = (operation.end_time - operation.start_time) * 1000
            
            return operation
            
        except Exception as e:
            operation.end_time = time.perf_counter()
            operation.duration_ms = (operation.end_time - operation.start_time) * 1000
            operation.error_message = str(e)
            raise

    async def _should_operation_fail(
        self,
        scenario_name: str,
        scenario: FailureScenario,
        operation: NetworkOperation
    ) -> tuple[bool, Dict[str, Any]]:
        """Determine if an operation should fail based on scenario configuration.
        
        Args:
            scenario_name: Name of the scenario
            scenario: Failure scenario configuration
            operation: Network operation to check
            
        Returns:
            Tuple of (should_fail, failure_details)
        """
        state = self._scenario_states[scenario_name]
        current_time = time.perf_counter() * 1000  # Convert to milliseconds
        
        failure_details = {
            "error_message": scenario.error_message,
            "scenario_name": scenario_name,
        }
        
        # Check pattern-based failure logic
        if scenario.pattern == "random":
            return random.random() < scenario.probability, failure_details
        
        elif scenario.pattern == "sequential":
            # Fail every N operations based on probability
            state["operations_since_last"] += 1
            operations_threshold = int(1.0 / scenario.probability) if scenario.probability > 0 else float('inf')
            
            if state["operations_since_last"] >= operations_threshold:
                state["operations_since_last"] = 0
                return True, failure_details
            
            return False, failure_details
        
        elif scenario.pattern == "burst":
            # Create bursts of failures followed by recovery periods
            if state["in_burst"]:
                if state["burst_counter"] < scenario.burst_count:
                    state["burst_counter"] += 1
                    return True, failure_details
                else:
                    # End burst, start recovery period
                    state["in_burst"] = False
                    state["burst_counter"] = 0
                    state["last_failure_time"] = current_time
                    return False, failure_details
            else:
                # Check if recovery time has passed
                if current_time - state["last_failure_time"] > scenario.recovery_time_ms:
                    # Start new burst based on probability
                    if random.random() < scenario.probability:
                        state["in_burst"] = True
                        state["burst_counter"] = 1
                        return True, failure_details
                
                return False, failure_details
        
        else:
            # Unknown pattern, default to random
            return random.random() < scenario.probability, failure_details

    @asynccontextmanager
    async def temporary_failure_scenario(
        self,
        scenario_name: str,
        failure_type: FailureType,
        probability: float = 1.0,
        duration_ms: Optional[int] = None
    ):
        """Temporarily add a failure scenario for testing.
        
        Args:
            scenario_name: Name for temporary scenario
            failure_type: Type of failure to simulate
            probability: Probability of failure
            duration_ms: Duration for timeouts/delays
        """
        # Add temporary scenario
        self.add_failure_scenario(
            scenario_name=scenario_name,
            failure_type=failure_type,
            probability=probability,
            duration_ms=duration_ms,
        )
        
        try:
            yield
        finally:
            # Remove temporary scenario
            self.remove_failure_scenario(scenario_name)

    async def simulate_http_request(
        self,
        url: str,
        method: str = "GET",
        timeout_ms: int = 5000
    ) -> NetworkOperation:
        """Simulate HTTP request with potential failures.
        
        Args:
            url: URL for the request
            method: HTTP method
            timeout_ms: Request timeout
            
        Returns:
            NetworkOperation result
        """
        return await self.simulate_network_operation(
            operation_name=f"http_{method.lower()}_{url}",
            target_host=url.split('/')[2] if '://' in url else url,
            target_port=443 if url.startswith('https') else 80,
            operation_type="http_request"
        )

    async def simulate_database_connection(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "test_db"
    ) -> NetworkOperation:
        """Simulate database connection with potential failures.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            
        Returns:
            NetworkOperation result
        """
        return await self.simulate_network_operation(
            operation_name=f"db_connect_{database}",
            target_host=host,
            target_port=port,
            operation_type="database_connection"
        )

    async def simulate_redis_operation(
        self,
        host: str = "localhost",
        port: int = 6379,
        operation: str = "get"
    ) -> NetworkOperation:
        """Simulate Redis operation with potential failures.
        
        Args:
            host: Redis host
            port: Redis port
            operation: Redis operation type
            
        Returns:
            NetworkOperation result
        """
        return await self.simulate_network_operation(
            operation_name=f"redis_{operation}",
            target_host=host,
            target_port=port,
            operation_type="redis_operation"
        )

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get comprehensive failure statistics.
        
        Returns:
            Dictionary with failure statistics and scenarios
        """
        total_operations = len(self._operations)
        successful_operations = sum(1 for op in self._operations.values() if op.success)
        failed_operations = total_operations - successful_operations
        
        return {
            "simulator_id": self.simulator_id,
            "is_running": self._is_running,
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "success_rate": successful_operations / total_operations if total_operations > 0 else 0,
            "failure_types": dict(self._operation_stats),
            "active_scenarios": {
                name: {
                    "failure_type": scenario.failure_type.value,
                    "probability": scenario.probability,
                    "pattern": scenario.pattern,
                    "duration_ms": scenario.duration_ms,
                }
                for name, scenario in self._active_scenarios.items()
            },
            "scenario_states": dict(self._scenario_states),
        }

    def get_operation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent operation history.
        
        Args:
            limit: Maximum number of operations to return
            
        Returns:
            List of operation details
        """
        operations = list(self._operations.values())
        # Sort by start time, most recent first
        operations.sort(key=lambda op: op.start_time, reverse=True)
        
        return [
            {
                "operation_id": op.operation_id,
                "operation_name": op.operation_name,
                "target": f"{op.target_host}:{op.target_port}",
                "operation_type": op.operation_type,
                "success": op.success,
                "duration_ms": op.duration_ms,
                "failure_type": op.failure_type.value if op.failure_type else None,
                "error_message": op.error_message if op.error_message else None,
            }
            for op in operations[:limit]
        ]

    async def _background_simulation_loop(self):
        """Background loop for scenario state management."""
        while self._is_running:
            try:
                # Update scenario states periodically
                current_time = time.perf_counter() * 1000
                
                for scenario_name, state in self._scenario_states.items():
                    scenario = self._active_scenarios[scenario_name]
                    
                    # Reset burst state if recovery time exceeded
                    if (state["in_burst"] and 
                        current_time - state["last_failure_time"] > scenario.recovery_time_ms):
                        state["in_burst"] = False
                        state["burst_counter"] = 0
                
                # Sleep for state update interval
                await asyncio.sleep(1.0)  # Update every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background simulation loop error: {e}")
                await asyncio.sleep(1.0)

    async def reset_statistics(self):
        """Reset all statistics and operation history."""
        self._operations.clear()
        self._operation_stats = {ft: 0 for ft in FailureType}
        
        # Reset scenario states
        for state in self._scenario_states.values():
            state.update({
                "last_failure_time": 0,
                "burst_counter": 0,
                "in_burst": False,
                "operations_since_last": 0,
            })
        
        logger.info("Network simulator statistics reset")

    def create_common_failure_scenarios(self) -> Dict[str, str]:
        """Create common failure scenarios for testing.
        
        Returns:
            Dictionary mapping scenario names to descriptions
        """
        scenarios = {
            "connection_timeout": "Simulates connection timeouts",
            "connection_refused": "Simulates refused connections",
            "dns_failure": "Simulates DNS resolution failures",
            "intermittent_errors": "Simulates intermittent network errors",
            "burst_failures": "Simulates burst failure patterns",
        }
        
        # Add common scenarios
        self.add_failure_scenario(
            "connection_timeout",
            FailureType.CONNECTION_TIMEOUT,
            probability=0.3,
            duration_ms=5000,
            pattern="random"
        )
        
        self.add_failure_scenario(
            "connection_refused",
            FailureType.CONNECTION_REFUSED,
            probability=0.2,
            pattern="sequential"
        )
        
        self.add_failure_scenario(
            "dns_failure",
            FailureType.DNS_RESOLUTION_FAILURE,
            probability=0.1,
            pattern="random"
        )
        
        self.add_failure_scenario(
            "intermittent_errors",
            FailureType.INTERMITTENT_ERRORS,
            probability=0.4,
            pattern="burst",
            burst_count=3,
            recovery_time_ms=2000
        )
        
        self.add_failure_scenario(
            "burst_failures",
            FailureType.HTTP_ERRORS,
            probability=0.8,
            pattern="burst",
            burst_count=5,
            recovery_time_ms=3000
        )
        
        logger.info("Created common failure scenarios")
        return scenarios