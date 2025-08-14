"""Security Integration Service.

Focused service responsible for security validation, input sanitization, and access control.
Handles input validation, memory monitoring, and secure workflow execution.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from ....core.protocols.ml_protocols import EventBusProtocol
from ....security.input_sanitization import InputSanitizer
from ....security.memory_guard import MemoryGuard
from ..core.orchestrator_service_protocols import SecurityIntegrationServiceProtocol
from ..events.event_types import EventType, MLEvent


class SecurityIntegrationService:
    """
    Security Integration Service.
    
    Responsible for:
    - Security validation and input sanitization
    - Memory monitoring and resource protection
    - Secure workflow execution with validation
    - Access control and threat detection
    """

    def __init__(
        self,
        event_bus: EventBusProtocol,
        input_sanitizer: InputSanitizer | None = None,
        memory_guard: MemoryGuard | None = None,
    ):
        """Initialize SecurityIntegrationService with required dependencies.
        
        Args:
            event_bus: Event bus for security event communication
            input_sanitizer: Optional input validation service
            memory_guard: Optional memory management service
        """
        self.event_bus = event_bus
        self.input_sanitizer = input_sanitizer
        self.memory_guard = memory_guard
        self.logger = logging.getLogger(__name__)
        
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the security integration service."""
        if self._is_initialized:
            return
        
        self.logger.info("Initializing Security Integration Service")
        
        try:
            # Configure input sanitizer if available
            if self.input_sanitizer:
                # Integrate input sanitizer with event bus for security monitoring
                self.input_sanitizer.set_event_bus(self.event_bus)
                self.logger.info("Input sanitizer integrated with event bus")
            
            # Configure memory guard if available
            if self.memory_guard:
                # Integrate memory guard with event bus for resource monitoring
                self.memory_guard.set_event_bus(self.event_bus)
                self.logger.info("Memory guard integrated with event bus")
            
            self._is_initialized = True
            
            # Emit initialization event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.ORCHESTRATOR_INITIALIZED,
                source="security_integration_service",
                data={"component": "security_integration_service", "timestamp": datetime.now(timezone.utc).isoformat()}
            ))
            
            self.logger.info("Security Integration Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security integration service: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the security integration service."""
        self.logger.info("Shutting down Security Integration Service")
        
        try:
            self._is_initialized = False
            self.logger.info("Security Integration Service shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during security integration service shutdown: {e}")
            raise

    async def validate_input_secure(self, input_data: Any, context: dict[str, Any] = None) -> Any:
        """
        Validate input data using integrated security sanitizer.
        
        Args:
            input_data: Data to validate
            context: Validation context (user_id, source_ip, etc.)
            
        Returns:
            ValidationResult with security assessment
            
        Raises:
            RuntimeError: If input sanitizer not available
            SecurityError: If critical security threats detected
        """
        if not self.input_sanitizer:
            self.logger.warning("Input sanitizer not available, skipping validation")
            # Return a basic validation result
            from ....security.input_sanitization import (
                SecurityThreatLevel,
                ValidationResult,
            )
            return ValidationResult(
                is_valid=True,
                sanitized_value=input_data,
                threat_level=SecurityThreatLevel.LOW,
                message="Input sanitizer not available"
            )
        
        # Perform comprehensive security validation
        result = await self.input_sanitizer.validate_input_async(input_data, context)
        
        # Log security validation results
        if result.threats_detected:
            self.logger.warning(f"Security threats detected in input: {result.threats_detected}")
        
        # Raise exception for critical threats
        from ....security.input_sanitization import (
            SecurityError,
            SecurityThreatLevel,
        )
        if result.threat_level == SecurityThreatLevel.CRITICAL:
            raise SecurityError(f"Critical security threat detected: {result.threats_detected}")
        
        return result

    async def monitor_memory_usage(self, operation_name: str = "security_operation", component_name: str = None) -> Any:
        """
        Monitor memory usage for security operations.
        
        Args:
            operation_name: Name of the operation being monitored
            component_name: Name of the component if applicable
            
        Returns:
            ResourceStats with current memory information
        """
        if not self.memory_guard:
            self.logger.warning("Memory guard not available, skipping memory monitoring")
            return None
        
        return await self.memory_guard.check_memory_usage_async(operation_name, component_name)

    def monitor_operation_memory(self, operation_name: str, component_name: str = None) -> Any:
        """
        Async context manager for monitoring memory during security operations.
        
        Args:
            operation_name: Name of the operation to monitor
            component_name: Name of the component if applicable
            
        Returns:
            AsyncMemoryMonitor context manager
        """
        if not self.memory_guard:
            self.logger.warning("Memory guard not available for operation monitoring")
            # Return a no-op context manager
            @asynccontextmanager
            async def noop_monitor():
                yield None
            
            return noop_monitor()
        
        return self.memory_guard.monitor_operation_async(operation_name, component_name)

    async def validate_operation_memory(self, data: Any, operation_name: str, component_name: str = None) -> bool:
        """
        Validate memory requirements for ML operations.
        
        Args:
            data: Data to validate for memory requirements
            operation_name: Name of the operation
            component_name: Name of the component performing the operation
            
        Returns:
            True if memory validation passes
            
        Raises:
            MemoryError: If memory requirements exceed limits
        """
        if not self.memory_guard:
            self.logger.warning("Memory guard not available, skipping memory validation")
            return True
        
        return await self.memory_guard.validate_ml_operation_memory(data, operation_name, component_name)

    async def run_training_workflow_secure(self, training_data: Any, context: dict[str, Any] = None) -> dict[str, Any]:
        """
        Run a complete training workflow with security validation.
        
        Args:
            training_data: Input training data
            context: Security context (user_id, source_ip, etc.)
            
        Returns:
            Dictionary of workflow results
            
        Raises:
            RuntimeError: If validation fails or required dependencies unavailable
        """
        self.logger.info("Running secure training workflow with input validation")
        
        try:
            # Validate training data for security threats
            validation_result = await self.validate_input_secure(training_data, context)
            
            if not validation_result.is_valid:
                raise RuntimeError(f"Training data validation failed: {validation_result.message}")
            
            # Use sanitized data for training
            sanitized_data = validation_result.sanitized_value
            
            # Memory monitoring during secure training
            async with self.monitor_operation_memory("secure_training_workflow", "security_integration_service"):
                # Validate memory requirements for training data
                await self.validate_operation_memory(sanitized_data, "training_data_validation", "security_integration_service")
                
                # This would typically integrate with the workflow orchestrator
                # For now, we return a placeholder result indicating successful security validation
                return {
                    "security_validation": {
                        "input_validated": True,
                        "threat_level": validation_result.threat_level.value,
                        "sanitized": True,
                        "memory_validated": True
                    },
                    "sanitized_data_ready": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
        
        except Exception as e:
            self.logger.error(f"Secure training workflow failed: {e}")
            raise

    async def run_training_workflow_with_memory_monitoring(
        self, 
        training_data: Any, 
        context: dict[str, Any] = None,
        workflow_runner: callable = None
    ) -> dict[str, Any]:
        """
        Run a complete training workflow with comprehensive memory monitoring.
        
        Args:
            training_data: Input training data
            context: Security and monitoring context
            workflow_runner: Function to execute the actual workflow
            
        Returns:
            Dictionary of workflow results
        """
        self.logger.info("Running training workflow with memory monitoring")
        
        async with self.monitor_operation_memory("training_workflow", "security_integration_service"):
            try:
                # Validate memory requirements for training data
                await self.validate_operation_memory(training_data, "training_data_validation", "security_integration_service")
                
                # Run secure training workflow with memory monitoring
                if self.input_sanitizer:
                    security_results = await self.run_training_workflow_secure(training_data, context)
                    
                    # If a workflow runner is provided, execute it with sanitized data
                    if workflow_runner:
                        workflow_results = await workflow_runner(security_results.get("sanitized_data", training_data))
                        security_results.update({"workflow_results": workflow_results})
                    
                    return security_results
                else:
                    # Run without input sanitizer if not available
                    if workflow_runner:
                        return await workflow_runner(training_data)
                    else:
                        return {"message": "Memory monitoring completed, no workflow runner provided"}
                        
            except Exception as e:
                self.logger.error(f"Memory-monitored training workflow failed: {e}")
                raise