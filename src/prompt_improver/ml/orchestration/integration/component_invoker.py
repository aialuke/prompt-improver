"""
Component Invoker for ML Pipeline Orchestrator.

Invokes methods on loaded ML components and handles results.
"""

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from .direct_component_loader import DirectComponentLoader, LoadedComponent

@dataclass
class InvocationResult:
    """Result of a component method invocation."""
    component_name: str
    method_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class ComponentInvoker:
    """
    Invokes methods on loaded ML components.
    
    Provides a unified interface for calling component methods with
    error handling, logging, and result tracking.
    """
    
    def __init__(self, component_loader: DirectComponentLoader, retry_manager=None, input_sanitizer=None, memory_guard=None):
        """Initialize the component invoker."""
        self.component_loader = component_loader
        self.logger = logging.getLogger(__name__)
        self.invocation_history: List[InvocationResult] = []
        self.retry_manager = retry_manager
        self.input_sanitizer = input_sanitizer
        self.memory_guard = memory_guard

    def set_retry_manager(self, retry_manager):
        """Set the retry manager for resilient component invocations."""
        self.retry_manager = retry_manager
        self.logger.info("Retry manager integrated with ComponentInvoker")

    def set_input_sanitizer(self, input_sanitizer):
        """Set the input sanitizer for secure component invocations."""
        self.input_sanitizer = input_sanitizer
        self.logger.info("Input sanitizer integrated with ComponentInvoker")

    def set_memory_guard(self, memory_guard):
        """Set the memory guard for memory-monitored component invocations."""
        self.memory_guard = memory_guard
        self.logger.info("Memory guard integrated with ComponentInvoker")

    async def invoke_component_method(
        self,
        component_name: str,
        method_name: str,
        *args,
        **kwargs
    ) -> InvocationResult:
        """
        Invoke a method on a loaded component.
        
        Args:
            component_name: Name of the component
            method_name: Name of the method to invoke
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            InvocationResult with execution details
        """
        start_time = datetime.now(timezone.utc)
        
        # Check if component is loaded
        loaded_component = self.component_loader.get_loaded_component(component_name)
        if not loaded_component:
            error_msg = f"Component {component_name} not loaded"
            self.logger.error(error_msg)
            return InvocationResult(
                component_name=component_name,
                method_name=method_name,
                success=False,
                error=error_msg,
                timestamp=start_time
            )
        
        # Check if component is initialized
        if not loaded_component.is_initialized:
            error_msg = f"Component {component_name} not initialized"
            self.logger.error(error_msg)
            return InvocationResult(
                component_name=component_name,
                method_name=method_name,
                success=False,
                error=error_msg,
                timestamp=start_time
            )
        
        # Get the method
        try:
            method = getattr(loaded_component.instance, method_name)
            if not callable(method):
                error_msg = f"Method {method_name} is not callable on {component_name}"
                self.logger.error(error_msg)
                return InvocationResult(
                    component_name=component_name,
                    method_name=method_name,
                    success=False,
                    error=error_msg,
                    timestamp=start_time
                )
        except AttributeError:
            error_msg = f"Method {method_name} not found on component {component_name}"
            self.logger.error(error_msg)
            return InvocationResult(
                component_name=component_name,
                method_name=method_name,
                success=False,
                error=error_msg,
                timestamp=start_time
            )
        
        # Invoke the method
        try:
            # Check if method is async
            if inspect.iscoroutinefunction(method):
                result = await method(*args, **kwargs)
            else:
                result = method(*args, **kwargs)
            
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()
            
            invocation_result = InvocationResult(
                component_name=component_name,
                method_name=method_name,
                success=True,
                result=result,
                execution_time=execution_time,
                timestamp=start_time
            )
            
            self.logger.info(f"Successfully invoked {component_name}.{method_name} in {execution_time:.3f}s")
            self.invocation_history.append(invocation_result)
            
            return invocation_result

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()
            error_msg = f"Error invoking {component_name}.{method_name}: {str(e)}"

            self.logger.error(error_msg)

            invocation_result = InvocationResult(
                component_name=component_name,
                method_name=method_name,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                timestamp=start_time
            )

            self.invocation_history.append(invocation_result)
            return invocation_result

    async def invoke_component_method_with_retry(
        self,
        component_name: str,
        method_name: str,
        *args,
        max_attempts: int = 3,
        enable_circuit_breaker: bool = True,
        **kwargs
    ) -> InvocationResult:
        """
        Invoke a component method with unified retry logic.

        Args:
            component_name: Name of the component
            method_name: Name of the method to invoke
            *args: Positional arguments for the method
            max_attempts: Maximum retry attempts
            enable_circuit_breaker: Whether to use circuit breaker
            **kwargs: Keyword arguments for the method

        Returns:
            InvocationResult with execution details
        """
        if not self.retry_manager:
            # Fallback to regular invocation if retry manager not available
            self.logger.warning(f"Retry manager not available for {component_name}.{method_name}, using direct invocation")
            return await self.invoke_component_method(component_name, method_name, *args, **kwargs)

        operation_name = f"{component_name}.{method_name}"
        start_time = datetime.now(timezone.utc)

        try:
            # Import retry configuration
            from ..core.unified_retry_manager import RetryConfig

            # Create retry configuration
            retry_config = RetryConfig(
                operation_name=operation_name,
                max_attempts=max_attempts,
                enable_circuit_breaker=enable_circuit_breaker
            )

            # Define the operation to retry
            async def component_operation():
                result = await self.invoke_component_method(component_name, method_name, *args, **kwargs)
                if not result.success:
                    # Convert failed invocation to exception for retry logic
                    raise RuntimeError(f"Component invocation failed: {result.error}")
                return result

            # Execute with retry
            result = await self.retry_manager.retry_async(component_operation, config=retry_config)

            self.logger.info(f"Successfully invoked {operation_name} with retry support")
            return result

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            error_msg = f"Resilient invocation failed after retries: {str(e)}"
            self.logger.error(f"Failed to invoke {operation_name} with retry: {e}")

            invocation_result = InvocationResult(
                component_name=component_name,
                method_name=method_name,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                timestamp=start_time
            )

            self.invocation_history.append(invocation_result)
            return invocation_result

    async def invoke_component_method_secure(
        self,
        component_name: str,
        method_name: str,
        *args,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> InvocationResult:
        """
        Invoke a component method with security validation.

        Args:
            component_name: Name of the component
            method_name: Name of the method to invoke
            *args: Positional arguments for the method
            context: Security context (user_id, source_ip, etc.)
            **kwargs: Keyword arguments for the method

        Returns:
            InvocationResult with execution details
        """
        start_time = datetime.now(timezone.utc)
        context = context or {}

        try:
            # Validate all input arguments if sanitizer is available
            if self.input_sanitizer:
                # Validate positional arguments
                validated_args = []
                for i, arg in enumerate(args):
                    validation_result = await self.input_sanitizer.validate_input_async(
                        arg, {**context, "parameter_type": f"arg_{i}"}
                    )

                    if not validation_result.is_valid:
                        error_msg = f"Security validation failed for argument {i}: {validation_result.message}"
                        self.logger.error(error_msg)
                        return InvocationResult(
                            component_name=component_name,
                            method_name=method_name,
                            success=False,
                            error=error_msg,
                            timestamp=start_time
                        )

                    validated_args.append(validation_result.sanitized_value)

                # Validate keyword arguments
                validated_kwargs = {}
                for key, value in kwargs.items():
                    validation_result = await self.input_sanitizer.validate_input_async(
                        value, {**context, "parameter_type": f"kwarg_{key}"}
                    )

                    if not validation_result.is_valid:
                        error_msg = f"Security validation failed for parameter {key}: {validation_result.message}"
                        self.logger.error(error_msg)
                        return InvocationResult(
                            component_name=component_name,
                            method_name=method_name,
                            success=False,
                            error=error_msg,
                            timestamp=start_time
                        )

                    validated_kwargs[key] = validation_result.sanitized_value

                # Use validated arguments
                args = tuple(validated_args)
                kwargs = validated_kwargs

                self.logger.info(f"Security validation passed for {component_name}.{method_name}")

            # Invoke the component method with validated arguments
            if self.retry_manager:
                return await self.invoke_component_method_with_retry(
                    component_name, method_name, *args, **kwargs
                )
            else:
                return await self.invoke_component_method(
                    component_name, method_name, *args, **kwargs
                )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            error_msg = f"Secure component invocation failed: {str(e)}"
            self.logger.error(f"Failed to securely invoke {component_name}.{method_name}: {e}")

            invocation_result = InvocationResult(
                component_name=component_name,
                method_name=method_name,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                timestamp=start_time
            )

            self.invocation_history.append(invocation_result)
            return invocation_result
    
    async def invoke_training_workflow(self, training_data: Any) -> Dict[str, InvocationResult]:
        """
        Invoke a complete training workflow using core components.
        
        Args:
            training_data: Input training data (for context, rule extraction, etc.)
            
        Returns:
            Dictionary of component results
        """
        results = {}
        
        # Import database session manager
        try:
            from ...database import get_sessionmanager
        except ImportError:
            # Fallback for testing environments
            results["data_loading"] = InvocationResult(
                component_name="training_data_loader",
                method_name="load_training_data", 
                success=False,
                error="Database session manager not available"
            )
            return results
        
        # Step 1: Load training data with proper database session
        try:
            async with get_sessionmanager().session() as db_session:
                result = await self.invoke_component_method(
                    "training_data_loader", 
                    "load_training_data", 
                    db_session  # Pass database session as required first parameter
                    # Note: rule_ids parameter is optional, can be extracted from training_data if needed
                )
                results["data_loading"] = result
                
                if not result.success:
                    return results
                
                # Use the loaded training data for subsequent steps
                loaded_training_data = result.result
                
        except Exception as e:
            results["data_loading"] = InvocationResult(
                component_name="training_data_loader",
                method_name="load_training_data",
                success=False,
                error=f"Database session error: {str(e)}"
            )
            return results
        
        # Step 2: Process with ML integration
        result = await self.invoke_component_method(
            "ml_integration",
            "discover_patterns",  # Use correct method name
            loaded_training_data if loaded_training_data else training_data
        )
        results["ml_processing"] = result
        
        if not result.success:
            return results
        
        # Step 3: Optimize rules (if we have performance data)
        if loaded_training_data and isinstance(loaded_training_data, dict) and loaded_training_data.get("features"):
            result = await self.invoke_component_method(
                "rule_optimizer",
                "optimize_rule_combination",  # Use correct method name
                loaded_training_data["features"][:5] if len(loaded_training_data["features"]) > 5 else loaded_training_data["features"]
            )
            results["rule_optimization"] = result
        else:
            # Skip rule optimization if no proper training data
            results["rule_optimization"] = InvocationResult(
                component_name="rule_optimizer",
                method_name="optimize_rule_combination",
                success=True,
                result="Skipped - no training data available"
            )
        
        return results
    
    async def invoke_evaluation_workflow(self, evaluation_data: Any) -> Dict[str, InvocationResult]:
        """
        Invoke a complete evaluation workflow using evaluation components.
        
        Args:
            evaluation_data: Input evaluation data
            
        Returns:
            Dictionary of component results
        """
        results = {}
        
        # Step 1: Statistical analysis
        result = await self.invoke_component_method(
            "statistical_analyzer",
            "perform_statistical_analysis",
            evaluation_data
        )
        results["statistical_analysis"] = result
        
        if not result.success:
            return results
        
        # Step 2: Pattern significance
        result = await self.invoke_component_method(
            "pattern_significance_analyzer",
            "analyze_patterns",
            result.result
        )
        results["pattern_analysis"] = result
        
        if not result.success:
            return results
        
        # Step 3: Structural analysis
        result = await self.invoke_component_method(
            "structural_analyzer",
            "analyze_structure",
            evaluation_data
        )
        results["structural_analysis"] = result
        
        return results
    
    async def batch_invoke_components(
        self,
        invocations: List[Dict[str, Any]]
    ) -> List[InvocationResult]:
        """
        Invoke multiple component methods in parallel.
        
        Args:
            invocations: List of invocation specifications with keys:
                        - component_name, method_name, args, kwargs
                        
        Returns:
            List of InvocationResults
        """
        tasks = []
        
        for invocation in invocations:
            component_name = invocation["component_name"]
            method_name = invocation["method_name"]
            args = invocation.get("args", [])
            kwargs = invocation.get("kwargs", {})
            
            task = self.invoke_component_method(
                component_name, method_name, *args, **kwargs
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                invocation = invocations[i]
                error_result = InvocationResult(
                    component_name=invocation["component_name"],
                    method_name=invocation["method_name"],
                    success=False,
                    error=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_available_methods(self, component_name: str) -> List[str]:
        """
        Get list of available methods for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            List of method names
        """
        loaded_component = self.component_loader.get_loaded_component(component_name)
        if not loaded_component or not loaded_component.is_initialized:
            return []
        
        methods = []
        for name in dir(loaded_component.instance):
            if not name.startswith('_') and callable(getattr(loaded_component.instance, name)):
                methods.append(name)
        
        return methods
    
    def get_method_signature(self, component_name: str, method_name: str) -> Optional[str]:
        """
        Get method signature for a component method.
        
        Args:
            component_name: Name of the component
            method_name: Name of the method
            
        Returns:
            Method signature string or None
        """
        loaded_component = self.component_loader.get_loaded_component(component_name)
        if not loaded_component or not loaded_component.is_initialized:
            return None
        
        try:
            method = getattr(loaded_component.instance, method_name)
            signature = inspect.signature(method)
            return str(signature)
        except (AttributeError, ValueError):
            return None
    
    def get_invocation_history(
        self, 
        component_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[InvocationResult]:
        """
        Get invocation history, optionally filtered by component.
        
        Args:
            component_name: Filter by component name
            limit: Limit number of results
            
        Returns:
            List of InvocationResults
        """
        history = self.invocation_history
        
        if component_name:
            history = [r for r in history if r.component_name == component_name]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_success_rate(self, component_name: Optional[str] = None) -> float:
        """
        Get success rate for invocations.
        
        Args:
            component_name: Filter by component name
            
        Returns:
            Success rate as float between 0.0 and 1.0
        """
        history = self.get_invocation_history(component_name)
        
        if not history:
            return 0.0
        
        successful = sum(1 for r in history if r.success)
        return successful / len(history)
    
    def clear_history(self) -> None:
        """Clear invocation history."""
        self.invocation_history.clear()
        self.logger.info("Cleared invocation history")

    async def invoke_component_method_with_memory_monitoring(
        self,
        component_name: str,
        method_name: str,
        *args,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> InvocationResult:
        """
        Invoke a component method with memory monitoring.

        Args:
            component_name: Name of the component
            method_name: Name of the method to invoke
            *args: Positional arguments for the method
            context: Monitoring context
            **kwargs: Keyword arguments for the method

        Returns:
            InvocationResult with execution details
        """
        operation_name = f"{component_name}.{method_name}"
        start_time = datetime.now(timezone.utc)
        context = context or {}

        try:
            # Monitor memory during component invocation
            if self.memory_guard:
                async with self.memory_guard.monitor_operation_async(operation_name, component_name):
                    # Validate memory requirements for arguments
                    for i, arg in enumerate(args):
                        await self.memory_guard.validate_ml_operation_memory(
                            arg, f"{operation_name}_arg_{i}", component_name
                        )

                    for key, value in kwargs.items():
                        await self.memory_guard.validate_ml_operation_memory(
                            value, f"{operation_name}_kwarg_{key}", component_name
                        )

                    # Invoke the component method with monitoring
                    if self.input_sanitizer:
                        return await self.invoke_component_method_secure(
                            component_name, method_name, *args, context=context, **kwargs
                        )
                    elif self.retry_manager:
                        return await self.invoke_component_method_with_retry(
                            component_name, method_name, *args, **kwargs
                        )
                    else:
                        return await self.invoke_component_method(
                            component_name, method_name, *args, **kwargs
                        )
            else:
                # Fallback to other invocation methods if memory guard not available
                self.logger.warning(f"Memory guard not available for {operation_name}, using fallback invocation")

                if self.input_sanitizer:
                    return await self.invoke_component_method_secure(
                        component_name, method_name, *args, context=context, **kwargs
                    )
                elif self.retry_manager:
                    return await self.invoke_component_method_with_retry(
                        component_name, method_name, *args, **kwargs
                    )
                else:
                    return await self.invoke_component_method(
                        component_name, method_name, *args, **kwargs
                    )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            error_msg = f"Memory-monitored component invocation failed: {str(e)}"
            self.logger.error(f"Failed to invoke {operation_name} with memory monitoring: {e}")

            invocation_result = InvocationResult(
                component_name=component_name,
                method_name=method_name,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                timestamp=start_time
            )

            self.invocation_history.append(invocation_result)
            return invocation_result