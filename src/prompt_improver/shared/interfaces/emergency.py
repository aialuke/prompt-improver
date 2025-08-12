"""Emergency operations interface to break circular dependencies"""

from typing import Protocol

from prompt_improver.shared.types.signals import OperationResult, SignalContext


class IEmergencyOperations(Protocol):
    """Interface for emergency operations management

    Abstracts emergency operations to break circular dependencies
    between signal handlers and emergency operation managers.
    """

    async def handle_operation(self, context: SignalContext) -> OperationResult:
        """Handle a signal-triggered emergency operation

        Args:
            context: Signal context with operation details

        Returns:
            Result of the operation
        """
        ...

    async def create_emergency_checkpoint(
        self, context: SignalContext
    ) -> OperationResult:
        """Create emergency checkpoint triggered by signal

        Args:
            context: Signal context

        Returns:
            Checkpoint creation result
        """
        ...

    async def generate_status_report(self, context: SignalContext) -> OperationResult:
        """Generate real-time status report

        Args:
            context: Signal context

        Returns:
            Status report generation result
        """
        ...

    async def reload_configuration(self, context: SignalContext) -> OperationResult:
        """Reload system configuration

        Args:
            context: Signal context

        Returns:
            Configuration reload result
        """
        ...

    async def emergency_shutdown(self, context: SignalContext) -> OperationResult:
        """Perform emergency shutdown operations

        Args:
            context: Signal context with shutdown details

        Returns:
            Shutdown operation result
        """
        ...
