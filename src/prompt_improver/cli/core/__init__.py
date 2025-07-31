"""
CLI Core Components for APES Training System

This package contains the core components for the ultra-minimal 3-command CLI:
- Training system management and lifecycle
- CLI orchestration and workflow coordination  
- Progress preservation and session management
- Signal handling and graceful shutdown
- Emergency operations and crash recovery
- Process management and monitoring

All components are designed for clean break modernization with:
- Zero legacy compatibility layers
- Pure training focus without MCP dependencies
- Optimal resource utilization
- 2025 Python best practices
"""

# Core training system components
from .training_system_manager import TrainingSystemManager
from .cli_orchestrator import CLIOrchestrator
from .progress_preservation import ProgressPreservationManager

# Signal handling and emergency operations
from .signal_handler import AsyncSignalHandler, ShutdownContext, ShutdownReason
from .emergency_operations import EmergencyOperationsManager

# Process and session management
from .unified_process_manager import UnifiedProcessManager
from .session_resume import SessionResumeManager

# System monitoring and reporting
from .system_state_reporter import SystemStateReporter
from .enhanced_workflow_manager import EnhancedWorkflowManager

# Validation services
from .rule_validation_service import RuleValidationService

__all__ = [
    # Core training components
    "TrainingSystemManager",
    "CLIOrchestrator", 
    "ProgressPreservationManager",
    
    # Signal handling
    "AsyncSignalHandler",
    "ShutdownContext",
    "ShutdownReason",
    
    # Emergency operations
    "EmergencyOperationsManager",
    
    # Process management
    "UnifiedProcessManager",
    "SessionResumeManager",
    
    # System monitoring
    "SystemStateReporter",
    "EnhancedWorkflowManager",
    
    # Validation
    "RuleValidationService",
]
