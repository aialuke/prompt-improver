"""
Comprehensive logging system for configuration loading and validation.
Provides detailed logging with multiple output formats and log levels.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class ConfigurationEvent:
    """Represents a configuration-related event for logging."""
    timestamp: datetime
    event_type: str  # 'load', 'validate', 'error', 'warning', 'migration'
    component: str
    message: str
    details: Optional[Dict[str, Any]] = None
    level: str = 'INFO'
    environment: Optional[str] = None
    config_file: Optional[str] = None


class ConfigurationLogger:
    """
    Specialized logger for configuration operations with structured output.
    
    Provides multiple output formats and detailed tracking of configuration
    loading, validation, and migration operations.
    """
    
    def __init__(self, 
                 logger_name: str = "prompt_improver.config",
                 log_level: str = "INFO",
                 enable_file_logging: bool = True,
                 log_file_path: Optional[str] = None):
        """
        Initialize configuration logger.
        
        Args:
            logger_name: Name for the logger instance
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_file_logging: Whether to log to file in addition to console
            log_file_path: Custom path for log file
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for structured logging
        if enable_file_logging:
            if log_file_path is None:
                log_file_path = f"config_operations_{datetime.now().strftime('%Y%m%d')}.log"
            
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)  # Always capture all details in file
            file_formatter = StructuredFormatter()
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file_path = log_file_path
        else:
            self.log_file_path = None
        
        # Event storage for reporting
        self.events: List[ConfigurationEvent] = []
        
    def log_configuration_load(self, 
                             config_file: str,
                             environment: str,
                             success: bool,
                             details: Optional[Dict[str, Any]] = None) -> None:
        """Log configuration file loading operation."""
        event_type = "load_success" if success else "load_error"
        level = "INFO" if success else "ERROR"
        
        message = f"Configuration loaded: {config_file}"
        if not success:
            message = f"Failed to load configuration: {config_file}"
        
        event = ConfigurationEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            component="config_loader",
            message=message,
            details=details,
            level=level,
            environment=environment,
            config_file=config_file
        )
        
        self._log_event(event)
    
    def log_validation_result(self,
                            component: str,
                            is_valid: bool,
                            message: str,
                            details: Optional[Dict[str, Any]] = None,
                            environment: Optional[str] = None) -> None:
        """Log configuration validation result."""
        event_type = "validation_success" if is_valid else "validation_error"
        level = "INFO" if is_valid else "WARNING"
        
        event = ConfigurationEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            component=component,
            message=message,
            details=details,
            level=level,
            environment=environment
        )
        
        self._log_event(event)
    
    def log_migration(self,
                     config_file: str,
                     from_version: str,
                     to_version: str,
                     success: bool,
                     details: Optional[Dict[str, Any]] = None) -> None:
        """Log configuration migration operation."""
        event_type = "migration_success" if success else "migration_error"
        level = "INFO" if success else "ERROR"
        
        message = f"Migration {from_version} -> {to_version}"
        if not success:
            message = f"Migration failed: {from_version} -> {to_version}"
        
        event = ConfigurationEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            component="config_migrator",
            message=message,
            details={
                **(details or {}),
                "from_version": from_version,
                "to_version": to_version
            },
            level=level,
            config_file=config_file
        )
        
        self._log_event(event)
    
    def log_connectivity_test(self,
                            service: str,
                            success: bool,
                            response_time_ms: Optional[float] = None,
                            error: Optional[str] = None) -> None:
        """Log service connectivity test result."""
        event_type = "connectivity_success" if success else "connectivity_error"
        level = "INFO" if success else "WARNING"
        
        message = f"{service} connectivity test"
        if not success:
            message = f"{service} connectivity failed"
        
        details = {}
        if response_time_ms is not None:
            details["response_time_ms"] = response_time_ms
        if error:
            details["error"] = error
        
        event = ConfigurationEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            component="connectivity_tester",
            message=message,
            details=details if details else None,
            level=level
        )
        
        self._log_event(event)
    
    def log_startup_summary(self,
                          environment: str,
                          config_files: List[str],
                          validation_results: Dict[str, bool],
                          startup_time_ms: float) -> None:
        """Log comprehensive startup configuration summary."""
        total_validations = len(validation_results)
        successful_validations = sum(validation_results.values())
        
        message = f"Startup configuration summary - {successful_validations}/{total_validations} validations passed"
        
        details = {
            "environment": environment,
            "config_files": config_files,
            "validation_results": validation_results,
            "startup_time_ms": startup_time_ms,
            "success_rate": successful_validations / total_validations if total_validations > 0 else 0
        }
        
        level = "INFO" if successful_validations == total_validations else "WARNING"
        
        event = ConfigurationEvent(
            timestamp=datetime.now(),
            event_type="startup_summary",
            component="startup_validator",
            message=message,
            details=details,
            level=level,
            environment=environment
        )
        
        self._log_event(event)
    
    def _log_event(self, event: ConfigurationEvent) -> None:
        """Internal method to log an event."""
        # Store event for reporting
        self.events.append(event)
        
        # Log to standard logging system
        log_level = getattr(logging, event.level.upper())
        self.logger.log(log_level, event.message, extra={
            'event_type': event.event_type,
            'component': event.component,
            'details': event.details,
            'environment': event.environment,
            'config_file': event.config_file
        })
    
    @contextmanager
    def log_operation(self, operation_name: str, component: str):
        """Context manager for logging operations with timing."""
        start_time = datetime.now()
        self.logger.info(f"Starting {operation_name}")
        
        try:
            yield
            
            # Operation succeeded
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            event = ConfigurationEvent(
                timestamp=end_time,
                event_type="operation_success",
                component=component,
                message=f"Completed {operation_name}",
                details={"duration_ms": duration_ms},
                level="INFO"
            )
            self._log_event(event)
            
        except Exception as e:
            # Operation failed
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            event = ConfigurationEvent(
                timestamp=end_time,
                event_type="operation_error",
                component=component,
                message=f"Failed {operation_name}: {str(e)}",
                details={
                    "duration_ms": duration_ms,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                level="ERROR"
            )
            self._log_event(event)
            raise
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive report of all logged events."""
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "total_events": len(self.events),
            "events_by_type": {},
            "events_by_level": {},
            "events_by_component": {},
            "timeline": []
        }
        
        # Aggregate statistics
        for event in self.events:
            # By type
            report["events_by_type"][event.event_type] = report["events_by_type"].get(event.event_type, 0) + 1
            
            # By level
            report["events_by_level"][event.level] = report["events_by_level"].get(event.level, 0) + 1
            
            # By component
            report["events_by_component"][event.component] = report["events_by_component"].get(event.component, 0) + 1
            
            # Timeline entry
            report["timeline"].append(asdict(event))
        
        # Save to file if requested
        if output_file:
            Path(output_file).write_text(json.dumps(report, indent=2, default=str))
            self.logger.info(f"Configuration report saved to: {output_file}")
        
        return report
    
    def get_error_summary(self) -> List[ConfigurationEvent]:
        """Get summary of all error events."""
        return [event for event in self.events if event.level in ['ERROR', 'WARNING']]
    
    def clear_events(self) -> None:
        """Clear stored events (useful for testing or memory management)."""
        self.events.clear()


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'event_type'):
            log_entry['event_type'] = record.event_type
        if hasattr(record, 'component'):
            log_entry['component'] = record.component
        if hasattr(record, 'details'):
            log_entry['details'] = record.details
        if hasattr(record, 'environment'):
            log_entry['environment'] = record.environment
        if hasattr(record, 'config_file'):
            log_entry['config_file'] = record.config_file
        
        return json.dumps(log_entry)


# Global configuration logger instance
config_logger = ConfigurationLogger()


def get_config_logger() -> ConfigurationLogger:
    """Get the global configuration logger instance."""
    return config_logger


def setup_config_logging(log_level: str = "INFO", 
                        enable_file_logging: bool = True,
                        log_file_path: Optional[str] = None) -> ConfigurationLogger:
    """
    Setup configuration logging with custom parameters.
    
    Args:
        log_level: Logging level
        enable_file_logging: Whether to enable file logging
        log_file_path: Custom log file path
        
    Returns:
        Configured ConfigurationLogger instance
    """
    global config_logger
    config_logger = ConfigurationLogger(
        log_level=log_level,
        enable_file_logging=enable_file_logging,
        log_file_path=log_file_path
    )
    return config_logger


if __name__ == "__main__":
    """Demo of configuration logging capabilities."""
    
    # Setup logger
    logger = setup_config_logging(log_level="DEBUG")
    
    # Demo various logging operations
    with logger.log_operation("configuration_load", "demo"):
        logger.log_configuration_load(
            config_file=".env.development",
            environment="development", 
            success=True,
            details={"variables_loaded": 45}
        )
    
    logger.log_validation_result(
        component="database",
        is_valid=True,
        message="Database configuration validated successfully",
        details={"host": "localhost", "port": 5432}
    )
    
    logger.log_connectivity_test(
        service="postgresql",
        success=True,
        response_time_ms=23.5
    )
    
    logger.log_connectivity_test(
        service="redis",
        success=False,
        error="Connection refused"
    )
    
    logger.log_startup_summary(
        environment="development",
        config_files=[".env.development"],
        validation_results={"database": True, "redis": False, "ml": True},
        startup_time_ms=1250.0
    )
    
    # Generate report
    report = logger.generate_report("config_logging_demo_report.json")
    print(f"\\nGenerated report with {report['total_events']} events")
    
    # Show error summary
    errors = logger.get_error_summary()
    if errors:
        print(f"Found {len(errors)} warnings/errors:")
        for error in errors:
            print(f"  - {error.component}: {error.message}")
    else:
        print("No errors or warnings found")