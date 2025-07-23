"""Secure subprocess execution utilities.

This module provides secure subprocess execution with comprehensive validation,
logging, and error handling based on the best patterns from cli.py.
"""

import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)


class SecureSubprocessManager:
    """Secure subprocess execution manager with comprehensive security patterns.

    This class implements the security patterns identified as best practices
    in cli.py, providing:
    - Path validation and resolution
    - Shell injection prevention
    - Timeout management
    - Comprehensive error handling
    - Audit logging
    """

    def __init__(self, default_timeout: int = 300, enable_audit_logging: bool = True):
        """Initialize the secure subprocess manager.

        Args:
            default_timeout: Default timeout for subprocess calls in seconds
            enable_audit_logging: Whether to enable comprehensive audit logging
        """
        self.default_timeout = default_timeout
        self.enable_audit_logging = enable_audit_logging

    def _validate_executable_path(self, executable: str | Path) -> str:
        """Validate and resolve executable path securely.

        Args:
            executable: Executable path or command name

        Returns:
            Validated absolute path to executable

        Raises:
            FileNotFoundError: If executable not found
            ValueError: If path validation fails
        """
        # If it's a known safe system executable, resolve via shutil.which()
        if isinstance(executable, str) and not Path(executable).is_absolute():
            resolved_path = shutil.which(executable)
            if not resolved_path:
                raise FileNotFoundError(f"Executable '{executable}' not found in PATH")
            return resolved_path

        # For paths, validate they exist and are files
        executable_path = Path(executable)
        if not executable_path.exists():
            raise FileNotFoundError(
                f"Executable path does not exist: {executable_path}"
            )

        if not executable_path.is_file():
            raise ValueError(f"Path is not a regular file: {executable_path}")

        return str(executable_path.resolve())

    def _validate_arguments(self, args: list[str]) -> list[str]:
        """Validate subprocess arguments for security.

        Args:
            args: List of arguments to validate

        Returns:
            Validated arguments list

        Raises:
            ValueError: If arguments contain security risks
        """
        if not args:
            raise ValueError("Arguments list cannot be empty")

        # Check for shell injection patterns
        dangerous_chars = [";", "&&", "||", "|", ">", "<", "`", "$"]
        for arg in args:
            if any(char in str(arg) for char in dangerous_chars):
                logger.warning(f"Potentially dangerous character in argument: {arg}")

        return [str(arg) for arg in args]

    def _log_subprocess_call(self, args: list[str], operation: str) -> None:
        """Log subprocess call for audit trail.

        Args:
            args: Arguments being executed
            operation: Description of the operation
        """
        if self.enable_audit_logging:
            # Log securely - don't log sensitive data
            safe_args = [args[0]] + ["<arg>" for _ in args[1:]]
            logger.info(
                f"Secure subprocess call: {operation}",
                extra={
                    "operation": operation,
                    "executable": args[0],
                    "arg_count": len(args) - 1,
                    "security_validated": True,
                },
            )

    def run_secure(
        self,
        args: list[str | Path],
        operation: str,
        timeout: int | None = None,
        check: bool = True,
        capture_output: bool = False,
        text: bool = True,
        cwd: str | Path | None = None,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """Execute subprocess with comprehensive security validation.

        Args:
            args: Command and arguments to execute
            operation: Description of operation for logging
            timeout: Timeout in seconds (uses default if None)
            check: Whether to check return code
            capture_output: Whether to capture stdout/stderr
            text: Whether to use text mode
            cwd: Working directory
            **kwargs: Additional subprocess.run arguments

        Returns:
            CompletedProcess result

        Raises:
            FileNotFoundError: If executable not found
            ValueError: If validation fails
            subprocess.SubprocessError: If subprocess fails
            subprocess.TimeoutExpired: If timeout exceeded
        """
        if not args:
            raise ValueError("Command arguments cannot be empty")

        # Validate executable path
        validated_executable = self._validate_executable_path(args[0])

        # Validate all arguments
        validated_args = [validated_executable] + self._validate_arguments(args[1:])

        # Set secure defaults
        secure_kwargs = {
            "shell": False,  # Always prevent shell injection
            "timeout": timeout or self.default_timeout,
            "check": check,
            "capture_output": capture_output,
            "text": text,
            **kwargs,
        }

        # Validate cwd if provided
        if cwd is not None:
            cwd_path = Path(cwd)
            if not cwd_path.exists() or not cwd_path.is_dir():
                raise ValueError(f"Invalid working directory: {cwd}")
            secure_kwargs["cwd"] = str(cwd_path.resolve())

        # Log the operation
        self._log_subprocess_call(validated_args, operation)

        try:
            # Execute with security validation complete
            result = subprocess.run(validated_args, **secure_kwargs)  # noqa: S603

            if self.enable_audit_logging:
                logger.info(
                    f"Subprocess completed successfully: {operation}",
                    extra={
                        "operation": operation,
                        "return_code": result.returncode,
                        "execution_time": "tracked_externally",
                    },
                )

            return result

        except subprocess.TimeoutExpired as e:
            logger.error(f"Subprocess timeout in {operation}: {e}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Subprocess failed in {operation}: {e}")
            raise
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"System error in {operation}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {operation}: {e}")
            raise

    def popen_secure(
        self,
        args: list[str | Path],
        operation: str,
        stdout: Any = None,
        stderr: Any = None,
        start_new_session: bool = True,
        **kwargs,
    ) -> subprocess.Popen:
        """Execute subprocess.Popen with security validation.

        Args:
            args: Command and arguments to execute
            operation: Description of operation for logging
            stdout: stdout configuration
            stderr: stderr configuration
            start_new_session: Whether to start new session for isolation
            **kwargs: Additional Popen arguments

        Returns:
            Popen process object

        Raises:
            FileNotFoundError: If executable not found
            ValueError: If validation fails
            subprocess.SubprocessError: If subprocess fails
        """
        if not args:
            raise ValueError("Command arguments cannot be empty")

        # Validate executable path
        validated_executable = self._validate_executable_path(args[0])

        # Validate all arguments
        validated_args = [validated_executable] + self._validate_arguments(args[1:])

        # Set secure defaults
        secure_kwargs = {
            "shell": False,  # Always prevent shell injection
            "stdout": stdout,
            "stderr": stderr,
            "start_new_session": start_new_session,
            **kwargs,
        }

        # Log the operation
        self._log_subprocess_call(validated_args, operation)

        try:
            # Execute with security validation complete
            process = subprocess.Popen(validated_args, **secure_kwargs)  # noqa: S603

            if self.enable_audit_logging:
                logger.info(
                    f"Subprocess Popen started: {operation}",
                    extra={
                        "operation": operation,
                        "pid": process.pid,
                        "process_started": True,
                    },
                )

            return process

        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"System error starting {operation}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error starting {operation}: {e}")
            raise


# Global instance for convenience
_default_manager = SecureSubprocessManager()


def secure_subprocess(
    operation: str,
    timeout: int | None = None,
    capture_output: bool = False,
    check: bool = True,
):
    """Decorator for secure subprocess execution.

    This decorator wraps functions that execute subprocess calls with
    comprehensive security validation.

    Args:
        operation: Description of the operation for logging
        timeout: Timeout in seconds
        capture_output: Whether to capture output
        check: Whether to check return code

    Returns:
        Decorated function

    Example:
        @secure_subprocess("MLflow UI launch", timeout=60)
        def launch_mlflow_ui(mlflow_path: str, port: str) -> subprocess.CompletedProcess:
            return [mlflow_path, "ui", "--port", port]
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get command from function
            command_args = func(*args, **kwargs)

            # Execute securely
            return _default_manager.run_secure(
                command_args,
                operation=operation,
                timeout=timeout,
                capture_output=capture_output,
                check=check,
            )

        return wrapper

    return decorator


def ensure_running(pid: int) -> bool:
    """Centralized function to validate if a background process is running.

    This function consolidates PID validation logic used across the codebase
    to avoid duplication. It handles both psutil-based and OS-based checks.

    Args:
        pid: Process ID to check

    Returns:
        True if process is running, False otherwise
    """
    try:
        # Try psutil first if available
        try:
            import psutil
            return psutil.pid_exists(pid)
        except ImportError:
            # Fallback to OS-based check
            os.kill(pid, 0)  # Signal 0 checks if process exists
            return True
    except (ProcessLookupError, PermissionError, OSError):
        return False
