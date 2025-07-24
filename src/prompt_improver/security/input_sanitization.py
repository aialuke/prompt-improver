"""
Enhanced Input Sanitization Service for ML Pipeline Orchestration.

Implements 2025 OWASP security best practices including:
- Prompt injection protection (OWASP LLM01:2025)
- Async validation patterns
- Comprehensive security monitoring
- ML-specific threat detection
- Event-driven security alerts
"""

import asyncio
import html
import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Exception raised for critical security threats."""
    pass

class SecurityThreatLevel(Enum):
    """Security threat levels for validation results."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

class ValidationResult:
    """Result of input validation with security context."""

    def __init__(self, is_valid: bool, sanitized_value: Any = None,
                 threat_level: SecurityThreatLevel = SecurityThreatLevel.LOW,
                 threats_detected: List[str] = None, message: str = ""):
        self.is_valid = is_valid
        self.sanitized_value = sanitized_value
        self.threat_level = threat_level
        self.threats_detected = threats_detected or []
        self.message = message
        self.timestamp = datetime.now(timezone.utc)

@dataclass
class SecurityEvent:
    """Security event for monitoring and alerting."""
    event_type: str
    threat_level: SecurityThreatLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    input_data: Optional[str] = None
    threats_detected: List[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class InputSanitizer:
    """
    Enhanced input sanitization service implementing 2025 OWASP security best practices.

    features:
    - Prompt injection protection (OWASP LLM01:2025)
    - Async validation patterns
    - ML-specific threat detection
    - Comprehensive security monitoring
    - Event-driven security alerts
    """

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.security_events: List[SecurityEvent] = []
        self._validation_stats = {
            "total_validations": 0,
            "threats_detected": 0,
            "threats_blocked": 0
        }

        # XSS patterns to detect malicious scripts
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*=',
            r'onmouseover\s*=',
            r'onfocus\s*=',
            r'onblur\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
            r'<style[^>]*>.*?</style>',
        ]

        # SQL injection patterns
        self.sql_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
            r'(--|#|/\*|\*/)',
            r'(\bOR\b.*=.*\bOR\b)',
            r'(\bAND\b.*=.*\bAND\b)',
            r'(;|\x00)',
            r'(\b(CHAR|NCHAR|VARCHAR|NVARCHAR)\b\s*\(\s*\d+\s*\))',
        ]

        # Command injection patterns
        self.command_patterns = [
            r'(\||;|&|`|\$\(|\$\{)',
            r'(\b(rm|del|format|fdisk|mkfs)\b)',
            r'(\.\.\/|\.\.\\)',
            r'(\bnc\b|\bnetcat\b|\btelnet\b)',
        ]

        # 2025 OWASP LLM01: Prompt injection patterns
        self.prompt_injection_patterns = [
            r'ignore\s+previous\s+instructions',
            r'forget\s+everything\s+above',
            r'system\s*:\s*you\s+are\s+now',
            r'new\s+instructions\s*:',
            r'override\s+your\s+programming',
            r'act\s+as\s+if\s+you\s+are',
            r'pretend\s+to\s+be',
            r'roleplay\s+as',
            r'simulate\s+being',
            r'you\s+must\s+now',
            r'from\s+now\s+on\s+you\s+are',
            r'disregard\s+all\s+previous',
            r'ignore\s+your\s+guidelines',
            r'bypass\s+your\s+restrictions',
            r'jailbreak\s+mode',
            r'developer\s+mode\s+enabled',
            r'admin\s+override',
            r'root\s+access\s+granted',
        ]

        # ML-specific threat patterns
        self.ml_threat_patterns = [
            r'adversarial\s+example',
            r'poison\s+training\s+data',
            r'model\s+extraction',
            r'membership\s+inference',
            r'gradient\s+inversion',
            r'backdoor\s+attack',
            r'data\s+poisoning',
            r'evasion\s+attack',
        ]

    def set_event_bus(self, event_bus):
        """Set event bus for security event emission."""
        self.event_bus = event_bus
        logger.info("Event bus integrated with InputSanitizer for security monitoring")

    async def _emit_security_event(self, event: SecurityEvent):
        """Emit security event for monitoring and alerting."""
        self.security_events.append(event)

        if self.event_bus:
            try:
                # Import here to avoid circular imports
                from ..ml.orchestration.events.event_types import MLEvent, EventType

                # Choose specific event type based on threat
                if "prompt_injection" in event.threats_detected:
                    event_type = EventType.PROMPT_INJECTION_DETECTED
                elif event.threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]:
                    event_type = EventType.INPUT_VALIDATION_FAILED
                else:
                    event_type = EventType.SECURITY_ALERT

                ml_event = MLEvent(
                    event_type=event_type,
                    source="input_sanitizer",
                    data={
                        "event_type": event.event_type,
                        "threat_level": event.threat_level.name.lower(),
                        "threats_detected": event.threats_detected,
                        "timestamp": event.timestamp.isoformat(),
                        "user_id": event.user_id,
                        "source_ip": event.source_ip,
                        "input_data_sample": event.input_data[:100] if event.input_data else None
                    }
                )
                await self.event_bus.emit(ml_event)
            except Exception as e:
                logger.error(f"Failed to emit security event: {e}")

        # Log security events
        if event.threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]:
            logger.error(f"SECURITY ALERT: {event.event_type} - {event.threat_level.name.lower()} - Threats: {event.threats_detected}")
        else:
            logger.warning(f"Security event: {event.event_type} - {event.threat_level.name.lower()}")

    def get_security_stats(self) -> Dict[str, Any]:
        """Get security validation statistics."""
        return {
            **self._validation_stats,
            "recent_events": len([e for e in self.security_events if
                                (datetime.now(timezone.utc) - e.timestamp).seconds < 3600]),
            "high_threat_events": len([e for e in self.security_events if
                                     e.threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]])
        }

    def sanitize_html_input(self, input_text: str) -> str:
        """Sanitize HTML input to prevent XSS attacks."""
        if not isinstance(input_text, str):
            return str(input_text)

        # HTML escape the input
        sanitized = html.escape(input_text, quote=True)

        # Remove any remaining script-like patterns
        for pattern in self.xss_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)

        return sanitized.strip()

    async def validate_input_async(self, input_data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """
        Comprehensive async input validation with 2025 security best practices.

        Args:
            input_data: Data to validate
            context: Validation context (user_id, source_ip, etc.)

        Returns:
            ValidationResult with security assessment
        """
        start_time = time.time()
        context = context or {}
        threats_detected = []
        threat_level = SecurityThreatLevel.LOW

        self._validation_stats["total_validations"] += 1

        try:
            # Handle different input types
            if isinstance(input_data, str):
                result = await self._validate_string_input(input_data, threats_detected)
                if not result.is_valid:
                    threat_level = result.threat_level

            elif isinstance(input_data, dict):
                result = await self._validate_dict_input(input_data, threats_detected)
                if not result.is_valid:
                    threat_level = result.threat_level

            elif isinstance(input_data, (list, tuple)):
                result = await self._validate_list_input(input_data, threats_detected)
                if not result.is_valid:
                    threat_level = result.threat_level

            elif isinstance(input_data, np.ndarray):
                result = await self._validate_numpy_input(input_data, threats_detected)
                if not result.is_valid:
                    threat_level = result.threat_level

            else:
                result = ValidationResult(
                    is_valid=True,
                    sanitized_value=input_data,
                    message="Input type validation passed"
                )

            # Update statistics
            if threats_detected:
                self._validation_stats["threats_detected"] += len(threats_detected)
                if not result.is_valid:
                    self._validation_stats["threats_blocked"] += 1

            # Emit security event if threats detected
            if threats_detected:
                await self._emit_security_event(SecurityEvent(
                    event_type="input_validation",
                    threat_level=threat_level,
                    source_ip=context.get("source_ip"),
                    user_id=context.get("user_id"),
                    input_data=str(input_data)[:500],  # Truncate for logging
                    threats_detected=threats_detected
                ))

            # Log validation performance
            validation_time = time.time() - start_time
            if validation_time > 0.1:  # Log slow validations
                logger.warning(f"Slow input validation: {validation_time:.3f}s for {type(input_data).__name__}")

            return result

        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return ValidationResult(
                is_valid=False,
                threat_level=SecurityThreatLevel.HIGH,
                threats_detected=["validation_error"],
                message=f"Validation failed: {str(e)}"
            )

    async def _validate_string_input(self, text: str, threats_detected: List[str]) -> ValidationResult:
        """Validate string input for various security threats."""
        if not isinstance(text, str):
            return ValidationResult(is_valid=False, message="Input must be string")

        sanitized_text = text
        threat_level = SecurityThreatLevel.LOW

        # Check for prompt injection (OWASP LLM01:2025)
        for pattern in self.prompt_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats_detected.append("prompt_injection")
                threat_level = SecurityThreatLevel.CRITICAL
                break

        # Check for XSS
        for pattern in self.xss_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                threats_detected.append("xss_attack")
                threat_level = max(threat_level, SecurityThreatLevel.HIGH)
                sanitized_text = self.sanitize_html_input(text)
                break

        # Check for SQL injection
        if not self.validate_sql_input(text):
            threats_detected.append("sql_injection")
            threat_level = max(threat_level, SecurityThreatLevel.HIGH)

        # Check for command injection
        if not self.validate_command_input(text):
            threats_detected.append("command_injection")
            threat_level = max(threat_level, SecurityThreatLevel.HIGH)

        # Check for ML-specific threats
        for pattern in self.ml_threat_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats_detected.append("ml_threat")
                threat_level = max(threat_level, SecurityThreatLevel.MEDIUM)
                break

        # For security, block HIGH and CRITICAL threats, allow LOW and MEDIUM with sanitization
        is_valid = threat_level < SecurityThreatLevel.HIGH

        return ValidationResult(
            is_valid=is_valid,
            sanitized_value=sanitized_text if is_valid else None,
            threat_level=threat_level,
            threats_detected=threats_detected,
            message="String validation completed"
        )

    def validate_sql_input(self, input_text: str) -> bool:
        """Validate input for SQL injection patterns."""
        if not isinstance(input_text, str):
            return True

        # Check for SQL injection patterns
        for pattern in self.sql_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                return False

        return True

    def validate_command_input(self, input_text: str) -> bool:
        """Validate input for command injection patterns."""
        if not isinstance(input_text, str):
            return True

        # Check for command injection patterns
        for pattern in self.command_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                return False

        return True

    async def _validate_dict_input(self, data: dict, threats_detected: List[str]) -> ValidationResult:
        """Validate dictionary input recursively."""
        sanitized_dict = {}
        threat_level = SecurityThreatLevel.LOW

        for key, value in data.items():
            # Validate key
            key_result = await self._validate_string_input(str(key), threats_detected)
            if not key_result.is_valid:
                threat_level = max(threat_level, key_result.threat_level)
                continue

            # Validate value recursively
            if isinstance(value, str):
                value_result = await self._validate_string_input(value, threats_detected)
                if value_result.is_valid:
                    sanitized_dict[key_result.sanitized_value] = value_result.sanitized_value
                else:
                    threat_level = max(threat_level, value_result.threat_level)
            elif isinstance(value, dict):
                value_result = await self._validate_dict_input(value, threats_detected)
                if value_result.is_valid:
                    sanitized_dict[key_result.sanitized_value] = value_result.sanitized_value
                else:
                    threat_level = max(threat_level, value_result.threat_level)
            elif isinstance(value, (list, tuple)):
                value_result = await self._validate_list_input(value, threats_detected)
                if value_result.is_valid:
                    sanitized_dict[key_result.sanitized_value] = value_result.sanitized_value
                else:
                    threat_level = max(threat_level, value_result.threat_level)
            else:
                sanitized_dict[key_result.sanitized_value] = value

        is_valid = threat_level != SecurityThreatLevel.CRITICAL

        return ValidationResult(
            is_valid=is_valid,
            sanitized_value=sanitized_dict if is_valid else None,
            threat_level=threat_level,
            threats_detected=threats_detected,
            message="Dictionary validation completed"
        )

    async def _validate_list_input(self, data: Union[list, tuple], threats_detected: List[str]) -> ValidationResult:
        """Validate list/tuple input recursively."""
        sanitized_list = []
        threat_level = SecurityThreatLevel.LOW

        for item in data:
            if isinstance(item, str):
                item_result = await self._validate_string_input(item, threats_detected)
                if item_result.is_valid:
                    sanitized_list.append(item_result.sanitized_value)
                else:
                    threat_level = max(threat_level, item_result.threat_level)
            elif isinstance(item, dict):
                item_result = await self._validate_dict_input(item, threats_detected)
                if item_result.is_valid:
                    sanitized_list.append(item_result.sanitized_value)
                else:
                    threat_level = max(threat_level, item_result.threat_level)
            elif isinstance(item, (list, tuple)):
                item_result = await self._validate_list_input(item, threats_detected)
                if item_result.is_valid:
                    sanitized_list.append(item_result.sanitized_value)
                else:
                    threat_level = max(threat_level, item_result.threat_level)
            else:
                sanitized_list.append(item)

        is_valid = threat_level != SecurityThreatLevel.CRITICAL

        return ValidationResult(
            is_valid=is_valid,
            sanitized_value=sanitized_list if is_valid else None,
            threat_level=threat_level,
            threats_detected=threats_detected,
            message="List validation completed"
        )

    async def _validate_numpy_input(self, data: np.ndarray, threats_detected: List[str]) -> ValidationResult:
        """Validate numpy array input for ML safety."""
        threat_level = SecurityThreatLevel.LOW

        # Check for NaN or infinite values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            threats_detected.append("invalid_numeric_data")
            threat_level = SecurityThreatLevel.MEDIUM

        # Check for extremely large values (potential adversarial examples)
        if np.any(np.abs(data) > 1e10):
            threats_detected.append("extreme_values")
            threat_level = SecurityThreatLevel.MEDIUM

        # Check for suspicious patterns that might indicate adversarial examples
        if data.size > 0:
            data_std = np.std(data)
            data_mean = np.mean(data)

            # Detect potential adversarial perturbations
            if data_std > 1000 or abs(data_mean) > 1000:
                threats_detected.append("suspicious_data_distribution")
                threat_level = SecurityThreatLevel.MEDIUM

        is_valid = threat_level != SecurityThreatLevel.CRITICAL

        return ValidationResult(
            is_valid=is_valid,
            sanitized_value=data if is_valid else None,
            threat_level=threat_level,
            threats_detected=threats_detected,
            message="NumPy array validation completed"
        )

    def validate_ml_input_data(self, data: Any) -> bool:
        """Validate ML input data for safety and consistency."""
        try:
            # Handle numpy arrays
            if isinstance(data, np.ndarray):
                # Check for NaN or infinite values
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    return False

                # Check for reasonable data ranges
                if np.any(np.abs(data) > 1e10):  # Extremely large values
                    return False

                return True

            # Handle lists
            if isinstance(data, (list, tuple)):
                if len(data) == 0:
                    return False

                for item in data:
                    if not self.validate_ml_input_data(item):
                        return False

                return True

            # Handle individual numbers
            if isinstance(data, (int, float)):
                if math.isnan(data) or math.isinf(data):
                    return False

                if abs(data) > 1e10:
                    return False

                return True

            # Handle dictionaries (for structured data)
            if isinstance(data, dict):
                for key, value in data.items():
                    if not isinstance(key, str) or not self.validate_ml_input_data(value):
                        return False

                return True

            # Reject other types
            return False

        except Exception:
            return False

    def validate_privacy_parameters(self, epsilon: float, delta: float = None) -> bool:
        """Validate differential privacy parameters."""
        # Epsilon must be positive and reasonable
        if not isinstance(epsilon, (int, float)) or epsilon <= 0 or epsilon > 10:
            return False

        # Delta must be positive and very small (if provided)
        if delta is not None:
            if not isinstance(delta, (int, float)) or delta <= 0 or delta >= 1:
                return False

        return True

    def sanitize_file_path(self, file_path: str) -> str:
        """Sanitize file path to prevent directory traversal attacks."""
        if not isinstance(file_path, str):
            return ""

        # Remove directory traversal patterns
        sanitized = re.sub(r'\.\.\/|\.\.\\', '', file_path)

        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')

        # Remove leading/trailing whitespace
        sanitized = sanitized.strip()

        # Only allow alphanumeric, dash, underscore, dot, and slash
        sanitized = re.sub(r'[^a-zA-Z0-9._/-]', '', sanitized)

        return sanitized

    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        if not isinstance(email, str):
            return False

        # RFC 5322 compliant regex (simplified)
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

        if not re.match(email_pattern, email):
            return False

        # Additional checks
        if len(email) > 254:  # RFC 5321 limit
            return False

        return True

    def validate_username(self, username: str) -> bool:
        """Validate username format."""
        if not isinstance(username, str):
            return False

        # Username requirements: 3-50 chars, alphanumeric + underscore/dash
        if not re.match(r'^[a-zA-Z0-9_-]{3,50}$', username):
            return False

        return True

    def validate_password_strength(self, password: str) -> dict[str, bool | list[str]]:
        """Validate password strength and return detailed feedback."""
        if not isinstance(password, str):
            return {"valid": False, "errors": ["Password must be a string"]}

        errors = []

        # Length check
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")

        if len(password) > 128:
            errors.append("Password must be no more than 128 characters long")

        # Character diversity checks
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")

        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")

        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")

        # Common pattern checks
        if re.search(r'(.)\1{3,}', password):  # 4+ repeated characters
            errors.append("Password must not contain 4 or more repeated characters")

        if re.search(r'(123|abc|qwe|asd|zxc)', password, re.IGNORECASE):
            errors.append("Password must not contain common patterns")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "strength_score": max(0, 100 - (len(errors) * 15))
        }

    def sanitize_json_input(self, json_data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize JSON input data."""
        if not isinstance(json_data, dict):
            return {}

        sanitized = {}

        for key, value in json_data.items():
            # Sanitize keys
            clean_key = self.sanitize_html_input(str(key))

            # Sanitize values recursively
            if isinstance(value, str):
                clean_value = self.sanitize_html_input(value)
            elif isinstance(value, dict):
                clean_value = self.sanitize_json_input(value)
            elif isinstance(value, list):
                clean_value = [self.sanitize_html_input(str(item)) if isinstance(item, str) else item for item in value]
            else:
                clean_value = value

            sanitized[clean_key] = clean_value

        return sanitized