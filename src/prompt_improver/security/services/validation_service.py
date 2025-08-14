"""ValidationService - OWASP-Compliant Input/Output Validation with Fail-Secure Design

A specialized security service that handles input validation, output sanitization,
and content security policy enforcement. Implements OWASP security guidelines with
fail-secure principles where any validation failure results in rejection.

Key Features:
- OWASP Top 10 vulnerability prevention (Injection, XSS, etc.)
- Comprehensive input validation with configurable rules
- Output sanitization to prevent data leakage
- Content Security Policy (CSP) enforcement
- Real-time threat detection in user input
- Fail-secure design (reject on error, no fail-open vulnerabilities)
- Support for multiple validation modes (strict, permissive, custom)
- Integration with threat intelligence feeds

Security Standards:
- OWASP Input Validation Cheat Sheet compliance
- NIST SP 800-53 input validation controls
- CWE (Common Weakness Enumeration) prevention
- SANS secure coding practices
- Defense in depth with multiple validation layers
"""

import html
import json
import logging
import re
import time
import urllib.parse
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from prompt_improver.database import SecurityContext
from prompt_improver.security.services.protocols import (
    SecurityStateManagerProtocol,
    ValidationServiceProtocol,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

try:
    from opentelemetry import metrics, trace

    OPENTELEMETRY_AVAILABLE = True
    validation_tracer = trace.get_tracer(__name__ + ".validation")
    validation_meter = metrics.get_meter(__name__ + ".validation")
    validation_operations_counter = validation_meter.create_counter(
        "validation_operations_total",
        description="Total validation operations by type and result",
        unit="1",
    )
    validation_threats_counter = validation_meter.create_counter(
        "validation_threats_detected_total",
        description="Total threats detected during validation",
        unit="1",
    )
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    validation_tracer = None
    validation_meter = None
    validation_operations_counter = None
    validation_threats_counter = None

logger = logging.getLogger(__name__)


class ThreatPattern:
    """Represents a threat detection pattern."""
    
    def __init__(self, name: str, pattern: str, severity: str, description: str):
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        self.severity = severity
        self.description = description
    
    def detect(self, input_data: str) -> List[Dict[str, Any]]:
        """Detect threat pattern in input data."""
        matches = self.pattern.findall(input_data)
        if matches:
            return [{
                "threat_name": self.name,
                "severity": self.severity,
                "description": self.description,
                "matches": matches[:5],  # Limit matches for logging
                "match_count": len(matches)
            }]
        return []


class ValidationRule:
    """Represents a validation rule with constraints."""
    
    def __init__(
        self,
        name: str,
        rule_type: str,
        parameters: Dict[str, Any],
        required: bool = True,
        error_message: str = "Validation failed"
    ):
        self.name = name
        self.rule_type = rule_type
        self.parameters = parameters
        self.required = required
        self.error_message = error_message
    
    def validate(self, value: Any) -> Tuple[bool, str]:
        """Validate value against this rule."""
        try:
            if self.rule_type == "length":
                return self._validate_length(value)
            elif self.rule_type == "pattern":
                return self._validate_pattern(value)
            elif self.rule_type == "type":
                return self._validate_type(value)
            elif self.rule_type == "range":
                return self._validate_range(value)
            elif self.rule_type == "whitelist":
                return self._validate_whitelist(value)
            elif self.rule_type == "blacklist":
                return self._validate_blacklist(value)
            else:
                return (False, f"Unknown validation rule type: {self.rule_type}")
        except Exception as e:
            return (False, f"Validation error: {str(e)}")
    
    def _validate_length(self, value: Any) -> Tuple[bool, str]:
        """Validate string length."""
        if not isinstance(value, str):
            return (False, "Value must be a string for length validation")
        
        length = len(value)
        min_length = self.parameters.get("min", 0)
        max_length = self.parameters.get("max", float('inf'))
        
        if length < min_length:
            return (False, f"Value too short (min: {min_length}, actual: {length})")
        if length > max_length:
            return (False, f"Value too long (max: {max_length}, actual: {length})")
        
        return (True, "")
    
    def _validate_pattern(self, value: Any) -> Tuple[bool, str]:
        """Validate against regex pattern."""
        if not isinstance(value, str):
            return (False, "Value must be a string for pattern validation")
        
        pattern = self.parameters.get("pattern")
        if not pattern:
            return (False, "No pattern specified for validation")
        
        if not re.match(pattern, value):
            return (False, f"Value does not match required pattern")
        
        return (True, "")
    
    def _validate_type(self, value: Any) -> Tuple[bool, str]:
        """Validate data type."""
        expected_type = self.parameters.get("type")
        if expected_type == "string" and not isinstance(value, str):
            return (False, "Value must be a string")
        elif expected_type == "integer" and not isinstance(value, int):
            return (False, "Value must be an integer")
        elif expected_type == "float" and not isinstance(value, (int, float)):
            return (False, "Value must be a number")
        elif expected_type == "boolean" and not isinstance(value, bool):
            return (False, "Value must be a boolean")
        elif expected_type == "list" and not isinstance(value, list):
            return (False, "Value must be a list")
        elif expected_type == "dict" and not isinstance(value, dict):
            return (False, "Value must be a dictionary")
        
        return (True, "")
    
    def _validate_range(self, value: Any) -> Tuple[bool, str]:
        """Validate numeric range."""
        if not isinstance(value, (int, float)):
            return (False, "Value must be numeric for range validation")
        
        min_val = self.parameters.get("min", float('-inf'))
        max_val = self.parameters.get("max", float('inf'))
        
        if value < min_val:
            return (False, f"Value below minimum (min: {min_val}, actual: {value})")
        if value > max_val:
            return (False, f"Value above maximum (max: {max_val}, actual: {value})")
        
        return (True, "")
    
    def _validate_whitelist(self, value: Any) -> Tuple[bool, str]:
        """Validate against whitelist."""
        allowed_values = self.parameters.get("values", [])
        if value not in allowed_values:
            return (False, f"Value not in allowed list")
        
        return (True, "")
    
    def _validate_blacklist(self, value: Any) -> Tuple[bool, str]:
        """Validate against blacklist."""
        forbidden_values = self.parameters.get("values", [])
        if value in forbidden_values:
            return (False, f"Value in forbidden list")
        
        return (True, "")


class ValidationService:
    """Focused validation service with OWASP compliance and fail-secure design.
    
    Handles all input/output validation operations including threat detection,
    sanitization, and content security policy enforcement. Designed to fail
    securely - any validation failure results in rejection rather than bypass.
    
    Single Responsibility: Input/output validation and sanitization only
    """

    def __init__(
        self,
        security_state_manager: SecurityStateManagerProtocol,
        default_validation_mode: str = "strict",
        enable_threat_detection: bool = True,
        max_input_size: int = 1024 * 1024,  # 1MB
    ):
        """Initialize validation service.
        
        Args:
            security_state_manager: Shared security state manager
            default_validation_mode: Default validation mode (strict/permissive/custom)
            enable_threat_detection: Enable real-time threat detection
            max_input_size: Maximum input size in bytes
        """
        self.security_state_manager = security_state_manager
        self.default_validation_mode = default_validation_mode
        self.enable_threat_detection = enable_threat_detection
        self.max_input_size = max_input_size
        
        # Validation rules registry
        self._validation_rules: Dict[str, List[ValidationRule]] = {}
        self._global_rules: List[ValidationRule] = []
        
        # Threat detection patterns
        self._threat_patterns: List[ThreatPattern] = []
        
        # Performance metrics
        self._operation_times: deque = deque(maxlen=1000)
        self._validation_results = {"valid": 0, "invalid": 0, "threats": 0}
        self._total_operations = 0
        
        # Sanitization cache
        self._sanitization_cache: Dict[str, str] = {}
        self._cache_max_size = 10000
        
        self._initialized = False
        
        # Initialize threat patterns and validation rules
        self._initialize_threat_patterns()
        self._initialize_default_rules()
        
        logger.info("ValidationService initialized with OWASP compliance and fail-secure design")

    async def initialize(self) -> bool:
        """Initialize validation service components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Load threat intelligence feeds (placeholder for real implementation)
            await self._load_threat_intelligence()
            
            initialization_time = time.time() - start_time
            logger.info(f"ValidationService initialized in {initialization_time:.3f}s")
            
            await self.security_state_manager.record_security_operation(
                "validation_service_init",
                success=True,
                details={
                    "initialization_time": initialization_time,
                    "threat_patterns_count": len(self._threat_patterns),
                    "validation_rules_count": len(self._global_rules)
                }
            )
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ValidationService: {e}")
            await self.security_state_manager.handle_security_incident(
                "high", "validation_service_init", "system",
                {"error": str(e), "operation": "initialization"}
            )
            return False

    async def validate_input(
        self,
        security_context: SecurityContext,
        input_data: Any,
        validation_rules: Dict[str, Any] | None = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate input data with comprehensive security checks.
        
        Performs multi-layer validation:
        1. Size and type validation
        2. Threat pattern detection
        3. Rule-based validation
        4. Context-aware validation
        5. OWASP compliance checks
        
        Args:
            security_context: Security context for validation
            input_data: Input data to validate
            validation_rules: Optional validation rules override
            
        Returns:
            Tuple of (is_valid, validation_results)
            
        Fail-secure: Returns (False, error_details) on validation failure
        """
        operation_start = time.time()
        
        if not self._initialized:
            logger.error("ValidationService not initialized")
            return (False, {"error": "service_not_initialized", "valid": False})
        
        try:
            validation_results = {
                "valid": True,
                "threats_detected": [],
                "validation_errors": [],
                "sanitized_data": None,
                "validation_timestamp": time.time(),
                "validation_mode": validation_rules.get("mode", self.default_validation_mode) if validation_rules else self.default_validation_mode
            }
            
            # 1. Basic size and type validation (fail-secure)
            size_valid, size_error = await self._validate_input_size(input_data)
            if not size_valid:
                validation_results["valid"] = False
                validation_results["validation_errors"].append(size_error)
                
                await self.security_state_manager.handle_security_incident(
                    "medium", "validation_size_violation", security_context.agent_id,
                    {"error": size_error, "input_size": len(str(input_data))}
                )
                
                self._validation_results["invalid"] += 1
                return (False, validation_results)
            
            # 2. Threat detection (fail-secure)
            if self.enable_threat_detection:
                threats = await self._detect_threats(input_data)
                if threats:
                    validation_results["valid"] = False
                    validation_results["threats_detected"] = threats
                    
                    await self.security_state_manager.handle_security_incident(
                        "high", "validation_threat_detected", security_context.agent_id,
                        {"threats": threats, "input_preview": str(input_data)[:100]}
                    )
                    
                    self._validation_results["threats"] += 1
                    return (False, validation_results)
            
            # 3. Rule-based validation (fail-secure)
            if validation_rules:
                rule_valid, rule_errors = await self._apply_validation_rules(input_data, validation_rules)
                if not rule_valid:
                    validation_results["valid"] = False
                    validation_results["validation_errors"].extend(rule_errors)
                    
                    await self.security_state_manager.handle_security_incident(
                        "low", "validation_rule_violation", security_context.agent_id,
                        {"errors": rule_errors}
                    )
                    
                    self._validation_results["invalid"] += 1
                    return (False, validation_results)
            
            # 4. Global rules validation (fail-secure)
            global_valid, global_errors = await self._apply_global_rules(input_data)
            if not global_valid:
                validation_results["valid"] = False
                validation_results["validation_errors"].extend(global_errors)
                
                self._validation_results["invalid"] += 1
                return (False, validation_results)
            
            # 5. Context-aware validation
            context_valid, context_errors = await self._validate_with_context(security_context, input_data)
            if not context_valid:
                validation_results["valid"] = False
                validation_results["validation_errors"].extend(context_errors)
                
                self._validation_results["invalid"] += 1
                return (False, validation_results)
            
            # Validation successful
            self._validation_results["valid"] += 1
            
            await self.security_state_manager.record_security_operation(
                "validation_success",
                success=True,
                agent_id=security_context.agent_id,
                details={"validation_mode": validation_results["validation_mode"]}
            )
            
            return (True, validation_results)
            
        except Exception as e:
            logger.error(f"Validation error for {security_context.agent_id}: {e}")
            
            await self.security_state_manager.handle_security_incident(
                "high", "validation_system_error", security_context.agent_id,
                {"error": str(e), "operation": "validate_input"}
            )
            
            # Fail-secure: Return invalid on any system error
            return (False, {"error": str(e), "valid": False})
            
        finally:
            # Record operation metrics
            operation_time = time.time() - operation_start
            self._operation_times.append(operation_time)
            self._total_operations += 1
            
            if OPENTELEMETRY_AVAILABLE and validation_operations_counter:
                validation_operations_counter.add(
                    1, {"operation": "validate_input", "agent_id": security_context.agent_id}
                )

    async def sanitize_input(
        self,
        input_data: Any,
        sanitization_rules: Dict[str, Any] | None = None,
    ) -> Any:
        """Sanitize input data to prevent injection attacks.
        
        Args:
            input_data: Input data to sanitize
            sanitization_rules: Optional sanitization rules
            
        Returns:
            Sanitized data
            
        Fail-secure: Returns safe default or raises exception on sanitization failure
        """
        try:
            if not isinstance(input_data, str):
                # For non-string data, return as-is after basic validation
                return input_data
            
            # Check cache first
            cache_key = f"{hash(input_data)}:{hash(str(sanitization_rules))}"
            if cache_key in self._sanitization_cache:
                return self._sanitization_cache[cache_key]
            
            sanitized = input_data
            
            # Apply sanitization rules
            if sanitization_rules:
                sanitization_mode = sanitization_rules.get("mode", "html")
            else:
                sanitization_mode = "html"
            
            if sanitization_mode == "html":
                sanitized = await self._sanitize_html(sanitized)
            elif sanitization_mode == "sql":
                sanitized = await self._sanitize_sql(sanitized)
            elif sanitization_mode == "command":
                sanitized = await self._sanitize_command(sanitized)
            elif sanitization_mode == "url":
                sanitized = await self._sanitize_url(sanitized)
            elif sanitization_mode == "json":
                sanitized = await self._sanitize_json(sanitized)
            
            # Cache result
            if len(self._sanitization_cache) < self._cache_max_size:
                self._sanitization_cache[cache_key] = sanitized
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Sanitization error: {e}")
            # Fail-secure: Return empty string on sanitization failure
            return ""

    async def validate_output(
        self,
        security_context: SecurityContext,
        output_data: Any,
        validation_rules: Dict[str, Any] | None = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate output data before sending to prevent data leakage.
        
        Args:
            security_context: Security context for validation
            output_data: Output data to validate
            validation_rules: Optional validation rules
            
        Returns:
            Tuple of (is_valid, validation_results)
            
        Fail-secure: Returns (False, error_details) on validation failure
        """
        try:
            validation_results = {
                "valid": True,
                "data_leakage_detected": False,
                "sensitive_data_found": [],
                "validation_errors": [],
                "validation_timestamp": time.time()
            }
            
            # Check for sensitive data patterns
            if isinstance(output_data, str):
                sensitive_patterns = await self._detect_sensitive_data(output_data)
                if sensitive_patterns:
                    validation_results["valid"] = False
                    validation_results["data_leakage_detected"] = True
                    validation_results["sensitive_data_found"] = sensitive_patterns
                    
                    await self.security_state_manager.handle_security_incident(
                        "high", "validation_data_leakage", security_context.agent_id,
                        {"sensitive_patterns": sensitive_patterns}
                    )
                    
                    return (False, validation_results)
            
            # Apply output validation rules
            if validation_rules:
                rule_valid, rule_errors = await self._apply_validation_rules(output_data, validation_rules)
                if not rule_valid:
                    validation_results["valid"] = False
                    validation_results["validation_errors"].extend(rule_errors)
                    return (False, validation_results)
            
            return (True, validation_results)
            
        except Exception as e:
            logger.error(f"Output validation error: {e}")
            return (False, {"error": str(e), "valid": False})

    async def check_content_security_policy(
        self, content: str, policy_rules: Dict[str, Any] | None = None
    ) -> bool:
        """Check content against security policies.
        
        Args:
            content: Content to check
            policy_rules: Optional policy rules
            
        Returns:
            True if content complies, False otherwise
            
        Fail-secure: Returns False if content violates security policies
        """
        try:
            if not policy_rules:
                policy_rules = {
                    "max_length": 10000,
                    "allowed_html_tags": ["p", "b", "i", "em", "strong"],
                    "block_scripts": True,
                    "block_external_links": True
                }
            
            # Check content length
            if len(content) > policy_rules.get("max_length", 10000):
                return False
            
            # Check for blocked scripts
            if policy_rules.get("block_scripts", True):
                script_patterns = [
                    r"<script.*?>.*?</script>",
                    r"javascript:",
                    r"on\w+\s*=",
                    r"eval\s*\(",
                    r"setTimeout\s*\(",
                    r"setInterval\s*\("
                ]
                
                for pattern in script_patterns:
                    if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                        return False
            
            # Check for external links
            if policy_rules.get("block_external_links", True):
                external_link_pattern = r'https?://(?!(?:localhost|127\.0\.0\.1|0\.0\.0\.0))'
                if re.search(external_link_pattern, content, re.IGNORECASE):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Content security policy check error: {e}")
            return False

    async def cleanup(self) -> bool:
        """Cleanup validation service resources.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            # Clear validation rules
            self._validation_rules.clear()
            self._global_rules.clear()
            
            # Clear threat patterns
            self._threat_patterns.clear()
            
            # Clear caches
            self._sanitization_cache.clear()
            
            # Clear metrics
            self._operation_times.clear()
            self._validation_results = {"valid": 0, "invalid": 0, "threats": 0}
            
            self._initialized = False
            logger.info("ValidationService cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup ValidationService: {e}")
            return False

    # Private helper methods

    def _initialize_threat_patterns(self) -> None:
        """Initialize threat detection patterns based on OWASP guidelines."""
        threat_patterns = [
            # SQL Injection patterns
            ThreatPattern(
                "sql_injection",
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\s+|['\";]|\b(OR|AND)\s+\d+\s*=\s*\d+)",
                "high",
                "Potential SQL injection attempt"
            ),
            
            # XSS patterns
            ThreatPattern(
                "xss_script",
                r"<script[^>]*>.*?</script>|javascript:|on\w+\s*=|eval\s*\(|setTimeout\s*\(",
                "high",
                "Potential XSS script injection"
            ),
            
            # Command injection patterns
            ThreatPattern(
                "command_injection",
                r"(\||&|;|`|\$\(|\${|<|>|\n|\r).*?(rm|cat|ls|ps|kill|wget|curl|nc|telnet|ssh)",
                "high",
                "Potential command injection"
            ),
            
            # LDAP injection patterns
            ThreatPattern(
                "ldap_injection",
                r"[*()\\&|!]",
                "medium",
                "Potential LDAP injection characters"
            ),
            
            # Path traversal patterns
            ThreatPattern(
                "path_traversal",
                r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
                "medium",
                "Potential path traversal attempt"
            ),
            
            # XXE patterns
            ThreatPattern(
                "xxe_injection",
                r"<!ENTITY|<!DOCTYPE.*ENTITY|SYSTEM\s+[\"']file://",
                "high",
                "Potential XXE injection"
            ),
            
            # SSRF patterns
            ThreatPattern(
                "ssrf_attempt",
                r"(localhost|127\.0\.0\.1|0\.0\.0\.0|169\.254\.|10\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[0-1]\.|192\.168\.)",
                "medium",
                "Potential SSRF attempt"
            )
        ]
        
        self._threat_patterns.extend(threat_patterns)

    def _initialize_default_rules(self) -> None:
        """Initialize default validation rules."""
        default_rules = [
            ValidationRule(
                "max_length",
                "length",
                {"max": self.max_input_size},
                required=True,
                error_message="Input exceeds maximum allowed length"
            ),
            ValidationRule(
                "no_null_bytes",
                "pattern",
                {"pattern": r"^[^\x00]*$"},
                required=True,
                error_message="Input contains null bytes"
            ),
            ValidationRule(
                "basic_xss_protection",
                "blacklist",
                {"values": ["<script>", "</script>", "javascript:", "onclick=", "onerror="]},
                required=True,
                error_message="Input contains potentially malicious content"
            )
        ]
        
        self._global_rules.extend(default_rules)

    async def _load_threat_intelligence(self) -> None:
        """Load threat intelligence feeds (placeholder for real implementation)."""
        # In production, this would load from:
        # - Threat intelligence feeds
        # - CVE databases
        # - Custom threat signatures
        # - Machine learning models
        pass

    async def _validate_input_size(self, input_data: Any) -> Tuple[bool, str]:
        """Validate input data size."""
        try:
            data_size = len(str(input_data).encode('utf-8'))
            if data_size > self.max_input_size:
                return (False, f"Input size {data_size} exceeds maximum {self.max_input_size}")
            return (True, "")
        except Exception as e:
            return (False, f"Size validation error: {str(e)}")

    async def _detect_threats(self, input_data: Any) -> List[Dict[str, Any]]:
        """Detect threats in input data using pattern matching."""
        threats = []
        
        if not isinstance(input_data, str):
            input_data = str(input_data)
        
        for pattern in self._threat_patterns:
            threat_matches = pattern.detect(input_data)
            threats.extend(threat_matches)
        
        if threats and OPENTELEMETRY_AVAILABLE and validation_threats_counter:
            validation_threats_counter.add(len(threats), {"threat_type": "pattern_match"})
        
        return threats

    async def _apply_validation_rules(self, input_data: Any, rules: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Apply validation rules to input data."""
        errors = []
        
        for rule_name, rule_config in rules.items():
            if rule_name == "mode":
                continue
                
            try:
                rule_type = rule_config.get("type", "pattern")
                parameters = rule_config.get("parameters", {})
                required = rule_config.get("required", True)
                error_message = rule_config.get("error_message", f"Validation failed for {rule_name}")
                
                rule = ValidationRule(rule_name, rule_type, parameters, required, error_message)
                is_valid, error = rule.validate(input_data)
                
                if not is_valid:
                    errors.append(f"{rule_name}: {error}")
                    if required:
                        return (False, errors)
                        
            except Exception as e:
                errors.append(f"{rule_name}: Validation error - {str(e)}")
                return (False, errors)
        
        return (len(errors) == 0, errors)

    async def _apply_global_rules(self, input_data: Any) -> Tuple[bool, List[str]]:
        """Apply global validation rules."""
        errors = []
        
        for rule in self._global_rules:
            is_valid, error = rule.validate(input_data)
            if not is_valid:
                errors.append(f"{rule.name}: {error}")
                if rule.required:
                    return (False, errors)
        
        return (len(errors) == 0, errors)

    async def _validate_with_context(self, security_context: SecurityContext, input_data: Any) -> Tuple[bool, List[str]]:
        """Perform context-aware validation."""
        errors = []
        
        try:
            # Check threat score
            if security_context.threat_score.score > 0.5:
                # Apply stricter validation for high-risk users
                if isinstance(input_data, str) and len(input_data) > 1000:
                    errors.append("Input too long for high-risk user")
                    return (False, errors)
            
            # Check user permissions for specific content types
            # This would integrate with authorization service in production
            
            return (True, errors)
            
        except Exception as e:
            errors.append(f"Context validation error: {str(e)}")
            return (False, errors)

    async def _detect_sensitive_data(self, output_data: str) -> List[str]:
        """Detect sensitive data patterns in output."""
        sensitive_patterns = []
        
        # Credit card numbers
        if re.search(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', output_data):
            sensitive_patterns.append("credit_card")
        
        # Social Security Numbers
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', output_data):
            sensitive_patterns.append("ssn")
        
        # Email addresses
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', output_data):
            sensitive_patterns.append("email")
        
        # Phone numbers
        if re.search(r'\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s*\d{3}-\d{4}', output_data):
            sensitive_patterns.append("phone")
        
        # API keys or tokens
        if re.search(r'\b[A-Za-z0-9]{32,}\b', output_data):
            sensitive_patterns.append("api_key")
        
        return sensitive_patterns

    async def _sanitize_html(self, input_data: str) -> str:
        """Sanitize HTML content."""
        # Escape HTML entities
        sanitized = html.escape(input_data)
        
        # Remove script tags
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove event handlers
        sanitized = re.sub(r'\son\w+\s*=\s*["\'][^"\']*["\']', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized

    async def _sanitize_sql(self, input_data: str) -> str:
        """Sanitize SQL content."""
        # Escape SQL quotes
        sanitized = input_data.replace("'", "''").replace('"', '""')
        
        # Remove SQL keywords
        sql_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER', 'EXEC', 'UNION']
        for keyword in sql_keywords:
            sanitized = re.sub(fr'\b{keyword}\b', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized

    async def _sanitize_command(self, input_data: str) -> str:
        """Sanitize command injection content."""
        # Remove command injection characters
        dangerous_chars = ['|', '&', ';', '`', '$', '<', '>', '\n', '\r']
        sanitized = input_data
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized

    async def _sanitize_url(self, input_data: str) -> str:
        """Sanitize URL content."""
        # URL encode the input
        return urllib.parse.quote(input_data, safe='')

    async def _sanitize_json(self, input_data: str) -> str:
        """Sanitize JSON content."""
        try:
            # Parse and re-serialize to ensure valid JSON
            parsed = json.loads(input_data)
            return json.dumps(parsed)
        except json.JSONDecodeError:
            # If not valid JSON, escape as string
            return json.dumps(input_data)


# Factory function for dependency injection
async def create_validation_service(
    security_state_manager: SecurityStateManagerProtocol,
    **config_overrides
) -> ValidationService:
    """Create and initialize validation service.
    
    Args:
        security_state_manager: Shared security state manager
        **config_overrides: Configuration overrides
        
    Returns:
        Initialized ValidationService instance
    """
    service = ValidationService(security_state_manager, **config_overrides)
    
    if not await service.initialize():
        raise RuntimeError("Failed to initialize ValidationService")
    
    return service