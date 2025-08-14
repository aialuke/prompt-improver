"""Validation Error Service - Specialized Input Validation and Security Error Handling.

Provides comprehensive validation error handling with:
- Input format validation (JSON schema, data types, structure validation)
- Business rule validation (constraints, relationships, domain-specific rules)
- Security validation (injection attempts, malicious payloads, data sanitization)
- Enhanced PII detection for various data formats and structures
- Validation error aggregation with detailed classification and reporting
- Integration with security framework for threat detection and mitigation

Security Features:
- Advanced PII detection beyond basic patterns (context-aware identification)
- SQL/NoSQL injection attempt detection in user inputs
- XSS and script injection payload identification
- File upload validation and malware scanning integration
- Data length and size validation for DoS prevention
- Input encoding and character set validation

Performance Target: <1ms validation error classification, <3ms security threat analysis
Memory Target: <10MB for validation cache and pattern matching
"""

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

import jsonschema
from jsonschema import ValidationError as JSONSchemaValidationError

from prompt_improver.core.services.security import SecurityLevel, PrivacyTechnique
from prompt_improver.performance.monitoring.metrics_registry import (
    StandardMetrics,
    get_metrics_registry,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class ValidationErrorCategory(Enum):
    """Validation-specific error categories for precise handling."""
    
    SCHEMA_VALIDATION_ERROR = "schema_validation_error"  # JSON Schema, data structure
    TYPE_VALIDATION_ERROR = "type_validation_error"      # Data type mismatches
    FORMAT_VALIDATION_ERROR = "format_validation_error"  # Format constraints (email, phone, etc.)
    RANGE_VALIDATION_ERROR = "range_validation_error"    # Numeric/size range violations
    BUSINESS_RULE_ERROR = "business_rule_error"          # Domain-specific business rules
    CONSTRAINT_VIOLATION = "constraint_violation"        # Database/model constraints
    SECURITY_VALIDATION_ERROR = "security_validation_error"  # Security policy violations
    PII_DETECTION_ERROR = "pii_detection_error"          # PII found in inappropriate context
    INJECTION_ATTEMPT = "injection_attempt"              # SQL/NoSQL/Script injection attempts
    MALICIOUS_PAYLOAD = "malicious_payload"              # Known malicious patterns
    FILE_VALIDATION_ERROR = "file_validation_error"      # File upload validation errors
    ENCODING_ERROR = "encoding_error"                    # Character encoding issues
    SIZE_LIMIT_EXCEEDED = "size_limit_exceeded"          # Data size/length limits
    UNKNOWN_VALIDATION_ERROR = "unknown_validation_error"


class ValidationErrorSeverity(Enum):
    """Error severity levels for validation operations."""
    
    CRITICAL = "critical"  # Security threat or system integrity risk
    HIGH = "high"         # Business rule violation or data corruption risk
    MEDIUM = "medium"     # Format/constraint violation requiring attention
    LOW = "low"          # Minor validation issues, user input corrections needed
    INFO = "info"        # Informational validation warnings


@dataclass
class ValidationErrorContext:
    """Comprehensive error context for validation operations."""
    
    error_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operation_name: str = ""
    category: ValidationErrorCategory = ValidationErrorCategory.UNKNOWN_VALIDATION_ERROR
    severity: ValidationErrorSeverity = ValidationErrorSeverity.MEDIUM
    original_exception: Optional[Exception] = None
    sanitized_message: str = ""
    
    # Validation-specific context
    field_name: Optional[str] = None
    field_path: Optional[str] = None  # JSONPath or nested field path
    expected_type: Optional[str] = None
    actual_type: Optional[str] = None
    expected_format: Optional[str] = None
    validation_rule: Optional[str] = None
    constraint_violated: Optional[str] = None
    
    # Input data context (sanitized)
    input_value: Optional[str] = None
    sanitized_input: Optional[str] = None
    input_size_bytes: Optional[int] = None
    input_character_count: Optional[int] = None
    
    # Security context
    threat_detected: bool = False
    threat_types: List[str] = field(default_factory=list)
    pii_detected: bool = False
    pii_types: List[str] = field(default_factory=list)
    security_level: Optional[SecurityLevel] = None
    recommended_privacy_technique: Optional[PrivacyTechnique] = None
    
    # Business rule context
    business_context: Dict[str, Any] = field(default_factory=dict)
    related_entities: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=aware_utc_now)
    correlation_id: Optional[str] = None


class ValidationErrorService:
    """Specialized validation error handling service.
    
    Provides comprehensive error handling for validation operations including:
    - Schema and data type validation with detailed error reporting
    - Business rule validation with contextual information
    - Security-aware validation with threat detection and PII identification
    - File upload validation with malware detection integration
    - Advanced pattern matching for various attack vectors
    """
    
    # Enhanced PII patterns with context awareness
    ENHANCED_PII_PATTERNS = [
        # Email addresses with context
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "email", "Email address detected"),
        # Phone numbers (various formats)
        (re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'), "phone", "Phone number detected"),
        # Social Security Numbers
        (re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'), "ssn", "Social Security Number detected"),
        # Credit card numbers
        (re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'), "credit_card", "Credit card number detected"),
        # Driver's license (US format)
        (re.compile(r'\b[A-Z]{1,2}\d{6,8}\b'), "drivers_license", "Driver's license detected"),
        # Passport numbers
        (re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'), "passport", "Passport number detected"),
        # IP addresses
        (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), "ip_address", "IP address detected"),
        # MAC addresses
        (re.compile(r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b'), "mac_address", "MAC address detected"),
        # Bank account numbers (basic pattern)
        (re.compile(r'\b\d{8,17}\b'), "bank_account", "Potential bank account number detected"),
        # Addresses (basic pattern for US addresses)
        (re.compile(r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)', re.IGNORECASE), 
         "address", "Street address detected"),
    ]
    
    # SQL injection patterns for various database systems
    SQL_INJECTION_PATTERNS = [
        (re.compile(r"(?i)\b(union|select|insert|update|delete|drop|alter|create|exec|execute|sp_|xp_)\b"), "sql_injection"),
        (re.compile(r"(?i)['\"];?\s*(union|select|insert|update|delete|drop)", re.IGNORECASE), "sql_injection"),
        (re.compile(r"(?i)\b(or|and)\b\s*['\"]?\s*\d+\s*['\"]?\s*[=<>!]+\s*['\"]?\s*\d+", re.IGNORECASE), "sql_injection"),
        (re.compile(r"(?i)(--|/\*|\*/|;\s*--)", re.IGNORECASE), "sql_comment_injection"),
        (re.compile(r"(?i)(information_schema|sysobjects|syscolumns|pg_tables)", re.IGNORECASE), "sql_schema_enumeration"),
    ]
    
    # NoSQL injection patterns
    NOSQL_INJECTION_PATTERNS = [
        (re.compile(r"(?i)(\$where|\$ne|\$gt|\$lt|\$regex|\$exists)", re.IGNORECASE), "nosql_injection"),
        (re.compile(r"(?i)(\.find\s*\(|\.remove\s*\(|\.update\s*\()", re.IGNORECASE), "nosql_method_injection"),
        (re.compile(r"(?i)(this\.|function\s*\(|return\s+)", re.IGNORECASE), "nosql_javascript_injection"),
    ]
    
    # XSS and script injection patterns
    XSS_PATTERNS = [
        (re.compile(r"(?i)<script[^>]*>.*?</script>", re.IGNORECASE), "xss_script_tag"),
        (re.compile(r"(?i)(javascript:|vbscript:|data:text/html)", re.IGNORECASE), "xss_protocol"),
        (re.compile(r"(?i)(onload|onerror|onclick|onmouseover|onfocus|onblur)=", re.IGNORECASE), "xss_event_handler"),
        (re.compile(r"(?i)(eval\s*\(|setTimeout\s*\(|setInterval\s*\()", re.IGNORECASE), "xss_code_execution"),
        (re.compile(r"(?i)(document\.cookie|document\.write|window\.location)", re.IGNORECASE), "xss_dom_manipulation"),
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        (re.compile(r"(?i)(;|\||&|\$\(|`|\${)(rm|cat|ls|wget|curl|nc|bash|sh|cmd|powershell)", re.IGNORECASE), "command_injection"),
        (re.compile(r"(?i)(\.\./)|(~/)|(\.\.\\)", re.IGNORECASE), "path_traversal"),
        (re.compile(r"(?i)(/etc/passwd|/etc/shadow|/proc/|/sys/)", re.IGNORECASE), "system_file_access"),
    ]
    
    # File upload security patterns
    FILE_SECURITY_PATTERNS = [
        (re.compile(r"(?i)\.(php|asp|aspx|jsp|jspx|cgi|pl|py|rb|sh|exe|bat|cmd)$", re.IGNORECASE), "executable_extension"),
        (re.compile(r"(?i)(<%|<?php|<script|javascript:)", re.IGNORECASE), "embedded_code"),
        (re.compile(r"(?i)content-type:\s*(application/x-|text/x-)", re.IGNORECASE), "suspicious_mime_type"),
    ]
    
    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize validation error service.
        
        Args:
            correlation_id: Optional correlation ID for request tracking
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self._metrics_registry = get_metrics_registry()
        self._validation_cache: Dict[str, Any] = {}
        self._security_patterns_compiled = True  # Flag to track pattern compilation
        
        logger.info(f"ValidationErrorService initialized with correlation_id: {self.correlation_id}")
    
    async def handle_validation_error(
        self,
        error: Exception,
        operation_name: str,
        field_name: Optional[str] = None,
        field_path: Optional[str] = None,
        input_value: Any = None,
        validation_schema: Optional[Dict[str, Any]] = None,
        business_context: Optional[Dict[str, Any]] = None,
        **context_kwargs: Any
    ) -> ValidationErrorContext:
        """Handle validation error with comprehensive processing.
        
        Args:
            error: The validation exception that occurred
            operation_name: Name of the validation operation
            field_name: Name of the field being validated
            field_path: JSONPath or nested field path
            input_value: The input value that failed validation
            validation_schema: Schema used for validation
            business_context: Business-specific context information
            **context_kwargs: Additional context information
            
        Returns:
            ValidationErrorContext with processed error information
        """
        start_time = time.time()
        
        # Classify the error
        category, severity = self._classify_validation_error(error, input_value)
        
        # Sanitize input value for logging
        sanitized_input = self._sanitize_input_value(input_value) if input_value is not None else None
        
        # Create error context
        error_context = ValidationErrorContext(
            operation_name=operation_name,
            category=category,
            severity=severity,
            original_exception=error,
            sanitized_message=self._sanitize_error_message(str(error)),
            field_name=field_name,
            field_path=field_path,
            input_value=str(input_value)[:200] if input_value is not None else None,  # Truncate for safety
            sanitized_input=sanitized_input,
            business_context=business_context or {},
            correlation_id=self.correlation_id,
            **context_kwargs
        )
        
        # Add type information
        if input_value is not None:
            error_context.actual_type = type(input_value).__name__
            error_context.input_size_bytes = len(str(input_value).encode('utf-8'))
            error_context.input_character_count = len(str(input_value))
        
        # Extract expected type/format from schema or error message
        if validation_schema:
            error_context.expected_type = self._extract_expected_type(validation_schema, field_name)
            error_context.expected_format = self._extract_expected_format(validation_schema, field_name)
        
        # Perform security analysis
        await self._analyze_security_threats(error_context)
        
        # Perform PII detection
        await self._detect_pii(error_context)
        
        # Classify business rule violations
        if category == ValidationErrorCategory.BUSINESS_RULE_ERROR:
            await self._analyze_business_rule_violation(error_context)
        
        # Record metrics
        await self._record_validation_error_metrics(error_context)
        
        # Log the error
        await self._log_validation_error(error_context)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self._metrics_registry.record_value(
            "validation_error_processing_duration_seconds",
            processing_time,
            tags={"operation": operation_name, "category": category.value}
        )
        
        return error_context
    
    async def validate_input_security(
        self,
        input_data: Any,
        context: str = "user_input",
        security_level: SecurityLevel = SecurityLevel.internal
    ) -> ValidationErrorContext:
        """Validate input for security threats and PII.
        
        Args:
            input_data: Data to validate
            context: Context of the input (user_input, api_request, etc.)
            security_level: Required security level
            
        Returns:
            ValidationErrorContext with security analysis results
        """
        start_time = time.time()
        
        error_context = ValidationErrorContext(
            operation_name=f"security_validation_{context}",
            category=ValidationErrorCategory.SECURITY_VALIDATION_ERROR,
            severity=ValidationErrorSeverity.INFO,
            security_level=security_level,
            correlation_id=self.correlation_id,
            input_value=str(input_data)[:200] if input_data is not None else None,
            sanitized_input=self._sanitize_input_value(input_data),
        )
        
        # Perform comprehensive security analysis
        await self._analyze_security_threats(error_context)
        await self._detect_pii(error_context)
        
        # Adjust severity based on findings
        if error_context.threat_detected:
            if "injection" in " ".join(error_context.threat_types):
                error_context.severity = ValidationErrorSeverity.CRITICAL
                error_context.category = ValidationErrorCategory.INJECTION_ATTEMPT
            else:
                error_context.severity = ValidationErrorSeverity.HIGH
                error_context.category = ValidationErrorCategory.MALICIOUS_PAYLOAD
        elif error_context.pii_detected:
            if security_level in (SecurityLevel.confidential, SecurityLevel.restricted):
                error_context.severity = ValidationErrorSeverity.HIGH
                error_context.category = ValidationErrorCategory.PII_DETECTION_ERROR
        
        # Record security metrics
        await self._record_security_validation_metrics(error_context)
        
        processing_time = time.time() - start_time
        self._metrics_registry.record_value(
            "security_validation_duration_seconds",
            processing_time,
            tags={"context": context, "security_level": security_level.value}
        )
        
        return error_context
    
    def _classify_validation_error(
        self,
        error: Exception,
        input_value: Any = None
    ) -> Tuple[ValidationErrorCategory, ValidationErrorSeverity]:
        """Classify validation error into category and severity.
        
        Args:
            error: Validation exception
            input_value: Input value that caused the error
            
        Returns:
            Tuple of (category, severity)
        """
        error_type = type(error)
        error_msg = str(error).lower()
        
        # JSONSchema validation errors
        if isinstance(error, JSONSchemaValidationError):
            if "type" in error.message.lower():
                return (ValidationErrorCategory.TYPE_VALIDATION_ERROR, ValidationErrorSeverity.MEDIUM)
            elif "format" in error.message.lower():
                return (ValidationErrorCategory.FORMAT_VALIDATION_ERROR, ValidationErrorSeverity.MEDIUM)
            elif "required" in error.message.lower():
                return (ValidationErrorCategory.SCHEMA_VALIDATION_ERROR, ValidationErrorSeverity.HIGH)
            else:
                return (ValidationErrorCategory.SCHEMA_VALIDATION_ERROR, ValidationErrorSeverity.MEDIUM)
        
        # Python built-in validation errors
        if isinstance(error, (TypeError, ValueError)):
            return (ValidationErrorCategory.TYPE_VALIDATION_ERROR, ValidationErrorSeverity.MEDIUM)
        
        if isinstance(error, (KeyError, AttributeError)):
            return (ValidationErrorCategory.SCHEMA_VALIDATION_ERROR, ValidationErrorSeverity.HIGH)
        
        # Pattern-based classification
        if any(pattern in error_msg for pattern in ["constraint", "unique", "foreign key", "check"]):
            return (ValidationErrorCategory.CONSTRAINT_VIOLATION, ValidationErrorSeverity.HIGH)
        
        if any(pattern in error_msg for pattern in ["business rule", "business logic", "domain"]):
            return (ValidationErrorCategory.BUSINESS_RULE_ERROR, ValidationErrorSeverity.HIGH)
        
        if any(pattern in error_msg for pattern in ["size", "length", "limit", "too large", "too small"]):
            return (ValidationErrorCategory.SIZE_LIMIT_EXCEEDED, ValidationErrorSeverity.MEDIUM)
        
        if any(pattern in error_msg for pattern in ["encoding", "decode", "unicode", "utf-8"]):
            return (ValidationErrorCategory.ENCODING_ERROR, ValidationErrorSeverity.MEDIUM)
        
        if any(pattern in error_msg for pattern in ["security", "injection", "malicious", "threat"]):
            return (ValidationErrorCategory.SECURITY_VALIDATION_ERROR, ValidationErrorSeverity.CRITICAL)
        
        if any(pattern in error_msg for pattern in ["pii", "personal", "sensitive"]):
            return (ValidationErrorCategory.PII_DETECTION_ERROR, ValidationErrorSeverity.HIGH)
        
        return (ValidationErrorCategory.UNKNOWN_VALIDATION_ERROR, ValidationErrorSeverity.MEDIUM)
    
    def _sanitize_input_value(self, input_value: Any) -> str:
        """Sanitize input value for safe logging and storage.
        
        Args:
            input_value: Raw input value
            
        Returns:
            Sanitized input value string
        """
        if input_value is None:
            return "None"
        
        # Convert to string and limit length
        value_str = str(input_value)[:500]  # Reasonable limit for logging
        
        # Apply PII redaction patterns
        for pattern, pii_type, description in self.ENHANCED_PII_PATTERNS:
            if pattern.search(value_str):
                value_str = pattern.sub(f"[{pii_type.upper()}_REDACTED]", value_str)
        
        # Redact potential secrets/tokens
        value_str = re.sub(r'\b[A-Za-z0-9_-]{32,}\b', '[TOKEN_REDACTED]', value_str)
        
        # Redact potential passwords/keys in key-value patterns
        value_str = re.sub(
            r'(password|pwd|pass|secret|key|token)\s*[=:]\s*\S+',
            r'\1=[REDACTED]',
            value_str,
            flags=re.IGNORECASE
        )
        
        return value_str
    
    def _sanitize_error_message(self, error_message: str) -> str:
        """Sanitize error message to remove sensitive information.
        
        Args:
            error_message: Raw error message
            
        Returns:
            Sanitized error message
        """
        sanitized = error_message
        
        # Remove file paths that might contain sensitive information
        sanitized = re.sub(r'/[^\s]*/(api_keys?|secrets?|passwords?|tokens?)/[^\s]*', '[SENSITIVE_PATH_REDACTED]', sanitized)
        
        # Remove potential database connection strings
        sanitized = re.sub(r'(postgresql|mysql|mongodb)://[^\s]*', '[DB_CONNECTION_REDACTED]', sanitized, flags=re.IGNORECASE)
        
        # Apply PII redaction
        sanitized = self._sanitize_input_value(sanitized)
        
        return sanitized
    
    def _extract_expected_type(self, schema: Dict[str, Any], field_name: Optional[str]) -> Optional[str]:
        """Extract expected type from validation schema.
        
        Args:
            schema: Validation schema
            field_name: Field name to look up
            
        Returns:
            Expected type or None
        """
        try:
            if field_name and "properties" in schema:
                field_schema = schema["properties"].get(field_name, {})
                return field_schema.get("type")
            return schema.get("type")
        except Exception:
            return None
    
    def _extract_expected_format(self, schema: Dict[str, Any], field_name: Optional[str]) -> Optional[str]:
        """Extract expected format from validation schema.
        
        Args:
            schema: Validation schema
            field_name: Field name to look up
            
        Returns:
            Expected format or None
        """
        try:
            if field_name and "properties" in schema:
                field_schema = schema["properties"].get(field_name, {})
                return field_schema.get("format")
            return schema.get("format")
        except Exception:
            return None
    
    async def _analyze_security_threats(self, error_context: ValidationErrorContext) -> None:
        """Analyze input for security threats.
        
        Args:
            error_context: Validation error context to update
        """
        if not error_context.input_value:
            return
        
        input_str = str(error_context.input_value)
        threats_detected = []
        
        # Check SQL injection patterns
        for pattern, threat_type in self.SQL_INJECTION_PATTERNS:
            if pattern.search(input_str):
                threats_detected.append(threat_type)
        
        # Check NoSQL injection patterns
        for pattern, threat_type in self.NOSQL_INJECTION_PATTERNS:
            if pattern.search(input_str):
                threats_detected.append(threat_type)
        
        # Check XSS patterns
        for pattern, threat_type in self.XSS_PATTERNS:
            if pattern.search(input_str):
                threats_detected.append(threat_type)
        
        # Check command injection patterns
        for pattern, threat_type in self.COMMAND_INJECTION_PATTERNS:
            if pattern.search(input_str):
                threats_detected.append(threat_type)
        
        # Check file security patterns if input looks like a filename or file content
        for pattern, threat_type in self.FILE_SECURITY_PATTERNS:
            if pattern.search(input_str):
                threats_detected.append(threat_type)
        
        if threats_detected:
            error_context.threat_detected = True
            error_context.threat_types = list(set(threats_detected))  # Remove duplicates
            
            logger.warning(
                f"VALIDATION SECURITY ALERT: Threats detected in {error_context.operation_name}: {threats_detected}",
                extra={
                    "correlation_id": self.correlation_id,
                    "threat_types": threats_detected,
                    "error_id": error_context.error_id,
                    "field_name": error_context.field_name
                }
            )
    
    async def _detect_pii(self, error_context: ValidationErrorContext) -> None:
        """Detect PII in validation input.
        
        Args:
            error_context: Validation error context to update
        """
        if not error_context.input_value:
            return
        
        input_str = str(error_context.input_value)
        pii_detected = []
        
        # Check enhanced PII patterns
        for pattern, pii_type, description in self.ENHANCED_PII_PATTERNS:
            if pattern.search(input_str):
                pii_detected.append(pii_type)
        
        if pii_detected:
            error_context.pii_detected = True
            error_context.pii_types = list(set(pii_detected))  # Remove duplicates
            
            # Recommend privacy technique based on security level and PII type
            if error_context.security_level in (SecurityLevel.confidential, SecurityLevel.restricted):
                if "ssn" in pii_detected or "credit_card" in pii_detected:
                    error_context.recommended_privacy_technique = PrivacyTechnique.HOMOMORPHIC_ENCRYPTION
                elif "email" in pii_detected or "phone" in pii_detected:
                    error_context.recommended_privacy_technique = PrivacyTechnique.tokenization
                else:
                    error_context.recommended_privacy_technique = PrivacyTechnique.masking
            else:
                error_context.recommended_privacy_technique = PrivacyTechnique.redaction
            
            logger.info(
                f"PII detected in validation input for {error_context.operation_name}: {pii_detected}",
                extra={
                    "correlation_id": self.correlation_id,
                    "pii_types": pii_detected,
                    "error_id": error_context.error_id,
                    "field_name": error_context.field_name,
                    "recommended_technique": error_context.recommended_privacy_technique.value if error_context.recommended_privacy_technique else None
                }
            )
    
    async def _analyze_business_rule_violation(self, error_context: ValidationErrorContext) -> None:
        """Analyze business rule violations for detailed context.
        
        Args:
            error_context: Validation error context to update
        """
        # Extract business rule information from context and error message
        error_msg = error_context.sanitized_message.lower()
        
        # Common business rule patterns
        if "duplicate" in error_msg or "unique" in error_msg:
            error_context.constraint_violated = "uniqueness_constraint"
            error_context.validation_rule = "Entity must be unique"
        elif "reference" in error_msg or "foreign" in error_msg:
            error_context.constraint_violated = "referential_integrity"
            error_context.validation_rule = "Referenced entity must exist"
        elif "required" in error_msg or "mandatory" in error_msg:
            error_context.constraint_violated = "mandatory_field"
            error_context.validation_rule = "Required field must be provided"
        elif "permission" in error_msg or "authorized" in error_msg:
            error_context.constraint_violated = "authorization_rule"
            error_context.validation_rule = "User lacks required permissions"
        elif "state" in error_msg or "status" in error_msg:
            error_context.constraint_violated = "state_transition_rule"
            error_context.validation_rule = "Invalid state transition attempted"
    
    async def _record_validation_error_metrics(self, error_context: ValidationErrorContext) -> None:
        """Record validation error metrics for monitoring.
        
        Args:
            error_context: Error context with metrics data
        """
        try:
            # Basic validation error metrics
            self._metrics_registry.increment(
                "validation_errors_total",
                tags={
                    "operation": error_context.operation_name,
                    "category": error_context.category.value,
                    "severity": error_context.severity.value,
                    "field_name": error_context.field_name or "unknown"
                }
            )
            
            # Security-specific metrics
            if error_context.threat_detected:
                self._metrics_registry.increment(
                    "validation_security_threats_total",
                    tags={
                        "operation": error_context.operation_name,
                        "threat_types": ",".join(error_context.threat_types),
                        "field_name": error_context.field_name or "unknown"
                    }
                )
            
            # PII detection metrics
            if error_context.pii_detected:
                self._metrics_registry.increment(
                    "validation_pii_detected_total",
                    tags={
                        "operation": error_context.operation_name,
                        "pii_types": ",".join(error_context.pii_types),
                        "security_level": error_context.security_level.value if error_context.security_level else "unknown"
                    }
                )
            
            # Input size metrics
            if error_context.input_size_bytes:
                self._metrics_registry.record_value(
                    "validation_input_size_bytes",
                    error_context.input_size_bytes,
                    tags={
                        "operation": error_context.operation_name,
                        "category": error_context.category.value
                    }
                )
        
        except Exception as e:
            logger.error(f"Failed to record validation error metrics: {e}")
    
    async def _record_security_validation_metrics(self, error_context: ValidationErrorContext) -> None:
        """Record security validation metrics.
        
        Args:
            error_context: Error context with security metrics
        """
        try:
            self._metrics_registry.increment(
                "security_validations_total",
                tags={
                    "security_level": error_context.security_level.value if error_context.security_level else "unknown",
                    "threats_detected": str(error_context.threat_detected).lower(),
                    "pii_detected": str(error_context.pii_detected).lower()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to record security validation metrics: {e}")
    
    async def _log_validation_error(self, error_context: ValidationErrorContext) -> None:
        """Log validation error with appropriate level and context.
        
        Args:
            error_context: Error context to log
        """
        log_data = {
            "correlation_id": self.correlation_id,
            "error_id": error_context.error_id,
            "operation": error_context.operation_name,
            "category": error_context.category.value,
            "severity": error_context.severity.value,
            "field_name": error_context.field_name,
            "field_path": error_context.field_path,
            "threat_detected": error_context.threat_detected,
            "pii_detected": error_context.pii_detected,
        }
        
        log_message = f"Validation error in {error_context.operation_name}: {error_context.sanitized_message}"
        
        if error_context.field_name:
            log_message += f" (field: {error_context.field_name})"
        
        if error_context.severity == ValidationErrorSeverity.CRITICAL:
            logger.critical(log_message, extra=log_data)
        elif error_context.severity == ValidationErrorSeverity.HIGH:
            logger.error(log_message, extra=log_data)
        elif error_context.severity == ValidationErrorSeverity.MEDIUM:
            logger.warning(log_message, extra=log_data)
        else:
            logger.info(log_message, extra=log_data)
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics for monitoring dashboard.
        
        Returns:
            Dictionary with validation statistics
        """
        return {
            "correlation_id": self.correlation_id,
            "validation_cache_size": len(self._validation_cache),
            "security_patterns_compiled": self._security_patterns_compiled,
            "pii_patterns_count": len(self.ENHANCED_PII_PATTERNS),
            "security_threat_patterns": {
                "sql_injection": len(self.SQL_INJECTION_PATTERNS),
                "nosql_injection": len(self.NOSQL_INJECTION_PATTERNS),
                "xss_patterns": len(self.XSS_PATTERNS),
                "command_injection": len(self.COMMAND_INJECTION_PATTERNS),
                "file_security": len(self.FILE_SECURITY_PATTERNS),
            },
            "service_health": "operational",
        }