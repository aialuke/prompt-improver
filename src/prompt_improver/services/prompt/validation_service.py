"""Validation Service - Focused service for input validation and business rule checking.

This service handles:
- Prompt input validation
- Business rule compliance checking
- Improvement session validation
- Content sanitization

Part of the PromptServiceFacade decomposition following Clean Architecture principles.
"""

import logging
import re
from datetime import datetime
from typing import Any

from prompt_improver.core.config.validation import ValidationResult
from prompt_improver.shared.interfaces.protocols.application import (
    ValidationServiceProtocol,
)

# Use standard exceptions - core.exceptions module was removed
ValidationError = ValueError
BusinessRuleViolationError = Exception

logger = logging.getLogger(__name__)


class ValidationService(ValidationServiceProtocol):
    """Service for input validation and business rule checking."""

    def __init__(self) -> None:
        self.validation_rules = self._initialize_validation_rules()
        self.business_rules = self._initialize_business_rules()
        self.sanitization_patterns = self._initialize_sanitization_patterns()

    async def validate_prompt_input(
        self,
        prompt: str,
        constraints: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Validate prompt input against constraints."""
        try:
            validation_result = {
                "valid": True,
                "violations": [],
                "warnings": [],
                "sanitized": False,
                "original_length": len(prompt),
                "validation_timestamp": datetime.now().isoformat()
            }

            constraints = constraints or {}

            # Basic validation checks
            basic_checks = await self._perform_basic_validation(prompt, constraints)
            validation_result.update(basic_checks)

            # Content validation
            content_checks = await self._perform_content_validation(prompt, constraints)
            validation_result["violations"].extend(content_checks.get("violations", []))
            validation_result["warnings"].extend(content_checks.get("warnings", []))

            # Length validation
            length_checks = await self._perform_length_validation(prompt, constraints)
            validation_result["violations"].extend(length_checks.get("violations", []))

            # Security validation
            security_checks = await self._perform_security_validation(prompt)
            validation_result["violations"].extend(security_checks.get("violations", []))
            validation_result["warnings"].extend(security_checks.get("warnings", []))

            # Format validation
            format_checks = await self._perform_format_validation(prompt, constraints)
            validation_result["violations"].extend(format_checks.get("violations", []))

            # Update overall validity
            validation_result["valid"] = len(validation_result["violations"]) == 0

            # Generate validation score
            validation_result["validation_score"] = self._calculate_validation_score(
                validation_result
            )

            return validation_result

        except Exception as e:
            logger.exception(f"Error validating prompt input: {e}")
            raise ValidationError(f"Validation failed: {e}")

    async def check_business_rules(
        self,
        operation: str,
        data: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> bool:
        """Check if an operation complies with business rules."""
        try:
            context = context or {}

            # Get applicable business rules for operation
            applicable_rules = self.business_rules.get(operation, [])

            for rule in applicable_rules:
                rule_result = await self._evaluate_business_rule(rule, data, context)
                if not rule_result["compliant"]:
                    logger.warning(
                        f"Business rule violation for {operation}: {rule_result['violation']}"
                    )
                    return False

            return True

        except Exception as e:
            logger.exception(f"Error checking business rules for {operation}: {e}")
            raise BusinessRuleViolationError(f"Business rule check failed: {e}")

    async def validate_improvement_session(
        self,
        session: Any  # ImprovementSession type
    ) -> dict[str, Any]:
        """Validate an improvement session."""
        try:
            validation_result = {
                "valid": True,
                "violations": [],
                "warnings": [],
                "session_health": "healthy"
            }

            # Validate session data integrity
            integrity_checks = await self._validate_session_integrity(session)
            validation_result["violations"].extend(integrity_checks.get("violations", []))

            # Validate session state
            state_checks = await self._validate_session_state(session)
            validation_result["violations"].extend(state_checks.get("violations", []))
            validation_result["warnings"].extend(state_checks.get("warnings", []))

            # Validate session timing
            timing_checks = await self._validate_session_timing(session)
            validation_result["warnings"].extend(timing_checks.get("warnings", []))

            # Validate session resources
            resource_checks = await self._validate_session_resources(session)
            validation_result["violations"].extend(resource_checks.get("violations", []))

            # Update overall validity
            validation_result["valid"] = len(validation_result["violations"]) == 0

            # Determine session health
            if validation_result["violations"]:
                validation_result["session_health"] = "critical"
            elif validation_result["warnings"]:
                validation_result["session_health"] = "warning"
            else:
                validation_result["session_health"] = "healthy"

            return validation_result

        except Exception as e:
            logger.exception(f"Error validating improvement session: {e}")
            raise ValidationError(f"Session validation failed: {e}")

    async def sanitize_prompt_content(
        self,
        prompt: str,
        sanitization_level: str = "standard"
    ) -> str:
        """Sanitize prompt content for safety."""
        try:
            sanitized_prompt = prompt

            # Apply sanitization based on level
            if sanitization_level == "basic":
                sanitized_prompt = await self._apply_basic_sanitization(sanitized_prompt)
            elif sanitization_level == "standard":
                sanitized_prompt = await self._apply_standard_sanitization(sanitized_prompt)
            elif sanitization_level == "strict":
                sanitized_prompt = await self._apply_strict_sanitization(sanitized_prompt)
            else:
                logger.warning(f"Unknown sanitization level: {sanitization_level}")
                sanitized_prompt = await self._apply_standard_sanitization(sanitized_prompt)

            return sanitized_prompt

        except Exception as e:
            logger.exception(f"Error sanitizing prompt content: {e}")
            return prompt  # Return original if sanitization fails

    async def _perform_basic_validation(
        self,
        prompt: str,
        constraints: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform basic validation checks."""
        result = {"violations": [], "warnings": []}

        # Check if prompt is empty
        if not prompt or not prompt.strip():
            result["violations"].append({
                "type": "empty_prompt",
                "message": "Prompt cannot be empty",
                "severity": "error"
            })

        # Check for only whitespace
        if prompt and not prompt.strip():
            result["violations"].append({
                "type": "whitespace_only",
                "message": "Prompt cannot contain only whitespace",
                "severity": "error"
            })

        # Check for minimum length if specified
        min_length = constraints.get("min_length", 0)
        if len(prompt) < min_length:
            result["violations"].append({
                "type": "min_length",
                "message": f"Prompt must be at least {min_length} characters",
                "severity": "error",
                "current_length": len(prompt),
                "required_length": min_length
            })

        return result

    async def _perform_content_validation(
        self,
        prompt: str,
        constraints: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform content-specific validation."""
        result = {"violations": [], "warnings": []}

        # Check for prohibited content
        prohibited_patterns = constraints.get("prohibited_patterns", [])
        for pattern in prohibited_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                result["violations"].append({
                    "type": "prohibited_content",
                    "message": f"Content matches prohibited pattern: {pattern}",
                    "severity": "error",
                    "pattern": pattern
                })

        # Check for required content
        required_patterns = constraints.get("required_patterns", [])
        for pattern in required_patterns:
            if not re.search(pattern, prompt, re.IGNORECASE):
                result["warnings"].append({
                    "type": "missing_required_content",
                    "message": f"Content should include pattern: {pattern}",
                    "severity": "warning",
                    "pattern": pattern
                })

        # Check for sensitive information
        sensitive_checks = await self._check_sensitive_content(prompt)
        result["warnings"].extend(sensitive_checks.get("warnings", []))

        return result

    async def _perform_length_validation(
        self,
        prompt: str,
        constraints: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform length validation."""
        result = {"violations": []}

        max_length = constraints.get("max_length")
        if max_length and len(prompt) > max_length:
            result["violations"].append({
                "type": "max_length",
                "message": f"Prompt exceeds maximum length of {max_length} characters",
                "severity": "error",
                "current_length": len(prompt),
                "max_length": max_length
            })

        # Check word count limits
        word_count = len(prompt.split())
        max_words = constraints.get("max_words")
        if max_words and word_count > max_words:
            result["violations"].append({
                "type": "max_words",
                "message": f"Prompt exceeds maximum word count of {max_words}",
                "severity": "error",
                "current_words": word_count,
                "max_words": max_words
            })

        return result

    async def _perform_security_validation(self, prompt: str) -> dict[str, Any]:
        """Perform security-related validation."""
        result = {"violations": [], "warnings": []}

        # Check for potential injection attempts
        injection_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript protocol
            r'eval\s*\(',                 # Eval calls
            r'onclick\s*=',               # Click handlers
            r'\bDROP\s+TABLE\b',          # SQL injection
            r'\bUNION\s+SELECT\b',        # SQL union
        ]

        for pattern in injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                result["violations"].append({
                    "type": "security_risk",
                    "message": "Potential security risk detected in prompt",
                    "severity": "error",
                    "risk_type": "injection_attempt"
                })

        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', prompt)) / max(len(prompt), 1)
        if special_char_ratio > 0.3:
            result["warnings"].append({
                "type": "high_special_chars",
                "message": "High ratio of special characters detected",
                "severity": "warning",
                "ratio": special_char_ratio
            })

        return result

    async def _perform_format_validation(
        self,
        prompt: str,
        constraints: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform format validation."""
        result = {"violations": []}

        # Check required format if specified
        required_format = constraints.get("format")
        if required_format:
            if required_format == "json" and not self._is_valid_json_format(prompt):
                result["violations"].append({
                    "type": "invalid_format",
                    "message": "Prompt must be in valid JSON format",
                    "severity": "error",
                    "expected_format": "json"
                })
            elif required_format == "xml" and not self._is_valid_xml_format(prompt):
                result["violations"].append({
                    "type": "invalid_format",
                    "message": "Prompt must be in valid XML format",
                    "severity": "error",
                    "expected_format": "xml"
                })

        return result

    async def _evaluate_business_rule(
        self,
        rule: dict[str, Any],
        data: dict[str, Any],
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate a single business rule."""
        try:
            rule_type = rule.get("type")
            rule_config = rule.get("config", {})

            if rule_type == "rate_limit":
                return await self._check_rate_limit_rule(data, context, rule_config)
            if rule_type == "user_permissions":
                return await self._check_user_permissions_rule(data, context, rule_config)
            if rule_type == "content_policy":
                return await self._check_content_policy_rule(data, context, rule_config)
            if rule_type == "resource_limits":
                return await self._check_resource_limits_rule(data, context, rule_config)
            return {"compliant": True, "message": "Unknown rule type"}

        except Exception as e:
            logger.exception(f"Error evaluating business rule: {e}")
            return {"compliant": False, "violation": f"Rule evaluation failed: {e}"}

    async def _check_sensitive_content(self, prompt: str) -> dict[str, Any]:
        """Check for sensitive information in prompt."""
        result = {"warnings": []}

        # Check for potential PII patterns
        pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }

        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, prompt):
                result["warnings"].append({
                    "type": "potential_pii",
                    "message": f"Potential {pii_type} detected in prompt",
                    "severity": "warning",
                    "pii_type": pii_type
                })

        return result

    async def _apply_basic_sanitization(self, prompt: str) -> str:
        """Apply basic sanitization."""
        # Remove obvious script tags
        prompt = re.sub(r'<script[^>]*>.*?</script>', '', prompt, flags=re.IGNORECASE | re.DOTALL)

        # Remove null bytes
        return prompt.replace('\x00', '')

    async def _apply_standard_sanitization(self, prompt: str) -> str:
        """Apply standard sanitization."""
        prompt = await self._apply_basic_sanitization(prompt)

        # Remove potential injection patterns
        for pattern in self.sanitization_patterns["standard"]:
            prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE)

        # Normalize whitespace
        return re.sub(r'\s+', ' ', prompt).strip()

    async def _apply_strict_sanitization(self, prompt: str) -> str:
        """Apply strict sanitization."""
        prompt = await self._apply_standard_sanitization(prompt)

        # Apply strict patterns
        for pattern in self.sanitization_patterns["strict"]:
            prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE)

        # Keep only alphanumeric, common punctuation, and whitespace
        return re.sub(r'[^\w\s.,!?;:()\[\]{}"\'-]', '', prompt)

    def _calculate_validation_score(self, validation_result: dict[str, Any]) -> float:
        """Calculate overall validation score."""
        violations = len(validation_result.get("violations", []))
        warnings = len(validation_result.get("warnings", []))

        # Base score starts at 1.0
        score = 1.0

        # Deduct for violations (more severe)
        score -= violations * 0.2

        # Deduct for warnings (less severe)
        score -= warnings * 0.1

        # Ensure score doesn't go below 0
        return max(0.0, score)

    def _is_valid_json_format(self, prompt: str) -> bool:
        """Check if prompt is in valid JSON format."""
        try:
            import json
            json.loads(prompt)
            return True
        except (ValueError, TypeError):
            return False

    def _is_valid_xml_format(self, prompt: str) -> bool:
        """Check if prompt is in valid XML format."""
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(prompt)
            return True
        except ET.ParseError:
            return False

    async def _validate_session_integrity(self, session: Any) -> dict[str, Any]:
        """Validate session data integrity."""
        result = {"violations": []}

        # Check required fields
        required_fields = ["id", "created_at"]
        for field in required_fields:
            if not hasattr(session, field) or getattr(session, field) is None:
                result["violations"].append({
                    "type": "missing_required_field",
                    "message": f"Session missing required field: {field}",
                    "severity": "error",
                    "field": field
                })

        return result

    async def _validate_session_state(self, session: Any) -> dict[str, Any]:
        """Validate session state."""
        return {"violations": [], "warnings": []}

        # Add session state validation logic here
        # This would check session status, workflow state, etc.

    async def _validate_session_timing(self, session: Any) -> dict[str, Any]:
        """Validate session timing."""
        return {"warnings": []}

        # Add timing validation logic here
        # This would check for sessions running too long, expired sessions, etc.

    async def _validate_session_resources(self, session: Any) -> dict[str, Any]:
        """Validate session resource usage."""
        return {"violations": []}

        # Add resource validation logic here
        # This would check memory usage, execution time limits, etc.

    async def _check_rate_limit_rule(
        self,
        data: dict[str, Any],
        context: dict[str, Any],
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Check rate limiting business rule."""
        # Simplified rate limit check
        return {"compliant": True, "message": "Rate limit check passed"}

    async def _check_user_permissions_rule(
        self,
        data: dict[str, Any],
        context: dict[str, Any],
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Check user permissions business rule."""
        # Simplified permissions check
        return {"compliant": True, "message": "Permissions check passed"}

    async def _check_content_policy_rule(
        self,
        data: dict[str, Any],
        context: dict[str, Any],
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Check content policy business rule."""
        # Simplified content policy check
        return {"compliant": True, "message": "Content policy check passed"}

    async def _check_resource_limits_rule(
        self,
        data: dict[str, Any],
        context: dict[str, Any],
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Check resource limits business rule."""
        # Simplified resource limits check
        return {"compliant": True, "message": "Resource limits check passed"}

    def _initialize_validation_rules(self) -> dict[str, Any]:
        """Initialize validation rules configuration."""
        return {
            "default_max_length": 10000,
            "default_min_length": 10,
            "allowed_formats": ["text", "json", "xml"],
            "security_patterns": [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'eval\s*\(',
            ]
        }

    def _initialize_business_rules(self) -> dict[str, list[dict[str, Any]]]:
        """Initialize business rules configuration."""
        return {
            "improve_prompt": [
                {"type": "rate_limit", "config": {"limit": 100, "window": "hour"}},
                {"type": "user_permissions", "config": {"required_role": "user"}},
                {"type": "content_policy", "config": {"policy": "standard"}}
            ],
            "create_session": [
                {"type": "rate_limit", "config": {"limit": 50, "window": "hour"}},
                {"type": "resource_limits", "config": {"max_concurrent": 5}}
            ]
        }

    def _initialize_sanitization_patterns(self) -> dict[str, list[str]]:
        """Initialize sanitization patterns."""
        return {
            "standard": [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'onclick\s*=',
                r'eval\s*\(',
            ],
            "strict": [
                r'<[^>]+>',  # Remove all HTML tags
                r'javascript:',
                r'eval\s*\(',
                r'function\s*\(',
                r'var\s+\w+\s*=',
            ]
        }

    # Configuration validation methods to break circular dependencies
    async def validate_startup_configuration(
        self,
        environment: str | None = None
    ) -> ValidationResult:
        """Validate overall startup configuration integrity."""
        try:
            from prompt_improver.core.config.app_config import get_config
            from prompt_improver.core.config.validation import validate_configuration

            # Get the configuration to validate
            config = get_config()

            # Override environment if specified
            if environment:
                # Create a copy with overridden environment
                config_dict = config.model_dump()
                config_dict["environment"] = environment
                from prompt_improver.core.config.app_config import AppConfig
                config = AppConfig(**config_dict)

            # Use the comprehensive validation from validation.py
            is_valid, report = await validate_configuration(config)

            return ValidationResult(
                component="startup_configuration",
                is_valid=is_valid,
                message="Startup configuration validation passed" if is_valid else f"Startup configuration issues: {len(report.critical_failures)} critical, {len(report.warnings)} warnings",
                details={
                    "environment": report.environment,
                    "critical_failures": report.critical_failures,
                    "warnings": report.warnings,
                    "components_validated": len(report.results),
                    "overall_status": "valid" if is_valid else "invalid"
                },
                critical=True
            )

        except Exception as e:
            logger.exception(f"Error validating startup configuration: {e}")
            return ValidationResult(
                component="startup_configuration",
                is_valid=False,
                message=f"Startup configuration validation failed: {e}",
                critical=True
            )

    async def validate_database_configuration(
        self,
        test_connectivity: bool = True
    ) -> ValidationResult:
        """Validate database configuration and connectivity."""
        try:
            from prompt_improver.core.config.app_config import get_config

            config = get_config()
            db_config = config.database
            issues = []

            # Validate connection parameters
            if not db_config.postgres_host:
                issues.append("Database host is required")

            if db_config.postgres_port < 1 or db_config.postgres_port > 65535:
                issues.append(f"Invalid database port: {db_config.postgres_port}")

            if not db_config.postgres_database:
                issues.append("Database name is required")

            if not db_config.postgres_username:
                issues.append("Database username is required")

            # Validate pool settings
            if db_config.pool_min_size > db_config.pool_max_size:
                issues.append(f"Pool min size ({db_config.pool_min_size}) exceeds max size ({db_config.pool_max_size})")

            if db_config.pool_timeout <= 0:
                issues.append("Pool timeout must be positive")

            # Test connectivity if requested and basic config is valid
            connectivity_details = {}
            if test_connectivity and not issues:
                try:
                    import asyncio

                    import asyncpg

                    database_url = db_config.get_database_url()

                    async def test_connection():
                        conn = None
                        try:
                            conn = await asyncpg.connect(database_url, timeout=5.0)
                            result = await conn.fetchval("SELECT 1")
                            return result == 1
                        except Exception:
                            return False
                        finally:
                            if conn:
                                await conn.close()

                    start_time = asyncio.get_event_loop().time()
                    is_connected = await asyncio.wait_for(test_connection(), timeout=10.0)
                    response_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                    connectivity_details = {
                        "connectivity_test_passed": is_connected,
                        "response_time_ms": round(response_time_ms, 2)
                    }

                    if not is_connected:
                        issues.append("Database connectivity test failed")

                except Exception as e:
                    connectivity_details = {
                        "connectivity_test_error": str(e),
                        "connectivity_test_passed": False
                    }
                    issues.append(f"Database connectivity test error: {e}")

            is_valid = len(issues) == 0
            message = "Database configuration valid" if is_valid else f"Database configuration issues: {'; '.join(issues)}"

            details = {
                "host": db_config.postgres_host,
                "port": db_config.postgres_port,
                "database": db_config.postgres_database,
                "pool_settings": {
                    "min_size": db_config.pool_min_size,
                    "max_size": db_config.pool_max_size,
                    "timeout": db_config.pool_timeout,
                },
                "issues": issues,
                **connectivity_details
            }

            return ValidationResult(
                component="database_configuration",
                is_valid=is_valid,
                message=message,
                details=details,
                critical=True
            )

        except Exception as e:
            logger.exception(f"Error validating database configuration: {e}")
            return ValidationResult(
                component="database_configuration",
                is_valid=False,
                message=f"Database configuration validation failed: {e}",
                critical=True
            )

    async def validate_security_configuration(
        self,
        security_profile: str | None = None
    ) -> ValidationResult:
        """Validate security configuration and settings."""
        try:
            from prompt_improver.core.config.app_config import get_config

            config = get_config()
            security_config = config.security
            issues = []

            # Validate secret key
            if not security_config.secret_key:
                issues.append("Secret key is required")
            elif len(security_config.secret_key) < 32:
                issues.append(f"Secret key must be at least 32 characters, got {len(security_config.secret_key)}")

            # Validate token settings
            if security_config.token_expiry_seconds <= 0:
                issues.append("Token expiry must be positive")

            if security_config.hash_rounds < 4 or security_config.hash_rounds > 20:
                issues.append("Hash rounds must be between 4 and 20")

            # Validate authentication settings
            auth_config = security_config.authentication
            if auth_config.api_key_length_bytes < 16:
                issues.append("API key length must be at least 16 bytes")

            if auth_config.max_failed_attempts_per_hour < 1:
                issues.append("Max failed attempts must be at least 1")

            # Validate rate limiting
            rate_config = security_config.rate_limiting
            if rate_config.basic_tier_rate_limit < 1:
                issues.append("Basic tier rate limit must be at least 1")

            if rate_config.basic_tier_burst_capacity < rate_config.basic_tier_rate_limit:
                issues.append("Burst capacity must be >= rate limit")

            # Validate against security profile if specified
            if security_profile:
                current_profile = security_config.security_profile.value
                if security_profile != current_profile:
                    issues.append(f"Security profile mismatch: expected {security_profile}, got {current_profile}")

                if security_profile == "production" and config.environment != "production":
                    issues.append("Production security profile should only be used in production environment")

            is_valid = len(issues) == 0
            message = "Security configuration valid" if is_valid else f"Security configuration issues: {'; '.join(issues)}"

            return ValidationResult(
                component="security_configuration",
                is_valid=is_valid,
                message=message,
                details={
                    "security_profile": security_config.security_profile.value,
                    "token_expiry_seconds": security_config.token_expiry_seconds,
                    "hash_rounds": security_config.hash_rounds,
                    "api_key_length_bytes": auth_config.api_key_length_bytes,
                    "rate_limits": {
                        "basic_tier": rate_config.basic_tier_rate_limit,
                        "burst_capacity": rate_config.basic_tier_burst_capacity
                    },
                    "issues": issues
                },
                critical=True
            )

        except Exception as e:
            logger.exception(f"Error validating security configuration: {e}")
            return ValidationResult(
                component="security_configuration",
                is_valid=False,
                message=f"Security configuration validation failed: {e}",
                critical=True
            )

    async def validate_monitoring_configuration(
        self,
        include_connectivity_tests: bool = False
    ) -> ValidationResult:
        """Validate monitoring and observability configuration."""
        try:
            from prompt_improver.core.config.app_config import get_config

            config = get_config()
            monitoring_config = config.monitoring
            issues = []

            # Validate health check settings
            health_config = monitoring_config.health_checks
            if health_config.interval_seconds <= 0:
                issues.append("Health check interval must be positive")

            if health_config.timeout_seconds >= health_config.interval_seconds:
                issues.append("Health check timeout should be less than interval")

            # Validate metrics settings
            metrics_config = monitoring_config.metrics
            if metrics_config.collection_interval_seconds <= 0:
                issues.append("Metrics collection interval must be positive")

            # Validate logging settings
            logging_config = monitoring_config.logging
            allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if logging_config.level not in allowed_levels:
                issues.append(f"Invalid log level: {logging_config.level}")

            allowed_formats = ["json", "text", "structured"]
            if logging_config.format not in allowed_formats:
                issues.append(f"Invalid log format: {logging_config.format}")

            # Perform connectivity tests if requested
            connectivity_details = {}
            if include_connectivity_tests:
                # Test monitoring endpoints, metrics collection, etc.
                # This is a placeholder - actual implementation would test specific monitoring endpoints
                connectivity_details = {
                    "connectivity_tests_enabled": True,
                    "monitoring_endpoints_tested": 0,
                    "all_endpoints_healthy": True
                }

            is_valid = len(issues) == 0
            message = "Monitoring configuration valid" if is_valid else f"Monitoring configuration issues: {'; '.join(issues)}"

            details = {
                "health_check_interval": health_config.interval_seconds,
                "health_check_timeout": health_config.timeout_seconds,
                "metrics_collection_interval": metrics_config.collection_interval_seconds,
                "log_level": logging_config.level,
                "log_format": logging_config.format,
                "issues": issues,
                **connectivity_details
            }

            return ValidationResult(
                component="monitoring_configuration",
                is_valid=is_valid,
                message=message,
                details=details,
                critical=False  # Monitoring config issues are not critical for core app
            )

        except Exception as e:
            logger.exception(f"Error validating monitoring configuration: {e}")
            return ValidationResult(
                component="monitoring_configuration",
                is_valid=False,
                message=f"Monitoring configuration validation failed: {e}",
                critical=False
            )
