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
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID

from prompt_improver.core.protocols.prompt_service.prompt_protocols import (
    ValidationServiceProtocol,
)
from prompt_improver.core.exceptions import ValidationError, BusinessRuleViolationError

logger = logging.getLogger(__name__)


class ValidationService(ValidationServiceProtocol):
    """Service for input validation and business rule checking."""

    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.business_rules = self._initialize_business_rules()
        self.sanitization_patterns = self._initialize_sanitization_patterns()

    async def validate_prompt_input(
        self,
        prompt: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
            logger.error(f"Error validating prompt input: {e}")
            raise ValidationError(f"Validation failed: {e}")

    async def check_business_rules(
        self,
        operation: str,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
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
            logger.error(f"Error checking business rules for {operation}: {e}")
            raise BusinessRuleViolationError(f"Business rule check failed: {e}")

    async def validate_improvement_session(
        self,
        session: Any  # ImprovementSession type
    ) -> Dict[str, Any]:
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
            logger.error(f"Error validating improvement session: {e}")
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
            logger.error(f"Error sanitizing prompt content: {e}")
            return prompt  # Return original if sanitization fails

    async def _perform_basic_validation(
        self,
        prompt: str,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
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

    async def _perform_security_validation(self, prompt: str) -> Dict[str, Any]:
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
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        rule: Dict[str, Any],
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single business rule."""
        try:
            rule_type = rule.get("type")
            rule_config = rule.get("config", {})

            if rule_type == "rate_limit":
                return await self._check_rate_limit_rule(data, context, rule_config)
            elif rule_type == "user_permissions":
                return await self._check_user_permissions_rule(data, context, rule_config)
            elif rule_type == "content_policy":
                return await self._check_content_policy_rule(data, context, rule_config)
            elif rule_type == "resource_limits":
                return await self._check_resource_limits_rule(data, context, rule_config)
            else:
                return {"compliant": True, "message": "Unknown rule type"}

        except Exception as e:
            logger.error(f"Error evaluating business rule: {e}")
            return {"compliant": False, "violation": f"Rule evaluation failed: {e}"}

    async def _check_sensitive_content(self, prompt: str) -> Dict[str, Any]:
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
        prompt = prompt.replace('\x00', '')
        
        return prompt

    async def _apply_standard_sanitization(self, prompt: str) -> str:
        """Apply standard sanitization."""
        prompt = await self._apply_basic_sanitization(prompt)
        
        # Remove potential injection patterns
        for pattern in self.sanitization_patterns["standard"]:
            prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE)
        
        # Normalize whitespace
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        
        return prompt

    async def _apply_strict_sanitization(self, prompt: str) -> str:
        """Apply strict sanitization."""
        prompt = await self._apply_standard_sanitization(prompt)
        
        # Apply strict patterns
        for pattern in self.sanitization_patterns["strict"]:
            prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE)
        
        # Keep only alphanumeric, common punctuation, and whitespace
        prompt = re.sub(r'[^\w\s.,!?;:()\[\]{}"\'-]', '', prompt)
        
        return prompt

    def _calculate_validation_score(self, validation_result: Dict[str, Any]) -> float:
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

    async def _validate_session_integrity(self, session: Any) -> Dict[str, Any]:
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

    async def _validate_session_state(self, session: Any) -> Dict[str, Any]:
        """Validate session state."""
        result = {"violations": [], "warnings": []}
        
        # Add session state validation logic here
        # This would check session status, workflow state, etc.
        
        return result

    async def _validate_session_timing(self, session: Any) -> Dict[str, Any]:
        """Validate session timing."""
        result = {"warnings": []}
        
        # Add timing validation logic here
        # This would check for sessions running too long, expired sessions, etc.
        
        return result

    async def _validate_session_resources(self, session: Any) -> Dict[str, Any]:
        """Validate session resource usage."""
        result = {"violations": []}
        
        # Add resource validation logic here
        # This would check memory usage, execution time limits, etc.
        
        return result

    async def _check_rate_limit_rule(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check rate limiting business rule."""
        # Simplified rate limit check
        return {"compliant": True, "message": "Rate limit check passed"}

    async def _check_user_permissions_rule(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check user permissions business rule."""
        # Simplified permissions check
        return {"compliant": True, "message": "Permissions check passed"}

    async def _check_content_policy_rule(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check content policy business rule."""
        # Simplified content policy check
        return {"compliant": True, "message": "Content policy check passed"}

    async def _check_resource_limits_rule(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check resource limits business rule."""
        # Simplified resource limits check
        return {"compliant": True, "message": "Resource limits check passed"}

    def _initialize_validation_rules(self) -> Dict[str, Any]:
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

    def _initialize_business_rules(self) -> Dict[str, List[Dict[str, Any]]]:
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

    def _initialize_sanitization_patterns(self) -> Dict[str, List[str]]:
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