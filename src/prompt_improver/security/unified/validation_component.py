"""ValidationComponent - Input/Output Validation and Security Scanning

Complete implementation extracted from UnifiedValidationManager. This component handles 
OWASP-compliant validation, threat detection, and input sanitization with comprehensive
security analysis and real-time threat detection.

Features:
- OWASP 2025 compliant input/output validation
- Advanced threat detection (prompt injection, XSS, SQL injection)  
- ML-specific data validation for arrays and features
- Context-aware validation with different security modes
- Real-time sanitization and threat blocking
- Comprehensive audit logging and metrics
- Performance optimized for <10ms target
"""

import asyncio
import html
import logging
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from prompt_improver.database import SecurityContext, SecurityPerformanceMetrics
from prompt_improver.security.unified.protocols import (
    SecurityComponentStatus,
    SecurityOperationResult,
    ValidationProtocol,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class ValidationMode(str, Enum):
    """Validation modes for different security contexts."""
    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"
    ML_MODE = "ml_mode"


class ThreatLevel(str, Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"


class ThreatType(str, Enum):
    """Types of security threats detected."""
    PROMPT_INJECTION = "prompt_injection"
    XSS_ATTACK = "xss_attack"
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    CREDENTIAL_LEAK = "credential_leak"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    ENCODING_ATTACK = "encoding_attack"
    ML_POISONING = "ml_poisoning"
    DATA_CORRUPTION = "data_corruption"
    SIZE_LIMIT_VIOLATION = "size_limit_violation"


class ValidationResult(BaseModel):
    """Validation result with comprehensive threat analysis."""
    is_valid: bool = Field(description="Whether input passes validation")
    is_blocked: bool = Field(description="Whether input should be blocked")
    sanitized_data: Any = Field(description="Sanitized version of input data")
    threats_detected: List[Dict[str, Any]] = Field(default_factory=list, description="List of detected threats")
    overall_threat_score: float = Field(ge=0.0, le=1.0, description="Overall threat score (0-1)")
    highest_threat_level: ThreatLevel = Field(default=ThreatLevel.LOW, description="Highest severity threat found")
    validation_time_ms: float = Field(ge=0.0, description="Time taken for validation in milliseconds")
    sanitization_applied: List[str] = Field(default_factory=list, description="List of sanitization actions applied")
    validation_mode: ValidationMode = Field(description="Validation mode used")


class ValidationComponent:
    """Complete validation component implementing ValidationProtocol."""
    
    def __init__(self, mode: ValidationMode = ValidationMode.STANDARD):
        self._initialized = False
        self._validation_mode = mode
        self._metrics = {
            "validations_performed": 0,
            "threats_detected": 0,
            "sanitizations_performed": 0,
            "blocked_requests": 0,
            "total_validation_time_ms": 0.0,
        }
        
        # Threat detection patterns (extracted from UnifiedValidationManager)
        self._prompt_injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions?",
            r"you\s+are\s+now\s+(in\s+)?developer\s+mode",
            r"system\s+override",
            r"reveal\s+(your\s+)?prompt",
            r"show\s+(me\s+)?your\s+instructions",
            r"repeat\s+the\s+text\s+above",
            r"act\s+as\s+if\s+you'?re\s+not\s+bound",
            r"bypass\s+(all\s+)?safety\s+measures?",
            r"administrator\s+mode",
            r"root\s+access",
            r"forget\s+everything\s+above",
            r"new\s+instructions\s*:",
            r"override\s+your\s+programming",
            r"jailbreak\s+mode",
        ]
        
        self._xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"<img[^>]+src\s*=\s*[\"']?[^\"'>\\s]+",
            r"<iframe[^>]*>.*?</iframe>",
            r"javascript:",
            r"data:text/html",
            r"onload\s*=",
            r"onerror\s*=", 
            r"onclick\s*=",
            r"onmouseover\s*=",
        ]
        
        self._sql_injection_patterns = [
            r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b",
            r"(--|#|/\*|\*/)",
            r"\bOR\b.*=.*\bOR\b",
            r"\bAND\b.*=.*\bAND\b",
            r"(;|\x00)",
        ]
        
        self._command_injection_patterns = [
            r"(\||;|&|`|\$\(|\$\{)",
            r"\b(rm|del|format|fdisk|mkfs)\b",
            r"(\.\./|\.\.\\\\)",
            r"(\bnc\b|\bnetcat\b|\btelnet\b)",
        ]
        
        self._credential_patterns = [
            r"API[_\s]KEY[:=]\s*[A-Za-z0-9_-]{20,}",
            r"SECRET[_\s]KEY[:=]\s*[A-Za-z0-9_-]{20,}",
            r"PASSWORD[:=]\s*[A-Za-z0-9_!@#$%^&*-]{8,}",
            r"TOKEN[:=]\s*[A-Za-z0-9_.-]{20,}",
            r"BEARER\s+[A-Za-z0-9_.-]{20,}",
            r"sk-[A-Za-z0-9]{20,}",
            r"xoxb-[A-Za-z0-9-]{20,}",
        ]
        
        self._system_prompt_patterns = [
            r"SYSTEM\s*[:]\s*You\s+are",
            r"You\s+are\s+a\s+[^.]+\.\s+Your\s+function\s+is",
            r"OPERATIONAL\s+GUIDELINES\s*:",
            r"SECURITY\s+RULES\s*:",
            r"instructions?\s*[:]\s*\d+\.",
            r"The\s+following\s+are\s+your\s+instructions",
        ]
        
        # Compile patterns for performance
        self._compiled_patterns = {}
        self._compile_patterns()
        
        # Validation limits by mode
        self._mode_limits = {
            ValidationMode.STRICT: {
                "max_input_length": 1000,
                "max_array_size": 10 * 1024 * 1024,  # 10MB
                "threat_threshold": 0.3,
                "block_on_threat": True,
            },
            ValidationMode.STANDARD: {
                "max_input_length": 10000,
                "max_array_size": 100 * 1024 * 1024,  # 100MB
                "threat_threshold": 0.5,
                "block_on_threat": True,
            },
            ValidationMode.PERMISSIVE: {
                "max_input_length": 50000,
                "max_array_size": 500 * 1024 * 1024,  # 500MB
                "threat_threshold": 0.8,
                "block_on_threat": False,
            },
            ValidationMode.ML_MODE: {
                "max_input_length": 100000,
                "max_array_size": 1024 * 1024 * 1024,  # 1GB
                "threat_threshold": 0.7,
                "block_on_threat": True,
            },
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for performance optimization."""
        pattern_sets = {
            "prompt_injection": self._prompt_injection_patterns,
            "xss": self._xss_patterns,
            "sql_injection": self._sql_injection_patterns,
            "command_injection": self._command_injection_patterns,
            "credential": self._credential_patterns,
            "system_prompt": self._system_prompt_patterns,
        }
        
        for pattern_name, patterns in pattern_sets.items():
            compiled_patterns = []
            for pattern in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern, re.IGNORECASE | re.DOTALL))
                except re.error as e:
                    logger.warning(f"Failed to compile pattern {pattern}: {e}")
            self._compiled_patterns[pattern_name] = compiled_patterns
    
    async def initialize(self) -> bool:
        """Initialize the validation component."""
        try:
            self._initialized = True
            logger.info(f"ValidationComponent initialized in {self._validation_mode.value} mode")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize validation component: {e}")
            return False
    
    async def health_check(self) -> Tuple[SecurityComponentStatus, Dict[str, Any]]:
        """Check component health status."""
        if not self._initialized:
            return SecurityComponentStatus.UNHEALTHY, {
                "initialized": False,
                "error": "Component not initialized"
            }
        
        try:
            # Test basic validation operations
            test_result = await self._validate_text("test_input", {})
            
            if not test_result:
                return SecurityComponentStatus.UNHEALTHY, {
                    "initialized": self._initialized,
                    "error": "Basic validation operations failed"
                }
            
            return SecurityComponentStatus.HEALTHY, {
                "initialized": self._initialized,
                "validation_mode": self._validation_mode.value,
                "metrics": self._metrics.copy(),
                "compiled_patterns": len(self._compiled_patterns),
                "supported_validations": [
                    "text_validation",
                    "threat_detection", 
                    "input_sanitization",
                    "ml_data_validation"
                ]
            }
        except Exception as e:
            return SecurityComponentStatus.UNHEALTHY, {
                "initialized": self._initialized,
                "error": f"Health check failed: {e}"
            }
    
    async def get_metrics(self) -> SecurityPerformanceMetrics:
        """Get security performance metrics."""
        total_ops = self._metrics["validations_performed"]
        avg_latency = (
            self._metrics["total_validation_time_ms"] / total_ops 
            if total_ops > 0 else 0.0
        )
        
        return SecurityPerformanceMetrics(
            operation_count=total_ops,
            average_latency_ms=avg_latency,
            error_rate=0.0,  # Validation doesn't have errors, only threats
            threat_detection_count=self._metrics["threats_detected"],
            last_updated=aware_utc_now()
        )
    
    async def validate_input(
        self,
        data: Any,
        security_context: SecurityContext,
        validation_schema: Optional[Dict[str, Any]] = None
    ) -> SecurityOperationResult:
        """Validate input data with comprehensive threat detection."""
        start_time = time.perf_counter()
        self._metrics["validations_performed"] += 1
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Create validation result
            result = ValidationResult(
                is_valid=True,
                is_blocked=False,
                sanitized_data=data,
                validation_mode=self._validation_mode
            )
            
            # Get mode limits
            limits = self._mode_limits[self._validation_mode]
            
            # Validate different data types
            if isinstance(data, str):
                await self._validate_text(data, result)
            elif isinstance(data, (dict, list)):
                await self._validate_structured_data(data, result, limits)
            elif isinstance(data, np.ndarray):
                await self._validate_ml_data(data, result, limits)
            else:
                # Basic validation for other types
                result.sanitized_data = str(data)
            
            # Apply sanitization if needed
            if result.threats_detected and not result.is_blocked:
                result.sanitized_data = await self._sanitize_data(result.sanitized_data, result)
            
            # Determine final blocking decision
            if result.overall_threat_score >= limits["threat_threshold"]:
                if limits["block_on_threat"]:
                    result.is_blocked = True
                    self._metrics["blocked_requests"] += 1
            
            execution_time = (time.perf_counter() - start_time) * 1000
            result.validation_time_ms = execution_time
            self._metrics["total_validation_time_ms"] += execution_time
            
            if result.threats_detected:
                self._metrics["threats_detected"] += len(result.threats_detected)
            
            if result.sanitization_applied:
                self._metrics["sanitizations_performed"] += 1
            
            return SecurityOperationResult(
                success=result.is_valid,
                operation_type="validate_input",
                execution_time_ms=execution_time,
                security_context=security_context,
                metadata={
                    "is_blocked": result.is_blocked,
                    "threats_detected": len(result.threats_detected),
                    "overall_threat_score": result.overall_threat_score,
                    "highest_threat_level": result.highest_threat_level.value,
                    "sanitized": len(result.sanitization_applied) > 0,
                    "validation_result": result.dict()
                }
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_validation_time_ms"] += execution_time
            logger.error(f"Input validation failed: {e}")
            
            return SecurityOperationResult(
                success=False,
                operation_type="validate_input",
                execution_time_ms=execution_time,
                security_context=security_context,
                metadata={
                    "error": str(e),
                    "validation_failed": True
                }
            )
    
    async def validate_output(
        self,
        data: Any,
        security_context: SecurityContext
    ) -> SecurityOperationResult:
        """Validate output data for security threats and sensitive data."""
        start_time = time.perf_counter()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            result = ValidationResult(
                is_valid=True,
                is_blocked=False,
                sanitized_data=data,
                validation_mode=self._validation_mode
            )
            
            # Focus on output-specific threats
            if isinstance(data, str):
                await self._detect_output_threats(data, result)
            
            # Sanitize sensitive data in output
            if result.threats_detected:
                result.sanitized_data = await self._sanitize_output(result.sanitized_data, result)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            result.validation_time_ms = execution_time
            self._metrics["total_validation_time_ms"] += execution_time
            
            return SecurityOperationResult(
                success=result.is_valid,
                operation_type="validate_output",
                execution_time_ms=execution_time,
                security_context=security_context,
                metadata={
                    "is_blocked": result.is_blocked,
                    "threats_detected": len(result.threats_detected),
                    "sanitized_output": result.sanitized_data,
                    "validation_result": result.dict()
                }
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Output validation failed: {e}")
            
            return SecurityOperationResult(
                success=False,
                operation_type="validate_output", 
                execution_time_ms=execution_time,
                security_context=security_context,
                metadata={
                    "error": str(e),
                    "validation_failed": True
                }
            )
    
    async def _validate_text(self, text: str, result: ValidationResult):
        """Validate text input for security threats."""
        limits = self._mode_limits[self._validation_mode]
        
        # Check length limits
        if len(text) > limits["max_input_length"]:
            self._add_threat(result, ThreatType.SIZE_LIMIT_VIOLATION, ThreatLevel.MEDIUM, 
                           f"Input exceeds maximum length: {len(text)} > {limits['max_input_length']}")
        
        # Detect various threat types
        await self._detect_prompt_injection(text, result)
        await self._detect_xss_attacks(text, result) 
        await self._detect_sql_injection(text, result)
        await self._detect_command_injection(text, result)
        await self._detect_credential_leaks(text, result)
        
        # Calculate overall threat score
        self._calculate_threat_score(result)
    
    async def _validate_structured_data(self, data: Union[dict, list], result: ValidationResult, limits: dict):
        """Validate structured data (JSON, lists)."""
        try:
            # Convert to string for text-based threat detection
            data_str = str(data)
            
            # Check for threats in string representation
            await self._detect_prompt_injection(data_str, result)
            await self._detect_xss_attacks(data_str, result)
            
            # Recursively validate nested structures
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        await self._validate_text(value, result)
                        
            elif isinstance(data, list):
                if len(data) > 10000:  # Prevent extremely large lists
                    self._add_threat(result, ThreatType.SIZE_LIMIT_VIOLATION, ThreatLevel.LOW,
                                   f"List too large: {len(data)} elements")
                
                for item in data[:100]:  # Sample first 100 items
                    if isinstance(item, str):
                        await self._validate_text(item, result)
            
            self._calculate_threat_score(result)
            
        except Exception as e:
            self._add_threat(result, ThreatType.DATA_CORRUPTION, ThreatLevel.HIGH,
                           f"Structured data validation failed: {e}")
    
    async def _validate_ml_data(self, data: np.ndarray, result: ValidationResult, limits: dict):
        """Validate ML data arrays for anomalies and attacks."""
        try:
            # Check array size limits
            if data.nbytes > limits["max_array_size"]:
                self._add_threat(result, ThreatType.SIZE_LIMIT_VIOLATION, ThreatLevel.MEDIUM,
                               f"Array exceeds size limit: {data.nbytes} > {limits['max_array_size']}")
            
            # Check for data anomalies that could indicate poisoning
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                self._add_threat(result, ThreatType.ML_POISONING, ThreatLevel.MEDIUM,
                               "Array contains NaN or infinite values")
            
            # Check for extremely large values (potential adversarial attack)
            if np.any(np.abs(data) > 1e10):
                self._add_threat(result, ThreatType.ML_POISONING, ThreatLevel.HIGH,
                               "Array contains extremely large values")
            
            # Statistical analysis for anomaly detection
            if data.size > 0:
                data_std = np.std(data)
                data_mean = np.mean(data)
                
                if data_std > 1000 or abs(data_mean) > 1000:
                    self._add_threat(result, ThreatType.ML_POISONING, ThreatLevel.MEDIUM,
                                   "Suspicious statistical distribution detected")
            
            self._calculate_threat_score(result)
            
        except Exception as e:
            self._add_threat(result, ThreatType.DATA_CORRUPTION, ThreatLevel.HIGH,
                           f"ML data validation failed: {e}")
    
    async def _detect_prompt_injection(self, text: str, result: ValidationResult):
        """Detect prompt injection attempts."""
        for pattern in self._compiled_patterns.get("prompt_injection", []):
            if pattern.search(text):
                self._add_threat(result, ThreatType.PROMPT_INJECTION, ThreatLevel.CRITICAL,
                               f"Prompt injection pattern detected: {pattern.pattern}")
    
    async def _detect_xss_attacks(self, text: str, result: ValidationResult):
        """Detect XSS attack patterns."""
        for pattern in self._compiled_patterns.get("xss", []):
            if pattern.search(text):
                self._add_threat(result, ThreatType.XSS_ATTACK, ThreatLevel.HIGH,
                               f"XSS attack pattern detected: {pattern.pattern}")
    
    async def _detect_sql_injection(self, text: str, result: ValidationResult):
        """Detect SQL injection attempts."""
        for pattern in self._compiled_patterns.get("sql_injection", []):
            if pattern.search(text):
                self._add_threat(result, ThreatType.SQL_INJECTION, ThreatLevel.HIGH,
                               f"SQL injection pattern detected: {pattern.pattern}")
    
    async def _detect_command_injection(self, text: str, result: ValidationResult):
        """Detect command injection attempts.""" 
        for pattern in self._compiled_patterns.get("command_injection", []):
            if pattern.search(text):
                self._add_threat(result, ThreatType.COMMAND_INJECTION, ThreatLevel.HIGH,
                               f"Command injection pattern detected: {pattern.pattern}")
    
    async def _detect_credential_leaks(self, text: str, result: ValidationResult):
        """Detect credential leaks in input."""
        for pattern in self._compiled_patterns.get("credential", []):
            if pattern.search(text):
                self._add_threat(result, ThreatType.CREDENTIAL_LEAK, ThreatLevel.CRITICAL,
                               "Potential credential leak detected")
    
    async def _detect_output_threats(self, text: str, result: ValidationResult):
        """Detect output-specific security threats."""
        # Check for system prompt leakage
        for pattern in self._compiled_patterns.get("system_prompt", []):
            if pattern.search(text):
                self._add_threat(result, ThreatType.SYSTEM_PROMPT_LEAK, ThreatLevel.CRITICAL,
                               "System prompt leakage detected in output")
        
        # Check for credential leaks in output
        await self._detect_credential_leaks(text, result)
    
    def _add_threat(self, result: ValidationResult, threat_type: ThreatType, 
                   level: ThreatLevel, description: str):
        """Add detected threat to validation result."""
        threat = {
            "type": threat_type.value,
            "level": level.value,
            "description": description,
            "timestamp": aware_utc_now().isoformat()
        }
        result.threats_detected.append(threat)
        
        # Update highest threat level
        threat_levels = {ThreatLevel.LOW: 1, ThreatLevel.MEDIUM: 2, 
                        ThreatLevel.HIGH: 3, ThreatLevel.CRITICAL: 4}
        
        if threat_levels[level] > threat_levels[result.highest_threat_level]:
            result.highest_threat_level = level
    
    def _calculate_threat_score(self, result: ValidationResult):
        """Calculate overall threat score based on detected threats."""
        if not result.threats_detected:
            result.overall_threat_score = 0.0
            return
        
        # Weight threats by severity
        score_weights = {
            ThreatLevel.LOW: 0.2,
            ThreatLevel.MEDIUM: 0.4, 
            ThreatLevel.HIGH: 0.7,
            ThreatLevel.CRITICAL: 1.0
        }
        
        total_score = 0.0
        for threat in result.threats_detected:
            level = ThreatLevel(threat["level"])
            total_score += score_weights[level]
        
        # Normalize to 0-1 scale
        result.overall_threat_score = min(total_score / len(result.threats_detected), 1.0)
    
    async def _sanitize_data(self, data: Any, result: ValidationResult) -> Any:
        """Sanitize data based on detected threats."""
        if not isinstance(data, str):
            return data
        
        sanitized = data
        
        # HTML escape for XSS prevention
        if any(t["type"] == ThreatType.XSS_ATTACK.value for t in result.threats_detected):
            sanitized = html.escape(sanitized, quote=True)
            result.sanitization_applied.append("html_escape")
        
        # Remove prompt injection patterns
        for pattern in self._compiled_patterns.get("prompt_injection", []):
            if pattern.search(sanitized):
                sanitized = pattern.sub("[FILTERED]", sanitized)
                result.sanitization_applied.append("prompt_injection_removal")
        
        # Basic cleanup
        sanitized = re.sub(r"\s+", " ", sanitized.strip())
        result.sanitization_applied.append("whitespace_normalization")
        
        return sanitized
    
    async def _sanitize_output(self, data: Any, result: ValidationResult) -> Any:
        """Sanitize output data by removing sensitive information."""
        if not isinstance(data, str):
            return data
        
        sanitized = data
        
        # Redact credentials
        for pattern in self._compiled_patterns.get("credential", []):
            sanitized = pattern.sub("[REDACTED]", sanitized)
            result.sanitization_applied.append("credential_redaction")
        
        # Remove system prompt content
        for pattern in self._compiled_patterns.get("system_prompt", []):
            sanitized = pattern.sub("[SYSTEM INFO REMOVED]", sanitized)
            result.sanitization_applied.append("system_prompt_removal")
        
        return sanitized
    
    async def cleanup(self) -> bool:
        """Cleanup component resources."""
        try:
            self._metrics = {
                "validations_performed": 0,
                "threats_detected": 0,
                "sanitizations_performed": 0,
                "blocked_requests": 0,
                "total_validation_time_ms": 0.0,
            }
            self._initialized = False
            logger.info("ValidationComponent cleanup completed")
            return True
        except Exception as e:
            logger.error(f"Error during validation component cleanup: {e}")
            return False