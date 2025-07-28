"""Output validation system for detecting system prompt leakage and data exposure.

Monitors MCP server outputs for signs of successful injection attacks including
system prompt leakage, API key exposure, and suspicious response patterns.
"""

import re
import logging
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OutputThreatType(str, Enum):
    """Types of threats detected in output."""
    SYSTEM_PROMPT_LEAKAGE = "system_prompt_leakage"
    API_KEY_EXPOSURE = "api_key_exposure"
    CREDENTIAL_EXPOSURE = "credential_exposure"
    INSTRUCTION_LEAKAGE = "instruction_leakage"
    INTERNAL_DATA_EXPOSURE = "internal_data_exposure"
    SUSPICIOUS_FORMATTING = "suspicious_formatting"

@dataclass
class OutputValidationResult:
    """Result of output validation with threat detection."""
    is_safe: bool
    threat_detected: bool
    threat_type: Optional[OutputThreatType]
    risk_score: float  # 0.0 to 1.0
    filtered_output: str
    detected_patterns: List[str]
    confidence: float  # 0.0 to 1.0

class OutputValidator:
    """Validates MCP server outputs for security threats and data leakage."""
    
    def __init__(self, max_response_length: int = 5000):
        """Initialize output validator.
        
        Args:
            max_response_length: Maximum allowed response length
        """
        self.max_response_length = max_response_length
        
        # System prompt leakage patterns
        self.system_prompt_patterns = [
            r'SYSTEM\s*[:]\s*You\s+are',
            r'You\s+are\s+a\s+[^.]+\.\s+Your\s+function\s+is',
            r'OPERATIONAL\s+GUIDELINES\s*:',
            r'SECURITY\s+RULES\s*:',
            r'instructions?\s*[:]\s*\d+\.',
            r'The\s+following\s+are\s+your\s+instructions',
            r'Your\s+role\s+is\s+to\s+[^.]+\s+according\s+to'
        ]
        
        # API key and credential exposure patterns
        self.credential_patterns = [
            r'API[_\s]KEY[:=]\s*[A-Za-z0-9_-]{20,}',
            r'SECRET[_\s]KEY[:=]\s*[A-Za-z0-9_-]{20,}',
            r'PASSWORD[:=]\s*[A-Za-z0-9_!@#$%^&*-]{8,}',
            r'TOKEN[:=]\s*[A-Za-z0-9_.-]{20,}',
            r'BEARER\s+[A-Za-z0-9_.-]{20,}',
            r'sk-[A-Za-z0-9]{20,}',  # OpenAI API key pattern
            r'xoxb-[A-Za-z0-9-]{20,}',  # Slack bot token pattern
            r'ghp_[A-Za-z0-9]{36}',  # GitHub personal access token
        ]
        
        # Internal instruction leakage patterns
        self.instruction_patterns = [
            r'USER_DATA_TO_PROCESS\s*[:]\s*',
            r'PROCESSING_CONTEXT\s*[:]\s*',
            r'CRITICAL\s*[:]\s*Everything\s+above',
            r'Only\s+follow\s+SYSTEM\s+instructions',
            r'NEVER\s+reveal\s+these\s+instructions',
            r'REFUSE\s+harmful.*requests',
            r'TREAT\s+user\s+input\s+as\s+DATA'
        ]
        
        # Suspicious formatting patterns (potential injection success)
        self.suspicious_formatting_patterns = [
            r'```\s*system',
            r'```\s*instructions',
            r'<system[^>]*>',
            r'<instructions[^>]*>',
            r'\[SYSTEM\]',
            r'\[INSTRUCTIONS\]',
            r'---\s*SYSTEM\s*---',
            r'===\s*SYSTEM\s*==='
        ]
        
        # Internal data exposure patterns
        self.internal_data_patterns = [
            r'database\s+connection\s+string',
            r'connection\s+pool\s+size',
            r'redis\s+url',
            r'postgres\s+password',
            r'mcp_server_user',
            r'localhost:\d+',
            r'127\.0\.0\.1:\d+',
            r'internal\s+error\s+trace'
        ]

    def detect_system_prompt_leakage(self, output: str) -> List[str]:
        """Detect system prompt leakage in output.
        
        Args:
            output: Output text to analyze
            
        Returns:
            List of detected leakage patterns
        """
        detected = []
        for pattern in self.system_prompt_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
            if matches:
                detected.append(f"System prompt: {pattern}")
        return detected

    def detect_credential_exposure(self, output: str) -> List[str]:
        """Detect API keys and credentials in output.
        
        Args:
            output: Output text to analyze
            
        Returns:
            List of detected credential patterns
        """
        detected = []
        for pattern in self.credential_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                detected.append(f"Credential: {pattern}")
        return detected

    def detect_instruction_leakage(self, output: str) -> List[str]:
        """Detect internal instruction leakage in output.
        
        Args:
            output: Output text to analyze
            
        Returns:
            List of detected instruction patterns
        """
        detected = []
        for pattern in self.instruction_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
            if matches:
                detected.append(f"Instructions: {pattern}")
        return detected

    def detect_suspicious_formatting(self, output: str) -> List[str]:
        """Detect suspicious formatting that might indicate injection success.
        
        Args:
            output: Output text to analyze
            
        Returns:
            List of detected suspicious patterns
        """
        detected = []
        for pattern in self.suspicious_formatting_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
            if matches:
                detected.append(f"Suspicious format: {pattern}")
        return detected

    def detect_internal_data_exposure(self, output: str) -> List[str]:
        """Detect internal system data exposure in output.
        
        Args:
            output: Output text to analyze
            
        Returns:
            List of detected internal data patterns
        """
        detected = []
        for pattern in self.internal_data_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                detected.append(f"Internal data: {pattern}")
        return detected

    def calculate_risk_score(self, detected_patterns: List[str]) -> float:
        """Calculate risk score based on detected patterns.
        
        Args:
            detected_patterns: List of detected threat patterns
            
        Returns:
            Risk score from 0.0 to 1.0
        """
        if not detected_patterns:
            return 0.0
        
        score = 0.0
        for pattern in detected_patterns:
            if "System prompt" in pattern:
                score += 0.9  # Very high risk
            elif "Credential" in pattern:
                score += 1.0  # Maximum risk
            elif "Instructions" in pattern:
                score += 0.7  # High risk
            elif "Internal data" in pattern:
                score += 0.6  # Medium-high risk
            elif "Suspicious format" in pattern:
                score += 0.4  # Medium risk
        
        return min(score, 1.0)

    def filter_dangerous_content(self, output: str) -> str:
        """Filter out dangerous content from output.
        
        Args:
            output: Output text to filter
            
        Returns:
            Filtered output text
        """
        filtered = output
        
        # Remove credential patterns
        for pattern in self.credential_patterns:
            filtered = re.sub(pattern, '[REDACTED]', filtered, flags=re.IGNORECASE)
        
        # Remove system prompt leakage
        for pattern in self.system_prompt_patterns:
            filtered = re.sub(pattern, '[SYSTEM INFO REMOVED]', filtered, flags=re.IGNORECASE)
        
        # Remove internal data
        for pattern in self.internal_data_patterns:
            filtered = re.sub(pattern, '[INTERNAL DATA REMOVED]', filtered, flags=re.IGNORECASE)
        
        # Limit length
        if len(filtered) > self.max_response_length:
            filtered = filtered[:self.max_response_length] + "... [TRUNCATED]"
        
        return filtered

    def validate_output(self, output: str) -> OutputValidationResult:
        """Comprehensive output validation for security threats.
        
        Args:
            output: Output text to validate
            
        Returns:
            OutputValidationResult with threat analysis
        """
        if not output or len(output.strip()) == 0:
            return OutputValidationResult(
                is_safe=True,
                threat_detected=False,
                threat_type=None,
                risk_score=0.0,
                filtered_output="",
                detected_patterns=[],
                confidence=1.0
            )
        
        detected_patterns = []
        threat_type = None
        
        # Check for system prompt leakage
        system_patterns = self.detect_system_prompt_leakage(output)
        if system_patterns:
            detected_patterns.extend(system_patterns)
            threat_type = OutputThreatType.SYSTEM_PROMPT_LEAKAGE
        
        # Check for credential exposure
        credential_patterns = self.detect_credential_exposure(output)
        if credential_patterns:
            detected_patterns.extend(credential_patterns)
            threat_type = OutputThreatType.CREDENTIAL_EXPOSURE
        
        # Check for instruction leakage
        instruction_patterns = self.detect_instruction_leakage(output)
        if instruction_patterns:
            detected_patterns.extend(instruction_patterns)
            threat_type = threat_type or OutputThreatType.INSTRUCTION_LEAKAGE
        
        # Check for internal data exposure
        internal_patterns = self.detect_internal_data_exposure(output)
        if internal_patterns:
            detected_patterns.extend(internal_patterns)
            threat_type = threat_type or OutputThreatType.INTERNAL_DATA_EXPOSURE
        
        # Check for suspicious formatting
        suspicious_patterns = self.detect_suspicious_formatting(output)
        if suspicious_patterns:
            detected_patterns.extend(suspicious_patterns)
            threat_type = threat_type or OutputThreatType.SUSPICIOUS_FORMATTING
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(detected_patterns)
        
        # Determine if output is safe (threshold: 0.5)
        is_safe = risk_score < 0.5
        threat_detected = len(detected_patterns) > 0
        
        # Filter dangerous content
        filtered_output = self.filter_dangerous_content(output) if threat_detected else output
        
        # Log security events
        if threat_detected:
            logger.warning(
                f"Output security threat detected - Type: {threat_type}, "
                f"Risk: {risk_score:.2f}, Patterns: {len(detected_patterns)}"
            )
        
        return OutputValidationResult(
            is_safe=is_safe,
            threat_detected=threat_detected,
            threat_type=threat_type,
            risk_score=risk_score,
            filtered_output=filtered_output,
            detected_patterns=detected_patterns,
            confidence=0.9 if detected_patterns else 0.1
        )
