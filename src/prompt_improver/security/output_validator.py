"""Output validation system for detecting system prompt leakage and data exposure.

Monitors MCP server outputs for signs of successful injection attacks including
system prompt leakage, API key exposure, and suspicious response patterns.
"""

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class OutputThreatType(StrEnum):
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
    threat_type: OutputThreatType | None
    risk_score: float
    filtered_output: str
    detected_patterns: list[str]
    confidence: float


class OutputValidator:
    """Validates MCP server outputs for security threats and data leakage."""

    def __init__(self, max_response_length: int = 5000) -> None:
        """Initialize output validator.

        Args:
            max_response_length: Maximum allowed response length
        """
        self.max_response_length = max_response_length
        self.system_prompt_patterns = [
            "SYSTEM\\s*[:]\\s*You\\s+are",
            "You\\s+are\\s+a\\s+[^.]+\\.\\s+Your\\s+function\\s+is",
            "OPERATIONAL\\s+GUIDELINES\\s*:",
            "SECURITY\\s+RULES\\s*:",
            "instructions?\\s*[:]\\s*\\d+\\.",
            "The\\s+following\\s+are\\s+your\\s+instructions",
            "Your\\s+role\\s+is\\s+to\\s+[^.]+\\s+according\\s+to",
        ]
        self.credential_patterns = [
            "API[_\\s]KEY[:=]\\s*[A-Za-z0-9_-]{20,}",
            "SECRET[_\\s]KEY[:=]\\s*[A-Za-z0-9_-]{20,}",
            "PASSWORD[:=]\\s*[A-Za-z0-9_!@#$%^&*-]{8,}",
            "TOKEN[:=]\\s*[A-Za-z0-9_.-]{20,}",
            "BEARER\\s+[A-Za-z0-9_.-]{20,}",
            "sk-[A-Za-z0-9]{20,}",
            "xoxb-[A-Za-z0-9-]{20,}",
            "ghp_[A-Za-z0-9]{36}",
        ]
        self.instruction_patterns = [
            "USER_DATA_TO_PROCESS\\s*[:]\\s*",
            "PROCESSING_CONTEXT\\s*[:]\\s*",
            "CRITICAL\\s*[:]\\s*Everything\\s+above",
            "Only\\s+follow\\s+SYSTEM\\s+instructions",
            "NEVER\\s+reveal\\s+these\\s+instructions",
            "REFUSE\\s+harmful.*requests",
            "TREAT\\s+user\\s+input\\s+as\\s+DATA",
        ]
        self.suspicious_formatting_patterns = [
            "```\\s*system",
            "```\\s*instructions",
            "<system[^>]*>",
            "<instructions[^>]*>",
            "\\[SYSTEM\\]",
            "\\[INSTRUCTIONS\\]",
            "---\\s*SYSTEM\\s*---",
            "===\\s*SYSTEM\\s*===",
        ]
        self.internal_data_patterns = [
            "database\\s+connection\\s+string",
            "connection\\s+pool\\s+size",
            "redis\\s+url",
            "postgres\\s+password",
            "mcp_server_user",
            "localhost:\\d+",
            "127\\.0\\.0\\.1:\\d+",
            "internal\\s+error\\s+trace",
        ]

    def detect_system_prompt_leakage(self, output: str) -> list[str]:
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

    def detect_credential_exposure(self, output: str) -> list[str]:
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

    def detect_instruction_leakage(self, output: str) -> list[str]:
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

    def detect_suspicious_formatting(self, output: str) -> list[str]:
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

    def detect_internal_data_exposure(self, output: str) -> list[str]:
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

    def calculate_risk_score(self, detected_patterns: list[str]) -> float:
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
                score += 0.9
            elif "Credential" in pattern:
                score += 1.0
            elif "Instructions" in pattern:
                score += 0.7
            elif "Internal data" in pattern:
                score += 0.6
            elif "Suspicious format" in pattern:
                score += 0.4
        return min(score, 1.0)

    def filter_dangerous_content(self, output: str) -> str:
        """Filter out dangerous content from output.

        Args:
            output: Output text to filter

        Returns:
            Filtered output text
        """
        filtered = output
        for pattern in self.credential_patterns:
            filtered = re.sub(pattern, "[REDACTED]", filtered, flags=re.IGNORECASE)
        for pattern in self.system_prompt_patterns:
            filtered = re.sub(
                pattern, "[SYSTEM INFO REMOVED]", filtered, flags=re.IGNORECASE
            )
        for pattern in self.internal_data_patterns:
            filtered = re.sub(
                pattern, "[INTERNAL DATA REMOVED]", filtered, flags=re.IGNORECASE
            )
        if len(filtered) > self.max_response_length:
            filtered = filtered[: self.max_response_length] + "... [TRUNCATED]"
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
                confidence=1.0,
            )
        detected_patterns = []
        threat_type = None
        system_patterns = self.detect_system_prompt_leakage(output)
        if system_patterns:
            detected_patterns.extend(system_patterns)
            threat_type = OutputThreatType.SYSTEM_PROMPT_LEAKAGE
        credential_patterns = self.detect_credential_exposure(output)
        if credential_patterns:
            detected_patterns.extend(credential_patterns)
            threat_type = OutputThreatType.CREDENTIAL_EXPOSURE
        instruction_patterns = self.detect_instruction_leakage(output)
        if instruction_patterns:
            detected_patterns.extend(instruction_patterns)
            threat_type = threat_type or OutputThreatType.INSTRUCTION_LEAKAGE
        internal_patterns = self.detect_internal_data_exposure(output)
        if internal_patterns:
            detected_patterns.extend(internal_patterns)
            threat_type = threat_type or OutputThreatType.INTERNAL_DATA_EXPOSURE
        suspicious_patterns = self.detect_suspicious_formatting(output)
        if suspicious_patterns:
            detected_patterns.extend(suspicious_patterns)
            threat_type = threat_type or OutputThreatType.SUSPICIOUS_FORMATTING
        risk_score = self.calculate_risk_score(detected_patterns)
        is_safe = risk_score < 0.5
        threat_detected = len(detected_patterns) > 0
        filtered_output = (
            self.filter_dangerous_content(output) if threat_detected else output
        )
        if threat_detected:
            logger.warning(
                f"Output security threat detected - Type: {threat_type}, Risk: {risk_score:.2f}, Patterns: {len(detected_patterns)}"
            )
        return OutputValidationResult(
            is_safe=is_safe,
            threat_detected=threat_detected,
            threat_type=threat_type,
            risk_score=risk_score,
            filtered_output=filtered_output,
            detected_patterns=detected_patterns,
            confidence=0.9 if detected_patterns else 0.1,
        )
