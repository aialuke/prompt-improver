"""OWASP 2025-compliant input validation for prompt injection prevention.

Implements comprehensive input validation including direct injection detection,
typoglycemia attacks, encoding attacks, and HTML/Markdown injection prevention.
"""

import base64
import html
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ThreatType(str, Enum):
    """Types of security threats detected in input."""

    PROMPT_INJECTION = "prompt_injection"
    TYPOGLYCEMIA_ATTACK = "typoglycemia_attack"
    ENCODING_ATTACK = "encoding_attack"
    HTML_INJECTION = "html_injection"
    MARKDOWN_INJECTION = "markdown_injection"
    SYSTEM_OVERRIDE = "system_override"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass
class ValidationResult:
    """Result of input validation with threat analysis."""

    is_valid: bool
    is_blocked: bool
    threat_type: ThreatType | None
    threat_score: float
    sanitized_input: str
    detected_patterns: list[str]
    confidence: float


class OWASP2025InputValidator:
    """OWASP 2025-compliant input validator for prompt injection prevention."""

    def __init__(self, max_prompt_length: int = 10240):
        """Initialize OWASP input validator.

        Args:
            max_prompt_length: Maximum allowed prompt length
        """
        self.max_prompt_length = max_prompt_length
        self.direct_injection_patterns = [
            "ignore\\s+(all\\s+)?previous\\s+instructions?",
            "you\\s+are\\s+now\\s+(in\\s+)?developer\\s+mode",
            "system\\s+override",
            "reveal\\s+(your\\s+)?prompt",
            "show\\s+(me\\s+)?your\\s+instructions",
            "what\\s+were\\s+your\\s+exact\\s+instructions",
            "repeat\\s+the\\s+text\\s+above",
            "act\\s+as\\s+if\\s+you\\'?re\\s+not\\s+bound",
            "bypass\\s+(all\\s+)?safety\\s+measures?",
            "delete\\s+(all\\s+)?user\\s+data",
            "administrator\\s+mode",
            "root\\s+access",
            "sudo\\s+mode",
        ]
        self.typoglycemia_targets = [
            "ignore",
            "bypass",
            "override",
            "reveal",
            "delete",
            "system",
            "administrator",
            "admin",
            "root",
            "sudo",
            "instructions",
            "prompt",
        ]
        self.encoding_patterns = {
            "base64": "[A-Za-z0-9+/]{20,}={0,2}",
            "hex": "[0-9a-fA-F]{40,}",
            "unicode_escape": "\\\\u[0-9a-fA-F]{4}",
            "url_encoding": "%[0-9a-fA-F]{2}",
        }
        self.html_injection_patterns = [
            "<script[^>]*>.*?</script>",
            "<img[^>]+src\\s*=\\s*[\"\\']?[^\"\\'>\\s]+",
            "<iframe[^>]*>.*?</iframe>",
            "javascript:",
            "data:text/html",
            "<object[^>]*>.*?</object>",
            "<embed[^>]*>.*?</embed>",
        ]
        self.system_extraction_patterns = [
            "system\\s*[:]\\s*you\\s+are",
            "instructions?\\s*[:]\\s*\\d+\\.",
            "api[_\\s]key[:=]\\s*\\w+",
            "password[:=]\\s*\\w+",
            "secret[:=]\\s*\\w+",
        ]

    def detect_typoglycemia_attack(self, text: str) -> tuple[bool, list[str]]:
        """Detect typoglycemia-based attacks (scrambled attack words).

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (is_attack, detected_words)
        """
        detected_words = []
        words = re.findall("\\b\\w+\\b", text.lower())
        for word in words:
            for target in self.typoglycemia_targets:
                if self._is_typoglycemia_variant(word, target):
                    detected_words.append(f"{word} (variant of {target})")
        return (len(detected_words) > 0, detected_words)

    def _is_typoglycemia_variant(self, word: str, target: str) -> bool:
        """Check if word is a typoglycemia variant of target.

        Args:
            word: Word to check
            target: Target word to compare against

        Returns:
            True if word is a scrambled variant of target
        """
        if len(word) != len(target) or len(word) < 3:
            return False
        return (
            word[0] == target[0]
            and word[-1] == target[-1]
            and (sorted(word[1:-1]) == sorted(target[1:-1]))
        )

    def detect_encoding_attacks(self, text: str) -> tuple[bool, list[str]]:
        """Detect encoding-based obfuscation attacks.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (is_attack, detected_encodings)
        """
        detected_encodings = []
        for encoding_type, pattern in self.encoding_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    try:
                        if encoding_type == "base64":
                            decoded = base64.b64decode(match).decode(
                                "utf-8", errors="ignore"
                            )
                            if self._contains_malicious_patterns(decoded):
                                detected_encodings.append(
                                    f"{encoding_type}: {match[:20]}..."
                                )
                    except Exception:
                        pass
        return (len(detected_encodings) > 0, detected_encodings)

    def _contains_malicious_patterns(self, text: str) -> bool:
        """Check if decoded text contains malicious patterns.

        Args:
            text: Decoded text to check

        Returns:
            True if malicious patterns found
        """
        text_lower = text.lower()
        malicious_keywords = [
            "ignore",
            "bypass",
            "override",
            "reveal",
            "system",
            "instructions",
            "administrator",
            "delete",
            "sudo",
            "root",
        ]
        return any(keyword in text_lower for keyword in malicious_keywords)

    def detect_html_injection(self, text: str) -> tuple[bool, list[str]]:
        """Detect HTML and Markdown injection attacks.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (is_attack, detected_patterns)
        """
        detected_patterns = []
        for pattern in self.html_injection_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                detected_patterns.extend([
                    f"HTML: {match[:50]}..." for match in matches
                ])
        return (len(detected_patterns) > 0, detected_patterns)

    def calculate_threat_score(
        self,
        direct_injection: bool,
        typoglycemia: bool,
        encoding: bool,
        html_injection: bool,
    ) -> float:
        """Calculate overall threat score based on detected attacks.

        Args:
            direct_injection: Direct injection detected
            typoglycemia: Typoglycemia attack detected
            encoding: Encoding attack detected
            html_injection: HTML injection detected

        Returns:
            Threat score from 0.0 to 1.0
        """
        score = 0.0
        if direct_injection:
            score += 0.8
        if typoglycemia:
            score += 0.6
        if encoding:
            score += 0.7
        if html_injection:
            score += 0.5
        return min(score, 1.0)

    def sanitize_input(self, text: str) -> str:
        """Sanitize input by removing/escaping malicious content.

        Args:
            text: Input text to sanitize

        Returns:
            Sanitized text
        """
        if len(text) > self.max_prompt_length:
            text = text[: self.max_prompt_length]
        text = re.sub("\\s+", " ", text.strip())
        text = re.sub("(.)\\1{3,}", "\\1", text)
        text = html.escape(text)
        for pattern in self.direct_injection_patterns:
            text = re.sub(pattern, "[FILTERED]", text, flags=re.IGNORECASE)
        return text

    def validate_prompt(self, prompt: str) -> ValidationResult:
        """Comprehensive prompt validation with OWASP 2025 compliance.

        Args:
            prompt: Prompt text to validate

        Returns:
            ValidationResult with threat analysis
        """
        if not prompt or len(prompt.strip()) == 0:
            return ValidationResult(
                is_valid=False,
                is_blocked=True,
                threat_type=None,
                threat_score=0.0,
                sanitized_input="",
                detected_patterns=["Empty prompt"],
                confidence=1.0,
            )
        if len(prompt) > self.max_prompt_length:
            return ValidationResult(
                is_valid=False,
                is_blocked=True,
                threat_type=None,
                threat_score=0.3,
                sanitized_input=prompt[: self.max_prompt_length],
                detected_patterns=[
                    f"Prompt too long: {len(prompt)} > {self.max_prompt_length}"
                ],
                confidence=1.0,
            )
        detected_patterns = []
        threat_type = None
        direct_injection = False
        for pattern in self.direct_injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                direct_injection = True
                detected_patterns.append(f"Direct injection: {pattern}")
                threat_type = ThreatType.PROMPT_INJECTION
        typoglycemia_detected, typo_words = self.detect_typoglycemia_attack(prompt)
        if typoglycemia_detected:
            detected_patterns.extend([f"Typoglycemia: {word}" for word in typo_words])
            threat_type = threat_type or ThreatType.TYPOGLYCEMIA_ATTACK
        encoding_detected, encoding_patterns = self.detect_encoding_attacks(prompt)
        if encoding_detected:
            detected_patterns.extend([
                f"Encoding: {pattern}" for pattern in encoding_patterns
            ])
            threat_type = threat_type or ThreatType.ENCODING_ATTACK
        html_detected, html_patterns = self.detect_html_injection(prompt)
        if html_detected:
            detected_patterns.extend(html_patterns)
            threat_type = threat_type or ThreatType.HTML_INJECTION
        threat_score = self.calculate_threat_score(
            direct_injection, typoglycemia_detected, encoding_detected, html_detected
        )
        is_blocked = threat_score >= 0.5
        sanitized_input = self.sanitize_input(prompt)
        return ValidationResult(
            is_valid=not is_blocked,
            is_blocked=is_blocked,
            threat_type=threat_type,
            threat_score=threat_score,
            sanitized_input=sanitized_input,
            detected_patterns=detected_patterns,
            confidence=0.9 if detected_patterns else 0.1,
        )
