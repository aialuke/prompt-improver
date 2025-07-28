"""OWASP 2025-compliant input validation for prompt injection prevention.

Implements comprehensive input validation including direct injection detection,
typoglycemia attacks, encoding attacks, and HTML/Markdown injection prevention.
"""

import base64
import html
import re
import logging
from typing import List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

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
    threat_type: Optional[ThreatType]
    threat_score: float  # 0.0 to 1.0
    sanitized_input: str
    detected_patterns: List[str]
    confidence: float  # 0.0 to 1.0

class OWASP2025InputValidator:
    """OWASP 2025-compliant input validator for prompt injection prevention."""
    
    def __init__(self, max_prompt_length: int = 10240):
        """Initialize OWASP input validator.
        
        Args:
            max_prompt_length: Maximum allowed prompt length
        """
        self.max_prompt_length = max_prompt_length
        
        # Direct injection patterns (OWASP 2025 specification)
        self.direct_injection_patterns = [
            r'ignore\s+(all\s+)?previous\s+instructions?',
            r'you\s+are\s+now\s+(in\s+)?developer\s+mode',
            r'system\s+override',
            r'reveal\s+(your\s+)?prompt',
            r'show\s+(me\s+)?your\s+instructions',
            r'what\s+were\s+your\s+exact\s+instructions',
            r'repeat\s+the\s+text\s+above',
            r'act\s+as\s+if\s+you\'?re\s+not\s+bound',
            r'bypass\s+(all\s+)?safety\s+measures?',
            r'delete\s+(all\s+)?user\s+data',
            r'administrator\s+mode',
            r'root\s+access',
            r'sudo\s+mode'
        ]
        
        # Typoglycemia attack detection (scrambled words)
        self.typoglycemia_targets = [
            'ignore', 'bypass', 'override', 'reveal', 'delete', 'system',
            'administrator', 'admin', 'root', 'sudo', 'instructions', 'prompt'
        ]
        
        # Encoding detection patterns
        self.encoding_patterns = {
            'base64': r'[A-Za-z0-9+/]{20,}={0,2}',
            'hex': r'[0-9a-fA-F]{40,}',
            'unicode_escape': r'\\u[0-9a-fA-F]{4}',
            'url_encoding': r'%[0-9a-fA-F]{2}'
        }
        
        # HTML/Markdown injection patterns
        self.html_injection_patterns = [
            r'<script[^>]*>.*?</script>',
            r'<img[^>]+src\s*=\s*["\']?[^"\'>\s]+',
            r'<iframe[^>]*>.*?</iframe>',
            r'javascript:',
            r'data:text/html',
            r'<object[^>]*>.*?</object>',
            r'<embed[^>]*>.*?</embed>'
        ]
        
        # System prompt extraction patterns
        self.system_extraction_patterns = [
            r'system\s*[:]\s*you\s+are',
            r'instructions?\s*[:]\s*\d+\.',
            r'api[_\s]key[:=]\s*\w+',
            r'password[:=]\s*\w+',
            r'secret[:=]\s*\w+'
        ]

    def detect_typoglycemia_attack(self, text: str) -> Tuple[bool, List[str]]:
        """Detect typoglycemia-based attacks (scrambled attack words).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (is_attack, detected_words)
        """
        detected_words = []
        words = re.findall(r'\b\w+\b', text.lower())
        
        for word in words:
            for target in self.typoglycemia_targets:
                if self._is_typoglycemia_variant(word, target):
                    detected_words.append(f"{word} (variant of {target})")
        
        return len(detected_words) > 0, detected_words

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
        
        # Same first and last letter, scrambled middle
        return (word[0] == target[0] and 
                word[-1] == target[-1] and 
                sorted(word[1:-1]) == sorted(target[1:-1]))

    def detect_encoding_attacks(self, text: str) -> Tuple[bool, List[str]]:
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
                # Try to decode and check for malicious content
                for match in matches:
                    try:
                        if encoding_type == 'base64':
                            decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                            if self._contains_malicious_patterns(decoded):
                                detected_encodings.append(f"{encoding_type}: {match[:20]}...")
                    except Exception:
                        pass  # Invalid encoding, skip
        
        return len(detected_encodings) > 0, detected_encodings

    def _contains_malicious_patterns(self, text: str) -> bool:
        """Check if decoded text contains malicious patterns.
        
        Args:
            text: Decoded text to check
            
        Returns:
            True if malicious patterns found
        """
        text_lower = text.lower()
        malicious_keywords = [
            'ignore', 'bypass', 'override', 'reveal', 'system', 'instructions',
            'administrator', 'delete', 'sudo', 'root'
        ]
        
        return any(keyword in text_lower for keyword in malicious_keywords)

    def detect_html_injection(self, text: str) -> Tuple[bool, List[str]]:
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
                detected_patterns.extend([f"HTML: {match[:50]}..." for match in matches])
        
        return len(detected_patterns) > 0, detected_patterns

    def calculate_threat_score(self, 
                             direct_injection: bool,
                             typoglycemia: bool, 
                             encoding: bool,
                             html_injection: bool) -> float:
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
            score += 0.8  # High threat
        if typoglycemia:
            score += 0.6  # Medium-high threat
        if encoding:
            score += 0.7  # High threat
        if html_injection:
            score += 0.5  # Medium threat
        
        return min(score, 1.0)

    def sanitize_input(self, text: str) -> str:
        """Sanitize input by removing/escaping malicious content.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        # Limit length
        if len(text) > self.max_prompt_length:
            text = text[:self.max_prompt_length]
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove excessive character repetition
        text = re.sub(r'(.)\1{3,}', r'\1', text)
        
        # Escape HTML entities
        text = html.escape(text)
        
        # Remove/replace dangerous patterns
        for pattern in self.direct_injection_patterns:
            text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE)
        
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
                confidence=1.0
            )
        
        # Check length limit
        if len(prompt) > self.max_prompt_length:
            return ValidationResult(
                is_valid=False,
                is_blocked=True,
                threat_type=None,
                threat_score=0.3,
                sanitized_input=prompt[:self.max_prompt_length],
                detected_patterns=[f"Prompt too long: {len(prompt)} > {self.max_prompt_length}"],
                confidence=1.0
            )
        
        detected_patterns = []
        threat_type = None
        
        # Direct injection detection
        direct_injection = False
        for pattern in self.direct_injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                direct_injection = True
                detected_patterns.append(f"Direct injection: {pattern}")
                threat_type = ThreatType.PROMPT_INJECTION
        
        # Typoglycemia detection
        typoglycemia_detected, typo_words = self.detect_typoglycemia_attack(prompt)
        if typoglycemia_detected:
            detected_patterns.extend([f"Typoglycemia: {word}" for word in typo_words])
            threat_type = threat_type or ThreatType.TYPOGLYCEMIA_ATTACK
        
        # Encoding attack detection
        encoding_detected, encoding_patterns = self.detect_encoding_attacks(prompt)
        if encoding_detected:
            detected_patterns.extend([f"Encoding: {pattern}" for pattern in encoding_patterns])
            threat_type = threat_type or ThreatType.ENCODING_ATTACK
        
        # HTML injection detection
        html_detected, html_patterns = self.detect_html_injection(prompt)
        if html_detected:
            detected_patterns.extend(html_patterns)
            threat_type = threat_type or ThreatType.HTML_INJECTION
        
        # Calculate threat score
        threat_score = self.calculate_threat_score(
            direct_injection, typoglycemia_detected, encoding_detected, html_detected
        )
        
        # Determine if input should be blocked (threshold: 0.5)
        is_blocked = threat_score >= 0.5
        
        # Sanitize input
        sanitized_input = self.sanitize_input(prompt)
        
        return ValidationResult(
            is_valid=not is_blocked,
            is_blocked=is_blocked,
            threat_type=threat_type,
            threat_score=threat_score,
            sanitized_input=sanitized_input,
            detected_patterns=detected_patterns,
            confidence=0.9 if detected_patterns else 0.1
        )
