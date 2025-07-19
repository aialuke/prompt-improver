"""Real input sanitization service for production use and integration testing."""

import html
import re
import math
from typing import Any, Dict, List, Union
import numpy as np


class InputSanitizer:
    """Real input sanitization service that implements comprehensive input validation."""
    
    def __init__(self):
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
            elif isinstance(data, (list, tuple)):
                if len(data) == 0:
                    return False
                
                for item in data:
                    if not self.validate_ml_input_data(item):
                        return False
                
                return True
            
            # Handle individual numbers
            elif isinstance(data, (int, float)):
                if math.isnan(data) or math.isinf(data):
                    return False
                
                if abs(data) > 1e10:
                    return False
                
                return True
            
            # Handle dictionaries (for structured data)
            elif isinstance(data, dict):
                for key, value in data.items():
                    if not isinstance(key, str) or not self.validate_ml_input_data(value):
                        return False
                
                return True
            
            # Reject other types
            else:
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
    
    def validate_password_strength(self, password: str) -> Dict[str, Union[bool, List[str]]]:
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
    
    def sanitize_json_input(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
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