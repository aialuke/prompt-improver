"""Comprehensive input validation framework for ML context learning.

Provides schema-based validation with security-first design for 2025 standards.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .input_sanitization import InputSanitizer


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(message)


@dataclass
class ValidationSchema:
    """Schema definition for input validation."""

    # String validation
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_chars: Optional[str] = None

    # Numeric validation
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

    # Array validation
    max_array_size: Optional[int] = None
    max_array_elements: Optional[int] = None
    allowed_dtypes: Optional[List[type]] = None

    # General validation
    required: bool = True
    allowed_types: Optional[List[type]] = None
    custom_validator: Optional[callable] = None


class InputValidator:
    """Comprehensive input validator for ML context learning system."""

    def __init__(self):
        self.sanitizer = InputSanitizer()

        # Define validation schemas for context learner inputs
        self.schemas = {
            "user_id": ValidationSchema(
                min_length=3,
                max_length=100,
                pattern=r"^[a-zA-Z0-9_-]+$",
                allowed_types=[str],
                required=True
            ),
            "session_id": ValidationSchema(
                min_length=8,
                max_length=128,
                pattern=r"^[a-zA-Z0-9_-]+$",
                allowed_types=[str],
                required=True
            ),
            "context_data": ValidationSchema(
                allowed_types=[dict],
                required=True,
                custom_validator=self._validate_context_data
            ),
            "numpy_array": ValidationSchema(
                max_array_size=100 * 1024 * 1024,  # 100MB max
                max_array_elements=1000000,  # 1M elements max
                allowed_dtypes=[np.float32, np.float64, np.int32, np.int64],
                allowed_types=[np.ndarray],
                required=True
            ),
            "ml_features": ValidationSchema(
                allowed_types=[list, np.ndarray],
                max_array_elements=10000,
                required=True,
                custom_validator=self._validate_ml_features
            ),
            "privacy_epsilon": ValidationSchema(
                min_value=0.001,
                max_value=10.0,
                allowed_types=[float, int],
                required=True
            ),
            "privacy_delta": ValidationSchema(
                min_value=1e-10,
                max_value=0.1,
                allowed_types=[float, int],
                required=False
            )
        }

    def validate_input(self, field_name: str, value: Any, schema_name: str = None) -> Any:
        """Validate single input field against schema.

        Args:
            field_name: Name of the field being validated
            value: Value to validate
            schema_name: Override schema name (defaults to field_name)

        Returns:
            Validated and potentially sanitized value

        Raises:
            ValidationError: If validation fails
        """
        schema_key = schema_name or field_name

        if schema_key not in self.schemas:
            raise ValidationError(f"No validation schema defined for {schema_key}", field_name, value)

        schema = self.schemas[schema_key]

        # Check if required field is missing
        if value is None:
            if schema.required:
                raise ValidationError(f"Required field {field_name} is missing", field_name, value)
            return None

        # Validate type
        if schema.allowed_types and not isinstance(value, tuple(schema.allowed_types)):
            expected_types = [t.__name__ for t in schema.allowed_types]
            raise ValidationError(
                f"Field {field_name} must be one of types {expected_types}, got {type(value).__name__}",
                field_name, value
            )

        # String validation
        if isinstance(value, str):
            value = self._validate_string(field_name, value, schema)

        # Numeric validation
        elif isinstance(value, (int, float)):
            self._validate_numeric(field_name, value, schema)

        # Array validation
        elif isinstance(value, np.ndarray):
            self._validate_numpy_array(field_name, value, schema)

        # Dict validation
        elif isinstance(value, dict):
            value = self._validate_dict(field_name, value, schema)

        # List validation
        elif isinstance(value, list):
            value = self._validate_list(field_name, value, schema)

        # Custom validation
        if schema.custom_validator:
            try:
                is_valid = schema.custom_validator(value)
                if not is_valid:
                    raise ValidationError(f"Custom validation failed for {field_name}", field_name, value)
            except Exception as e:
                raise ValidationError(f"Custom validation error for {field_name}: {str(e)}", field_name, value)

        return value

    def validate_multiple(self, data: Dict[str, Any], field_schemas: Dict[str, str] = None) -> Dict[str, Any]:
        """Validate multiple fields at once.

        Args:
            data: Dictionary of field names to values
            field_schemas: Optional mapping of field names to schema names

        Returns:
            Dictionary of validated values

        Raises:
            ValidationError: If any validation fails
        """
        validated = {}
        field_schemas = field_schemas or {}

        for field_name, value in data.items():
            schema_name = field_schemas.get(field_name, field_name)
            validated[field_name] = self.validate_input(field_name, value, schema_name)

        return validated

    def _validate_string(self, field_name: str, value: str, schema: ValidationSchema) -> str:
        """Validate string value against schema."""
        # Length validation
        if schema.min_length and len(value) < schema.min_length:
            raise ValidationError(
                f"Field {field_name} must be at least {schema.min_length} characters",
                field_name, value
            )

        if schema.max_length and len(value) > schema.max_length:
            raise ValidationError(
                f"Field {field_name} must be no more than {schema.max_length} characters",
                field_name, value
            )

        # Pattern validation
        if schema.pattern and not re.match(schema.pattern, value):
            raise ValidationError(
                f"Field {field_name} does not match required pattern",
                field_name, value
            )

        # Character validation
        if schema.allowed_chars:
            invalid_chars = set(value) - set(schema.allowed_chars)
            if invalid_chars:
                raise ValidationError(
                    f"Field {field_name} contains invalid characters: {invalid_chars}",
                    field_name, value
                )

        # Security sanitization
        if not self.sanitizer.validate_sql_input(value):
            raise ValidationError(f"Field {field_name} contains SQL injection patterns", field_name, value)

        if not self.sanitizer.validate_command_input(value):
            raise ValidationError(f"Field {field_name} contains command injection patterns", field_name, value)

        # HTML sanitization for certain fields
        if field_name in ["context_data", "prompt", "description"]:
            value = self.sanitizer.sanitize_html_input(value)

        return value

    def _validate_numeric(self, field_name: str, value: Union[int, float], schema: ValidationSchema):
        """Validate numeric value against schema."""
        if schema.min_value is not None and value < schema.min_value:
            raise ValidationError(
                f"Field {field_name} must be at least {schema.min_value}",
                field_name, value
            )

        if schema.max_value is not None and value > schema.max_value:
            raise ValidationError(
                f"Field {field_name} must be no more than {schema.max_value}",
                field_name, value
            )

        # Check for NaN/infinity
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                raise ValidationError(f"Field {field_name} cannot be NaN or infinite", field_name, value)

    def _validate_numpy_array(self, field_name: str, value: np.ndarray, schema: ValidationSchema):
        """Validate numpy array against schema."""
        # Size validation (in bytes)
        if schema.max_array_size and value.nbytes > schema.max_array_size:
            raise ValidationError(
                f"Array {field_name} exceeds maximum size of {schema.max_array_size} bytes",
                field_name, value
            )

        # Element count validation
        if schema.max_array_elements and value.size > schema.max_array_elements:
            raise ValidationError(
                f"Array {field_name} exceeds maximum elements of {schema.max_array_elements}",
                field_name, value
            )

        # Data type validation
        if schema.allowed_dtypes and value.dtype.type not in schema.allowed_dtypes:
            allowed_types = [t.__name__ for t in schema.allowed_dtypes]
            raise ValidationError(
                f"Array {field_name} has invalid dtype {value.dtype}, allowed: {allowed_types}",
                field_name, value
            )

        # Use existing ML validation
        if not self.sanitizer.validate_ml_input_data(value):
            raise ValidationError(f"Array {field_name} contains invalid ML data", field_name, value)

    def _validate_dict(self, field_name: str, value: dict, schema: ValidationSchema) -> dict:
        """Validate dictionary value against schema."""
        # Sanitize the dictionary
        sanitized = self.sanitizer.sanitize_json_input(value)

        # Additional validation for specific dict types
        if field_name == "context_data":
            return self._validate_context_data(sanitized)

        return sanitized

    def _validate_list(self, field_name: str, value: list, schema: ValidationSchema) -> list:
        """Validate list value against schema."""
        if schema.max_array_elements and len(value) > schema.max_array_elements:
            raise ValidationError(
                f"List {field_name} exceeds maximum length of {schema.max_array_elements}",
                field_name, value
            )

        # Validate each element if it's a string
        sanitized = []
        for i, item in enumerate(value):
            if isinstance(item, str):
                sanitized.append(self.sanitizer.sanitize_html_input(item))
            else:
                sanitized.append(item)

        return sanitized

    def _validate_context_data(self, context_data: dict) -> bool:
        """Custom validator for context data structure."""
        if not isinstance(context_data, dict):
            return False

        # Required fields check
        required_fields = ["domain", "project_type"]
        for field in required_fields:
            if field not in context_data:
                return False

        # Validate field values
        if "domain" in context_data:
            domain = context_data["domain"]
            if not isinstance(domain, str) or len(domain) > 100:
                return False

        if "project_type" in context_data:
            project_type = context_data["project_type"]
            allowed_types = ["web", "mobile", "api", "data", "ml", "other"]
            if project_type not in allowed_types:
                return False

        return True

    def _validate_ml_features(self, features: Union[list, np.ndarray]) -> bool:
        """Custom validator for ML features."""
        if isinstance(features, np.ndarray):
            return self.sanitizer.validate_ml_input_data(features)
        elif isinstance(features, list):
            # Check each feature value
            for feature in features:
                if not isinstance(feature, (int, float)):
                    return False
                if np.isnan(feature) or np.isinf(feature):
                    return False
                if abs(feature) > 1e6:  # Reasonable range for features
                    return False
            return True
        return False

    def create_hash_id(self, *args) -> str:
        """Create a secure hash ID from input arguments."""
        combined = "|".join(str(arg) for arg in args)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]