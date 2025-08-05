"""Real security services for production use and integration testing."""

from .adversarial_defense import AdversarialDefenseSystem
from .authorization import AuthorizationService, Permission, Role
from .differential_privacy import DifferentialPrivacyService
from .federated_learning import FederatedLearningService
from .input_sanitization import InputSanitizer
from .input_validator import InputValidator, ValidationError, ValidationSchema
# Temporarily commented out to fix circular import
# from .unified_validation_manager import (
#     UnifiedValidationManager,
#     ValidationMode,
#     ThreatType,
#     ThreatSeverity,
#     ValidationResult,
#     ValidationConfiguration,
#     ThreatDetection,
#     get_unified_validation_manager,
#     get_mcp_validation_manager,
#     get_api_validation_manager,
#     get_internal_validation_manager,
#     get_admin_validation_manager,
#     get_high_security_validation_manager,
#     get_ml_validation_manager,
#     create_security_aware_validation_manager,
#     create_validation_test_adapter
# )
from .memory_guard import MemoryGuard, get_memory_guard
from .key_manager import UnifiedKeyManager, KeyRotationConfig, get_key_manager
from .unified_crypto_manager import (
    UnifiedCryptoManager,
    HashAlgorithm,
    RandomType,
    CryptoAuditEvent,
    get_crypto_manager,
    hash_sha256,
    hash_md5,
    generate_token_bytes,
    generate_token_hex,
    generate_token_urlsafe,
    generate_cache_key,
    encrypt_data,
    decrypt_data,
    secure_compare
)
from .secure_logging import SecureLogger

__all__ = [
    "AuthorizationService",
    "Permission",
    "Role",
    "DifferentialPrivacyService",
    "AdversarialDefenseSystem",
    "FederatedLearningService",
    "InputSanitizer",
    "InputValidator",
    "ValidationError",
    "ValidationSchema",
    "UnifiedValidationManager",
    "ValidationMode",
    "ThreatType",
    "ThreatSeverity",
    "ValidationResult",
    "ValidationConfiguration",
    "ThreatDetection",
    "get_unified_validation_manager",
    "get_mcp_validation_manager",
    "get_api_validation_manager",
    "get_internal_validation_manager",
    "get_admin_validation_manager",
    "get_high_security_validation_manager",
    "get_ml_validation_manager",
    "create_security_aware_validation_manager",
    "create_validation_test_adapter",
    "MemoryGuard",
    "get_memory_guard",
    "UnifiedKeyManager", 
    "KeyRotationConfig",
    "get_key_manager",
    "UnifiedCryptoManager",
    "HashAlgorithm",
    "RandomType", 
    "CryptoAuditEvent",
    "get_crypto_manager",
    "hash_sha256",
    "hash_md5",
    "generate_token_bytes",
    "generate_token_hex",
    "generate_token_urlsafe",
    "generate_cache_key",
    "encrypt_data",
    "decrypt_data",
    "secure_compare",
    "SecureLogger",
]