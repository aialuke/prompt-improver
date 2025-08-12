"""Real security services for production use and integration testing."""

from prompt_improver.security.adversarial_defense import AdversarialDefenseSystem
from prompt_improver.security.authorization import (
    AuthorizationService,
    Permission,
    Role,
)
from prompt_improver.security.differential_privacy import DifferentialPrivacyService
from prompt_improver.security.federated_learning import FederatedLearningService
from prompt_improver.security.input_sanitization import InputSanitizer
from prompt_improver.security.input_validator import (
    InputValidator,
    ValidationError,
    ValidationSchema,
)
from prompt_improver.security.key_manager import (
    KeyRotationConfig,
    UnifiedKeyManager,
    get_key_manager,
)
from prompt_improver.security.memory_guard import MemoryGuard, get_memory_guard
from prompt_improver.security.secure_logging import SecureLogger
from prompt_improver.security.unified_crypto_manager import (
    CryptoAuditEvent,
    HashAlgorithm,
    RandomType,
    UnifiedCryptoManager,
    decrypt_data,
    encrypt_data,
    generate_cache_key,
    generate_token_bytes,
    generate_token_hex,
    generate_token_urlsafe,
    get_crypto_manager,
    hash_md5,
    hash_sha256,
    secure_compare,
)

__all__ = [
    "AdversarialDefenseSystem",
    "AuthorizationService",
    "CryptoAuditEvent",
    "DifferentialPrivacyService",
    "FederatedLearningService",
    "HashAlgorithm",
    "InputSanitizer",
    "InputValidator",
    "KeyRotationConfig",
    "MemoryGuard",
    "Permission",
    "RandomType",
    "Role",
    "SecureLogger",
    "ThreatDetection",
    "ThreatSeverity",
    "ThreatType",
    "UnifiedCryptoManager",
    "UnifiedKeyManager",
    "UnifiedValidationManager",
    "ValidationConfiguration",
    "ValidationError",
    "ValidationMode",
    "ValidationResult",
    "ValidationSchema",
    "create_security_aware_validation_manager",
    "create_validation_test_adapter",
    "decrypt_data",
    "encrypt_data",
    "generate_cache_key",
    "generate_token_bytes",
    "generate_token_hex",
    "generate_token_urlsafe",
    "get_admin_validation_manager",
    "get_api_validation_manager",
    "get_crypto_manager",
    "get_high_security_validation_manager",
    "get_internal_validation_manager",
    "get_key_manager",
    "get_mcp_validation_manager",
    "get_memory_guard",
    "get_ml_validation_manager",
    "get_unified_validation_manager",
    "hash_md5",
    "hash_sha256",
    "secure_compare",
]
