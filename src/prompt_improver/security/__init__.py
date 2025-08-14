"""Modern security services - Clean architecture with decomposed services.

Provides secure, fail-secure security operations through specialized services
following the 2025 security patterns with zero legacy compatibility layers.
"""

from prompt_improver.security.adversarial_defense import AdversarialDefenseSystem
from prompt_improver.security.authorization import (
    AuthorizationService,
    Permission,
    Role,
)
from prompt_improver.security.differential_privacy import DifferentialPrivacyService
from prompt_improver.security.federated_learning import FederatedLearningService
from prompt_improver.security.input_sanitization import InputSanitizer
from prompt_improver.security.owasp_input_validator import (
    OWASP2025InputValidator,
    ThreatType,
    ValidationResult,
)
from prompt_improver.security.input_validator import (
    InputValidator,
    ValidationError,
    ValidationSchema,
)
from prompt_improver.security.key_manager import (
    KeyRotationConfig,
    UnifiedKeyService,
    get_key_manager,
)
from prompt_improver.security.memory_guard import MemoryGuard, get_memory_guard
from prompt_improver.security.secure_logging import SecureLogger
from prompt_improver.security.services.security_service_facade import (
    SecurityServiceFacade,
    get_security_service_facade,
)
from prompt_improver.security.unified_crypto_manager import (
    CryptoAuditEvent,
    HashAlgorithm,
    RandomType,
    UnifiedCryptoService,
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
    # Core security services (modern architecture)
    "SecurityServiceFacade",
    "get_security_service_facade",
    
    # Specialized security components
    "AdversarialDefenseSystem",
    "AuthorizationService",
    "DifferentialPrivacyService",
    "FederatedLearningService",
    "InputSanitizer",
    "InputValidator",  # Legacy validator (being phased out)
    "OWASP2025InputValidator",  # Modern OWASP-compliant validator
    "MemoryGuard",
    "SecureLogger",
    
    # Security models and enums
    "Permission",
    "Role",
    "ThreatType",
    "ValidationError",
    "ValidationResult",
    "ValidationSchema",
    
    # Cryptographic services
    "CryptoAuditEvent",
    "HashAlgorithm",
    "KeyRotationConfig",
    "RandomType",
    "UnifiedCryptoService",
    "UnifiedKeyService",
    
    # Utility functions
    "decrypt_data",
    "encrypt_data",
    "generate_cache_key",
    "generate_token_bytes",
    "generate_token_hex",
    "generate_token_urlsafe",
    "get_crypto_manager",
    "get_key_manager",
    "get_memory_guard",
    "hash_md5",
    "hash_sha256",
    "secure_compare",
]
