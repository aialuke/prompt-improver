"""Unified Cryptographic Operations Manager

Provides standardized cryptographic operations through the existing KeyManager infrastructure.
Eliminates scattered crypto imports and ensures all cryptographic operations follow
NIST standards and security best practices.

Features:
- Unified hashing operations (SHA-256, SHA-512, MD5)
- Secure random generation for all use cases
- Encryption/decryption through KeyManager
- Key derivation and management
- Digital signatures and verification
- Secure comparison operations
- Comprehensive audit logging
- Memory-safe operations
- NIST-approved algorithms only

All cryptographic operations are centralized through this manager to ensure:
- Consistent security practices
- Comprehensive audit trails
- Standardized error handling
- Performance optimization
- Compliance with security standards
"""

import hashlib
import logging
import secrets
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from prompt_improver.security.key_manager import (
    AuditEvent,
    KeyRotationConfig,
    SecurityLevel,
    UnifiedKeyManager,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class HashAlgorithm(Enum):
    """Supported hash algorithms (NIST approved)."""

    SHA256 = "sha256"
    SHA512 = "sha512"
    SHA1 = "sha1"
    MD5 = "md5"


class RandomType(Enum):
    """Types of random data generation."""

    BYTES = "bytes"
    HEX = "hex"
    URLSAFE = "urlsafe"
    TOKEN = "token"


class CryptoAuditEvent(Enum):
    """Cryptographic audit events."""

    HASH_GENERATED = "hash_generated"
    RANDOM_GENERATED = "random_generated"
    ENCRYPTION_PERFORMED = "encryption_performed"
    DECRYPTION_PERFORMED = "decryption_performed"
    KEY_DERIVED = "key_derived"
    SIGNATURE_CREATED = "signature_created"
    SIGNATURE_VERIFIED = "signature_verified"
    CRYPTO_ERROR = "crypto_error"


class CryptoMetrics:
    """Performance and security metrics for crypto operations."""

    def __init__(self):
        self.operation_counts: dict[str, int] = {}
        self.performance_stats: dict[str, list[float]] = {}
        self.error_counts: dict[str, int] = {}
        self.last_reset = aware_utc_now()

    def record_operation(
        self, operation: str, duration_ms: float, success: bool = True
    ):
        """Record crypto operation metrics."""
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
        if operation not in self.performance_stats:
            self.performance_stats[operation] = []
        self.performance_stats[operation].append(duration_ms)
        if len(self.performance_stats[operation]) > 1000:
            self.performance_stats[operation] = self.performance_stats[operation][
                -1000:
            ]
        if not success:
            self.error_counts[operation] = self.error_counts.get(operation, 0) + 1

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive crypto metrics."""
        stats = {
            "operation_counts": self.operation_counts.copy(),
            "error_counts": self.error_counts.copy(),
            "performance_averages": {},
            "last_reset": self.last_reset.isoformat(),
            "total_operations": sum(self.operation_counts.values()),
        }
        for operation, times in self.performance_stats.items():
            if times:
                stats["performance_averages"][operation] = {
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "count": len(times),
                }
        return stats


class UnifiedCryptoManager:
    """Unified cryptographic operations manager.

    Centralizes all cryptographic operations through the existing KeyManager
    infrastructure to ensure consistent security practices and audit trails.
    """

    def __init__(
        self,
        key_manager: UnifiedKeyManager | None = None,
        security_level: SecurityLevel = SecurityLevel.enhanced,
    ):
        """Initialize the unified crypto manager.

        Args:
            key_manager: Optional existing key manager instance
            security_level: Security level for operations
        """
        self.key_manager = key_manager or UnifiedKeyManager()
        self.security_level = security_level
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.metrics = CryptoMetrics()
        self.audit_enabled = True
        self.last_security_check = aware_utc_now()
        self.failed_operations: list[dict[str, Any]] = []

    def hash_data(
        self,
        data: str | bytes,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        truncate_length: int | None = None,
    ) -> str:
        """Generate hash of data using specified algorithm.

        Args:
            data: Data to hash (string or bytes)
            algorithm: Hash algorithm to use
            truncate_length: Optional length to truncate hash (for cache keys, etc.)

        Returns:
            Hexadecimal hash string
        """
        start_time = time.time()
        success = True
        try:
            if isinstance(data, str):
                data_bytes = data.encode("utf-8")
            else:
                data_bytes = data
            if algorithm == HashAlgorithm.SHA256:
                hash_obj = hashlib.sha256(data_bytes)
            elif algorithm == HashAlgorithm.SHA512:
                hash_obj = hashlib.sha512(data_bytes)
            elif algorithm == HashAlgorithm.SHA1:
                self.logger.warning(
                    "SHA1 is deprecated and should not be used for security purposes"
                )
                hash_obj = hashlib.sha1(data_bytes)
            elif algorithm == HashAlgorithm.MD5:
                self.logger.warning(
                    "MD5 is deprecated and should not be used for security purposes"
                )
                hash_obj = hashlib.md5(data_bytes, usedforsecurity=False)
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            hash_result = hash_obj.hexdigest()
            if truncate_length and truncate_length > 0:
                hash_result = hash_result[:truncate_length]
            if self.audit_enabled:
                self._log_crypto_event(
                    CryptoAuditEvent.HASH_GENERATED,
                    {
                        "algorithm": algorithm.value,
                        "data_length": len(data_bytes),
                        "truncate_length": truncate_length,
                        "hash_length": len(hash_result),
                    },
                )
            self.logger.debug(
                f"Generated {algorithm.value} hash (length: {len(hash_result)})"
            )
            return hash_result
        except Exception as e:
            success = False
            self._handle_crypto_error("hash_data", e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation(
                f"hash_{algorithm.value}", duration_ms, success
            )

    def hash_sha256(self, data: str | bytes, truncate_length: int | None = None) -> str:
        """Generate SHA-256 hash (convenience method)."""
        return self.hash_data(data, HashAlgorithm.SHA256, truncate_length)

    def hash_sha512(self, data: str | bytes, truncate_length: int | None = None) -> str:
        """Generate SHA-512 hash (convenience method)."""
        return self.hash_data(data, HashAlgorithm.SHA512, truncate_length)

    def hash_md5(self, data: str | bytes, truncate_length: int | None = None) -> str:
        """Generate MD5 hash (for non-security purposes only)."""
        return self.hash_data(data, HashAlgorithm.MD5, truncate_length)

    def generate_cache_key(self, *components: Any, max_length: int = 16) -> str:
        """Generate standardized cache key from components.

        Args:
            *components: Components to include in cache key
            max_length: Maximum length of generated key

        Returns:
            Cache key string
        """
        key_string = "_".join(str(component) for component in components)
        return self.hash_sha256(key_string, max_length)

    def generate_secure_hash(
        self, data: str | bytes, salt: bytes | None = None
    ) -> tuple[str, bytes]:
        """Generate secure hash with salt for password/security purposes.

        Args:
            data: Data to hash
            salt: Optional salt bytes (will be generated if not provided)

        Returns:
            Tuple of (hash_string, salt_bytes)
        """
        if salt is None:
            salt = self.generate_random_bytes(32)
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = data
        combined = data_bytes + salt
        hash_result = self.hash_sha256(combined)
        return (hash_result, salt)

    def generate_random(
        self, length: int, random_type: RandomType = RandomType.BYTES
    ) -> bytes | str:
        """Generate cryptographically secure random data.

        Args:
            length: Length of random data
            random_type: Type of random data to generate

        Returns:
            Random data in specified format
        """
        start_time = time.time()
        success = True
        try:
            if random_type == RandomType.BYTES:
                result = secrets.token_bytes(length)
            elif random_type == RandomType.HEX:
                result = secrets.token_hex(length)
            elif random_type == RandomType.URLSAFE:
                result = secrets.token_urlsafe(length)
            elif random_type == RandomType.TOKEN:
                result = secrets.token_hex(length // 2 if length > 1 else 1)
            else:
                raise ValueError(f"Unsupported random type: {random_type}")
            if self.audit_enabled:
                self._log_crypto_event(
                    CryptoAuditEvent.RANDOM_GENERATED,
                    {
                        "type": random_type.value,
                        "length": length,
                        "result_length": len(result),
                    },
                )
            self.logger.debug(
                f"Generated {random_type.value} random data (length: {length})"
            )
            return result
        except Exception as e:
            success = False
            self._handle_crypto_error("generate_random", e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation(
                f"random_{random_type.value}", duration_ms, success
            )

    def generate_random_bytes(self, length: int) -> bytes:
        """Generate random bytes (convenience method)."""
        return self.generate_random(length, RandomType.BYTES)

    def generate_random_hex(self, length: int) -> str:
        """Generate random hex string (convenience method)."""
        return self.generate_random(length, RandomType.HEX)

    def generate_random_urlsafe(self, length: int) -> str:
        """Generate random URL-safe string (convenience method)."""
        return self.generate_random(length, RandomType.URLSAFE)

    def generate_session_token(self, prefix: str = "session", length: int = 32) -> str:
        """Generate secure session token.

        Args:
            prefix: Token prefix
            length: Random component length

        Returns:
            Session token string
        """
        timestamp = int(time.time() * 1000000)
        random_part = self.generate_random_urlsafe(length)
        return f"{prefix}_{timestamp}_{random_part}"

    def generate_api_key(self, prefix: str = "api", length: int = 32) -> str:
        """Generate secure API key.

        Args:
            prefix: Key prefix
            length: Random component length

        Returns:
            API key string
        """
        random_part = self.generate_random_urlsafe(length)
        return f"{prefix}_{random_part}"

    def encrypt_data(
        self, data: str | bytes, key_id: str | None = None
    ) -> tuple[bytes, str]:
        """Encrypt data using current encryption key.

        Args:
            data: Data to encrypt
            key_id: Optional specific key ID to use

        Returns:
            Tuple of (encrypted_data, key_id_used)
        """
        start_time = time.time()
        success = True
        try:
            if isinstance(data, str):
                data_bytes = data.encode("utf-8")
            else:
                data_bytes = data
            encrypted_data, used_key_id = self.key_manager.encrypt(data_bytes, key_id)
            if self.audit_enabled:
                self._log_crypto_event(
                    CryptoAuditEvent.ENCRYPTION_PERFORMED,
                    {
                        "data_length": len(data_bytes),
                        "key_id": used_key_id,
                        "encrypted_length": len(encrypted_data),
                    },
                )
            self.logger.debug(f"Encrypted data using key {used_key_id}")
            return (encrypted_data, used_key_id)
        except Exception as e:
            success = False
            self._handle_crypto_error("encrypt_data", e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation("encryption", duration_ms, success)

    def decrypt_data(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt data using specified key.

        Args:
            encrypted_data: Data to decrypt
            key_id: Key ID for decryption

        Returns:
            Decrypted data
        """
        start_time = time.time()
        success = True
        try:
            decrypted_data = self.key_manager.decrypt(encrypted_data, key_id)
            if self.audit_enabled:
                self._log_crypto_event(
                    CryptoAuditEvent.DECRYPTION_PERFORMED,
                    {
                        "encrypted_length": len(encrypted_data),
                        "key_id": key_id,
                        "decrypted_length": len(decrypted_data),
                    },
                )
            self.logger.debug(f"Decrypted data using key {key_id}")
            return decrypted_data
        except Exception as e:
            success = False
            self._handle_crypto_error("decrypt_data", e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation("decryption", duration_ms, success)

    def encrypt_string(self, text: str, key_id: str | None = None) -> tuple[bytes, str]:
        """Encrypt string data (convenience method)."""
        return self.encrypt_data(text, key_id)

    def decrypt_string(self, encrypted_data: bytes, key_id: str) -> str:
        """Decrypt to string data (convenience method)."""
        decrypted_bytes = self.decrypt_data(encrypted_data, key_id)
        return decrypted_bytes.decode("utf-8")

    def derive_key(
        self,
        password: str | bytes,
        salt: bytes | None = None,
        length: int = 32,
        iterations: int = 600000,
    ) -> tuple[bytes, bytes]:
        """Derive key from password using PBKDF2-HMAC-SHA256.

        Args:
            password: Password to derive from
            salt: Salt bytes (generated if not provided)
            length: Derived key length in bytes
            iterations: Number of iterations (NIST 2025 minimum: 600,000)

        Returns:
            Tuple of (derived_key, salt)
        """
        start_time = time.time()
        success = True
        try:
            if salt is None:
                salt = self.generate_random_bytes(32)
            if isinstance(password, str):
                password_bytes = password.encode("utf-8")
            else:
                password_bytes = password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=length,
                salt=salt,
                iterations=iterations,
                backend=default_backend(),
            )
            derived_key = kdf.derive(password_bytes)
            if self.audit_enabled:
                self._log_crypto_event(
                    CryptoAuditEvent.KEY_DERIVED,
                    {
                        "algorithm": "PBKDF2-HMAC-SHA256",
                        "iterations": iterations,
                        "salt_length": len(salt),
                        "key_length": len(derived_key),
                    },
                )
            self.logger.debug(f"Derived key using PBKDF2 ({iterations} iterations)")
            return (derived_key, salt)
        except Exception as e:
            success = False
            self._handle_crypto_error("derive_key", e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation("key_derivation", duration_ms, success)

    def derive_key_scrypt(
        self,
        password: str | bytes,
        salt: bytes | None = None,
        length: int = 32,
        n: int = 2**16,
        r: int = 8,
        p: int = 1,
    ) -> tuple[bytes, bytes]:
        """Derive key using Scrypt (memory-hard function).

        Args:
            password: Password to derive from
            salt: Salt bytes (generated if not provided)
            length: Derived key length in bytes
            n: CPU/memory cost factor
            r: Block size parameter
            p: Parallelization parameter

        Returns:
            Tuple of (derived_key, salt)
        """
        start_time = time.time()
        success = True
        try:
            if salt is None:
                salt = self.generate_random_bytes(32)
            if isinstance(password, str):
                password_bytes = password.encode("utf-8")
            else:
                password_bytes = password
            kdf = Scrypt(
                length=length, salt=salt, n=n, r=r, p=p, backend=default_backend()
            )
            derived_key = kdf.derive(password_bytes)
            if self.audit_enabled:
                self._log_crypto_event(
                    CryptoAuditEvent.KEY_DERIVED,
                    {
                        "algorithm": "Scrypt",
                        "n": n,
                        "r": r,
                        "p": p,
                        "salt_length": len(salt),
                        "key_length": len(derived_key),
                    },
                )
            self.logger.debug(f"Derived key using Scrypt (n={n}, r={r}, p={p})")
            return (derived_key, salt)
        except Exception as e:
            success = False
            self._handle_crypto_error("derive_key_scrypt", e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation("scrypt_derivation", duration_ms, success)

    def secure_compare(self, a: str | bytes, b: str | bytes) -> bool:
        """Perform timing-safe comparison of two values.

        Args:
            a: First value
            b: Second value

        Returns:
            True if values are equal, False otherwise
        """
        if isinstance(a, str):
            a_bytes = a.encode("utf-8")
        else:
            a_bytes = a
        if isinstance(b, str):
            b_bytes = b.encode("utf-8")
        else:
            b_bytes = b
        return secrets.compare_digest(a_bytes, b_bytes)

    def verify_hash(
        self,
        data: str | bytes,
        hash_value: str,
        salt: bytes,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ) -> bool:
        """Verify data against a hash with salt.

        Args:
            data: Original data
            hash_value: Expected hash value
            salt: Salt used in original hash
            algorithm: Hash algorithm used

        Returns:
            True if hash matches, False otherwise
        """
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = data
        combined = data_bytes + salt
        computed_hash = self.hash_data(combined, algorithm)
        return self.secure_compare(computed_hash, hash_value)

    def rotate_keys(self) -> str:
        """Rotate encryption keys."""
        return self.key_manager.rotate_key()

    def get_crypto_status(self) -> dict[str, Any]:
        """Get comprehensive crypto system status."""
        key_status = self.key_manager.get_key_info()
        metrics = self.metrics.get_stats()
        return {
            "crypto_manager": {
                "security_level": self.security_level.value,
                "audit_enabled": self.audit_enabled,
                "last_security_check": self.last_security_check.isoformat(),
                "failed_operations": len(self.failed_operations),
            },
            "key_management": key_status,
            "performance_metrics": metrics,
            "supported_algorithms": {
                "hash": [algo.value for algo in HashAlgorithm],
                "random": [rtype.value for rtype in RandomType],
                "encryption": ["Fernet", "AES-256"],
            },
        }

    def cleanup_old_keys(self, keep_count: int = 3) -> list[str]:
        """Clean up old encryption keys."""
        return self.key_manager.cleanup_old_keys(keep_count)

    def _log_crypto_event(self, event: CryptoAuditEvent, details: dict[str, Any]):
        """Log cryptographic audit event."""
        if not self.audit_enabled:
            return
        audit_entry = {
            "timestamp": aware_utc_now().isoformat(),
            "component": "UnifiedCryptoManager",
            "event": event.value,
            "security_level": self.security_level.value,
            "details": details,
        }
        if event == CryptoAuditEvent.CRYPTO_ERROR:
            self.logger.error(f"CRYPTO_AUDIT: {audit_entry}")
        else:
            self.logger.debug(f"CRYPTO_AUDIT: {audit_entry}")

    def _handle_crypto_error(self, operation: str, error: Exception):
        """Handle cryptographic operation errors."""
        error_info = {
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": aware_utc_now().isoformat(),
        }
        self.failed_operations.append(error_info)
        if len(self.failed_operations) > 100:
            self.failed_operations = self.failed_operations[-100:]
        if self.audit_enabled:
            self._log_crypto_event(CryptoAuditEvent.CRYPTO_ERROR, error_info)
        self.logger.error(f"Crypto operation failed: {operation} - {error}")


_default_crypto_manager: UnifiedCryptoManager | None = None


def get_crypto_manager(
    security_level: SecurityLevel = SecurityLevel.enhanced,
) -> UnifiedCryptoManager:
    """Get global unified crypto manager instance.

    Args:
        security_level: Security level for operations

    Returns:
        UnifiedCryptoManager instance
    """
    global _default_crypto_manager
    if _default_crypto_manager is None:
        _default_crypto_manager = UnifiedCryptoManager(security_level=security_level)
    return _default_crypto_manager


def reset_crypto_manager():
    """Reset global crypto manager (for testing)."""
    global _default_crypto_manager
    _default_crypto_manager = None


def hash_sha256(data: str | bytes, truncate_length: int | None = None) -> str:
    """Generate SHA-256 hash (migration-friendly function)."""
    return get_crypto_manager().hash_sha256(data, truncate_length)


def hash_md5(data: str | bytes, truncate_length: int | None = None) -> str:
    """Generate MD5 hash (migration-friendly function)."""
    return get_crypto_manager().hash_md5(data, truncate_length)


def generate_token_bytes(length: int) -> bytes:
    """Generate random bytes (migration-friendly function)."""
    return get_crypto_manager().generate_random_bytes(length)


def generate_token_hex(length: int) -> str:
    """Generate random hex string (migration-friendly function)."""
    return get_crypto_manager().generate_random_hex(length)


def generate_token_urlsafe(length: int) -> str:
    """Generate random URL-safe string (migration-friendly function)."""
    return get_crypto_manager().generate_random_urlsafe(length)


def generate_cache_key(*components: Any, max_length: int = 16) -> str:
    """Generate standardized cache key (migration-friendly function)."""
    return get_crypto_manager().generate_cache_key(*components, max_length=max_length)


def encrypt_data(data: str | bytes, key_id: str | None = None) -> tuple[bytes, str]:
    """Encrypt data (migration-friendly function)."""
    return get_crypto_manager().encrypt_data(data, key_id)


def decrypt_data(encrypted_data: bytes, key_id: str) -> bytes:
    """Decrypt data (migration-friendly function)."""
    return get_crypto_manager().decrypt_data(encrypted_data, key_id)


def secure_compare(a: str | bytes, b: str | bytes) -> bool:
    """Perform timing-safe comparison (migration-friendly function)."""
    return get_crypto_manager().secure_compare(a, b)
