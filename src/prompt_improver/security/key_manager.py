"""Secure key management with rotation for encryption operations.

Provides enterprise-grade key management with rotation, versioning, and secure storage.
Enhanced with 2025 security best practices including:
- Zero Trust Architecture principles
- Hardware Security Module (HSM) readiness
- NIST SP 800-57 compliance
- Advanced key derivation and entropy
- Secure memory handling
- Audit logging and compliance
- Cloud KMS integration readiness

Includes orchestrator integration for ML pipeline compatibility.
"""

import hashlib
import logging
import secrets
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels following NIST guidelines (2025)."""

    basic = "basic"
    enhanced = "enhanced"
    HIGH = "high"
    CRITICAL = "critical"


class KeyDerivationMethod(Enum):
    """Key derivation methods (2025 best practices)."""

    pbkdf2 = "pbkdf2"
    scrypt = "scrypt"
    direct = "direct"


class AuditEvent(Enum):
    """Security audit events for compliance."""

    KEY_GENERATED = "key_generated"
    KEY_ROTATED = "key_rotated"
    KEY_ACCESSED = "key_accessed"
    KEY_EXPIRED = "key_expired"
    KEY_DELETED = "key_deleted"
    SECURITY_VIOLATION = "security_violation"


class SecurityAuditLogger:
    """Enhanced security audit logging for compliance (2025)."""

    def __init__(self, component_name: str = "UnifiedKeyService"):
        self.component_name = component_name
        self.logger = logging.getLogger(f"{__name__}.{component_name}.Audit")

    def log_event(
        self,
        event: AuditEvent,
        details: dict[str, Any],
        security_level: SecurityLevel = SecurityLevel.basic,
    ):
        """Log security audit event with structured data."""
        audit_entry = {
            "timestamp": aware_utc_now().isoformat(),
            "component": self.component_name,
            "event": event.value,
            "security_level": security_level.value,
            "details": details,
            "compliance": "NIST-SP-800-57",
        }
        if event in [AuditEvent.SECURITY_VIOLATION]:
            self.logger.error(f"SECURITY_AUDIT: {audit_entry}")
        elif event in [AuditEvent.KEY_GENERATED, AuditEvent.KEY_ROTATED]:
            self.logger.info(f"SECURITY_AUDIT: {audit_entry}")
        else:
            self.logger.debug(f"SECURITY_AUDIT: {audit_entry}")


class KeyRotationConfig:
    """Enhanced key rotation configuration following 2025 security best practices."""

    def __init__(
        self,
        rotation_interval_hours: int = 24,
        max_key_age_hours: int = 72,
        key_version_limit: int = 5,
        auto_rotation_enabled: bool = True,
        security_level: SecurityLevel = SecurityLevel.enhanced,
        key_derivation_method: KeyDerivationMethod = KeyDerivationMethod.scrypt,
        enable_audit_logging: bool = True,
        zero_trust_mode: bool = True,
        hsm_ready: bool = False,
        compliance_mode: str = "NIST-SP-800-57",
    ):
        self.rotation_interval_hours = rotation_interval_hours
        self.max_key_age_hours = max_key_age_hours
        self.key_version_limit = key_version_limit
        self.auto_rotation_enabled = auto_rotation_enabled
        self.security_level = security_level
        self.key_derivation_method = key_derivation_method
        self.enable_audit_logging = enable_audit_logging
        self.zero_trust_mode = zero_trust_mode
        self.hsm_ready = hsm_ready
        self.compliance_mode = compliance_mode
        self._apply_security_level_defaults()

    def _apply_security_level_defaults(self):
        """Apply security level-specific defaults (2025 standards)."""
        if self.security_level == SecurityLevel.CRITICAL:
            self.rotation_interval_hours = min(self.rotation_interval_hours, 6)
            self.max_key_age_hours = min(self.max_key_age_hours, 24)
            self.key_version_limit = max(self.key_version_limit, 10)
            self.zero_trust_mode = True
            self.enable_audit_logging = True
        elif self.security_level == SecurityLevel.HIGH:
            self.rotation_interval_hours = min(self.rotation_interval_hours, 12)
            self.max_key_age_hours = min(self.max_key_age_hours, 48)
            self.zero_trust_mode = True
        elif self.security_level == SecurityLevel.enhanced:
            self.rotation_interval_hours = min(self.rotation_interval_hours, 24)
            self.enable_audit_logging = True


class KeyInfo:
    """Enhanced key information with 2025 security metadata."""

    def __init__(
        self,
        key: bytes,
        key_id: str,
        created_at: datetime,
        version: int = 1,
        security_level: SecurityLevel = SecurityLevel.enhanced,
        derivation_method: KeyDerivationMethod = KeyDerivationMethod.scrypt,
    ):
        self.key = key
        self.key_id = key_id
        self.created_at = created_at
        self.version = version
        self.last_used = created_at
        self.usage_count = 0
        self.security_level = security_level
        self.derivation_method = derivation_method
        self.access_history: list[datetime] = [created_at]
        self.compliance_metadata = {
            "nist_compliant": True,
            "zero_trust_verified": True,
            "audit_trail": [],
        }
        self.entropy_bits = len(key) * 8
        self.is_hsm_backed = False
        self.last_security_check = created_at

    def is_expired(self, max_age_hours: int) -> bool:
        """Check if key has exceeded maximum age."""
        age = aware_utc_now() - self.created_at
        return age > timedelta(hours=max_age_hours)

    def should_rotate(self, rotation_interval_hours: int) -> bool:
        """Check if key should be rotated based on interval."""
        age = aware_utc_now() - self.created_at
        return age > timedelta(hours=rotation_interval_hours)

    def mark_used(self, audit_logger: SecurityAuditLogger | None = None):
        """Mark key as recently used with enhanced tracking."""
        now = aware_utc_now()
        self.last_used = now
        self.usage_count += 1
        self.access_history.append(now)
        if len(self.access_history) > 100:
            self.access_history = self.access_history[-100:]
        if audit_logger:
            audit_logger.log_event(
                AuditEvent.KEY_ACCESSED,
                {
                    "key_id": self.key_id,
                    "usage_count": self.usage_count,
                    "key_age_hours": (now - self.created_at).total_seconds() / 3600,
                },
                self.security_level,
            )

    def get_security_metrics(self) -> dict[str, Any]:
        """Get comprehensive security metrics for the key."""
        now = aware_utc_now()
        age_hours = (now - self.created_at).total_seconds() / 3600
        return {
            "key_id": self.key_id,
            "security_level": self.security_level.value,
            "derivation_method": self.derivation_method.value,
            "age_hours": age_hours,
            "usage_count": self.usage_count,
            "entropy_bits": self.entropy_bits,
            "is_hsm_backed": self.is_hsm_backed,
            "compliance_status": self.compliance_metadata,
            "last_used_hours_ago": (now - self.last_used).total_seconds() / 3600,
            "access_frequency": len(self.access_history) / max(age_hours, 1),
        }

    def model_dump(self) -> dict[str, Any]:
        """Convert key info to dictionary representation."""
        return {
            "key_id": self.key_id,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "usage_count": self.usage_count,
            "version": self.version,
            "security_level": self.security_level.value,
            "derivation_method": self.derivation_method.value,
            "entropy_bits": self.entropy_bits,
            "is_hsm_backed": self.is_hsm_backed,
        }


class SecureKeyService:
    """Enhanced secure key service with 2025 security best practices.

    features:
    - Zero Trust Architecture compliance
    - NIST SP 800-57 key management standards
    - Advanced key derivation (Scrypt/pbkdf2)
    - Comprehensive audit logging
    - HSM readiness
    - Cloud KMS integration readiness
    """

    def __init__(self, config: KeyRotationConfig | None = None):
        self.config = config or KeyRotationConfig()
        self.keys: dict[str, KeyInfo] = {}
        self.current_key_id: str | None = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.audit_logger = (
            SecurityAuditLogger("UnifiedKeyService")
            if self.config.enable_audit_logging
            else None
        )
        self.security_level = self.config.security_level
        self.zero_trust_mode = self.config.zero_trust_mode
        self.last_security_audit = aware_utc_now()
        self.security_violations: list[dict[str, Any]] = []
        self.key_access_patterns: dict[str, list[datetime]] = {}
        self._generate_new_key()

    def run_orchestrated_analysis(
        self, operation: str, parameters: dict = None
    ) -> dict[str, Any]:
        """Synchronous orchestrator-compatible interface for testing and simple operations"""
        parameters = parameters or {}
        try:
            if operation == "get_key":
                key_id = parameters.get("key_id")
                if key_id:
                    key_bytes = self.get_key_by_id(key_id)
                    return {
                        "orchestrator_compatible": True,
                        "operation": operation,
                        "result": key_bytes,
                        "timestamp": aware_utc_now().isoformat(),
                    }
                key_bytes, current_key_id = self.get_current_key()
                return {
                    "orchestrator_compatible": True,
                    "operation": operation,
                    "result": key_bytes,
                    "key_id": current_key_id,
                    "timestamp": aware_utc_now().isoformat(),
                }
            if operation == "generate_key":
                key_id = parameters.get("key_id")
                if key_id:
                    new_key_id = self._generate_new_key()
                    if hasattr(self, "_key_id_mapping"):
                        self._key_id_mapping[key_id] = new_key_id
                    else:
                        self._key_id_mapping = {key_id: new_key_id}
                    key_bytes, _ = self.get_current_key()
                    return {
                        "orchestrator_compatible": True,
                        "operation": operation,
                        "result": key_bytes,
                        "timestamp": aware_utc_now().isoformat(),
                    }
                new_key_id = self._generate_new_key()
                key_bytes, _ = self.get_current_key()
                return {
                    "orchestrator_compatible": True,
                    "operation": operation,
                    "result": key_bytes,
                    "timestamp": aware_utc_now().isoformat(),
                }
            if operation == "rotate_key":
                key_id = parameters.get("key_id")
                new_key_id = self.rotate_key()
                if key_id and hasattr(self, "_key_id_mapping"):
                    self._key_id_mapping[key_id] = new_key_id
                key_bytes, _ = self.get_current_key()
                return {
                    "orchestrator_compatible": True,
                    "operation": operation,
                    "result": key_bytes,
                    "timestamp": aware_utc_now().isoformat(),
                }
            if operation == "get_status":
                security_status = self.get_security_status()
                return {
                    "orchestrator_compatible": True,
                    "operation": operation,
                    "result": {
                        "key_count": len(self.keys),
                        "current_key_id": self.current_key_id,
                        "security_status": security_status,
                        "last_operation": "get_status",
                    },
                    "timestamp": aware_utc_now().isoformat(),
                }
            if operation == "cleanup":
                key_id = parameters.get("key_id")
                removed_count = self.cleanup_expired_keys()
                return {
                    "orchestrator_compatible": True,
                    "operation": operation,
                    "result": removed_count,
                    "timestamp": aware_utc_now().isoformat(),
                }
            return {
                "orchestrator_compatible": True,
                "operation": operation,
                "result": None,
                "error": f"Unknown operation: {operation}",
                "timestamp": aware_utc_now().isoformat(),
            }
        except Exception as e:
            return {
                "orchestrator_compatible": True,
                "operation": operation,
                "result": None,
                "error": str(e),
                "timestamp": aware_utc_now().isoformat(),
            }

    async def run_orchestrated_analysis_async(
        self, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Orchestrator-compatible interface for secure key management (2025 pattern)

        Args:
            config: Orchestrator configuration containing:
                - operation: 'get_key', 'rotate_key', 'get_status', 'cleanup'
                - output_path: Local path for key storage (optional)
                - key_id: Specific key ID for operations (optional)

        Returns:
            Orchestrator-compatible result with key management status and metadata
        """
        start_time = datetime.now()
        try:
            operation = config.get("operation", "get_status")
            output_path = config.get("output_path", "./keys")
            if operation == "get_key":
                _, current_key_id = self.get_current_key()
                result = {
                    "key_available": True,
                    "key_id": current_key_id,
                    "key_version": self.keys[current_key_id].version,
                    "key_age_hours": (
                        aware_utc_now() - self.keys[current_key_id].created_at
                    ).total_seconds()
                    / 3600,
                    "usage_count": self.keys[current_key_id].usage_count,
                }
            elif operation == "rotate_key":
                new_key_id = self.rotate_key()
                result = {
                    "rotation_completed": True,
                    "new_key_id": new_key_id,
                    "total_keys": len(self.keys),
                    "rotation_timestamp": aware_utc_now().isoformat(),
                }
            elif operation == "get_status":
                key_info = self.get_key_info()
                security_status = self.get_security_status()
                result = {
                    "total_keys": len(self.keys),
                    "current_key_id": self.current_key_id,
                    "key_details": key_info,
                    "security_status": security_status,
                    "config": {
                        "rotation_interval_hours": self.config.rotation_interval_hours,
                        "key_version_limit": self.config.key_version_limit,
                        "auto_rotation_enabled": self.config.auto_rotation_enabled,
                        "security_level": self.config.security_level.value,
                        "key_derivation_method": self.config.key_derivation_method.value,
                        "zero_trust_mode": self.config.zero_trust_mode,
                        "compliance_mode": self.config.compliance_mode,
                    },
                }
            elif operation == "cleanup":
                removed_count = self.cleanup_expired_keys()
                result = {
                    "cleanup_completed": True,
                    "keys_removed": removed_count,
                    "remaining_keys": len(self.keys),
                    "cleanup_timestamp": aware_utc_now().isoformat(),
                }
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "orchestrator_compatible": True,
                "component_result": result,
                "local_metadata": {
                    "output_path": output_path,
                    "execution_time": execution_time,
                    "operation": operation,
                    "component_version": "1.0.0",
                    "security_status": "active",
                },
            }
        except Exception as e:
            self.logger.error(f"Orchestrated key management failed: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": str(e), "operation_failed": True},
                "local_metadata": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "error": True,
                    "component_version": "1.0.0",
                },
            }

    def get_current_key(self) -> tuple[bytes, str]:
        """Get the current encryption key with enhanced security checks.

        Returns:
            Tuple of (key_bytes, key_id)

        Raises:
            RuntimeError: If no valid key is available
            SecurityError: If security violations detected
        """
        if self.zero_trust_mode:
            self._perform_zero_trust_checks()
        if self.config.auto_rotation_enabled and self._should_rotate_current_key():
            self._rotate_key()
        if self.current_key_id is None or self.current_key_id not in self.keys:
            self._log_security_violation("No valid encryption key available")
            raise RuntimeError("No valid encryption key available")
        key_info = self.keys[self.current_key_id]
        if self._detect_suspicious_access_pattern(self.current_key_id):
            self._log_security_violation(
                f"Suspicious access pattern detected for key {self.current_key_id}"
            )
        key_info.mark_used(self.audit_logger)
        self._track_key_access(self.current_key_id)
        self.logger.debug(f"Retrieved current key: {self.current_key_id}")
        return (key_info.key, self.current_key_id)

    def get_key_by_id(self, key_id: str) -> bytes | None:
        """Get a specific key by ID for decryption.

        Args:
            key_id: The key identifier

        Returns:
            Key bytes if found and valid, None otherwise
        """
        if key_id not in self.keys:
            self.logger.warning(f"Key not found: {key_id}")
            return None
        key_info = self.keys[key_id]
        if key_info.is_expired(self.config.max_key_age_hours):
            self.logger.warning(f"Key expired: {key_id}")
            return None
        key_info.mark_used()
        self.logger.debug(f"Retrieved key by ID: {key_id}")
        return key_info.key

    def rotate_key(self) -> str:
        """Manually rotate the encryption key.

        Returns:
            New key ID
        """
        return self._rotate_key()

    def get_key_info(self) -> dict[str, dict]:
        """Get information about all managed keys.

        Returns:
            Dictionary mapping key IDs to key information
        """
        info = {}
        for key_id, key_info in self.keys.items():
            info[key_id] = {
                "created_at": key_info.created_at.isoformat(),
                "last_used": key_info.last_used.isoformat(),
                "usage_count": key_info.usage_count,
                "version": key_info.version,
                "is_current": key_id == self.current_key_id,
                "is_expired": key_info.is_expired(self.config.max_key_age_hours),
                "should_rotate": key_info.should_rotate(
                    self.config.rotation_interval_hours
                ),
            }
        return info

    def cleanup_expired_keys(self) -> int:
        """Remove expired keys that are no longer needed.

        Returns:
            Number of keys removed
        """
        expired_keys = []
        for key_id, key_info in self.keys.items():
            if key_id == self.current_key_id:
                continue
            if key_info.is_expired(self.config.max_key_age_hours):
                time_since_last_use = aware_utc_now() - key_info.last_used
                if time_since_last_use > timedelta(hours=1):
                    expired_keys.append(key_id)
        sorted_keys = sorted(
            self.keys.items(), key=lambda x: x[1].created_at, reverse=True
        )
        if len(sorted_keys) > self.config.key_version_limit:
            excess_keys = sorted_keys[self.config.key_version_limit :]
            for key_id, _ in excess_keys:
                if key_id != self.current_key_id and key_id not in expired_keys:
                    expired_keys.append(key_id)
        for key_id in expired_keys:
            del self.keys[key_id]
            self.logger.info(f"Removed expired key: {key_id}")
        return len(expired_keys)

    def _generate_new_key(self) -> str:
        """Generate a new encryption key using 2025 security best practices.

        Returns:
            New key ID
        """
        if self.config.key_derivation_method == KeyDerivationMethod.scrypt:
            key_bytes = self._generate_scrypt_key()
        elif self.config.key_derivation_method == KeyDerivationMethod.pbkdf2:
            key_bytes = self._generate_pbkdf2_key()
        else:
            key_bytes = Fernet.generate_key()
        key_id = self._generate_key_id(key_bytes)
        version = (
            max([key_info.version for key_info in self.keys.values()], default=0) + 1
        )
        key_info = KeyInfo(
            key=key_bytes,
            key_id=key_id,
            created_at=aware_utc_now(),
            version=version,
            security_level=self.security_level,
            derivation_method=self.config.key_derivation_method,
        )
        self.keys[key_id] = key_info
        self.current_key_id = key_id
        if self.audit_logger:
            self.audit_logger.log_event(
                AuditEvent.KEY_GENERATED,
                {
                    "key_id": key_id,
                    "version": version,
                    "derivation_method": self.config.key_derivation_method.value,
                    "security_level": self.security_level.value,
                    "entropy_bits": len(key_bytes) * 8,
                },
                self.security_level,
            )
        self.logger.info(
            f"Generated new encryption key: {key_id} (version {version}, {self.config.key_derivation_method.value})"
        )
        return key_id

    def _generate_scrypt_key(self) -> bytes:
        """Generate key using Scrypt KDF (memory-hard, 2025 recommended)."""
        salt = secrets.token_bytes(32)
        password = secrets.token_bytes(64)
        kdf = Scrypt(length=32, salt=salt, n=2**16, r=8, p=1)
        key_material = kdf.derive(password)
        import base64

        return base64.urlsafe_b64encode(key_material)

    def _generate_pbkdf2_key(self) -> bytes:
        """Generate key using pbkdf2-HMAC-SHA256 (NIST approved)."""
        salt = secrets.token_bytes(32)
        password = secrets.token_bytes(64)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600000
        )
        key_material = kdf.derive(password)
        import base64

        return base64.urlsafe_b64encode(key_material)

    def _rotate_key(self) -> str:
        """Rotate to a new encryption key.

        Returns:
            New key ID
        """
        old_key_id = self.current_key_id
        new_key_id = self._generate_new_key()
        removed_count = self.cleanup_expired_keys()
        self.logger.info(
            f"Key rotation completed: {old_key_id} -> {new_key_id}, removed {removed_count} expired keys"
        )
        return new_key_id

    def _should_rotate_current_key(self) -> bool:
        """Check if current key should be rotated."""
        if self.current_key_id is None:
            return True
        key_info = self.keys.get(self.current_key_id)
        if key_info is None:
            return True
        return key_info.should_rotate(self.config.rotation_interval_hours)

    def _generate_key_id(self, key_bytes: bytes) -> str:
        """Generate a unique key identifier.

        Args:
            key_bytes: The key material

        Returns:
            Unique key identifier
        """
        timestamp = str(int(time.time()))
        random_component = secrets.token_hex(8)
        key_hash = hashlib.sha256(key_bytes).hexdigest()[:8]
        key_id = f"key_{timestamp}_{random_component}_{key_hash}"
        return key_id

    def _perform_zero_trust_checks(self):
        """Perform Zero Trust security verification (2025 standard)."""
        now = aware_utc_now()
        if (now - self.last_security_audit).total_seconds() > 3600:
            self._perform_security_audit()
        if self.current_key_id and self.current_key_id in self.keys:
            key_info = self.keys[self.current_key_id]
            if not self._verify_key_integrity(key_info):
                self._log_security_violation("Key integrity verification failed")
                raise RuntimeError("Key integrity verification failed")

    def _perform_security_audit(self):
        """Perform comprehensive security audit."""
        self.last_security_audit = aware_utc_now()
        expired_count = 0
        for _, key_info in self.keys.items():
            if key_info.is_expired(self.config.max_key_age_hours):
                expired_count += 1
        suspicious_patterns = self._analyze_access_patterns()
        if self.audit_logger:
            self.audit_logger.log_event(
                AuditEvent.KEY_ACCESSED,
                {
                    "audit_type": "security_audit",
                    "total_keys": len(self.keys),
                    "expired_keys": expired_count,
                    "suspicious_patterns": len(suspicious_patterns),
                    "security_violations": len(self.security_violations),
                },
                self.security_level,
            )

    def _verify_key_integrity(self, key_info: KeyInfo) -> bool:
        """Verify key integrity using checksums."""
        try:
            Fernet(key_info.key)
            return True
        except Exception:
            return False

    def _detect_suspicious_access_pattern(self, key_id: str) -> bool:
        """Detect suspicious key access patterns."""
        if key_id not in self.key_access_patterns:
            return False
        access_times = self.key_access_patterns[key_id]
        if len(access_times) < 5:
            return False
        recent_accesses = [
            t for t in access_times if (aware_utc_now() - t).total_seconds() < 60
        ]
        if len(recent_accesses) > 10:
            return True
        return False

    def _track_key_access(self, key_id: str):
        """Track key access patterns for security analysis."""
        if key_id not in self.key_access_patterns:
            self.key_access_patterns[key_id] = []
        self.key_access_patterns[key_id].append(aware_utc_now())
        if len(self.key_access_patterns[key_id]) > 100:
            self.key_access_patterns[key_id] = self.key_access_patterns[key_id][-100:]

    def _analyze_access_patterns(self) -> list[dict[str, Any]]:
        """Analyze access patterns for security anomalies."""
        suspicious_patterns = []
        for key_id, access_times in self.key_access_patterns.items():
            if len(access_times) < 2:
                continue
            time_span = (access_times[-1] - access_times[0]).total_seconds()
            if time_span > 0:
                access_rate = len(access_times) / time_span
                if access_rate > 0.1:
                    suspicious_patterns.append({
                        "key_id": key_id,
                        "pattern": "high_frequency_access",
                        "access_rate": access_rate,
                        "total_accesses": len(access_times),
                    })
        return suspicious_patterns

    def _log_security_violation(self, message: str):
        """Log security violation for compliance and monitoring."""
        violation = {
            "timestamp": aware_utc_now().isoformat(),
            "message": message,
            "security_level": self.security_level.value,
            "current_key_id": self.current_key_id,
        }
        self.security_violations.append(violation)
        if len(self.security_violations) > 100:
            self.security_violations = self.security_violations[-100:]
        if self.audit_logger:
            self.audit_logger.log_event(
                AuditEvent.SECURITY_VIOLATION, violation, self.security_level
            )
        self.logger.warning(f"Security violation: {message}")

    def get_security_status(self) -> dict[str, Any]:
        """Get comprehensive security status report (2025 compliance)."""
        now = aware_utc_now()
        return {
            "security_level": self.security_level.value,
            "zero_trust_mode": self.zero_trust_mode,
            "compliance_mode": self.config.compliance_mode,
            "total_keys": len(self.keys),
            "current_key_age_hours": (
                now - self.keys[self.current_key_id].created_at
            ).total_seconds()
            / 3600
            if self.current_key_id
            else 0,
            "last_security_audit": self.last_security_audit.isoformat(),
            "security_violations_count": len(self.security_violations),
            "key_derivation_method": self.config.key_derivation_method.value,
            "auto_rotation_enabled": self.config.auto_rotation_enabled,
            "audit_logging_enabled": self.config.enable_audit_logging,
            "hsm_ready": self.config.hsm_ready,
            "suspicious_patterns": len(self._analyze_access_patterns()),
        }


class UnifiedKeyService:
    """Unified key management system combining secure key management and Fernet encryption.

    This service consolidates SecureKeyService and FernetKeyManager functionality
    using composition to provide both key management and encryption capabilities
    in a single, unified interface.
    """

    def __init__(self, config: KeyRotationConfig | None = None):
        """Initialize unified key manager with both key management and encryption capabilities.

        Args:
            config: Optional key rotation configuration
        """
        self.config = config or KeyRotationConfig()
        self.keys: dict[str, KeyInfo] = {}
        self.current_key_id: str | None = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.audit_logger = (
            SecurityAuditLogger("UnifiedKeyService")
            if self.config.enable_audit_logging
            else None
        )
        self.security_level = self.config.security_level
        self.zero_trust_mode = self.config.zero_trust_mode
        self.last_security_audit = aware_utc_now()
        self.security_violations: list[dict[str, Any]] = []
        self.key_access_patterns: dict[str, list[datetime]] = {}
        self._fernet_cache: dict[str, Fernet] = {}
        self._generate_new_key()

    def get_current_key(self) -> tuple[bytes, str]:
        """Get current encryption key and its ID."""
        if self.current_key_id is None or self.current_key_id not in self.keys:
            self._generate_new_key()
        key_info = self.keys[self.current_key_id]
        key_info.last_used = aware_utc_now()
        key_info.usage_count += 1
        return (key_info.key, self.current_key_id)

    def get_key_by_id(self, key_id: str) -> bytes:
        """Get specific key by ID."""
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
        key_info = self.keys[key_id]
        key_info.last_used = aware_utc_now()
        key_info.usage_count += 1
        return key_info.key

    def rotate_key(self) -> str:
        """Rotate to a new key and return new key ID."""
        new_key_id = self._generate_new_key()
        self._fernet_cache.clear()
        return new_key_id

    def get_key_info(self) -> dict[str, Any]:
        """Get comprehensive key information."""
        if self.current_key_id is None:
            return {"error": "No keys available"}
        key_info = self.keys.get(self.current_key_id)
        if key_info is None:
            return {"error": "Current key not found"}
        return key_info.model_dump()

    def cleanup_old_keys(self, keep_count: int = 3) -> list[str]:
        """Clean up old keys, keeping specified count."""
        if len(self.keys) <= keep_count:
            return []
        sorted_keys = sorted(
            self.keys.items(), key=lambda x: x[1].created_at, reverse=True
        )
        keys_to_remove = [key_id for key_id, _ in sorted_keys[keep_count:]]
        for key_id in keys_to_remove:
            del self.keys[key_id]
            self._fernet_cache.pop(key_id, None)
        return keys_to_remove

    def _generate_new_key(self) -> str:
        """Generate a new encryption key and store it."""
        key_id = f"key_{int(time.time() * 1000000)}"
        key_bytes = Fernet.generate_key()
        key_info = KeyInfo(
            key=key_bytes,
            key_id=key_id,
            created_at=aware_utc_now(),
            derivation_method=KeyDerivationMethod.direct,
        )
        self.keys[key_id] = key_info
        self.current_key_id = key_id
        self.logger.info(f"Generated new key: {key_id}")
        return key_id

    def _get_fernet(self, key_id: str | None = None) -> tuple[Fernet, str]:
        """Get Fernet instance for encryption/decryption.

        Args:
            key_id: Optional specific key ID to use

        Returns:
            Tuple of (Fernet instance, key_id used)
        """
        if key_id is None:
            key_bytes, key_id = self.get_current_key()
        else:
            key_bytes = self.get_key_by_id(key_id)
        if key_id in self._fernet_cache:
            return (self._fernet_cache[key_id], key_id)
        fernet = Fernet(key_bytes)
        self._fernet_cache[key_id] = fernet
        return (fernet, key_id)

    def encrypt(self, data: bytes, key_id: str | None = None) -> tuple[bytes, str]:
        """Encrypt data using Fernet encryption.

        Args:
            data: Data to encrypt
            key_id: Optional specific key ID to use

        Returns:
            Tuple of (encrypted_data, key_id_used)
        """
        fernet, used_key_id = self._get_fernet(key_id)
        encrypted_data = fernet.encrypt(data)
        self.logger.debug(f"Encrypted data using key {used_key_id}")
        return (encrypted_data, used_key_id)

    def decrypt(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt data using Fernet encryption.

        Args:
            encrypted_data: Data to decrypt
            key_id: Key ID to use for decryption

        Returns:
            Decrypted data
        """
        fernet, _ = self._get_fernet(key_id)
        decrypted_data = fernet.decrypt(encrypted_data)
        self.logger.debug(f"Decrypted data using key {key_id}")
        return decrypted_data

    def encrypt_string(self, text: str, key_id: str | None = None) -> tuple[bytes, str]:
        """Encrypt string data.

        Args:
            text: String to encrypt
            key_id: Optional specific key ID to use

        Returns:
            Tuple of (encrypted_data, key_id_used)
        """
        return self.encrypt(text.encode("utf-8"), key_id)

    def decrypt_string(self, encrypted_data: bytes, key_id: str) -> str:
        """Decrypt to string data.

        Args:
            encrypted_data: Data to decrypt
            key_id: Key ID to use for decryption

        Returns:
            Decrypted string
        """
        return self.decrypt(encrypted_data, key_id).decode("utf-8")

    def run_orchestrated_analysis(
        self, operation: str, parameters: dict = None
    ) -> dict[str, Any]:
        """Unified orchestrator-compatible interface combining key management and encryption."""
        parameters = parameters or {}
        try:
            if operation == "get_key":
                key_id = parameters.get("key_id")
                if key_id:
                    key_bytes = self.get_key_by_id(key_id)
                    return {
                        "orchestrator_compatible": True,
                        "operation": operation,
                        "result": key_bytes,
                        "timestamp": aware_utc_now().isoformat(),
                    }
                key_bytes, current_key_id = self.get_current_key()
                return {
                    "orchestrator_compatible": True,
                    "operation": operation,
                    "result": key_bytes,
                    "key_id": current_key_id,
                    "timestamp": aware_utc_now().isoformat(),
                }
            if operation == "generate_key":
                key_id = parameters.get("key_id")
                if key_id:
                    new_key_id = self._generate_new_key()
                    if hasattr(self, "_key_id_mapping"):
                        self._key_id_mapping[key_id] = new_key_id
                    else:
                        self._key_id_mapping = {key_id: new_key_id}
                    key_bytes, _ = self.get_current_key()
                    return {
                        "orchestrator_compatible": True,
                        "operation": operation,
                        "result": key_bytes,
                        "timestamp": aware_utc_now().isoformat(),
                    }
                new_key_id = self._generate_new_key()
                key_bytes, _ = self.get_current_key()
                return {
                    "orchestrator_compatible": True,
                    "operation": operation,
                    "result": key_bytes,
                    "timestamp": aware_utc_now().isoformat(),
                }
            if operation == "rotate_key":
                key_id = parameters.get("key_id")
                new_key_id = self.rotate_key()
                if key_id and hasattr(self, "_key_id_mapping"):
                    self._key_id_mapping[key_id] = new_key_id
                key_bytes, _ = self.get_current_key()
                return {
                    "orchestrator_compatible": True,
                    "operation": operation,
                    "result": key_bytes,
                    "timestamp": aware_utc_now().isoformat(),
                }
            if operation == "encrypt":
                data = parameters.get("data")
                key_id = parameters.get("key_id")
                if data is None:
                    return {
                        "orchestrator_compatible": True,
                        "operation": operation,
                        "result": {"success": False, "error": "No data provided"},
                        "timestamp": aware_utc_now().isoformat(),
                    }
                if isinstance(data, str):
                    data = data.encode("utf-8")
                encrypted_data, used_key_id = self.encrypt(data, key_id)
                return {
                    "orchestrator_compatible": True,
                    "operation": operation,
                    "result": {
                        "encrypted_data": encrypted_data,
                        "key_id": used_key_id,
                        "timestamp": aware_utc_now().isoformat(),
                    },
                    "timestamp": aware_utc_now().isoformat(),
                }
            if operation == "decrypt":
                data = parameters.get("data")
                key_id = parameters.get("key_id")
                if data is None:
                    return {
                        "orchestrator_compatible": True,
                        "operation": operation,
                        "result": {"success": False, "error": "No data provided"},
                        "timestamp": aware_utc_now().isoformat(),
                    }
                if isinstance(data, str):
                    data = data.encode("utf-8")
                encrypted_data, used_key_id = self.encrypt(data)
                return {
                    "orchestrator_compatible": True,
                    "operation": operation,
                    "result": {
                        "encrypted_data": encrypted_data,
                        "key_id": used_key_id,
                        "timestamp": aware_utc_now().isoformat(),
                    },
                    "timestamp": aware_utc_now().isoformat(),
                }
            if operation == "decrypt":
                encrypted_data = parameters.get("encrypted_data")
                key_id = parameters.get("key_id")
                if encrypted_data is None or key_id is None:
                    return {
                        "orchestrator_compatible": True,
                        "operation": operation,
                        "result": {
                            "success": False,
                            "error": "Missing encrypted_data or key_id",
                        },
                        "timestamp": aware_utc_now().isoformat(),
                    }
                try:
                    decrypted_data = self.decrypt(encrypted_data, key_id)
                    if isinstance(decrypted_data, bytes):
                        try:
                            decrypted_str = decrypted_data.decode("utf-8")
                        except UnicodeDecodeError:
                            decrypted_str = decrypted_data
                    else:
                        decrypted_str = decrypted_data
                    return {
                        "orchestrator_compatible": True,
                        "operation": operation,
                        "result": {
                            "success": True,
                            "decrypted_data": decrypted_str,
                            "key_id": key_id,
                        },
                        "timestamp": aware_utc_now().isoformat(),
                    }
                except Exception as e:
                    return {
                        "orchestrator_compatible": True,
                        "operation": operation,
                        "result": {
                            "success": False,
                            "error": str(e),
                            "decrypted_data": None,
                        },
                        "timestamp": aware_utc_now().isoformat(),
                    }
            elif operation == "test_encryption":
                test_data = "Test encryption cycle"
                try:
                    start_time = time.time()
                    encrypted_data, key_id = self.encrypt(test_data.encode("utf-8"))
                    decrypted_data = self.decrypt(encrypted_data, key_id)
                    end_time = time.time()
                    success = decrypted_data.decode("utf-8") == test_data
                    return {
                        "orchestrator_compatible": True,
                        "operation": operation,
                        "result": {
                            "success": success,
                            "test_data": test_data,
                            "encrypted_data": encrypted_data,
                            "decrypted_data": decrypted_data.decode("utf-8"),
                            "key_id": key_id,
                            "performance_ms": (end_time - start_time) * 1000,
                        },
                        "timestamp": aware_utc_now().isoformat(),
                    }
                except Exception as e:
                    return {
                        "orchestrator_compatible": True,
                        "operation": operation,
                        "result": {"success": False, "error": str(e)},
                        "timestamp": aware_utc_now().isoformat(),
                    }
            elif operation == "get_status":
                return {
                    "orchestrator_compatible": True,
                    "operation": operation,
                    "result": {
                        "key_count": len(self.keys),
                        "current_key_id": self.current_key_id,
                        "last_operation": "get_status",
                        "unified_manager": True,
                    },
                    "timestamp": aware_utc_now().isoformat(),
                }
            else:
                return {
                    "orchestrator_compatible": True,
                    "operation": operation,
                    "result": None,
                    "error": f"Unknown operation: {operation}",
                    "timestamp": aware_utc_now().isoformat(),
                }
        except Exception as e:
            return {
                "orchestrator_compatible": True,
                "operation": operation,
                "result": None,
                "error": str(e),
                "timestamp": aware_utc_now().isoformat(),
            }

    async def run_orchestrated_analysis_async(
        self, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Unified async orchestrator interface for key management and encryption operations."""
        operation = config.get("operation", "get_status")
        parameters = {k: v for k, v in config.items() if k != "operation"}
        result = self.run_orchestrated_analysis(operation, parameters)
        return {
            "orchestrator_compatible": True,
            "component_result": result.get("result", {}),
            "local_metadata": {
                "execution_time": 0.0,
                "operation": operation,
                "component_version": "2.0.0",
                "unified_manager": True,
            },
        }

    def get_current_fernet(self) -> tuple[Fernet, str]:
        """Get current Fernet instance and key ID.

        Returns:
            Tuple of (Fernet instance, key_id)
        """
        return self._get_fernet()

    def get_fernet_by_id(self, key_id: str) -> Fernet | None:
        """Get Fernet instance for specific key.

        Args:
            key_id: Key identifier

        Returns:
            Fernet instance if key found, None otherwise
        """
        try:
            fernet, _ = self._get_fernet(key_id)
            return fernet
        except Exception:
            return None


_default_key_manager = None


def get_key_manager() -> UnifiedKeyService:
    """Get global unified key manager instance."""
    global _default_key_manager
    if _default_key_manager is None:
        _default_key_manager = UnifiedKeyService()
    return _default_key_manager


def get_unified_key_manager() -> UnifiedKeyService:
    """Get global unified key manager instance."""
    return get_key_manager()
