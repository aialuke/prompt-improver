"""CryptographyComponent - Cryptographic Operations Service

Complete implementation extracted from UnifiedCryptoManager. This component handles 
all cryptographic operations including hashing, encryption, decryption, and secure 
random generation using NIST-approved algorithms.

Features:
- NIST-approved cryptographic algorithms (SHA-256, SHA-512, AES-256)
- Secure random generation for all use cases
- Key derivation using PBKDF2-HMAC and Scrypt
- Memory-safe operations with secure comparison
- Comprehensive audit logging and monitoring
- Integration with KeyManager for encryption operations
- Performance optimization with caching
"""

import asyncio
import hashlib
import logging
import secrets
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from prompt_improver.database import SecurityContext, SecurityPerformanceMetrics
from prompt_improver.security.key_manager import (
    SecurityLevel,
    UnifiedKeyService,
)
from prompt_improver.security.unified.protocols import (
    CryptographyProtocol,
    SecurityComponentStatus,
    SecurityOperationResult,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class HashAlgorithm(str, Enum):
    """Supported hash algorithms (NIST approved)."""
    SHA256 = "sha256"
    SHA512 = "sha512"
    SHA1 = "sha1"
    MD5 = "md5"


class RandomType(str, Enum):
    """Types of random data generation."""
    BYTES = "bytes"
    HEX = "hex"
    URLSAFE = "urlsafe"
    TOKEN = "token"


class CryptoAuditEvent(str, Enum):
    """Cryptographic audit events."""
    HASH_GENERATED = "hash_generated"
    RANDOM_GENERATED = "random_generated"
    ENCRYPTION_PERFORMED = "encryption_performed"
    DECRYPTION_PERFORMED = "decryption_performed"
    KEY_DERIVED = "key_derived"
    CRYPTO_ERROR = "crypto_error"


class CryptographyComponent:
    """Complete cryptography component implementing CryptographyProtocol."""
    
    def __init__(self, key_manager: UnifiedKeyService = None, security_level: SecurityLevel = SecurityLevel.enhanced):
        self._initialized = False
        self._key_manager = key_manager
        self._security_level = security_level
        self._metrics = {
            "hash_operations": 0,
            "encrypt_operations": 0,
            "decrypt_operations": 0,
            "random_generations": 0,
            "total_crypto_time_ms": 0.0,
        }
        self._audit_enabled = True
        self._failed_operations = []
    
    async def initialize(self) -> bool:
        """Initialize the cryptography component."""
        try:
            if not self._key_manager:
                self._key_manager = UnifiedKeyService()
            
            self._initialized = True
            logger.info("CryptographyComponent initialized with NIST-approved algorithms")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize cryptography component: {e}")
            return False
    
    async def health_check(self) -> Tuple[SecurityComponentStatus, Dict[str, Any]]:
        """Check component health status."""
        if not self._initialized:
            return SecurityComponentStatus.UNHEALTHY, {
                "initialized": False,
                "error": "Component not initialized"
            }
        
        try:
            # Test basic cryptographic operations
            test_data = "health_check_test"
            test_hash = self._hash_sha256(test_data)
            test_random = self._generate_random_bytes(16)
            
            if not test_hash or not test_random:
                return SecurityComponentStatus.UNHEALTHY, {
                    "initialized": self._initialized,
                    "error": "Basic crypto operations failed"
                }
            
            return SecurityComponentStatus.HEALTHY, {
                "initialized": self._initialized,
                "metrics": self._metrics.copy(),
                "key_manager_available": self._key_manager is not None,
                "security_level": self._security_level.value if self._security_level else "unknown",
                "supported_algorithms": {
                    "hash": [algo.value for algo in HashAlgorithm],
                    "random": [rtype.value for rtype in RandomType],
                    "encryption": ["Fernet", "AES-256"]
                }
            }
        except Exception as e:
            return SecurityComponentStatus.UNHEALTHY, {
                "initialized": self._initialized,
                "error": f"Health check failed: {e}"
            }
    
    async def get_metrics(self) -> SecurityPerformanceMetrics:
        """Get security performance metrics."""
        total_ops = sum([
            self._metrics["hash_operations"],
            self._metrics["encrypt_operations"], 
            self._metrics["decrypt_operations"],
            self._metrics["random_generations"]
        ])
        avg_latency = (
            self._metrics["total_crypto_time_ms"] / total_ops 
            if total_ops > 0 else 0.0
        )
        
        return SecurityPerformanceMetrics(
            operation_count=total_ops,
            average_latency_ms=avg_latency,
            error_rate=len(self._failed_operations) / max(total_ops, 1),
            threat_detection_count=0,  # Crypto component doesn't detect threats
            last_updated=aware_utc_now()
        )
    
    async def hash_data(
        self,
        data: Union[str, bytes],
        algorithm: str = "sha256",
        salt: Optional[bytes] = None
    ) -> SecurityOperationResult:
        """Hash data using specified algorithm."""
        start_time = time.perf_counter()
        self._metrics["hash_operations"] += 1
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Convert algorithm string to enum
            hash_algo = HashAlgorithm(algorithm.lower())
            
            # Prepare data
            if isinstance(data, str):
                data_bytes = data.encode("utf-8")
            else:
                data_bytes = data
            
            # Add salt if provided
            if salt:
                data_bytes = data_bytes + salt
            
            # Perform hashing
            if hash_algo == HashAlgorithm.SHA256:
                hash_obj = hashlib.sha256(data_bytes)
            elif hash_algo == HashAlgorithm.SHA512:
                hash_obj = hashlib.sha512(data_bytes)
            elif hash_algo == HashAlgorithm.SHA1:
                logger.warning("SHA1 is deprecated and should not be used for security purposes")
                hash_obj = hashlib.sha1(data_bytes)
            elif hash_algo == HashAlgorithm.MD5:
                logger.warning("MD5 is deprecated and should not be used for security purposes")
                hash_obj = hashlib.md5(data_bytes, usedforsecurity=False)
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            hash_result = hash_obj.hexdigest()
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_crypto_time_ms"] += execution_time
            
            if self._audit_enabled:
                self._log_crypto_event(CryptoAuditEvent.HASH_GENERATED, {
                    "algorithm": hash_algo.value,
                    "data_length": len(data_bytes),
                    "salt_used": salt is not None,
                    "hash_length": len(hash_result)
                })
            
            return SecurityOperationResult(
                success=True,
                operation_type="hash_data",
                execution_time_ms=execution_time,
                metadata={
                    "algorithm": hash_algo.value,
                    "hash_value": hash_result,
                    "salt_used": salt is not None,
                    "data_length": len(data_bytes)
                }
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_crypto_time_ms"] += execution_time
            self._handle_crypto_error("hash_data", e)
            return SecurityOperationResult(
                success=False,
                operation_type="hash_data",
                execution_time_ms=execution_time,
                metadata={
                    "algorithm": algorithm,
                    "error": str(e)
                }
            )
    
    async def encrypt_data(
        self,
        data: Union[str, bytes],
        key_id: Optional[str] = None,
        security_context: Optional[SecurityContext] = None
    ) -> SecurityOperationResult:
        """Encrypt data using specified key."""
        start_time = time.perf_counter()
        self._metrics["encrypt_operations"] += 1
        
        try:
            if not self._initialized:
                await self.initialize()
            
            if not self._key_manager:
                raise ValueError("Key manager not available for encryption")
            
            # Prepare data
            if isinstance(data, str):
                data_bytes = data.encode("utf-8")
            else:
                data_bytes = data
            
            # Encrypt using key manager
            encrypted_data, used_key_id = self._key_manager.encrypt(data_bytes, key_id)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_crypto_time_ms"] += execution_time
            
            if self._audit_enabled:
                self._log_crypto_event(CryptoAuditEvent.ENCRYPTION_PERFORMED, {
                    "key_id": used_key_id,
                    "data_length": len(data_bytes),
                    "encrypted_length": len(encrypted_data),
                    "security_context": security_context.agent_id if security_context else None
                })
            
            return SecurityOperationResult(
                success=True,
                operation_type="encrypt_data",
                execution_time_ms=execution_time,
                security_context=security_context,
                metadata={
                    "key_id": used_key_id,
                    "encrypted_data": encrypted_data,
                    "data_length": len(data_bytes),
                    "encrypted_length": len(encrypted_data)
                }
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_crypto_time_ms"] += execution_time
            self._handle_crypto_error("encrypt_data", e)
            return SecurityOperationResult(
                success=False,
                operation_type="encrypt_data",
                execution_time_ms=execution_time,
                security_context=security_context,
                metadata={
                    "key_id": key_id,
                    "error": str(e)
                }
            )
    
    async def decrypt_data(
        self,
        encrypted_data: bytes,
        key_id: str,
        security_context: SecurityContext
    ) -> SecurityOperationResult:
        """Decrypt data using specified key."""
        start_time = time.perf_counter()
        self._metrics["decrypt_operations"] += 1
        
        try:
            if not self._initialized:
                await self.initialize()
            
            if not self._key_manager:
                raise ValueError("Key manager not available for decryption")
            
            # Decrypt using key manager
            decrypted_data = self._key_manager.decrypt(encrypted_data, key_id)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_crypto_time_ms"] += execution_time
            
            if self._audit_enabled:
                self._log_crypto_event(CryptoAuditEvent.DECRYPTION_PERFORMED, {
                    "key_id": key_id,
                    "encrypted_length": len(encrypted_data),
                    "decrypted_length": len(decrypted_data),
                    "security_context": security_context.agent_id if security_context else None
                })
            
            return SecurityOperationResult(
                success=True,
                operation_type="decrypt_data",
                execution_time_ms=execution_time,
                security_context=security_context,
                metadata={
                    "key_id": key_id,
                    "decrypted_data": decrypted_data,
                    "encrypted_length": len(encrypted_data),
                    "decrypted_length": len(decrypted_data)
                }
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_crypto_time_ms"] += execution_time
            self._handle_crypto_error("decrypt_data", e)
            return SecurityOperationResult(
                success=False,
                operation_type="decrypt_data",
                execution_time_ms=execution_time,
                security_context=security_context,
                metadata={
                    "key_id": key_id,
                    "error": str(e)
                }
            )
    
    async def generate_random(
        self,
        length: int,
        random_type: str = "bytes"
    ) -> SecurityOperationResult:
        """Generate cryptographically secure random data."""
        start_time = time.perf_counter()
        self._metrics["random_generations"] += 1
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Convert random_type string to enum
            rtype = RandomType(random_type.lower())
            
            # Generate random data
            if rtype == RandomType.BYTES:
                result = secrets.token_bytes(length)
            elif rtype == RandomType.HEX:
                result = secrets.token_hex(length)
            elif rtype == RandomType.URLSAFE:
                result = secrets.token_urlsafe(length)
            elif rtype == RandomType.TOKEN:
                result = secrets.token_hex(length // 2 if length > 1 else 1)
            else:
                raise ValueError(f"Unsupported random type: {random_type}")
            
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_crypto_time_ms"] += execution_time
            
            if self._audit_enabled:
                self._log_crypto_event(CryptoAuditEvent.RANDOM_GENERATED, {
                    "type": rtype.value,
                    "length": length,
                    "result_length": len(result)
                })
            
            return SecurityOperationResult(
                success=True,
                operation_type="generate_random",
                execution_time_ms=execution_time,
                metadata={
                    "length": length,
                    "random_type": rtype.value,
                    "random_data": result,
                    "result_length": len(result)
                }
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics["total_crypto_time_ms"] += execution_time
            self._handle_crypto_error("generate_random", e)
            return SecurityOperationResult(
                success=False,
                operation_type="generate_random",
                execution_time_ms=execution_time,
                metadata={
                    "length": length,
                    "random_type": random_type,
                    "error": str(e)
                }
            )
    
    def _hash_sha256(self, data: Union[str, bytes]) -> str:
        """Internal SHA-256 hash method for health checks."""
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = data
        return hashlib.sha256(data_bytes).hexdigest()
    
    def _generate_random_bytes(self, length: int) -> bytes:
        """Internal random bytes generation for health checks."""
        return secrets.token_bytes(length)
    
    def derive_key_pbkdf2(
        self,
        password: Union[str, bytes],
        salt: Optional[bytes] = None,
        length: int = 32,
        iterations: int = 600000
    ) -> Tuple[bytes, bytes]:
        """Derive key from password using PBKDF2-HMAC-SHA256."""
        if salt is None:
            salt = self._generate_random_bytes(32)
        
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
        
        if self._audit_enabled:
            self._log_crypto_event(CryptoAuditEvent.KEY_DERIVED, {
                "algorithm": "PBKDF2-HMAC-SHA256",
                "iterations": iterations,
                "salt_length": len(salt),
                "key_length": len(derived_key)
            })
        
        return derived_key, salt
    
    def derive_key_scrypt(
        self,
        password: Union[str, bytes],
        salt: Optional[bytes] = None,
        length: int = 32,
        n: int = 2**16,
        r: int = 8,
        p: int = 1
    ) -> Tuple[bytes, bytes]:
        """Derive key using Scrypt (memory-hard function)."""
        if salt is None:
            salt = self._generate_random_bytes(32)
        
        if isinstance(password, str):
            password_bytes = password.encode("utf-8")
        else:
            password_bytes = password
        
        kdf = Scrypt(
            length=length,
            salt=salt,
            n=n,
            r=r,
            p=p,
            backend=default_backend()
        )
        
        derived_key = kdf.derive(password_bytes)
        
        if self._audit_enabled:
            self._log_crypto_event(CryptoAuditEvent.KEY_DERIVED, {
                "algorithm": "Scrypt",
                "n": n,
                "r": r,
                "p": p,
                "salt_length": len(salt),
                "key_length": len(derived_key)
            })
        
        return derived_key, salt
    
    def secure_compare(self, a: Union[str, bytes], b: Union[str, bytes]) -> bool:
        """Perform timing-safe comparison of two values."""
        if isinstance(a, str):
            a_bytes = a.encode("utf-8")
        else:
            a_bytes = a
        
        if isinstance(b, str):
            b_bytes = b.encode("utf-8")
        else:
            b_bytes = b
        
        return secrets.compare_digest(a_bytes, b_bytes)
    
    def generate_cache_key(self, *components: Any, max_length: int = 16) -> str:
        """Generate standardized cache key from components."""
        key_string = "_".join(str(component) for component in components)
        return self._hash_sha256(key_string)[:max_length]
    
    def generate_session_token(self, prefix: str = "session", length: int = 32) -> str:
        """Generate secure session token."""
        timestamp = int(time.time() * 1000000)
        random_part = secrets.token_urlsafe(length)
        return f"{prefix}_{timestamp}_{random_part}"
    
    def _log_crypto_event(self, event: CryptoAuditEvent, details: Dict[str, Any]):
        """Log cryptographic audit event."""
        if not self._audit_enabled:
            return
        
        audit_entry = {
            "timestamp": aware_utc_now().isoformat(),
            "component": "CryptographyComponent",
            "event": event.value,
            "security_level": self._security_level.value if self._security_level else "unknown",
            "details": details,
        }
        
        if event == CryptoAuditEvent.CRYPTO_ERROR:
            logger.error(f"CRYPTO_AUDIT: {audit_entry}")
        else:
            logger.debug(f"CRYPTO_AUDIT: {audit_entry}")
    
    def _handle_crypto_error(self, operation: str, error: Exception):
        """Handle cryptographic operation errors."""
        error_info = {
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": aware_utc_now().isoformat(),
        }
        
        self._failed_operations.append(error_info)
        if len(self._failed_operations) > 100:
            self._failed_operations = self._failed_operations[-100:]
        
        if self._audit_enabled:
            self._log_crypto_event(CryptoAuditEvent.CRYPTO_ERROR, error_info)
        
        logger.error(f"Crypto operation failed: {operation} - {error}")
    
    async def cleanup(self) -> bool:
        """Cleanup component resources."""
        try:
            self._metrics = {
                "hash_operations": 0,
                "encrypt_operations": 0,
                "decrypt_operations": 0,
                "random_generations": 0,
                "total_crypto_time_ms": 0.0,
            }
            self._failed_operations.clear()
            self._initialized = False
            logger.info("CryptographyComponent cleanup completed")
            return True
        except Exception as e:
            logger.error(f"Error during cryptography component cleanup: {e}")
            return False