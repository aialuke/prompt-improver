"""Comprehensive integration tests for unified crypto manager consolidation.

Tests the complete migration from scattered crypto operations to the
unified crypto manager system, including performance, security, and
compatibility validations.
"""
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
import pytest
from prompt_improver.security import HashAlgorithm, RandomType, SecurityLevel, UnifiedCryptoManager, decrypt_data, encrypt_data, generate_cache_key, generate_token_bytes, generate_token_hex, generate_token_urlsafe, get_crypto_manager, hash_md5, hash_sha256, secure_compare

class TestUnifiedCryptoManagerBasics:
    """Test basic crypto manager functionality."""

    def test_crypto_manager_initialization(self):
        """Test crypto manager initializes correctly."""
        crypto_manager = UnifiedCryptoManager()
        assert crypto_manager is not None
        assert crypto_manager.security_level == SecurityLevel.enhanced
        assert crypto_manager.audit_enabled is True
        assert crypto_manager.key_manager is not None

    def test_global_crypto_manager(self):
        """Test global crypto manager instance."""
        manager1 = get_crypto_manager()
        manager2 = get_crypto_manager()
        assert manager1 is manager2

    def test_crypto_manager_status(self):
        """Test crypto manager status reporting."""
        crypto_manager = get_crypto_manager()
        status = crypto_manager.get_crypto_status()
        assert 'crypto_manager' in status
        assert 'key_management' in status
        assert 'performance_metrics' in status
        assert 'supported_algorithms' in status
        assert 'hash' in status['supported_algorithms']
        assert 'random' in status['supported_algorithms']
        assert 'encryption' in status['supported_algorithms']

class TestHashOperations:
    """Test hash operation standardization."""

    def test_sha256_hashing(self):
        """Test SHA-256 hash operations."""
        test_data = 'Hello, World!'
        hash1 = hash_sha256(test_data)
        assert len(hash1) == 64
        hash2 = hash_sha256(test_data.encode())
        assert hash1 == hash2
        hash3 = hash_sha256(test_data)
        assert hash1 == hash3

    def test_sha256_truncation(self):
        """Test SHA-256 hash truncation."""
        test_data = 'Test truncation'
        hash_8 = hash_sha256(test_data, truncate_length=8)
        hash_16 = hash_sha256(test_data, truncate_length=16)
        hash_full = hash_sha256(test_data)
        assert len(hash_8) == 8
        assert len(hash_16) == 16
        assert len(hash_full) == 64
        assert hash_full.startswith(hash_8)
        assert hash_full.startswith(hash_16)

    def test_md5_hashing(self):
        """Test MD5 hash operations."""
        test_data = 'MD5 test data'
        hash1 = hash_md5(test_data)
        assert len(hash1) == 32
        hash2 = hash_md5(test_data)
        assert hash1 == hash2

    def test_hash_algorithm_enum(self):
        """Test different hash algorithms through manager."""
        crypto_manager = get_crypto_manager()
        test_data = 'Algorithm test'
        sha256_hash = crypto_manager.hash_data(test_data, HashAlgorithm.SHA256)
        sha512_hash = crypto_manager.hash_data(test_data, HashAlgorithm.SHA512)
        md5_hash = crypto_manager.hash_data(test_data, HashAlgorithm.MD5)
        assert len(sha256_hash) == 64
        assert len(sha512_hash) == 128
        assert len(md5_hash) == 32
        assert sha256_hash != sha512_hash != md5_hash

    def test_cache_key_generation(self):
        """Test standardized cache key generation."""
        key1 = generate_cache_key('user', 123, 'action')
        key2 = generate_cache_key('user', 123, 'action')
        key3 = generate_cache_key('user', 124, 'action')
        assert key1 == key2
        assert key1 != key3
        key_short = generate_cache_key('user', 123, 'action', max_length=8)
        assert len(key_short) == 8

    def test_secure_hash_with_salt(self):
        """Test secure hash generation with salt."""
        crypto_manager = get_crypto_manager()
        test_data = 'password123'
        hash1, salt1 = crypto_manager.generate_secure_hash(test_data)
        hash2, salt2 = crypto_manager.generate_secure_hash(test_data)
        assert salt1 != salt2
        assert hash1 != hash2
        assert crypto_manager.verify_hash(test_data, hash1, salt1)
        assert crypto_manager.verify_hash(test_data, hash2, salt2)
        assert not crypto_manager.verify_hash('wrong', hash1, salt1)

class TestRandomGeneration:
    """Test random generation standardization."""

    def test_random_bytes(self):
        """Test random bytes generation."""
        bytes1 = generate_token_bytes(16)
        bytes2 = generate_token_bytes(16)
        assert len(bytes1) == 16
        assert len(bytes2) == 16
        assert bytes1 != bytes2

    def test_random_hex(self):
        """Test random hex generation."""
        hex1 = generate_token_hex(8)
        hex2 = generate_token_hex(8)
        assert len(hex1) == 16
        assert len(hex2) == 16
        assert hex1 != hex2
        int(hex1, 16)
        int(hex2, 16)

    def test_random_urlsafe(self):
        """Test random URL-safe generation."""
        url1 = generate_token_urlsafe(16)
        url2 = generate_token_urlsafe(16)
        assert url1 != url2
        unsafe_chars = {'+', '/', '='}
        assert not any((char in url1 for char in unsafe_chars))
        assert not any((char in url2 for char in unsafe_chars))

    def test_random_type_enum(self):
        """Test different random types through manager."""
        crypto_manager = get_crypto_manager()
        random_bytes = crypto_manager.generate_random(16, RandomType.BYTES)
        random_hex = crypto_manager.generate_random(8, RandomType.HEX)
        random_urlsafe = crypto_manager.generate_random(16, RandomType.URLSAFE)
        random_token = crypto_manager.generate_random(8, RandomType.TOKEN)
        assert isinstance(random_bytes, bytes)
        assert isinstance(random_hex, str)
        assert isinstance(random_urlsafe, str)
        assert isinstance(random_token, str)
        assert len(random_bytes) == 16
        assert len(random_hex) == 16

    def test_session_token_generation(self):
        """Test session token generation."""
        crypto_manager = get_crypto_manager()
        token1 = crypto_manager.generate_session_token()
        token2 = crypto_manager.generate_session_token('custom', 24)
        assert token1.startswith('session_')
        assert token2.startswith('custom_')
        assert token1 != token2
        parts1 = token1.split('_')
        assert len(parts1) == 3

    def test_api_key_generation(self):
        """Test API key generation."""
        crypto_manager = get_crypto_manager()
        key1 = crypto_manager.generate_api_key()
        key2 = crypto_manager.generate_api_key('custom', 40)
        assert key1.startswith('api_')
        assert key2.startswith('custom_')
        assert key1 != key2

class TestEncryptionDecryption:
    """Test encryption/decryption through crypto manager."""

    def test_basic_encryption_decryption(self):
        """Test basic encrypt/decrypt cycle."""
        crypto_manager = get_crypto_manager()
        test_data = 'Secret message for encryption'
        encrypted_data, key_id = crypto_manager.encrypt_string(test_data)
        assert isinstance(encrypted_data, bytes)
        assert isinstance(key_id, str)
        decrypted_text = crypto_manager.decrypt_string(encrypted_data, key_id)
        assert decrypted_text == test_data

    def test_convenience_functions(self):
        """Test convenience encrypt/decrypt functions."""
        test_data = 'Test data for convenience functions'
        encrypted_data, key_id = encrypt_data(test_data)
        decrypted_data = decrypt_data(encrypted_data, key_id)
        assert decrypted_data.decode('utf-8') == test_data

    def test_bytes_encryption(self):
        """Test bytes encryption/decryption."""
        crypto_manager = get_crypto_manager()
        test_bytes = b'Binary data \x00\x01\x02\xff'
        encrypted_data, key_id = crypto_manager.encrypt_data(test_bytes)
        decrypted_bytes = crypto_manager.decrypt_data(encrypted_data, key_id)
        assert decrypted_bytes == test_bytes

    def test_key_rotation_encryption(self):
        """Test encryption with key rotation."""
        crypto_manager = get_crypto_manager()
        test_data = 'Data for key rotation test'
        encrypted1, key_id1 = crypto_manager.encrypt_string(test_data)
        new_key_id = crypto_manager.rotate_keys()
        assert new_key_id != key_id1
        encrypted2, key_id2 = crypto_manager.encrypt_string(test_data)
        assert key_id2 == new_key_id
        assert encrypted1 != encrypted2
        decrypted1 = crypto_manager.decrypt_string(encrypted1, key_id1)
        decrypted2 = crypto_manager.decrypt_string(encrypted2, key_id2)
        assert decrypted1 == test_data
        assert decrypted2 == test_data

class TestSecureComparison:
    """Test secure comparison operations."""

    def test_secure_compare_strings(self):
        """Test secure string comparison."""
        assert secure_compare('test', 'test') is True
        assert secure_compare('test', 'different') is False
        assert secure_compare('', '') is True

    def test_secure_compare_bytes(self):
        """Test secure bytes comparison."""
        bytes1 = b'test_data'
        bytes2 = b'test_data'
        bytes3 = b'different'
        assert secure_compare(bytes1, bytes2) is True
        assert secure_compare(bytes1, bytes3) is False

    def test_secure_compare_mixed(self):
        """Test secure comparison with mixed types."""
        text = 'test_data'
        bytes_data = b'test_data'
        assert secure_compare(text, bytes_data) is True
        assert secure_compare(text, b'different') is False

class TestKeyDerivation:
    """Test key derivation functions."""

    def test_pbkdf2_key_derivation(self):
        """Test PBKDF2 key derivation."""
        crypto_manager = get_crypto_manager()
        password = 'strong_password'
        key1, salt1 = crypto_manager.derive_key(password)
        key2, salt2 = crypto_manager.derive_key(password)
        assert len(key1) == 32
        assert len(key2) == 32
        assert len(salt1) == 32
        assert len(salt2) == 32
        assert salt1 != salt2
        assert key1 != key2

    def test_pbkdf2_with_custom_parameters(self):
        """Test PBKDF2 with custom parameters."""
        crypto_manager = get_crypto_manager()
        password = 'test_password'
        custom_salt = b'custom_salt_16_bytes'
        key, salt = crypto_manager.derive_key(password, salt=custom_salt, length=16, iterations=100000)
        assert len(key) == 16
        assert salt == custom_salt

    def test_scrypt_key_derivation(self):
        """Test Scrypt key derivation."""
        crypto_manager = get_crypto_manager()
        password = 'scrypt_password'
        key1, salt1 = crypto_manager.derive_key_scrypt(password)
        key2, salt2 = crypto_manager.derive_key_scrypt(password)
        assert len(key1) == 32
        assert len(key2) == 32
        assert key1 != key2

    def test_scrypt_with_custom_parameters(self):
        """Test Scrypt with custom parameters."""
        crypto_manager = get_crypto_manager()
        password = 'test_password'
        key, salt = crypto_manager.derive_key_scrypt(password, length=16, n=2 ** 14, r=8, p=1)
        assert len(key) == 16
        assert len(salt) == 32

class TestPerformanceAndMetrics:
    """Test performance monitoring and metrics."""

    def test_performance_metrics(self):
        """Test crypto operation performance tracking."""
        crypto_manager = get_crypto_manager()
        for i in range(10):
            hash_sha256(f'test_data_{i}')
            generate_token_hex(16)
            crypto_manager.encrypt_string(f'message_{i}')
        metrics = crypto_manager.metrics.get_stats()
        assert metrics['total_operations'] > 0
        assert 'operation_counts' in metrics
        assert 'performance_averages' in metrics
        assert any(('hash' in op for op in metrics['operation_counts']))

    def test_concurrent_operations(self):
        """Test crypto manager under concurrent load."""
        crypto_manager = get_crypto_manager()
        results = []
        errors = []

        def worker():
            try:
                for i in range(5):
                    hash_result = hash_sha256(f'data_{threading.current_thread().ident}_{i}')
                    results.append(('hash', hash_result))
                    random_data = generate_token_hex(8)
                    results.append(('random', random_data))
                    encrypted, key_id = crypto_manager.encrypt_string(f'message_{i}')
                    decrypted = crypto_manager.decrypt_string(encrypted, key_id)
                    results.append(('encrypt', decrypted))
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f'Errors in concurrent operations: {errors}'
        assert len(results) > 0
        hash_results = [r[1] for r in results if r[0] == 'hash']
        random_results = [r[1] for r in results if r[0] == 'random']
        assert len(set(random_results)) == len(random_results)

    def test_operation_timing(self):
        """Test operation timing is reasonable."""
        crypto_manager = get_crypto_manager()
        start_time = time.time()
        for _ in range(100):
            hash_sha256('test_data')
        hash_time = time.time() - start_time
        assert hash_time < 1.0, 'Hash operations too slow'
        start_time = time.time()
        for _ in range(50):
            encrypted, key_id = crypto_manager.encrypt_string('test_message')
            crypto_manager.decrypt_string(encrypted, key_id)
        encrypt_time = time.time() - start_time
        assert encrypt_time < 2.0, 'Encryption operations too slow'

class TestSecurityFeatures:
    """Test security features and audit logging."""

    def test_audit_logging(self):
        """Test crypto operation audit logging."""
        crypto_manager = UnifiedCryptoManager(security_level=SecurityLevel.HIGH)
        crypto_manager.audit_enabled = True
        crypto_manager.hash_sha256('audit_test')
        crypto_manager.generate_random_bytes(16)
        crypto_manager.encrypt_string('audit_message')
        status = crypto_manager.get_crypto_status()
        metrics = status['performance_metrics']
        assert metrics['total_operations'] > 0

    def test_security_levels(self):
        """Test different security levels."""
        basic_manager = UnifiedCryptoManager(security_level=SecurityLevel.basic)
        assert basic_manager.security_level == SecurityLevel.basic
        high_manager = UnifiedCryptoManager(security_level=SecurityLevel.HIGH)
        assert high_manager.security_level == SecurityLevel.HIGH
        hash1 = basic_manager.hash_sha256('test')
        hash2 = high_manager.hash_sha256('test')
        assert hash1 == hash2

    def test_error_handling(self):
        """Test crypto error handling."""
        crypto_manager = get_crypto_manager()
        with pytest.raises(ValueError):
            crypto_manager.decrypt_string(b'invalid_data', 'nonexistent_key')
        assert len(crypto_manager.failed_operations) > 0

    def test_key_cleanup(self):
        """Test key cleanup functionality."""
        crypto_manager = get_crypto_manager()
        original_key_count = len(crypto_manager.key_manager.keys)
        for _ in range(5):
            crypto_manager.rotate_keys()
        assert len(crypto_manager.key_manager.keys) > original_key_count
        removed_keys = crypto_manager.cleanup_old_keys(keep_count=2)
        assert len(crypto_manager.key_manager.keys) <= original_key_count + 2

class TestMigrationCompatibility:
    """Test backward compatibility and migration patterns."""

    def test_migration_function_compatibility(self):
        """Test that migration functions work like originals."""
        test_data = 'migration_test_data'
        sha256_result = hash_sha256(test_data)
        md5_result = hash_md5(test_data)
        assert isinstance(sha256_result, str)
        assert isinstance(md5_result, str)
        assert len(sha256_result) == 64
        assert len(md5_result) == 32
        random_bytes = generate_token_bytes(16)
        random_hex = generate_token_hex(8)
        random_urlsafe = generate_token_urlsafe(16)
        assert isinstance(random_bytes, bytes)
        assert isinstance(random_hex, str)
        assert isinstance(random_urlsafe, str)

    def test_cache_key_patterns(self):
        """Test common cache key generation patterns."""
        key1 = generate_cache_key('user', 123)
        assert isinstance(key1, str)
        assert len(key1) == 16
        key2 = generate_cache_key('action', 'user_123', 'timestamp', 1234567890)
        assert isinstance(key2, str)
        key3 = generate_cache_key('test', 'key', max_length=8)
        assert len(key3) == 8

    def test_encryption_patterns(self):
        """Test common encryption patterns."""
        data = 'sensitive_information'
        encrypted, key_id = encrypt_data(data)
        decrypted = decrypt_data(encrypted, key_id)
        assert decrypted.decode('utf-8') == data
        binary_data = b'binary\x00\x01\x02data'
        encrypted_bin, key_id_bin = encrypt_data(binary_data)
        decrypted_bin = decrypt_data(encrypted_bin, key_id_bin)
        assert decrypted_bin == binary_data
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
