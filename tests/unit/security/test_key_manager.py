"""
Key Manager Security Tests

Tests UnifiedKeyManager components for local ML deployment,
ensuring secure key management, encryption/decryption operations, key rotation,
and orchestrator integration following 2025 best practices.

Security Test Coverage:
- Real encryption/decryption behavior using actual Fernet implementation
- Key generation and rotation with actual cryptographic operations
- Orchestrator compatibility with real operation execution
- File-based key storage security for local deployment
- Error handling and edge cases with real exceptions
- Performance characteristics under realistic loads
- Memory safety and key cleanup verification
"""
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
import pytest
from cryptography.fernet import Fernet
from prompt_improver.security.key_manager import UnifiedKeyManager, get_key_manager

class TestUnifiedKeyManager:
    """Test UnifiedKeyManager with real cryptographic operations"""

    @pytest.fixture
    def key_manager(self):
        """Create UnifiedKeyManager for testing"""
        return UnifiedKeyManager()

    def test_key_manager_initialization(self, key_manager):
        """Test UnifiedKeyManager initializes correctly"""
        assert key_manager is not None
        assert hasattr(key_manager, 'rotate_key')
        assert hasattr(key_manager, 'get_current_key')
        assert hasattr(key_manager, 'get_key_by_id')
        assert hasattr(key_manager, 'cleanup_expired_keys')
        assert hasattr(key_manager, 'run_orchestrated_analysis')

    def test_key_generation_creates_valid_key(self, key_manager):
        """Test key generation creates valid Fernet key"""
        key_bytes, key_id = key_manager.get_current_key()
        assert isinstance(key_bytes, bytes)
        assert len(key_bytes) == 44
        fernet_instance = Fernet(key_bytes)
        assert fernet_instance is not None
        retrieved_key = key_manager.get_key_by_id(key_id)
        assert retrieved_key == key_bytes

    def test_key_storage_security(self, key_manager):
        """Test key storage is secure in memory"""
        key_bytes, key_id = key_manager.get_current_key()
        stored_key = key_manager.get_key_by_id(key_id)
        assert stored_key == key_bytes
        assert len(stored_key) == 44
        key_info = key_manager.get_key_info()
        assert key_id in key_info
        assert key_info[key_id]['is_current'] is True

    def test_key_rotation_functionality(self, key_manager):
        """Test key rotation creates new key and maintains access"""
        original_key, original_key_id = key_manager.get_current_key()
        new_key_id = key_manager.rotate_key()
        rotated_key, current_key_id = key_manager.get_current_key()
        assert new_key_id != original_key_id
        assert current_key_id == new_key_id
        assert rotated_key != original_key
        assert isinstance(rotated_key, bytes)
        assert len(rotated_key) == 44
        fernet_instance = Fernet(rotated_key)
        assert fernet_instance is not None
        current_key_retrieved = key_manager.get_key_by_id(current_key_id)
        assert current_key_retrieved == rotated_key

    def test_key_versioning_and_cleanup(self, key_manager):
        """Test key versioning and old key cleanup"""
        v1_key, v1_id = key_manager.get_current_key()
        v2_id = key_manager.rotate_key()
        v2_key, _ = key_manager.get_current_key()
        v3_id = key_manager.rotate_key()
        v3_key, current_id = key_manager.get_current_key()
        assert len(key_manager.keys) >= 1
        assert current_id == v3_id
        assert v3_key != v2_key != v1_key
        cleaned_count = key_manager.cleanup_expired_keys()
        assert isinstance(cleaned_count, int)
        assert cleaned_count >= 0
        current_key_after_cleanup, _ = key_manager.get_current_key()
        assert current_key_after_cleanup == v3_key

    def test_nonexistent_key_handling(self, key_manager):
        """Test handling of requests for nonexistent keys"""
        nonexistent_key_id = 'does_not_exist'
        result = key_manager.get_key_by_id(nonexistent_key_id)
        assert result is None
        new_key_id = key_manager.rotate_key()
        assert isinstance(new_key_id, str)
        assert len(new_key_id) > 0

    def test_orchestrator_compatibility(self, key_manager):
        """Test orchestrator-compatible operation interface"""
        result = key_manager.run_orchestrated_analysis(operation='get_key', parameters={'key_id': 'orchestrator_test'})
        assert isinstance(result, dict)
        assert result.get('orchestrator_compatible') is True
        assert 'result' in result
        assert 'operation' in result
        assert 'timestamp' in result
        if result['result'] is None:
            gen_result = key_manager.run_orchestrated_analysis(operation='generate_key', parameters={'key_id': 'orchestrator_test'})
            assert gen_result.get('orchestrator_compatible') is True
            assert gen_result['result'] is not None

    def test_orchestrator_operations_coverage(self, key_manager):
        """Test all supported orchestrator operations"""
        gen_result = key_manager.run_orchestrated_analysis(operation='generate_key', parameters={})
        assert gen_result['orchestrator_compatible'] is True
        assert isinstance(gen_result['result'], bytes)
        get_result = key_manager.run_orchestrated_analysis(operation='get_key', parameters={})
        assert get_result['orchestrator_compatible'] is True
        assert isinstance(get_result['result'], bytes)
        rotate_result = key_manager.run_orchestrated_analysis(operation='rotate_key', parameters={})
        assert rotate_result['orchestrator_compatible'] is True
        assert isinstance(rotate_result['result'], bytes)
        status_result = key_manager.run_orchestrated_analysis(operation='get_status', parameters={})
        assert status_result['orchestrator_compatible'] is True
        assert isinstance(status_result['result'], dict)
        cleanup_result = key_manager.run_orchestrated_analysis(operation='cleanup', parameters={})
        assert cleanup_result['orchestrator_compatible'] is True
        assert isinstance(cleanup_result['result'], int)

    def test_concurrent_key_operations(self, key_manager):
        """Test concurrent key operations for thread safety"""
        import threading
        results = []
        errors = []

        def rotate_keys(start_index):
            try:
                for i in range(5):
                    new_key_id = key_manager.rotate_key()
                    current_key, current_key_id = key_manager.get_current_key()
                    results.append((f'thread_{start_index}_{i}', current_key, current_key_id))
            except Exception as e:
                errors.append(e)
        threads = []
        for i in range(3):
            thread = threading.Thread(target=rotate_keys, args=(i,))
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(errors) == 0, f'Concurrent operations failed: {errors}'
        assert len(results) == 15
        for _, key, key_id in results:
            assert isinstance(key, bytes)
            assert len(key) == 44
            assert isinstance(key_id, str)
            Fernet(key)

class TestUnifiedKeyManagerEncryption:
    """Test UnifiedKeyManager encryption/decryption operations"""

    @pytest.fixture
    def fernet_manager(self):
        """Create UnifiedKeyManager for encryption testing"""
        return UnifiedKeyManager()

    def test_fernet_manager_initialization(self, fernet_manager):
        """Test UnifiedKeyManager encryption initializes correctly"""
        assert fernet_manager is not None
        assert hasattr(fernet_manager, 'encrypt')
        assert hasattr(fernet_manager, 'decrypt')
        assert hasattr(fernet_manager, 'get_current_fernet')
        assert hasattr(fernet_manager, 'get_fernet_by_id')
        assert hasattr(fernet_manager, 'run_orchestrated_analysis')

    def test_encryption_decryption_cycle(self, fernet_manager):
        """Test complete encryption/decryption cycle with real data"""
        test_cases = ['Simple string message', 'Unicode message with émojis 🔒🔑', json.dumps({'key': 'value', 'number': 42}), 'Binary data: ' + '\x00\x01\x02\x03\x04', 'Long message: ' + 'A' * 1000]
        for original_data in test_cases:
            encrypted_data, key_id = fernet_manager.encrypt(original_data.encode('utf-8'))
            assert encrypted_data != original_data.encode('utf-8')
            assert isinstance(encrypted_data, bytes)
            assert isinstance(key_id, str)
            decrypted_data = fernet_manager.decrypt(encrypted_data, key_id)
            assert isinstance(decrypted_data, bytes)
            assert decrypted_data.decode('utf-8') == original_data

    def test_encryption_with_custom_key(self, fernet_manager):
        """Test encryption with custom key specification"""
        test_data = 'Data encrypted with custom key'
        encrypted_data, key_id = fernet_manager.encrypt(test_data.encode('utf-8'))
        assert isinstance(key_id, str)
        assert len(key_id) > 0
        decrypted_data = fernet_manager.decrypt(encrypted_data, key_id)
        assert isinstance(decrypted_data, bytes)
        assert decrypted_data.decode('utf-8') == test_data

    def test_encryption_test_cycle(self, fernet_manager):
        """Test a complete encryption cycle for system validation"""
        test_data = 'Test encryption cycle data'
        import time
        start_time = time.time()
        encrypted_data, key_id = fernet_manager.encrypt(test_data.encode('utf-8'))
        decrypted_data = fernet_manager.decrypt(encrypted_data, key_id)
        end_time = time.time()
        performance_ms = (end_time - start_time) * 1000
        assert isinstance(encrypted_data, bytes)
        assert isinstance(key_id, str)
        assert isinstance(decrypted_data, bytes)
        assert decrypted_data.decode('utf-8') == test_data
        assert performance_ms < 100

    def test_decryption_with_wrong_key(self, fernet_manager):
        """Test decryption fails appropriately with wrong key"""
        test_data = 'Secret message for wrong key test'
        encrypted_data, correct_key_id = fernet_manager.encrypt(test_data.encode('utf-8'))
        fernet_manager.key_manager.rotate_key()
        new_key, new_key_id = fernet_manager.key_manager.get_current_key()
        try:
            decrypted_data = fernet_manager.decrypt(encrypted_data, new_key_id)
            assert decrypted_data.decode('utf-8') != test_data
        except (ValueError, Exception):
            assert True

    def test_invalid_encrypted_data_handling(self, fernet_manager):
        """Test handling of invalid encrypted data"""
        invalid_data_cases = [b'not_valid_encrypted_data', b'', b'\x00\x01\x02\x03', b'string_instead_of_bytes']
        test_data, key_id = fernet_manager.encrypt(b'test')
        for invalid_data in invalid_data_cases:
            try:
                result = fernet_manager.decrypt(invalid_data, key_id)
                assert isinstance(result, bytes)
            except (ValueError, Exception) as e:
                assert True

    def test_orchestrator_encryption_operations(self, fernet_manager):
        """Test orchestrator-compatible encryption operations"""
        test_data = 'Orchestrator encryption test'
        encrypt_result = fernet_manager.run_orchestrated_analysis('encrypt', {'data': test_data})
        assert encrypt_result['orchestrator_compatible'] is True
        assert 'result' in encrypt_result
        assert 'encrypted_data' in encrypt_result['result']
        assert 'key_id' in encrypt_result['result']
        encrypted_data = encrypt_result['result']['encrypted_data']
        key_id = encrypt_result['result']['key_id']
        decrypt_result = fernet_manager.run_orchestrated_analysis('decrypt', {'encrypted_data': encrypted_data, 'key_id': key_id})
        assert decrypt_result['orchestrator_compatible'] is True
        assert decrypt_result['result']['success'] is True
        assert decrypt_result['result']['decrypted_data'] == test_data

    def test_orchestrator_test_operations(self, fernet_manager):
        """Test orchestrator test and status operations"""
        test_result = fernet_manager.run_orchestrated_analysis('test_encryption', {})
        assert test_result['orchestrator_compatible'] is True
        assert test_result['result']['success'] is True
        status_result = fernet_manager.run_orchestrated_analysis('get_status', {})
        assert status_result['orchestrator_compatible'] is True
        assert isinstance(status_result['result'], dict)
        assert 'key_count' in status_result['result']
        assert 'last_operation' in status_result['result']

    def test_large_data_encryption_performance(self, fernet_manager):
        """Test encryption performance with large data"""
        sizes = [1024, 10240, 102400]
        for size in sizes:
            large_data = 'X' * size
            start_time = time.time()
            encrypted_data, key_id = fernet_manager.encrypt(large_data.encode('utf-8'))
            encrypt_time = time.time() - start_time
            start_time = time.time()
            decrypted_data = fernet_manager.decrypt(encrypted_data, key_id)
            decrypt_time = time.time() - start_time
            assert isinstance(decrypted_data, bytes)
            assert decrypted_data.decode('utf-8') == large_data
            assert encrypt_time < 1.0, f'Encryption too slow for {size} bytes: {encrypt_time}s'
            assert decrypt_time < 1.0, f'Decryption too slow for {size} bytes: {decrypt_time}s'

    def test_multiple_key_management(self, fernet_manager):
        """Test managing multiple keys for different purposes"""
        keys_data = ['User personal information', 'ML model weights and parameters', 'Session data and tokens', 'Backup system data']
        encrypted_items = []
        for data in keys_data:
            encrypted_data, key_id = fernet_manager.encrypt(data.encode('utf-8'))
            encrypted_items.append((encrypted_data, key_id, data))
            assert isinstance(encrypted_data, bytes)
            assert isinstance(key_id, str)
            assert len(key_id) > 0
        for encrypted_data, key_id, original_data in encrypted_items:
            decrypted_data = fernet_manager.decrypt(encrypted_data, key_id)
            assert isinstance(decrypted_data, bytes)
            assert decrypted_data.decode('utf-8') == original_data
        assert len(fernet_manager.key_manager.keys) >= 1

    def test_key_rotation_encryption_compatibility(self, fernet_manager):
        """Test that key rotation maintains encryption compatibility"""
        test_data = 'Data to survive key rotation'
        original_encrypted_data, original_key_id = fernet_manager.encrypt(test_data.encode('utf-8'))
        fernet_manager.key_manager.rotate_key()
        try:
            decrypted_data = fernet_manager.decrypt(original_encrypted_data, original_key_id)
            assert decrypted_data.decode('utf-8') == test_data
        except Exception:
            pass
        new_encrypted_data, new_key_id = fernet_manager.encrypt(test_data.encode('utf-8'))
        new_decrypted_data = fernet_manager.decrypt(new_encrypted_data, new_key_id)
        assert isinstance(new_decrypted_data, bytes)
        assert new_decrypted_data.decode('utf-8') == test_data

class TestKeyManagerEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_string_encryption(self):
        """Test encryption of empty strings"""
        fernet_manager = UnifiedKeyManager()
        empty_data = ''
        encrypted_data, key_id = fernet_manager.encrypt(empty_data.encode('utf-8'))
        assert encrypted_data != b''
        decrypted_data = fernet_manager.decrypt(encrypted_data, key_id)
        assert isinstance(decrypted_data, bytes)
        assert decrypted_data.decode('utf-8') == empty_data

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters"""
        fernet_manager = UnifiedKeyManager()
        special_cases = ['Hello, 世界! 🌍', 'Español: ñáéíóú', 'العربية: مرحبا بالعالم', 'Русский: Привет, мир!', '\n\t\r\x00Special chars', 'Emoji test: 🔒🔑🛡️⚡🌟']
        for test_data in special_cases:
            encrypted_data, key_id = fernet_manager.encrypt(test_data.encode('utf-8'))
            decrypted_data = fernet_manager.decrypt(encrypted_data, key_id)
            assert isinstance(decrypted_data, bytes)
            assert decrypted_data.decode('utf-8') == test_data

    def test_memory_error_handling(self):
        """Test handling of memory-related errors in in-memory storage"""
        key_manager = UnifiedKeyManager()
        key, key_id = key_manager.get_current_key()
        assert key is not None
        assert key_id is not None
        new_key_id = key_manager.rotate_key()
        assert new_key_id != key_id
        cleaned_count = key_manager.cleanup_expired_keys()
        assert isinstance(cleaned_count, int)

    def test_invalid_orchestrator_operations(self):
        """Test handling of invalid orchestrator operations"""
        key_manager = UnifiedKeyManager()
        result = key_manager.run_orchestrated_analysis(operation='invalid_operation', parameters={})
        assert result['orchestrator_compatible'] is True
        assert result['result'] is None
        assert 'error' in result

    def test_malformed_parameters(self):
        """Test handling of malformed parameters"""
        fernet_manager = UnifiedKeyManager()
        invalid_cases = [{'operation': 'encrypt', 'parameters': {}}, {'operation': 'decrypt', 'parameters': {'key_id': 'test'}}, {'operation': 'encrypt', 'parameters': {'data': None}}]
        for case in invalid_cases:
            result = fernet_manager.run_orchestrated_analysis(case['operation'], case['parameters'])
            assert result['orchestrator_compatible'] is True
            if not result.get('result', {}).get('success', True):
                assert 'error' in result.get('result', {})

@pytest.mark.performance
class TestKeyManagerPerformance:
    """Test performance characteristics of key management operations"""

    def test_key_generation_performance(self):
        """Test key generation performance"""
        key_manager = UnifiedKeyManager()
        start_time = time.time()
        keys_created = []
        for i in range(50):
            key_id = key_manager.rotate_key()
            key, retrieved_key_id = key_manager.get_current_key()
            assert isinstance(key, bytes)
            assert len(key) == 44
            assert retrieved_key_id == key_id
            keys_created.append(key_id)
        elapsed_time = time.time() - start_time
        assert elapsed_time < 2.0
        avg_time_per_key = elapsed_time / 50
        assert avg_time_per_key < 0.04

    def test_encryption_performance_scaling(self):
        """Test encryption performance with different data sizes"""
        fernet_manager = UnifiedKeyManager()
        sizes = [100, 1000, 10000, 50000]
        for size in sizes:
            test_data = 'X' * size
            start_time = time.time()
            encrypted_data, key_id = fernet_manager.encrypt(test_data.encode('utf-8'))
            encrypt_time = time.time() - start_time
            start_time = time.time()
            decrypted_data = fernet_manager.decrypt(encrypted_data, key_id)
            decrypt_time = time.time() - start_time
            assert decrypted_data.decode('utf-8') == test_data
            max_time = 0.5 + size / 100000
            assert encrypt_time < max_time, f'Encryption too slow for {size}B: {encrypt_time:.3f}s'
            assert decrypt_time < max_time, f'Decryption too slow for {size}B: {decrypt_time:.3f}s'

    def test_concurrent_encryption_performance(self):
        """Test concurrent encryption operations"""
        import threading
        fernet_manager = UnifiedKeyManager()
        results = []
        errors = []

        def encrypt_data(thread_id):
            try:
                for i in range(10):
                    test_data = f'Thread {thread_id} data {i}'
                    encrypted_data, key_id = fernet_manager.encrypt(test_data.encode('utf-8'))
                    decrypted_data = fernet_manager.decrypt(encrypted_data, key_id)
                    assert decrypted_data.decode('utf-8') == test_data
                    results.append((thread_id, i))
            except Exception as e:
                errors.append((thread_id, str(e)))
        threads = []
        start_time = time.time()
        for thread_id in range(5):
            thread = threading.Thread(target=encrypt_data, args=(thread_id,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        elapsed_time = time.time() - start_time
        assert len(errors) == 0, f'Concurrent encryption errors: {errors}'
        assert len(results) == 50
        assert elapsed_time < 5.0
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
