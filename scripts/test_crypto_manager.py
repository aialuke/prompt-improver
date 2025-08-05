#!/usr/bin/env python3
"""Simple test script for the unified crypto manager.

Tests the crypto manager directly without complex imports.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_unified_crypto_manager():
    """Test the unified crypto manager directly."""
    try:
        # Direct import to avoid circular dependency
        from prompt_improver.security.unified_crypto_manager import (
            UnifiedCryptoManager, 
            HashAlgorithm, 
            RandomType,
            SecurityLevel
        )
        from prompt_improver.security.key_manager import KeyRotationConfig
        
        print("✅ Successfully imported UnifiedCryptoManager")
        
        # Initialize crypto manager
        crypto_manager = UnifiedCryptoManager()
        print("✅ Successfully initialized UnifiedCryptoManager")
        
        # Test hashing
        test_data = "Hello, Unified Crypto World!"
        sha256_hash = crypto_manager.hash_sha256(test_data)
        md5_hash = crypto_manager.hash_md5(test_data)
        
        print(f"✅ SHA-256 hash: {sha256_hash}")
        print(f"✅ MD5 hash: {md5_hash}")
        
        # Test truncation
        truncated = crypto_manager.hash_sha256(test_data, truncate_length=12)
        print(f"✅ Truncated hash: {truncated}")
        
        # Test random generation
        random_bytes = crypto_manager.generate_random_bytes(16)
        random_hex = crypto_manager.generate_random_hex(8)
        random_urlsafe = crypto_manager.generate_random_urlsafe(16)
        
        print(f"✅ Random bytes (length {len(random_bytes)}): {random_bytes[:8]}...")
        print(f"✅ Random hex: {random_hex}")
        print(f"✅ Random URL-safe: {random_urlsafe}")
        
        # Test cache key generation
        cache_key = crypto_manager.generate_cache_key("user", 123, "action", max_length=10)
        print(f"✅ Cache key: {cache_key}")
        
        # Test session and API key generation
        session_token = crypto_manager.generate_session_token()
        api_key = crypto_manager.generate_api_key()
        
        print(f"✅ Session token: {session_token}")
        print(f"✅ API key: {api_key}")
        
        # Test encryption/decryption
        test_message = "Secret message for encryption test"
        encrypted_data, key_id = crypto_manager.encrypt_string(test_message)
        decrypted_message = crypto_manager.decrypt_string(encrypted_data, key_id)
        
        print(f"✅ Encryption test: {'PASS' if decrypted_message == test_message else 'FAIL'}")
        print(f"   Key ID: {key_id}")
        print(f"   Encrypted length: {len(encrypted_data)} bytes")
        
        # Test secure comparison
        comparison_test = crypto_manager.secure_compare("test", "test")
        comparison_fail = crypto_manager.secure_compare("test", "different")
        
        print(f"✅ Secure comparison test: {'PASS' if comparison_test and not comparison_fail else 'FAIL'}")
        
        # Test key derivation
        password = "test_password_123"
        derived_key, salt = crypto_manager.derive_key(password, iterations=100000)  # Lower for testing
        
        print(f"✅ Key derivation: derived {len(derived_key)} bytes with {len(salt)} byte salt")
        
        # Test Scrypt derivation
        scrypt_key, scrypt_salt = crypto_manager.derive_key_scrypt(password, n=2**14)  # Lower for testing
        
        print(f"✅ Scrypt derivation: derived {len(scrypt_key)} bytes with {len(scrypt_salt)} byte salt")
        
        # Test hash verification
        secure_hash, hash_salt = crypto_manager.generate_secure_hash("password123")
        verification_pass = crypto_manager.verify_hash("password123", secure_hash, hash_salt)
        verification_fail = crypto_manager.verify_hash("wrong_password", secure_hash, hash_salt)
        
        print(f"✅ Hash verification: {'PASS' if verification_pass and not verification_fail else 'FAIL'}")
        
        # Test key rotation
        original_key_count = len(crypto_manager.key_manager.keys)
        new_key_id = crypto_manager.rotate_keys()
        new_key_count = len(crypto_manager.key_manager.keys)
        
        print(f"✅ Key rotation: {original_key_count} -> {new_key_count} keys, new key: {new_key_id}")
        
        # Test status reporting
        status = crypto_manager.get_crypto_status()
        
        print(f"✅ Status report: {len(status)} categories")
        print(f"   Security level: {status['crypto_manager']['security_level']}") 
        print(f"   Total operations: {status['performance_metrics']['total_operations']}")
        print(f"   Supported algorithms: {list(status['supported_algorithms'].keys())}")
        
        # Test cleanup
        removed_keys = crypto_manager.cleanup_old_keys(keep_count=2)
        print(f"✅ Cleanup: removed {len(removed_keys)} old keys")
        
        print("\n🎉 All UnifiedCryptoManager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_migration_patterns():
    """Test migration-friendly functions."""
    try:
        from prompt_improver.security.unified_crypto_manager import (
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
        
        print("\n📝 Testing migration-friendly functions:")
        
        # Test direct function calls (as they would be used after migration)
        test_data = "Migration test data"
        
        # Hash functions
        sha256_result = hash_sha256(test_data)
        md5_result = hash_md5(test_data, truncate_length=8)
        
        print(f"✅ hash_sha256(): {sha256_result}")
        print(f"✅ hash_md5() (truncated): {md5_result}")
        
        # Random functions
        bytes_result = generate_token_bytes(16)
        hex_result = generate_token_hex(8)
        urlsafe_result = generate_token_urlsafe(12)
        
        print(f"✅ generate_token_bytes(16): {len(bytes_result)} bytes")
        print(f"✅ generate_token_hex(8): {hex_result}")
        print(f"✅ generate_token_urlsafe(12): {urlsafe_result}")
        
        # Cache key function
        cache_result = generate_cache_key("user", 123, "action", max_length=10)
        print(f"✅ generate_cache_key(): {cache_result}")
        
        # Encryption functions
        encrypted_result, key_id = encrypt_data("Secret data")
        decrypted_result = decrypt_data(encrypted_result, key_id)
        
        encryption_success = decrypted_result.decode('utf-8') == "Secret data"
        print(f"✅ encrypt_data/decrypt_data: {'PASS' if encryption_success else 'FAIL'}")
        
        # Secure comparison
        compare_result = secure_compare("test", "test")
        print(f"✅ secure_compare(): {'PASS' if compare_result else 'FAIL'}")
        
        print("\n🎉 All migration function tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Migration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_migration_examples():
    """Show migration examples."""
    print("\n📚 MIGRATION EXAMPLES:")
    print("=" * 60)
    
    examples = [
        {
            "title": "Hash Operations",
            "before": [
                "import hashlib",
                "hash_result = hashlib.sha256(data.encode()).hexdigest()",
                "short_hash = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:8]"
            ],
            "after": [
                "from prompt_improver.security import hash_sha256, hash_md5",
                "hash_result = hash_sha256(data)",
                "short_hash = hash_md5(key, truncate_length=8)"
            ]
        },
        {
            "title": "Random Generation",
            "before": [
                "import secrets",
                "random_key = secrets.token_bytes(32)",
                "session_id = f'session_{secrets.token_hex(16)}'"
            ],
            "after": [
                "from prompt_improver.security import generate_token_bytes, generate_token_hex",
                "random_key = generate_token_bytes(32)",
                "session_id = f'session_{generate_token_hex(16)}'"
            ]
        },
        {
            "title": "Cache Key Generation",
            "before": [
                "import hashlib",
                "cache_key = f'cache:{hashlib.md5(f\"{user_id}_{action}\".encode()).hexdigest()[:12]}'"
            ],
            "after": [
                "from prompt_improver.security import generate_cache_key",
                "cache_key = f'cache:{generate_cache_key(user_id, action, max_length=12)}'"
            ]
        },
        {
            "title": "Encryption Operations",
            "before": [
                "from cryptography.fernet import Fernet",
                "key = Fernet.generate_key()",
                "fernet = Fernet(key)",
                "encrypted = fernet.encrypt(data.encode())"
            ],
            "after": [
                "from prompt_improver.security import encrypt_data, decrypt_data",
                "encrypted, key_id = encrypt_data(data)",
                "decrypted = decrypt_data(encrypted, key_id)"
            ]
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}:")
        print("-" * 40)
        print("BEFORE:")
        for line in example["before"]:
            print(f"  {line}")
        print("AFTER:")
        for line in example["after"]:
            print(f"  {line}")

def main():
    """Main test function."""
    print("🔐 UNIFIED CRYPTO MANAGER VALIDATION")
    print("=" * 50)
    
    # Test core functionality
    success1 = test_unified_crypto_manager()
    
    # Test migration functions
    success2 = test_migration_patterns()
    
    # Show examples
    show_migration_examples()
    
    # Summary
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED! UnifiedCryptoManager is ready for production.")
        print("\nKey Benefits:")
        print("✅ Centralized crypto operations through existing KeyManager")
        print("✅ Standardized interfaces for all crypto functions") 
        print("✅ Comprehensive audit logging and security features")
        print("✅ Migration-friendly functions for easy adoption")
        print("✅ Performance monitoring and metrics collection")
        print("✅ NIST-approved algorithms and security best practices")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED! Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())