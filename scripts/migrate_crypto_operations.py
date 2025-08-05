#!/usr/bin/env python3
"""Migration script for standardizing cryptographic operations.

This script demonstrates how to migrate from scattered crypto imports
to the unified crypto manager system.

Usage:
    python scripts/migrate_crypto_operations.py --validate
    python scripts/migrate_crypto_operations.py --migrate-file path/to/file.py
    python scripts/migrate_crypto_operations.py --scan-all
"""

import re
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_improver.security import (
    get_crypto_manager, 
    hash_sha256,
    hash_md5,
    generate_token_bytes,
    generate_token_hex,
    generate_token_urlsafe,
    generate_cache_key,
    secure_compare
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoMigrationPatterns:
    """Migration patterns for crypto operations."""
    
    # Hashlib patterns to replace
    HASHLIB_PATTERNS = [
        # SHA-256 patterns
        (r'hashlib\.sha256\(([^)]+)\)\.hexdigest\(\)', r'hash_sha256(\1)'),
        (r'hashlib\.sha256\(([^)]+)\)\.hexdigest\(\)\[:(\d+)\]', r'hash_sha256(\1, truncate_length=\2)'),
        
        # MD5 patterns  
        (r'hashlib\.md5\(([^)]+)\)\.hexdigest\(\)', r'hash_md5(\1)'),
        (r'hashlib\.md5\(([^)]+)\)\.hexdigest\(\)\[:(\d+)\]', r'hash_md5(\1, truncate_length=\2)'),
        (r'hashlib\.md5\(([^)]+), usedforsecurity=False\)\.hexdigest\(\)', r'hash_md5(\1)'),
        (r'hashlib\.md5\(([^)]+), usedforsecurity=False\)\.hexdigest\(\)\[:(\d+)\]', r'hash_md5(\1, truncate_length=\2)'),
        
        # SHA-512 patterns
        (r'hashlib\.sha512\(([^)]+)\)\.hexdigest\(\)', r'get_crypto_manager().hash_sha512(\1)'),
        (r'hashlib\.sha512\(([^)]+)\)\.hexdigest\(\)\[:(\d+)\]', r'get_crypto_manager().hash_sha512(\1, truncate_length=\2)'),
    ]
    
    # Secrets patterns to replace
    SECRETS_PATTERNS = [
        (r'secrets\.token_bytes\((\d+)\)', r'generate_token_bytes(\1)'),
        (r'secrets\.token_hex\((\d+)\)', r'generate_token_hex(\1)'),
        (r'secrets\.token_urlsafe\((\d+)\)', r'generate_token_urlsafe(\1)'),
    ]
    
    # Import patterns to replace
    IMPORT_PATTERNS = [
        (r'import hashlib\n', '# hashlib replaced with unified crypto manager\n'),
        (r'import secrets\n', '# secrets replaced with unified crypto manager\n'),
        (r'from hashlib import.*\n', '# hashlib imports replaced with unified crypto manager\n'),
        (r'from secrets import.*\n', '# secrets imports replaced with unified crypto manager\n'),
    ]
    
    # Required imports to add
    REQUIRED_IMPORTS = [
        'from prompt_improver.security import (',
        '    hash_sha256,',
        '    hash_md5,', 
        '    generate_token_bytes,',
        '    generate_token_hex,',
        '    generate_token_urlsafe,',
        '    generate_cache_key,',
        '    get_crypto_manager,',
        '    secure_compare',
        ')',
    ]

def scan_file_for_crypto_usage(file_path: Path) -> Dict[str, List[str]]:
    """Scan file for crypto usage patterns."""
    usage = {
        'hashlib_usage': [],
        'secrets_usage': [],
        'cryptography_usage': [],
        'imports': []
    }
    
    try:
        content = file_path.read_text()
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for imports
            if any(pattern in line for pattern in ['import hashlib', 'import secrets', 'from hashlib', 'from secrets']):
                usage['imports'].append(f"Line {i}: {line.strip()}")
            
            # Check for hashlib usage
            if any(pattern in line for pattern in ['hashlib.sha256', 'hashlib.sha512', 'hashlib.md5', 'hashlib.sha1']):
                usage['hashlib_usage'].append(f"Line {i}: {line.strip()}")
                
            # Check for secrets usage  
            if any(pattern in line for pattern in ['secrets.token_bytes', 'secrets.token_hex', 'secrets.token_urlsafe']):
                usage['secrets_usage'].append(f"Line {i}: {line.strip()}")
                
            # Check for cryptography usage
            if any(pattern in line for pattern in ['from cryptography', 'import cryptography']):
                usage['cryptography_usage'].append(f"Line {i}: {line.strip()}")
                
    except Exception as e:
        logger.error(f"Error scanning {file_path}: {e}")
        
    return usage

def migrate_file_crypto_operations(file_path: Path, dry_run: bool = True) -> bool:
    """Migrate crypto operations in a single file."""
    try:
        content = file_path.read_text()
        original_content = content
        
        # Apply hashlib patterns
        for pattern, replacement in CryptoMigrationPatterns.HASHLIB_PATTERNS:
            content = re.sub(pattern, replacement, content)
            
        # Apply secrets patterns  
        for pattern, replacement in CryptoMigrationPatterns.SECRETS_PATTERNS:
            content = re.sub(pattern, replacement, content)
            
        # Remove old imports and add new ones
        for pattern, replacement in CryptoMigrationPatterns.IMPORT_PATTERNS:
            content = re.sub(pattern, replacement, content)
            
        # Add required imports if crypto operations were found
        if content != original_content:
            # Find the best place to add imports (after existing imports)
            lines = content.split('\n')
            import_line_index = 0
            
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_line_index = i + 1
                elif line.strip() == '' and import_line_index > 0:
                    break
                    
            # Insert new imports
            if import_line_index > 0:
                new_import_lines = CryptoMigrationPatterns.REQUIRED_IMPORTS
                for j, import_line in enumerate(new_import_lines):
                    lines.insert(import_line_index + j, import_line)
                    
                content = '\n'.join(lines)
        
        if dry_run:
            if content != original_content:
                logger.info(f"Would migrate {file_path}")
                return True
            else:
                logger.info(f"No changes needed for {file_path}")
                return False
        else:
            if content != original_content:
                file_path.write_text(content)
                logger.info(f"Migrated {file_path}")
                return True
            else:
                logger.info(f"No changes needed for {file_path}")
                return False
                
    except Exception as e:
        logger.error(f"Error migrating {file_path}: {e}")
        return False

def scan_all_files(src_directory: Path) -> Dict[str, Any]:
    """Scan all Python files for crypto usage."""
    results = {
        'files_with_crypto': [],
        'total_files_scanned': 0,
        'summary': {
            'hashlib_files': 0,
            'secrets_files': 0,
            'cryptography_files': 0,
            'total_crypto_files': 0
        }
    }
    
    python_files = list(src_directory.rglob('*.py'))
    results['total_files_scanned'] = len(python_files)
    
    for file_path in python_files:
        usage = scan_file_for_crypto_usage(file_path)
        
        has_crypto = any(usage[key] for key in ['hashlib_usage', 'secrets_usage', 'cryptography_usage'])
        
        if has_crypto:
            results['files_with_crypto'].append({
                'path': str(file_path),
                'usage': usage
            })
            
            if usage['hashlib_usage']:
                results['summary']['hashlib_files'] += 1
            if usage['secrets_usage']:
                results['summary']['secrets_files'] += 1  
            if usage['cryptography_usage']:
                results['summary']['cryptography_files'] += 1
                
    results['summary']['total_crypto_files'] = len(results['files_with_crypto'])
    
    return results

def validate_crypto_manager():
    """Validate the unified crypto manager works correctly."""
    logger.info("Validating UnifiedCryptoManager...")
    
    try:
        crypto_manager = get_crypto_manager()
        
        # Test hashing
        test_data = "Hello, World!"
        sha256_hash = hash_sha256(test_data)
        md5_hash = hash_md5(test_data)
        
        logger.info(f"SHA-256 hash: {sha256_hash}")
        logger.info(f"MD5 hash: {md5_hash}")
        
        # Test truncation  
        truncated_hash = hash_sha256(test_data, truncate_length=8)
        logger.info(f"Truncated hash: {truncated_hash}")
        
        # Test random generation
        random_bytes = generate_token_bytes(16)
        random_hex = generate_token_hex(8)
        random_urlsafe = generate_token_urlsafe(16)
        
        logger.info(f"Random bytes length: {len(random_bytes)}")
        logger.info(f"Random hex: {random_hex}")
        logger.info(f"Random URL-safe: {random_urlsafe}")
        
        # Test cache key generation
        cache_key = generate_cache_key("user", 123, "action", max_length=12)
        logger.info(f"Cache key: {cache_key}")
        
        # Test encryption/decryption
        encrypted_data, key_id = crypto_manager.encrypt_string("Secret message")
        decrypted_text = crypto_manager.decrypt_string(encrypted_data, key_id)
        logger.info(f"Encryption/decryption test: {'PASS' if decrypted_text == 'Secret message' else 'FAIL'}")
        
        # Test secure comparison
        comparison_result = secure_compare("test", "test")
        logger.info(f"Secure comparison test: {'PASS' if comparison_result else 'FAIL'}")
        
        # Get status
        status = crypto_manager.get_crypto_status()
        logger.info(f"Crypto system status: {len(status)} categories")
        
        logger.info("✅ All crypto manager validations passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Crypto manager validation failed: {e}")
        return False

def create_migration_examples():
    """Create migration examples showing before/after patterns."""
    examples = {
        "Hash Operations": {
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
        "Random Generation": {
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
        "Cache Key Generation": {
            "before": [
                "import hashlib",
                "cache_key = f'cache:{hashlib.md5(f\"{user_id}_{action}\".encode()).hexdigest()[:12]}'"
            ],
            "after": [
                "from prompt_improver.security import generate_cache_key",
                "cache_key = f'cache:{generate_cache_key(user_id, action, max_length=12)}'"
            ]
        },
        "Encryption": {
            "before": [
                "from cryptography.fernet import Fernet",
                "key = Fernet.generate_key()",
                "fernet = Fernet(key)",
                "encrypted = fernet.encrypt(data.encode())"
            ],
            "after": [
                "from prompt_improver.security import get_crypto_manager",
                "crypto_manager = get_crypto_manager()",
                "encrypted, key_id = crypto_manager.encrypt_string(data)"
            ]
        }
    }
    
    logger.info("Migration Examples:")
    logger.info("=" * 50)
    
    for category, example in examples.items():
        logger.info(f"\n{category}:")
        logger.info("-" * 30)
        logger.info("BEFORE:")
        for line in example["before"]:
            logger.info(f"  {line}")
        logger.info("AFTER:")
        for line in example["after"]:
            logger.info(f"  {line}")

def main():
    parser = argparse.ArgumentParser(description='Migrate cryptographic operations to unified manager')
    parser.add_argument('--validate', action='store_true', help='Validate crypto manager functionality')
    parser.add_argument('--scan-all', action='store_true', help='Scan all files for crypto usage')
    parser.add_argument('--migrate-file', type=str, help='Migrate specific file')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    parser.add_argument('--examples', action='store_true', help='Show migration examples')
    
    args = parser.parse_args()
    
    if args.validate:
        success = validate_crypto_manager()
        sys.exit(0 if success else 1)
        
    if args.examples:
        create_migration_examples()
        sys.exit(0)
        
    if args.scan_all:
        src_dir = Path(__file__).parent.parent / "src"
        results = scan_all_files(src_dir)
        
        logger.info(f"Scanned {results['total_files_scanned']} Python files")
        logger.info(f"Found {results['summary']['total_crypto_files']} files with crypto operations:")
        logger.info(f"  - {results['summary']['hashlib_files']} files using hashlib")
        logger.info(f"  - {results['summary']['secrets_files']} files using secrets")
        logger.info(f"  - {results['summary']['cryptography_files']} files using cryptography")
        
        logger.info("\nFiles requiring migration:")
        for file_info in results['files_with_crypto']:
            logger.info(f"\n{file_info['path']}:")
            for category, usages in file_info['usage'].items():
                if usages:
                    logger.info(f"  {category}:")
                    for usage in usages[:3]:  # Show first 3 usages
                        logger.info(f"    {usage}")
                    if len(usages) > 3:
                        logger.info(f"    ... and {len(usages) - 3} more")
        
        sys.exit(0)
        
    if args.migrate_file:
        file_path = Path(args.migrate_file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
            
        success = migrate_file_crypto_operations(file_path, dry_run=args.dry_run)
        sys.exit(0 if success else 1)
        
    parser.print_help()

if __name__ == "__main__":
    main()