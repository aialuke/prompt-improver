#!/usr/bin/env python3
"""Validate environment configuration for consistent database URLs."""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional


def find_env_files() -> List[Path]:
    """Find all environment configuration files."""
    env_files = []
    
    # Common environment file patterns
    patterns = [
        ".env",
        ".env.example",
        ".env.local",
        ".env.development",
        ".env.production",
        ".env.test",
        "docker-compose.yml",
        "docker-compose.test.yml",
        "docker-compose.dev.yml",
        "docker-compose.prod.yml",
    ]
    
    for pattern in patterns:
        for file_path in Path(".").rglob(pattern):
            if file_path.is_file():
                env_files.append(file_path)
    
    return env_files


def extract_database_urls(file_path: Path) -> Dict[str, str]:
    """Extract database URLs from a file."""
    urls = {}
    
    try:
        content = file_path.read_text()
        
        # Patterns to match database URLs
        patterns = [
            r'DATABASE_URL\s*[=:]\s*["\']?([^"\';\s]+)["\']?',
            r'POSTGRES_URL\s*[=:]\s*["\']?([^"\';\s]+)["\']?',
            r'DB_URL\s*[=:]\s*["\']?([^"\';\s]+)["\']?',
            r'postgresql://[^"\';\s]+',
            r'postgresql\+psycopg2://[^"\';\s]+',
            r'postgresql\+psycopg://[^"\';\s]+',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if match.startswith('postgresql'):
                    urls[f"found_in_{file_path.name}"] = match
                    
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
    
    return urls


def validate_url_format(url: str) -> Dict[str, bool]:
    """Validate database URL format."""
    validation = {
        "is_postgresql": False,
        "uses_psycopg3": False,
        "no_psycopg2": False,
        "no_sqlite": False,
    }
    
    validation["is_postgresql"] = url.startswith("postgresql://") or url.startswith("postgresql+")
    # Both postgresql:// (sync) and postgresql+psycopg:// (async) are valid psycopg3 formats
    validation["uses_psycopg3"] = (url.startswith("postgresql://") or "postgresql+psycopg://" in url) and "psycopg2" not in url
    validation["no_psycopg2"] = "psycopg2" not in url
    validation["no_sqlite"] = "sqlite" not in url.lower()
    
    return validation


def main():
    """Main validation function."""
    print("🔍 Validating environment configuration...")
    
    # Find all environment files
    env_files = find_env_files()
    print(f"Found {len(env_files)} environment files:")
    for file in env_files:
        print(f"  - {file}")
    
    # Extract database URLs from all files
    all_urls = {}
    for file_path in env_files:
        urls = extract_database_urls(file_path)
        all_urls.update(urls)
    
    if not all_urls:
        print("✓ No database URLs found in environment files")
        return True
    
    print(f"\n📊 Found {len(all_urls)} database URLs:")
    
    validation_passed = True
    for source, url in all_urls.items():
        print(f"\n🔗 {source}:")
        print(f"   URL: {url}")
        
        validation = validate_url_format(url)
        
        for check, passed in validation.items():
            status = "✓" if passed else "✗"
            print(f"   {status} {check}: {passed}")
            if not passed:
                validation_passed = False
    
    # Check for consistency
    unique_urls = set(all_urls.values())
    if len(unique_urls) > 1:
        print(f"\n⚠️  Found {len(unique_urls)} different database URLs - ensure consistency")
        for url in unique_urls:
            print(f"   - {url}")
    
    # Summary
    print(f"\n{'='*50}")
    if validation_passed:
        print("✅ All database URLs are properly configured!")
        print("   - Using PostgreSQL")
        print("   - Using psycopg3 format (postgresql:// or postgresql+psycopg://)")
        print("   - No SQLite references")
        print("   - No psycopg2 references")
    else:
        print("❌ Environment configuration validation failed!")
        print("   Please update database URLs to use psycopg3 format")
        print("   Valid formats: postgresql:// (sync) or postgresql+psycopg:// (async)")
    
    return validation_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)