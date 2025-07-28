#!/usr/bin/env python3
"""
Test Environment Validation Script

This script validates that all required environment variables are properly
configured for integration tests in the Prompt Improver project.

Usage:
    python scripts/validate_test_environment.py
    
    # With specific env file
    python scripts/validate_test_environment.py --env-file .env.test.local
"""

import asyncio
import os
import sys
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message: str, status: str, details: str = ""):
    """Print a status message with color coding."""
    if status == "PASS":
        color = Colors.GREEN
        symbol = "✓"
    elif status == "FAIL":
        color = Colors.RED
        symbol = "✗"
    elif status == "WARN":
        color = Colors.YELLOW
        symbol = "⚠"
    else:
        color = Colors.BLUE
        symbol = "ℹ"
    
    status_str = f"{color}{symbol} {message}{Colors.END}"
    if details:
        status_str += f" - {details}"
    print(status_str)

def load_env_file(env_file: str) -> None:
    """Load environment variables from a file."""
    if not os.path.exists(env_file):
        print_status(f"Environment file {env_file} not found", "WARN")
        return
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
    
    print_status(f"Loaded environment from {env_file}", "INFO")

def validate_required_variables() -> Tuple[bool, List[str]]:
    """Validate that all required environment variables are set."""
    required_vars = {
        # Critical for all tests
        'POSTGRES_HOST': 'Database host',
        'POSTGRES_PORT': 'Database port',
        'POSTGRES_DATABASE': 'Database name',
        'POSTGRES_USERNAME': 'Database username',
        'POSTGRES_PASSWORD': 'Database password',
        
        # JWT Authentication
        'MCP_JWT_SECRET_KEY': 'JWT secret for MCP authentication',
        
        # Redis
        'REDIS_URL': 'Redis connection URL',
        
        # MCP specific
        'MCP_POSTGRES_PASSWORD': 'MCP database password',
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            print_status(f"{var} ({description})", "FAIL", "Not set")
            missing_vars.append(var)
        else:
            # Check JWT secret length
            if var == 'MCP_JWT_SECRET_KEY' and len(value) < 32:
                print_status(f"{var}", "FAIL", f"Too short ({len(value)} chars, need 32+)")
                missing_vars.append(var)
            else:
                print_status(f"{var} ({description})", "PASS", "Set")
    
    return len(missing_vars) == 0, missing_vars

def validate_optional_variables() -> None:
    """Validate optional but recommended environment variables."""
    optional_vars = {
        'JWT_SECRET_KEY': 'General JWT secret',
        'TEST_REDIS_URL': 'Test-specific Redis URL',
        'MCP_RATE_LIMIT_REDIS_URL': 'Rate limiting Redis URL',
        'MCP_CACHE_REDIS_URL': 'Caching Redis URL',
        'TEST_DB_NAME': 'Test database name',
        'ENVIRONMENT': 'Environment setting',
        'LOG_LEVEL': 'Logging level',
    }
    
    print(f"\n{Colors.BOLD}Optional Variables:{Colors.END}")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print_status(f"{var} ({description})", "PASS", "Set")
        else:
            print_status(f"{var} ({description})", "WARN", "Not set (using defaults)")

async def test_database_connection() -> bool:
    """Test database connection."""
    try:
        import asyncpg
        
        conn = await asyncpg.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            database=os.getenv('POSTGRES_DATABASE', 'apes_production'),
            user=os.getenv('POSTGRES_USERNAME', 'apes_user'),
            password=os.getenv('POSTGRES_PASSWORD')
        )
        
        # Test a simple query
        result = await conn.fetchval('SELECT version()')
        await conn.close()
        
        print_status("Database connection", "PASS", f"Connected to PostgreSQL")
        return True
        
    except ImportError:
        print_status("Database connection", "WARN", "asyncpg not installed - skipping test")
        return True
    except Exception as e:
        print_status("Database connection", "FAIL", str(e))
        return False

def test_redis_connection() -> bool:
    """Test Redis connection."""
    try:
        import redis
        
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/15')
        r = redis.from_url(redis_url)
        r.ping()
        
        # Test basic operations
        r.set('test_key', 'test_value', ex=10)
        value = r.get('test_key')
        r.delete('test_key')
        
        print_status("Redis connection", "PASS", f"Connected to {redis_url}")
        return True
        
    except ImportError:
        print_status("Redis connection", "WARN", "redis package not installed - skipping test")
        return True
    except Exception as e:
        print_status("Redis connection", "FAIL", str(e))
        return False

def test_jwt_functionality() -> bool:
    """Test JWT secret functionality."""
    try:
        import jwt
        
        secret = os.getenv('MCP_JWT_SECRET_KEY')
        if not secret:
            print_status("JWT functionality", "FAIL", "MCP_JWT_SECRET_KEY not set")
            return False
        
        # Test encoding/decoding
        payload = {"test": "data", "user": "test_user"}
        token = jwt.encode(payload, secret, algorithm="HS256")
        decoded = jwt.decode(token, secret, algorithms=["HS256"])
        
        if decoded['test'] == 'data':
            print_status("JWT functionality", "PASS", "Encoding/decoding works")
            return True
        else:
            print_status("JWT functionality", "FAIL", "Decoded payload doesn't match")
            return False
            
    except ImportError:
        print_status("JWT functionality", "WARN", "PyJWT not installed - skipping test")
        return True
    except Exception as e:
        print_status("JWT functionality", "FAIL", str(e))
        return False

def check_file_permissions() -> bool:
    """Check file permissions for test directories."""
    test_dirs = [
        'tests/integration',
        'tests/fixtures',
        'logs',
        'test_models'
    ]
    
    all_good = True
    for dir_path in test_dirs:
        path = Path(dir_path)
        if path.exists():
            if path.is_dir() and os.access(path, os.R_OK | os.W_OK):
                print_status(f"Directory {dir_path}", "PASS", "Readable and writable")
            else:
                print_status(f"Directory {dir_path}", "FAIL", "Permission issues")
                all_good = False
        else:
            # Create directory if it doesn't exist
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_status(f"Directory {dir_path}", "PASS", "Created")
            except Exception as e:
                print_status(f"Directory {dir_path}", "FAIL", f"Cannot create: {e}")
                all_good = False
    
    return all_good

def generate_sample_env_file() -> None:
    """Generate a sample .env.test.local file."""
    sample_content = """# Test Environment Configuration
# Copy from .env.test and customize these values for your environment

# Database (customize for your setup)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=apes_production
POSTGRES_USERNAME=apes_user
POSTGRES_PASSWORD=YOUR_DATABASE_PASSWORD_HERE

# Redis (customize for your setup)
REDIS_URL=redis://localhost:6379/15

# JWT Secret (MUST be at least 32 characters)
MCP_JWT_SECRET_KEY=test_jwt_secret_key_for_phase1_integration_testing_32_chars_minimum

# MCP Database Password
MCP_POSTGRES_PASSWORD=secure_mcp_user_password

# Optional: Environment setting
ENVIRONMENT=test
LOG_LEVEL=INFO
"""
    
    sample_file = '.env.test.local.sample'
    with open(sample_file, 'w') as f:
        f.write(sample_content)
    
    print_status(f"Generated sample file", "INFO", f"{sample_file} created")

async def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate test environment setup')
    parser.add_argument('--env-file', help='Environment file to load')
    parser.add_argument('--generate-sample', action='store_true', 
                       help='Generate sample .env.test.local file')
    
    args = parser.parse_args()
    
    if args.generate_sample:
        generate_sample_env_file()
        return
    
    print(f"{Colors.BOLD}Test Environment Validation{Colors.END}")
    print("=" * 50)
    
    # Load environment file if specified
    if args.env_file:
        load_env_file(args.env_file)
    else:
        # Try to load common env files
        for env_file in ['.env.test.local', '.env.test', '.env']:
            if os.path.exists(env_file):
                load_env_file(env_file)
                break
    
    print(f"\n{Colors.BOLD}Required Variables:{Colors.END}")
    required_ok, missing_vars = validate_required_variables()
    
    validate_optional_variables()
    
    print(f"\n{Colors.BOLD}Connectivity Tests:{Colors.END}")
    db_ok = await test_database_connection()
    redis_ok = test_redis_connection()
    jwt_ok = test_jwt_functionality()
    
    print(f"\n{Colors.BOLD}File System Tests:{Colors.END}")
    files_ok = check_file_permissions()
    
    # Summary
    print(f"\n{Colors.BOLD}Summary:{Colors.END}")
    all_tests_passed = required_ok and db_ok and redis_ok and jwt_ok and files_ok
    
    if all_tests_passed:
        print_status("All tests passed", "PASS", "Environment is ready for integration tests")
        sys.exit(0)
    else:
        print_status("Some tests failed", "FAIL", "Environment needs attention")
        
        if missing_vars:
            print(f"\n{Colors.YELLOW}Missing required variables:{Colors.END}")
            for var in missing_vars:
                print(f"  - {var}")
        
        print(f"\n{Colors.BLUE}Next steps:{Colors.END}")
        print("1. Copy .env.test to .env.test.local")
        print("2. Customize values in .env.test.local for your environment")
        print("3. Ensure PostgreSQL and Redis are running")
        print("4. Run this script again to validate")
        
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())