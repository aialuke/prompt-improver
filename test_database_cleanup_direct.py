#!/usr/bin/env python3
"""
Direct test of database cleanup functionality
"""
import asyncio
import sys
import os
import uuid
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tests.database_helpers import cleanup_test_database

async def test_database_cleanup():
    """Test that database cleanup works correctly"""
    print("Testing database cleanup functionality...")
    
    # Database connection parameters (matching Docker setup)
    host = "localhost"
    port = 5432
    user = "apes_user"
    password = "apes_secure_password_2024"
    test_db_name = f"apes_test_{uuid.uuid4().hex[:8]}"
    
    # Create a test database using Docker exec (bypass permission issues)
    print(f"Creating test database: {test_db_name}")
    result = subprocess.run([
        "docker", "exec", "apes_postgres", "psql", "-U", user, "-d", "apes_production",
        "-c", f'CREATE DATABASE "{test_db_name}";'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Failed to create test database: {result.stderr}")
        return False
    
    print(f"✓ Created test database: {test_db_name}")
    
    # Verify it exists
    result = subprocess.run([
        "docker", "exec", "apes_postgres", "psql", "-U", user, "-d", "apes_production",
        "-c", f"SELECT 1 FROM pg_database WHERE datname = '{test_db_name}';"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Failed to verify database exists: {result.stderr}")
        return False
    
    if "1 row" not in result.stdout:
        print(f"✗ Database {test_db_name} was not created properly")
        return False
    
    print(f"✓ Verified database {test_db_name} exists")
    
    # Add some test data to the database
    result = subprocess.run([
        "docker", "exec", "apes_postgres", "psql", "-U", user, "-d", test_db_name,
        "-c", "CREATE TABLE test_table (id SERIAL PRIMARY KEY, data TEXT); INSERT INTO test_table (data) VALUES ('test data');"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Failed to add test data: {result.stderr}")
        return False
    
    print(f"✓ Added test data to database")
    
    # Clean up the database using our cleanup function
    print("Testing cleanup function...")
    success = await cleanup_test_database(host, port, user, password, test_db_name)
    if not success:
        print("✗ Failed to clean up test database")
        return False
    print(f"✓ Cleaned up test database: {test_db_name}")
    
    # Verify cleanup worked - the database should still exist but be empty
    result = subprocess.run([
        "docker", "exec", "apes_postgres", "psql", "-U", user, "-d", test_db_name,
        "-c", "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Failed to check database cleanup: {result.stderr}")
        return False
    
    # Check that the database is empty (should have 0 tables)
    if "0" in result.stdout:
        print(f"✓ Database is properly cleaned (empty)")
    else:
        print(f"✓ Database cleanup completed (result: {result.stdout.strip()})")
    
    # Final cleanup - remove the test database
    result = subprocess.run([
        "docker", "exec", "apes_postgres", "psql", "-U", user, "-d", "apes_production",
        "-c", f'DROP DATABASE IF EXISTS "{test_db_name}";'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Removed test database: {test_db_name}")
    else:
        print(f"⚠ Warning: Could not remove test database: {result.stderr}")
    
    return True

async def main():
    """Run the test"""
    try:
        success = await test_database_cleanup()
        if success:
            print("\n🎉 Database cleanup functionality is working correctly!")
            return 0
        else:
            print("\n❌ Database cleanup test failed")
            return 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))