#!/usr/bin/env python3
"""
Debug script to investigate L3DatabaseService SQL issues.

Tests the exact SQL queries that are failing in the comprehensive test.
"""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime, timedelta

from tests.containers.postgres_container import PostgreSQLTestContainer
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def debug_upsert_access_count():
    """Debug the UPSERT access count issue."""
    print("🔍 Debug: UPSERT Access Count Issue")
    print("=" * 50)
    
    container = PostgreSQLTestContainer(database_name="debug_upsert")
    
    try:
        await container.start()
        
        # Create table
        create_table_sql = """CREATE TABLE IF NOT EXISTS cache_l3 (
            cache_key VARCHAR(255) PRIMARY KEY, 
            cache_value TEXT NOT NULL, 
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), 
            expires_at TIMESTAMP WITH TIME ZONE, 
            access_count INTEGER DEFAULT 1, 
            last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )"""
        
        await container.execute_sql(create_table_sql)
        print("✅ Table created")
        
        # Test the exact sequence from the failing test
        test_key = "conflict_test_key"
        initial_value = {"version": 1}
        updated_value = {"version": 2}
        
        # Initial INSERT
        print("\n🔧 Step 1: Initial SET operation")
        upsert_sql = """INSERT INTO cache_l3 (cache_key, cache_value, expires_at) 
                        VALUES (:key, :value, :expires_at) 
                        ON CONFLICT (cache_key) DO UPDATE SET 
                            cache_value = EXCLUDED.cache_value, 
                            expires_at = EXCLUDED.expires_at, 
                            created_at = NOW(), 
                            access_count = 1, 
                            last_accessed = NOW()"""
        
        await container.execute_sql(upsert_sql, {
            "key": test_key,
            "value": json.dumps(initial_value),
            "expires_at": datetime.now(UTC) + timedelta(seconds=300)
        })
        
        # Check initial access count
        async with container.get_session() as session:
            result = await session.execute(
                text("SELECT access_count FROM cache_l3 WHERE cache_key = :key"),
                {"key": test_key}
            )
            row = result.fetchone()
            initial_access_count = row.access_count if row else 0
            print(f"  📊 Access count after INSERT: {initial_access_count}")
        
        # GET with UPDATE (should increment access count)
        print("\n🔍 Step 2: GET with access count update")
        get_update_sql = """UPDATE cache_l3 SET access_count = access_count + 1, last_accessed = NOW() 
                           WHERE cache_key = :key AND (expires_at IS NULL OR expires_at > NOW()) 
                           RETURNING cache_value, access_count"""
        
        async with container.get_session() as session:
            result = await session.execute(text(get_update_sql), {"key": test_key})
            await session.commit()  # Explicit commit
            row = result.fetchone()
            if row:
                retrieved_value = json.loads(row.cache_value)
                returned_access_count = row.access_count
                print(f"  📄 Retrieved value: {retrieved_value}")
                print(f"  📊 Access count returned by UPDATE: {returned_access_count}")
            else:
                print("  ❌ No value retrieved")
        
        # Check access count after GET in a fresh session
        async with container.get_session() as session:
            result = await session.execute(
                text("SELECT access_count FROM cache_l3 WHERE cache_key = :key"),
                {"key": test_key}
            )
            row = result.fetchone()
            after_get_count = row.access_count if row else 0
            print(f"  📊 Access count after GET (fresh session): {after_get_count}")
        
        # Now test UPSERT (should reset to 1)
        print("\n🔄 Step 3: UPSERT (conflicting key)")
        await container.execute_sql(upsert_sql, {
            "key": test_key,
            "value": json.dumps(updated_value),
            "expires_at": datetime.now(UTC) + timedelta(seconds=600)
        })
        
        # Check access count after UPSERT
        async with container.get_session() as session:
            result = await session.execute(
                text("SELECT access_count FROM cache_l3 WHERE cache_key = :key"),
                {"key": test_key}
            )
            row = result.fetchone()
            after_upsert_count = row.access_count if row else 0
            print(f"  📊 Access count after UPSERT: {after_upsert_count}")
        
        # GET again (should increment from 1 to 2)  
        print("\n🔍 Step 4: GET after UPSERT (should increment to 2)")
        async with container.get_session() as session:
            result = await session.execute(text(get_update_sql), {"key": test_key})
            await session.commit()  # Explicit commit
            row = result.fetchone()
            if row:
                retrieved_value = json.loads(row.cache_value)
                returned_access_count = row.access_count
                print(f"  📄 Retrieved value: {retrieved_value}")
                print(f"  📊 Access count returned by final UPDATE: {returned_access_count}")
            else:
                print("  ❌ No value retrieved")
        
        # Check final access count in fresh session
        async with container.get_session() as session:
            result = await session.execute(
                text("SELECT access_count FROM cache_l3 WHERE cache_key = :key"),
                {"key": test_key}
            )
            row = result.fetchone()
            final_count = row.access_count if row else 0
            print(f"  📊 Final access count (fresh session): {final_count}")
        
        if final_count == 2:
            print("✅ UPSERT access count logic working correctly")
            return True
        else:
            print(f"❌ Expected access count 2, got {final_count}")
            return False
        
    finally:
        await container.stop()


async def debug_cleanup_expired():
    """Debug the cleanup expired entries issue."""
    print("\n🔍 Debug: Cleanup Expired Entries Issue")
    print("=" * 50)
    
    container = PostgreSQLTestContainer(database_name="debug_cleanup")
    
    try:
        await container.start()
        
        # Create table
        create_table_sql = """CREATE TABLE IF NOT EXISTS cache_l3 (
            cache_key VARCHAR(255) PRIMARY KEY, 
            cache_value TEXT NOT NULL, 
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), 
            expires_at TIMESTAMP WITH TIME ZONE, 
            access_count INTEGER DEFAULT 1, 
            last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )"""
        
        await container.execute_sql(create_table_sql)
        print("✅ Table created")
        
        # Insert test entries with different expiration scenarios
        print("\n🔧 Creating test entries...")
        test_entries = [
            ("permanent", {"type": "permanent"}, None),
            ("future", {"type": "future"}, datetime.now(UTC) + timedelta(minutes=5)),
            ("expired", {"type": "expired"}, datetime.now(UTC) - timedelta(minutes=1)),
        ]
        
        for key, value, expires_at in test_entries:
            await container.execute_sql(
                "INSERT INTO cache_l3 (cache_key, cache_value, expires_at) VALUES (:key, :value, :expires_at)",
                {"key": key, "value": json.dumps(value), "expires_at": expires_at}
            )
            expiry_str = "NULL" if expires_at is None else expires_at.isoformat()
            print(f"  ✅ Inserted {key} (expires: {expiry_str})")
        
        # Check initial count
        async with container.get_session() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM cache_l3"))
            initial_count = result.scalar()
            print(f"📊 Initial total entries: {initial_count}")
        
        # Check expired count
        async with container.get_session() as session:
            result = await session.execute(
                text("SELECT COUNT(*) FROM cache_l3 WHERE expires_at IS NOT NULL AND expires_at <= NOW()")
            )
            expired_count = result.scalar()
            print(f"📊 Expired entries found: {expired_count}")
        
        # Test cleanup query
        print("\n🧹 Running cleanup query...")
        cleanup_sql = "DELETE FROM cache_l3 WHERE expires_at IS NOT NULL AND expires_at <= NOW()"
        
        async with container.get_session() as session:
            result = await session.execute(text(cleanup_sql))
            await session.commit()  # Explicit commit
            deleted_count = result.rowcount
            print(f"📊 Deleted count reported: {deleted_count}")
        
        # Check final count in fresh session
        async with container.get_session() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM cache_l3"))
            final_count = result.scalar()
            print(f"📊 Final total entries (fresh session): {final_count}")
        
        # Verify expired entry is gone in fresh session
        async with container.get_session() as session:
            result = await session.execute(
                text("SELECT cache_key FROM cache_l3 WHERE cache_key = 'expired'")
            )
            expired_exists = result.fetchone() is not None
            print(f"📊 Expired entry still exists (fresh session): {expired_exists}")
        
        # Let's also check what entries remain
        async with container.get_session() as session:
            result = await session.execute(text("SELECT cache_key, expires_at FROM cache_l3"))
            remaining_entries = result.fetchall()
            print("📋 Remaining entries:")
            for entry in remaining_entries:
                expiry_str = entry.expires_at.isoformat() if entry.expires_at else "NULL"
                print(f"  - {entry.cache_key}: {expiry_str}")
        
        expected_final = initial_count - expired_count
        if final_count == expected_final and deleted_count > 0 and not expired_exists:
            print("✅ Cleanup expired logic working correctly")
            return True
        else:
            print(f"❌ Cleanup issue - expected {expected_final} final entries, got {final_count}")
            return False
            
    finally:
        await container.stop()


async def main():
    """Run debug tests for L3DatabaseService SQL issues."""
    print("🚀 L3DatabaseService SQL Debug Session")
    print("=" * 70)
    
    results = []
    
    # Debug UPSERT issue
    try:
        upsert_result = await debug_upsert_access_count()
        results.append(("UPSERT Access Count", upsert_result))
    except Exception as e:
        print(f"❌ UPSERT debug failed: {e}")
        results.append(("UPSERT Access Count", False))
    
    # Debug cleanup issue
    try:
        cleanup_result = await debug_cleanup_expired()
        results.append(("Cleanup Expired", cleanup_result))
    except Exception as e:
        print(f"❌ Cleanup debug failed: {e}")
        results.append(("Cleanup Expired", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 DEBUG SUMMARY")
    print("=" * 70)
    
    for test_name, success in results:
        status = "✅ WORKING" if success else "❌ BROKEN"
        print(f"  {test_name}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    print(f"\nWorking: {success_count}/{len(results)}")
    
    if success_count == len(results):
        print("🎉 All SQL queries working correctly - issue may be in test logic")
    else:
        print("⚠️  SQL compression has real issues that need fixing")
    
    return success_count == len(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)