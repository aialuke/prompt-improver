"""
Comprehensive validation test for L3DatabaseService compressed SQL queries.

This test validates all SQL query compressions and simplifications made during 
aggressive code reduction, ensuring they work correctly against real PostgreSQL.

Critical validation areas:
1. Compressed single-line SQL queries execute without syntax errors
2. Table creation and schema validation with constraints and indexes
3. UPSERT conflict resolution with ON CONFLICT DO UPDATE logic
4. TTL expiration logic with NULL vs future/past timestamps
5. Performance tracking and <50ms response time targets
6. JSON serialization in TEXT fields
7. Statistics collection and health check functionality
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional
from unittest.mock import Mock

import pytest
from sqlalchemy import text

from tests.containers.postgres_container import PostgreSQLTestContainer
from prompt_improver.services.cache.l3_database_service import L3DatabaseService, DatabaseSessionProtocol
from src.prompt_improver.database import ManagerMode

logger = logging.getLogger(__name__)


class PostgreSQLSessionAdapter:
    """Adapter to bridge PostgreSQL testcontainer to DatabaseSessionProtocol."""
    
    def __init__(self, container: PostgreSQLTestContainer):
        self.container = container
        self._session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = self.container.get_session()
        self._actual_session = await self._session.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)
    
    async def execute(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """Execute database query."""
        if parameters:
            # Convert named parameters (:key) to SQLAlchemy format
            result = await self._actual_session.execute(text(query), parameters)
        else:
            result = await self._actual_session.execute(text(query))
        
        # Capture rowcount before potential commit
        rowcount = getattr(result, 'rowcount', 0)
        
        # Only commit for non-SELECT queries
        if not query.strip().upper().startswith('SELECT'):
            await self._actual_session.commit()
        
        # Preserve rowcount attribute
        result.rowcount = rowcount
        return result
    
    async def fetch_one(self, query: str, parameters: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Fetch single row from database."""
        if parameters:
            result = await self._actual_session.execute(text(query), parameters)
        else:
            result = await self._actual_session.execute(text(query))
        
        # Commit if this is an UPDATE/DELETE...RETURNING query
        if 'UPDATE' in query.upper() or 'DELETE' in query.upper():
            await self._actual_session.commit()
        
        row = result.fetchone()
        if row:
            # Convert SQLAlchemy Row to dict
            return dict(row._mapping)
        return None
    
    async def fetch_all(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Fetch multiple rows from database."""
        if parameters:
            result = await self._actual_session.execute(text(query), parameters)
        else:
            result = await self._actual_session.execute(text(query))
        
        rows = result.fetchall()
        return [dict(row._mapping) for row in rows]


class TestDatabaseConfig:
    """Simple database config for testing."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
    
    def get_database_url(self) -> str:
        return self.database_url
    
    def get_connection_pool_config(self) -> dict:
        return {
            "pool_pre_ping": True,
            "pool_recycle": 3600,
        }


class MockSessionManager:
    """Mock session manager that provides PostgreSQL adapter sessions."""
    
    def __init__(self, container: PostgreSQLTestContainer):
        self.container = container
    
    def get_session(self):
        """Return PostgreSQL session adapter."""
        return PostgreSQLSessionAdapter(self.container)


class L3DatabaseServiceValidator:
    """Comprehensive validator for L3DatabaseService SQL compressions."""
    
    def __init__(self):
        self.container = None
        self.service = None
        self.test_results = {}
    
    async def setup(self):
        """Set up PostgreSQL testcontainer and L3DatabaseService."""
        print("🚀 Setting up PostgreSQL testcontainer...")
        
        # Create PostgreSQL container
        self.container = PostgreSQLTestContainer(
            postgres_version="16",
            database_name=f"l3_cache_test_{uuid.uuid4().hex[:8]}"
        )
        
        await self.container.start()
        print(f"✅ PostgreSQL container started: {self.container.database_name}")
        
        # Create L3DatabaseService with PostgreSQL adapter
        session_manager = MockSessionManager(self.container)
        self.service = L3DatabaseService(session_manager=session_manager)
        
        print("✅ L3DatabaseService configured with real PostgreSQL")
    
    async def teardown(self):
        """Clean up test resources."""
        if self.container:
            await self.container.stop()
            print("✅ PostgreSQL container stopped")
    
    async def validate_schema_creation(self) -> bool:
        """Test 1: Validate compressed table and index creation SQL."""
        print("\n📋 Test 1: Schema Creation Validation")
        print("=" * 50)
        
        try:
            # Test table creation
            print("🔨 Testing compressed table creation SQL...")
            table_created = await self.service.ensure_table_exists()
            
            if not table_created:
                print("❌ Table creation failed")
                return False
            
            print("✅ Table creation successful")
            
            # Verify table structure
            print("🔍 Verifying table schema...")
            async with self.container.get_session() as session:
                # Check table exists
                table_check = await session.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'cache_l3'
                    )
                """))
                table_exists = table_check.scalar()
                
                if not table_exists:
                    print("❌ Table 'cache_l3' not found")
                    return False
                
                # Check column structure
                columns_check = await session.execute(text("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = 'cache_l3' 
                    ORDER BY ordinal_position
                """))
                columns = columns_check.fetchall()
                
                expected_columns = {
                    'cache_key': ('character varying', 'NO'),
                    'cache_value': ('text', 'NO'), 
                    'created_at': ('timestamp with time zone', 'YES'),
                    'expires_at': ('timestamp with time zone', 'YES'),
                    'access_count': ('integer', 'YES'),
                    'last_accessed': ('timestamp with time zone', 'YES'),
                }
                
                for col in columns:
                    col_name = col.column_name
                    if col_name in expected_columns:
                        expected_type, expected_nullable = expected_columns[col_name]
                        if col.data_type == expected_type and col.is_nullable == expected_nullable:
                            print(f"  ✅ {col_name}: {col.data_type} (nullable: {col.is_nullable})")
                        else:
                            print(f"  ❌ {col_name}: Expected {expected_type}, got {col.data_type}")
                            return False
                    else:
                        print(f"  ⚠️  Unexpected column: {col_name}")
                
                # Check primary key constraint
                pk_check = await session.execute(text("""
                    SELECT constraint_name 
                    FROM information_schema.table_constraints 
                    WHERE table_name = 'cache_l3' 
                    AND constraint_type = 'PRIMARY KEY'
                """))
                pk_exists = pk_check.scalar()
                
                if pk_exists:
                    print("  ✅ Primary key constraint exists")
                else:
                    print("  ❌ Primary key constraint missing")
                    return False
                
                # Check index exists
                index_check = await session.execute(text("""
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE tablename = 'cache_l3' 
                    AND indexname = 'idx_cache_l3_expires_at'
                """))
                index_exists = index_check.scalar()
                
                if index_exists:
                    print("  ✅ Conditional index on expires_at exists")
                else:
                    print("  ❌ Conditional index on expires_at missing")
                    return False
            
            print("✅ Schema validation: PASSED")
            self.test_results['schema_creation'] = True
            return True
            
        except Exception as e:
            print(f"❌ Schema validation: FAILED - {e}")
            self.test_results['schema_creation'] = False
            return False
    
    async def validate_crud_operations(self) -> bool:
        """Test 2: Validate all compressed CRUD SQL queries."""
        print("\n📝 Test 2: CRUD Operations Validation")
        print("=" * 50)
        
        try:
            test_data = [
                ("permanent_key", {"data": "permanent_value"}, None),
                ("ttl_key", {"data": "ttl_value"}, 300),
                ("complex_key", {"nested": {"data": [1, 2, 3]}, "bool": True}, 600),
            ]
            
            # Test SET operations
            print("🔧 Testing compressed SET SQL...")
            for key, value, ttl in test_data:
                success = await self.service.set(key, value, ttl)
                if not success:
                    print(f"  ❌ SET failed for key: {key}")
                    return False
                print(f"  ✅ SET successful for key: {key}")
            
            # Test GET operations  
            print("🔍 Testing compressed GET SQL...")
            for key, expected_value, _ in test_data:
                retrieved_value = await self.service.get(key)
                if retrieved_value != expected_value:
                    print(f"  ❌ GET mismatch for key {key}: expected {expected_value}, got {retrieved_value}")
                    return False
                print(f"  ✅ GET successful for key: {key}")
            
            # Test EXISTS operations
            print("🔎 Testing compressed EXISTS SQL...")
            for key, _, _ in test_data:
                exists = await self.service.exists(key)
                if not exists:
                    print(f"  ❌ EXISTS returned False for key: {key}")
                    return False
                print(f"  ✅ EXISTS successful for key: {key}")
            
            # Test DELETE operations
            print("🗑️ Testing compressed DELETE SQL...")
            delete_success = await self.service.delete(test_data[1][0])  # Delete ttl_key
            if not delete_success:
                print("  ❌ DELETE operation failed")
                return False
            
            # Verify deletion
            deleted_value = await self.service.get(test_data[1][0])
            if deleted_value is not None:
                print("  ❌ Key still exists after deletion")
                return False
            print("  ✅ DELETE successful")
            
            print("✅ CRUD operations validation: PASSED")
            self.test_results['crud_operations'] = True
            return True
            
        except Exception as e:
            print(f"❌ CRUD operations validation: FAILED - {e}")
            self.test_results['crud_operations'] = False
            return False
    
    async def validate_upsert_conflict_resolution(self) -> bool:
        """Test 3: Validate INSERT...ON CONFLICT DO UPDATE logic."""
        print("\n🔄 Test 3: UPSERT Conflict Resolution Validation")
        print("=" * 50)
        
        try:
            test_key = "conflict_test_key"
            initial_value = {"version": 1}
            updated_value = {"version": 2}
            
            # Initial set
            print("🔧 Testing initial SET operation...")
            success1 = await self.service.set(test_key, initial_value, 300)
            if not success1:
                print("  ❌ Initial SET failed")
                return False
            
            # Verify initial value
            retrieved1 = await self.service.get(test_key)
            if retrieved1 != initial_value:
                print(f"  ❌ Initial value mismatch: expected {initial_value}, got {retrieved1}")
                return False
            print("  ✅ Initial SET successful")
            
            # Get initial access count
            async with self.container.get_session() as session:
                result = await session.execute(
                    text("SELECT access_count FROM cache_l3 WHERE cache_key = :key"),
                    {"key": test_key}
                )
                row = result.fetchone()
                initial_access_count = row.access_count if row else 0
                print(f"  📊 Initial access count: {initial_access_count}")
            
            # Conflicting set (should trigger UPDATE)
            print("🔄 Testing UPSERT conflict resolution...")
            success2 = await self.service.set(test_key, updated_value, 600)
            if not success2:
                print("  ❌ UPSERT SET failed")
                return False
            
            # Verify updated value
            retrieved2 = await self.service.get(test_key)
            if retrieved2 != updated_value:
                print(f"  ❌ Updated value mismatch: expected {updated_value}, got {retrieved2}")
                return False
            
            # Verify access count was reset to 1 (as per UPSERT logic)
            async with self.container.get_session() as session:
                result = await session.execute(
                    text("SELECT access_count FROM cache_l3 WHERE cache_key = :key"),
                    {"key": test_key}
                )
                row = result.fetchone()
                final_access_count = row.access_count if row else 0
                
                # Note: GET operation increments access_count, so it should be 2
                # (1 from UPSERT reset + 1 from GET)
                expected_count = 2
                if final_access_count != expected_count:
                    print(f"  ❌ Access count mismatch: expected {expected_count}, got {final_access_count}")
                    return False
                print(f"  📊 Final access count: {final_access_count}")
            
            print("  ✅ UPSERT conflict resolution successful")
            print("✅ UPSERT validation: PASSED")
            self.test_results['upsert_conflict'] = True
            return True
            
        except Exception as e:
            print(f"❌ UPSERT validation: FAILED - {e}")
            self.test_results['upsert_conflict'] = False
            return False
    
    async def validate_ttl_expiration_logic(self) -> bool:
        """Test 4: Validate TTL and expiration logic."""
        print("\n⏰ Test 4: TTL and Expiration Logic Validation")
        print("=" * 50)
        
        try:
            # Test data with different TTL scenarios
            permanent_key = "permanent_key"
            valid_key = "valid_key"  
            expired_key = "expired_key"
            
            # Set permanent entry (no TTL)
            print("🔧 Testing permanent entry (NULL expires_at)...")
            await self.service.set(permanent_key, {"type": "permanent"}, None)
            
            # Set valid entry (future expiration)
            print("🔧 Testing valid entry (future expires_at)...")
            await self.service.set(valid_key, {"type": "valid"}, 300)  # 5 minutes
            
            # Set expired entry (past expiration) - simulate by directly inserting
            print("🔧 Creating expired entry (past expires_at)...")
            past_time = datetime.now(UTC) - timedelta(minutes=1)
            async with self.container.get_session() as session:
                await session.execute(text("""
                    INSERT INTO cache_l3 (cache_key, cache_value, expires_at) 
                    VALUES (:key, :value, :expires_at)
                """), {
                    "key": expired_key,
                    "value": json.dumps({"type": "expired"}),
                    "expires_at": past_time
                })
                await session.commit()  # Explicit commit
            
            # Test retrieval behavior
            print("🔍 Testing retrieval of different TTL scenarios...")
            
            # Permanent entry should be retrievable
            permanent_result = await self.service.get(permanent_key)
            if permanent_result != {"type": "permanent"}:
                print(f"  ❌ Permanent entry not retrievable: {permanent_result}")
                return False
            print("  ✅ Permanent entry (NULL expires_at) retrievable")
            
            # Valid entry should be retrievable
            valid_result = await self.service.get(valid_key)
            if valid_result != {"type": "valid"}:
                print(f"  ❌ Valid entry not retrievable: {valid_result}")
                return False
            print("  ✅ Valid entry (future expires_at) retrievable")
            
            # Expired entry should NOT be retrievable
            expired_result = await self.service.get(expired_key)
            if expired_result is not None:
                print(f"  ❌ Expired entry unexpectedly retrievable: {expired_result}")
                return False
            print("  ✅ Expired entry (past expires_at) correctly filtered")
            
            # Test cleanup of expired entries
            print("🧹 Testing cleanup of expired entries...")
            
            # First, let's verify what entries exist before cleanup
            async with self.container.get_session() as session:
                result = await session.execute(text(
                    "SELECT cache_key, expires_at FROM cache_l3"
                ))
                entries = result.fetchall()
                print("  📋 Entries before cleanup:")
                for entry in entries:
                    expiry_str = entry.expires_at.isoformat() if entry.expires_at else "NULL"
                    print(f"    - {entry.cache_key}: {expiry_str}")
            
            # Check expired entries count
            async with self.container.get_session() as session:
                result = await session.execute(text(
                    "SELECT COUNT(*) FROM cache_l3 WHERE expires_at IS NOT NULL AND expires_at <= NOW()"
                ))
                expired_count = result.scalar()
                print(f"  📊 Expired entries to clean: {expired_count}")
            
            cleanup_count = await self.service.cleanup_expired()
            print(f"  📊 Cleanup count returned: {cleanup_count}")
            
            if cleanup_count < 1:
                print(f"  ❌ Cleanup did not remove expired entries: {cleanup_count}")
                return False
            print(f"  ✅ Cleanup removed {cleanup_count} expired entries")
            
            # Verify expired entry is actually removed
            async with self.container.get_session() as session:
                result = await session.execute(
                    text("SELECT COUNT(*) FROM cache_l3 WHERE cache_key = :key"),
                    {"key": expired_key}
                )
                count = result.scalar()
                if count > 0:
                    print("  ❌ Expired entry still exists in database after cleanup")
                    return False
                print("  ✅ Expired entry properly removed from database")
            
            print("✅ TTL and expiration validation: PASSED")
            self.test_results['ttl_expiration'] = True
            return True
            
        except Exception as e:
            print(f"❌ TTL and expiration validation: FAILED - {e}")
            self.test_results['ttl_expiration'] = False
            return False
    
    async def validate_performance_targets(self) -> bool:
        """Test 5: Validate <50ms performance targets."""
        print("\n⚡ Test 5: Performance Target Validation")
        print("=" * 50)
        
        try:
            performance_results = {}
            target_ms = 50.0  # As specified in L3DatabaseService
            
            # Test GET performance
            print("🏁 Testing GET operation performance...")
            test_key = "perf_test_key"
            test_value = {"performance": "test"}
            await self.service.set(test_key, test_value, 300)
            
            get_times = []
            for i in range(10):
                start_time = time.perf_counter()
                result = await self.service.get(test_key)
                end_time = time.perf_counter()
                get_times.append((end_time - start_time) * 1000)
            
            avg_get_time = sum(get_times) / len(get_times)
            max_get_time = max(get_times)
            performance_results['GET'] = {'avg': avg_get_time, 'max': max_get_time}
            print(f"  📊 GET: avg={avg_get_time:.2f}ms, max={max_get_time:.2f}ms")
            
            # Test SET performance
            print("🏁 Testing SET operation performance...")
            set_times = []
            for i in range(10):
                key = f"perf_set_{i}"
                start_time = time.perf_counter()
                await self.service.set(key, {"index": i}, 300)
                end_time = time.perf_counter()
                set_times.append((end_time - start_time) * 1000)
            
            avg_set_time = sum(set_times) / len(set_times)
            max_set_time = max(set_times)
            performance_results['SET'] = {'avg': avg_set_time, 'max': max_set_time}
            print(f"  📊 SET: avg={avg_set_time:.2f}ms, max={max_set_time:.2f}ms")
            
            # Test DELETE performance
            print("🏁 Testing DELETE operation performance...")
            delete_times = []
            for i in range(10):
                key = f"perf_set_{i}"  # Delete the keys we just created
                start_time = time.perf_counter()
                await self.service.delete(key)
                end_time = time.perf_counter()
                delete_times.append((end_time - start_time) * 1000)
            
            avg_delete_time = sum(delete_times) / len(delete_times)
            max_delete_time = max(delete_times)
            performance_results['DELETE'] = {'avg': avg_delete_time, 'max': max_delete_time}
            print(f"  📊 DELETE: avg={avg_delete_time:.2f}ms, max={max_delete_time:.2f}ms")
            
            # Check if all operations meet <50ms target
            all_operations_fast = True
            for operation, times in performance_results.items():
                if times['max'] > target_ms:
                    print(f"  ❌ {operation} max time {times['max']:.2f}ms exceeds {target_ms}ms target")
                    all_operations_fast = False
                else:
                    print(f"  ✅ {operation} meets <{target_ms}ms target")
            
            if all_operations_fast:
                print("✅ Performance validation: PASSED")
                self.test_results['performance'] = True
                return True
            else:
                print("❌ Performance validation: FAILED - Some operations exceed target")
                self.test_results['performance'] = False
                return False
                
        except Exception as e:
            print(f"❌ Performance validation: FAILED - {e}")
            self.test_results['performance'] = False
            return False
    
    async def validate_statistics_and_monitoring(self) -> bool:
        """Test 6: Validate statistics and health check functionality."""
        print("\n📊 Test 6: Statistics and Monitoring Validation")
        print("=" * 50)
        
        try:
            # Perform some operations to generate statistics
            print("🔧 Generating test data for statistics...")
            test_operations = [
                ("stats_key_1", {"data": "value1"}, 300),
                ("stats_key_2", {"data": "value2"}, 600),
                ("stats_key_3", {"data": "value3"}, None),
            ]
            
            for key, value, ttl in test_operations:
                await self.service.set(key, value, ttl)
                await self.service.get(key)  # Generate cache hit
            
            # Test get_stats method
            print("📈 Testing service statistics...")
            service_stats = self.service.get_stats()
            
            required_stats = [
                'total_operations', 'successful_operations', 'failed_operations',
                'success_rate', 'avg_response_time_ms', 'cache_hits', 'cache_misses',
                'total_requests', 'hit_rate', 'session_manager_available',
                'slo_target_ms', 'slo_compliant', 'health_status', 'uptime_seconds'
            ]
            
            for stat in required_stats:
                if stat not in service_stats:
                    print(f"  ❌ Missing statistic: {stat}")
                    return False
                print(f"  📊 {stat}: {service_stats[stat]}")
            
            # Verify key metrics
            if service_stats['total_operations'] < 6:  # Should have at least 6 operations
                print(f"  ❌ Expected at least 6 operations, got {service_stats['total_operations']}")
                return False
            
            if service_stats['success_rate'] != 1.0:
                print(f"  ❌ Expected 100% success rate, got {service_stats['success_rate']}")
                return False
            
            print("  ✅ Service statistics collection working")
            
            # Test get_database_stats method
            print("🗃️ Testing database statistics...")
            db_stats = await self.service.get_database_stats()
            
            if "error" in db_stats:
                print(f"  ❌ Database statistics error: {db_stats['error']}")
                return False
            
            required_db_stats = [
                'total_entries', 'valid_entries', 'expired_entries',
                'avg_access_count', 'last_access_time'
            ]
            
            for stat in required_db_stats:
                if stat not in db_stats:
                    print(f"  ❌ Missing database statistic: {stat}")
                    return False
                print(f"  📊 {stat}: {db_stats[stat]}")
            
            if db_stats['total_entries'] < 3:
                print(f"  ❌ Expected at least 3 entries, got {db_stats['total_entries']}")
                return False
            
            print("  ✅ Database statistics collection working")
            
            # Test health check
            print("🏥 Testing health check functionality...")
            health_result = await self.service.health_check()
            
            required_health_fields = ['healthy', 'checks', 'performance', 'timestamp']
            for field in required_health_fields:
                if field not in health_result:
                    print(f"  ❌ Missing health check field: {field}")
                    return False
            
            if not health_result['healthy']:
                print(f"  ❌ Health check failed: {health_result}")
                return False
            
            if not health_result['checks']['table_exists']:
                print("  ❌ Health check: table_exists failed")
                return False
            
            if not health_result['checks']['operations']:
                print("  ❌ Health check: operations failed")
                return False
            
            print(f"  🏥 Health status: {health_result['healthy']}")
            print(f"  📊 Check time: {health_result['performance']['total_check_time_ms']:.2f}ms")
            print("  ✅ Health check functionality working")
            
            print("✅ Statistics and monitoring validation: PASSED")
            self.test_results['statistics_monitoring'] = True
            return True
            
        except Exception as e:
            print(f"❌ Statistics and monitoring validation: FAILED - {e}")
            self.test_results['statistics_monitoring'] = False
            return False
    
    async def validate_json_serialization(self) -> bool:
        """Test 7: Validate JSON serialization in TEXT fields."""
        print("\n🔤 Test 7: JSON Serialization Validation")
        print("=" * 50)
        
        try:
            # Test various JSON data types
            test_cases = [
                ("json_string", "simple string"),
                ("json_number", 12345),
                ("json_float", 123.456),
                ("json_boolean", True),
                ("json_list", [1, 2, 3, "four", True]),
                ("json_dict", {"nested": {"key": "value"}, "list": [1, 2, 3]}),
                ("json_complex", {
                    "string": "test",
                    "number": 42,
                    "float": 3.14,
                    "boolean": False,
                    "null": None,
                    "list": [1, "two", {"three": 3}],
                    "nested": {"deep": {"deeper": {"deepest": "value"}}}
                }),
            ]
            
            print("🔧 Testing JSON serialization for various data types...")
            for key, value in test_cases:
                # Set value
                success = await self.service.set(key, value, 300)
                if not success:
                    print(f"  ❌ Failed to set {key}")
                    return False
                
                # Get value and verify
                retrieved = await self.service.get(key)
                if retrieved != value:
                    print(f"  ❌ Serialization mismatch for {key}:")
                    print(f"      Expected: {value}")
                    print(f"      Got: {retrieved}")
                    return False
                
                print(f"  ✅ {key}: {type(value).__name__} serialized correctly")
            
            # Test edge cases
            print("🔧 Testing JSON serialization edge cases...")
            edge_cases = [
                ("empty_string", ""),
                ("empty_list", []),
                ("empty_dict", {}),
                ("unicode_string", "Hello 世界 🌍"),
                ("special_chars", "\"quotes\" and \\backslashes\\ and \nnewlines"),
            ]
            
            for key, value in edge_cases:
                success = await self.service.set(key, value, 300)
                if not success:
                    print(f"  ❌ Failed to set edge case {key}")
                    return False
                
                retrieved = await self.service.get(key)
                if retrieved != value:
                    print(f"  ❌ Edge case mismatch for {key}:")
                    print(f"      Expected: {repr(value)}")
                    print(f"      Got: {repr(retrieved)}")
                    return False
                
                print(f"  ✅ {key}: Edge case handled correctly")
            
            # Verify data is actually stored as JSON TEXT in database
            print("🔍 Verifying TEXT field contains valid JSON...")
            async with self.container.get_session() as session:
                result = await session.execute(
                    text("SELECT cache_key, cache_value FROM cache_l3 WHERE cache_key = :key"),
                    {"key": "json_complex"}
                )
                row = result.fetchone()
                if not row:
                    print("  ❌ Complex JSON test case not found in database")
                    return False
                
                # Verify the stored value is valid JSON
                try:
                    stored_json = json.loads(row.cache_value)
                    expected = test_cases[6][1]  # json_complex test case
                    if stored_json != expected:
                        print(f"  ❌ Stored JSON doesn't match expected:")
                        print(f"      Expected: {expected}")
                        print(f"      Stored: {stored_json}")
                        return False
                    print("  ✅ JSON data correctly stored as TEXT in database")
                except json.JSONDecodeError as e:
                    print(f"  ❌ Stored data is not valid JSON: {e}")
                    return False
            
            print("✅ JSON serialization validation: PASSED")
            self.test_results['json_serialization'] = True
            return True
            
        except Exception as e:
            print(f"❌ JSON serialization validation: FAILED - {e}")
            self.test_results['json_serialization'] = False
            return False
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests and return comprehensive results."""
        print("🚀 L3DatabaseService SQL Compression Validation")
        print("=" * 70)
        print("Validating compressed SQL queries against real PostgreSQL...")
        print()
        
        start_time = time.perf_counter()
        
        # Run all validation tests
        tests = [
            ("Schema Creation", self.validate_schema_creation),
            ("CRUD Operations", self.validate_crud_operations), 
            ("UPSERT Conflict Resolution", self.validate_upsert_conflict_resolution),
            ("TTL and Expiration Logic", self.validate_ttl_expiration_logic),
            ("Performance Targets", self.validate_performance_targets),
            ("Statistics and Monitoring", self.validate_statistics_and_monitoring),
            ("JSON Serialization", self.validate_json_serialization),
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                success = await test_func()
                results[test_name.lower().replace(' ', '_')] = success
            except Exception as e:
                print(f"❌ {test_name}: FAILED with exception - {e}")
                results[test_name.lower().replace(' ', '_')] = False
        
        total_time = time.perf_counter() - start_time
        
        # Summary
        print("\n" + "=" * 70)
        print("📋 VALIDATION SUMMARY")
        print("=" * 70)
        
        passed_tests = sum(1 for success in results.values() if success)
        total_tests = len(results)
        success_rate = (passed_tests / total_tests) * 100
        
        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
        
        print()
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Time: {total_time:.2f} seconds")
        print()
        
        if success_rate == 100:
            print("🎉 ALL SQL COMPRESSION VALIDATIONS PASSED!")
            print("✅ L3DatabaseService compressed queries work correctly with PostgreSQL")
        else:
            print("❌ SOME VALIDATIONS FAILED")
            print("⚠️  SQL compression issues detected - review failed tests")
        
        return {
            "success_rate": success_rate,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "total_time": total_time,
            "test_results": results,
            "overall_success": success_rate == 100
        }


async def main():
    """Main function to run L3DatabaseService validation."""
    validator = L3DatabaseServiceValidator()
    
    try:
        await validator.setup()
        results = await validator.run_comprehensive_validation()
        return results["overall_success"]
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return False
    finally:
        await validator.teardown()


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)