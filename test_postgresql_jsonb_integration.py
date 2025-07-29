#!/usr/bin/env python3
"""
PostgreSQL JSONB integration testing for API metrics.
Tests serialization, storage, and retrieval of metrics data.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

from src.prompt_improver.metrics.api_metrics import (
    APIUsageMetric,
    UserJourneyMetric,
    RateLimitMetric,
    AuthenticationMetric,
    EndpointCategory,
    HTTPMethod,
    UserJourneyStage,
    AuthenticationMethod
)

# Try to import database components
try:
    from src.prompt_improver.database.connection_manager import get_connection_manager
    import psycopg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("PostgreSQL components not available - skipping database tests")


class PostgreSQLJSONBTester:
    """Test PostgreSQL JSONB integration for metrics."""
    
    def __init__(self):
        self.connection_manager = None
        self.test_table_created = False
    
    async def setup_database(self):
        """Set up test database and tables."""
        if not POSTGRES_AVAILABLE:
            return False
        
        try:
            self.connection_manager = get_connection_manager()
            
            # Create test table for metrics storage
            async with self.connection_manager.get_connection() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS test_api_metrics (
                        id SERIAL PRIMARY KEY,
                        metric_type VARCHAR(50) NOT NULL,
                        metric_data JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        endpoint VARCHAR(255),
                        user_id VARCHAR(100),
                        session_id VARCHAR(100)
                    )
                """)
                
                # Create indexes for JSONB queries
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_test_api_metrics_jsonb_gin 
                    ON test_api_metrics USING GIN (metric_data)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_test_api_metrics_endpoint 
                    ON test_api_metrics (endpoint)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_test_api_metrics_user_id 
                    ON test_api_metrics (user_id)
                """)
                
                await conn.commit()
            
            self.test_table_created = True
            print("✓ Test database tables created")
            return True
            
        except Exception as e:
            print(f"✗ Database setup failed: {e}")
            return False
    
    async def cleanup_database(self):
        """Clean up test data."""
        if not self.test_table_created:
            return
        
        try:
            async with self.connection_manager.get_connection() as conn:
                await conn.execute("DROP TABLE IF EXISTS test_api_metrics")
                await conn.commit()
            print("✓ Test database cleaned up")
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")
    
    def serialize_metric_to_jsonb(self, metric) -> Dict[str, Any]:
        """Serialize metric object to JSONB-compatible dict."""
        if isinstance(metric, APIUsageMetric):
            return {
                "endpoint": metric.endpoint,
                "method": metric.method.value,
                "category": metric.category.value,
                "status_code": metric.status_code,
                "response_time_ms": metric.response_time_ms,
                "request_size_bytes": metric.request_size_bytes,
                "response_size_bytes": metric.response_size_bytes,
                "user_id": metric.user_id,
                "session_id": metric.session_id,
                "ip_address": metric.ip_address,
                "user_agent": metric.user_agent,
                "timestamp": metric.timestamp.isoformat(),
                "query_parameters_count": metric.query_parameters_count,
                "payload_type": metric.payload_type,
                "rate_limited": metric.rate_limited,
                "cache_hit": metric.cache_hit,
                "authentication_method": metric.authentication_method.value,
                "api_version": metric.api_version
            }
        elif isinstance(metric, UserJourneyMetric):
            return {
                "user_id": metric.user_id,
                "session_id": metric.session_id,
                "journey_stage": metric.journey_stage.value,
                "event_type": metric.event_type,
                "endpoint": metric.endpoint,
                "success": metric.success,
                "conversion_value": metric.conversion_value,
                "time_to_action_seconds": metric.time_to_action_seconds,
                "previous_stage": metric.previous_stage.value if metric.previous_stage else None,
                "feature_flags_active": metric.feature_flags_active,
                "cohort_id": metric.cohort_id,
                "timestamp": metric.timestamp.isoformat(),
                "metadata": metric.metadata
            }
        else:
            raise ValueError(f"Unsupported metric type: {type(metric)}")
    
    async def test_jsonb_serialization(self):
        """Test JSONB serialization and deserialization."""
        print("🧪 Test: JSONB Serialization")
        
        try:
            # Create test metrics
            api_metric = APIUsageMetric(
                endpoint="/api/v1/test",
                method=HTTPMethod.POST,
                category=EndpointCategory.PROMPT_IMPROVEMENT,
                status_code=200,
                response_time_ms=150.5,
                request_size_bytes=1024,
                response_size_bytes=2048,
                user_id="test_user",
                session_id="test_session",
                ip_address="192.168.1.1",
                user_agent="TestAgent/1.0",
                timestamp=datetime.now(timezone.utc),
                query_parameters_count=3,
                payload_type="application/json",
                rate_limited=False,
                cache_hit=True,
                authentication_method=AuthenticationMethod.JWT_TOKEN,
                api_version="v1"
            )
            
            journey_metric = UserJourneyMetric(
                user_id="test_user",
                session_id="test_session",
                journey_stage=UserJourneyStage.FIRST_USE,
                event_type="test_event",
                endpoint="/api/v1/test",
                success=True,
                conversion_value=10.0,
                time_to_action_seconds=30.0,
                previous_stage=UserJourneyStage.ONBOARDING,
                feature_flags_active=["test_flag"],
                cohort_id="test_cohort",
                timestamp=datetime.now(timezone.utc),
                metadata={"test": "data", "nested": {"key": "value"}}
            )
            
            # Test serialization
            api_data = self.serialize_metric_to_jsonb(api_metric)
            journey_data = self.serialize_metric_to_jsonb(journey_metric)
            
            # Verify JSON serialization works
            api_json = json.dumps(api_data)
            journey_json = json.dumps(journey_data)
            
            # Verify deserialization works
            api_restored = json.loads(api_json)
            journey_restored = json.loads(journey_json)
            
            # Verify data integrity
            assert api_restored["endpoint"] == api_metric.endpoint
            assert api_restored["method"] == api_metric.method.value
            assert api_restored["status_code"] == api_metric.status_code
            
            assert journey_restored["user_id"] == journey_metric.user_id
            assert journey_restored["journey_stage"] == journey_metric.journey_stage.value
            assert journey_restored["metadata"]["nested"]["key"] == "value"
            
            print("✓ JSONB serialization: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ JSONB serialization: FAILED - {e}")
            return False
    
    async def test_database_storage_retrieval(self):
        """Test storing and retrieving metrics from PostgreSQL."""
        print("🧪 Test: Database Storage & Retrieval")
        
        if not self.test_table_created:
            print("✗ Database storage: SKIPPED - No database connection")
            return False
        
        try:
            # Create test metrics
            metrics = []
            for i in range(5):
                api_metric = APIUsageMetric(
                    endpoint=f"/api/v1/test_{i}",
                    method=HTTPMethod.GET if i % 2 == 0 else HTTPMethod.POST,
                    category=EndpointCategory.PROMPT_IMPROVEMENT,
                    status_code=200,
                    response_time_ms=100.0 + i * 10,
                    request_size_bytes=500 + i * 100,
                    response_size_bytes=1000 + i * 200,
                    user_id=f"user_{i}",
                    session_id=f"session_{i}",
                    ip_address="192.168.1.1",
                    user_agent="TestAgent/1.0",
                    timestamp=datetime.now(timezone.utc),
                    query_parameters_count=i,
                    payload_type="application/json",
                    rate_limited=False,
                    cache_hit=i % 2 == 0,
                    authentication_method=AuthenticationMethod.JWT_TOKEN,
                    api_version="v1"
                )
                metrics.append(("api_usage", api_metric))
            
            # Store metrics in database
            async with self.connection_manager.get_connection() as conn:
                for metric_type, metric in metrics:
                    metric_data = self.serialize_metric_to_jsonb(metric)
                    
                    await conn.execute("""
                        INSERT INTO test_api_metrics 
                        (metric_type, metric_data, endpoint, user_id, session_id)
                        VALUES ($1, $2, $3, $4, $5)
                    """, metric_type, json.dumps(metric_data), 
                        metric.endpoint, metric.user_id, metric.session_id)
                
                await conn.commit()
            
            # Retrieve and verify metrics
            async with self.connection_manager.get_connection() as conn:
                # Test basic retrieval
                rows = await conn.fetch("""
                    SELECT metric_type, metric_data, endpoint, user_id 
                    FROM test_api_metrics 
                    ORDER BY id
                """)
                
                assert len(rows) == 5
                
                # Test JSONB queries
                jsonb_rows = await conn.fetch("""
                    SELECT metric_data 
                    FROM test_api_metrics 
                    WHERE metric_data->>'method' = 'GET'
                """)
                
                assert len(jsonb_rows) == 3  # 3 GET requests
                
                # Test JSONB indexing performance
                start_time = time.time()
                performance_rows = await conn.fetch("""
                    SELECT COUNT(*) 
                    FROM test_api_metrics 
                    WHERE metric_data->>'status_code' = '200'
                """)
                query_time = (time.time() - start_time) * 1000
                
                assert performance_rows[0][0] == 5  # All have status 200
                
                print(f"  JSONB query time: {query_time:.2f}ms")
            
            print("✓ Database storage & retrieval: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Database storage & retrieval: FAILED - {e}")
            return False
    
    async def test_jsonb_indexing_performance(self):
        """Test JSONB indexing and query performance."""
        print("🧪 Test: JSONB Indexing Performance")
        
        if not self.test_table_created:
            print("✗ JSONB indexing: SKIPPED - No database connection")
            return False
        
        try:
            # Insert larger dataset for performance testing
            async with self.connection_manager.get_connection() as conn:
                for i in range(100):
                    api_metric = APIUsageMetric(
                        endpoint=f"/api/v1/perf_test_{i % 10}",
                        method=HTTPMethod.GET if i % 2 == 0 else HTTPMethod.POST,
                        category=EndpointCategory.PROMPT_IMPROVEMENT,
                        status_code=200 if i % 10 != 9 else 500,
                        response_time_ms=100.0 + (i % 50) * 5,
                        request_size_bytes=500 + i * 10,
                        response_size_bytes=1000 + i * 20,
                        user_id=f"perf_user_{i % 20}",
                        session_id=f"perf_session_{i}",
                        ip_address="192.168.1.1",
                        user_agent="PerfTestAgent/1.0",
                        timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                        query_parameters_count=i % 5,
                        payload_type="application/json",
                        rate_limited=i % 20 == 0,
                        cache_hit=i % 3 == 0,
                        authentication_method=AuthenticationMethod.JWT_TOKEN,
                        api_version="v1"
                    )
                    
                    metric_data = self.serialize_metric_to_jsonb(api_metric)
                    await conn.execute("""
                        INSERT INTO test_api_metrics 
                        (metric_type, metric_data, endpoint, user_id, session_id)
                        VALUES ($1, $2, $3, $4, $5)
                    """, "api_usage", json.dumps(metric_data), 
                        api_metric.endpoint, api_metric.user_id, api_metric.session_id)
                
                await conn.commit()
            
            # Test various JSONB query patterns
            query_tests = [
                ("Simple JSONB field access", "SELECT COUNT(*) FROM test_api_metrics WHERE metric_data->>'method' = 'GET'"),
                ("JSONB numeric comparison", "SELECT COUNT(*) FROM test_api_metrics WHERE (metric_data->>'status_code')::int = 200"),
                ("JSONB boolean query", "SELECT COUNT(*) FROM test_api_metrics WHERE (metric_data->>'cache_hit')::boolean = true"),
                ("Complex JSONB query", "SELECT COUNT(*) FROM test_api_metrics WHERE metric_data->>'method' = 'POST' AND (metric_data->>'status_code')::int >= 400"),
                ("JSONB aggregation", "SELECT AVG((metric_data->>'response_time_ms')::float) FROM test_api_metrics")
            ]
            
            performance_results = {}
            
            async with self.connection_manager.get_connection() as conn:
                for test_name, query in query_tests:
                    start_time = time.time()
                    result = await conn.fetch(query)
                    query_time = (time.time() - start_time) * 1000
                    
                    performance_results[test_name] = query_time
                    print(f"  {test_name}: {query_time:.2f}ms")
            
            # Check if all queries are reasonably fast (under 50ms for this dataset size)
            max_acceptable_time = 50.0  # ms
            all_fast = all(time < max_acceptable_time for time in performance_results.values())
            
            if all_fast:
                print("✓ JSONB indexing performance: PASSED")
            else:
                slow_queries = [name for name, time in performance_results.items() if time >= max_acceptable_time]
                print(f"✗ JSONB indexing performance: FAILED - Slow queries: {slow_queries}")
            
            return all_fast
            
        except Exception as e:
            print(f"✗ JSONB indexing performance: FAILED - {e}")
            return False


async def main():
    """Run PostgreSQL JSONB integration tests."""
    print("🚀 PostgreSQL JSONB Integration Testing")
    print("=" * 50)
    
    if not POSTGRES_AVAILABLE:
        print("❌ PostgreSQL not available - skipping all database tests")
        return False
    
    tester = PostgreSQLJSONBTester()
    
    try:
        # Setup
        setup_success = await tester.setup_database()
        if not setup_success:
            print("❌ Database setup failed - skipping tests")
            return False
        
        # Run tests
        tests = [
            tester.test_jsonb_serialization,
            tester.test_database_storage_retrieval,
            tester.test_jsonb_indexing_performance
        ]
        
        results = []
        for test in tests:
            result = await test()
            results.append(result)
            print()
        
        # Summary
        passed = sum(results)
        total = len(results)
        success_rate = (passed / total) * 100
        
        print("📊 POSTGRESQL JSONB TEST SUMMARY")
        print("=" * 50)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("🎉 All PostgreSQL JSONB tests passed!")
            return True
        else:
            print("❌ Some PostgreSQL JSONB tests failed")
            return False
    
    finally:
        await tester.cleanup_database()


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
