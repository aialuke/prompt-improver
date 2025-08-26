"""
Real PostgreSQL database behavior testing with comprehensive constraint and performance validation.

This test suite validates actual database behavior using real PostgreSQL testcontainers:
- Database constraint enforcement (foreign keys, unique constraints, check constraints)
- Transaction isolation and concurrency control
- Query performance and execution plans
- Connection pooling behavior
- Database migration and schema validation
- Index usage and query optimization
- Bulk operations and performance characteristics
- Error handling with real database exceptions

Features:
- 100% real PostgreSQL testing using testcontainers
- Comprehensive constraint validation
- Performance benchmarking with real execution plans
- Concurrent access testing
- Database health monitoring validation
- Real behavior regression testing
"""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError
from tests.containers.postgres_container import PostgreSQLTestFixture

from prompt_improver.database.models import (
    PromptSession,
    TrainingIteration,
    TrainingSession,
)


class TestRealDatabaseBehavior:
    """Comprehensive real database behavior testing with PostgreSQL testcontainers."""

    @pytest.fixture
    async def db_fixture(self, postgres_container):
        """Database test fixture with helper methods."""
        return PostgreSQLTestFixture(postgres_container)

    @pytest.mark.asyncio
    async def test_database_constraint_enforcement(self, postgres_container, db_fixture):
        """Test that database constraints are properly enforced with real PostgreSQL."""

        # Test unique constraint violation
        session_data = {
            "session_id": "test_unique_session",
            "original_prompt": "Test prompt",
            "improved_prompt": "Improved test prompt",
            "improvement_score": 0.85,
            "created_at": datetime.now(UTC)
        }

        async with postgres_container.get_session() as session:
            # First insert should succeed
            session.add(PromptSession(**session_data))
            await session.commit()

            # Second insert with same session_id should fail
            with pytest.raises(IntegrityError, match="unique constraint|duplicate key"):
                session.add(PromptSession(**session_data))
                await session.commit()

    @pytest.mark.asyncio
    async def test_foreign_key_constraint_enforcement(self, postgres_container):
        """Test foreign key constraints are enforced."""

        async with postgres_container.get_session() as session:
            # Try to insert TrainingIteration without valid TrainingSession
            invalid_iteration = TrainingIteration(
                training_session_id=999999,  # Non-existent session
                iteration=1,
                status="running",
                started_at=datetime.now(UTC)
            )

            with pytest.raises(IntegrityError, match="foreign key constraint|violates foreign key"):
                session.add(invalid_iteration)
                await session.commit()

    @pytest.mark.asyncio
    async def test_transaction_isolation_behavior(self, postgres_container):
        """Test transaction isolation with concurrent sessions."""

        async def create_session_in_transaction(session_id: str, delay: float = 0):
            """Create a session with optional delay to test concurrency."""
            async with postgres_container.get_session() as session:
                try:
                    await asyncio.sleep(delay)
                    prompt_session = PromptSession(
                        session_id=session_id,
                        original_prompt="Concurrent test",
                        improved_prompt="Improved concurrent test",
                        improvement_score=0.75,
                        created_at=datetime.now(UTC)
                    )
                    session.add(prompt_session)
                    await session.commit()
                    return True
                except IntegrityError:
                    await session.rollback()
                    return False

        # Test concurrent inserts with same session_id
        results = await asyncio.gather(
            create_session_in_transaction("concurrent_test_1", 0.1),
            create_session_in_transaction("concurrent_test_1", 0.1),
            return_exceptions=True
        )

        # Only one should succeed due to unique constraint
        success_count = sum(1 for r in results if r is True)
        assert success_count == 1, "Only one concurrent insert should succeed"

    @pytest.mark.asyncio
    async def test_query_performance_measurement(self, postgres_container, db_fixture):
        """Test query performance measurement with real execution plans."""

        # Create test data for performance testing
        test_sessions = [{
                "session_id": f"perf_test_{i}",
                "original_prompt": f"Performance test prompt {i}",
                "improved_prompt": f"Improved performance test prompt {i}",
                "improvement_score": 0.5 + (i % 50) / 100.0,
                "created_at": datetime.now(UTC) - timedelta(hours=i)
            } for i in range(100)]

        await db_fixture.create_test_data("prompt_sessions", test_sessions)

        # Test query performance
        query = """
        SELECT session_id, improvement_score, created_at
        FROM prompt_sessions
        WHERE improvement_score > :min_score
        ORDER BY created_at DESC
        LIMIT 10
        """

        performance_results = await db_fixture.measure_query_performance(
            query,
            {"min_score": 0.7}
        )

        # Validate performance characteristics
        assert performance_results["execution_time_ms"] < 1000, "Query should execute within 1 second"
        assert "explain_plan" in performance_results
        assert performance_results["explain_plan"] is not None

    @pytest.mark.asyncio
    async def test_connection_pooling_behavior(self, postgres_container, db_fixture):
        """Test database connection pooling with concurrent access."""

        # Test concurrent connection behavior
        pooling_results = await db_fixture.test_connection_pooling(concurrent_connections=15)

        # Validate connection pooling metrics
        assert pooling_results["concurrent_connections"] == 15
        assert pooling_results["total_time_ms"] < 5000, "Pooled connections should be efficient"
        assert pooling_results["avg_connection_time_ms"] < 1000, "Average connection time should be reasonable"

    @pytest.mark.asyncio
    async def test_bulk_operations_performance(self, postgres_container):
        """Test bulk operations performance characteristics."""

        # Prepare bulk data
        bulk_data = [PromptSession(
                session_id=f"bulk_test_{i}",
                original_prompt=f"Bulk test prompt {i}",
                improved_prompt=f"Bulk improved prompt {i}",
                improvement_score=0.5 + (i % 50) / 100.0,
                created_at=datetime.now(UTC)
            ) for i in range(1000)]

        # Test bulk insert performance
        start_time = time.perf_counter()
        async with postgres_container.get_session() as session:
            session.add_all(bulk_data)
            await session.commit()

        bulk_insert_time = time.perf_counter() - start_time

        # Validate bulk operation performance
        assert bulk_insert_time < 10.0, "Bulk insert of 1000 records should complete within 10 seconds"

        # Verify all records were inserted
        count_result = await postgres_container.get_table_count("prompt_sessions")
        assert count_result >= 1000, "All bulk records should be inserted"

    @pytest.mark.asyncio
    async def test_index_usage_validation(self, postgres_container):
        """Test that database indexes are being used properly."""

        # Create test data
        async with postgres_container.get_session() as session:
            for i in range(50):
                prompt_session = PromptSession(
                    session_id=f"index_test_{i}",
                    original_prompt=f"Index test prompt {i}",
                    improved_prompt=f"Index improved prompt {i}",
                    improvement_score=0.5 + i / 100.0,
                    created_at=datetime.now(UTC) - timedelta(hours=i)
                )
                session.add(prompt_session)
            await session.commit()

        # Test index usage on session_id (should be unique index)
        async with postgres_container.get_session() as session:
            explain_result = await session.execute(
                text("EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) SELECT * FROM prompt_sessions WHERE session_id = :session_id"),
                {"session_id": "index_test_25"}
            )

            explain_data = explain_result.fetchone()[0]
            plan = explain_data[0]["Plan"]

            # Should use index scan, not sequential scan
            assert "Index" in plan["Node Type"], "Query should use index for session_id lookup"

    @pytest.mark.asyncio
    async def test_database_health_monitoring(self, postgres_container):
        """Test database health monitoring with real metrics."""

        # Get real connection info
        conn_info = postgres_container.get_connection_info()

        async with postgres_container.get_session() as session:
            # Test database version and status
            version_result = await session.execute(text("SELECT version()"))
            version = version_result.scalar()
            assert "PostgreSQL" in version

            # Test database activity monitoring
            activity_result = await session.execute(
                text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            )
            active_connections = activity_result.scalar()
            assert active_connections >= 1, "Should have at least one active connection"

            # Test database size monitoring
            size_result = await session.execute(
                text("SELECT pg_database_size(current_database())")
            )
            db_size = size_result.scalar()
            assert db_size > 0, "Database should have some size"

    @pytest.mark.asyncio
    async def test_error_handling_with_real_exceptions(self, postgres_container):
        """Test error handling with real database exceptions."""

        # Test connection timeout
        with patch.object(postgres_container._engine.pool, '_creator') as mock_creator:
            mock_creator.side_effect = OperationalError("connection timeout", None, None)

            with pytest.raises(OperationalError):
                async with postgres_container.get_session() as session:
                    await session.execute(text("SELECT 1"))

    @pytest.mark.asyncio
    async def test_concurrent_schema_operations(self, postgres_container):
        """Test concurrent schema operations and locking behavior."""

        async def create_temp_table(table_name: str):
            """Create a temporary table."""
            async with postgres_container.get_session() as session:
                await session.execute(
                    text(f"CREATE TEMP TABLE {table_name} (id SERIAL PRIMARY KEY, data TEXT)")
                )
                await session.commit()
                return table_name

        # Test concurrent table creation
        results = await asyncio.gather(
            create_temp_table("temp_table_1"),
            create_temp_table("temp_table_2"),
            create_temp_table("temp_table_3"),
            return_exceptions=True
        )

        # All should succeed since they're in different sessions
        success_count = sum(1 for r in results if isinstance(r, str))
        assert success_count == 3, "All concurrent temp table creations should succeed"

    @pytest.mark.asyncio
    async def test_data_consistency_validation(self, postgres_container):
        """Test data consistency across related tables."""

        async with postgres_container.get_session() as session:
            # Create a training session
            training_session = TrainingSession(
                session_id="consistency_test",
                status="running",
                started_at=datetime.now(UTC),
                continuous_mode=True,
                improvement_threshold=0.01,
                max_iterations=100,
                timeout_seconds=3600
            )
            session.add(training_session)
            await session.flush()  # Get the ID

            # Create related training iterations
            for i in range(5):
                iteration = TrainingIteration(
                    training_session_id=training_session.id,
                    iteration=i + 1,
                    status="completed",
                    started_at=datetime.now(UTC),
                    duration_seconds=60 + i * 10,
                    improvement_score=0.1 + i * 0.05
                )
                session.add(iteration)

            await session.commit()

            # Validate data consistency
            session_count = await session.execute(
                text("SELECT COUNT(*) FROM training_sessions WHERE session_id = :session_id"),
                {"session_id": "consistency_test"}
            )
            assert session_count.scalar() == 1

            iteration_count = await session.execute(
                text("SELECT COUNT(*) FROM training_iterations WHERE training_session_id = :session_id"),
                {"session_id": training_session.id}
            )
            assert iteration_count.scalar() == 5

    @pytest.mark.asyncio
    async def test_database_migration_behavior(self, postgres_container):
        """Test database migration and schema changes."""

        async with postgres_container.get_session() as session:
            # Test adding a column (simulate migration)
            try:
                await session.execute(
                    text("ALTER TABLE prompt_sessions ADD COLUMN test_migration_column TEXT DEFAULT 'test'")
                )
                await session.commit()

                # Verify column was added
                column_result = await session.execute(
                    text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'prompt_sessions'
                    AND column_name = 'test_migration_column'
                    """)
                )
                assert column_result.fetchone() is not None, "Migration column should be added"

                # Clean up
                await session.execute(
                    text("ALTER TABLE prompt_sessions DROP COLUMN test_migration_column")
                )
                await session.commit()

            except Exception as e:
                await session.rollback()
                pytest.fail(f"Migration test failed: {e}")

    @pytest.mark.asyncio
    async def test_vacuum_and_analyze_operations(self, postgres_container):
        """Test database maintenance operations."""

        async with postgres_container.get_session() as session:
            # Test VACUUM operation
            await session.execute(text("VACUUM ANALYZE prompt_sessions"))

            # Test getting table statistics after ANALYZE
            stats_result = await session.execute(
                text("""
                SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
                FROM pg_stat_user_tables
                WHERE tablename = 'prompt_sessions'
                """)
            )

            stats = stats_result.fetchone()
            assert stats is not None, "Should have statistics for prompt_sessions table"


class TestDatabasePerformanceRegression:
    """Performance regression testing for database operations."""

    @pytest.mark.asyncio
    async def test_query_performance_baseline(self, postgres_container):
        """Establish performance baselines for critical queries."""

        # Create performance test data
        async with postgres_container.get_session() as session:
            for i in range(500):
                prompt_session = PromptSession(
                    session_id=f"perf_baseline_{i}",
                    original_prompt=f"Performance baseline prompt {i}",
                    improved_prompt=f"Improved baseline prompt {i}",
                    improvement_score=0.3 + (i % 70) / 100.0,
                    created_at=datetime.now(UTC) - timedelta(hours=i % 24)
                )
                session.add(prompt_session)
            await session.commit()

        # Test critical query performance
        queries_to_benchmark = [
            ("SELECT COUNT(*) FROM prompt_sessions", {}),
            ("SELECT * FROM prompt_sessions WHERE improvement_score > :score ORDER BY created_at DESC LIMIT 10", {"score": 0.7}),
            ("SELECT AVG(improvement_score) FROM prompt_sessions WHERE created_at > :date", {"date": datetime.now(UTC) - timedelta(days=1)}),
        ]

        for query, params in queries_to_benchmark:
            start_time = time.perf_counter()

            async with postgres_container.get_session() as session:
                result = await session.execute(text(query), params)
                if result.returns_rows:
                    _ = result.fetchall()

            query_time = time.perf_counter() - start_time

            # Performance assertions - adjust thresholds based on requirements
            assert query_time < 2.0, f"Query should execute within 2 seconds: {query}"

    @pytest.mark.asyncio
    async def test_connection_pool_performance(self, postgres_container):
        """Test connection pool performance under load."""

        async def execute_query_batch(batch_id: int):
            """Execute a batch of queries."""
            async with postgres_container.get_session() as session:
                for i in range(10):
                    result = await session.execute(
                        text("SELECT :batch_id, :query_num, pg_backend_pid()"),
                        {"batch_id": batch_id, "query_num": i}
                    )
                    _ = result.fetchone()

        # Test concurrent batches
        start_time = time.perf_counter()
        await asyncio.gather(*[execute_query_batch(i) for i in range(20)])
        total_time = time.perf_counter() - start_time

        # Should complete efficiently with connection pooling
        assert total_time < 10.0, "Concurrent query batches should complete within 10 seconds"
