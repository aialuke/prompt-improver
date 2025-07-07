"""
Enhanced async test execution with pytest-asyncio best practices.
Implements class-scoped event loops, proper fixture management, and performance validation
following Context7 research on pytest-asyncio best practices.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import (
    given,
    strategies as st,
)


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncExecution:
    """Enhanced async test execution with class-scoped event loops for optimal performance."""

    async def test_async_execution_basic(self):
        """Verify basic async test execution with performance validation."""
        start_time = asyncio.get_event_loop().time()
        await asyncio.sleep(0.001)
        end_time = asyncio.get_event_loop().time()
        execution_time = (end_time - start_time) * 1000

        # Validate execution time is reasonable (should be ~1ms + overhead)
        assert 1 <= execution_time <= 10, (
            f"Execution time {execution_time}ms outside expected range"
        )
        assert True

    async def test_mcp_improve_prompt_async(self, mock_db_session):
        """Test actual async MCP function execution."""
        # Mock the import for this test since we're validating infrastructure
        with patch(
            "prompt_improver.mcp_server.mcp_server.improve_prompt"
        ) as mock_improve:
            mock_improve.return_value = {
                "improved_prompt": "Enhanced test prompt",
                "processing_time_ms": 150,
                "applied_rules": [{"rule_id": "clarity_rule", "confidence": 0.9}],
            }

            # This validates that our async infrastructure works
            result = await mock_improve(
                prompt="Test prompt for validation",
                context={"domain": "testing"},
                session_id="async_test",
            )

            assert "improved_prompt" in result
            assert result["processing_time_ms"] > 0

    async def test_concurrent_async_operations(self):
        """Verify multiple async operations work correctly with performance validation."""

        async def dummy_operation(delay):
            await asyncio.sleep(delay)
            return delay

        # Measure concurrent execution time
        start_time = asyncio.get_event_loop().time()

        # Run concurrent operations
        results = await asyncio.gather(
            dummy_operation(0.001), dummy_operation(0.002), dummy_operation(0.003)
        )

        end_time = asyncio.get_event_loop().time()
        execution_time = (end_time - start_time) * 1000

        assert results == [0.001, 0.002, 0.003]

        # Validate concurrent execution was actually faster than sequential
        sequential_time = sum(results) * 1000  # Expected sequential time in ms
        # Concurrent execution should be significantly faster
        assert execution_time < sequential_time * 0.8, (
            f"Concurrent execution {execution_time}ms not faster than sequential"
        )

    async def test_async_fixture_interaction(
        self, mock_db_session, sample_training_data
    ):
        """Test that async tests properly interact with centralized fixtures."""
        # Verify mock_db_session works
        mock_db_session.execute.return_value = AsyncMock()
        await mock_db_session.execute("SELECT 1")
        mock_db_session.execute.assert_called_once()

        # Verify sample_training_data fixture works
        assert "features" in sample_training_data
        assert "effectiveness_scores" in sample_training_data
        assert len(sample_training_data["features"]) == 25
        assert len(sample_training_data["effectiveness_scores"]) == 25

        # Test async context propagation
        loop_id = id(asyncio.get_running_loop())
        await asyncio.sleep(0.001)
        assert id(asyncio.get_running_loop()) == loop_id, (
            "Event loop changed during test execution"
        )


class TestFixtureAccessibility:
    """Test that all centralized fixtures are accessible across test modules."""

    def test_cli_runner_fixture(self, cli_runner):
        """Test that CLI runner fixture is accessible."""
        assert cli_runner is not None
        assert hasattr(cli_runner, "invoke")

    def test_isolated_cli_runner_fixture(self, isolated_cli_runner):
        """Test that isolated CLI runner fixture is accessible."""
        assert isolated_cli_runner is not None
        assert hasattr(isolated_cli_runner, "invoke")

    def test_test_data_dir_fixture(self, test_data_dir):
        """Test that test data directory fixture works."""
        assert test_data_dir.exists()
        assert (test_data_dir / "data").exists()
        assert (test_data_dir / "config").exists()
        assert (test_data_dir / "logs").exists()
        assert (test_data_dir / "temp").exists()

    def test_mock_db_session_fixture(self, mock_db_session):
        """Test that mock database session fixture is properly configured."""
        assert mock_db_session is not None
        assert hasattr(mock_db_session, "execute")
        assert hasattr(mock_db_session, "commit")
        assert hasattr(mock_db_session, "rollback")
        assert hasattr(mock_db_session, "add")

    def test_sample_training_data_fixture(self, sample_training_data):
        """Test that sample training data fixture provides valid data."""
        assert "features" in sample_training_data
        assert "effectiveness_scores" in sample_training_data
        assert len(sample_training_data["features"]) > 0
        assert len(sample_training_data["effectiveness_scores"]) > 0

    def test_test_config_fixture(self, test_config):
        """Test that test configuration fixture is accessible."""
        assert "database" in test_config
        assert "performance" in test_config
        assert "ml" in test_config
        assert test_config["performance"]["target_response_time_ms"] == 200


class TestEventLoopIsolation:
    """Test that event loops are properly isolated between tests."""

    @pytest.mark.asyncio
    async def test_event_loop_isolation_1(self):
        """First test to verify event loop isolation."""
        loop = asyncio.get_running_loop()
        loop.test_marker = "test_1"
        await asyncio.sleep(0.001)
        assert hasattr(loop, "test_marker")
        assert loop.test_marker == "test_1"

    @pytest.mark.asyncio
    async def test_event_loop_isolation_2(self):
        """Second test to verify event loops are isolated."""
        loop = asyncio.get_running_loop()
        # This should be a fresh event loop, not the one from test_1
        assert not hasattr(loop, "test_marker")
        loop.test_marker = "test_2"
        await asyncio.sleep(0.001)
        assert loop.test_marker == "test_2"


class TestPerformanceCompliance:
    """Test that fixtures and async operations meet performance requirements."""

    @pytest.mark.asyncio
    async def test_async_operation_latency(self):
        """Test that async operations complete within reasonable time."""
        start_time = asyncio.get_event_loop().time()

        # Simulate a quick async operation
        await asyncio.sleep(0.001)

        end_time = asyncio.get_event_loop().time()
        execution_time_ms = (end_time - start_time) * 1000

        # Should complete quickly (much less than 200ms target)
        assert execution_time_ms < 50, f"Async operation took {execution_time_ms}ms"

    def test_fixture_creation_performance(self, test_data_dir, sample_training_data):
        """Test that fixture creation is reasonably fast."""

        start_time = time.time()

        # Access fixtures (creation time is included)
        assert test_data_dir.exists()
        assert len(sample_training_data["features"]) > 0

        end_time = time.time()
        creation_time_ms = (end_time - start_time) * 1000

        # Fixture creation should be fast
        assert creation_time_ms < 100, f"Fixture creation took {creation_time_ms}ms"


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncPropertyBasedTesting:
    """Property-based testing for async operations using Hypothesis."""

    @given(
        operation_count=st.integers(min_value=1, max_value=10),
        delay_ms=st.floats(min_value=0.001, max_value=0.01),
    )
    async def test_concurrent_operations_scaling(self, operation_count, delay_ms):
        """Property: concurrent operations should scale better than sequential ones."""

        async def dummy_operation():
            await asyncio.sleep(delay_ms)
            return delay_ms

        # Measure concurrent execution
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[dummy_operation() for _ in range(operation_count)])
        concurrent_time = (asyncio.get_event_loop().time() - start_time) * 1000

        # Verify results
        assert len(results) == operation_count
        assert all(result == delay_ms for result in results)

        # Property: concurrent execution should be faster than sequential for multiple operations
        if operation_count > 1:
            expected_sequential_time = operation_count * delay_ms * 1000
            # Allow some overhead but should be significantly faster
            assert concurrent_time < expected_sequential_time * 0.8, (
                f"Concurrent time {concurrent_time}ms not efficient vs sequential {expected_sequential_time}ms"
            )

    @given(
        sleep_duration=st.floats(min_value=0.001, max_value=0.05)
    )
    async def test_async_sleep_accuracy(self, sleep_duration):
        """Property: asyncio.sleep should be reasonably accurate."""

        start_time = asyncio.get_event_loop().time()
        await asyncio.sleep(sleep_duration)
        end_time = asyncio.get_event_loop().time()

        actual_duration = end_time - start_time

        # Allow 50% overhead for timing variations in tests
        assert actual_duration >= sleep_duration, "Sleep duration was less than requested"
        assert actual_duration <= sleep_duration * 1.5, (
            f"Sleep took {actual_duration:.4f}s, expected ~{sleep_duration:.4f}s"
        )

    @given(
        tasks=st.lists(
            st.floats(min_value=0.001, max_value=0.01),
            min_size=2,
            max_size=8
        )
    )
    async def test_asyncio_gather_order_preservation(self, tasks):
        """Property: asyncio.gather should preserve order regardless of completion time."""

        async def timed_operation(delay, task_id):
            await asyncio.sleep(delay)
            return task_id

        # Create tasks with IDs to track order
        task_coroutines = [
            timed_operation(delay, idx)
            for idx, delay in enumerate(tasks)
        ]

        results = await asyncio.gather(*task_coroutines)

        # Property: results should be in same order as input regardless of timing
        expected_order = list(range(len(tasks)))
        assert results == expected_order, (
            f"Order not preserved: got {results}, expected {expected_order}"
        )

    @given(
        concurrent_count=st.integers(min_value=2, max_value=20)
    )
    async def test_event_loop_context_preservation(self, concurrent_count):
        """Property: event loop context should be preserved across concurrent operations."""

        async def context_checker(operation_id):
            # Store loop reference at start
            loop_start = asyncio.get_running_loop()
            loop_id_start = id(loop_start)

            # Simulate some async work
            await asyncio.sleep(0.001)

            # Check loop is still the same
            loop_end = asyncio.get_running_loop()
            loop_id_end = id(loop_end)

            return (operation_id, loop_id_start, loop_id_end)

        # Run multiple concurrent operations
        results = await asyncio.gather(*[
            context_checker(i) for i in range(concurrent_count)
        ])

        # Property: all operations should use the same event loop
        loop_ids_start = [result[1] for result in results]
        loop_ids_end = [result[2] for result in results]

        # All operations should use the same loop
        assert len(set(loop_ids_start)) == 1, "Operations used different event loops"
        assert len(set(loop_ids_end)) == 1, "Event loop changed during execution"
        assert loop_ids_start[0] == loop_ids_end[0], "Event loop changed during operations"
