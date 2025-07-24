"""Example usage of AsyncErrorBoundary and structured logging features.

This module demonstrates how to use the new centralized error handling
and structured logging features in error_handlers.py.
"""

import asyncio
import logging
from typing import Any, Dict

from .error_handlers import (
    AsyncContextLogger,
    AsyncErrorBoundary,
    PIIRedactionFilter,
    async_error_boundary,
    configure_structured_logging,
)

async def example_background_task(
    task_name: str, should_fail: bool = False
) -> dict[str, Any]:
    """Example background task that might succeed or fail."""
    await asyncio.sleep(0.1)  # Simulate some work

    if should_fail:
        raise ValueError(f"Task {task_name} failed as requested")

    return {
        "task": task_name,
        "status": "completed",
        "result": f"Success for {task_name}",
    }

async def example_with_class_based_boundary():
    """Example using the AsyncErrorBoundary class directly."""
    print("\n=== Class-based AsyncErrorBoundary Example ===")

    # Configure logging
    logger = configure_structured_logging(
        logger_name="example_app",
        enable_pii_redaction=True,
        json_format=False,  # Use readable format for demo
    )

    # Success case
    async with AsyncErrorBoundary(
        operation_name="successful_background_task", logger=logger, reraise=True
    ) as boundary:
        result = await example_background_task("success_task", should_fail=False)
        logger.info("Task completed", task_result=result)
        print(f"✅ Success: {result}")

    # Failure case with reraise=False
    async with AsyncErrorBoundary(
        operation_name="failing_background_task",
        logger=logger,
        reraise=False,  # Don't reraise exceptions
        fallback_result={"status": "fallback", "error": "handled"},
    ) as boundary:
        try:
            result = await example_background_task("fail_task", should_fail=True)
            print(f"✅ Task result: {result}")
        except Exception as e:
            print("❌ This shouldn't be reached because reraise=False")

        # Check if exception occurred
        exception_info = boundary.get_exception_info()
        if exception_info:
            print(f"🔍 Exception was handled: {exception_info['type']}")

async def example_with_context_manager_factory():
    """Example using the async_error_boundary context manager factory."""
    print("\n=== Context Manager Factory Example ===")

    # Configure logging
    logger = configure_structured_logging(
        logger_name="factory_example", enable_pii_redaction=True, json_format=False
    )

    # Success case
    async with async_error_boundary(
        operation_name="api_call_simulation",
        logger=logger,
        reraise=False,
        fallback_result={"error": "api_unavailable", "fallback": True},
    ) as boundary:
        result = await example_background_task("api_task", should_fail=False)
        logger.info("API call successful", api_result=result)
        print(f"✅ API Success: {result}")

async def example_timeout_handling():
    """Example showing timeout handling."""
    print("\n=== Timeout Handling Example ===")

    logger = configure_structured_logging(
        logger_name="timeout_example", enable_pii_redaction=True, json_format=False
    )

    async def slow_task():
        await asyncio.sleep(2)  # Takes longer than timeout
        return {"status": "completed"}

    async with AsyncErrorBoundary(
        operation_name="slow_operation",
        logger=logger,
        reraise=False,
        timeout=1.0,  # 1 second timeout
    ) as boundary:
        try:
            # This will timeout
            result = await asyncio.wait_for(slow_task(), timeout=boundary.timeout)
            print(f"✅ Task completed: {result}")
        except TimeoutError:
            print("⏰ Task timed out as expected")

def example_pii_redaction():
    """Example showing PII redaction in logs."""
    print("\n=== PII Redaction Example ===")

    # Configure logging with PII redaction
    logger = configure_structured_logging(
        logger_name="pii_example", enable_pii_redaction=True, json_format=False
    )

    # Log some data that contains PII
    user_data = {
        "user_id": "12345",
        "email": "user@example.com",
        "phone": "555-123-4567",
        "credit_card": "4111-1111-1111-1111",
        "name": "John Doe",
        "password": "secret123",
    }

    print("Original data:", user_data)
    print("Logging this data (PII should be redacted):")

    # This will automatically redact PII
    logger.info("User registration", user_data=user_data)

    # Also test string-based PII redaction
    logger.info("Processing payment for user@example.com with card 4111-1111-1111-1111")

async def example_correlation_id_tracking():
    """Example showing correlation ID tracking across operations."""
    print("\n=== Correlation ID Tracking Example ===")

    logger = configure_structured_logging(
        logger_name="correlation_example", enable_pii_redaction=True, json_format=False
    )

    # Set up correlation context
    logger.set_context(user_id="user123", session_id="sess456")

    async with AsyncErrorBoundary(
        operation_name="user_workflow", logger=logger, reraise=False
    ) as boundary:
        logger.info("Starting user workflow")

        # Simulate multiple operations with same correlation ID
        await example_background_task("step1", should_fail=False)
        logger.info("Completed step 1")

        await example_background_task("step2", should_fail=False)
        logger.info("Completed step 2")

        logger.info("User workflow completed successfully")

async def main():
    """Run all examples."""
    print("🚀 AsyncErrorBoundary and Structured Logging Examples")
    print("=" * 60)

    await example_with_class_based_boundary()
    await example_with_context_manager_factory()
    await example_timeout_handling()
    example_pii_redaction()
    await example_correlation_id_tracking()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")

if __name__ == "__main__":
    asyncio.run(main())
