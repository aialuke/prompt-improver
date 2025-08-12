"""DateTime service interface for dependency injection.

This interface defines the contract for datetime services, enabling
dependency injection and improved testability across the ML Pipeline
Orchestrator system.

Following 2025 best practices:
- Protocol-based interfaces for duck typing
- Async-compatible design
- Timezone-aware operations by default
- Easy mocking for testing
"""

from datetime import UTC, datetime, timezone
from typing import Optional, Protocol
from zoneinfo import ZoneInfo


class DateTimeServiceProtocol(Protocol):
    """Protocol defining the datetime service interface.

    This protocol enables dependency injection while maintaining
    type safety and allowing for easy testing with mock implementations.
    """

    async def utc_now(self) -> datetime:
        """Get current UTC time as timezone-aware datetime.

        Returns:
            datetime: Current UTC time with timezone info
        """
        ...

    async def aware_utc_now(self) -> datetime:
        """Get current UTC time as timezone-aware datetime.

        Returns:
            datetime: Current UTC time with timezone info
        """
        ...

    async def naive_utc_now(self) -> datetime:
        """Get current UTC time as naive datetime.

        For database compatibility where naive UTC is expected.

        Returns:
            datetime: Current UTC time without timezone info
        """
        ...

    async def from_timestamp(self, timestamp: float, aware: bool = True) -> datetime:
        """Convert Unix timestamp to datetime.

        Args:
            timestamp: Unix timestamp (seconds since epoch)
            aware: If True, return timezone-aware datetime; if False, naive

        Returns:
            datetime: Converted datetime object
        """
        ...

    async def to_timezone(self, dt: datetime, tz: ZoneInfo) -> datetime:
        """Convert datetime to specified timezone.

        Args:
            dt: Input datetime (naive or aware)
            tz: Target timezone

        Returns:
            datetime: Datetime converted to target timezone
        """
        ...

    async def format_iso(self, dt: datetime) -> str:
        """Format datetime as ISO 8601 string.

        Args:
            dt: Datetime to format

        Returns:
            str: ISO 8601 formatted string
        """
        ...

    async def ensure_aware_utc(self, dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware UTC.

        Args:
            dt: Input datetime (naive or aware)

        Returns:
            datetime: Timezone-aware UTC datetime
        """
        ...

    async def ensure_naive_utc(self, dt: datetime) -> datetime:
        """Ensure datetime is naive UTC.

        Args:
            dt: Input datetime (naive or aware)

        Returns:
            datetime: Naive UTC datetime
        """
        ...


class MockDateTimeService:
    """Mock implementation for testing.

    Provides a controllable datetime service for unit testing
    that implements the DateTimeServiceProtocol.
    """

    def __init__(self, fixed_time: datetime | None = None):
        """Initialize mock service.

        Args:
            fixed_time: Fixed time to return, or None for real time
        """
        self.fixed_time = fixed_time
        self.call_count = 0
        self.method_calls = []

    async def utc_now(self) -> datetime:
        """Mock implementation of utc_now."""
        self.call_count += 1
        self.method_calls.append("utc_now")
        if self.fixed_time:
            return self.fixed_time.replace(tzinfo=UTC)
        return datetime.now(UTC)

    async def aware_utc_now(self) -> datetime:
        """Mock implementation of aware_utc_now."""
        return await self.utc_now()

    async def naive_utc_now(self) -> datetime:
        """Mock implementation of naive_utc_now."""
        self.call_count += 1
        self.method_calls.append("naive_utc_now")
        if self.fixed_time:
            return self.fixed_time.replace(tzinfo=None)
        return datetime.now(UTC).replace(tzinfo=None)

    async def from_timestamp(self, timestamp: float, aware: bool = True) -> datetime:
        """Mock implementation of from_timestamp."""
        self.call_count += 1
        self.method_calls.append("from_timestamp")
        dt = datetime.fromtimestamp(timestamp, UTC)
        return dt if aware else dt.replace(tzinfo=None)

    async def to_timezone(self, dt: datetime, tz: ZoneInfo) -> datetime:
        """Mock implementation of to_timezone."""
        self.call_count += 1
        self.method_calls.append("to_timezone")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(tz)

    async def format_iso(self, dt: datetime) -> str:
        """Mock implementation of format_iso."""
        self.call_count += 1
        self.method_calls.append("format_iso")
        return dt.isoformat()

    async def ensure_aware_utc(self, dt: datetime) -> datetime:
        """Mock implementation of ensure_aware_utc."""
        self.call_count += 1
        self.method_calls.append("ensure_aware_utc")
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    async def ensure_naive_utc(self, dt: datetime) -> datetime:
        """Mock implementation of ensure_naive_utc."""
        self.call_count += 1
        self.method_calls.append("ensure_naive_utc")
        if dt.tzinfo is None:
            return dt
        return dt.astimezone(UTC).replace(tzinfo=None)

    def reset_counters(self):
        """Reset call counters for testing."""
        self.call_count = 0
        self.method_calls = []
