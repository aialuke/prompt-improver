"""Production datetime service implementation.

This service provides timezone-aware datetime operations following
2025 best practices for the ML Pipeline Orchestrator system.

Features:
- Async-compatible interface
- Timezone-aware operations by default
- Performance optimized for high-frequency calls
- Comprehensive error handling
- Logging for debugging and monitoring
"""
import logging
from datetime import UTC, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo
from prompt_improver.core.interfaces.datetime_service import DateTimeServiceProtocol

class DateTimeService(DateTimeServiceProtocol):
    """Production datetime service with timezone awareness.

    This service implements the DateTimeServiceProtocol and provides
    robust datetime operations for the ML Pipeline Orchestrator.
    """

    def __init__(self, logger: logging.Logger | None=None):
        """Initialize the datetime service.

        Args:
            logger: Optional logger for debugging and monitoring
        """
        self.logger = logger or logging.getLogger(__name__)
        self._call_count = 0
        self._utc_tz = UTC
        self.logger.debug('DateTimeService initialized')

    async def utc_now(self) -> datetime:
        """Get current UTC time as timezone-aware datetime.

        Returns:
            datetime: Current UTC time with timezone info
        """
        self._call_count += 1
        return datetime.now(self._utc_tz)

    async def naive_utc_now(self) -> datetime:
        """Get current UTC time as naive datetime.

        For database compatibility where naive UTC is expected.

        Returns:
            datetime: Current UTC time without timezone info
        """
        self._call_count += 1
        return datetime.now(self._utc_tz).replace(tzinfo=None)

    async def from_timestamp(self, timestamp: float, aware: bool=True) -> datetime:
        """Convert Unix timestamp to datetime.

        Args:
            timestamp: Unix timestamp (seconds since epoch)
            aware: If True, return timezone-aware datetime; if False, naive

        Returns:
            datetime: Converted datetime object

        Raises:
            ValueError: If timestamp is invalid
        """
        self._call_count += 1
        try:
            dt = datetime.fromtimestamp(timestamp, self._utc_tz)
            return dt if aware else dt.replace(tzinfo=None)
        except (ValueError, OSError) as e:
            self.logger.error('Invalid timestamp {timestamp}: %s', e)
            raise ValueError(f'Invalid timestamp: {timestamp}') from e

    async def to_timezone(self, dt: datetime, tz: ZoneInfo) -> datetime:
        """Convert datetime to specified timezone.

        Args:
            dt: Input datetime (naive or aware)
            tz: Target timezone

        Returns:
            datetime: Datetime converted to target timezone

        Raises:
            ValueError: If timezone conversion fails
        """
        self._call_count += 1
        try:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=self._utc_tz)
            return dt.astimezone(tz)
        except Exception as e:
            self.logger.error('Timezone conversion failed for {dt} to {tz}: %s', e)
            raise ValueError(f'Timezone conversion failed: {e}') from e

    async def format_iso(self, dt: datetime) -> str:
        """Format datetime as ISO 8601 string.

        Args:
            dt: Datetime to format

        Returns:
            str: ISO 8601 formatted string
        """
        self._call_count += 1
        try:
            return dt.isoformat()
        except Exception as e:
            self.logger.error('ISO formatting failed for {dt}: %s', e)
            raise ValueError(f'ISO formatting failed: {e}') from e

    async def ensure_aware_utc(self, dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware UTC.

        If the datetime is already aware, converts to UTC if necessary.
        If the datetime is naive, assumes it's UTC and adds timezone info.

        Args:
            dt: Input datetime (naive or aware)

        Returns:
            datetime: Timezone-aware UTC datetime
        """
        self._call_count += 1
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self._utc_tz)
        return dt.astimezone(self._utc_tz)

    async def ensure_naive_utc(self, dt: datetime) -> datetime:
        """Ensure datetime is naive UTC.

        If the datetime is already naive, returns it unchanged (assumes UTC).
        If the datetime is timezone-aware, converts to UTC and removes timezone info.

        Args:
            dt: Input datetime (naive or aware)

        Returns:
            datetime: Naive UTC datetime
        """
        self._call_count += 1
        if dt.tzinfo is None:
            return dt
        return dt.astimezone(self._utc_tz).replace(tzinfo=None)

    async def add_seconds(self, dt: datetime, seconds: int) -> datetime:
        """Add seconds to datetime while preserving timezone info.

        Args:
            dt: Input datetime
            seconds: Seconds to add (can be negative)

        Returns:
            datetime: Modified datetime
        """
        from datetime import timedelta
        return dt + timedelta(seconds=seconds)

    async def get_age_seconds(self, dt: datetime) -> float:
        """Get age of datetime in seconds from now.

        Args:
            dt: Input datetime

        Returns:
            float: Age in seconds (positive for past, negative for future)
        """
        now = await self.utc_now()
        if dt.tzinfo is None:
            dt = await self.ensure_aware_utc(dt)
        delta = now - dt
        return delta.total_seconds()

    async def is_recent(self, dt: datetime, max_age_seconds: int=300) -> bool:
        """Check if datetime is recent (within max_age_seconds).

        Args:
            dt: Datetime to check
            max_age_seconds: Maximum age in seconds (default: 5 minutes)

        Returns:
            bool: True if datetime is recent
        """
        age = await self.get_age_seconds(dt)
        return 0 <= age <= max_age_seconds

    def get_call_count(self) -> int:
        """Get total number of service calls for monitoring.

        Returns:
            int: Total call count
        """
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset call counter for testing/monitoring."""
        self._call_count = 0
        self.logger.debug('DateTimeService call count reset')

    async def health_check(self) -> dict:
        """Perform health check of the datetime service.

        Returns:
            dict: Health check results
        """
        try:
            now = await self.utc_now()
            naive_now = await self.naive_utc_now()
            iso_format = await self.format_iso(now)
            return {'status': 'healthy', 'call_count': self._call_count, 'current_utc': iso_format, 'service_type': 'DateTimeService'}
        except Exception as e:
            self.logger.error('DateTimeService health check failed: %s', e)
            return {'status': 'unhealthy', 'error': str(e), 'service_type': 'DateTimeService'}
