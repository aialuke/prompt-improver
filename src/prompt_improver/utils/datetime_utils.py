"""Comprehensive datetime utilities for UTC handling and formatting.

This consolidated module provides utilities to replace deprecated datetime.utcnow() calls
following Python 3.12+ best practices and comprehensive formatting functions for consistent
string representation across the application.

UTC Handling (Python 3.12+ best practices):
- naive_utc_now(): Returns naive UTC datetime for database compatibility  
- aware_utc_now(): Returns timezone-aware UTC datetime for services/logging
- ensure_naive_utc() / ensure_aware_utc(): Timezone conversion utilities

Formatting Functions (consolidated from common.datetime_utils):
- format_timestamp(): ISO timestamp format
- format_display_date(): Human-readable display format
- format_log_timestamp(): Logging with millisecond precision
- format_compact_timestamp(): Compact format for file naming
- format_date_only() / format_time_only(): Date or time only
- format_duration(): Human-readable duration strings
- format_utc_timestamp(): UTC timestamp with Z suffix

This module replaces both deprecated datetime.utcnow() usage and scattered
strftime() formatting throughout the codebase for improved maintainability.
"""

from datetime import UTC, datetime


def naive_utc_now() -> datetime:
    """Get current UTC time as a naive datetime object.

    This function provides identical behavior to the deprecated datetime.utcnow()
    and is intended for use with database models and systems that expect naive
    UTC timestamps.

    Returns:
        datetime: Current UTC time as naive datetime (no timezone info)

    Example:
        >>> dt = naive_utc_now()
        >>> dt.tzinfo is None
        True
    """
    return datetime.now(UTC).replace(tzinfo=None)


def aware_utc_now() -> datetime:
    """Get current UTC time as a timezone-aware datetime object.

    This function returns a timezone-aware datetime in UTC, which is the
    recommended approach for services, logging, and API responses that need
    proper timezone handling.

    Returns:
        datetime: Current UTC time with timezone info attached

    Example:
        >>> dt = aware_utc_now()
        >>> dt.tzinfo is not None
        True
        >>> str(dt.tzinfo)
        'UTC'
    """
    return datetime.now(UTC)


def naive_utc_from_timestamp(timestamp: float) -> datetime:
    """Convert Unix timestamp to naive UTC datetime.

    This function provides identical behavior to the deprecated
    datetime.utcfromtimestamp() and is intended for database compatibility.

    Args:
        timestamp: Unix timestamp (seconds since epoch)

    Returns:
        datetime: UTC datetime as naive object (no timezone info)

    Example:
        >>> dt = naive_utc_from_timestamp(0)
        >>> dt.tzinfo is None
        True
    """
    return datetime.fromtimestamp(timestamp, UTC).replace(tzinfo=None)


def aware_utc_from_timestamp(timestamp: float) -> datetime:
    """Convert Unix timestamp to timezone-aware UTC datetime.

    This function returns a timezone-aware datetime in UTC from a Unix timestamp,
    which is the recommended approach for services and API responses.

    Args:
        timestamp: Unix timestamp (seconds since epoch)

    Returns:
        datetime: UTC datetime with timezone info attached

    Example:
        >>> dt = aware_utc_from_timestamp(0)
        >>> dt.tzinfo is not None
        True
    """
    return datetime.fromtimestamp(timestamp, UTC)


def ensure_naive_utc(dt: datetime) -> datetime:
    """Ensure datetime is naive UTC.

    If the datetime is already naive, returns it unchanged (assumes UTC).
    If the datetime is timezone-aware, converts to UTC and removes timezone info.

    Args:
        dt: Input datetime (naive or aware)

    Returns:
        datetime: Naive UTC datetime

    Example:
        >>> from datetime import timezone
        >>> aware_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        >>> naive_dt = ensure_naive_utc(aware_dt)
        >>> naive_dt.tzinfo is None
        True
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(UTC).replace(tzinfo=None)


def ensure_aware_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware UTC.

    If the datetime is already aware, converts to UTC if necessary.
    If the datetime is naive, assumes it's UTC and adds timezone info.

    Args:
        dt: Input datetime (naive or aware)

    Returns:
        datetime: Timezone-aware UTC datetime

    Example:
        >>> naive_dt = datetime(2023, 1, 1, 12, 0, 0)
        >>> aware_dt = ensure_aware_utc(naive_dt)
        >>> aware_dt.tzinfo is not None
        True
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


# ============================================================================
# DATETIME FORMATTING UTILITIES (from common.datetime_utils)
# ============================================================================

def format_timestamp(dt: datetime) -> str:
    """Format datetime as ISO timestamp.

    Args:
        dt: Datetime to format

    Returns:
        ISO formatted timestamp string (YYYY-MM-DDTHH:MM:SS.ffffff)

    Example:
        >>> dt = datetime(2023, 12, 25, 14, 30, 45, 123456)
        >>> format_timestamp(dt)
        '2023-12-25T14:30:45.123456'
    """
    return dt.isoformat()


def format_display_date(dt: datetime) -> str:
    """Format datetime for user-friendly display.

    Args:
        dt: Datetime to format

    Returns:
        Human-readable date string (YYYY-MM-DD HH:MM:SS)

    Example:
        >>> dt = datetime(2023, 12, 25, 14, 30, 45)
        >>> format_display_date(dt)
        '2023-12-25 14:30:45'
    """
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"


def format_log_timestamp(dt: datetime) -> str:
    """Format datetime for logging with millisecond precision.

    Args:
        dt: Datetime to format

    Returns:
        Log timestamp string with milliseconds (YYYY-MM-DD HH:MM:SS.mmm)

    Example:
        >>> dt = datetime(2023, 12, 25, 14, 30, 45, 123456)
        >>> format_log_timestamp(dt)
        '2023-12-25 14:30:45.123'
    """
    milliseconds = dt.microsecond // 1000
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{milliseconds:03d}"


def format_compact_timestamp(dt: datetime) -> str:
    """Format datetime as compact timestamp without separators.

    Args:
        dt: Datetime to format

    Returns:
        Compact timestamp string (YYYYMMDD_HHMMSS)

    Example:
        >>> dt = datetime(2023, 12, 25, 14, 30, 45)
        >>> format_compact_timestamp(dt)
        '20231225_143045'
    """
    return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"


def format_date_only(dt: datetime) -> str:
    """Format datetime as date only.

    Args:
        dt: Datetime to format

    Returns:
        Date string (YYYY-MM-DD)

    Example:
        >>> dt = datetime(2023, 12, 25, 14, 30, 45)
        >>> format_date_only(dt)
        '2023-12-25'
    """
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"


def format_time_only(dt: datetime) -> str:
    """Format datetime as time only.

    Args:
        dt: Datetime to format

    Returns:
        Time string (HH:MM:SS)

    Example:
        >>> dt = datetime(2023, 12, 25, 14, 30, 45)
        >>> format_time_only(dt)
        '14:30:45'
    """
    return f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"


def format_duration(seconds: float) -> str:
    """Format duration in seconds as human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string

    Example:
        >>> format_duration(3661.5)
        '1h 1m 1.5s'
        >>> format_duration(75.234)
        '1m 15.23s'
        >>> format_duration(0.123)
        '0.12s'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.2f}s"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60

    return f"{hours}h {remaining_minutes}m {remaining_seconds:.2f}s"


def format_utc_timestamp(dt: datetime) -> str:
    """Format datetime as UTC timestamp with Z suffix.

    Args:
        dt: Datetime to format (assumed to be UTC)

    Returns:
        UTC timestamp string (YYYY-MM-DDTHH:MM:SS.fffffZ)

    Example:
        >>> dt = datetime(2023, 12, 25, 14, 30, 45, 123456)
        >>> format_utc_timestamp(dt)
        '2023-12-25T14:30:45.123456Z'
    """
    return f"{dt.isoformat()}Z"
