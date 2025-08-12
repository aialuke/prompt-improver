"""Common datetime formatting utilities.

This module provides standardized datetime formatting functions to replace
scattered strftime() usage throughout the codebase. This centralization
improves maintainability and ensures consistent formatting patterns.
"""

from datetime import datetime
from typing import Optional


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
