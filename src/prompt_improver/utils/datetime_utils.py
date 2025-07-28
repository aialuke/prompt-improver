"""Datetime utilities for consistent UTC handling across the application.

This module provides utilities to replace deprecated datetime.utcnow() calls
following Python 3.12+ best practices and 2025 recommendations.

Key functions:
- naive_utc_now(): Returns naive UTC datetime for database compatibility
- aware_utc_now(): Returns timezone-aware UTC datetime for services/logging
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
        # Already naive, assume UTC
        return dt
    # Convert to UTC and remove timezone
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
        # Naive, assume UTC
        return dt.replace(tzinfo=UTC)
    # Already aware, convert to UTC
    return dt.astimezone(UTC)