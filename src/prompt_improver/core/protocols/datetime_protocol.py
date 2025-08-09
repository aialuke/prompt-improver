"""Protocol definitions for datetime utilities.

Provides type-safe interface contracts for datetime operations,
enabling dependency inversion and improved testability.
"""
from datetime import datetime, timezone
from typing import Protocol

class DateTimeServiceProtocol(Protocol):
    """Protocol for datetime service operations"""

    def aware_utc_now(self) -> datetime:
        """Get current UTC datetime with timezone awareness"""
        ...

    def naive_utc_now(self) -> datetime:
        """Get current UTC datetime without timezone awareness"""
        ...

    def to_aware_utc(self, dt: datetime) -> datetime:
        """Convert datetime to timezone-aware UTC"""
        ...

    def to_naive_utc(self, dt: datetime) -> datetime:
        """Convert datetime to naive UTC"""
        ...

    def format_iso(self, dt: datetime) -> str:
        """Format datetime as ISO string"""
        ...

    def parse_iso(self, iso_string: str) -> datetime:
        """Parse ISO string to datetime"""
        ...

class TimeZoneServiceProtocol(Protocol):
    """Protocol for timezone operations"""

    def get_utc_timezone(self) -> timezone:
        """Get UTC timezone object"""
        ...

    def convert_timezone(self, dt: datetime, target_tz: timezone) -> datetime:
        """Convert datetime to target timezone"""
        ...

    def is_aware(self, dt: datetime) -> bool:
        """Check if datetime is timezone-aware"""
        ...

class DateTimeUtilsProtocol(DateTimeServiceProtocol, TimeZoneServiceProtocol):
    """Combined protocol for all datetime utilities"""
