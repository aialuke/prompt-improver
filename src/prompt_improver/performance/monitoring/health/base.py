"""Base health checker interface and result types for APES health monitoring.
PHASE 3: Health Check Consolidation - Composite Pattern Implementation
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

class HealthStatus(Enum):
    """Health check status levels"""

    HEALTHY = "healthy"
    WARNING = "warning"
    FAILED = "failed"

@dataclass
class HealthResult:
    """Standardized health check result"""

    status: HealthStatus
    component: str
    response_time_ms: float | None = None
    message: str = ""
    error: str | None = None
    details: dict[str, Any] | None = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class AggregatedHealthResult:
    """Aggregated health check results from multiple components"""

    overall_status: HealthStatus
    checks: dict[str, HealthResult]
    timestamp: datetime
    failed_checks: list[str] = None
    warning_checks: list[str] = None

    def __post_init__(self):
        if self.failed_checks is None:
            self.failed_checks = [
                name
                for name, result in self.checks.items()
                if result.status == HealthStatus.FAILED
            ]
        if self.warning_checks is None:
            self.warning_checks = [
                name
                for name, result in self.checks.items()
                if result.status == HealthStatus.WARNING
            ]

class HealthChecker(ABC):
    """Base interface for health check components"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def check(self) -> HealthResult:
        """Perform health check and return result"""

    def _time_check(self, func):
        """Utility to time a health check function"""

        async def wrapper():
            start_time = time.time()
            try:
                result = await func()
                end_time = time.time()
                if (
                    hasattr(result, "response_time_ms")
                    and result.response_time_ms is None
                ):
                    result.response_time_ms = (end_time - start_time) * 1000
                return result
            except Exception as e:
                end_time = time.time()
                return HealthResult(
                    status=HealthStatus.FAILED,
                    component=self.name,
                    response_time_ms=(end_time - start_time) * 1000,
                    error=str(e),
                    message=f"Health check failed: {e!s}",
                )

        return wrapper
