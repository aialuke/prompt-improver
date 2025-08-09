"""Base health checker interface and result types for APES health monitoring.
PHASE 3: Health Check Consolidation - Composite Pattern Implementation
"""
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any
from sqlmodel import Field, SQLModel

class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = 'healthy'
    WARNING = 'warning'
    FAILED = 'failed'

class HealthResult(SQLModel):
    """Standardized health check result"""
    status: HealthStatus
    component: str = Field(min_length=1, max_length=255)
    response_time_ms: float | None = Field(default=None, ge=0.0, le=60000.0)
    message: str = Field(default='', max_length=1000)
    error: str | None = Field(default=None, max_length=1000)
    details: dict[str, Any] | None = Field(default=None)
    timestamp: datetime = Field(default=None)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AggregatedHealthResult(SQLModel):
    """Aggregated health check results from multiple components"""
    overall_status: HealthStatus
    checks: dict[str, HealthResult]
    timestamp: datetime
    failed_checks: list[str] = Field(default=None)
    warning_checks: list[str] = Field(default=None)

    def __post_init__(self):
        if self.failed_checks is None:
            self.failed_checks = [name for name, result in self.checks.items() if result.status == HealthStatus.FAILED]
        if self.warning_checks is None:
            self.warning_checks = [name for name, result in self.checks.items() if result.status == HealthStatus.WARNING]

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
                if hasattr(result, 'response_time_ms') and result.response_time_ms is None:
                    result.response_time_ms = (end_time - start_time) * 1000
                return result
            except Exception as e:
                end_time = time.time()
                return HealthResult(status=HealthStatus.FAILED, component=self.name, response_time_ms=(end_time - start_time) * 1000, error=str(e), message=f'Health check failed: {e!s}')
        return wrapper
