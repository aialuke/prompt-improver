"""Database Health Monitoring Infrastructure

Comprehensive PostgreSQL health monitoring with real metrics collection,
connection pool optimization, and performance analysis.
"""

from .database_health_monitor import DatabaseHealthMonitor, get_database_health_monitor
from .index_health_assessor import IndexHealthAssessor
from .table_bloat_detector import TableBloatDetector

# Note: ConnectionPoolMonitor and QueryPerformanceAnalyzer have been consolidated 
# into DatabaseHealthMonitor for improved performance and reduced code duplication.
# Use DatabaseHealthMonitor.get_comprehensive_health() for all database health metrics.

__all__ = [
    "DatabaseHealthMonitor",
    "get_database_health_monitor", 
    "IndexHealthAssessor",
    "TableBloatDetector",
]