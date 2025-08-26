"""Table Bloat Detector with Advanced PostgreSQL Analytics

Provides comprehensive table bloat detection including:
- Table and index bloat estimation algorithms
- Dead tuple analysis
- VACUUM and ANALYZE recommendations
- Storage optimization suggestions
- Maintenance scheduling guidance
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

# psycopg_client removed in Phase 1 - using database_services instead
from prompt_improver.database import (
    ManagerMode,
    get_database_services,
    get_session_context,
)

logger = logging.getLogger(__name__)


@dataclass
class TableBloatInfo:
    """Table bloat information and statistics"""

    schema_name: str
    table_name: str

    # Size metrics
    table_size_bytes: int = 0
    table_size_pretty: str = ""
    index_size_bytes: int = 0
    total_size_bytes: int = 0

    # Row statistics
    estimated_rows: int = 0
    dead_tuples: int = 0
    live_tuples: int = 0

    # Bloat estimation
    bloat_bytes: int = 0
    bloat_ratio_percent: float = 0.0

    # VACUUM statistics
    last_vacuum: datetime | None = None
    last_autovacuum: datetime | None = None
    last_analyze: datetime | None = None
    last_autoanalyze: datetime | None = None
    vacuum_count: int = 0
    autovacuum_count: int = 0

    # Activity metrics
    seq_scan: int = 0
    seq_tup_read: int = 0
    idx_scan: int = 0
    n_tup_ins: int = 0
    n_tup_upd: int = 0
    n_tup_del: int = 0
    n_tup_hot_upd: int = 0

    # Health indicators
    needs_vacuum: bool = False
    needs_analyze: bool = False
    high_bloat: bool = False

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    maintenance_priority: str = "low"  # low, medium, high, critical


@dataclass
class IndexBloatInfo:
    """Index bloat information"""

    schema_name: str
    table_name: str
    index_name: str

    # Size metrics
    index_size_bytes: int = 0
    index_size_pretty: str = ""

    # Bloat estimation
    bloat_bytes: int = 0
    bloat_ratio_percent: float = 0.0

    # Index statistics
    leaf_fragmentation: float = 0.0
    avg_leaf_density: float = 0.0

    needs_reindex: bool = False
    recommendations: list[str] = field(default_factory=list)


@dataclass
class BloatDetectionReport:
    """Comprehensive bloat detection report"""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Table bloat summary
    total_tables_analyzed: int = 0
    bloated_tables: list[TableBloatInfo] = field(default_factory=list)
    tables_needing_vacuum: list[TableBloatInfo] = field(default_factory=list)
    tables_needing_analyze: list[TableBloatInfo] = field(default_factory=list)

    # Index bloat summary
    total_indexes_analyzed: int = 0
    bloated_indexes: list[IndexBloatInfo] = field(default_factory=list)
    indexes_needing_reindex: list[IndexBloatInfo] = field(default_factory=list)

    # Space savings potential
    total_bloat_bytes: int = 0
    potential_space_savings_bytes: int = 0

    # Maintenance recommendations
    immediate_actions: list[str] = field(default_factory=list)
    scheduled_maintenance: list[str] = field(default_factory=list)

    # Health score
    bloat_health_score: float = 100.0


class TableBloatDetector:
    """Detect and analyze table and index bloat using PostgreSQL statistics
    and estimation algorithms
    """

    def __init__(self, client: Any | None = None):
        self.client = client

        # Bloat thresholds
        self.table_bloat_threshold_percent = 20.0  # 20% bloat threshold
        self.index_bloat_threshold_percent = 30.0  # 30% index bloat threshold
        self.large_table_threshold_mb = 100  # 100MB+ tables get more attention
        self.dead_tuple_threshold_percent = 10.0  # 10% dead tuples threshold

        # Maintenance thresholds
        self.vacuum_threshold_days = 7  # Vacuum if older than 7 days
        self.analyze_threshold_days = 3  # Analyze if older than 3 days
        self.high_activity_threshold = 10000  # High activity = 10K+ modifications

        # Page and tuple size estimates (PostgreSQL defaults)
        self.page_size = 8192  # 8KB page size
        self.page_header_size = 24
        self.tuple_header_size = 23
        self.alignment = 8

    async def get_client(self):
        """Get database client"""
        if self.client is None:
            return await get_database_services(ManagerMode.ASYNC_MODERN)
        return self.client

    async def detect_table_bloat(self) -> dict[str, Any]:
        """Comprehensive table bloat detection and analysis"""
        logger.debug("Starting table bloat detection")
        start_time = time.perf_counter()

        try:
            # Get table bloat information
            table_bloat_info = await self._analyze_table_bloat()

            # Get index bloat information
            index_bloat_info = await self._analyze_index_bloat()

            # Build comprehensive report
            report = BloatDetectionReport()

            # Process table results
            report.total_tables_analyzed = len(table_bloat_info)
            for table_info in table_bloat_info:
                if table_info.bloat_ratio_percent > self.table_bloat_threshold_percent:
                    table_info.high_bloat = True
                    report.bloated_tables.append(table_info)

                if table_info.needs_vacuum:
                    report.tables_needing_vacuum.append(table_info)

                if table_info.needs_analyze:
                    report.tables_needing_analyze.append(table_info)

            # Process index results
            report.total_indexes_analyzed = len(index_bloat_info)
            for index_info in index_bloat_info:
                if index_info.bloat_ratio_percent > self.index_bloat_threshold_percent:
                    index_info.needs_reindex = True
                    report.bloated_indexes.append(index_info)
                    report.indexes_needing_reindex.append(index_info)

            # Calculate space savings
            report.total_bloat_bytes = sum(
                t.bloat_bytes for t in report.bloated_tables
            ) + sum(i.bloat_bytes for i in report.bloated_indexes)
            report.potential_space_savings_bytes = report.total_bloat_bytes

            # Generate maintenance recommendations
            report.immediate_actions = self._generate_immediate_actions(report)
            report.scheduled_maintenance = self._generate_scheduled_maintenance(report)

            # Calculate health score
            report.bloat_health_score = self._calculate_bloat_health_score(report)

            detection_time = (time.perf_counter() - start_time) * 1000
            logger.info("Table bloat detection completed in %.2fms", detection_time)

            return {
                "timestamp": report.timestamp.isoformat(),
                "summary": {
                    "total_tables_analyzed": report.total_tables_analyzed,
                    "bloated_tables_count": len(report.bloated_tables),
                    "tables_needing_vacuum": len(report.tables_needing_vacuum),
                    "tables_needing_analyze": len(report.tables_needing_analyze),
                    "total_indexes_analyzed": report.total_indexes_analyzed,
                    "bloated_indexes_count": len(report.bloated_indexes),
                    "total_bloat_mb": report.total_bloat_bytes / (1024 * 1024),
                    "potential_savings_mb": report.potential_space_savings_bytes
                    / (1024 * 1024),
                    "bloat_health_score": report.bloat_health_score,
                },
                "bloated_tables": [
                    {
                        "table": f"{t.schema_name}.{t.table_name}",
                        "size_mb": t.total_size_bytes / (1024 * 1024),
                        "bloat_mb": t.bloat_bytes / (1024 * 1024),
                        "bloat_percent": t.bloat_ratio_percent,
                        "dead_tuples": t.dead_tuples,
                        "priority": t.maintenance_priority,
                        "recommendations": t.recommendations,
                    }
                    for t in report.bloated_tables[:10]  # Top 10 most bloated
                ],
                "bloated_indexes": [
                    {
                        "index": f"{i.schema_name}.{i.index_name}",
                        "table": f"{i.schema_name}.{i.table_name}",
                        "size_mb": i.index_size_bytes / (1024 * 1024),
                        "bloat_mb": i.bloat_bytes / (1024 * 1024),
                        "bloat_percent": i.bloat_ratio_percent,
                        "recommendations": i.recommendations,
                    }
                    for i in report.bloated_indexes[:10]  # Top 10 most bloated
                ],
                "maintenance_actions": {
                    "immediate": report.immediate_actions,
                    "scheduled": report.scheduled_maintenance,
                },
                "detection_time_ms": round(detection_time, 2),
            }

        except Exception as e:
            logger.error(f"Table bloat detection failed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
                "detection_time_ms": (time.perf_counter() - start_time) * 1000,
            }

    async def _analyze_table_bloat(self) -> list[TableBloatInfo]:
        """Analyze table bloat using pg_stat_user_tables and size estimation"""
        async with get_session_context() as session:
            query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_total_relation_size(schemaname||'.'||tablename) as total_size,
                    pg_relation_size(schemaname||'.'||tablename) as table_size,
                    pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename) as index_size,
                    reltuples::bigint as estimated_rows,
                    n_dead_tup as dead_tuples,
                    n_live_tup as live_tuples,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    n_tup_ins,
                    n_tup_upd,
                    n_tup_del,
                    n_tup_hot_upd,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze,
                    vacuum_count,
                    autovacuum_count,
                    -- Additional metrics for bloat estimation
                    relpages,
                    relallvisible
                FROM pg_stat_user_tables st
                JOIN pg_class c ON c.relname = st.tablename
                WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """)

            result = await session.execute(query)
            table_bloat_list = []

            for row in result:
                table_info = TableBloatInfo(
                    schema_name=row[0],
                    table_name=row[1],
                    total_size_bytes=int(row[2]),
                    table_size_bytes=int(row[3]),
                    index_size_bytes=int(row[4]),
                    estimated_rows=int(row[5]),
                    dead_tuples=int(row[6]),
                    live_tuples=int(row[7]),
                    seq_scan=int(row[8]),
                    seq_tup_read=int(row[9]),
                    idx_scan=int(row[10] or 0),
                    n_tup_ins=int(row[11]),
                    n_tup_upd=int(row[12]),
                    n_tup_del=int(row[13]),
                    n_tup_hot_upd=int(row[14]),
                    last_vacuum=row[15],
                    last_autovacuum=row[16],
                    last_analyze=row[17],
                    last_autoanalyze=row[18],
                    vacuum_count=int(row[19]),
                    autovacuum_count=int(row[20]),
                )

                table_info.table_size_pretty = self._format_bytes(
                    table_info.table_size_bytes
                )

                # Estimate bloat using page and tuple analysis
                relpages = int(row[21]) if row[21] else 0
                relallvisible = int(row[22]) if row[22] else 0

                table_info.bloat_bytes, table_info.bloat_ratio_percent = (
                    self._estimate_table_bloat(table_info, relpages, relallvisible)
                )

                # Determine maintenance needs
                self._assess_maintenance_needs(table_info)

                # Generate recommendations
                table_info.recommendations = self._generate_table_recommendations(
                    table_info
                )

                # Set maintenance priority
                table_info.maintenance_priority = self._determine_maintenance_priority(
                    table_info
                )

                table_bloat_list.append(table_info)

            return table_bloat_list

    async def _analyze_index_bloat(self) -> list[IndexBloatInfo]:
        """Analyze index bloat using pg_stat_user_indexes and estimation algorithms"""
        async with get_session_context() as session:
            query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    pg_relation_size(schemaname||'.'||indexname) as index_size,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes
                WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                ORDER BY pg_relation_size(schemaname||'.'||indexname) DESC
            """)

            result = await session.execute(query)
            index_bloat_list = []

            for row in result:
                index_info = IndexBloatInfo(
                    schema_name=row[0],
                    table_name=row[1],
                    index_name=row[2],
                    index_size_bytes=int(row[3]),
                )

                index_info.index_size_pretty = self._format_bytes(
                    index_info.index_size_bytes
                )

                # Estimate index bloat (simplified approach)
                # In a full implementation, this would use more sophisticated algorithms
                # like the ones in pg_stat_tuple or pgstattuple extension
                index_info.bloat_bytes, index_info.bloat_ratio_percent = (
                    self._estimate_index_bloat(
                        index_info, int(row[4] or 0), int(row[5] or 0)
                    )
                )

                # Generate recommendations
                index_info.recommendations = self._generate_index_recommendations(
                    index_info
                )

                index_bloat_list.append(index_info)

            return index_bloat_list

    def _estimate_table_bloat(
        self, table_info: TableBloatInfo, relpages: int, relallvisible: int
    ) -> tuple[int, float]:
        """Estimate table bloat using page statistics and tuple counts

        This is a simplified estimation. For production use, consider using
        pgstattuple extension for more accurate bloat calculations.
        """
        try:
            total_tuples = table_info.live_tuples + table_info.dead_tuples
            if total_tuples == 0:
                return 0, 0.0

            # Estimate average tuple size (rough approximation)
            if table_info.table_size_bytes > 0 and table_info.live_tuples > 0:
                avg_tuple_size = table_info.table_size_bytes / table_info.live_tuples
            else:
                avg_tuple_size = 100  # Default estimate

            # Calculate expected size for live tuples
            expected_tuple_space = table_info.live_tuples * avg_tuple_size

            # Add page overhead
            expected_pages = max(
                1, int(expected_tuple_space / (self.page_size - self.page_header_size))
            )
            expected_size = expected_pages * self.page_size

            # Bloat is the difference between actual and expected size
            if expected_size > 0 and table_info.table_size_bytes > expected_size:
                bloat_bytes = table_info.table_size_bytes - expected_size
                bloat_ratio = (bloat_bytes / table_info.table_size_bytes) * 100
            else:
                bloat_bytes = 0
                bloat_ratio = 0.0

            # Alternative calculation using dead tuple ratio
            dead_tuple_ratio = (
                (table_info.dead_tuples / total_tuples) * 100 if total_tuples > 0 else 0
            )
            if dead_tuple_ratio > self.dead_tuple_threshold_percent:
                # Estimate bloat based on dead tuples
                dead_tuple_bloat_bytes = int(
                    table_info.table_size_bytes * (dead_tuple_ratio / 100)
                )
                if dead_tuple_bloat_bytes > bloat_bytes:
                    bloat_bytes = dead_tuple_bloat_bytes
                    bloat_ratio = dead_tuple_ratio

            return bloat_bytes, round(bloat_ratio, 2)

        except Exception as e:
            logger.debug(f"Bloat estimation failed for {table_info.table_name}: {e}")
            return 0, 0.0

    def _estimate_index_bloat(
        self, index_info: IndexBloatInfo, idx_scan: int, idx_tup_read: int
    ) -> tuple[int, float]:
        """Estimate index bloat (simplified approach)

        For production use, consider using pgstattuple extension or
        more sophisticated bloat estimation queries.
        """
        try:
            # This is a very simplified estimation
            # Real bloat detection would analyze index structure, fill factors, etc.

            if index_info.index_size_bytes < 1024 * 1024:  # Less than 1MB
                return 0, 0.0

            # Heuristic: indexes with low scan/read ratios might have bloat
            if idx_scan > 0 and idx_tup_read > 0:
                read_efficiency = idx_scan / idx_tup_read
                if read_efficiency < 0.1:  # Very low efficiency might indicate bloat
                    estimated_bloat_ratio = min(30.0, (0.1 - read_efficiency) * 1000)
                    bloat_bytes = int(
                        index_info.index_size_bytes * (estimated_bloat_ratio / 100)
                    )
                    return bloat_bytes, estimated_bloat_ratio

            # For large indexes with no usage, assume some bloat
            if (
                idx_scan == 0 and index_info.index_size_bytes > 10 * 1024 * 1024
            ):  # > 10MB
                return int(index_info.index_size_bytes * 0.1), 10.0  # Assume 10% bloat

            return 0, 0.0

        except Exception as e:
            logger.debug(
                f"Index bloat estimation failed for {index_info.index_name}: {e}"
            )
            return 0, 0.0

    def _assess_maintenance_needs(self, table_info: TableBloatInfo) -> None:
        """Assess whether table needs VACUUM or ANALYZE"""
        now = datetime.now(UTC)

        # Check VACUUM needs
        last_vacuum_time = None
        if table_info.last_autovacuum:
            last_vacuum_time = table_info.last_autovacuum.replace(tzinfo=UTC)
        if table_info.last_vacuum:
            manual_vacuum = table_info.last_vacuum.replace(tzinfo=UTC)
            if not last_vacuum_time or manual_vacuum > last_vacuum_time:
                last_vacuum_time = manual_vacuum

        # Needs vacuum if:
        # 1. Never vacuumed or vacuum is old
        # 2. High percentage of dead tuples
        # 3. High modification activity
        vacuum_age_days = (
            (now - last_vacuum_time).days if last_vacuum_time else float("inf")
        )
        dead_tuple_ratio = (
            (
                table_info.dead_tuples
                / (table_info.live_tuples + table_info.dead_tuples)
                * 100
            )
            if (table_info.live_tuples + table_info.dead_tuples) > 0
            else 0
        )
        total_modifications = (
            table_info.n_tup_ins + table_info.n_tup_upd + table_info.n_tup_del
        )

        table_info.needs_vacuum = (
            vacuum_age_days > self.vacuum_threshold_days
            or dead_tuple_ratio > self.dead_tuple_threshold_percent
            or total_modifications > self.high_activity_threshold
        )

        # Check ANALYZE needs
        last_analyze_time = None
        if table_info.last_autoanalyze:
            last_analyze_time = table_info.last_autoanalyze.replace(tzinfo=UTC)
        if table_info.last_analyze:
            manual_analyze = table_info.last_analyze.replace(tzinfo=UTC)
            if not last_analyze_time or manual_analyze > last_analyze_time:
                last_analyze_time = manual_analyze

        analyze_age_days = (
            (now - last_analyze_time).days if last_analyze_time else float("inf")
        )

        table_info.needs_analyze = (
            analyze_age_days > self.analyze_threshold_days
            or total_modifications > self.high_activity_threshold
        )

    def _generate_table_recommendations(self, table_info: TableBloatInfo) -> list[str]:
        """Generate specific recommendations for table maintenance"""
        recommendations = []

        # VACUUM recommendations
        if table_info.needs_vacuum:
            if table_info.bloat_ratio_percent > 50:
                recommendations.append("VACUUM FULL recommended - high bloat detected")
            else:
                recommendations.append("VACUUM recommended - remove dead tuples")

        # ANALYZE recommendations
        if table_info.needs_analyze:
            recommendations.append("ANALYZE recommended - update table statistics")

        # Bloat-specific recommendations
        if table_info.bloat_ratio_percent > self.table_bloat_threshold_percent:
            if (
                table_info.table_size_bytes
                > self.large_table_threshold_mb * 1024 * 1024
            ):
                recommendations.append(
                    "Consider scheduled VACUUM during maintenance window"
                )
            else:
                recommendations.append("Run VACUUM to reclaim space")

        # Activity-based recommendations
        total_modifications = (
            table_info.n_tup_ins + table_info.n_tup_upd + table_info.n_tup_del
        )
        if total_modifications > self.high_activity_threshold:
            recommendations.append(
                "High activity table - consider more frequent maintenance"
            )

        # Hot update recommendations
        if table_info.n_tup_upd > 0:
            hot_update_ratio = (table_info.n_tup_hot_upd / table_info.n_tup_upd) * 100
            if hot_update_ratio < 50:
                recommendations.append(
                    "Low HOT update ratio - consider increasing fillfactor"
                )

        return recommendations

    def _generate_index_recommendations(self, index_info: IndexBloatInfo) -> list[str]:
        """Generate specific recommendations for index maintenance"""
        recommendations = []

        if index_info.bloat_ratio_percent > self.index_bloat_threshold_percent:
            if index_info.index_size_bytes > 100 * 1024 * 1024:  # > 100MB
                recommendations.append("REINDEX recommended during maintenance window")
            else:
                recommendations.append("REINDEX recommended - high bloat detected")

        if index_info.index_size_bytes > 500 * 1024 * 1024:  # > 500MB
            recommendations.append("Monitor index growth - consider partitioning")

        return recommendations

    def _determine_maintenance_priority(self, table_info: TableBloatInfo) -> str:
        """Determine maintenance priority based on bloat and impact"""
        # Critical priority
        if (
            table_info.bloat_ratio_percent > 60
            or table_info.dead_tuples > 100000
            or (
                table_info.table_size_bytes > 1024 * 1024 * 1024
                and table_info.bloat_ratio_percent > 30
            )
        ):  # 1GB+ with 30%+ bloat
            return "critical"

        # High priority
        if table_info.bloat_ratio_percent > 40 or (
            table_info.needs_vacuum and table_info.table_size_bytes > 100 * 1024 * 1024
        ):  # 100MB+
            return "high"

        # Medium priority
        if (
            table_info.bloat_ratio_percent > self.table_bloat_threshold_percent
            or table_info.needs_vacuum
            or table_info.needs_analyze
        ):
            return "medium"

        return "low"

    def _generate_immediate_actions(self, report: BloatDetectionReport) -> list[str]:
        """Generate immediate action recommendations"""
        actions = []

        # Critical bloat tables
        critical_tables = [
            t for t in report.bloated_tables if t.maintenance_priority == "critical"
        ]
        if critical_tables:
            actions.append(
                f"URGENT: {len(critical_tables)} tables with critical bloat need immediate attention"
            )

        # High-impact vacuum operations
        high_priority_vacuum = [
            t
            for t in report.tables_needing_vacuum
            if t.maintenance_priority in ["critical", "high"]
        ]
        if high_priority_vacuum:
            table_names = [
                f"{t.schema_name}.{t.table_name}" for t in high_priority_vacuum[:3]
            ]
            actions.append(
                f"Run VACUUM on high-priority tables: {', '.join(table_names)}"
            )

        # Immediate reindex needs
        critical_indexes = [
            i for i in report.bloated_indexes if i.bloat_ratio_percent > 50
        ]
        if critical_indexes:
            actions.append(
                f"REINDEX {len(critical_indexes)} critically bloated indexes"
            )

        return actions

    def _generate_scheduled_maintenance(
        self, report: BloatDetectionReport
    ) -> list[str]:
        """Generate scheduled maintenance recommendations"""
        maintenance = []

        # Regular vacuum schedule
        if report.tables_needing_vacuum:
            maintenance.append(
                f"Schedule regular VACUUM for {len(report.tables_needing_vacuum)} tables"
            )

        # Analyze schedule
        if report.tables_needing_analyze:
            maintenance.append(
                f"Schedule ANALYZE for {len(report.tables_needing_analyze)} tables"
            )

        # Bulk reindex operations
        if len(report.bloated_indexes) > 5:
            maintenance.append(
                f"Plan maintenance window for bulk REINDEX of {len(report.bloated_indexes)} indexes"
            )

        # Autovacuum tuning
        high_activity_tables = [
            t
            for t in report.bloated_tables
            if (t.n_tup_ins + t.n_tup_upd + t.n_tup_del) > 50000
        ]
        if high_activity_tables:
            maintenance.append(
                f"Review autovacuum settings for {len(high_activity_tables)} high-activity tables"
            )

        return maintenance

    def _calculate_bloat_health_score(self, report: BloatDetectionReport) -> float:
        """Calculate overall bloat health score (0-100)"""
        if report.total_tables_analyzed == 0:
            return 100.0

        score = 100.0

        # Penalize bloated tables
        bloated_ratio = len(report.bloated_tables) / report.total_tables_analyzed
        score -= bloated_ratio * 40  # Up to 40 point penalty

        # Penalize tables needing vacuum
        vacuum_ratio = len(report.tables_needing_vacuum) / report.total_tables_analyzed
        score -= vacuum_ratio * 20  # Up to 20 point penalty

        # Penalize based on total bloat size
        if report.total_bloat_bytes > 1024 * 1024 * 1024:  # > 1GB total bloat
            score -= 20
        elif report.total_bloat_bytes > 100 * 1024 * 1024:  # > 100MB total bloat
            score -= 10

        # Penalize critical priority tables more heavily
        critical_tables = len([
            t for t in report.bloated_tables if t.maintenance_priority == "critical"
        ])
        if critical_tables > 0:
            score -= critical_tables * 10  # 10 points per critical table

        return max(0.0, min(100.0, score))

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes in human-readable format"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

    async def get_bloat_summary(self) -> dict[str, Any]:
        """Get a concise summary of bloat status"""
        try:
            bloat_data = await self.detect_table_bloat()

            if "error" in bloat_data:
                return {
                    "status": "error",
                    "error": bloat_data["error"],
                    "timestamp": bloat_data["timestamp"],
                }

            summary = bloat_data["summary"]

            # Determine overall status
            if summary["bloat_health_score"] < 60:
                status = "critical"
            elif summary["bloat_health_score"] < 80:
                status = "warning"
            else:
                status = "healthy"

            return {
                "status": status,
                "bloat_health_score": summary["bloat_health_score"],
                "bloated_tables_count": summary["bloated_tables_count"],
                "bloated_indexes_count": summary["bloated_indexes_count"],
                "total_bloat_mb": summary["total_bloat_mb"],
                "tables_needing_vacuum": summary["tables_needing_vacuum"],
                "tables_needing_analyze": summary["tables_needing_analyze"],
                "immediate_actions": bloat_data["maintenance_actions"]["immediate"][:2],
                "timestamp": bloat_data["timestamp"],
            }

        except Exception as e:
            logger.error(f"Failed to get bloat summary: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
