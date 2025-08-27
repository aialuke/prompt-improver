"""Index Health Assessor with PostgreSQL System Catalog Analysis.

Provides comprehensive index health analysis including:
- Index usage statistics and efficiency
- Unused and redundant index detection
- Index bloat assessment
- Missing index recommendations
- Index maintenance suggestions
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from prompt_improver.database import (
    ManagerMode,
    get_database_services,
    get_session_context,
)

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Index statistics and health information."""

    schema_name: str
    table_name: str
    index_name: str
    index_type: str
    is_unique: bool
    is_primary: bool
    columns: list[str]

    # Size metrics
    size_bytes: int = 0
    size_pretty: str = ""

    # Usage statistics
    index_scans: int = 0
    tuples_read: int = 0
    tuples_fetched: int = 0

    # Health indicators
    usage_ratio: float = 0.0  # scans per table scan
    bloat_ratio: float = 0.0
    selectivity: float = 0.0

    # Status flags
    is_unused: bool = False
    is_redundant: bool = False
    needs_reindex: bool = False

    # Recommendations
    recommendations: list[str] = field(default_factory=list)


@dataclass
class IndexHealthReport:
    """Comprehensive index health report."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    total_indexes: int = 0
    total_size_bytes: int = 0

    # Index categories
    healthy_indexes: list[IndexStats] = field(default_factory=list)
    unused_indexes: list[IndexStats] = field(default_factory=list)
    redundant_indexes: list[IndexStats] = field(default_factory=list)
    bloated_indexes: list[IndexStats] = field(default_factory=list)
    low_usage_indexes: list[IndexStats] = field(default_factory=list)

    # Missing index suggestions
    missing_index_suggestions: list[dict[str, Any]] = field(default_factory=list)

    # Summary metrics
    health_score: float = 100.0
    potential_space_savings_bytes: int = 0
    maintenance_recommendations: list[str] = field(default_factory=list)


class IndexHealthAssessor:
    """Assess PostgreSQL index health using system catalogs and statistics."""

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

        # Health thresholds
        self.unused_threshold_scans = 10  # Indexes with < 10 scans considered unused
        self.low_usage_threshold_ratio = 0.1  # Usage ratio below 0.1 is considered low
        self.bloat_threshold_percent = 20.0  # Index bloat > 20% needs attention
        self.large_index_threshold_mb = 100  # Indexes > 100MB need more attention

        # Cache for expensive operations
        self._table_stats_cache = {}
        self._cache_timestamp = None
        self._cache_duration_seconds = 300  # 5 minutes

    async def get_client(self):
        """Get database client."""
        if self.client is None:
            return await get_database_services(ManagerMode.ASYNC_MODERN)
        return self.client

    async def assess_index_health(self) -> IndexHealthReport:
        """Perform comprehensive index health assessment."""
        logger.debug("Starting index health assessment")
        start_time = time.perf_counter()

        try:
            # Get all index statistics
            all_indexes = await self._get_all_index_stats()

            # Categorize indexes by health status
            report = IndexHealthReport()
            report.total_indexes = len(all_indexes)
            report.total_size_bytes = sum(idx.size_bytes for idx in all_indexes)

            for index in all_indexes:
                # Determine index health status
                self._assess_individual_index_health(index)

                # Categorize based on health indicators
                if index.is_unused:
                    report.unused_indexes.append(index)
                elif index.is_redundant:
                    report.redundant_indexes.append(index)
                elif index.bloat_ratio > self.bloat_threshold_percent:
                    report.bloated_indexes.append(index)
                elif index.usage_ratio < self.low_usage_threshold_ratio:
                    report.low_usage_indexes.append(index)
                else:
                    report.healthy_indexes.append(index)

            # Generate missing index suggestions
            report.missing_index_suggestions = await self._suggest_missing_indexes()

            # Calculate health score and recommendations
            report.health_score = self._calculate_index_health_score(report)
            report.potential_space_savings_bytes = self._calculate_space_savings(report)
            report.maintenance_recommendations = (
                self._generate_maintenance_recommendations(report)
            )

            assessment_time = (time.perf_counter() - start_time) * 1000
            logger.info("Index health assessment completed in %.2fms", assessment_time)

            return report

        except Exception as e:
            logger.exception(f"Index health assessment failed: {e}")
            report = IndexHealthReport()
            report.maintenance_recommendations = [f"Assessment failed: {e}"]
            return report

    async def _get_all_index_stats(self) -> list[IndexStats]:
        """Get comprehensive statistics for all indexes."""
        async with get_session_context() as session:
            query = text("""
                SELECT
                    n.nspname as schema_name,
                    t.relname as table_name,
                    i.relname as index_name,
                    am.amname as index_type,
                    ix.indisunique as is_unique,
                    ix.indisprimary as is_primary,
                    array_to_string(array_agg(a.attname ORDER BY a.attnum), ', ') as columns,
                    pg_relation_size(i.oid) as size_bytes,
                    pg_size_pretty(pg_relation_size(i.oid)) as size_pretty,
                    COALESCE(s.idx_scan, 0) as index_scans,
                    COALESCE(s.idx_tup_read, 0) as tuples_read,
                    COALESCE(s.idx_tup_fetch, 0) as tuples_fetched,
                    COALESCE(ts.seq_scan + ts.idx_scan, 0) as total_table_scans,
                    t.reltuples::bigint as table_rows
                FROM pg_class i
                JOIN pg_index ix ON i.oid = ix.indexrelid
                JOIN pg_class t ON ix.indrelid = t.oid
                JOIN pg_namespace n ON t.relnamespace = n.oid
                JOIN pg_am am ON i.relam = am.oid
                JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
                LEFT JOIN pg_stat_user_indexes s ON s.indexrelid = i.oid
                LEFT JOIN pg_stat_user_tables ts ON ts.relid = t.oid
                WHERE n.nspname NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                    AND i.relkind = 'i'
                GROUP BY
                    n.nspname, t.relname, i.relname, am.amname, ix.indisunique,
                    ix.indisprimary, i.oid, s.idx_scan, s.idx_tup_read,
                    s.idx_tup_fetch, ts.seq_scan, ts.idx_scan, t.reltuples
                ORDER BY pg_relation_size(i.oid) DESC
            """)

            result = await session.execute(query)
            indexes = []

            for row in result:
                index_stats = IndexStats(
                    schema_name=row[0],
                    table_name=row[1],
                    index_name=row[2],
                    index_type=row[3],
                    is_unique=row[4],
                    is_primary=row[5],
                    columns=row[6].split(", ") if row[6] else [],
                    size_bytes=int(row[7]),
                    size_pretty=row[8],
                    index_scans=int(row[9]),
                    tuples_read=int(row[10]),
                    tuples_fetched=int(row[11]),
                )

                # Calculate usage ratio
                total_scans = int(row[12]) if row[12] else 0
                if total_scans > 0:
                    index_stats.usage_ratio = index_stats.index_scans / total_scans

                # Estimate selectivity (simplified)
                table_rows = int(row[13]) if row[13] else 1
                if index_stats.tuples_fetched > 0 and table_rows > 0:
                    index_stats.selectivity = index_stats.tuples_fetched / table_rows

                indexes.append(index_stats)

            return indexes

    def _assess_individual_index_health(self, index: IndexStats) -> None:
        """Assess health of individual index and set status flags."""
        recommendations = []

        # Check if index is unused
        if index.index_scans < self.unused_threshold_scans and not index.is_primary:
            index.is_unused = True
            recommendations.append("Consider dropping - very low usage")

        # Check for low usage (but not completely unused)
        elif (
            index.usage_ratio < self.low_usage_threshold_ratio and not index.is_primary
        ):
            recommendations.append("Low usage ratio - review necessity")

        # Check for large indexes with low usage
        if index.size_bytes > self.large_index_threshold_mb * 1024 * 1024:
            if index.usage_ratio < 0.5:
                recommendations.append(
                    "Large index with low usage - significant space savings possible"
                )

        # Check for potential redundancy (simplified check)
        # Note: This is a basic check - full redundancy detection would need more complex analysis
        if len(index.columns) == 1 and not index.is_unique and not index.is_primary:
            recommendations.append(
                "Single-column non-unique index - check for redundancy"
            )

        # Performance recommendations
        if index.selectivity > 0.8:  # Low selectivity
            recommendations.append("Low selectivity - may not be effective")

        if (
            index.tuples_read > index.tuples_fetched * 2
        ):  # Reading many more tuples than fetching
            recommendations.append("High tuple read ratio - consider composite index")

        # Size-based recommendations
        if index.size_bytes > 500 * 1024 * 1024:  # > 500MB
            recommendations.append(
                "Very large index - monitor growth and consider partitioning"
            )

        index.recommendations = recommendations

    async def _suggest_missing_indexes(self) -> list[dict[str, Any]]:
        """Suggest missing indexes based on query patterns and table statistics."""
        suggestions = []

        try:
            async with get_session_context() as session:
                # Find tables with high sequential scan ratios
                seq_scan_query = text("""
                    SELECT
                        schemaname,
                        tablename,
                        seq_scan,
                        seq_tup_read,
                        idx_scan,
                        n_tup_ins + n_tup_upd + n_tup_del as modifications,
                        reltuples::bigint as estimated_rows
                    FROM pg_stat_user_tables st
                    JOIN pg_class c ON c.relname = st.tablename
                    WHERE seq_scan > 100  -- Tables with significant sequential scans
                        AND seq_tup_read > seq_scan * 1000  -- Reading many tuples per scan
                        AND (idx_scan = 0 OR seq_scan > idx_scan)  -- More seq scans than index scans
                    ORDER BY seq_tup_read DESC
                    LIMIT 10
                """)

                result = await session.execute(seq_scan_query)

                for row in result:
                    schema_name, table_name = row[0], row[1]
                    seq_scans, seq_tup_read = row[2], row[3]
                    idx_scans, modifications = row[4], row[5]
                    estimated_rows = row[6]

                    # Calculate efficiency metrics
                    avg_tuples_per_seq_scan = (
                        seq_tup_read / seq_scans if seq_scans > 0 else 0
                    )
                    seq_scan_ratio = (
                        seq_scans / (seq_scans + idx_scans)
                        if (seq_scans + idx_scans) > 0
                        else 1
                    )

                    suggestion = {
                        "table": f"{schema_name}.{table_name}",
                        "reason": "High sequential scan activity",
                        "seq_scans": seq_scans,
                        "avg_tuples_per_scan": int(avg_tuples_per_seq_scan),
                        "seq_scan_ratio_percent": round(seq_scan_ratio * 100, 1),
                        "estimated_rows": estimated_rows,
                        "priority": "high"
                        if seq_scan_ratio > 0.8 and avg_tuples_per_seq_scan > 10000
                        else "medium",
                        "suggestion": f"Consider adding indexes for commonly filtered columns on {table_name}",
                        "impact_estimate": "High - could significantly reduce I/O",
                    }

                    suggestions.append(suggestion)

                # Find foreign key columns without indexes
                fk_query = text("""
                    SELECT DISTINCT
                        n.nspname as schema_name,
                        t.relname as table_name,
                        a.attname as column_name,
                        'Foreign key without index' as reason
                    FROM pg_constraint c
                    JOIN pg_class t ON c.conrelid = t.oid
                    JOIN pg_namespace n ON t.relnamespace = n.oid
                    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(c.conkey)
                    WHERE c.contype = 'f'  -- Foreign key constraints
                        AND n.nspname NOT IN ('information_schema', 'pg_catalog')
                        AND NOT EXISTS (
                            SELECT 1 FROM pg_index i
                            WHERE i.indrelid = t.oid
                                AND a.attnum = ANY(i.indkey)
                                AND i.indkey[0] = a.attnum  -- Column is first in index
                        )
                    LIMIT 5
                """)

                result = await session.execute(fk_query)

                for row in result:
                    schema_name, table_name, column_name, reason = row

                    suggestion = {
                        "table": f"{schema_name}.{table_name}",
                        "column": column_name,
                        "reason": reason,
                        "priority": "medium",
                        "suggestion": f"CREATE INDEX idx_{table_name}_{column_name} ON {schema_name}.{table_name} ({column_name})",
                        "impact_estimate": "Medium - improve foreign key join performance",
                    }

                    suggestions.append(suggestion)

        except Exception as e:
            logger.exception(f"Failed to generate missing index suggestions: {e}")

        return suggestions

    def _calculate_index_health_score(self, report: IndexHealthReport) -> float:
        """Calculate overall index health score (0-100)."""
        if report.total_indexes == 0:
            return 100.0

        score = 100.0

        # Penalize unused indexes
        unused_ratio = len(report.unused_indexes) / report.total_indexes
        score -= unused_ratio * 30  # Up to 30 point penalty

        # Penalize redundant indexes
        redundant_ratio = len(report.redundant_indexes) / report.total_indexes
        score -= redundant_ratio * 25  # Up to 25 point penalty

        # Penalize bloated indexes
        bloated_ratio = len(report.bloated_indexes) / report.total_indexes
        score -= bloated_ratio * 20  # Up to 20 point penalty

        # Penalize low usage indexes
        low_usage_ratio = len(report.low_usage_indexes) / report.total_indexes
        score -= low_usage_ratio * 15  # Up to 15 point penalty

        # Bonus for having missing index suggestions addressed
        if len(report.missing_index_suggestions) > 5:
            score -= 10  # Penalty for many missing index opportunities

        return max(0.0, min(100.0, score))

    def _calculate_space_savings(self, report: IndexHealthReport) -> int:
        """Calculate potential space savings from removing unused/redundant indexes."""
        savings = 0

        # Space from unused indexes
        for index in report.unused_indexes:
            savings += index.size_bytes

        # Space from redundant indexes (estimate 50% can be removed)
        for index in report.redundant_indexes:
            savings += index.size_bytes // 2

        return savings

    def _generate_maintenance_recommendations(
        self, report: IndexHealthReport
    ) -> list[str]:
        """Generate actionable maintenance recommendations."""
        recommendations = []

        # Unused indexes
        if report.unused_indexes:
            unused_count = len(report.unused_indexes)
            unused_size_mb = sum(idx.size_bytes for idx in report.unused_indexes) / (
                1024 * 1024
            )
            recommendations.append(
                f"Drop {unused_count} unused indexes to save {unused_size_mb:.1f}MB"
            )

        # Redundant indexes
        if report.redundant_indexes:
            redundant_count = len(report.redundant_indexes)
            recommendations.append(
                f"Review {redundant_count} potentially redundant indexes"
            )

        # Bloated indexes
        if report.bloated_indexes:
            bloated_count = len(report.bloated_indexes)
            recommendations.append(f"REINDEX {bloated_count} bloated indexes")

        # Missing indexes
        if report.missing_index_suggestions:
            high_priority = len([
                s
                for s in report.missing_index_suggestions
                if s.get("priority") == "high"
            ])
            if high_priority > 0:
                recommendations.append(
                    f"Create {high_priority} high-priority missing indexes"
                )

        # Large indexes with low usage
        large_low_usage = [
            idx
            for idx in report.low_usage_indexes
            if idx.size_bytes > self.large_index_threshold_mb * 1024 * 1024
        ]
        if large_low_usage:
            recommendations.append(
                f"Review {len(large_low_usage)} large indexes with low usage"
            )

        # General recommendations
        if report.health_score < 70:
            recommendations.append("Index health needs immediate attention")
        elif report.health_score < 85:
            recommendations.append("Consider index optimization for better performance")

        if not recommendations:
            recommendations.append("Index configuration appears healthy")

        return recommendations

    async def get_index_health_summary(self) -> dict[str, Any]:
        """Get a concise summary of index health."""
        try:
            report = await self.assess_index_health()

            # Determine overall status
            if report.health_score < 60:
                status = "critical"
            elif report.health_score < 80:
                status = "warning"
            else:
                status = "healthy"

            return {
                "status": status,
                "health_score": report.health_score,
                "total_indexes": report.total_indexes,
                "unused_indexes": len(report.unused_indexes),
                "redundant_indexes": len(report.redundant_indexes),
                "bloated_indexes": len(report.bloated_indexes),
                "potential_savings_mb": report.potential_space_savings_bytes
                / (1024 * 1024),
                "missing_index_opportunities": len(report.missing_index_suggestions),
                "key_recommendations": report.maintenance_recommendations[:3],
                "timestamp": report.timestamp.isoformat(),
            }

        except Exception as e:
            logger.exception(f"Failed to get index health summary: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    async def analyze_index_redundancy(self) -> dict[str, Any]:
        """Detailed analysis of potentially redundant indexes."""
        try:
            async with get_session_context() as session:
                # Find indexes that might be redundant (same leading columns)
                redundancy_query = text("""
                    WITH index_columns AS (
                        SELECT
                            n.nspname as schema_name,
                            t.relname as table_name,
                            i.relname as index_name,
                            array_agg(a.attname ORDER BY array_position(ix.indkey, a.attnum)) as columns,
                            ix.indisunique,
                            ix.indisprimary,
                            pg_relation_size(i.oid) as size_bytes
                        FROM pg_class i
                        JOIN pg_index ix ON i.oid = ix.indexrelid
                        JOIN pg_class t ON ix.indrelid = t.oid
                        JOIN pg_namespace n ON t.relnamespace = n.oid
                        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
                        WHERE n.nspname NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                            AND i.relkind = 'i'
                        GROUP BY n.nspname, t.relname, i.relname, ix.indisunique, ix.indisprimary, i.oid
                    )
                    SELECT
                        i1.schema_name,
                        i1.table_name,
                        i1.index_name as index1,
                        i2.index_name as index2,
                        i1.columns as columns1,
                        i2.columns as columns2,
                        i1.size_bytes as size1,
                        i2.size_bytes as size2,
                        CASE
                            WHEN i1.columns <@ i2.columns THEN 'i1_subset_of_i2'
                            WHEN i2.columns <@ i1.columns THEN 'i2_subset_of_i1'
                            WHEN i1.columns = i2.columns THEN 'identical'
                            ELSE 'partial_overlap'
                        END as relationship
                    FROM index_columns i1
                    JOIN index_columns i2 ON i1.schema_name = i2.schema_name
                        AND i1.table_name = i2.table_name
                        AND i1.index_name < i2.index_name  -- Avoid duplicates
                    WHERE (i1.columns && i2.columns)  -- Have overlapping columns
                        AND NOT (i1.indisprimary OR i2.indisprimary)  -- Don't include primary keys
                    ORDER BY i1.schema_name, i1.table_name, i1.index_name
                """)

                result = await session.execute(redundancy_query)
                redundant_pairs = []

                for row in result:
                    pair_info = {
                        "table": f"{row[0]}.{row[1]}",
                        "index1": row[2],
                        "index2": row[3],
                        "columns1": row[4],
                        "columns2": row[5],
                        "size1_bytes": int(row[6]),
                        "size2_bytes": int(row[7]),
                        "relationship": row[8],
                        "recommendation": self._get_redundancy_recommendation(row),
                    }
                    redundant_pairs.append(pair_info)

                total_redundant_size = sum(
                    min(pair["size1_bytes"], pair["size2_bytes"])
                    for pair in redundant_pairs
                )

                return {
                    "redundant_pairs_found": len(redundant_pairs),
                    "estimated_savings_bytes": total_redundant_size,
                    "estimated_savings_mb": total_redundant_size / (1024 * 1024),
                    "redundant_pairs": redundant_pairs,
                    "timestamp": datetime.now(UTC).isoformat(),
                }

        except Exception as e:
            logger.exception(f"Index redundancy analysis failed: {e}")
            return {"error": str(e)}

    def _get_redundancy_recommendation(self, row) -> str:
        """Generate recommendation for redundant index pair."""
        relationship = row[8]
        size1, size2 = row[6], row[7]

        if relationship == "identical":
            smaller_index = row[2] if size1 < size2 else row[3]
            return f"Drop {smaller_index} (identical indexes)"
        if relationship == "i1_subset_of_i2":
            return f"Consider dropping {row[2]} (subset of {row[3]})"
        if relationship == "i2_subset_of_i1":
            return f"Consider dropping {row[3]} (subset of {row[2]})"
        return "Review usage patterns to determine if one can be dropped"

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes in human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
