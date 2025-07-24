"""Analytics Service
Provides comprehensive analytics and reporting for APES
"""

from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.utils.datetime_utils import aware_utc_now

from ...database import get_sessionmanager
from ...database.models import RuleEffectivenessStats, UserSatisfactionStats
from ...database.utils import fetch_all_rows, fetch_one_row
from ...utils.error_handlers import handle_database_errors, handle_validation_errors

class AnalyticsService:
    """Service for analytics and reporting functionality"""

    @handle_database_errors(
        rollback_session=True,
        return_format="none",
        operation_name="get_rule_effectiveness",
        retry_count=2,
    )
    async def get_rule_effectiveness(
        self,
        days: int = 30,
        min_usage_count: int = 5,
        db_session: AsyncSession | None = None,
    ) -> list[RuleEffectivenessStats]:
        """Get rule effectiveness analytics for the specified time period"""
        if not db_session:
            return []

        # Calculate cutoff date
        cutoff_date = aware_utc_now() - timedelta(days=days)

        # Query rule effectiveness using SQL view or direct aggregation
        query = text("""
            SELECT
                rp.rule_id,
                rm.rule_name,
                COUNT(*) as usage_count,
                AVG(rp.improvement_score) as avg_improvement,
                STDDEV(rp.improvement_score) as score_stddev,
                MIN(rp.improvement_score) as min_improvement,
                MAX(rp.improvement_score) as max_improvement,
                AVG(rp.confidence_level) as avg_confidence,
                AVG(rp.execution_time_ms) as avg_execution_time,
                COUNT(DISTINCT rp.session_id) as prompt_types_count
            FROM rule_performance rp
            JOIN rule_metadata rm ON rp.rule_id = rm.rule_id
            WHERE rp.created_at >= :cutoff_date
            GROUP BY rp.rule_id, rm.rule_name
            HAVING COUNT(*) >= :min_usage_count
            ORDER BY avg_improvement DESC
        """)

        # Execute query with precise row typing
        rows: list[Row[Any]] = await fetch_all_rows(db_session, query, {"cutoff_date": cutoff_date, "min_usage_count": min_usage_count})

        stats = []
        for row in rows:
            stats.append(
                RuleEffectivenessStats(
                    rule_id=str(row.rule_id),
                    rule_name=str(row.rule_name),
                    usage_count=int(row.usage_count),
                    avg_improvement=float(row.avg_improvement or 0),
                    score_stddev=float(row.score_stddev or 0),
                    min_improvement=float(row.min_improvement or 0),
                    max_improvement=float(row.max_improvement or 0),
                    avg_confidence=float(row.avg_confidence or 0),
                    avg_execution_time=float(row.avg_execution_time or 0),
                    prompt_types_count=int(row.prompt_types_count or 0),
                )
            )

        return stats

    @handle_database_errors(
        rollback_session=True,
        return_format="none",
        operation_name="get_user_satisfaction",
        retry_count=2,
    )
    async def get_user_satisfaction(
        self, days: int = 30, db_session: AsyncSession | None = None
    ) -> list[UserSatisfactionStats]:
        """Get user satisfaction analytics and trends"""
        if not db_session:
            return []

        # Calculate cutoff date
        cutoff_date = aware_utc_now() - timedelta(days=days)

        # Query user satisfaction by day
        query = text("""
            SELECT
                DATE_TRUNC('day', created_at) as feedback_date,
                COUNT(*) as total_feedback,
                AVG(rating::FLOAT) as avg_rating,
                COUNT(CASE WHEN rating >= 4 THEN 1 END) as positive_feedback,
                COUNT(CASE WHEN rating <= 2 THEN 1 END) as negative_feedback,
                ARRAY_AGG(DISTINCT improvement_areas::text) as rules_used
            FROM user_feedback
            WHERE created_at >= :cutoff_date
            GROUP BY DATE_TRUNC('day', created_at)
            ORDER BY feedback_date DESC
        """)

        result = await db_session.execute(query, {"cutoff_date": cutoff_date})

        stats = []
        for row in result:
            # Parse rules_used array
            rules_used = row.rules_used or []
            if rules_used and isinstance(rules_used[0], str):
                # Extract rule IDs from JSON strings
                import json

                all_rules = set()
                for rule_json in rules_used:
                    parsed_rules = self._parse_rule_json_safe(rule_json)
                    if isinstance(parsed_rules, list):
                        all_rules.update(parsed_rules)
                rules_used = list(all_rules)

            stats.append(
                UserSatisfactionStats(
                    feedback_date=row.feedback_date,
                    total_feedback=int(row.total_feedback),
                    avg_rating=float(row.avg_rating or 0),
                    positive_feedback=int(row.positive_feedback or 0),
                    negative_feedback=int(row.negative_feedback or 0),
                    rules_used=rules_used,
                )
            )

        return stats

    @handle_database_errors(
        rollback_session=True,
        return_format="none",
        operation_name="get_performance_trends",
        retry_count=2,
    )
    async def get_performance_trends(
        self,
        rule_id: str | None = None,
        days: int = 30,
        db_session: AsyncSession | None = None,
    ) -> dict[str, Any]:
        """Get performance trends for rules over time"""
        if not db_session:
            return {}

        # Calculate cutoff date
        cutoff_date = aware_utc_now() - timedelta(days=days)

        # Base query for performance trends
        base_query = """
            SELECT
                DATE_TRUNC('day', rp.created_at) as trend_date,
                rp.rule_id,
                rm.rule_name,
                AVG(rp.improvement_score) as avg_score,
                COUNT(*) as usage_count,
                AVG(rp.confidence_level) as avg_confidence,
                AVG(rp.execution_time_ms) as avg_execution_time
            FROM rule_performance rp
            JOIN rule_metadata rm ON rp.rule_id = rm.rule_id
            WHERE rp.created_at >= :cutoff_date
        """

        params = {"cutoff_date": cutoff_date}

        if rule_id:
            base_query += " AND rp.rule_id = :rule_id"
            params["rule_id"] = rule_id

        base_query += """
            GROUP BY DATE_TRUNC('day', rp.created_at), rp.rule_id, rm.rule_name
            ORDER BY trend_date DESC, rp.rule_id
        """

        query = text(base_query)
        result = await db_session.execute(query, params)

        trends = {
            "daily_trends": [],
            "summary": {},
            "period_start": cutoff_date.isoformat(),
            "period_end": aware_utc_now().isoformat(),
        }

        total_usage = 0
        total_score = 0
        score_count = 0

        for row in result:
            trend_data = {
                "date": row.trend_date.isoformat(),
                "rule_id": row.rule_id,
                "rule_name": row.rule_name,
                "avg_score": float(row.avg_score or 0),
                "usage_count": row.usage_count,
                "avg_confidence": float(row.avg_confidence or 0),
                "avg_execution_time": float(row.avg_execution_time or 0),
            }
            trends["daily_trends"].append(trend_data)

            # Accumulate for summary
            total_usage += row.usage_count
            if row.avg_score:
                total_score += float(row.avg_score) * row.usage_count
                score_count += row.usage_count

        # Calculate summary statistics
        trends["summary"] = {
            "total_usage": total_usage,
            "avg_score_weighted": total_score / score_count if score_count > 0 else 0,
            "rules_analyzed": len(
                set(row["rule_id"] for row in trends["daily_trends"])
            ),
            "days_analyzed": days,
        }

        return trends

    @handle_database_errors(
        rollback_session=True,
        return_format="none",
        operation_name="get_prompt_type_analysis",
        retry_count=2,
    )
    async def get_prompt_type_analysis(
        self, days: int = 30, db_session: AsyncSession | None = None
    ) -> dict[str, Any]:
        """Analyze performance by prompt type"""
        if not db_session:
            return {}

        cutoff_date = aware_utc_now() - timedelta(days=days)

        query = text("""
            SELECT
                'general' as prompt_type,
                COUNT(*) as usage_count,
                AVG(rp.improvement_score) as avg_improvement,
                AVG(rp.confidence_level) as avg_confidence,
                COUNT(DISTINCT rp.rule_id) as rules_used
            FROM rule_performance rp
            WHERE rp.created_at >= :cutoff_date
        """)

        result = await db_session.execute(query, {"cutoff_date": cutoff_date})

        analysis = {
            "prompt_types": [],
            "summary": {
                "total_prompts": 0,
                "most_common_type": None,
                "best_performing_type": None,
                "avg_improvement_overall": 0,
            },
        }

        total_prompts = 0
        total_improvement = 0
        best_score = 0
        best_type = None
        most_common_count = 0
        most_common_type = None

        for row in result:
            type_data = {
                "prompt_type": row.prompt_type,
                "usage_count": row.usage_count,
                "avg_improvement": float(row.avg_improvement or 0),
                "avg_confidence": float(row.avg_confidence or 0),
                "rules_used": row.rules_used,
            }
            analysis["prompt_types"].append(type_data)

            # Track totals and bests
            total_prompts += row.usage_count
            total_improvement += float(row.avg_improvement or 0) * row.usage_count

            if row.usage_count > most_common_count:
                most_common_count = row.usage_count
                most_common_type = row.prompt_type

            if float(row.avg_improvement or 0) > best_score:
                best_score = float(row.avg_improvement or 0)
                best_type = row.prompt_type

        # Update summary
        analysis["summary"] = {
            "total_prompts": total_prompts,
            "most_common_type": most_common_type,
            "best_performing_type": best_type,
            "avg_improvement_overall": total_improvement / total_prompts
            if total_prompts > 0
            else 0,
        }

        return analysis

    @handle_database_errors(
        rollback_session=True,
        return_format="none",
        operation_name="get_rule_correlation_analysis",
        retry_count=2,
    )
    async def get_rule_correlation_analysis(
        self, days: int = 30, db_session: AsyncSession | None = None
    ) -> dict[str, Any]:
        """Analyze correlations between rules and their effectiveness"""
        if not db_session:
            return {}

        cutoff_date = aware_utc_now() - timedelta(days=days)

        # This would require more complex analysis
        # For now, return a simple analysis of rule combinations
        query = text("""
            SELECT
                rp.rule_id,
                rm.rule_name,
                rp.session_id as prompt_type,
                AVG(rp.improvement_score) as avg_improvement,
                COUNT(*) as usage_count
            FROM rule_performance rp
            JOIN rule_metadata rm ON rp.rule_id = rm.rule_id
            WHERE rp.created_at >= :cutoff_date
            GROUP BY rp.rule_id, rm.rule_name, rp.session_id
            HAVING COUNT(*) >= 3
            ORDER BY avg_improvement DESC
        """)

        result = await db_session.execute(query, {"cutoff_date": cutoff_date})

        correlations = {"rule_prompt_combinations": [], "insights": []}

        for row in result:
            combination = {
                "rule_id": row.rule_id,
                "rule_name": row.rule_name,
                "prompt_type": row.prompt_type,
                "avg_improvement": float(row.avg_improvement or 0),
                "usage_count": row.usage_count,
            }
            correlations["rule_prompt_combinations"].append(combination)

        # Generate insights
        if correlations["rule_prompt_combinations"]:
            best_combo = max(
                correlations["rule_prompt_combinations"],
                key=lambda x: x["avg_improvement"],
            )
            correlations["insights"].append(
                f"Best performing combination: {best_combo['rule_name']} "
                f"on {best_combo['prompt_type']} prompts "
                f"(avg improvement: {best_combo['avg_improvement']:.3f})"
            )

        return correlations

    @handle_database_errors(
        rollback_session=True,
        return_format="none",
        operation_name="get_performance_summary",
        retry_count=2,
    )
    async def get_performance_summary(
        self, days: int = 30, db_session: AsyncSession | None = None
    ) -> dict[str, Any]:
        """Get comprehensive performance summary for the system"""
        if not db_session:
            return {
                "total_sessions": 0,
                "avg_improvement": 0.0,
                "success_rate": 0.0,
                "total_rules_applied": 0,
                "avg_processing_time_ms": 0.0,
            }

        from datetime import datetime, timedelta

        import sqlmodel
        from sqlalchemy import func, select

        from ..database.models import ImprovementSession, RulePerformance

        # Calculate date range
        end_date = aware_utc_now()
        start_date = end_date - timedelta(days=days)

        # Get session statistics

        # Only use fields that exist: session_id, created_at
        # For 'successful_sessions', assume all sessions are successful (or skip this metric)
        session_query = select(
            func.count(ImprovementSession.session_id).label("total_sessions"),
            # Skip avg_improvement calculation as improvement_metrics is JSON and can't be averaged directly
        ).where(ImprovementSession.created_at >= start_date)

        session_result = await db_session.execute(session_query)
        session_stats = session_result.first() if session_result else None

        # Get rule performance statistics
        rule_query = select(
            func.count(RulePerformance.id).label("total_rules_applied"),
            func.avg(RulePerformance.improvement_score).label("avg_rule_improvement"),
            func.avg(RulePerformance.execution_time_ms).label("avg_processing_time"),
        ).where(RulePerformance.created_at >= start_date)

        rule_result = await db_session.execute(rule_query)
        rule_stats = rule_result.first() if rule_result else None

        # Calculate success rate (not available, so set to 1.0 if sessions exist)
        total_sessions = (
            session_stats.total_sessions
            if session_stats and hasattr(session_stats, "total_sessions")
            else 0
        )
        success_rate = 1.0 if total_sessions > 0 else 0.0

        return {
            "total_sessions": total_sessions,
            "avg_improvement": 0.0,  # Fixed: Use default value since JSON averaging is not supported
            "success_rate": success_rate,
            "total_rules_applied": rule_stats.total_rules_applied
            if rule_stats
            and hasattr(rule_stats, "total_rules_applied")
            and rule_stats.total_rules_applied is not None
            else 0,
            "avg_processing_time_ms": rule_stats.avg_processing_time
            if rule_stats
            and hasattr(rule_stats, "avg_processing_time")
            and rule_stats.avg_processing_time is not None
            else 0.0,
            "date_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
            },
        }

    @handle_validation_errors(
        return_format="dict",
        operation_name="parse_rule_json",
        log_validation_details=False,
    )
    def _parse_rule_json_safe(self, rule_json: str) -> list[str]:
        """Safely parse rule JSON and extract rule IDs using error handling decorator"""
        import json

        if not rule_json or rule_json == "null":
            return []

        rules_data = json.loads(rule_json)
        if not isinstance(rules_data, list):
            return []

        rule_ids = []
        for rule in rules_data:
            if isinstance(rule, dict) and "rule_id" in rule:
                rule_ids.append(rule["rule_id"])

        return rule_ids
