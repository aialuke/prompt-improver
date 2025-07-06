"""
Analytics Service
Provides comprehensive analytics and reporting for APES
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from sqlmodel import select as sqlmodel_select

from ..database.models import (
    RulePerformance,
    UserFeedback,
    RuleEffectivenessStats,
    UserSatisfactionStats,
)


class AnalyticsService:
    """
    Service for analytics and reporting functionality
    """

    async def get_rule_effectiveness(
        self, days: int = 30, min_usage_count: int = 5, db_session: AsyncSession = None
    ) -> List[RuleEffectivenessStats]:
        """
        Get rule effectiveness analytics for the specified time period
        """
        if not db_session:
            return []

        try:
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Query rule effectiveness using SQL view or direct aggregation
            query = text("""
                SELECT 
                    rule_id,
                    rule_name,
                    COUNT(*) as usage_count,
                    AVG(improvement_score) as avg_improvement,
                    STDDEV(improvement_score) as score_stddev,
                    MIN(improvement_score) as min_improvement,
                    MAX(improvement_score) as max_improvement,
                    AVG(confidence_level) as avg_confidence,
                    AVG(execution_time_ms) as avg_execution_time,
                    COUNT(DISTINCT prompt_type) as prompt_types_count
                FROM rule_performance 
                WHERE created_at >= :cutoff_date
                GROUP BY rule_id, rule_name
                HAVING COUNT(*) >= :min_usage_count
                ORDER BY avg_improvement DESC
            """)

            result = await db_session.execute(
                query, {"cutoff_date": cutoff_date, "min_usage_count": min_usage_count}
            )

            stats = []
            for row in result:
                stats.append(
                    RuleEffectivenessStats(
                        rule_id=row.rule_id,
                        rule_name=row.rule_name,
                        usage_count=row.usage_count,
                        avg_improvement=float(row.avg_improvement or 0),
                        score_stddev=float(row.score_stddev or 0),
                        min_improvement=float(row.min_improvement or 0),
                        max_improvement=float(row.max_improvement or 0),
                        avg_confidence=float(row.avg_confidence or 0),
                        avg_execution_time=float(row.avg_execution_time or 0),
                        prompt_types_count=row.prompt_types_count or 0,
                    )
                )

            return stats

        except Exception as e:
            print(f"Error getting rule effectiveness: {e}")
            return []

    async def get_user_satisfaction(
        self, days: int = 30, db_session: AsyncSession = None
    ) -> List[UserSatisfactionStats]:
        """
        Get user satisfaction analytics and trends
        """
        if not db_session:
            return []

        try:
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Query user satisfaction by day
            query = text("""
                SELECT 
                    DATE_TRUNC('day', created_at) as feedback_date,
                    COUNT(*) as total_feedback,
                    AVG(user_rating::FLOAT) as avg_rating,
                    COUNT(CASE WHEN user_rating >= 4 THEN 1 END) as positive_feedback,
                    COUNT(CASE WHEN user_rating <= 2 THEN 1 END) as negative_feedback,
                    ARRAY_AGG(DISTINCT applied_rules::text) as rules_used
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
                        try:
                            if rule_json and rule_json != "null":
                                rules_data = json.loads(rule_json)
                                if isinstance(rules_data, list):
                                    for rule in rules_data:
                                        if isinstance(rule, dict) and "rule_id" in rule:
                                            all_rules.add(rule["rule_id"])
                        except (json.JSONDecodeError, TypeError):
                            continue
                    rules_used = list(all_rules)

                stats.append(
                    UserSatisfactionStats(
                        feedback_date=row.feedback_date,
                        total_feedback=row.total_feedback,
                        avg_rating=float(row.avg_rating or 0),
                        positive_feedback=row.positive_feedback or 0,
                        negative_feedback=row.negative_feedback or 0,
                        rules_used=rules_used,
                    )
                )

            return stats

        except Exception as e:
            print(f"Error getting user satisfaction: {e}")
            return []

    async def get_performance_trends(
        self,
        rule_id: Optional[str] = None,
        days: int = 30,
        db_session: AsyncSession = None,
    ) -> Dict[str, Any]:
        """
        Get performance trends for rules over time
        """
        if not db_session:
            return {}

        try:
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Base query for performance trends
            base_query = """
                SELECT 
                    DATE_TRUNC('day', created_at) as trend_date,
                    rule_id,
                    rule_name,
                    AVG(improvement_score) as avg_score,
                    COUNT(*) as usage_count,
                    AVG(confidence_level) as avg_confidence,
                    AVG(execution_time_ms) as avg_execution_time
                FROM rule_performance 
                WHERE created_at >= :cutoff_date
            """

            params = {"cutoff_date": cutoff_date}

            if rule_id:
                base_query += " AND rule_id = :rule_id"
                params["rule_id"] = rule_id

            base_query += """
                GROUP BY DATE_TRUNC('day', created_at), rule_id, rule_name
                ORDER BY trend_date DESC, rule_id
            """

            query = text(base_query)
            result = await db_session.execute(query, params)

            trends = {
                "daily_trends": [],
                "summary": {},
                "period_start": cutoff_date.isoformat(),
                "period_end": datetime.utcnow().isoformat(),
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
                "avg_score_weighted": total_score / score_count
                if score_count > 0
                else 0,
                "rules_analyzed": len(
                    set(row["rule_id"] for row in trends["daily_trends"])
                ),
                "days_analyzed": days,
            }

            return trends

        except Exception as e:
            print(f"Error getting performance trends: {e}")
            return {}

    async def get_prompt_type_analysis(
        self, days: int = 30, db_session: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Analyze performance by prompt type
        """
        if not db_session:
            return {}

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            query = text("""
                SELECT 
                    COALESCE(prompt_type, 'unknown') as prompt_type,
                    COUNT(*) as usage_count,
                    AVG(improvement_score) as avg_improvement,
                    AVG(confidence_level) as avg_confidence,
                    COUNT(DISTINCT rule_id) as rules_used
                FROM rule_performance 
                WHERE created_at >= :cutoff_date
                GROUP BY COALESCE(prompt_type, 'unknown')
                ORDER BY usage_count DESC
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

        except Exception as e:
            print(f"Error getting prompt type analysis: {e}")
            return {}

    async def get_rule_correlation_analysis(
        self, days: int = 30, db_session: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Analyze correlations between rules and their effectiveness
        """
        if not db_session:
            return {}

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # This would require more complex analysis
            # For now, return a simple analysis of rule combinations
            query = text("""
                SELECT 
                    rule_id,
                    rule_name,
                    prompt_type,
                    AVG(improvement_score) as avg_improvement,
                    COUNT(*) as usage_count
                FROM rule_performance 
                WHERE created_at >= :cutoff_date
                GROUP BY rule_id, rule_name, prompt_type
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

        except Exception as e:
            print(f"Error getting correlation analysis: {e}")
            return {}
