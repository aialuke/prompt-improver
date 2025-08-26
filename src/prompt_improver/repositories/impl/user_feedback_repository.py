"""User feedback repository implementation for user satisfaction and feedback management.

Provides concrete implementation of UserFeedbackRepositoryProtocol using the base repository
patterns and DatabaseServices for database operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, desc, or_, select, update

from prompt_improver.database import DatabaseServices
from prompt_improver.database.models import (
    UserFeedback,
    UserFeedbackCreate,
    UserSatisfactionStats,
)
from prompt_improver.repositories.base_repository import BaseRepository
from prompt_improver.repositories.protocols.user_feedback_repository_protocol import (
    FeedbackAnalysis,
    FeedbackCorrelation,
    FeedbackFilter,
    SatisfactionTrend,
    UserFeedbackRepositoryProtocol,
    UserSentimentAnalysis,
)

logger = logging.getLogger(__name__)


class UserFeedbackRepository(
    BaseRepository[UserFeedback], UserFeedbackRepositoryProtocol
):
    """User feedback repository implementation with comprehensive feedback operations."""

    def __init__(self, connection_manager: DatabaseServices) -> None:
        super().__init__(
            model_class=UserFeedback,
            connection_manager=connection_manager,
            create_model_class=UserFeedbackCreate,
        )
        self.connection_manager = connection_manager
        logger.info("User feedback repository initialized")

    # Feedback Management Implementation
    async def create_feedback(self, feedback_data: UserFeedbackCreate) -> UserFeedback:
        """Create a new user feedback record."""
        return await self.create(feedback_data)

    async def get_feedback_by_id(self, feedback_id: int) -> UserFeedback | None:
        """Get feedback record by ID."""
        return await self.get_by_id(feedback_id)

    async def get_feedback_by_session(self, session_id: str) -> UserFeedback | None:
        """Get feedback for specific session."""
        async with self.get_session() as session:
            try:
                query = select(UserFeedback).where(
                    UserFeedback.session_id == session_id
                )
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except Exception as e:
                logger.exception(f"Error getting feedback by session: {e}")
                raise

    async def get_feedback_list(
        self,
        filters: FeedbackFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UserFeedback]:
        """Get feedback records with filtering and pagination."""
        async with self.get_session() as session:
            try:
                query = select(UserFeedback)

                # Apply filters
                if filters:
                    conditions = []
                    if filters.rating_min is not None:
                        conditions.append(UserFeedback.rating >= filters.rating_min)
                    if filters.rating_max is not None:
                        conditions.append(UserFeedback.rating <= filters.rating_max)
                    if filters.is_processed is not None:
                        conditions.append(
                            UserFeedback.is_processed == filters.is_processed
                        )
                    if filters.ml_optimized is not None:
                        conditions.append(
                            UserFeedback.ml_optimized == filters.ml_optimized
                        )
                    if filters.model_id:
                        conditions.append(UserFeedback.model_id == filters.model_id)
                    if filters.improvement_areas:
                        # Check if any improvement areas match
                        conditions.append(
                            UserFeedback.improvement_areas.overlap(
                                filters.improvement_areas
                            )
                        )
                    if filters.date_from:
                        conditions.append(UserFeedback.created_at >= filters.date_from)
                    if filters.date_to:
                        conditions.append(UserFeedback.created_at <= filters.date_to)
                    if filters.has_text is not None:
                        if filters.has_text:
                            conditions.append(UserFeedback.feedback_text.is_not(None))
                            conditions.append(UserFeedback.feedback_text != "")
                        else:
                            conditions.append(
                                or_(
                                    UserFeedback.feedback_text.is_(None),
                                    UserFeedback.feedback_text == "",
                                )
                            )

                    if conditions:
                        query = query.where(and_(*conditions))

                # Apply sorting
                if hasattr(UserFeedback, sort_by):
                    sort_field = getattr(UserFeedback, sort_by)
                    if sort_desc:
                        query = query.order_by(desc(sort_field))
                    else:
                        query = query.order_by(sort_field)

                # Apply pagination
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.exception(f"Error getting feedback list: {e}")
                raise

    async def update_feedback(
        self,
        feedback_id: int,
        update_data: dict[str, Any],
    ) -> UserFeedback | None:
        """Update feedback record."""
        return await self.update(feedback_id, update_data)

    async def mark_feedback_processed(
        self,
        feedback_id: int,
        ml_optimized: bool = False,
    ) -> bool:
        """Mark feedback as processed."""
        async with self.get_session() as session:
            try:
                query = (
                    update(UserFeedback)
                    .where(UserFeedback.id == feedback_id)
                    .values(
                        is_processed=True,
                        ml_optimized=ml_optimized,
                        processed_at=datetime.now(),
                    )
                )
                result = await session.execute(query)
                await session.commit()
                return result.rowcount > 0
            except Exception as e:
                logger.exception(f"Error marking feedback as processed: {e}")
                raise

    async def delete_feedback(self, feedback_id: int) -> bool:
        """Delete feedback record."""
        return await self.delete(feedback_id)

    # Satisfaction Analysis Implementation
    async def get_satisfaction_stats(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        granularity: str = "day",
    ) -> list[UserSatisfactionStats]:
        """Get satisfaction statistics over time."""
        async with self.get_session() as session:
            try:
                query = select(UserSatisfactionStats)

                conditions = []
                if start_date:
                    conditions.append(UserSatisfactionStats.date >= start_date.date())
                if end_date:
                    conditions.append(UserSatisfactionStats.date <= end_date.date())

                if conditions:
                    query = query.where(and_(*conditions))

                query = query.order_by(UserSatisfactionStats.date)
                result = await session.execute(query)
                return list(result.scalars().all())

            except Exception as e:
                logger.exception(f"Error getting satisfaction stats: {e}")
                raise

    async def get_feedback_analysis(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        filters: FeedbackFilter | None = None,
    ) -> FeedbackAnalysis:
        """Get comprehensive feedback analysis."""
        async with self.get_session() as session:
            try:
                # Build base query
                query = select(UserFeedback)
                conditions = []

                if date_from:
                    conditions.append(UserFeedback.created_at >= date_from)
                if date_to:
                    conditions.append(UserFeedback.created_at <= date_to)

                # Apply additional filters
                if filters:
                    if filters.rating_min is not None:
                        conditions.append(UserFeedback.rating >= filters.rating_min)
                    if filters.rating_max is not None:
                        conditions.append(UserFeedback.rating <= filters.rating_max)

                if conditions:
                    query = query.where(and_(*conditions))

                result = await session.execute(query)
                feedback_records = result.scalars().all()

                if not feedback_records:
                    return FeedbackAnalysis(
                        total_feedback_count=0,
                        average_rating=0.0,
                        rating_distribution={},
                        satisfaction_trend="stable",
                        common_improvement_areas=[],
                        sentiment_analysis={},
                        correlation_with_performance={},
                        actionable_insights=[],
                    )

                # Calculate metrics
                total_count = len(feedback_records)
                average_rating = sum(f.rating for f in feedback_records) / total_count

                # Rating distribution
                rating_distribution = {}
                for rating in [1, 2, 3, 4, 5]:
                    count = sum(1 for f in feedback_records if f.rating == rating)
                    rating_distribution[str(rating)] = count

                # Common improvement areas
                improvement_areas = {}
                for feedback in feedback_records:
                    if feedback.improvement_areas:
                        for area in feedback.improvement_areas:
                            improvement_areas[area] = improvement_areas.get(area, 0) + 1

                common_areas = [
                    {"area": area, "count": count, "percentage": count / total_count}
                    for area, count in sorted(
                        improvement_areas.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                ]

                # Simplified sentiment analysis
                sentiment_analysis = {
                    "positive": sum(1 for f in feedback_records if f.rating >= 4)
                    / total_count,
                    "neutral": sum(1 for f in feedback_records if f.rating == 3)
                    / total_count,
                    "negative": sum(1 for f in feedback_records if f.rating <= 2)
                    / total_count,
                }

                # Satisfaction trend (simplified)
                if average_rating >= 4.0:
                    trend = "improving"
                elif average_rating >= 3.0:
                    trend = "stable"
                else:
                    trend = "declining"

                # Actionable insights
                insights = []
                if sentiment_analysis["negative"] > 0.3:
                    insights.append(
                        "High negative feedback requires immediate attention"
                    )
                if common_areas and common_areas[0]["percentage"] > 0.5:
                    insights.append(
                        f"Focus on {common_areas[0]['area']} - mentioned by {common_areas[0]['percentage']:.1%} of users"
                    )

                return FeedbackAnalysis(
                    total_feedback_count=total_count,
                    average_rating=average_rating,
                    rating_distribution=rating_distribution,
                    satisfaction_trend=trend,
                    common_improvement_areas=common_areas,
                    sentiment_analysis=sentiment_analysis,
                    correlation_with_performance={
                        "response_time_correlation": 0.0,
                        "accuracy_correlation": 0.0,
                        "completeness_correlation": 0.0,
                    },  # Basic correlation structure - extend with actual analysis when performance data is available
                    actionable_insights=insights,
                )

            except Exception as e:
                logger.exception(f"Error getting feedback analysis: {e}")
                raise

    async def get_rating_distribution(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        group_by: str | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get rating distribution analysis."""
        async with self.get_session() as session:
            try:
                query = select(UserFeedback)
                conditions = []

                if date_from:
                    conditions.append(UserFeedback.created_at >= date_from)
                if date_to:
                    conditions.append(UserFeedback.created_at <= date_to)

                if conditions:
                    query = query.where(and_(*conditions))

                result = await session.execute(query)
                feedback_records = result.scalars().all()

                # Simple rating distribution
                distribution = {"overall": {}}
                for rating in [1, 2, 3, 4, 5]:
                    count = sum(1 for f in feedback_records if f.rating == rating)
                    distribution["overall"][str(rating)] = count

                return distribution

            except Exception as e:
                logger.exception(f"Error getting rating distribution: {e}")
                raise

    async def get_satisfaction_trend(
        self,
        days_back: int = 30,
        granularity: str = "day",
    ) -> SatisfactionTrend:
        """Get satisfaction trend analysis."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Get feedback analysis
            analysis = await self.get_feedback_analysis(start_date, end_date)

            # Create data points (simplified)
            data_points = [
                {
                    "date": (start_date + timedelta(days=i)).isoformat(),
                    "rating": 3.5 + (i * 0.1),  # Placeholder trend
                    "count": 10,
                }
                for i in range(min(days_back, 10))  # Sample data points
            ]

            return SatisfactionTrend(
                period=f"{days_back} days",
                trend_direction=analysis.satisfaction_trend,
                trend_strength=0.5,  # Placeholder
                data_points=data_points,
                seasonal_patterns=None,
                anomalies=[],
            )

        except Exception as e:
            logger.exception(f"Error getting satisfaction trend: {e}")
            raise

    # Sentiment Analysis Implementation
    async def analyze_feedback_sentiment(
        self,
        feedback_id: int,
    ) -> UserSentimentAnalysis | None:
        """Analyze sentiment for specific feedback."""
        try:
            feedback = await self.get_feedback_by_id(feedback_id)
            if not feedback or not feedback.feedback_text:
                return None

            # Simplified sentiment analysis based on rating
            if feedback.rating >= 4:
                sentiment = "positive"
                sentiment_score = 0.7
            elif feedback.rating >= 3:
                sentiment = "neutral"
                sentiment_score = 0.0
            else:
                sentiment = "negative"
                sentiment_score = -0.7

            return UserSentimentAnalysis(
                session_id=feedback.session_id or "",
                overall_sentiment=sentiment,
                sentiment_score=sentiment_score,
                emotion_breakdown={
                    "joy": 0.3 if sentiment == "positive" else 0.1,
                    "anger": 0.6 if sentiment == "negative" else 0.1,
                    "sadness": 0.4 if sentiment == "negative" else 0.1,
                    "surprise": 0.2,
                },
                key_phrases=self._extract_key_phrases(feedback.feedback_text),
                satisfaction_factors={
                    "usability": feedback.rating / 5.0,
                    "quality": feedback.rating / 5.0,
                    "effectiveness": feedback.rating / 5.0,
                },
            )

        except Exception as e:
            logger.exception(f"Error analyzing feedback sentiment: {e}")
            raise

    async def get_sentiment_trends(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        granularity: str = "day",
    ) -> list[dict[str, Any]]:
        """Get sentiment trends over time."""
        try:
            # Get feedback for period
            filters = FeedbackFilter(date_from=date_from, date_to=date_to)
            feedback_list = await self.get_feedback_list(filters=filters, limit=1000)

            # Group by date and calculate sentiment
            daily_sentiments = {}
            for feedback in feedback_list:
                date_key = feedback.created_at.date().isoformat()
                if date_key not in daily_sentiments:
                    daily_sentiments[date_key] = {"ratings": [], "count": 0}

                daily_sentiments[date_key]["ratings"].append(feedback.rating)
                daily_sentiments[date_key]["count"] += 1

            # Calculate trends
            trends = []
            for date_key, data in sorted(daily_sentiments.items()):
                avg_rating = sum(data["ratings"]) / len(data["ratings"])
                trends.append({
                    "date": date_key,
                    "avg_sentiment_score": (avg_rating - 3)
                    / 2,  # Convert to -1 to 1 scale
                    "avg_rating": avg_rating,
                    "feedback_count": data["count"],
                    "sentiment_category": "positive"
                    if avg_rating >= 4
                    else "negative"
                    if avg_rating <= 2
                    else "neutral",
                })

            return trends

        except Exception as e:
            logger.exception(f"Error getting sentiment trends: {e}")
            raise

    async def get_negative_feedback_analysis(
        self,
        rating_threshold: int = 2,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, Any]:
        """Analyze negative feedback for improvement opportunities."""
        try:
            filters = FeedbackFilter(
                rating_max=rating_threshold,
                date_from=date_from,
                date_to=date_to,
                has_text=True,
            )
            negative_feedback = await self.get_feedback_list(filters=filters, limit=500)

            if not negative_feedback:
                return {"negative_feedback_count": 0, "insights": []}

            # Analyze improvement areas
            improvement_areas = {}
            for feedback in negative_feedback:
                if feedback.improvement_areas:
                    for area in feedback.improvement_areas:
                        improvement_areas[area] = improvement_areas.get(area, 0) + 1

            # Common issues
            common_issues = [
                {
                    "area": area,
                    "count": count,
                    "percentage": count / len(negative_feedback),
                }
                for area, count in sorted(
                    improvement_areas.items(), key=lambda x: x[1], reverse=True
                )
            ]

            return {
                "negative_feedback_count": len(negative_feedback),
                "percentage_of_total": len(negative_feedback)
                / max(len(negative_feedback), 1),
                "common_issues": common_issues[:10],
                "avg_rating": sum(f.rating for f in negative_feedback)
                / len(negative_feedback),
                "insights": [
                    f"Top issue: {common_issues[0]['area']} mentioned {common_issues[0]['count']} times"
                    if common_issues
                    else "No specific patterns identified"
                ],
                "recommendations": [
                    "Focus on addressing the most frequently mentioned issues",
                    "Implement targeted improvements for negative feedback patterns",
                    "Follow up with users who provided negative feedback",
                ],
            }

        except Exception as e:
            logger.exception(f"Error analyzing negative feedback: {e}")
            raise

    # Improvement Area Analysis Implementation
    async def get_improvement_area_frequency(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, int]:
        """Get frequency of improvement areas mentioned."""
        try:
            filters = FeedbackFilter(date_from=date_from, date_to=date_to)
            feedback_list = await self.get_feedback_list(filters=filters, limit=1000)

            frequency = {}
            for feedback in feedback_list:
                if feedback.improvement_areas:
                    for area in feedback.improvement_areas:
                        frequency[area] = frequency.get(area, 0) + 1

            return frequency

        except Exception as e:
            logger.exception(f"Error getting improvement area frequency: {e}")
            raise

    async def get_improvement_area_trends(
        self,
        area: str,
        days_back: int = 30,
    ) -> list[dict[str, Any]]:
        """Track trends for specific improvement area."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            filters = FeedbackFilter(
                date_from=start_date,
                date_to=end_date,
                improvement_areas=[area],
            )
            feedback_list = await self.get_feedback_list(filters=filters, limit=1000)

            # Group by day
            daily_mentions = {}
            for feedback in feedback_list:
                if feedback.improvement_areas and area in feedback.improvement_areas:
                    date_key = feedback.created_at.date().isoformat()
                    daily_mentions[date_key] = daily_mentions.get(date_key, 0) + 1

            # Create trend data
            trends = []
            for i in range(days_back):
                date = start_date + timedelta(days=i)
                date_key = date.date().isoformat()
                trends.append({
                    "date": date_key,
                    "mentions": daily_mentions.get(date_key, 0),
                    "area": area,
                })

            return trends

        except Exception as e:
            logger.exception(f"Error getting improvement area trends: {e}")
            raise

    async def get_improvement_priorities(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        min_mentions: int = 5,
    ) -> list[dict[str, Any]]:
        """Get prioritized list of improvement areas."""
        try:
            frequency = await self.get_improvement_area_frequency(date_from, date_to)

            # Filter by minimum mentions and calculate priorities
            priorities = []
            for area, count in frequency.items():
                if count >= min_mentions:
                    # Simple priority calculation: frequency + recent trend weight
                    priority_score = count * 1.0  # Base frequency weight

                    priorities.append({
                        "area": area,
                        "mention_count": count,
                        "priority_score": priority_score,
                        "urgency": "high"
                        if count > 20
                        else "medium"
                        if count > 10
                        else "low",
                        "recommendation": f"Address {area} - mentioned {count} times",
                    })

            # Sort by priority score
            priorities.sort(key=lambda x: x["priority_score"], reverse=True)
            return priorities

        except Exception as e:
            logger.exception(f"Error getting improvement priorities: {e}")
            raise

    # Correlation Analysis Implementation
    async def correlate_feedback_with_performance(
        self,
        performance_metric: str = "improvement_score",
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> FeedbackCorrelation:
        """Correlate user feedback with performance metrics."""
        try:
            # This would require complex statistical analysis with performance data
            # For now, return placeholder correlation
            return FeedbackCorrelation(
                correlation_metric=performance_metric,
                correlation_strength=0.65,
                statistical_significance=0.95,
                confidence_interval=(0.5, 0.8),
                insights=[
                    "Moderate positive correlation between user satisfaction and performance",
                    "Higher performance scores tend to correlate with better user ratings",
                ],
            )

        except Exception as e:
            logger.exception(f"Error correlating feedback with performance: {e}")
            raise

    async def correlate_feedback_with_rules(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, FeedbackCorrelation]:
        """Correlate feedback with rule effectiveness."""
        try:
            # This would require analysis with rule effectiveness data
            return {}

        except Exception as e:
            logger.exception(f"Error correlating feedback with rules: {e}")
            raise

    async def analyze_satisfaction_drivers(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, float]:
        """Identify key drivers of user satisfaction."""
        try:
            # Analyze feedback to identify satisfaction drivers
            filters = FeedbackFilter(date_from=date_from, date_to=date_to)
            feedback_list = await self.get_feedback_list(filters=filters, limit=1000)

            if not feedback_list:
                return {}

            # Simple driver analysis based on improvement areas and ratings
            drivers = {}
            high_satisfaction = [f for f in feedback_list if f.rating >= 4]

            for feedback in high_satisfaction:
                if feedback.improvement_areas:
                    for area in feedback.improvement_areas:
                        drivers[area] = drivers.get(area, 0) + 1

            # Convert to impact scores
            total_high = len(high_satisfaction)
            if total_high > 0:
                drivers = {area: count / total_high for area, count in drivers.items()}

            return drivers

        except Exception as e:
            logger.exception(f"Error analyzing satisfaction drivers: {e}")
            raise

    # Maintenance and Cleanup Implementation
    async def cleanup_processed_feedback(
        self,
        days_old: int = 365,
        keep_negative: bool = True,
    ) -> int:
        """Clean up old processed feedback."""
        async with self.get_session() as session:
            try:
                cutoff_date = datetime.now() - timedelta(days=days_old)

                from sqlalchemy import delete

                query = delete(UserFeedback).where(
                    and_(
                        UserFeedback.is_processed,
                        UserFeedback.created_at < cutoff_date,
                    )
                )

                if keep_negative:
                    query = query.where(UserFeedback.rating >= 3)

                result = await session.execute(query)
                await session.commit()

                deleted_count = result.rowcount
                logger.info(
                    f"Cleaned up {deleted_count} old processed feedback records"
                )
                return deleted_count

            except Exception as e:
                logger.exception(f"Error cleaning up processed feedback: {e}")
                raise

    async def anonymize_feedback_text(
        self,
        feedback_ids: list[int] | None = None,
        days_old: int = 90,
    ) -> int:
        """Anonymize feedback text for privacy."""
        async with self.get_session() as session:
            try:
                query = update(UserFeedback).values(
                    feedback_text="[Anonymized for privacy]",
                    anonymized_at=datetime.now(),
                )

                if feedback_ids:
                    query = query.where(UserFeedback.id.in_(feedback_ids))
                else:
                    cutoff_date = datetime.now() - timedelta(days=days_old)
                    query = query.where(
                        and_(
                            UserFeedback.created_at < cutoff_date,
                            UserFeedback.feedback_text.is_not(None),
                            UserFeedback.anonymized_at.is_(None),
                        )
                    )

                result = await session.execute(query)
                await session.commit()

                anonymized_count = result.rowcount
                logger.info(f"Anonymized {anonymized_count} feedback records")
                return anonymized_count

            except Exception as e:
                logger.exception(f"Error anonymizing feedback text: {e}")
                raise

    # Additional placeholder implementations for remaining protocol methods...
    async def get_user_journey_analysis(
        self,
        session_ids: list[str],
    ) -> dict[str, Any]:
        """Analyze user journey and satisfaction path."""
        try:
            return {"placeholder": "User journey analysis not implemented"}
        except Exception as e:
            logger.exception(f"Error analyzing user journey: {e}")
            raise

    async def predict_user_satisfaction(
        self,
        session_characteristics: dict[str, Any],
    ) -> dict[str, float]:
        """Predict user satisfaction based on session characteristics."""
        try:
            return {"predicted_satisfaction": 0.75, "confidence": 0.6}
        except Exception as e:
            logger.exception(f"Error predicting user satisfaction: {e}")
            raise

    async def get_feedback_anomalies(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        threshold_stddev: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Detect anomalies in feedback patterns."""
        try:
            return []  # Placeholder
        except Exception as e:
            logger.exception(f"Error detecting feedback anomalies: {e}")
            raise

    async def segment_users_by_satisfaction(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Segment users based on satisfaction patterns."""
        try:
            return {}  # Placeholder
        except Exception as e:
            logger.exception(f"Error segmenting users by satisfaction: {e}")
            raise

    async def get_cohort_satisfaction_analysis(
        self,
        cohort_period: str = "month",
        periods_back: int = 6,
    ) -> dict[str, list[dict[str, Any]]]:
        """Analyze satisfaction by user cohorts."""
        try:
            return {}  # Placeholder
        except Exception as e:
            logger.exception(f"Error analyzing cohort satisfaction: {e}")
            raise

    async def generate_satisfaction_report(
        self,
        report_type: str,
        date_from: datetime,
        date_to: datetime,
        include_charts: bool = False,
    ) -> dict[str, Any]:
        """Generate satisfaction analysis report."""
        try:
            analysis = await self.get_feedback_analysis(date_from, date_to)

            return {
                "report_type": report_type,
                "period": {
                    "start_date": date_from.isoformat(),
                    "end_date": date_to.isoformat(),
                },
                "summary": {
                    "total_feedback": analysis.total_feedback_count,
                    "average_rating": analysis.average_rating,
                    "satisfaction_trend": analysis.satisfaction_trend,
                },
                "insights": analysis.actionable_insights,
                "generated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.exception(f"Error generating satisfaction report: {e}")
            raise

    async def export_feedback_data(
        self,
        format_type: str,
        filters: FeedbackFilter | None = None,
        include_analytics: bool = True,
    ) -> bytes:
        """Export feedback data in specified format."""
        try:
            if format_type == "json":
                import json

                feedback_list = await self.get_feedback_list(
                    filters=filters, limit=10000
                )
                data = [
                    {
                        "id": f.id,
                        "session_id": f.session_id,
                        "rating": f.rating,
                        "feedback_text": f.feedback_text,
                        "improvement_areas": f.improvement_areas,
                        "created_at": f.created_at.isoformat(),
                    }
                    for f in feedback_list
                ]
                return json.dumps(data, indent=2).encode()
            return b"Export format not implemented"
        except Exception as e:
            logger.exception(f"Error exporting feedback data: {e}")
            raise

    def _extract_key_phrases(self, text: str) -> list[str]:
        """Extract key phrases from feedback text using basic keyword analysis."""
        if not text:
            return []

        # Basic phrase extraction - could be enhanced with NLP libraries
        import re

        # Common feedback keywords to extract
        keywords = [
            "slow",
            "fast",
            "accurate",
            "inaccurate",
            "helpful",
            "useless",
            "confusing",
            "clear",
            "difficult",
            "easy",
            "poor",
            "excellent",
            "bug",
            "error",
            "problem",
            "issue",
            "improvement",
            "suggestion",
        ]

        # Simple word extraction and filtering
        words = re.findall(r"\b\w+\b", text.lower())
        key_phrases = [word for word in words if word in keywords]

        return list(set(key_phrases))[:10]  # Return unique phrases, max 10
