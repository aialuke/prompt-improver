"""MCP ML Data Collector.

Collects rule application data and user feedback for the ML pipeline.
Maintains strict architectural separation - MCP only collects and formats data,
never performs ML operations directly.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from prompt_improver.performance.monitoring.health.background_manager import (
    TaskPriority,
    get_background_task_manager,
)
from prompt_improver.repositories.factory import (
    RepositoryFactory,
    get_repository_factory,
)
from prompt_improver.database.models import (
    UserFeedbackCreate,
)
from prompt_improver.repositories.protocols.user_feedback_repository_protocol import (
    FeedbackFilter,
    UserFeedbackRepositoryProtocol,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

if TYPE_CHECKING:
    from prompt_improver.repositories.protocols.analytics_repository_protocol import (
        AnalyticsRepositoryProtocol,
    )
    from prompt_improver.database import DatabaseServices

logger = logging.getLogger(__name__)


@dataclass
class RuleApplicationData:
    """Data structure for rule application results."""

    rule_id: str
    prompt_text: str
    enhanced_prompt: str
    improvement_score: float
    confidence_level: float
    response_time_ms: float
    prompt_characteristics: dict[str, Any]
    applied_rules: list[str]
    user_agent: str
    session_id: str
    timestamp: datetime


@dataclass
class UserFeedbackData:
    """Data structure for user feedback."""

    feedback_id: str
    rule_application_id: str
    user_rating: float
    effectiveness_score: float
    satisfaction_score: float
    feedback_text: str | None
    improvement_suggestions: list[str]
    timestamp: datetime


@dataclass
class MLDataPackage:
    """Complete data package for ML pipeline consumption."""

    rule_applications: list[RuleApplicationData]
    user_feedback: list[UserFeedbackData]
    collection_period: dict[str, datetime]
    data_quality_metrics: dict[str, float]
    total_samples: int


class MCPMLDataCollector:
    """MCP ML Data Collector.

    Responsible for:
    - Collecting rule application results
    - Gathering user feedback
    - Formatting data for ML pipeline consumption
    - Maintaining data quality metrics
    - Providing clean handoff to existing ML system

    NOT responsible for:
    - ML analysis or pattern recognition
    - Rule generation or optimization
    - Performance analytics or predictions
    """

    def __init__(self, database_services = None) -> None:
        """Initialize MCP ML data collector.

        Uses repository layer for all database operations following clean architecture.

        Args:
            database_services: Database services instance for repository initialization
        """
        self.batch_size = 100
        self.collection_interval_seconds = 300
        self.data_retention_days = 90
        self.quality_metrics: dict[str, float] = {
            "total_collected": 0.0,
            "successful_collections": 0.0,
            "failed_collections": 0.0,
            "data_completeness_rate": 0.0,
            "collection_latency_ms": 0.0,
        }
        self._collection_task_id: str | None = None
        self._running = False

        # Initialize repositories through factory
        if database_services:
            self._repository_factory = RepositoryFactory(database_services)
            self._repository_factory.initialize()
        else:
            self._repository_factory = None

        self._analytics_repository: AnalyticsRepositoryProtocol | None = None
        self._feedback_repository: UserFeedbackRepositoryProtocol | None = None

    async def start_collection(self) -> None:
        """Start background data collection for ML pipeline."""
        if self._running:
            logger.warning("Data collection already running")
            return
        self._running = True
        task_manager = get_background_task_manager()
        task_id = await task_manager.submit_enhanced_task(
            task_id=f"ml_data_collection_{id(self)}",
            coroutine=self._collection_loop,
            priority=TaskPriority.HIGH,
            tags={
                "service": "ml_data_collector",
                "type": "collection_loop",
                "component": "mcp_server",
            },
        )
        self._collection_task_id = task_id
        logger.info("Started MCP ML data collection with enhanced task management")

    async def _ensure_repositories_initialized(self) -> None:
        """Ensure repositories are properly initialized."""
        if not self._repository_factory:
            # Try to get factory from global instance if not provided
            try:
                from prompt_improver.database import DatabaseServices

                # This is a fallback - in production, database_services should be injected
                database_services = DatabaseServices()
                self._repository_factory = get_repository_factory(database_services)
            except Exception as e:
                logger.exception(f"Failed to initialize repository factory: {e}")
                raise RuntimeError(
                    "Repository factory not available and could not be initialized"
                )

        if not self._analytics_repository:
            self._analytics_repository = (
                self._repository_factory.get_analytics_repository()
            )

        if not self._feedback_repository:
            self._feedback_repository = (
                self._repository_factory.get_user_feedback_repository()
            )

    async def stop_collection(self) -> None:
        """Stop background data collection."""
        self._running = False
        if self._collection_task_id:
            task_manager = get_background_task_manager()
            await task_manager.cancel_task(self._collection_task_id)
            self._collection_task_id = None
        logger.info("Stopped MCP ML data collection")

    async def collect_rule_application(
        self,
        rule_id: str,
        prompt_text: str,
        enhanced_prompt: str,
        improvement_score: float,
        confidence_level: float,
        response_time_ms: float,
        prompt_characteristics: dict[str, Any],
        applied_rules: list[str],
        user_agent: str,
        session_id: str,
    ) -> str:
        """Collect rule application data for ML pipeline.

        Args:
            rule_id: Applied rule identifier
            prompt_text: Original prompt
            enhanced_prompt: Enhanced prompt result
            improvement_score: Calculated improvement score
            confidence_level: Rule application confidence
            response_time_ms: Processing time
            prompt_characteristics: Analyzed prompt characteristics
            applied_rules: All rules applied in combination
            user_agent: User agent string
            session_id: Session identifier

        Returns:
            Rule application ID for feedback correlation
        """
        try:
            application_data = RuleApplicationData(
                rule_id=rule_id,
                prompt_text=prompt_text,
                enhanced_prompt=enhanced_prompt,
                improvement_score=improvement_score,
                confidence_level=confidence_level,
                response_time_ms=response_time_ms,
                prompt_characteristics=prompt_characteristics,
                applied_rules=applied_rules,
                user_agent=user_agent,
                session_id=session_id,
                timestamp=aware_utc_now(),
            )
            application_id = await self._store_rule_application(application_data)
            self.quality_metrics["total_collected"] += 1
            self.quality_metrics["successful_collections"] += 1
            return application_id
        except Exception as e:
            logger.exception(f"Failed to collect rule application data: {e}")
            self.quality_metrics["failed_collections"] += 1
            raise

    async def collect_user_feedback(
        self,
        rule_application_id: str,
        user_rating: float,
        effectiveness_score: float,
        satisfaction_score: float,
        feedback_text: str | None = None,
        improvement_suggestions: list[str] | None = None,
    ) -> str:
        """Collect user feedback for ML pipeline.

        Args:
            rule_application_id: Associated rule application ID
            user_rating: User rating (1-5 scale)
            effectiveness_score: Effectiveness rating (0-1 scale)
            satisfaction_score: Satisfaction rating (0-1 scale)
            feedback_text: Optional feedback text
            improvement_suggestions: Optional improvement suggestions

        Returns:
            Feedback ID
        """
        try:
            feedback_data = UserFeedbackData(
                feedback_id=f"feedback_{int(time.time() * 1000)}",
                rule_application_id=rule_application_id,
                user_rating=user_rating,
                effectiveness_score=effectiveness_score,
                satisfaction_score=satisfaction_score,
                feedback_text=feedback_text,
                improvement_suggestions=improvement_suggestions or [],
                timestamp=aware_utc_now(),
            )
            return await self._store_user_feedback(feedback_data)
        except Exception as e:
            logger.exception(f"Failed to collect user feedback: {e}")
            raise

    async def prepare_ml_data_package(
        self, hours_back: int = 24, min_samples: int = 10
    ) -> MLDataPackage | None:
        """Prepare data package for ML pipeline consumption.

        Args:
            hours_back: Hours of data to include
            min_samples: Minimum samples required

        Returns:
            MLDataPackage for ML pipeline or None if insufficient data
        """
        try:
            start_time = aware_utc_now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            start_time = start_time.replace(hour=start_time.hour - hours_back)
            rule_applications = await self._get_rule_applications(start_time)
            user_feedback = await self._get_user_feedback(start_time)
            if len(rule_applications) < min_samples:
                logger.info(
                    f"Insufficient data for ML package: {len(rule_applications)} < {min_samples}"
                )
                return None
            data_quality = self._calculate_data_quality(
                rule_applications, user_feedback
            )
            ml_package = MLDataPackage(
                rule_applications=rule_applications,
                user_feedback=user_feedback,
                collection_period={"start": start_time, "end": aware_utc_now()},
                data_quality_metrics=data_quality,
                total_samples=len(rule_applications),
            )
            logger.info(
                f"Prepared ML data package with {len(rule_applications)} applications and {len(user_feedback)} feedback items"
            )
            return ml_package
        except Exception as e:
            logger.exception(f"Failed to prepare ML data package: {e}")
            return None

    async def _collection_loop(self) -> None:
        """Background collection loop for periodic ML data preparation."""
        while self._running:
            try:
                ml_package = await self.prepare_ml_data_package()
                if ml_package:
                    await self._signal_ml_pipeline(ml_package)
                await asyncio.sleep(self.collection_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in collection loop: {e}")
                await asyncio.sleep(60)

    async def _store_rule_application(self, data: RuleApplicationData) -> str:
        """Store rule application data using repository layer."""
        try:
            # Ensure repositories are initialized
            await self._ensure_repositories_initialized()

            if not self._analytics_repository:
                raise RuntimeError("Analytics repository not available")

            # Create ImprovementSession directly using the database connection
            # Since there's no direct create method in analytics repo, we'll use the base repo functionality
            from prompt_improver.database.models import ImprovementSession

            improvement_session = ImprovementSession(
                session_id=data.session_id,
                original_prompt=data.prompt_text,
                final_prompt=data.enhanced_prompt,
                rules_applied=data.applied_rules,
                user_context={
                    "user_agent": data.user_agent,
                    "rule_id": data.rule_id,
                    "prompt_characteristics": data.prompt_characteristics,
                },
                improvement_metrics={
                    "improvement_score": data.improvement_score,
                    "confidence_level": data.confidence_level,
                    "response_time_ms": data.response_time_ms,
                },
                created_at=data.timestamp,
            )

            # Use analytics repository's connection to create the record
            async with self._analytics_repository.get_session() as session:
                session.add(improvement_session)
                await session.commit()
                await session.refresh(improvement_session)
                return str(improvement_session.id)

        except Exception as e:
            logger.exception(f"Failed to store rule application via repository: {e}")
            raise

    async def _store_user_feedback(self, data: UserFeedbackData) -> str:
        """Store user feedback data using repository layer."""
        try:
            # Ensure repositories are initialized
            await self._ensure_repositories_initialized()

            if not self._feedback_repository:
                raise RuntimeError("Feedback repository not available")

            # Map collector data to UserFeedback model
            feedback_data = UserFeedbackCreate(
                session_id=data.rule_application_id,
                rating=int(data.user_rating),
                feedback_text=data.feedback_text,
                improvement_areas=data.improvement_suggestions,
                applied_rules=[
                    data.rule_application_id
                ],
                is_processed=False,
                ml_optimized=False,
            )

            # Create feedback through repository
            feedback_record = await self._feedback_repository.create_feedback(
                feedback_data
            )
            return str(feedback_record.id)

        except Exception as e:
            logger.exception(f"Failed to store user feedback via repository: {e}")
            raise

    async def _get_rule_applications(
        self, start_time: datetime
    ) -> list[RuleApplicationData]:
        """Get rule applications using repository layer."""
        try:
            # Ensure repositories are initialized
            await self._ensure_repositories_initialized()

            if not self._analytics_repository:
                raise RuntimeError("Analytics repository not available")

            # Use analytics repository to get improvement sessions
            sessions = await self._analytics_repository.get_improvement_sessions(
                start_date=start_time,
                end_date=aware_utc_now(),
                limit=1000,  # Reasonable limit for ML data collection
            )

            # Map sessions to RuleApplicationData
            applications: list[RuleApplicationData] = []
            for session in sessions:
                # Extract metrics from improvement_metrics if available
                metrics = session.improvement_metrics or {}
                context = session.user_context or {}

                applications.append(
                    RuleApplicationData(
                        rule_id=context.get("rule_id", str(session.id)),
                        prompt_text=session.original_prompt,
                        enhanced_prompt=session.final_prompt,
                        improvement_score=metrics.get("improvement_score", 0.8),
                        confidence_level=metrics.get("confidence_level", 0.9),
                        response_time_ms=metrics.get("response_time_ms", 100.0),
                        prompt_characteristics=context.get(
                            "prompt_characteristics", {}
                        ),
                        applied_rules=session.rules_applied or [],
                        user_agent=context.get("user_agent", "unknown"),
                        session_id=session.session_id,
                        timestamp=session.created_at,
                    )
                )

            return applications

        except Exception as e:
            logger.exception(f"Failed to get rule applications via repository: {e}")
            raise

    async def _get_user_feedback(self, start_time: datetime) -> list[UserFeedbackData]:
        """Get user feedback using repository layer."""
        try:
            # Ensure repositories are initialized
            await self._ensure_repositories_initialized()

            if not self._feedback_repository:
                raise RuntimeError("Feedback repository not available")

            # Create filter for feedback retrieval
            feedback_filter = FeedbackFilter(
                date_from=start_time, date_to=aware_utc_now()
            )

            # Get feedback through repository
            feedback_records = await self._feedback_repository.get_feedback_list(
                filters=feedback_filter,
                limit=1000,  # Reasonable limit for ML data collection
                sort_by="created_at",
                sort_desc=True,
            )

            # Map feedback records to UserFeedbackData
            feedback_list: list[UserFeedbackData] = [UserFeedbackData(
                        feedback_id=str(record.id),
                        rule_application_id=record.session_id,
                        user_rating=float(record.rating),
                        effectiveness_score=float(record.rating) / 5.0,
                        satisfaction_score=float(record.rating) / 5.0,
                        feedback_text=record.feedback_text,
                        improvement_suggestions=record.improvement_areas or [],
                        timestamp=record.created_at,
                    ) for record in feedback_records]

            return feedback_list

        except Exception as e:
            logger.exception(f"Failed to get user feedback via repository: {e}")
            raise

    def _calculate_data_quality(
        self, applications: list[RuleApplicationData], feedback: list[UserFeedbackData]
    ) -> dict[str, float]:
        """Calculate data quality metrics."""
        if not applications:
            return {"completeness": 0.0, "feedback_rate": 0.0, "avg_confidence": 0.0}
        complete_applications = sum(
            1
            for app in applications
            if app.prompt_text and app.enhanced_prompt and (app.improvement_score > 0)
        )
        completeness = complete_applications / len(applications)
        feedback_rate = len(feedback) / len(applications) if applications else 0.0
        avg_confidence = sum(app.confidence_level for app in applications) / len(
            applications
        )
        return {
            "completeness": completeness,
            "feedback_rate": feedback_rate,
            "avg_confidence": avg_confidence,
        }

    async def _signal_ml_pipeline(self, ml_package: MLDataPackage) -> None:
        """Signal ML pipeline with new data package.

        This is where MCP hands off data to the existing ML system.
        MCP's responsibility ends here - no ML operations performed.

        For now, we just log the package availability since the ML pipeline
        will read directly from the existing tables.
        """
        try:
            package_id = f"ml_package_{int(time.time())}"
            logger.info(
                f"ML data package {package_id} ready: {len(ml_package.rule_applications)} applications, {len(ml_package.user_feedback)} feedback items, quality: {ml_package.data_quality_metrics}"
            )
        except Exception as e:
            logger.exception(f"Failed to signal ML pipeline: {e}")

    def get_collection_statistics(self) -> dict[str, Any]:
        """Get data collection statistics."""
        return {
            "quality_metrics": self.quality_metrics.copy(),
            "collection_running": self._running,
            "batch_size": self.batch_size,
            "collection_interval_seconds": self.collection_interval_seconds,
            "data_retention_days": self.data_retention_days,
        }
