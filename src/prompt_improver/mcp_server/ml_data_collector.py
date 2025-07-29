"""MCP ML Data Collector.

Collects rule application data and user feedback for the ML pipeline.
Maintains strict architectural separation - MCP only collects and formats data,
never performs ML operations directly.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from prompt_improver.utils.datetime_utils import aware_utc_now
from prompt_improver.database.utils import fetch_all_rows

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
    prompt_characteristics: Dict[str, Any]
    applied_rules: List[str]
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
    feedback_text: Optional[str]
    improvement_suggestions: List[str]
    timestamp: datetime

@dataclass
class MLDataPackage:
    """Complete data package for ML pipeline consumption."""
    rule_applications: List[RuleApplicationData]
    user_feedback: List[UserFeedbackData]
    collection_period: Dict[str, datetime]
    data_quality_metrics: Dict[str, float]
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

    def __init__(self, db_session: AsyncSession):
        """Initialize MCP ML data collector.

        Args:
            db_session: Database session for data storage
        """
        self.db_session = db_session

        # Data collection configuration
        self.batch_size = 100
        self.collection_interval_seconds = 300  # 5 minutes
        self.data_retention_days = 90

        # Data quality tracking
        self.quality_metrics: Dict[str, float] = {
            "total_collected": 0.0,
            "successful_collections": 0.0,
            "failed_collections": 0.0,
            "data_completeness_rate": 0.0,
            "collection_latency_ms": 0.0
        }

        # Background collection task
        self._collection_task: Optional[asyncio.Task[None]] = None
        self._running = False

    async def start_collection(self) -> None:
        """Start background data collection for ML pipeline."""
        if self._running:
            logger.warning("Data collection already running")
            return

        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started MCP ML data collection")

    async def stop_collection(self) -> None:
        """Stop background data collection."""
        self._running = False

        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped MCP ML data collection")

    async def collect_rule_application(
        self,
        rule_id: str,
        prompt_text: str,
        enhanced_prompt: str,
        improvement_score: float,
        confidence_level: float,
        response_time_ms: float,
        prompt_characteristics: Dict[str, Any],
        applied_rules: List[str],
        user_agent: str,
        session_id: str
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
            # Create rule application data
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
                timestamp=aware_utc_now()
            )

            # Store in database for ML pipeline consumption
            application_id = await self._store_rule_application(application_data)

            # Update quality metrics
            self.quality_metrics["total_collected"] += 1
            self.quality_metrics["successful_collections"] += 1

            return application_id

        except Exception as e:
            logger.error(f"Failed to collect rule application data: {e}")
            self.quality_metrics["failed_collections"] += 1
            raise

    async def collect_user_feedback(
        self,
        rule_application_id: str,
        user_rating: float,
        effectiveness_score: float,
        satisfaction_score: float,
        feedback_text: Optional[str] = None,
        improvement_suggestions: Optional[List[str]] = None
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
            # Create feedback data
            feedback_data = UserFeedbackData(
                feedback_id=f"feedback_{int(time.time() * 1000)}",
                rule_application_id=rule_application_id,
                user_rating=user_rating,
                effectiveness_score=effectiveness_score,
                satisfaction_score=satisfaction_score,
                feedback_text=feedback_text,
                improvement_suggestions=improvement_suggestions or [],
                timestamp=aware_utc_now()
            )

            # Store in database for ML pipeline consumption
            feedback_id = await self._store_user_feedback(feedback_data)

            return feedback_id

        except Exception as e:
            logger.error(f"Failed to collect user feedback: {e}")
            raise

    async def prepare_ml_data_package(
        self,
        hours_back: int = 24,
        min_samples: int = 10
    ) -> Optional[MLDataPackage]:
        """Prepare data package for ML pipeline consumption.

        Args:
            hours_back: Hours of data to include
            min_samples: Minimum samples required

        Returns:
            MLDataPackage for ML pipeline or None if insufficient data
        """
        try:
            start_time = aware_utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = start_time.replace(hour=start_time.hour - hours_back)

            # Collect rule applications
            rule_applications = await self._get_rule_applications(start_time)

            # Collect user feedback
            user_feedback = await self._get_user_feedback(start_time)

            if len(rule_applications) < min_samples:
                logger.info(f"Insufficient data for ML package: {len(rule_applications)} < {min_samples}")
                return None

            # Calculate data quality metrics
            data_quality = self._calculate_data_quality(rule_applications, user_feedback)

            # Create ML data package
            ml_package = MLDataPackage(
                rule_applications=rule_applications,
                user_feedback=user_feedback,
                collection_period={
                    "start": start_time,
                    "end": aware_utc_now()
                },
                data_quality_metrics=data_quality,
                total_samples=len(rule_applications)
            )

            logger.info(f"Prepared ML data package with {len(rule_applications)} applications and {len(user_feedback)} feedback items")
            return ml_package

        except Exception as e:
            logger.error(f"Failed to prepare ML data package: {e}")
            return None

    async def _collection_loop(self) -> None:
        """Background collection loop for periodic ML data preparation."""
        while self._running:
            try:
                # Prepare data package for ML pipeline
                ml_package = await self.prepare_ml_data_package()

                if ml_package:
                    # Signal ML pipeline (this would trigger existing ML system)
                    await self._signal_ml_pipeline(ml_package)

                # Wait for next collection interval
                await asyncio.sleep(self.collection_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _store_rule_application(self, data: RuleApplicationData) -> str:
        """Store rule application data in database using existing schema."""
        application_id = f"app_{int(time.time() * 1000)}"

        # Use prompt_improvement_sessions table which exists for MCP feedback collection
        query = text("""
            INSERT INTO prompt_improvement_sessions (
                original_prompt, enhanced_prompt, applied_rules,
                response_time_ms, agent_type, session_timestamp,
                anonymized_user_hash, created_at
            ) VALUES (
                :original_prompt, :enhanced_prompt, :applied_rules,
                :response_time_ms, :agent_type, :session_timestamp,
                :anonymized_user_hash, :created_at
            ) RETURNING id
        """)

        result = await self.db_session.execute(query, {
            "original_prompt": data.prompt_text,
            "enhanced_prompt": data.enhanced_prompt,
            "applied_rules": data.applied_rules,
            "response_time_ms": int(data.response_time_ms),
            "agent_type": data.user_agent,
            "session_timestamp": data.timestamp,
            "anonymized_user_hash": data.session_id,  # Use session_id as hash
            "created_at": data.timestamp
        })

        await self.db_session.commit()
        row = result.first()
        return str(row[0]) if row else application_id

    async def _store_user_feedback(self, data: UserFeedbackData) -> str:
        """Store user feedback data in database using existing schema."""
        # Use user_feedback table which exists in the schema
        query = text("""
            INSERT INTO user_feedback (
                session_id, user_rating, feedback_text,
                improvement_areas, applied_rules, created_at
            ) VALUES (
                :session_id, :user_rating, :feedback_text,
                :improvement_areas, :applied_rules, :created_at
            ) RETURNING id
        """)

        result = await self.db_session.execute(query, {
            "session_id": data.rule_application_id,
            "user_rating": int(data.user_rating),
            "feedback_text": data.feedback_text,
            "improvement_areas": data.improvement_suggestions,
            "applied_rules": [data.rule_application_id],  # Store as array
            "created_at": data.timestamp
        })

        await self.db_session.commit()
        row = result.first()
        return str(row[0]) if row else data.feedback_id

    async def _get_rule_applications(self, start_time: datetime) -> List[RuleApplicationData]:
        """Get rule applications from database using existing schema."""
        query = text("""
            SELECT id, original_prompt, enhanced_prompt, applied_rules,
                   response_time_ms, agent_type, anonymized_user_hash,
                   session_timestamp, created_at
            FROM prompt_improvement_sessions
            WHERE created_at >= :start_time
            ORDER BY created_at DESC
        """)

        rows = await fetch_all_rows(self.db_session, query, {"start_time": start_time})

        applications: List[RuleApplicationData] = []
        for row in rows:
            # Use proper row attribute access instead of protected _mapping
            applications.append(RuleApplicationData(
                rule_id=str(row.id),  # Use session ID as rule ID
                prompt_text=row.original_prompt,
                enhanced_prompt=row.enhanced_prompt,
                improvement_score=0.8,  # Default score since not stored
                confidence_level=0.9,   # Default confidence since not stored
                response_time_ms=float(row.response_time_ms),
                prompt_characteristics={},  # Empty since not stored in this table
                applied_rules=row.applied_rules if row.applied_rules else [],
                user_agent=row.agent_type or "unknown",
                session_id=row.anonymized_user_hash or str(row.id),
                timestamp=row.created_at
            ))

        return applications

    async def _get_user_feedback(self, start_time: datetime) -> List[UserFeedbackData]:
        """Get user feedback from database using existing schema."""
        query = text("""
            SELECT id, session_id, user_rating, feedback_text,
                   improvement_areas, applied_rules, created_at
            FROM user_feedback
            WHERE created_at >= :start_time
            ORDER BY created_at DESC
        """)

        rows = await fetch_all_rows(self.db_session, query, {"start_time": start_time})

        feedback_list: List[UserFeedbackData] = []
        for row in rows:
            # Use proper row attribute access instead of protected _mapping
            feedback_list.append(UserFeedbackData(
                feedback_id=str(row.id),
                rule_application_id=row.session_id,
                user_rating=float(row.user_rating),
                effectiveness_score=float(row.user_rating) / 5.0,  # Convert 1-5 to 0-1 scale
                satisfaction_score=float(row.user_rating) / 5.0,   # Convert 1-5 to 0-1 scale
                feedback_text=row.feedback_text,
                improvement_suggestions=row.improvement_areas if row.improvement_areas else [],
                timestamp=row.created_at
            ))

        return feedback_list

    def _calculate_data_quality(
        self,
        applications: List[RuleApplicationData],
        feedback: List[UserFeedbackData]
    ) -> Dict[str, float]:
        """Calculate data quality metrics."""
        if not applications:
            return {"completeness": 0.0, "feedback_rate": 0.0, "avg_confidence": 0.0}

        # Completeness (non-null required fields)
        complete_applications = sum(
            1 for app in applications
            if app.prompt_text and app.enhanced_prompt and app.improvement_score > 0
        )
        completeness = complete_applications / len(applications)

        # Feedback rate
        feedback_rate = len(feedback) / len(applications) if applications else 0.0

        # Average confidence
        avg_confidence = sum(app.confidence_level for app in applications) / len(applications)

        return {
            "completeness": completeness,
            "feedback_rate": feedback_rate,
            "avg_confidence": avg_confidence
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
                f"ML data package {package_id} ready: "
                f"{len(ml_package.rule_applications)} applications, "
                f"{len(ml_package.user_feedback)} feedback items, "
                f"quality: {ml_package.data_quality_metrics}"
            )

            # The ML pipeline can read directly from prompt_improvement_sessions
            # and user_feedback tables using the time range in ml_package.collection_period

        except Exception as e:
            logger.error(f"Failed to signal ML pipeline: {e}")

    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get data collection statistics."""
        return {
            "quality_metrics": self.quality_metrics.copy(),
            "collection_running": self._running,
            "batch_size": self.batch_size,
            "collection_interval_seconds": self.collection_interval_seconds,
            "data_retention_days": self.data_retention_days
        }
