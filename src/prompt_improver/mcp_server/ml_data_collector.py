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
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from prompt_improver.database.utils import fetch_all_rows
from prompt_improver.performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager
from prompt_improver.utils.datetime_utils import aware_utc_now
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

    def __init__(self):
        """Initialize MCP ML data collector.

        Uses unified session factory from database package for all database operations.
        """
        self.batch_size = 100
        self.collection_interval_seconds = 300
        self.data_retention_days = 90
        self.quality_metrics: dict[str, float] = {'total_collected': 0.0, 'successful_collections': 0.0, 'failed_collections': 0.0, 'data_completeness_rate': 0.0, 'collection_latency_ms': 0.0}
        self._collection_task_id: str | None = None
        self._running = False

    async def start_collection(self) -> None:
        """Start background data collection for ML pipeline."""
        if self._running:
            logger.warning('Data collection already running')
            return
        self._running = True
        task_manager = get_background_task_manager()
        task_id = await task_manager.submit_enhanced_task(task_id=f'ml_data_collection_{id(self)}', coroutine=self._collection_loop, priority=TaskPriority.HIGH, tags={'service': 'ml_data_collector', 'type': 'collection_loop', 'component': 'mcp_server'})
        self._collection_task_id = task_id
        logger.info('Started MCP ML data collection with enhanced task management')

    async def stop_collection(self) -> None:
        """Stop background data collection."""
        self._running = False
        if self._collection_task_id:
            task_manager = get_background_task_manager()
            await task_manager.cancel_task(self._collection_task_id)
            self._collection_task_id = None
        logger.info('Stopped MCP ML data collection')

    async def collect_rule_application(self, rule_id: str, prompt_text: str, enhanced_prompt: str, improvement_score: float, confidence_level: float, response_time_ms: float, prompt_characteristics: dict[str, Any], applied_rules: list[str], user_agent: str, session_id: str) -> str:
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
            application_data = RuleApplicationData(rule_id=rule_id, prompt_text=prompt_text, enhanced_prompt=enhanced_prompt, improvement_score=improvement_score, confidence_level=confidence_level, response_time_ms=response_time_ms, prompt_characteristics=prompt_characteristics, applied_rules=applied_rules, user_agent=user_agent, session_id=session_id, timestamp=aware_utc_now())
            application_id = await self._store_rule_application(application_data)
            self.quality_metrics['total_collected'] += 1
            self.quality_metrics['successful_collections'] += 1
            return application_id
        except Exception as e:
            logger.error('Failed to collect rule application data: %s', e)
            self.quality_metrics['failed_collections'] += 1
            raise

    async def collect_user_feedback(self, rule_application_id: str, user_rating: float, effectiveness_score: float, satisfaction_score: float, feedback_text: str | None=None, improvement_suggestions: list[str] | None=None) -> str:
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
            feedback_data = UserFeedbackData(feedback_id=f'feedback_{int(time.time() * 1000)}', rule_application_id=rule_application_id, user_rating=user_rating, effectiveness_score=effectiveness_score, satisfaction_score=satisfaction_score, feedback_text=feedback_text, improvement_suggestions=improvement_suggestions or [], timestamp=aware_utc_now())
            feedback_id = await self._store_user_feedback(feedback_data)
            return feedback_id
        except Exception as e:
            logger.error('Failed to collect user feedback: %s', e)
            raise

    async def prepare_ml_data_package(self, hours_back: int=24, min_samples: int=10) -> MLDataPackage | None:
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
            rule_applications = await self._get_rule_applications(start_time)
            user_feedback = await self._get_user_feedback(start_time)
            if len(rule_applications) < min_samples:
                logger.info('Insufficient data for ML package: %s < %s', len(rule_applications), min_samples)
                return None
            data_quality = self._calculate_data_quality(rule_applications, user_feedback)
            ml_package = MLDataPackage(rule_applications=rule_applications, user_feedback=user_feedback, collection_period={'start': start_time, 'end': aware_utc_now()}, data_quality_metrics=data_quality, total_samples=len(rule_applications))
            logger.info('Prepared ML data package with %s applications and %s feedback items', len(rule_applications), len(user_feedback))
            return ml_package
        except Exception as e:
            logger.error('Failed to prepare ML data package: %s', e)
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
                logger.error('Error in collection loop: %s', e)
                await asyncio.sleep(60)

    async def _store_rule_application(self, data: RuleApplicationData) -> str:
        """Store rule application data in database using existing schema."""
        application_id = f'app_{int(time.time() * 1000)}'
        query = text('\n            INSERT INTO prompt_improvement_sessions (\n                original_prompt, enhanced_prompt, applied_rules,\n                response_time_ms, agent_type, session_timestamp,\n                anonymized_user_hash, created_at\n            ) VALUES (\n                :original_prompt, :enhanced_prompt, :applied_rules,\n                :response_time_ms, :agent_type, :session_timestamp,\n                :anonymized_user_hash, :created_at\n            ) RETURNING id\n        ')
        from prompt_improver.database import get_session_context
        async with get_session_context() as session:
            result = await session.execute(query, {'original_prompt': data.prompt_text, 'enhanced_prompt': data.enhanced_prompt, 'applied_rules': data.applied_rules, 'response_time_ms': int(data.response_time_ms), 'agent_type': data.user_agent, 'session_timestamp': data.timestamp, 'anonymized_user_hash': data.session_id, 'created_at': data.timestamp})
            await session.commit()
            row = result.first()
            return str(row[0]) if row else application_id

    async def _store_user_feedback(self, data: UserFeedbackData) -> str:
        """Store user feedback data in database using existing schema."""
        query = text('\n            INSERT INTO user_feedback (\n                session_id, user_rating, feedback_text,\n                improvement_areas, applied_rules, created_at\n            ) VALUES (\n                :session_id, :user_rating, :feedback_text,\n                :improvement_areas, :applied_rules, :created_at\n            ) RETURNING id\n        ')
        from prompt_improver.database import get_session_context
        async with get_session_context() as session:
            result = await session.execute(query, {'session_id': data.rule_application_id, 'user_rating': int(data.user_rating), 'feedback_text': data.feedback_text, 'improvement_areas': data.improvement_suggestions, 'applied_rules': [data.rule_application_id], 'created_at': data.timestamp})
            await session.commit()
            row = result.first()
            return str(row[0]) if row else data.feedback_id

    async def _get_rule_applications(self, start_time: datetime) -> list[RuleApplicationData]:
        """Get rule applications from database using existing schema."""
        query = text('\n            SELECT id, original_prompt, enhanced_prompt, applied_rules,\n                   response_time_ms, agent_type, anonymized_user_hash,\n                   session_timestamp, created_at\n            FROM prompt_improvement_sessions\n            WHERE created_at >= :start_time\n            ORDER BY created_at DESC\n        ')
        from prompt_improver.database import get_session_context
        from prompt_improver.database.utils import fetch_all_rows
        async with get_session_context() as session:
            rows = await fetch_all_rows(session, query, {'start_time': start_time})
        applications: list[RuleApplicationData] = []
        for row in rows:
            applications.append(RuleApplicationData(rule_id=str(row.id), prompt_text=row.original_prompt, enhanced_prompt=row.enhanced_prompt, improvement_score=0.8, confidence_level=0.9, response_time_ms=float(row.response_time_ms), prompt_characteristics={}, applied_rules=row.applied_rules if row.applied_rules else [], user_agent=row.agent_type or 'unknown', session_id=row.anonymized_user_hash or str(row.id), timestamp=row.created_at))
        return applications

    async def _get_user_feedback(self, start_time: datetime) -> list[UserFeedbackData]:
        """Get user feedback from database using existing schema."""
        query = text('\n            SELECT id, session_id, user_rating, feedback_text,\n                   improvement_areas, applied_rules, created_at\n            FROM user_feedback\n            WHERE created_at >= :start_time\n            ORDER BY created_at DESC\n        ')
        from prompt_improver.database import get_session_context
        async with get_session_context() as session:
            rows = await fetch_all_rows(session, query, {'start_time': start_time})
        feedback_list: list[UserFeedbackData] = []
        for row in rows:
            feedback_list.append(UserFeedbackData(feedback_id=str(row.id), rule_application_id=row.session_id, user_rating=float(row.user_rating), effectiveness_score=float(row.user_rating) / 5.0, satisfaction_score=float(row.user_rating) / 5.0, feedback_text=row.feedback_text, improvement_suggestions=row.improvement_areas if row.improvement_areas else [], timestamp=row.created_at))
        return feedback_list

    def _calculate_data_quality(self, applications: list[RuleApplicationData], feedback: list[UserFeedbackData]) -> dict[str, float]:
        """Calculate data quality metrics."""
        if not applications:
            return {'completeness': 0.0, 'feedback_rate': 0.0, 'avg_confidence': 0.0}
        complete_applications = sum((1 for app in applications if app.prompt_text and app.enhanced_prompt and (app.improvement_score > 0)))
        completeness = complete_applications / len(applications)
        feedback_rate = len(feedback) / len(applications) if applications else 0.0
        avg_confidence = sum((app.confidence_level for app in applications)) / len(applications)
        return {'completeness': completeness, 'feedback_rate': feedback_rate, 'avg_confidence': avg_confidence}

    async def _signal_ml_pipeline(self, ml_package: MLDataPackage) -> None:
        """Signal ML pipeline with new data package.

        This is where MCP hands off data to the existing ML system.
        MCP's responsibility ends here - no ML operations performed.

        For now, we just log the package availability since the ML pipeline
        will read directly from the existing tables.
        """
        try:
            package_id = f'ml_package_{int(time.time())}'
            logger.info('ML data package %s ready: %s applications, %s feedback items, quality: %s', package_id, len(ml_package.rule_applications), len(ml_package.user_feedback), ml_package.data_quality_metrics)
        except Exception as e:
            logger.error('Failed to signal ML pipeline: %s', e)

    def get_collection_statistics(self) -> dict[str, Any]:
        """Get data collection statistics."""
        return {'quality_metrics': self.quality_metrics.copy(), 'collection_running': self._running, 'batch_size': self.batch_size, 'collection_interval_seconds': self.collection_interval_seconds, 'data_retention_days': self.data_retention_days}
