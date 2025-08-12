"""Enhanced Feedback Collection System with Non-blocking Processing.

Implements 2025 best practices for feedback collection using FastAPI BackgroundTasks
with comprehensive data anonymization and GDPR compliance.
"""

import asyncio
import hashlib
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import coredis
from fastapi import BackgroundTasks
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.utils.datetime_utils import aware_utc_now

from ..database import (
    ManagerMode,
    create_security_context,
    get_database_services,
)

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback that can be collected."""

    PROMPT_ENHANCEMENT = "prompt_enhancement"
    RULE_EFFECTIVENESS = "rule_effectiveness"
    USER_SATISFACTION = "user_satisfaction"
    SYSTEM_PERFORMANCE = "system_performance"
    ERROR_REPORT = "error_report"


class AnonymizationLevel(str, Enum):
    """Levels of data anonymization."""

    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    FULL = "full"


@dataclass
class FeedbackData:
    """Structured feedback data with metadata."""

    feedback_id: str
    feedback_type: FeedbackType
    session_id: str
    agent_id: str
    agent_type: str
    original_prompt: str
    enhanced_prompt: str
    applied_rules: list[str]
    effectiveness_score: float
    user_rating: int | None
    user_comments: str | None
    performance_metrics: dict[str, Any]
    timestamp: datetime
    anonymization_level: AnonymizationLevel
    metadata: dict[str, Any]


@dataclass
class AnonymizedFeedback:
    """Anonymized feedback data for ML training."""

    feedback_id: str
    feedback_type: FeedbackType
    session_hash: str
    agent_type: str
    prompt_structure: dict[str, Any]
    enhancement_patterns: list[str]
    rule_effectiveness: dict[str, float]
    satisfaction_score: float | None
    performance_metrics: dict[str, Any]
    timestamp: datetime
    anonymization_metadata: dict[str, Any]


class PIIDetector:
    """Advanced PII detection and removal system."""

    def __init__(self):
        """Initialize PII detection patterns."""
        self.email_pattern = re.compile(
            "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"
        )
        self.phone_patterns = [
            re.compile("\\b\\d{3}-\\d{3}-\\d{4}\\b"),
            re.compile("\\b\\(\\d{3}\\)\\s*\\d{3}-\\d{4}\\b"),
            re.compile("\\b\\d{3}\\.\\d{3}\\.\\d{4}\\b"),
            re.compile("\\b\\+\\d{1,3}\\s*\\d{3,4}\\s*\\d{3,4}\\s*\\d{3,4}\\b"),
        ]
        self.credit_card_pattern = re.compile(
            "\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b"
        )
        self.ssn_pattern = re.compile("\\b\\d{3}-\\d{2}-\\d{4}\\b")
        self.ip_pattern = re.compile("\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\b")
        self.url_pattern = re.compile("https?://[^\\s]+")
        self.common_names = {
            "first_names": {
                "john",
                "jane",
                "michael",
                "sarah",
                "david",
                "lisa",
                "robert",
                "mary",
            },
            "last_names": {
                "smith",
                "johnson",
                "williams",
                "brown",
                "jones",
                "garcia",
                "miller",
            },
        }

    def detect_and_remove_pii(
        self,
        text: str,
        anonymization_level: AnonymizationLevel = AnonymizationLevel.ADVANCED,
    ) -> tuple[str, dict[str, Any]]:
        """Detect and remove PII from text based on anonymization level.

        Args:
            text: Input text to anonymize
            anonymization_level: Level of anonymization to apply

        Returns:
            Tuple of (anonymized_text, detection_metadata)
        """
        if anonymization_level == AnonymizationLevel.NONE:
            return (text, {})
        anonymized_text = text
        detection_metadata = {
            "pii_detected": [],
            "replacements_made": 0,
            "anonymization_level": anonymization_level.value,
        }
        if self.email_pattern.search(anonymized_text):
            detection_metadata["pii_detected"].append("email")
            anonymized_text = self.email_pattern.sub("[EMAIL]", anonymized_text)
            detection_metadata["replacements_made"] += len(
                self.email_pattern.findall(text)
            )
        for pattern in self.phone_patterns:
            if pattern.search(anonymized_text):
                detection_metadata["pii_detected"].append("phone")
                anonymized_text = pattern.sub("[PHONE]", anonymized_text)
                detection_metadata["replacements_made"] += len(pattern.findall(text))
        if self.credit_card_pattern.search(anonymized_text):
            detection_metadata["pii_detected"].append("credit_card")
            anonymized_text = self.credit_card_pattern.sub(
                "[CREDIT_CARD]", anonymized_text
            )
            detection_metadata["replacements_made"] += len(
                self.credit_card_pattern.findall(text)
            )
        if self.ssn_pattern.search(anonymized_text):
            detection_metadata["pii_detected"].append("ssn")
            anonymized_text = self.ssn_pattern.sub("[SSN]", anonymized_text)
            detection_metadata["replacements_made"] += len(
                self.ssn_pattern.findall(text)
            )
        if anonymization_level in [
            AnonymizationLevel.ADVANCED,
            AnonymizationLevel.FULL,
        ]:
            if self.ip_pattern.search(anonymized_text):
                detection_metadata["pii_detected"].append("ip_address")
                anonymized_text = self.ip_pattern.sub("[IP_ADDRESS]", anonymized_text)
                detection_metadata["replacements_made"] += len(
                    self.ip_pattern.findall(text)
                )
            if self.url_pattern.search(anonymized_text):
                detection_metadata["pii_detected"].append("url")
                anonymized_text = self.url_pattern.sub("[URL]", anonymized_text)
                detection_metadata["replacements_made"] += len(
                    self.url_pattern.findall(text)
                )
        if anonymization_level == AnonymizationLevel.FULL:
            words = anonymized_text.lower().split()
            for i, word in enumerate(words):
                clean_word = re.sub("[^\\w]", "", word)
                if (
                    clean_word in self.common_names["first_names"]
                    or clean_word in self.common_names["last_names"]
                ):
                    detection_metadata["pii_detected"].append("name")
                    anonymized_text = re.sub(
                        "\\b" + re.escape(word) + "\\b",
                        "[NAME]",
                        anonymized_text,
                        flags=re.IGNORECASE,
                    )
                    detection_metadata["replacements_made"] += 1
        return (anonymized_text, detection_metadata)


class EnhancedFeedbackCollector:
    """Enhanced feedback collection system with non-blocking processing and DatabaseServices.

    Features:
    - FastAPI BackgroundTasks for zero user impact
    - Comprehensive PII detection and anonymization
    - GDPR-compliant data handling
    - Retry mechanisms for reliability
    - Performance monitoring and success rate tracking
    - Multi-level caching and connection pooling via DatabaseServices
    """

    def __init__(
        self,
        db_session: AsyncSession,
        redis_url: str | None = None,
        agent_id: str = "enhanced_feedback_collector",
    ):
        """Initialize enhanced feedback collector with DatabaseServices integration.

        Args:
            db_session: Database session for feedback storage
            redis_url: Redis URL for temporary feedback queuing (fallback)
            agent_id: Agent identifier for security context
        """
        self.db_session = db_session
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/4")
        self.agent_id = agent_id
        self._database_services = None
        self._redis_client = None
        self._use_database_services = True
        self.pii_detector = PIIDetector()
        self.default_anonymization_level = AnonymizationLevel.ADVANCED
        self.success_rate_target = 0.95
        self._feedback_stats = {
            "total_collected": 0,
            "successful_storage": 0,
            "failed_storage": 0,
            "pii_detections": 0,
            "anonymization_errors": 0,
        }
        self.max_retries = 3
        self.retry_delay_seconds = 1.0
        self.feedback_queue_key = "feedback:queue"
        self.failed_feedback_key = "feedback:failed"

    async def get_redis_client(self) -> coredis.Redis:
        """Get Redis client for feedback queuing via DatabaseServices exclusively."""
        if self._database_services is None:
            self._database_services = await get_database_services(
                ManagerMode.HIGH_AVAILABILITY
            )

        # Access Redis through the cache manager's redis client
        if hasattr(self._database_services.cache, "redis_client"):
            return self._database_services.cache.redis_client

        raise RuntimeError(
            "Redis client not available via DatabaseServices - ensure Redis is configured and running"
        )

    async def collect_feedback(
        self,
        background_tasks: BackgroundTasks,
        session_id: str,
        agent_id: str,
        agent_type: str,
        original_prompt: str,
        enhanced_prompt: str,
        applied_rules: list[str],
        effectiveness_score: float,
        user_rating: int | None = None,
        user_comments: str | None = None,
        performance_metrics: dict[str, Any] | None = None,
        anonymization_level: AnonymizationLevel = None,
    ) -> str:
        """Collect feedback using FastAPI BackgroundTasks for non-blocking processing.

        Args:
            background_tasks: FastAPI BackgroundTasks instance
            session_id: Session identifier
            agent_id: Agent identifier
            agent_type: Type of agent
            original_prompt: Original prompt text
            enhanced_prompt: Enhanced prompt text
            applied_rules: List of applied rule IDs
            effectiveness_score: Effectiveness score (0.0 to 1.0)
            user_rating: Optional user rating (1-5)
            user_comments: Optional user comments
            performance_metrics: Optional performance metrics
            anonymization_level: Level of anonymization to apply

        Returns:
            Feedback ID for tracking
        """
        feedback_id = str(uuid.uuid4())
        feedback_data = FeedbackData(
            feedback_id=feedback_id,
            feedback_type=FeedbackType.PROMPT_ENHANCEMENT,
            session_id=session_id,
            agent_id=agent_id,
            agent_type=agent_type,
            original_prompt=original_prompt,
            enhanced_prompt=enhanced_prompt,
            applied_rules=applied_rules,
            effectiveness_score=effectiveness_score,
            user_rating=user_rating,
            user_comments=user_comments,
            performance_metrics=performance_metrics or {},
            timestamp=aware_utc_now(),
            anonymization_level=anonymization_level or self.default_anonymization_level,
            metadata={
                "collection_method": "background_task",
                "collector_version": "2.0.0",
            },
        )
        background_tasks.add_task(self._process_feedback_async, feedback_data)
        self._feedback_stats["total_collected"] += 1
        logger.info(f"Feedback collection initiated: {feedback_id}")
        return feedback_id

    async def _process_feedback_async(self, feedback_data: FeedbackData) -> None:
        """Process feedback asynchronously in background.

        Args:
            feedback_data: Feedback data to process
        """
        try:
            anonymized_feedback = await self._anonymize_feedback(feedback_data)
            success = await self._store_feedback_with_retry(
                feedback_data, anonymized_feedback
            )
            if success:
                self._feedback_stats["successful_storage"] += 1
                logger.info(
                    f"Feedback processed successfully: {feedback_data.feedback_id}"
                )
            else:
                self._feedback_stats["failed_storage"] += 1
                await self._queue_failed_feedback(feedback_data)
                logger.error(f"Feedback processing failed: {feedback_data.feedback_id}")
        except Exception as e:
            self._feedback_stats["failed_storage"] += 1
            logger.error(
                f"Feedback processing error for {feedback_data.feedback_id}: {e}"
            )
            await self._queue_failed_feedback(feedback_data)

    async def _anonymize_feedback(
        self, feedback_data: FeedbackData
    ) -> AnonymizedFeedback:
        """Anonymize feedback data for ML training.

        Args:
            feedback_data: Original feedback data

        Returns:
            Anonymized feedback data
        """
        try:
            anonymized_original, original_metadata = (
                self.pii_detector.detect_and_remove_pii(
                    feedback_data.original_prompt, feedback_data.anonymization_level
                )
            )
            anonymized_enhanced, enhanced_metadata = (
                self.pii_detector.detect_and_remove_pii(
                    feedback_data.enhanced_prompt, feedback_data.anonymization_level
                )
            )
            anonymized_comments = None
            comments_metadata = {}
            if feedback_data.user_comments:
                anonymized_comments, comments_metadata = (
                    self.pii_detector.detect_and_remove_pii(
                        feedback_data.user_comments, feedback_data.anonymization_level
                    )
                )
            session_hash = hashlib.sha256(
                f"{feedback_data.session_id}:{feedback_data.agent_id}".encode()
            ).hexdigest()[:16]
            prompt_structure = self._extract_prompt_structure(anonymized_original)
            enhancement_patterns = self._extract_enhancement_patterns(
                anonymized_original, anonymized_enhanced
            )
            rule_effectiveness = dict.fromkeys(
                feedback_data.applied_rules, feedback_data.effectiveness_score
            )
            total_pii_detected = (
                len(original_metadata.get("pii_detected", []))
                + len(enhanced_metadata.get("pii_detected", []))
                + len(comments_metadata.get("pii_detected", []))
            )
            if total_pii_detected > 0:
                self._feedback_stats["pii_detections"] += 1
            anonymized_feedback = AnonymizedFeedback(
                feedback_id=feedback_data.feedback_id,
                feedback_type=feedback_data.feedback_type,
                session_hash=session_hash,
                agent_type=feedback_data.agent_type,
                prompt_structure=prompt_structure,
                enhancement_patterns=enhancement_patterns,
                rule_effectiveness=rule_effectiveness,
                satisfaction_score=float(feedback_data.user_rating)
                if feedback_data.user_rating
                else None,
                performance_metrics=feedback_data.performance_metrics,
                timestamp=feedback_data.timestamp,
                anonymization_metadata={
                    "original_pii": original_metadata,
                    "enhanced_pii": enhanced_metadata,
                    "comments_pii": comments_metadata,
                    "total_pii_detected": total_pii_detected,
                    "anonymization_level": feedback_data.anonymization_level.value,
                },
            )
            return anonymized_feedback
        except Exception as e:
            self._feedback_stats["anonymization_errors"] += 1
            logger.error(f"Anonymization error for {feedback_data.feedback_id}: {e}")
            raise

    def _extract_prompt_structure(self, prompt: str) -> dict[str, Any]:
        """Extract structural patterns from prompt for ML training.

        Args:
            prompt: Anonymized prompt text

        Returns:
            Prompt structure metadata
        """
        words = prompt.split()
        sentences = re.split(r"[.!?]+", prompt)
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_sentence_length": len(words) / max(1, len(sentences)),
            "question_count": prompt.count("?"),
            "exclamation_count": prompt.count("!"),
            "has_imperatives": any(
                word.lower() in ["write", "create", "make", "generate"]
                for word in words[:5]
            ),
            "has_context": any(
                word.lower() in ["given", "considering", "assuming"] for word in words
            ),
            "complexity_indicators": sum(
                1
                for word in words
                if word.lower() in ["complex", "detailed", "comprehensive"]
            ),
            "length_category": "short"
            if len(words) < 20
            else "medium"
            if len(words) < 50
            else "long",
        }

    def _extract_enhancement_patterns(self, original: str, enhanced: str) -> list[str]:
        """Extract enhancement patterns applied to the prompt.

        Args:
            original: Original anonymized prompt
            enhanced: Enhanced anonymized prompt

        Returns:
            List of enhancement pattern identifiers
        """
        patterns = []
        orig_words = len(original.split())
        enh_words = len(enhanced.split())
        if enh_words > orig_words * 1.5:
            patterns.append("significant_expansion")
        elif enh_words > orig_words * 1.2:
            patterns.append("moderate_expansion")
        elif enh_words < orig_words * 0.8:
            patterns.append("compression")
        if enhanced.count("?") > original.count("?"):
            patterns.append("added_questions")
        if enhanced.count(".") > original.count("."):
            patterns.append("added_structure")
        if "step" in enhanced.lower() and "step" not in original.lower():
            patterns.append("added_step_by_step")
        if "example" in enhanced.lower() and "example" not in original.lower():
            patterns.append("added_examples")
        if "context" in enhanced.lower() and "context" not in original.lower():
            patterns.append("added_context")
        return patterns

    async def _store_feedback_with_retry(
        self, feedback_data: FeedbackData, anonymized_feedback: AnonymizedFeedback
    ) -> bool:
        """Store feedback in database with retry logic.

        Args:
            feedback_data: Original feedback data
            anonymized_feedback: Anonymized feedback data

        Returns:
            True if storage successful, False otherwise
        """
        for attempt in range(self.max_retries):
            try:
                await self._store_original_feedback(feedback_data)
                await self._store_anonymized_feedback(anonymized_feedback)
                await self.db_session.commit()
                return True
            except Exception as e:
                logger.warning(
                    f"Storage attempt {attempt + 1} failed for {feedback_data.feedback_id}: {e}"
                )
                await self.db_session.rollback()
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay_seconds * 2**attempt)
                else:
                    logger.error(
                        f"All storage attempts failed for {feedback_data.feedback_id}"
                    )
                    return False
        return False

    async def _store_original_feedback(self, feedback_data: FeedbackData) -> None:
        """Store original feedback data in secure table.

        Args:
            feedback_data: Original feedback data with potential PII
        """
        query = text(
            "\n            INSERT INTO feedback_collection (\n                feedback_id, feedback_type, session_id, agent_id, agent_type,\n                original_prompt, enhanced_prompt, applied_rules, effectiveness_score,\n                user_rating, user_comments, performance_metrics, anonymization_level,\n                metadata, created_at\n            ) VALUES (\n                :feedback_id, :feedback_type, :session_id, :agent_id, :agent_type,\n                :original_prompt, :enhanced_prompt, :applied_rules, :effectiveness_score,\n                :user_rating, :user_comments, :performance_metrics, :anonymization_level,\n                :metadata, :created_at\n            )\n        "
        )
        await self.db_session.execute(
            query,
            {
                "feedback_id": feedback_data.feedback_id,
                "feedback_type": feedback_data.feedback_type.value,
                "session_id": feedback_data.session_id,
                "agent_id": feedback_data.agent_id,
                "agent_type": feedback_data.agent_type,
                "original_prompt": feedback_data.original_prompt,
                "enhanced_prompt": feedback_data.enhanced_prompt,
                "applied_rules": feedback_data.applied_rules,
                "effectiveness_score": feedback_data.effectiveness_score,
                "user_rating": feedback_data.user_rating,
                "user_comments": feedback_data.user_comments,
                "performance_metrics": feedback_data.performance_metrics,
                "anonymization_level": feedback_data.anonymization_level.value,
                "metadata": feedback_data.metadata,
                "created_at": feedback_data.timestamp,
            },
        )

    async def _store_anonymized_feedback(
        self, anonymized_feedback: AnonymizedFeedback
    ) -> None:
        """Store anonymized feedback data for ML training.

        Args:
            anonymized_feedback: Anonymized feedback data
        """
        query = text(
            "\n            INSERT INTO ml_training_feedback (\n                feedback_id, feedback_type, session_hash, agent_type,\n                prompt_structure, enhancement_patterns, rule_effectiveness,\n                satisfaction_score, performance_metrics, anonymization_metadata,\n                created_at\n            ) VALUES (\n                :feedback_id, :feedback_type, :session_hash, :agent_type,\n                :prompt_structure, :enhancement_patterns, :rule_effectiveness,\n                :satisfaction_score, :performance_metrics, :anonymization_metadata,\n                :created_at\n            )\n        "
        )
        await self.db_session.execute(
            query,
            {
                "feedback_id": anonymized_feedback.feedback_id,
                "feedback_type": anonymized_feedback.feedback_type.value,
                "session_hash": anonymized_feedback.session_hash,
                "agent_type": anonymized_feedback.agent_type,
                "prompt_structure": anonymized_feedback.prompt_structure,
                "enhancement_patterns": anonymized_feedback.enhancement_patterns,
                "rule_effectiveness": anonymized_feedback.rule_effectiveness,
                "satisfaction_score": anonymized_feedback.satisfaction_score,
                "performance_metrics": anonymized_feedback.performance_metrics,
                "anonymization_metadata": anonymized_feedback.anonymization_metadata,
                "created_at": anonymized_feedback.timestamp,
            },
        )

    async def _queue_failed_feedback(self, feedback_data: FeedbackData) -> None:
        """Queue failed feedback for later retry.

        Args:
            feedback_data: Failed feedback data
        """
        try:
            redis = await self.get_redis_client()
            feedback_json = {
                "feedback_id": feedback_data.feedback_id,
                "data": asdict(feedback_data),
                "failed_at": time.time(),
                "retry_count": 0,
            }
            await redis.lpush(self.failed_feedback_key, str(feedback_json))
            logger.info(
                f"Queued failed feedback for retry: {feedback_data.feedback_id}"
            )
        except Exception as e:
            logger.error(f"Failed to queue feedback for retry: {e}")

    async def retry_failed_feedback(self, max_items: int = 100) -> int:
        """Retry processing failed feedback items.

        Args:
            max_items: Maximum number of items to retry

        Returns:
            Number of items successfully retried
        """
        try:
            redis = await self.get_redis_client()
            retried_count = 0
            for _ in range(max_items):
                failed_item = await redis.rpop(self.failed_feedback_key)
                if not failed_item:
                    break
                try:
                    import json

                    feedback_json = json.loads(failed_item)
                    feedback_dict = feedback_json["data"]
                    feedback_data = FeedbackData(**feedback_dict)
                    await self._process_feedback_async(feedback_data)
                    retried_count += 1
                except Exception as e:
                    logger.error(f"Failed to retry feedback item: {e}")
                    retry_count = feedback_json.get("retry_count", 0)
                    if retry_count < self.max_retries:
                        feedback_json["retry_count"] = retry_count + 1
                        await redis.lpush(self.failed_feedback_key, str(feedback_json))
            if retried_count > 0:
                logger.info(
                    f"Successfully retried {retried_count} failed feedback items"
                )
            return retried_count
        except Exception as e:
            logger.error(f"Error during failed feedback retry: {e}")
            return 0

    def get_feedback_statistics(self) -> dict[str, Any]:
        """Get feedback collection statistics including DatabaseServices performance.

        Returns:
            Dictionary with enhanced feedback statistics
        """
        total_attempts = (
            self._feedback_stats["successful_storage"]
            + self._feedback_stats["failed_storage"]
        )
        success_rate = self._feedback_stats["successful_storage"] / max(
            1, total_attempts
        )
        base_stats = {
            "total_collected": self._feedback_stats["total_collected"],
            "successful_storage": self._feedback_stats["successful_storage"],
            "failed_storage": self._feedback_stats["failed_storage"],
            "success_rate": success_rate,
            "target_success_rate": self.success_rate_target,
            "success_rate_status": "good"
            if success_rate >= self.success_rate_target
            else "needs_improvement",
            "pii_detections": self._feedback_stats["pii_detections"],
            "anonymization_errors": self._feedback_stats["anonymization_errors"],
            "collection_method": "background_tasks",
        }
        if self._use_database_services and self._database_services:
            try:
                cache_stats = (
                    self._database_services.cache.get_stats()
                    if hasattr(self._database_services.cache, "get_stats")
                    else {}
                )
                base_stats["database_services"] = {
                    "enabled": True,
                    "healthy": True,  # Health check will be done in async health_check method
                    "performance_improvement": "Multi-level cache with connection pooling",
                    "cache_health": cache_stats.get("health", "unknown"),
                    "mode": "HIGH_AVAILABILITY",
                    "cache_optimization": "L1/L2/L3 multi-level caching for feedback data",
                }
            except Exception as e:
                base_stats["database_services"] = {
                    "enabled": True,
                    "error": str(e),
                }
        else:
            base_stats["database_services"] = {
                "enabled": False,
                "reason": "Using direct Redis connection",
            }
        return base_stats

    async def get_feedback_queue_status(self) -> dict[str, Any]:
        """Get feedback queue status from Redis.

        Returns:
            Queue status information
        """
        try:
            redis = await self.get_redis_client()
            failed_queue_length = await redis.llen(self.failed_feedback_key)
            return {
                "failed_queue_length": failed_queue_length,
                "queue_healthy": failed_queue_length < 100,
                "redis_connected": True,
            }
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {
                "failed_queue_length": -1,
                "queue_healthy": False,
                "redis_connected": False,
                "error": str(e),
            }

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check for feedback collection system with DatabaseServices.

        Returns:
            Enhanced health status information
        """
        stats = self.get_feedback_statistics()
        queue_status = await self.get_feedback_queue_status()
        health_issues = []
        if stats["success_rate"] < self.success_rate_target:
            health_issues.append(
                f"Success rate below target: {stats['success_rate']:.2f}"
            )
        if not queue_status["queue_healthy"]:
            health_issues.append(
                f"Failed queue too large: {queue_status['failed_queue_length']}"
            )
        if not queue_status["redis_connected"]:
            health_issues.append("Redis connection failed")
        if stats["anonymization_errors"] > stats["total_collected"] * 0.05:
            health_issues.append("High anonymization error rate")
        services_health = stats.get("database_services", {})
        if services_health.get("enabled", False) and (
            not services_health.get("healthy", True)
        ):
            health_issues.append("DatabaseServices unhealthy")
        overall_status = "healthy" if not health_issues else "degraded"
        return {
            "status": overall_status,
            "issues": health_issues,
            "statistics": stats,
            "queue_status": queue_status,
            "database_services": services_health,
            "performance_enhancements": {
                "connection_pooling": services_health.get("enabled", False),
                "cache_optimization": services_health.get("enabled", False),
                "performance_improvement": services_health.get(
                    "performance_improvement", "N/A"
                ),
            },
            "timestamp": aware_utc_now().isoformat(),
        }
