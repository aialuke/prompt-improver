"""Prompt Application Service.

Orchestrates prompt improvement workflows, coordinating between presentation
and domain layers while managing transaction boundaries and cross-cutting concerns.
"""

import hashlib
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

# Cache manager handled through dependency injection - no direct database imports
from prompt_improver.repositories.protocols.prompt_repository_protocol import (
    PromptRepositoryProtocol,
)
from prompt_improver.rule_engine import RuleEngine
from prompt_improver.security.owasp_input_validator import OWASP2025InputValidator
from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.services.cache.cache_factory import CacheFactory
from prompt_improver.services.prompt.facade import (
    PromptServiceFacade as PromptImprovementService,
)
from prompt_improver.shared.interfaces.protocols.database import (
    SessionManagerProtocol,
)

logger = logging.getLogger(__name__)


class PromptApplicationService:
    """Application service for prompt improvement workflows.

    Orchestrates the complete prompt improvement process including:
    - Session management and tracking
    - Rule application and validation
    - Feedback collection and storage
    - Transaction boundary management
    - Cross-cutting concerns (logging, monitoring, security)
    """

    def __init__(
        self,
        session_manager: SessionManagerProtocol,
        prompt_repository: PromptRepositoryProtocol,
        rule_engine: RuleEngine,
        cache_facade: CacheFacade,
        input_validator: OWASP2025InputValidator,
        prompt_improvement_service: PromptImprovementService,
        cache_manager=None,  # Cache manager from performance layer
    ) -> None:
        self.session_manager = session_manager
        self.prompt_repository = prompt_repository
        self.rule_engine = rule_engine
        # Use factory for prompt cache while maintaining compatibility with injected cache
        self.cache_facade = cache_facade or CacheFactory.get_prompt_cache()
        self.input_validator = input_validator
        self.prompt_improvement_service = prompt_improvement_service
        self.logger = logger

        # Performance optimization - multi-level caching with factory fallback
        self.cache_manager = cache_manager or CacheFactory.get_utility_cache()
        self._cache_enabled = True  # Always enabled with factory pattern

        # Performance metrics
        self._performance_metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time_ms": 0.0,
            "total_requests": 0,
        }

    async def initialize(self) -> None:
        """Initialize the application service."""
        self.logger.info("Initializing PromptApplicationService")

        # Initialize any required resources
        await self.rule_engine.initialize()

        # Cache manager is always available through factory pattern
        try:
            # Factory-based caches are pre-initialized
            self.logger.info("Cache manager initialized via factory pattern for prompt service")
        except Exception as e:
            self.logger.warning(f"Factory cache access failed: {e}")
            self._cache_enabled = False

    async def cleanup(self) -> None:
        """Clean up application service resources."""
        self.logger.info("Cleaning up PromptApplicationService")
        await self.rule_engine.cleanup()

        # Factory-based caches handle their own lifecycle management
        try:
            self.logger.info("Cache manager cleanup handled by factory pattern")
        except Exception as e:
            self.logger.warning(f"Factory cache cleanup warning: {e}")

    async def improve_prompt(
        self,
        prompt: str,
        session_id: str,
        improvement_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Orchestrate the complete prompt improvement workflow.

        This method coordinates the entire prompt improvement process:
        1. Validate input and security
        2. Check cache for existing improvements (performance optimization)
        3. Load or create session context
        4. Apply rule selection and improvement logic
        5. Store results and update session
        6. Cache results for future requests
        7. Return comprehensive results with metadata

        Args:
            prompt: The prompt text to improve
            session_id: Session identifier for tracking
            improvement_options: Optional configuration for improvement

        Returns:
            Dict containing improved prompt, rules applied, metrics, and metadata
        """
        start_time = datetime.now(UTC)
        self._performance_metrics["total_requests"] += 1

        # Performance optimization: Generate cache key for this improvement request
        cache_key = None
        if self._cache_enabled:
            cache_key = self._generate_cache_key(prompt, session_id, improvement_options)

            # Check cache first (target: <5ms)
            try:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    self._performance_metrics["cache_hits"] += 1
                    self._update_performance_metrics(start_time)
                    self.logger.debug(f"Cache hit for prompt improvement: {session_id}")

                    # Add cache metadata
                    cached_result["cache_hit"] = True
                    cached_result["served_from_cache"] = True
                    return cached_result

            except Exception as e:
                self.logger.warning(f"Cache lookup failed: {e}")

            self._performance_metrics["cache_misses"] += 1

        try:
            self.logger.info(f"Starting prompt improvement workflow for session {session_id}")

            # 1. Input validation and security
            if not await self._validate_prompt_input(prompt):
                return {
                    "status": "error",
                    "error": "Invalid prompt input",
                    "timestamp": start_time.isoformat(),
                    "cache_hit": False,
                    "served_from_cache": False,
                }

            # 2. Session management
            session_context = await self._load_or_create_session_context(
                session_id, improvement_options
            )

            # 3. Transaction boundary - entire workflow
            async with self.session_manager.get_session() as db_session:
                try:
                    # 4. Core improvement workflow
                    improvement_result = await self.prompt_improvement_service.improve_prompt(
                        prompt=prompt,
                        session_context=session_context,
                        db_session=db_session,
                    )

                    # 5. Store improvement results
                    await self._store_improvement_results(
                        session_id, prompt, improvement_result, db_session
                    )

                    # 6. Update session state
                    await self._update_session_state(
                        session_id, improvement_result, session_context
                    )

                    # 7. Commit transaction
                    await db_session.commit()

                    # 8. Prepare comprehensive response
                    end_time = datetime.now(UTC)
                    duration_ms = (end_time - start_time).total_seconds() * 1000

                    result = {
                        "status": "success",
                        "session_id": session_id,
                        "original_prompt": prompt,
                        "improved_prompt": improvement_result.get("improved_prompt"),
                        "rules_applied": improvement_result.get("rules_applied", []),
                        "improvement_metrics": improvement_result.get("metrics", {}),
                        "session_metrics": session_context.get("metrics", {}),
                        "processing_time_ms": duration_ms,
                        "timestamp": end_time.isoformat(),
                        "cache_hit": False,
                        "served_from_cache": False,
                        "metadata": {
                            "workflow_version": "2.0",
                            "improvement_options": improvement_options or {},
                            "session_context_loaded": True,
                        }
                    }

                    # Cache successful results for future requests (TTL: 1 hour)
                    if self._cache_enabled and cache_key:
                        try:
                            await self.cache_manager.set(
                                cache_key,
                                result,
                                ttl_seconds=3600  # Cache for 1 hour
                            )
                            self.logger.debug(f"Cached prompt improvement result: {session_id}")
                        except Exception as e:
                            self.logger.warning(f"Failed to cache result: {e}")

                    # Update performance metrics
                    self._update_performance_metrics(start_time)

                    return result

                except Exception as e:
                    await db_session.rollback()
                    self.logger.exception(f"Error in improvement workflow: {e}")
                    raise

        except Exception as e:
            self.logger.exception(f"Prompt improvement workflow failed: {e}")
            self._update_performance_metrics(start_time)
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "cache_hit": False,
                "served_from_cache": False,
            }

    async def apply_rules_to_prompt(
        self,
        prompt: str,
        rule_ids: list[str],
        session_id: str,
    ) -> dict[str, Any]:
        """Apply specific rules to a prompt with session tracking.

        Args:
            prompt: The prompt text to process
            rule_ids: List of rule IDs to apply
            session_id: Session identifier for tracking

        Returns:
            Dict containing results of rule application
        """
        start_time = datetime.now(UTC)

        try:
            self.logger.info(f"Applying rules {rule_ids} to prompt in session {session_id}")

            # Input validation
            if not await self._validate_prompt_input(prompt):
                return {"status": "error", "error": "Invalid prompt input"}

            if not rule_ids:
                return {"status": "error", "error": "No rules specified"}

            # Load session context
            session_context = await self._load_session_context(session_id)

            # Transaction boundary for rule application
            async with self.session_manager.get_session() as db_session:
                try:
                    # Apply rules through rule engine
                    rule_results = await self.rule_engine.apply_rules(
                        prompt=prompt,
                        rule_ids=rule_ids,
                        context=session_context,
                        db_session=db_session,
                    )

                    # Store rule application results
                    await self._store_rule_application(
                        session_id, prompt, rule_ids, rule_results, db_session
                    )

                    await db_session.commit()

                    end_time = datetime.now(UTC)
                    duration_ms = (end_time - start_time).total_seconds() * 1000

                    return {
                        "status": "success",
                        "session_id": session_id,
                        "original_prompt": prompt,
                        "modified_prompt": rule_results.get("modified_prompt"),
                        "rules_applied": rule_results.get("rules_applied", []),
                        "rule_metrics": rule_results.get("metrics", {}),
                        "processing_time_ms": duration_ms,
                        "timestamp": end_time.isoformat(),
                    }

                except Exception as e:
                    await db_session.rollback()
                    raise

        except Exception as e:
            self.logger.exception(f"Rule application failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }

    async def create_improvement_session(
        self,
        initial_prompt: str,
        user_preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new prompt improvement session.

        Args:
            initial_prompt: Starting prompt for the session
            user_preferences: Optional user preferences for improvement

        Returns:
            Dict containing new session details
        """
        try:
            session_id = str(uuid.uuid4())

            self.logger.info(f"Creating new improvement session {session_id}")

            # Validate initial prompt
            if not await self._validate_prompt_input(initial_prompt):
                return {"status": "error", "error": "Invalid initial prompt"}

            # Create session with transaction boundary
            async with self.session_manager.get_session() as db_session:
                try:
                    # Create session record
                    session_data = {
                        "session_id": session_id,
                        "initial_prompt": initial_prompt,
                        "user_preferences": user_preferences or {},
                        "created_at": datetime.now(UTC).isoformat(),
                        "status": "active",
                        "improvement_history": [],
                        "metrics": {
                            "total_improvements": 0,
                            "rules_applied_count": 0,
                            "avg_processing_time_ms": 0,
                        }
                    }

                    # Store in session store
                    await self.cache_facade.set_session(session_id, session_data, ttl=3600)

                    # Store in database
                    await self.session_manager.create_session(
                        session_id, session_data, db_session
                    )

                    await db_session.commit()

                    return {
                        "status": "success",
                        "session_id": session_id,
                        "session_data": session_data,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }

                except Exception as e:
                    await db_session.rollback()
                    raise

        except Exception as e:
            self.logger.exception(f"Session creation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    async def finalize_improvement_session(
        self,
        session_id: str,
        feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finalize an improvement session with user feedback.

        Args:
            session_id: Session identifier to finalize
            feedback: Optional user feedback on the session

        Returns:
            Dict containing finalization results
        """
        try:
            self.logger.info(f"Finalizing improvement session {session_id}")

            # Transaction boundary for session finalization
            async with self.session_manager.get_session() as db_session:
                try:
                    # Load session data
                    session_data = await self.cache_facade.get_session(session_id)
                    if not session_data:
                        return {"status": "error", "error": "Session not found"}

                    # Update session with feedback
                    session_data["status"] = "completed"
                    session_data["completed_at"] = datetime.now(UTC).isoformat()
                    session_data["user_feedback"] = feedback or {}

                    # Store final session state
                    await self.session_manager.finalize_session(
                        session_id, session_data, db_session
                    )

                    # Update session store
                    await self.cache_facade.set_session(session_id, session_data, ttl=3600)

                    await db_session.commit()

                    return {
                        "status": "success",
                        "session_id": session_id,
                        "final_metrics": session_data.get("metrics", {}),
                        "completion_summary": {
                            "total_improvements": session_data["metrics"]["total_improvements"],
                            "session_duration": self._calculate_session_duration(session_data),
                            "user_feedback": feedback,
                        },
                        "timestamp": datetime.now(UTC).isoformat(),
                    }

                except Exception as e:
                    await db_session.rollback()
                    raise

        except Exception as e:
            self.logger.exception(f"Session finalization failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }

    # Private helper methods

    async def _validate_prompt_input(self, prompt: str) -> bool:
        """Validate prompt input for security and format using OWASP 2025 compliance."""
        try:
            validation_result = self.input_validator.validate_prompt(prompt)
            if validation_result.is_blocked:
                self.logger.warning(
                    f"Prompt blocked by OWASP validator - Threat: {validation_result.threat_type}, "
                    f"Score: {validation_result.threat_score:.2f}, "
                    f"Patterns: {validation_result.detected_patterns}"
                )
            return validation_result.is_valid
        except Exception as e:
            self.logger.exception(f"OWASP prompt validation failed: {e}")
            return False

    async def _load_or_create_session_context(
        self, session_id: str, improvement_options: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Load existing session context or create new one."""
        session_data = await self.cache_facade.get_session(session_id)

        if not session_data:
            # Create new session context
            session_data = {
                "session_id": session_id,
                "created_at": datetime.now(UTC).isoformat(),
                "improvement_options": improvement_options or {},
                "improvement_history": [],
                "metrics": {
                    "total_improvements": 0,
                    "rules_applied_count": 0,
                    "avg_processing_time_ms": 0,
                }
            }
            await self.cache_facade.set_session(session_id, session_data, ttl=3600)

        return session_data

    def _generate_cache_key(
        self,
        prompt: str,
        session_id: str,
        improvement_options: dict[str, Any] | None
    ) -> str:
        """Generate cache key for prompt improvement request."""
        # Create deterministic cache key based on inputs
        content = f"{prompt}:{session_id}:{improvement_options or {}}"
        return f"prompt_improvement:{hashlib.md5(content.encode()).hexdigest()}"

    def _update_performance_metrics(self, start_time: datetime) -> None:
        """Update performance metrics for monitoring."""
        duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

        # Update rolling average response time
        total_requests = self._performance_metrics["total_requests"]
        current_avg = self._performance_metrics["avg_response_time_ms"]

        self._performance_metrics["avg_response_time_ms"] = (
            (current_avg * (total_requests - 1) + duration_ms) / total_requests
        )

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics for monitoring."""
        cache_hit_rate = (
            self._performance_metrics["cache_hits"] /
            max(self._performance_metrics["total_requests"], 1)
        )

        metrics = {
            "service": "prompt_application_service",
            "performance": dict(self._performance_metrics),
            "cache_hit_rate": cache_hit_rate,
            "cache_enabled": self._cache_enabled,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Add cache manager stats if available
        if self.cache_manager:
            try:
                cache_stats = await self.cache_manager.get_stats()
                metrics["cache_stats"] = cache_stats
            except Exception as e:
                self.logger.warning(f"Failed to get cache stats: {e}")

        return metrics

    async def _load_session_context(self, session_id: str) -> dict[str, Any]:
        """Load existing session context."""
        session_data = await self.cache_facade.get_session(session_id)
        if not session_data:
            raise ValueError(f"Session {session_id} not found")
        return session_data

    async def _store_improvement_results(
        self,
        session_id: str,
        prompt: str,
        improvement_result: dict[str, Any],
        db_session,
    ) -> None:
        """Store improvement results in the database."""
        await self.prompt_repository.store_improvement_result(
            session_id=session_id,
            original_prompt=prompt,
            improvement_result=improvement_result,
            db_session=db_session,
        )

    async def _store_rule_application(
        self,
        session_id: str,
        prompt: str,
        rule_ids: list[str],
        rule_results: dict[str, Any],
        db_session,
    ) -> None:
        """Store rule application results."""
        await self.prompt_repository.store_rule_application(
            session_id=session_id,
            original_prompt=prompt,
            rule_ids=rule_ids,
            rule_results=rule_results,
            db_session=db_session,
        )

    async def _update_session_state(
        self,
        session_id: str,
        improvement_result: dict[str, Any],
        session_context: dict[str, Any],
    ) -> None:
        """Update session state with improvement results."""
        # Update metrics
        session_context["metrics"]["total_improvements"] += 1
        session_context["metrics"]["rules_applied_count"] += len(
            improvement_result.get("rules_applied", [])
        )

        # Add to improvement history
        session_context["improvement_history"].append({
            "timestamp": datetime.now(UTC).isoformat(),
            "improvement_result": improvement_result,
        })

        # Update session store
        await self.cache_facade.set_session(session_id, session_context, ttl=3600)

    def _calculate_session_duration(self, session_data: dict[str, Any]) -> float:
        """Calculate session duration in seconds."""
        try:
            created_at = datetime.fromisoformat(session_data["created_at"].replace("Z", "+00:00"))
            completed_at = datetime.fromisoformat(session_data.get("completed_at", datetime.now(UTC).isoformat()).replace("Z", "+00:00"))
            return (completed_at - created_at).total_seconds()
        except Exception:
            return 0.0
