"""
Adaptive Training Coordinator - 2025 Best Practices Implementation
Integrates adaptive data generation with continuous training workflows.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from ....performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority

from sqlalchemy.ext.asyncio import AsyncSession

from ....database import get_sessionmanager
from ....database.models import TrainingSession, TrainingSessionUpdate
from ..core.ml_pipeline_orchestrator import MLPipelineOrchestrator
from ...analysis.performance_gap_analyzer import PerformanceGapAnalyzer, GapAnalysisResult
from ...analysis.generation_strategy_analyzer import GenerationStrategyAnalyzer, StrategyRecommendation
from ...analysis.difficulty_distribution_analyzer import DifficultyDistributionAnalyzer, DifficultyProfile
from ...preprocessing.orchestrator import ProductionSyntheticDataGenerator
from ...analytics.generation_analytics import GenerationHistoryTracker, GenerationAnalytics
from ...optimization.batch import UnifiedBatchProcessor, UnifiedBatchConfig, ProcessingStrategy


@dataclass
class AdaptiveTrainingIteration:
    """Represents one iteration of adaptive training."""
    iteration_id: str
    session_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    performance_gaps: Optional[GapAnalysisResult] = None
    strategy_recommendation: Optional[StrategyRecommendation] = None
    difficulty_profile: Optional[DifficultyProfile] = None
    generated_data_count: int = 0
    performance_improvement: float = 0.0
    stopping_criteria_met: bool = False
    metadata: Dict[str, Any] = None


class AdaptiveTrainingCoordinator:
    """
    2025 best practices adaptive training coordinator.

    Integrates adaptive data generation with continuous training workflows:
    - Performance gap analysis and strategy determination
    - Targeted synthetic data generation
    - Continuous training loop management
    - Intelligent stopping criteria
    - Session persistence and recovery
    """

    def __init__(
        self,
        orchestrator: MLPipelineOrchestrator,
        data_generator: ProductionSyntheticDataGenerator,
        db_session: Optional[AsyncSession] = None
    ):
        self.logger = logging.getLogger("apes.adaptive_training_coordinator")
        self.orchestrator = orchestrator
        self.data_generator = data_generator

        # Analysis components
        self.gap_analyzer = PerformanceGapAnalyzer()
        self.strategy_analyzer = GenerationStrategyAnalyzer()
        self.difficulty_analyzer = DifficultyDistributionAnalyzer()

        # Week 6 enhancements
        self.generation_tracker: Optional[GenerationHistoryTracker] = None
        self.generation_analytics: Optional[GenerationAnalytics] = None
        self.batch_optimizer: Optional[UnifiedBatchProcessor] = None

        # Initialize Week 6 features if database session provided
        if db_session:
            self.generation_tracker = GenerationHistoryTracker(db_session)
            self.generation_analytics = GenerationAnalytics(db_session)
            self.data_generator.enable_history_tracking(db_session)

            # Configure unified batch processing
            batch_config = UnifiedBatchConfig(
                strategy=ProcessingStrategy.OPTIMIZED,
                max_memory_mb=4000.0,  # 4GB for training
                enable_optimization=True
            )
            self.batch_optimizer = UnifiedBatchProcessor(batch_config)

        # Training state
        self.active_sessions: Dict[str, AdaptiveTrainingIteration] = {}
        self.session_history: Dict[str, List[AdaptiveTrainingIteration]] = {}

        # Enhanced configuration with Week 6 features
        self.config = {
            "max_iterations_per_session": 50,
            "performance_improvement_threshold": 0.02,
            "plateau_detection_window": 5,
            "min_data_generation_samples": 100,
            "max_data_generation_samples": 2000,
            "session_timeout_hours": 24,
            "checkpoint_frequency": 5,  # Save every 5 iterations
            # Week 6 enhancements
            "enable_history_tracking": self.generation_tracker is not None,
            "enable_dynamic_batching": self.batch_optimizer is not None,
            "quality_threshold": 0.75,
            "enable_method_auto_selection": True,
            "analytics_reporting_frequency": 10  # Generate analytics every 10 iterations
        }

    async def start_adaptive_training_session(
        self,
        session_config: Dict[str, Any],
        initial_data: Optional[Any] = None,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """
        Start a new adaptive training session with continuous improvement.

        Args:
            session_config: Configuration for the training session
            initial_data: Initial training data (optional)
            focus_areas: Specific areas to focus improvement on

        Returns:
            Session ID for tracking
        """
        session_id = f"adaptive_training_{uuid.uuid4().hex[:8]}"
        self.logger.info(f"Starting adaptive training session: {session_id}")

        try:
            # 1. Create database session record
            await self._create_training_session_record(session_id, session_config)

            # 2. Initialize first iteration
            first_iteration = AdaptiveTrainingIteration(
                iteration_id=f"{session_id}_iter_1",
                session_id=session_id,
                started_at=datetime.now(timezone.utc),
                metadata={
                    "session_config": session_config,
                    "focus_areas": focus_areas or [],
                    "initial_data_provided": initial_data is not None
                }
            )

            # 3. Store active session
            self.active_sessions[session_id] = first_iteration
            self.session_history[session_id] = [first_iteration]

            # 4. Start continuous training loop with CRITICAL priority for ML training operations
            task_manager = get_background_task_manager()
            training_task_id = await task_manager.submit_enhanced_task(
                task_id=f"ml_adaptive_training_{session_id}_{str(uuid.uuid4())[:8]}",
                coroutine=self._run_continuous_training_loop(session_id, initial_data, focus_areas),
                priority=TaskPriority.CRITICAL,
                tags={
                    "service": "ml",
                    "type": "training",
                    "component": "adaptive_training_coordinator",
                    "session_id": session_id,
                    "module": "adaptive_training_coordinator"
                }
            )
            
            # Store task ID for tracking
            self.active_training_tasks[session_id] = training_task_id

            self.logger.info(f"Adaptive training session started: {session_id}")
            return session_id

        except Exception as e:
            self.logger.error(f"Failed to start adaptive training session: {e}")
            raise

    async def _run_continuous_training_loop(
        self,
        session_id: str,
        initial_data: Optional[Any],
        focus_areas: Optional[List[str]]
    ) -> None:
        """Run the main continuous training loop with adaptive data generation."""

        self.logger.info(f"Starting continuous training loop for session: {session_id}")
        iteration_count = 0

        try:
            while iteration_count < self.config["max_iterations_per_session"]:
                iteration_count += 1
                iteration_id = f"{session_id}_iter_{iteration_count}"

                self.logger.info(f"Starting iteration {iteration_count} for session {session_id}")

                # Create new iteration
                current_iteration = AdaptiveTrainingIteration(
                    iteration_id=iteration_id,
                    session_id=session_id,
                    started_at=datetime.now(timezone.utc)
                )

                try:
                    # Step 1: Analyze performance gaps
                    self.logger.info(f"Analyzing performance gaps for iteration {iteration_count}")
                    gap_analysis = await self._analyze_performance_gaps(session_id, focus_areas)
                    current_iteration.performance_gaps = gap_analysis

                    # Step 2: Check stopping criteria
                    if gap_analysis.stopping_criteria_met:
                        self.logger.info(f"Stopping criteria met for session {session_id}")
                        current_iteration.stopping_criteria_met = True
                        await self._complete_training_session(session_id, "stopping_criteria_met")
                        break

                    # Step 3: Determine generation strategy
                    self.logger.info(f"Determining generation strategy for iteration {iteration_count}")
                    strategy_recommendation = await self._determine_generation_strategy(
                        gap_analysis, focus_areas
                    )
                    current_iteration.strategy_recommendation = strategy_recommendation

                    # Step 4: Analyze difficulty distribution
                    self.logger.info(f"Analyzing difficulty distribution for iteration {iteration_count}")
                    difficulty_profile = await self._analyze_difficulty_distribution(
                        gap_analysis, focus_areas
                    )
                    current_iteration.difficulty_profile = difficulty_profile

                    # Step 5: Generate targeted synthetic data
                    self.logger.info(f"Generating targeted data for iteration {iteration_count}")
                    generated_data = await self._generate_targeted_data(
                        gap_analysis, strategy_recommendation, difficulty_profile, session_id
                    )
                    current_iteration.generated_data_count = len(generated_data.get("features", []))

                    # Step 6: Execute training with new data
                    self.logger.info(f"Executing training for iteration {iteration_count}")
                    training_results = await self._execute_training_iteration(
                        session_id, generated_data, current_iteration
                    )

                    # Step 7: Evaluate performance improvement
                    performance_improvement = await self._evaluate_performance_improvement(
                        session_id, training_results
                    )
                    current_iteration.performance_improvement = performance_improvement

                    # Step 8: Update session and complete iteration
                    current_iteration.completed_at = datetime.now(timezone.utc)
                    await self._update_training_session(session_id, current_iteration)

                    # Step 9: Generate analytics report if enabled (Week 6)
                    if (self.config["enable_history_tracking"] and
                        iteration_count % self.config["analytics_reporting_frequency"] == 0):
                        await self._generate_analytics_report(session_id, iteration_count)

                    # Step 10: Checkpoint if needed
                    if iteration_count % self.config["checkpoint_frequency"] == 0:
                        await self._checkpoint_session(session_id)

                    self.logger.info(f"Completed iteration {iteration_count} for session {session_id}")

                except Exception as e:
                    self.logger.error(f"Error in iteration {iteration_count} for session {session_id}: {e}")
                    current_iteration.metadata = {"error": str(e)}
                    await self._handle_iteration_error(session_id, current_iteration, e)

                finally:
                    # Store iteration in history
                    self.session_history[session_id].append(current_iteration)
                    self.active_sessions[session_id] = current_iteration

            # Complete session if max iterations reached
            if iteration_count >= self.config["max_iterations_per_session"]:
                await self._complete_training_session(session_id, "max_iterations_reached")

        except Exception as e:
            self.logger.error(f"Critical error in continuous training loop for session {session_id}: {e}")
            await self._complete_training_session(session_id, "error", str(e))

    async def _analyze_performance_gaps(
        self,
        session_id: str,
        focus_areas: Optional[List[str]]
    ) -> GapAnalysisResult:
        """Analyze current performance gaps for the session."""

        async with get_sessionmanager().get_session() as db_session:
            # Use enhanced gap analysis for targeted generation
            gap_analysis = await self.gap_analyzer.analyze_gaps_for_targeted_generation(
                session=db_session,
                rule_ids=None,  # Analyze all rules
                focus_areas=focus_areas
            )

            return gap_analysis["standard_analysis"]

    async def _determine_generation_strategy(
        self,
        gap_analysis: GapAnalysisResult,
        focus_areas: Optional[List[str]]
    ) -> StrategyRecommendation:
        """Determine optimal generation strategy based on gap analysis."""

        # Get hardness analysis from gap analysis metadata
        hardness_analysis = gap_analysis.metadata.get("hardness_analysis", {})

        strategy_recommendation = await self.strategy_analyzer.analyze_optimal_strategy(
            gap_analysis=gap_analysis,
            hardness_analysis=hardness_analysis,
            focus_areas=focus_areas
        )

        return strategy_recommendation

    async def _analyze_difficulty_distribution(
        self,
        gap_analysis: GapAnalysisResult,
        focus_areas: Optional[List[str]]
    ) -> DifficultyProfile:
        """Analyze optimal difficulty distribution for data generation."""

        all_gaps = gap_analysis.critical_gaps + gap_analysis.improvement_opportunities
        hardness_analysis = gap_analysis.metadata.get("hardness_analysis", {})

        difficulty_profile = await self.difficulty_analyzer.analyze_optimal_difficulty_distribution(
            performance_gaps=all_gaps,
            hardness_analysis=hardness_analysis,
            focus_areas=focus_areas
        )

        return difficulty_profile

    async def _generate_targeted_data(
        self,
        gap_analysis: GapAnalysisResult,
        strategy_recommendation: StrategyRecommendation,
        difficulty_profile: DifficultyProfile,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate targeted synthetic data using Week 6 enhanced features."""

        # Extract performance gaps for targeting
        all_gaps = gap_analysis.critical_gaps + gap_analysis.improvement_opportunities
        performance_gaps = {
            f"{gap.gap_type}_{gap.rule_id}": gap.gap_magnitude
            for gap in all_gaps
        }

        # Use enhanced generation with history tracking if available
        if self.config["enable_history_tracking"] and self.generation_tracker:
            self.logger.info("Using enhanced generation with history tracking")

            generated_data = await self.data_generator.generate_with_history_tracking(
                total_samples=strategy_recommendation.estimated_samples,
                performance_gaps=performance_gaps,
                strategy=strategy_recommendation.primary_strategy.value,
                training_session_id=session_id
            )

        elif self.config["enable_dynamic_batching"] and self.batch_optimizer:
            self.logger.info("Using dynamic batch optimization")

            generated_data = await self.data_generator.generate_with_dynamic_batching(
                total_samples=strategy_recommendation.estimated_samples,
                performance_gaps=performance_gaps,
                strategy=strategy_recommendation.primary_strategy.value
            )

        else:
            # Fallback to standard targeted generation
            self.logger.info("Using standard targeted generation")

            generated_data = await self.data_generator.generate_targeted_data(
                performance_gaps=performance_gaps,
                strategy=strategy_recommendation.primary_strategy.value,
                batch_size=strategy_recommendation.estimated_samples,
                focus_areas=strategy_recommendation.focus_areas
            )

        # Log generation results
        sample_count = len(generated_data.get("features", []))
        generation_method = generated_data.get("metadata", {}).get("generation_method", "unknown")
        self.logger.info(f"Generated {sample_count} samples using {generation_method} method")

        return generated_data

    async def _generate_analytics_report(
        self,
        session_id: str,
        iteration_count: int
    ) -> None:
        """Generate comprehensive analytics report (Week 6 feature)"""

        if not self.generation_analytics:
            return

        try:
            # Generate effectiveness report
            effectiveness_report = await self.generation_analytics.get_effectiveness_report(
                session_id=session_id,
                days_back=7
            )

            # Get method comparison
            method_comparison = await self.generation_analytics.get_method_comparison(
                days_back=7
            )

            # Get performance trends
            performance_trends = await self.generation_analytics.get_performance_trends(
                days_back=7
            )

            # Log key insights
            self.logger.info(f"Analytics Report for Session {session_id} (Iteration {iteration_count}):")
            self.logger.info(f"  Overall Effectiveness: {effectiveness_report['overall_effectiveness']['effectiveness_score']:.3f}")
            self.logger.info(f"  Best Method: {method_comparison.get('best_method', 'N/A')}")
            self.logger.info(f"  Quality Trend: {performance_trends['trends']['quality_trend']:.3f}")
            self.logger.info(f"  Efficiency Trend: {performance_trends['trends']['efficiency_trend']:.3f}")

            # Log recommendations
            recommendations = effectiveness_report.get('recommendations', [])
            if recommendations:
                self.logger.info("  Recommendations:")
                for rec in recommendations[:3]:  # Top 3 recommendations
                    self.logger.info(f"    - {rec}")

        except Exception as e:
            self.logger.error(f"Failed to generate analytics report: {e}")

    async def _execute_training_iteration(
        self,
        session_id: str,
        generated_data: Dict[str, Any],
        current_iteration: AdaptiveTrainingIteration
    ) -> Dict[str, Any]:
        """Execute one training iteration with the generated data."""

        # Prepare training data for orchestrator
        training_data = {
            "features": generated_data.get("features", []),
            "effectiveness": generated_data.get("effectiveness", []),
            "prompts": generated_data.get("prompts", []),
            "metadata": {
                "iteration_id": current_iteration.iteration_id,
                "session_id": session_id,
                "generation_method": generated_data.get("metadata", {}).get("generation_method", "unknown"),
                "adaptive_training": True
            }
        }

        # Execute training workflow through orchestrator
        training_results = await self.orchestrator.run_training_workflow_with_memory_monitoring(
            training_data=training_data,
            context={
                "session_id": session_id,
                "iteration_id": current_iteration.iteration_id,
                "adaptive_mode": True
            }
        )

        return training_results

    async def _evaluate_performance_improvement(
        self,
        session_id: str,
        training_results: Dict[str, Any]
    ) -> float:
        """Evaluate performance improvement from the training iteration."""

        # Get previous performance from session history
        session_iterations = self.session_history.get(session_id, [])
        if len(session_iterations) < 2:
            return 0.0  # No previous iteration to compare

        # Extract performance metrics from training results
        current_performance = training_results.get("performance_metrics", {}).get("overall_score", 0.0)

        # Get previous performance
        previous_iteration = session_iterations[-2]  # Second to last (current is last)
        previous_performance = previous_iteration.metadata.get("performance_score", 0.0)

        # Calculate improvement
        improvement = current_performance - previous_performance

        return improvement

    async def _create_training_session_record(
        self,
        session_id: str,
        session_config: Dict[str, Any]
    ) -> None:
        """Create database record for the training session."""

        async with get_sessionmanager().get_session() as db_session:
            from sqlalchemy import select

            # Check if session already exists
            existing_session = await db_session.execute(
                select(TrainingSession).where(TrainingSession.session_id == session_id)
            )

            if existing_session.scalar_one_or_none():
                self.logger.warning(f"Training session {session_id} already exists")
                return

            # Create new training session
            training_session = TrainingSession(
                session_id=session_id,
                continuous_mode=True,
                max_iterations=session_config.get("max_iterations", self.config["max_iterations_per_session"]),
                improvement_threshold=session_config.get("improvement_threshold", self.config["performance_improvement_threshold"]),
                timeout_seconds=session_config.get("timeout_hours", self.config["session_timeout_hours"]) * 3600,
                status="running",
                current_iteration=0,
                checkpoint_data={
                    "adaptive_training": True,
                    "session_config": session_config,
                    "coordinator_version": "2025.1"
                }
            )

            db_session.add(training_session)
            await db_session.commit()

            self.logger.info(f"Created training session record: {session_id}")

    async def _update_training_session(
        self,
        session_id: str,
        current_iteration: AdaptiveTrainingIteration
    ) -> None:
        """Update training session with iteration results."""

        async with get_sessionmanager().get_session() as db_session:
            from sqlalchemy import select

            # Get existing session
            result = await db_session.execute(
                select(TrainingSession).where(TrainingSession.session_id == session_id)
            )
            training_session = result.scalar_one_or_none()

            if not training_session:
                self.logger.error(f"Training session {session_id} not found for update")
                return

            # Update session with iteration data
            update_data = TrainingSessionUpdate(
                current_iteration=training_session.current_iteration + 1,
                current_performance=current_iteration.performance_improvement,
                data_points_processed=training_session.data_points_processed + current_iteration.generated_data_count,
                last_activity_at=datetime.now(timezone.utc)
            )

            # Update performance history
            if training_session.performance_history:
                performance_history = training_session.performance_history.copy()
            else:
                performance_history = []

            performance_history.append(current_iteration.performance_improvement)
            update_data.performance_history = performance_history

            # Update best performance if improved
            if (training_session.best_performance is None or
                current_iteration.performance_improvement > training_session.best_performance):
                update_data.best_performance = current_iteration.performance_improvement

            # Apply updates
            for field, value in update_data.model_dump(exclude_unset=True).items():
                setattr(training_session, field, value)

            await db_session.commit()

            self.logger.info(f"Updated training session {session_id} with iteration results")

    async def _checkpoint_session(self, session_id: str) -> None:
        """Create checkpoint for session recovery."""

        session_iterations = self.session_history.get(session_id, [])
        if not session_iterations:
            return

        checkpoint_data = {
            "session_id": session_id,
            "checkpoint_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_iterations": len(session_iterations),
            "latest_iteration": session_iterations[-1].__dict__ if session_iterations else None,
            "performance_trend": [iter.performance_improvement for iter in session_iterations],
            "active_session_state": self.active_sessions.get(session_id).__dict__ if session_id in self.active_sessions else None
        }

        async with get_sessionmanager().get_session() as db_session:
            from sqlalchemy import select

            result = await db_session.execute(
                select(TrainingSession).where(TrainingSession.session_id == session_id)
            )
            training_session = result.scalar_one_or_none()

            if training_session:
                training_session.checkpoint_data = checkpoint_data
                training_session.last_checkpoint_at = datetime.now(timezone.utc)
                await db_session.commit()

                self.logger.info(f"Created checkpoint for session {session_id}")

    async def _complete_training_session(
        self,
        session_id: str,
        completion_reason: str,
        error_message: Optional[str] = None
    ) -> None:
        """Complete and finalize training session."""

        self.logger.info(f"Completing training session {session_id}: {completion_reason}")

        async with get_sessionmanager().get_session() as db_session:
            from sqlalchemy import select

            result = await db_session.execute(
                select(TrainingSession).where(TrainingSession.session_id == session_id)
            )
            training_session = result.scalar_one_or_none()

            if training_session:
                training_session.status = "completed" if not error_message else "failed"
                training_session.completed_at = datetime.now(timezone.utc)
                training_session.last_error = error_message

                # Add completion metadata
                if training_session.checkpoint_data:
                    training_session.checkpoint_data["completion_reason"] = completion_reason
                    training_session.checkpoint_data["completion_timestamp"] = datetime.now(timezone.utc).isoformat()

                await db_session.commit()

        # Clean up active session
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        self.logger.info(f"Training session {session_id} completed successfully")

    async def _handle_iteration_error(
        self,
        session_id: str,
        current_iteration: AdaptiveTrainingIteration,
        error: Exception
    ) -> None:
        """Handle errors during training iterations."""

        self.logger.error(f"Handling iteration error for session {session_id}: {error}")

        # Update session error tracking
        async with get_sessionmanager().get_session() as db_session:
            from sqlalchemy import select

            result = await db_session.execute(
                select(TrainingSession).where(TrainingSession.session_id == session_id)
            )
            training_session = result.scalar_one_or_none()

            if training_session:
                training_session.error_count += 1
                training_session.last_error = str(error)
                training_session.last_activity_at = datetime.now(timezone.utc)

                # If too many errors, fail the session
                if training_session.error_count >= 5:
                    await self._complete_training_session(session_id, "too_many_errors", str(error))

                await db_session.commit()

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a training session."""

        # Get from active sessions first
        if session_id in self.active_sessions:
            current_iteration = self.active_sessions[session_id]
            session_iterations = self.session_history.get(session_id, [])

            return {
                "session_id": session_id,
                "status": "active",
                "current_iteration": current_iteration.__dict__,
                "total_iterations": len(session_iterations),
                "performance_trend": [iter.performance_improvement for iter in session_iterations],
                "last_update": current_iteration.started_at.isoformat()
            }

        # Get from database
        async with get_sessionmanager().get_session() as db_session:
            from sqlalchemy import select

            result = await db_session.execute(
                select(TrainingSession).where(TrainingSession.session_id == session_id)
            )
            training_session = result.scalar_one_or_none()

            if training_session:
                return {
                    "session_id": session_id,
                    "status": training_session.status,
                    "current_iteration": training_session.current_iteration,
                    "performance_history": training_session.performance_history or [],
                    "best_performance": training_session.best_performance,
                    "last_update": training_session.last_activity_at.isoformat() if training_session.last_activity_at else None
                }

            return {"session_id": session_id, "status": "not_found"}
