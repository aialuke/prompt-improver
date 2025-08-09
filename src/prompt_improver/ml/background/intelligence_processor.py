"""
ML Background Intelligence Processor

Runs ML analysis in background and stores pre-computed results in database.
Maintains strict architectural separation from MCP serving layer.

This service:
1. Processes rule performance data using existing ML components
2. Generates pre-computed intelligence for rule selection
3. Stores results in database for MCP read-only access
4. Runs as separate background service/job
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import time
from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database.connection import get_session_context

# Import existing ML components (allowed in ML background service)
from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import (
    AdvancedPatternDiscovery,
)
from prompt_improver.ml.optimization.algorithms.rule_optimizer import RuleOptimizer

# 2025 Circuit Breaker Integration for ML Resilience
from prompt_improver.performance.monitoring.health.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
)
from prompt_improver.rule_engine.models import PromptCharacteristics

logger = logging.getLogger(__name__)


class MLIntelligenceProcessor:
    """Background ML processor for generating pre-computed rule intelligence.

    Architectural Role:
    - Runs in ML system (separate from MCP)
    - Uses existing ML components for analysis
    - Stores results in database for MCP consumption
    - Maintains batch processing schedule
    """

    def __init__(self):
        """Initialize ML components for background processing with 2025 resilience patterns."""
        self.pattern_discovery = AdvancedPatternDiscovery()
        self.rule_optimizer = RuleOptimizer()

        # 2025 Circuit Breaker Configuration for ML Operations
        self._setup_circuit_breakers()

        # Processing configuration - 2025 Performance Optimization
        self.batch_size = 100  # Rules per batch for parallel processing
        self.processing_interval_hours = 6  # 6-hour batch cycles
        self.cache_ttl_hours = 12  # Cache TTL
        self.max_parallel_workers = 4  # Parallel processing workers
        self.incremental_update_threshold = 0.1  # 10% change threshold for updates

        # ML Prediction Pipeline Configuration
        self.confidence_scoring_enabled = True
        self.min_confidence_threshold = 0.6
        self.prediction_batch_size = 50

        # Performance tracking with proper typing
        self.processing_stats: Dict[str, Any] = {
            "rules_processed": 0,
            "combinations_generated": 0,
            "patterns_discovered": 0,
            "processing_time_ms": 0,
            "last_run": None
        }

    def _setup_circuit_breakers(self):
        """Setup circuit breakers for ML operations following 2025 best practices."""
        # Pattern Discovery Circuit Breaker - More lenient for exploratory ML
        pattern_config = CircuitBreakerConfig(
            failure_threshold=3,  # Allow more failures for pattern discovery
            recovery_timeout=300,  # 5 minutes recovery
            half_open_max_calls=2,
            response_time_threshold_ms=30000,  # 30s for complex pattern analysis
            success_rate_threshold=0.7  # Lower threshold for experimental ML
        )
        self.pattern_discovery_breaker = CircuitBreaker(
            name="pattern_discovery",
            config=pattern_config,
            on_state_change=self._on_breaker_state_change
        )

        # Rule Optimization Circuit Breaker - Stricter for production optimization
        optimizer_config = CircuitBreakerConfig(
            failure_threshold=2,  # Stricter for optimization
            recovery_timeout=180,  # 3 minutes recovery
            half_open_max_calls=1,
            response_time_threshold_ms=15000,  # 15s for optimization
            success_rate_threshold=0.85  # Higher threshold for production
        )
        self.rule_optimizer_breaker = CircuitBreaker(
            name="rule_optimizer",
            config=optimizer_config,
            on_state_change=self._on_breaker_state_change
        )

        # Database Circuit Breaker - Critical for data persistence
        database_config = CircuitBreakerConfig(
            failure_threshold=5,  # More tolerance for DB operations
            recovery_timeout=60,  # 1 minute recovery
            half_open_max_calls=3,
            response_time_threshold_ms=5000,  # 5s for DB operations
            success_rate_threshold=0.95  # High threshold for DB reliability
        )
        self.database_breaker = CircuitBreaker(
            name="database_operations",
            config=database_config,
            on_state_change=self._on_breaker_state_change
        )

    def _on_breaker_state_change(self, component_name: str, new_state: CircuitState) -> None:
        """Handle circuit breaker state changes with 2025 observability patterns."""
        logger.warning(
            f"ML Circuit breaker state change: {component_name} -> {new_state.value}",
            extra={
                "component": component_name,
                "new_state": new_state.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": "ml_intelligence_processor"
            }
        )

        # Update processing stats for monitoring
        if new_state.value == "open":
            self.processing_stats[f"{component_name}_circuit_open"] = True
            logger.error("ML component %s circuit breaker OPEN - degraded mode activated", component_name)
        elif new_state.value == "closed":
            self.processing_stats[f"{component_name}_circuit_open"] = False
            logger.info("ML component %s circuit breaker CLOSED - normal operation restored", component_name)

    async def run_intelligence_processing(self) -> Dict[str, Any]:
        """Run complete ML intelligence processing pipeline with 2025 circuit breaker protection.

        Returns:
            Processing results and statistics
        """
        start_time = time.time()
        logger.info("Starting ML intelligence processing pipeline with circuit breaker protection")

        # Initialize results with fallback values
        results = {
            "status": "partial_success",
            "rules_processed": 0,
            "combinations_generated": 0,
            "patterns_discovered": 0,
            "predictions_generated": 0,
            "cache_cleaned": 0,
            "circuit_breaker_events": [],
            "processing_time_ms": 0,
            "degraded_mode": False
        }

        try:
            # Get database session with circuit breaker protection
            async def get_db_session():
                return get_session_context()

            db_session = await self.database_breaker.call(get_db_session)

            async with db_session:
                # Step 1: Process rule effectiveness intelligence (with circuit breaker)
                rule_results = await self._process_rule_intelligence_protected(db_session)
                results.update(rule_results)

                # Step 2: Generate rule combination intelligence (with circuit breaker)
                combination_results = await self._process_combination_intelligence_protected(db_session)
                results.update(combination_results)

                # Step 3: Discover and cache patterns (with circuit breaker)
                pattern_results = await self._process_pattern_discovery_protected(db_session)
                results.update(pattern_results)

                # Step 4: Generate ML predictions (with circuit breaker)
                prediction_results = await self._process_ml_predictions_protected(db_session)
                results.update(prediction_results)

                # Step 5: Clean expired cache entries (always attempt)
                cleanup_results = await self._cleanup_expired_cache(db_session)
                results.update(cleanup_results)

                # Commit all changes
                await db_session.commit()

                # Determine final status
                if any(results.get("circuit_breaker_events", [])):
                    results["status"] = "partial_success_degraded"
                else:
                    results["status"] = "success"

                processing_time = (time.time() - start_time) * 1000
                results["processing_time_ms"] = processing_time

                # Update statistics
                self.processing_stats.update({
                    "rules_processed": results.get("rules_processed", 0),
                    "combinations_generated": results.get("combinations_generated", 0),
                    "patterns_discovered": results.get("patterns_discovered", 0),
                    "processing_time_ms": processing_time,
                    "last_run": datetime.now(timezone.utc).isoformat(),
                    "degraded_mode": results.get("degraded_mode", False)
                })

                results["statistics"] = self.processing_stats

                logger.info(
                    f"ML intelligence processing completed in {processing_time:.1f}ms "
                    f"(status: {results['status']}, degraded: {results.get('degraded_mode', False)})"
                )
                return results

        except CircuitBreakerOpen as e:
            # Database circuit breaker is open - complete failure
            logger.error("Database circuit breaker open - ML processing completely unavailable: %s", e)
            results.update({
                "status": "failed_circuit_breaker",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "circuit_breaker_events": ["database_unavailable"],
                "degraded_mode": True
            })
            return results
        except Exception as e:
            logger.error("ML intelligence processing failed: %s", e)
            results.update({
                "status": "failed",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            })
            return results

    # Circuit breaker protected methods
    async def _process_rule_intelligence_protected(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Process rule intelligence with circuit breaker protection."""
        try:
            return await self.rule_optimizer_breaker.call(
                self._process_rule_intelligence, db_session
            )
        except CircuitBreakerOpen:
            logger.warning("Rule intelligence processing circuit breaker open - using fallback")
            return {"rules_processed": 0, "circuit_breaker_events": ["rule_intelligence_unavailable"]}

    async def _process_combination_intelligence_protected(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Process combination intelligence with circuit breaker protection."""
        try:
            return await self.rule_optimizer_breaker.call(
                self._process_combination_intelligence, db_session
            )
        except CircuitBreakerOpen:
            logger.warning("Combination intelligence processing circuit breaker open - using fallback")
            return {"combinations_generated": 0, "circuit_breaker_events": ["combination_intelligence_unavailable"]}

    async def _process_pattern_discovery_protected(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Process pattern discovery with circuit breaker protection."""
        try:
            return await self.pattern_discovery_breaker.call(
                self._process_pattern_discovery, db_session
            )
        except CircuitBreakerOpen:
            logger.warning("Pattern discovery circuit breaker open - using fallback")
            return {"patterns_discovered": 0, "circuit_breaker_events": ["pattern_discovery_unavailable"]}

    async def _process_ml_predictions_protected(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Process ML predictions with circuit breaker protection."""
        try:
            return await self.rule_optimizer_breaker.call(
                self._process_ml_predictions, db_session
            )
        except CircuitBreakerOpen:
            logger.warning("ML predictions circuit breaker open - using fallback")
            return {"predictions_generated": 0, "circuit_breaker_events": ["ml_predictions_unavailable"]}

    async def _process_rule_intelligence(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Process rule effectiveness intelligence using ML analysis."""
        logger.info("Processing rule effectiveness intelligence")

        # Get distinct prompt characteristics from recent performance data
        characteristics_query = text("""
            SELECT DISTINCT
                prompt_characteristics,
                prompt_type,
                COUNT(*) as sample_count
            FROM rule_performance
            WHERE created_at > NOW() - INTERVAL '30 days'
              AND prompt_characteristics IS NOT NULL
            GROUP BY prompt_characteristics, prompt_type
            HAVING COUNT(*) >= 5
            ORDER BY sample_count DESC
            LIMIT :batch_size
        """)

        result = await db_session.execute(characteristics_query, {"batch_size": self.batch_size})
        characteristic_groups = result.fetchall()

        rules_processed = 0

        for group in characteristic_groups:
            characteristics_data = group.prompt_characteristics

            # Convert to PromptCharacteristics object
            try:
                characteristics = PromptCharacteristics(**characteristics_data)
            except Exception as e:
                logger.warning("Failed to parse characteristics: %s", e)
                continue

            # Generate characteristics hash
            characteristics_hash = self._hash_characteristics(characteristics)

            # Get rules for this characteristic set
            rules_query = text("""
                SELECT
                    rule_id,
                    rule_name,
                    AVG(improvement_score) as avg_effectiveness,
                    AVG(confidence_level) as avg_confidence,
                    COUNT(*) as sample_size,
                    MAX(created_at) as last_used
                FROM rule_performance
                WHERE prompt_characteristics = :characteristics
                  AND created_at > NOW() - INTERVAL '30 days'
                GROUP BY rule_id, rule_name
                HAVING COUNT(*) >= 3
            """)

            rules_result = await db_session.execute(rules_query, {
                "characteristics": json.dumps(characteristics_data)
            })
            rules = rules_result.fetchall()

            # Process each rule with ML analysis
            for rule in rules:
                await self._generate_rule_intelligence(
                    db_session, rule, characteristics, characteristics_hash
                )
                rules_processed += 1

        return {
            "rules_processed": rules_processed,
            "characteristic_groups": len(characteristic_groups)
        }

    async def _generate_rule_intelligence(
        self,
        db_session: AsyncSession,
        rule_data: Any,
        characteristics: PromptCharacteristics,
        characteristics_hash: str
    ) -> None:
        """Generate comprehensive rule intelligence using ML components."""

        # Calculate enhanced scores using ML insights
        effectiveness_score = float(rule_data.avg_effectiveness)

        # Use pattern discovery for characteristic matching
        try:
            pattern_results = await self.pattern_discovery.discover_advanced_patterns(
                db_session=db_session,
                min_effectiveness=0.6,
                min_support=3,
                pattern_types=["parameter", "performance"],
                use_ensemble=True
            )

            # Extract characteristic match score from patterns
            characteristic_match_score = self._calculate_characteristic_match(
                characteristics, pattern_results
            )

            # Extract pattern insights
            pattern_insights = {
                "parameter_patterns": pattern_results.get("parameter_patterns", {}),
                "performance_patterns": pattern_results.get("performance_patterns", {}),
                "confidence": pattern_results.get("ensemble_analysis", {}).get("confidence", 0.5)
            }

        except Exception as e:
            logger.warning("Pattern discovery failed for rule {rule_data.rule_id}: %s", e)
            characteristic_match_score = 0.5
            pattern_insights = {}

        # Use rule optimizer for recommendations
        try:
            optimization_results = await self.rule_optimizer.optimize_rule(
                rule_id=rule_data.rule_id,
                performance_data={rule_data.rule_id: {
                    "total_applications": rule_data.sample_size,
                    "avg_effectiveness": effectiveness_score,
                    "confidence_level": float(rule_data.avg_confidence)
                }}
            )

            optimization_recommendations = optimization_results.get("recommendations", [])
            performance_trend = optimization_results.get("trend", "stable")

        except Exception as e:
            logger.warning("Rule optimization failed for rule {rule_data.rule_id}: %s", e)
            optimization_recommendations = []
            performance_trend = "stable"

        # Calculate composite scores
        historical_performance_score = min(1.0, effectiveness_score * 1.2)
        recency_score = self._calculate_recency_score(rule_data.last_used)

        # Calculate total score using weighted algorithm
        total_score = (
            effectiveness_score * 0.35 +
            characteristic_match_score * 0.25 +
            historical_performance_score * 0.20 +
            0.5 * 0.15 +  # ML prediction placeholder
            recency_score * 0.05
        )

        # Generate cache key
        cache_key = f"{rule_data.rule_id}_{characteristics_hash}"

        # Store in rule intelligence cache
        insert_query = text("""
            INSERT INTO rule_intelligence_cache (
                cache_key, rule_id, rule_name,
                effectiveness_score, characteristic_match_score,
                historical_performance_score, ml_prediction_score, recency_score,
                total_score, confidence_level, sample_size,
                pattern_insights, optimization_recommendations, performance_trend,
                prompt_characteristics_hash, computed_at, expires_at
            ) VALUES (
                :cache_key, :rule_id, :rule_name,
                :effectiveness_score, :characteristic_match_score,
                :historical_performance_score, :ml_prediction_score, :recency_score,
                :total_score, :confidence_level, :sample_size,
                :pattern_insights, :optimization_recommendations, :performance_trend,
                :characteristics_hash, NOW(), NOW() + INTERVAL ':ttl_hours hours'
            )
            ON CONFLICT (cache_key) DO UPDATE SET
                effectiveness_score = EXCLUDED.effectiveness_score,
                characteristic_match_score = EXCLUDED.characteristic_match_score,
                historical_performance_score = EXCLUDED.historical_performance_score,
                ml_prediction_score = EXCLUDED.ml_prediction_score,
                recency_score = EXCLUDED.recency_score,
                total_score = EXCLUDED.total_score,
                confidence_level = EXCLUDED.confidence_level,
                sample_size = EXCLUDED.sample_size,
                pattern_insights = EXCLUDED.pattern_insights,
                optimization_recommendations = EXCLUDED.optimization_recommendations,
                performance_trend = EXCLUDED.performance_trend,
                computed_at = NOW(),
                expires_at = NOW() + INTERVAL ':ttl_hours hours'
        """)

        await db_session.execute(insert_query, {
            "cache_key": cache_key,
            "rule_id": rule_data.rule_id,
            "rule_name": rule_data.rule_name,
            "effectiveness_score": effectiveness_score,
            "characteristic_match_score": characteristic_match_score,
            "historical_performance_score": historical_performance_score,
            "ml_prediction_score": None,  # Will be filled by ML prediction step
            "recency_score": recency_score,
            "total_score": total_score,
            "confidence_level": float(rule_data.avg_confidence),
            "sample_size": rule_data.sample_size,
            "pattern_insights": json.dumps(pattern_insights),
            "optimization_recommendations": json.dumps(optimization_recommendations),
            "performance_trend": performance_trend,
            "characteristics_hash": characteristics_hash,
            "ttl_hours": self.cache_ttl_hours
        })

    async def _process_combination_intelligence(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Process rule combination intelligence."""
        logger.info("Processing rule combination intelligence")

        # Get successful rule combinations from historical data
        combinations_query = text("""
            SELECT
                rule_set,
                prompt_type,
                AVG(combined_effectiveness) as avg_effectiveness,
                COUNT(*) as sample_size,
                AVG(statistical_confidence) as avg_confidence
            FROM rule_combinations
            WHERE created_at > NOW() - INTERVAL '60 days'
              AND combined_effectiveness >= 0.6
            GROUP BY rule_set, prompt_type
            HAVING COUNT(*) >= 3
            ORDER BY avg_effectiveness DESC
            LIMIT :batch_size
        """)

        result = await db_session.execute(combinations_query, {"batch_size": self.batch_size})
        combinations = result.fetchall()

        combinations_generated = 0

        for combo in combinations:
            # Generate combination intelligence
            await self._generate_combination_intelligence(db_session, combo)
            combinations_generated += 1

        return {"combinations_generated": combinations_generated}

    async def _generate_combination_intelligence(
        self,
        db_session: AsyncSession,
        combination_data: Any
    ) -> None:
        """Generate intelligence for a specific rule combination."""

        rule_set = combination_data.rule_set
        prompt_type = combination_data.prompt_type

        # Generate combination key
        combination_key = f"{json.dumps(sorted(rule_set))}_{prompt_type}"
        combination_hash = hashlib.md5(combination_key.encode()).hexdigest()

        # Analyze synergies and conflicts (simplified for now)
        synergy_pairs = []
        conflict_pairs = []

        # Calculate synergy score (placeholder - would use ML analysis)
        synergy_score = min(1.0, combination_data.avg_effectiveness - 0.5)

        # Store combination intelligence
        insert_query = text("""
            INSERT INTO rule_combination_intelligence (
                combination_key, rule_set, combined_effectiveness,
                synergy_score, individual_scores, synergy_pairs, conflict_pairs,
                sample_size, statistical_confidence, prompt_type,
                computed_at, expires_at
            ) VALUES (
                :combination_key, :rule_set, :combined_effectiveness,
                :synergy_score, :individual_scores, :synergy_pairs, :conflict_pairs,
                :sample_size, :statistical_confidence, :prompt_type,
                NOW(), NOW() + INTERVAL ':ttl_hours hours'
            )
            ON CONFLICT (combination_key) DO UPDATE SET
                combined_effectiveness = EXCLUDED.combined_effectiveness,
                synergy_score = EXCLUDED.synergy_score,
                sample_size = EXCLUDED.sample_size,
                statistical_confidence = EXCLUDED.statistical_confidence,
                computed_at = NOW(),
                expires_at = NOW() + INTERVAL ':ttl_hours hours'
        """)

        await db_session.execute(insert_query, {
            "combination_key": combination_hash,
            "rule_set": json.dumps(rule_set),
            "combined_effectiveness": float(combination_data.avg_effectiveness),
            "synergy_score": synergy_score,
            "individual_scores": json.dumps({}),  # Placeholder
            "synergy_pairs": json.dumps(synergy_pairs),
            "conflict_pairs": json.dumps(conflict_pairs),
            "sample_size": combination_data.sample_size,
            "statistical_confidence": float(combination_data.avg_confidence),
            "prompt_type": prompt_type,
            "ttl_hours": self.cache_ttl_hours
        })

    async def _process_pattern_discovery(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Process pattern discovery and cache results."""
        logger.info("Processing pattern discovery")

        try:
            # Run comprehensive pattern discovery
            pattern_results = await self.pattern_discovery.discover_advanced_patterns(
                db_session=db_session,
                min_effectiveness=0.7,
                min_support=5,
                pattern_types=["parameter", "sequence", "performance", "semantic"],
                use_ensemble=True,
                include_apriori=True
            )

            # Generate discovery key
            discovery_key = f"comprehensive_{int(time.time())}"

            # Store pattern discovery results
            insert_query = text("""
                INSERT INTO pattern_discovery_cache (
                    discovery_key, parameter_patterns, sequence_patterns,
                    performance_patterns, semantic_patterns, apriori_patterns,
                    ensemble_analysis, cross_validation, discovery_method,
                    min_effectiveness, min_support, confidence_level,
                    patterns_found, computed_at, expires_at
                ) VALUES (
                    :discovery_key, :parameter_patterns, :sequence_patterns,
                    :performance_patterns, :semantic_patterns, :apriori_patterns,
                    :ensemble_analysis, :cross_validation, :discovery_method,
                    :min_effectiveness, :min_support, :confidence_level,
                    :patterns_found, NOW(), NOW() + INTERVAL '24 hours'
                )
            """)

            patterns_found = len(pattern_results.get("parameter_patterns", {}).get("clusters", []))

            await db_session.execute(insert_query, {
                "discovery_key": discovery_key,
                "parameter_patterns": json.dumps(pattern_results.get("parameter_patterns", {})),
                "sequence_patterns": json.dumps(pattern_results.get("sequence_patterns", {})),
                "performance_patterns": json.dumps(pattern_results.get("performance_patterns", {})),
                "semantic_patterns": json.dumps(pattern_results.get("semantic_patterns", {})),
                "apriori_patterns": json.dumps(pattern_results.get("apriori_patterns", {})),
                "ensemble_analysis": json.dumps(pattern_results.get("ensemble_analysis", {})),
                "cross_validation": json.dumps(pattern_results.get("cross_validation", {})),
                "discovery_method": "ensemble",
                "min_effectiveness": 0.7,
                "min_support": 5,
                "confidence_level": pattern_results.get("ensemble_analysis", {}).get("confidence", 0.8),
                "patterns_found": patterns_found
            })

            return {
                "patterns_discovered": patterns_found,
                "discovery_key": discovery_key
            }

        except Exception as e:
            logger.error("Pattern discovery failed: %s", e)
            return {"patterns_discovered": 0, "error": str(e)}

    async def _process_ml_predictions(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Process ML predictions for rule effectiveness."""
        # Placeholder for ML prediction processing
        # Would integrate with trained models for effectiveness prediction
        return {"predictions_generated": 0}

    async def _cleanup_expired_cache(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Clean up expired cache entries."""
        cleanup_query = text("SELECT clean_expired_intelligence_cache()")
        result = await db_session.execute(cleanup_query)
        deleted_count = result.scalar()

        logger.info("Cleaned up %s expired cache entries", deleted_count)
        return {"deleted_entries": deleted_count}

    def _hash_characteristics(self, characteristics: PromptCharacteristics) -> str:
        """Generate hash for prompt characteristics."""
        char_data: Dict[str, Any] = {
            "prompt_type": characteristics.prompt_type,
            "domain": characteristics.domain,
            "complexity_level": round(characteristics.complexity_level, 1),
            "length_category": characteristics.length_category,
            "reasoning_required": characteristics.reasoning_required,
            "specificity_level": round(characteristics.specificity_level, 1),
            "task_type": characteristics.task_type
        }

        char_string = json.dumps(char_data, sort_keys=True)
        return hashlib.sha256(char_string.encode()).hexdigest()

    def _calculate_characteristic_match(
        self,
        characteristics: PromptCharacteristics,
        pattern_results: Dict[str, Any]
    ) -> float:
        """Calculate characteristic match score from pattern analysis."""
        # Simplified calculation - would use more sophisticated ML analysis
        base_score = 0.5

        # Boost score based on pattern confidence
        pattern_confidence = pattern_results.get("ensemble_analysis", {}).get("confidence", 0.5)
        return min(1.0, base_score + pattern_confidence * 0.3)

    def _calculate_recency_score(self, last_used: datetime) -> float:
        """Calculate recency score based on last usage."""
        if not last_used:
            return 0.3

        days_ago = (datetime.now(timezone.utc) - last_used).days

        if days_ago <= 7:
            return 1.0
        elif days_ago <= 30:
            return 0.8
        elif days_ago <= 90:
            return 0.5
        else:
            return 0.2

    # 2025 Performance-Optimized Background Service Methods
    def _calculate_batch_ranges(self, total_rules: int) -> List[Dict[str, int]]:
        """Calculate optimal batch ranges for parallel processing."""
        # Calculate batch size to ensure we don't exceed max_parallel_workers
        batch_size = max(1, total_rules // self.max_parallel_workers)
        if total_rules % self.max_parallel_workers != 0:
            batch_size += 1  # Round up to ensure all rules are covered

        batches: List[Dict[str, int]] = []

        for i in range(0, total_rules, batch_size):
            batches.append({
                "start_offset": i,
                "batch_size": min(batch_size, total_rules - i),
                "batch_id": len(batches)
            })

            # Limit to max_parallel_workers batches
            if len(batches) >= self.max_parallel_workers:
                break

        return batches

    async def _check_incremental_update_needed(
        self,
        db_session: AsyncSession,
        rule_id: str
    ) -> bool:
        """Check if rule needs incremental update based on performance changes."""
        # Check last update time and performance changes
        check_query = text("""
            SELECT
                ric.updated_at,
                ric.total_score as cached_score,
                AVG(rp.improvement_score) as current_avg_score,
                COUNT(rp.id) as recent_samples
            FROM rule_intelligence_cache ric
            LEFT JOIN rule_performance rp ON ric.rule_id = rp.rule_id
                AND rp.created_at > ric.updated_at
            WHERE ric.rule_id = :rule_id
            GROUP BY ric.rule_id, ric.updated_at, ric.total_score
        """)

        result = await db_session.execute(check_query, {"rule_id": rule_id})
        cache_data = result.fetchone()

        if not cache_data:
            return True  # No cache entry, needs processing

        # Check if enough new samples or significant score change
        recent_samples = cache_data.recent_samples or 0
        if recent_samples < 5:
            return False  # Not enough new data

        cached_score = cache_data.cached_score or 0
        current_score = cache_data.current_avg_score or 0

        # Check if score changed beyond threshold
        score_change = abs(current_score - cached_score) / max(cached_score, 0.1)

        return score_change > self.incremental_update_threshold

    async def _generate_ml_predictions_with_confidence(
        self,
        db_session: AsyncSession,
        rule_data: Any,
        characteristics: PromptCharacteristics
    ) -> Dict[str, Any]:
        """Generate ML predictions with confidence scoring for 2025 pipeline."""
        if not self.confidence_scoring_enabled:
            return {"confidence": 0.5, "predictions": {}}

        try:
            # Use existing ML components for prediction
            rule_id = rule_data.rule_id

            # Get historical performance for confidence calculation
            performance_query = text("""
                SELECT
                    improvement_score,
                    effectiveness_rating,
                    created_at
                FROM rule_performance
                WHERE rule_id = :rule_id
                  AND created_at > NOW() - INTERVAL '90 days'
                ORDER BY created_at DESC
                LIMIT 100
            """)

            result = await db_session.execute(performance_query, {"rule_id": rule_id})
            performance_data = result.fetchall()

            if len(performance_data) < 5:
                return {"confidence": 0.3, "predictions": {"insufficient_data": True}}

            # Calculate prediction confidence based on data quality
            scores = [float(p.improvement_score) for p in performance_data if p.improvement_score is not None]

            if not scores:
                return {"confidence": 0.2, "predictions": {"no_valid_scores": True}}

            # Statistical confidence calculation
            import statistics
            score_mean = statistics.mean(scores)
            score_stdev = statistics.stdev(scores) if len(scores) > 1 else 0.5

            # Confidence based on sample size, recency, and consistency
            sample_confidence = min(1.0, len(scores) / 50)  # More samples = higher confidence
            consistency_confidence = max(0.1, 1.0 - (score_stdev / max(score_mean, 0.1)))
            recency_confidence = min(1.0, len([p for p in performance_data if
                                             (datetime.now() - p.created_at).days < 30]) / 20)

            overall_confidence = (sample_confidence + consistency_confidence + recency_confidence) / 3

            # Generate predictions using ML components
            predictions: Dict[str, Any] = {
                "effectiveness_prediction": score_mean,
                "confidence_interval": score_stdev * 1.96,  # 95% confidence interval
                "trend_direction": "improving" if len(scores) > 10 and scores[0] > scores[-1] else "stable",
                "sample_size": len(scores),
                "data_quality_score": overall_confidence
            }

            return {
                "confidence": max(self.min_confidence_threshold, overall_confidence),
                "predictions": predictions
            }

        except Exception as e:
            logger.warning("ML prediction generation failed for rule {rule_data.rule_id}: %s", e)
            return {"confidence": 0.1, "predictions": {"error": str(e)}}

    async def _process_batch_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        batch_info: Dict[str, int],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Process a single batch with semaphore control for parallel execution."""
        async with semaphore:
            batch_id = batch_info["batch_id"]
            start_offset = batch_info["start_offset"]
            batch_size = batch_info["batch_size"]

            logger.info("Processing batch {batch_id}: offset {start_offset}, size %s", batch_size)

            try:
                # Process rules in this batch
                batch_query = text("""
                    SELECT DISTINCT
                        rule_id,
                        rule_name,
                        prompt_characteristics,
                        prompt_type
                    FROM rule_performance
                    WHERE created_at > NOW() - INTERVAL '30 days'
                      AND prompt_characteristics IS NOT NULL
                    ORDER BY rule_id
                    LIMIT :batch_size OFFSET :start_offset
                """)

                result = await db_session.execute(batch_query, {
                    "batch_size": batch_size,
                    "start_offset": start_offset
                })

                batch_rules = result.fetchall()
                rules_processed = 0

                for rule_data in batch_rules:
                    try:
                        # Process individual rule with incremental updates
                        await self._process_rule_with_incremental_update(db_session, rule_data)
                        rules_processed += 1
                    except Exception as e:
                        logger.warning("Failed to process rule {rule_data.rule_id} in batch {batch_id}: %s", e)
                        continue

                logger.info("Batch {batch_id} completed: %s rules processed", rules_processed)
                return {
                    "batch_id": batch_id,
                    "rules_processed": rules_processed,
                    "status": "success"
                }

            except Exception as e:
                logger.error("Batch {batch_id} failed: %s", e)
                return {
                    "batch_id": batch_id,
                    "rules_processed": 0,
                    "status": "failed",
                    "error": str(e)
                }

    async def _process_rule_with_incremental_update(
        self,
        db_session: AsyncSession,
        rule_data: Any
    ) -> None:
        """Process rule with incremental update logic to avoid unnecessary recomputation."""
        rule_id = rule_data.rule_id

        # Check if rule needs update based on incremental threshold
        needs_update = await self._check_incremental_update_needed(db_session, rule_id)

        if not needs_update:
            logger.debug("Rule %s skipped - no significant changes detected", rule_id)
            return

        # Process rule with ML prediction pipeline
        try:
            characteristics = PromptCharacteristics(**rule_data.prompt_characteristics)
            characteristics_hash = self._hash_characteristics(characteristics)

            # Generate ML predictions with confidence scoring
            ml_predictions = await self._generate_ml_predictions_with_confidence(
                db_session, rule_data, characteristics
            )

            # Update rule intelligence cache with new predictions
            await self._update_rule_intelligence_incremental(
                db_session, rule_data, characteristics_hash, ml_predictions
            )

            logger.debug("Rule {rule_id} updated with ML predictions (confidence: %s)", ml_predictions.get('confidence', 0):.2f)

        except Exception as e:
            logger.warning("Failed to process rule {rule_id} incrementally: %s", e)

    async def _update_rule_intelligence_incremental(
        self,
        db_session: AsyncSession,
        rule_data: Any,
        characteristics_hash: str,
        ml_predictions: Dict[str, Any]
    ) -> None:
        """Update rule intelligence cache with incremental ML predictions."""
        confidence = ml_predictions.get("confidence", 0.5)
        predictions = ml_predictions.get("predictions", {})

        # Only update if confidence meets threshold
        if confidence < self.min_confidence_threshold:
            logger.debug("Skipping update for rule {rule_data.rule_id} - confidence too low: %s", confidence:.2f)
            return

        update_query = text("""
            UPDATE rule_intelligence_cache
            SET
                ml_prediction_score = :ml_score,
                confidence_level = :confidence,
                pattern_insights = :insights,
                updated_at = NOW(),
                expires_at = NOW() + INTERVAL ':ttl_hours hours'
            WHERE rule_id = :rule_id
              AND prompt_characteristics_hash = :characteristics_hash
        """)

        await db_session.execute(update_query, {
            "rule_id": rule_data.rule_id,
            "characteristics_hash": characteristics_hash,
            "ml_score": predictions.get("effectiveness_prediction", 0.5),
            "confidence": confidence,
            "insights": json.dumps(predictions),
            "ttl_hours": self.cache_ttl_hours
        })

    async def run_parallel_batch_processing(self) -> Dict[str, Any]:
        """Run parallel batch processing with 2025 performance optimization patterns."""
        start_time = time.time()
        logger.info("Starting parallel batch processing with %s workers", self.max_parallel_workers)

        try:
            async with get_session_context() as db_session:
                # Get total rule count for batch planning
                total_rules_query = text("""
                    SELECT COUNT(DISTINCT rule_id) as total_rules
                    FROM rule_performance
                    WHERE created_at > NOW() - INTERVAL '30 days'
                """)
                result = await db_session.execute(total_rules_query)
                total_rules = result.scalar() or 0

                if total_rules == 0:
                    logger.warning("No rules found for processing")
                    return {"status": "no_data", "rules_processed": 0}

                # Calculate batch ranges for parallel processing
                batches = self._calculate_batch_ranges(total_rules)
                logger.info("Processing {total_rules} rules in %s parallel batches", len(batches))

                # Create semaphore to limit concurrent workers
                semaphore = asyncio.Semaphore(self.max_parallel_workers)

                # Process batches in parallel
                tasks = [
                    self._process_batch_with_semaphore(semaphore, batch_info, db_session)
                    for batch_info in batches
                ]

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Aggregate results
                total_processed = 0
                total_errors = 0

                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error("Batch {i} failed: %s", result)
                        total_errors += 1
                    elif isinstance(result, dict):
                        total_processed += result.get("rules_processed", 0)
                    else:
                        logger.warning("Batch {i} returned unexpected result type: %s", type(result))
                        total_errors += 1

                processing_time = (time.time() - start_time) * 1000

                return {
                    "status": "success" if total_errors == 0 else "partial_success",
                    "rules_processed": total_processed,
                    "batches_processed": len(batches),
                    "batch_errors": total_errors,
                    "processing_time_ms": processing_time,
                    "parallel_workers": self.max_parallel_workers
                }

        except Exception as e:
            logger.error("Parallel batch processing failed: %s", e)
            return {
                "status": "failed",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }


# Background service runner
async def run_intelligence_processor():
    """Run the ML intelligence processor as a background service."""
    processor = MLIntelligenceProcessor()

    while True:
        try:
            results = await processor.run_intelligence_processing()
            logger.info("Intelligence processing completed: %s", results['status'])

            # Wait for next processing cycle
            await asyncio.sleep(processor.processing_interval_hours * 3600)

        except Exception as e:
            logger.error("Intelligence processing failed: %s", e)
            # Wait shorter time on error before retrying
            await asyncio.sleep(300)  # 5 minutes





if __name__ == "__main__":
    # Run as standalone background service
    asyncio.run(run_intelligence_processor())
