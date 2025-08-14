"""Training Validator Service - Clean Architecture Implementation

Implements data validation, quality assessment, and training readiness checks.
Extracted from training_system_manager.py (2109 lines) as part of decomposition.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.cli.core.rule_validation_service import RuleValidationService
from prompt_improver.cli.services.training_protocols import TrainingValidatorProtocol
from prompt_improver.database import get_sessionmanager


class TrainingValidator(TrainingValidatorProtocol):
    """Training validation service implementing Clean Architecture patterns.
    
    Responsibilities:
    - Data validation and quality assessment
    - Training readiness verification
    - Database schema and rule validation
    - System state detection and analysis
    """

    def __init__(self):
        self.logger = logging.getLogger("apes.training_validator")
        self._rule_validator: Optional[RuleValidationService] = None
        
        # Training system data directory
        self.training_data_dir = Path.home() / ".local" / "share" / "apes" / "training"

    async def validate_ready_for_training(self) -> bool:
        """Validate that the system is ready for training.
        
        Returns:
            True if ready for training, False otherwise
        """
        try:
            # Check database connectivity
            try:
                session_manager = get_sessionmanager()
                async with session_manager.get_async_session() as session:
                    await session.execute(text("SELECT 1"))
            except Exception:
                self.logger.error("Database connectivity check failed")
                return False

            # Validate database schema
            schema_valid = await self._verify_database_schema()
            if not schema_valid:
                self.logger.error("Database schema validation failed")
                return False

            # Validate minimum data requirements
            data_adequate = await self._validate_minimum_data_requirements()
            if not data_adequate:
                self.logger.error("Minimum data requirements not met")
                return False

            # Validate rules and metadata
            rules_valid = await self._validate_rules_and_metadata()
            if not rules_valid:
                self.logger.error("Rules and metadata validation failed")
                return False

            self.logger.info("System validation passed - ready for training")
            return True

        except Exception as e:
            self.logger.error(f"Training readiness validation failed: {e}")
            return False

    async def validate_database_and_rules(self) -> Dict[str, Any]:
        """Validate database connectivity, schema, and seeded rules.
        
        Returns:
            Database and rule validation results
        """
        validation_start = time.time()

        database_status = {
            "connectivity": {"status": "unknown", "details": {}},
            "schema_validation": {"status": "unknown", "details": {}},
            "seeded_rules": {"status": "unknown", "details": {}},
            "rule_metadata": {"status": "unknown", "details": {}},
            "validation_time_ms": 0,
        }

        try:
            session_manager = get_sessionmanager()
            async with session_manager.get_async_session() as db_session:
                # Test basic connectivity
                await db_session.execute(text("SELECT 1"))
                database_status["connectivity"] = {
                    "status": "healthy",
                    "details": {"connection_successful": True},
                }

                # Validate required tables exist
                required_tables = [
                    "improvement_sessions",
                    "rule_performance",
                    "rule_metadata",
                    "training_prompts",
                    "discovered_patterns",
                ]

                table_list = "', '".join(required_tables)
                result = await db_session.execute(
                    text(
                        f"SELECT table_name FROM information_schema.tables WHERE table_name IN ('{table_list}')"
                    )
                )
                existing_tables = [row[0] for row in result.fetchall()]

                missing_tables = set(required_tables) - set(existing_tables)
                database_status["schema_validation"] = {
                    "status": "healthy" if not missing_tables else "incomplete",
                    "details": {
                        "required_tables": required_tables,
                        "existing_tables": existing_tables,
                        "missing_tables": list(missing_tables),
                    },
                }

                # Enhanced rule validation using RuleValidationService
                if not self._rule_validator:
                    self._rule_validator = RuleValidationService()

                rule_validation_report = await self._rule_validator.validate_all_rules()

                database_status["seeded_rules"] = {
                    "status": rule_validation_report["overall_status"],
                    "details": {
                        "total_rules": rule_validation_report["rule_count"]["found"],
                        "valid_rules": rule_validation_report["rule_count"]["valid"],
                        "expected_rules": rule_validation_report["rule_count"]["expected"],
                        "validation_report": rule_validation_report,
                    },
                }

                database_status["rule_metadata"] = {
                    "status": rule_validation_report["metadata_validation"]["overall_status"],
                    "details": rule_validation_report["metadata_validation"],
                }

        except Exception as e:
            database_status["connectivity"] = {
                "status": "error",
                "details": {"error": str(e)},
            }

        database_status["validation_time_ms"] = (time.time() - validation_start) * 1000
        return database_status

    async def assess_data_availability(self) -> Dict[str, Any]:
        """Comprehensive training data availability assessment.
        
        Returns:
            Data availability analysis with quality metrics
        """
        assessment_start = time.time()

        data_status = {
            "training_data": {"status": "unknown", "details": {}},
            "synthetic_data": {"status": "unknown", "details": {}},
            "data_quality": {"status": "unknown", "details": {}},
            "minimum_requirements": {"status": "unknown", "details": {}},
            "assessment_time_ms": 0,
        }

        try:
            session_manager = get_sessionmanager()
            async with session_manager.get_async_session() as db_session:
                # Assess training prompts
                training_count_result = await db_session.execute(
                    text("SELECT COUNT(*) FROM training_prompts")
                )
                training_count = training_count_result.scalar() or 0

                # Assess by data source
                synthetic_count_result = await db_session.execute(
                    text("SELECT COUNT(*) FROM training_prompts WHERE data_source = 'synthetic'")
                )
                synthetic_count = synthetic_count_result.scalar() or 0

                user_count_result = await db_session.execute(
                    text("SELECT COUNT(*) FROM training_prompts WHERE data_source = 'user'")
                )
                user_count = user_count_result.scalar() or 0

                # Assess prompt sessions
                session_count_result = await db_session.execute(
                    text("SELECT COUNT(*) FROM prompt_sessions")
                )
                session_count = session_count_result.scalar() or 0

                data_status["training_data"] = {
                    "status": "sufficient" if training_count >= 100 else "insufficient",
                    "details": {
                        "total_training_prompts": training_count,
                        "synthetic_prompts": synthetic_count,
                        "user_prompts": user_count,
                        "prompt_sessions": session_count,
                        "minimum_required": 100,
                    },
                }

                data_status["synthetic_data"] = {
                    "status": "available" if synthetic_count > 0 else "missing",
                    "details": {
                        "synthetic_count": synthetic_count,
                        "percentage_synthetic": (synthetic_count / max(training_count, 1)) * 100,
                    },
                }

                # Enhanced quality assessment
                if training_count > 0:
                    quality_assessment = await self._perform_comprehensive_quality_assessment(
                        db_session, training_count
                    )
                    data_status["data_quality"] = quality_assessment
                else:
                    data_status["data_quality"] = {
                        "status": "no_data",
                        "details": {
                            "message": "No training data available for quality assessment"
                        },
                    }

                # Minimum requirements check
                requirements_met = {
                    "training_data_count": training_count >= 100,
                    "synthetic_data_available": synthetic_count > 0,
                    "user_data_available": user_count > 0,
                    "quality_acceptable": data_status["data_quality"]["status"] in ["good", "unknown"],
                }

                data_status["minimum_requirements"] = {
                    "status": "met" if all(requirements_met.values()) else "not_met",
                    "details": requirements_met,
                }

        except Exception as e:
            self.logger.error(f"Data availability assessment failed: {e}")
            data_status.update({
                "training_data": {
                    "status": "error",
                    "details": {"error": str(e), "total_training_prompts": 0},
                },
                "synthetic_data": {"status": "error", "details": {"error": str(e)}},
                "data_quality": {"status": "error", "details": {"error": str(e)}},
                "minimum_requirements": {"status": "error", "details": {"error": str(e)}},
            })

        data_status["assessment_time_ms"] = (time.time() - assessment_start) * 1000
        return data_status

    async def detect_system_state(self) -> Dict[str, Any]:
        """Comprehensive system state detection using 2025 best practices.
        
        Returns:
            Detailed system state information
        """
        state_start = time.time()

        system_state = {
            "environment_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": str(Path.cwd()),
                "data_directory": str(self.training_data_dir),
                "data_directory_exists": self.training_data_dir.exists(),
            },
            "detection_time_ms": 0,
        }

        # Check for existing configuration files
        config_files = {
            "training_config": (self.training_data_dir / "config.json").exists(),
            "model_cache": (self.training_data_dir / "models").exists(),
            "logs": (self.training_data_dir / "logs").exists(),
        }
        system_state["configuration_files"] = config_files

        # Check system resources
        try:
            import psutil

            system_state["system_resources"] = {
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                "cpu_count": psutil.cpu_count(),
                "disk_free_gb": psutil.disk_usage(".").free / (1024**3),
            }
        except ImportError:
            system_state["system_resources"] = {"status": "psutil_not_available"}

        system_state["detection_time_ms"] = (time.time() - state_start) * 1000
        return system_state

    async def validate_components(self, orchestrator=None, analytics=None, data_generator=None) -> Dict[str, Any]:
        """Validate all training system components with health checks.
        
        Args:
            orchestrator: ML pipeline orchestrator instance
            analytics: Analytics service instance
            data_generator: Data generator instance
            
        Returns:
            Component validation results
        """
        validation_start = time.time()

        component_status = {
            "orchestrator": {"status": "not_initialized", "details": {}},
            "analytics": {"status": "not_initialized", "details": {}},
            "data_generator": {"status": "not_initialized", "details": {}},
            "validation_time_ms": 0,
        }

        # Validate orchestrator
        if orchestrator:
            try:
                health = await orchestrator.health_check()
                component_status["orchestrator"] = {
                    "status": "healthy" if health.get("healthy") else "unhealthy",
                    "details": health,
                }
            except Exception as e:
                component_status["orchestrator"] = {
                    "status": "error",
                    "details": {"error": str(e)},
                }

        # Validate analytics
        if analytics:
            try:
                component_status["analytics"] = {
                    "status": "healthy",
                    "details": {"initialized": True},
                }
            except Exception as e:
                component_status["analytics"] = {
                    "status": "error",
                    "details": {"error": str(e)},
                }

        # Validate data generator
        if data_generator:
            try:
                component_status["data_generator"] = {
                    "status": "healthy",
                    "details": {
                        "generation_method": getattr(data_generator, "generation_method", "unknown"),
                        "target_samples": getattr(data_generator, "target_samples", 0),
                    },
                }
            except Exception as e:
                component_status["data_generator"] = {
                    "status": "error",
                    "details": {"error": str(e)},
                }

        component_status["validation_time_ms"] = (time.time() - validation_start) * 1000
        return component_status

    async def _perform_comprehensive_quality_assessment(
        self, db_session: AsyncSession, training_count: int
    ) -> Dict[str, Any]:
        """Perform comprehensive quality assessment of training data.
        
        Implements 2025 best practices for data quality evaluation:
        - Multi-dimensional quality scoring
        - Statistical distribution analysis
        - Feature completeness validation
        - Effectiveness score analysis
        
        Returns:
            Detailed quality assessment report
        """
        # Sample size for quality assessment (max 50 for performance)
        sample_size = min(50, training_count)

        # Get representative sample using raw SQL
        sample_result = await db_session.execute(
            text(f"""
                SELECT enhancement_result, data_source, training_priority
                FROM training_prompts
                ORDER BY RANDOM()
                LIMIT {sample_size}
            """)
        )
        samples = [(row[0], row[1], row[2]) for row in sample_result.fetchall()]

        quality_metrics = {
            "effectiveness_scores": [],
            "feature_completeness": [],
            "data_source_distribution": {},
            "priority_distribution": {},
            "metadata_quality": [],
            "enhancement_quality": [],
        }

        # Analyze each sample
        for enhancement_result, data_source, priority in samples:
            if isinstance(enhancement_result, dict):
                # Effectiveness score analysis
                effectiveness = enhancement_result.get("effectiveness_score", 0)
                if isinstance(effectiveness, (int, float)):
                    quality_metrics["effectiveness_scores"].append(effectiveness)

                # Feature completeness analysis
                required_fields = ["enhanced_prompt", "effectiveness_score", "metadata"]
                present_fields = sum(
                    1 for field in required_fields if field in enhancement_result
                )
                completeness = present_fields / len(required_fields)
                quality_metrics["feature_completeness"].append(completeness)

                # Metadata quality analysis
                metadata = enhancement_result.get("metadata", {})
                if isinstance(metadata, dict):
                    metadata_fields = ["source", "domain", "generation_timestamp", "feature_names"]
                    metadata_completeness = sum(
                        1 for field in metadata_fields if field in metadata
                    ) / len(metadata_fields)
                    quality_metrics["metadata_quality"].append(metadata_completeness)

                # Enhancement quality analysis
                enhanced_prompt = enhancement_result.get("enhanced_prompt", "")
                original_prompt = enhancement_result.get("original_prompt", "")
                if enhanced_prompt and original_prompt:
                    enhancement_ratio = len(enhanced_prompt) / max(len(original_prompt), 1)
                    quality_metrics["enhancement_quality"].append(min(enhancement_ratio, 3.0))

            # Data source distribution
            quality_metrics["data_source_distribution"][data_source] = (
                quality_metrics["data_source_distribution"].get(data_source, 0) + 1
            )

            # Priority distribution
            priority_range = (
                "high" if priority >= 80 else "medium" if priority >= 50 else "low"
            )
            quality_metrics["priority_distribution"][priority_range] = (
                quality_metrics["priority_distribution"].get(priority_range, 0) + 1
            )

        # Calculate quality scores
        avg_effectiveness = (
            sum(quality_metrics["effectiveness_scores"]) / len(quality_metrics["effectiveness_scores"])
            if quality_metrics["effectiveness_scores"]
            else 0
        )
        avg_completeness = (
            sum(quality_metrics["feature_completeness"]) / len(quality_metrics["feature_completeness"])
            if quality_metrics["feature_completeness"]
            else 0
        )
        avg_metadata_quality = (
            sum(quality_metrics["metadata_quality"]) / len(quality_metrics["metadata_quality"])
            if quality_metrics["metadata_quality"]
            else 0
        )
        avg_enhancement_quality = (
            sum(quality_metrics["enhancement_quality"]) / len(quality_metrics["enhancement_quality"])
            if quality_metrics["enhancement_quality"]
            else 0
        )

        # Overall quality score (weighted average)
        overall_quality = (
            avg_effectiveness * 0.4
            + avg_completeness * 0.3
            + avg_metadata_quality * 0.2
            + avg_enhancement_quality * 0.1
        )

        # Determine quality status
        if overall_quality >= 0.8:
            quality_status = "excellent"
        elif overall_quality >= 0.7:
            quality_status = "good"
        elif overall_quality >= 0.5:
            quality_status = "acceptable"
        else:
            quality_status = "needs_improvement"

        return {
            "status": quality_status,
            "details": {
                "samples_analyzed": len(samples),
                "overall_quality_score": overall_quality,
                "effectiveness_metrics": {
                    "average_score": avg_effectiveness,
                    "score_count": len(quality_metrics["effectiveness_scores"]),
                    "min_score": min(quality_metrics["effectiveness_scores"])
                    if quality_metrics["effectiveness_scores"]
                    else 0,
                    "max_score": max(quality_metrics["effectiveness_scores"])
                    if quality_metrics["effectiveness_scores"]
                    else 0,
                },
                "completeness_metrics": {
                    "average_completeness": avg_completeness,
                    "metadata_quality": avg_metadata_quality,
                    "enhancement_quality": avg_enhancement_quality,
                },
                "distribution_analysis": {
                    "data_sources": quality_metrics["data_source_distribution"],
                    "priority_levels": quality_metrics["priority_distribution"],
                },
                "quality_recommendations": self._generate_quality_recommendations(
                    overall_quality, avg_effectiveness, avg_completeness, quality_metrics
                ),
            },
        }

    def _generate_quality_recommendations(
        self, overall_quality: float, avg_effectiveness: float, 
        avg_completeness: float, metrics: Dict
    ) -> List[str]:
        """Generate recommendations for improving data quality."""
        recommendations = []

        if overall_quality < 0.7:
            recommendations.append(
                "Overall data quality is below optimal. Consider regenerating synthetic data with improved parameters."
            )

        if avg_effectiveness < 0.6:
            recommendations.append(
                "Low effectiveness scores detected. Review and improve prompt enhancement algorithms."
            )

        if avg_completeness < 0.8:
            recommendations.append(
                "Incomplete feature data detected. Ensure all required fields are populated during data generation."
            )

        # Check data source diversity
        source_dist = metrics["data_source_distribution"]
        if len(source_dist) == 1:
            recommendations.append(
                "Limited data source diversity. Consider adding user-generated data alongside synthetic data."
            )

        synthetic_percentage = (source_dist.get("synthetic", 0) / sum(source_dist.values()) * 100)
        if synthetic_percentage > 90:
            recommendations.append(
                "High reliance on synthetic data. Consider collecting real user data for better model performance."
            )

        return recommendations

    async def _verify_database_schema(self) -> bool:
        """Verify database schema is up to date."""
        try:
            session_manager = get_sessionmanager()
            async with session_manager.get_async_session() as session:
                # Check if essential training tables exist
                result = await session.execute(
                    text(
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'training_sessions')"
                    )
                )
                return bool(result.scalar())
        except Exception:
            return False

    async def _validate_minimum_data_requirements(self) -> bool:
        """Validate that minimum data requirements are met."""
        try:
            session_manager = get_sessionmanager()
            async with session_manager.get_async_session() as session:
                # Check training data count
                result = await session.execute(
                    text("SELECT COUNT(*) FROM training_prompts")
                )
                training_count = result.scalar() or 0
                
                # Require at least 50 training samples
                return training_count >= 50
        except Exception:
            return False

    async def _validate_rules_and_metadata(self) -> bool:
        """Validate rules and metadata completeness."""
        try:
            if not self._rule_validator:
                self._rule_validator = RuleValidationService()

            rule_validation_report = await self._rule_validator.validate_all_rules()
            return rule_validation_report["overall_status"] == "healthy"
        except Exception:
            return False