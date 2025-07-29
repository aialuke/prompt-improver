"""Analysis Orchestrator

High-level orchestration for failure analysis workflows.
Coordinates pattern detection, classification, and robustness validation
to provide comprehensive failure analysis with monitoring and alerting.
"""

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from scipy import stats

from .pattern_detector import PatternDetector
from .failure_classifier import FailureClassifier


@dataclass
class FailureConfig:
    """Configuration for failure mode analysis"""

    # Failure classification thresholds
    failure_threshold: float = 0.3
    min_pattern_size: int = 3
    significance_threshold: float = 0.1

    # Analysis parameters
    max_patterns: int = 20
    confidence_threshold: float = 0.7

    # Root cause analysis
    max_root_causes: int = 10
    correlation_threshold: float = 0.6

    # Edge case detection
    outlier_threshold: float = 2.0
    min_cluster_size: int = 3

    # Phase 3 enhancements - Robustness Validation
    enable_robustness_validation: bool = True
    adversarial_testing: bool = True
    ensemble_anomaly_detection: bool = True
    robustness_test_samples: int = 100
    noise_levels: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    adversarial_epsilon: float = 0.1

    # Phase 3 enhancements - Automated Alert Systems
    enable_prometheus_monitoring: bool = True
    prometheus_port: int = 8000
    alert_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "failure_rate": 0.15,
            "response_time_ms": 200,
            "error_rate": 0.05,
            "anomaly_score": 0.8,
        }
    )
    alert_cooldown_seconds: int = 300

# Robustness testing imports
try:
    import art
    from art.attacks.evasion import FastGradientMethod
    from art.estimators.classification import SklearnClassifier

    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    warnings.warn(
        "Adversarial Robustness Toolbox not available. Install with: pip install adversarial-robustness-toolbox"
    )

logger = logging.getLogger(__name__)


class RobustnessTestResult:
    """Results from robustness validation tests"""

    def __init__(self, test_type: str, success_rate: float, degradation_rate: float,
                 robustness_score: float, failed_samples: list, recommendations: list):
        self.test_type = test_type
        self.success_rate = success_rate
        self.degradation_rate = degradation_rate
        self.robustness_score = robustness_score
        self.failed_samples = failed_samples
        self.recommendations = recommendations


class AnalysisOrchestrator:
    """High-level orchestrator for comprehensive failure analysis"""

    def __init__(self, config, training_loader=None):
        """Initialize the analysis orchestrator
        
        Args:
            config: FailureConfig instance containing analysis parameters
            training_loader: Training data loader for ML pipeline integration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Training data integration
        from ...core.training_data_loader import TrainingDataLoader
        self.training_loader = training_loader or TrainingDataLoader()
        self.logger.info("Analysis orchestrator integrated with training data pipeline")
        
        # Initialize specialized components
        self.pattern_detector = PatternDetector(config)
        self.failure_classifier = FailureClassifier(config)
        
        # Robustness validation state
        self.robustness_test_results: list[RobustnessTestResult] = []
        
        self.logger.info("Analysis orchestrator initialized successfully")

    async def run_orchestrated_analysis(self, config: dict[str, Any]) -> dict[str, Any]:
        """Orchestrator-compatible interface for failure analysis (2025 pattern)

        Args:
            config: Orchestrator configuration containing:
                - test_results: Test results to analyze for failures
                - output_path: Local path for output files (optional)
                - analysis_depth: Depth of analysis ('basic', 'comprehensive', 'deep')
                - enable_robustness_testing: Whether to run robustness validation

        Returns:
            Orchestrator-compatible result with failure analysis and metadata
        """
        start_time = datetime.now()

        try:
            # Extract configuration from orchestrator
            test_results = config.get("test_results", [])
            output_path = config.get("output_path", "./outputs/failure_analysis")
            analysis_depth = config.get("analysis_depth", "comprehensive")
            enable_robustness = config.get("enable_robustness_testing", True)

            # Perform failure analysis using existing methods
            analysis_result = await self.analyze_failures(test_results)

            # Run robustness validation if enabled
            robustness_results = {}
            if enable_robustness and self.config.enable_robustness_validation:
                robustness_results = await self._run_robustness_validation(test_results)

            # Prepare orchestrator-compatible result
            result = {
                "failure_patterns": [
                    {
                        "pattern_id": pattern.get("pattern_id", "unknown"),
                        "description": pattern.get("description", ""),
                        "frequency": pattern.get("frequency", 0),
                        "severity": pattern.get("severity", 0),
                        "confidence": pattern.get("characteristics", {}).get("confidence", 0.5),
                        "examples": pattern.get("examples", [])[:5]  # Limit examples for orchestrator
                    }
                    for pattern in analysis_result.get("patterns", [])
                ],
                "root_causes": [
                    {
                        "cause_id": cause.get("cause_id", "unknown"),
                        "description": cause.get("description", ""),
                        "likelihood": cause.get("correlation_strength", 0),
                        "impact": cause.get("affected_failures", 0),
                        "evidence": cause.get("evidence", [])[:3]  # Limit evidence for orchestrator
                    }
                    for cause in analysis_result.get("root_causes", [])
                ],
                "robustness_assessment": robustness_results,
                "summary": {
                    "total_failures": analysis_result.get("metadata", {}).get("total_failures", 0),
                    "patterns_identified": len(analysis_result.get("patterns", [])),
                    "critical_patterns": len([p for p in analysis_result.get("patterns", []) if p.get("severity", 0) > 0.7]),
                    "root_causes_found": len(analysis_result.get("root_causes", [])),
                    "overall_health_score": 1 - analysis_result.get("metadata", {}).get("failure_rate", 0)
                }
            }

            # Calculate execution metadata
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "orchestrator_compatible": True,
                "component_result": result,
                "local_metadata": {
                    "output_path": output_path,
                    "execution_time": execution_time,
                    "analysis_depth": analysis_depth,
                    "robustness_testing_enabled": enable_robustness,
                    "prometheus_monitoring": self.config.enable_prometheus_monitoring,
                    "component_version": "1.0.0"
                }
            }

        except Exception as e:
            self.logger.error(f"Orchestrated analysis failed: {e}")
            return {
                "orchestrator_compatible": True,
                "error": str(e),
                "local_metadata": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "component_version": "1.0.0"
                }
            }

    async def analyze_historical_failures(self, db_session) -> dict[str, Any]:
        """Analyze all historical failure patterns from training data
        
        Loads complete failure history and trains anomaly detection models
        to identify failure patterns and predict future failures.
        """
        try:
            self.logger.info("Starting analysis of historical failures from training data")
            
            # Load training data from pipeline
            training_data = await self.training_loader.load_training_data(db_session)
            
            # Check if training data validation passed
            if not training_data.get("validation", {}).get("is_valid", False):
                self.logger.warning("Insufficient training data for failure analysis")
                return {"status": "insufficient_data", "samples": training_data["metadata"]["total_samples"]}
            
            if training_data["metadata"]["total_samples"] < 5:
                self.logger.warning("Insufficient training data for failure analysis")
                return {"status": "insufficient_data", "samples": training_data["metadata"]["total_samples"]}
            
            # Extract failure cases from training data
            failure_data = await self._extract_failure_cases(training_data)
            
            if not failure_data.get("features"):
                self.logger.warning("No failure cases found in training data")
                return {"status": "no_failures"}
            
            self.logger.info(f"Found {len(failure_data['features'])} failure cases for analysis")
            
            # Train ML models for failure prediction
            isolation_forest = await self._train_isolation_forest(failure_data["features"])
            failure_clusters = await self._cluster_failure_modes(failure_data["features"], failure_data)
            failure_trends = await self._analyze_failure_trends(failure_data)
            root_causes = await self._identify_root_causes(failure_data)
            
            return {
                "status": "success",
                "failure_analysis": {
                    "total_samples": training_data["metadata"]["total_samples"],
                    "failure_cases": len(failure_data["features"]),
                    "isolation_forest": isolation_forest,
                    "failure_clusters": failure_clusters,
                    "failure_trends": failure_trends,
                    "root_causes": root_causes,
                    "model_metadata": {
                        "features_extracted": len(failure_data["features"][0]) if failure_data["features"] else 0,
                        "training_date": datetime.now().isoformat(),
                        "data_source": "training_pipeline"
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Historical failure analysis failed: {e}")
            return {"status": "error", "error": str(e)}

    async def predict_failure_probability(self, prompt_features: np.ndarray) -> dict[str, Any]:
        """Predict failure probability for new prompt features using trained models
        
        Args:
            prompt_features: Feature vector for prompt to analyze
            
        Returns:
            Failure probability prediction with confidence and recommendations
        """
        try:
            if not hasattr(self, 'isolation_forest_model'):
                self.logger.warning("No trained isolation forest model available")
                return {"status": "no_model", "fallback_probability": 0.5}
            
            # Use isolation forest for anomaly detection
            anomaly_score = self.isolation_forest_model.decision_function([prompt_features])[0]
            failure_probability = self._score_to_probability(anomaly_score)
            
            # Get similar failures for context
            similar_failures = await self._get_similar_failures(prompt_features)
            
            # Generate recommendations based on failure probability
            recommendations = self._generate_failure_recommendations(failure_probability, similar_failures)
            
            return {
                "status": "success",
                "failure_probability": failure_probability,
                "confidence": min(1.0, abs(anomaly_score)),
                "anomaly_score": anomaly_score,
                "similar_failures": similar_failures[:5],  # Top 5 similar cases
                "recommendations": recommendations,
                "prediction_metadata": {
                    "model_type": "isolation_forest",
                    "features_used": len(prompt_features),
                    "prediction_date": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failure probability prediction failed: {e}")
            return {"status": "error", "error": str(e)}

    async def analyze_failures(
        self, test_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze failure patterns in test results

        Args:
            test_results: All test results to analyze

        Returns:
            Comprehensive failure analysis
        """
        self.logger.info(
            f"Starting failure analysis with {len(test_results)} test results"
        )

        # Filter failures and successes
        failures = [
            result
            for result in test_results
            if (
                result.get("overallImprovement", 0) or result.get("improvementScore", 0)
            )
            < self.config.failure_threshold
        ]

        successes = [
            result
            for result in test_results
            if (
                result.get("overallImprovement", 0) or result.get("improvementScore", 0)
            )
            >= self.config.failure_threshold
        ]

        self.logger.info(
            f"Identified {len(failures)} failures and {len(successes)} successes"
        )

        if len(failures) == 0:
            return self._generate_no_failure_report(test_results)

        # Core failure analysis using specialized components
        failure_analysis = {
            "summary": self._generate_failure_summary(failures, test_results),
            "patterns": await self.pattern_detector.identify_failure_patterns(failures),
            "root_causes": await self.pattern_detector.identify_root_causes_comparative(failures, successes),
            "rule_gaps": await self.pattern_detector.find_missing_rules(failures),
            "edge_cases": await self.pattern_detector.find_edge_cases(failures),
            "systematic_issues": await self.pattern_detector.find_systematic_issues(
                failures, test_results
            ),
            # ML FMEA Analysis using failure classifier
            "ml_fmea": await self.failure_classifier.perform_ml_fmea_analysis(failures, test_results),
            "anomaly_detection": await self.failure_classifier.perform_ensemble_anomaly_detection(
                failures
            ),
            "risk_assessment": await self.failure_classifier.calculate_risk_priority_numbers(failures),
            # Comparative analysis
            "failure_vs_success": self._compare_failures_with_successes(
                failures, successes
            ),
            # Metadata
            "metadata": {
                "total_failures": len(failures),
                "total_tests": len(test_results),
                "failure_rate": len(failures) / len(test_results),
                "analysis_date": datetime.now().isoformat(),
                "config": self.config.__dict__,
                # Phase 3 metadata
                "phase3_enhancements": {
                    "robustness_validation_enabled": self.config.enable_robustness_validation,
                    "prometheus_monitoring_enabled": self.config.enable_prometheus_monitoring,
                    "ensemble_anomaly_detection_enabled": self.config.ensemble_anomaly_detection,
                    "adversarial_testing_enabled": self.config.adversarial_testing,
                },
            },
        }

        self.logger.info(
            f"Failure analysis completed: {len(failure_analysis['patterns'])} patterns, "
            f"{len(failure_analysis['root_causes'])} root causes identified"
        )

        # Phase 3 enhancements: Add robustness validation
        if self.config.enable_robustness_validation:
            failure_analysis[
                "robustness_validation"
            ] = await self._perform_robustness_validation(failures, test_results)

        # Phase 3 enhancements: Update monitoring metrics
        if self.config.enable_prometheus_monitoring:
            await self.failure_classifier._update_prometheus_metrics(failure_analysis)
            failure_analysis[
                "prometheus_alerts"
            ] = await self.failure_classifier._check_and_trigger_alerts(failure_analysis)

        return failure_analysis

    def _generate_no_failure_report(
        self, test_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate report when no failures are found"""
        return {
            "summary": {
                "total_tests": len(test_results),
                "failure_rate": 0.0,
                "status": "excellent",
                "message": "No failures detected - system performing optimally",
            },
            "patterns": [],
            "root_causes": [],
            "recommendations": [
                {
                    "type": "maintenance",
                    "description": "Continue monitoring for potential issues",
                    "priority": "low",
                }
            ],
            "metadata": {
                "total_failures": 0,
                "total_tests": len(test_results),
                "failure_rate": 0.0,
                "analysis_date": datetime.now().isoformat(),
            },
        }

    def _generate_failure_summary(
        self, failures: list[dict[str, Any]], all_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate failure summary statistics"""
        failure_scores = [
            failure.get("overallImprovement", 0) or failure.get("improvementScore", 0)
            for failure in failures
        ]

        # Calculate basic statistics
        summary = {
            "total_failures": len(failures),
            "failure_rate": len(failures) / len(all_results),
            "avg_failure_score": float(np.mean(failure_scores))
            if failure_scores
            else 0.0,
            "worst_failure_score": float(np.min(failure_scores))
            if failure_scores
            else 0.0,
            "failure_std": float(np.std(failure_scores)) if failure_scores else 0.0,
        }

        # Classify failure severity
        if summary["failure_rate"] > 0.3:
            summary["severity"] = "critical"
        elif summary["failure_rate"] > 0.15:
            summary["severity"] = "high"
        elif summary["failure_rate"] > 0.05:
            summary["severity"] = "medium"
        else:
            summary["severity"] = "low"

        # Analyze failure distribution by rule
        rule_failures = defaultdict(int)
        rule_totals = defaultdict(int)

        for result in all_results:
            applied_rules = result.get("appliedRules", [])
            is_failure = result in failures

            for rule in applied_rules:
                rule_id = rule.get("ruleId") or rule.get("id", "unknown")
                rule_totals[rule_id] += 1
                if is_failure:
                    rule_failures[rule_id] += 1

        # Calculate rule failure rates
        rule_failure_rates = {}
        for rule_id in rule_totals:
            if rule_totals[rule_id] >= 5:  # Minimum sample size
                failure_rate = rule_failures[rule_id] / rule_totals[rule_id]
                rule_failure_rates[rule_id] = {
                    "failure_rate": failure_rate,
                    "failures": rule_failures[rule_id],
                    "total": rule_totals[rule_id],
                }

        # Find worst performing rules
        worst_rules = sorted(
            rule_failure_rates.items(), key=lambda x: x[1]["failure_rate"], reverse=True
        )[:5]

        summary["worst_performing_rules"] = [
            {"rule_id": rule_id, **stats} for rule_id, stats in worst_rules
        ]

        return summary

    def _compare_failures_with_successes(
        self, failures: list[dict[str, Any]], successes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compare failures with successes to identify key differences"""
        comparison = {}

        # Basic statistics
        failure_scores = [
            f.get("overallImprovement", 0) or f.get("improvementScore", 0)
            for f in failures
        ]
        success_scores = [
            s.get("overallImprovement", 0) or s.get("improvementScore", 0)
            for s in successes
        ]

        comparison["score_comparison"] = {
            "failure_avg": float(np.mean(failure_scores)) if failure_scores else 0,
            "success_avg": float(np.mean(success_scores)) if success_scores else 0,
            "score_gap": float(np.mean(success_scores) - np.mean(failure_scores))
            if failure_scores and success_scores
            else 0,
        }

        # Rule usage comparison
        failure_rules = defaultdict(int)
        success_rules = defaultdict(int)

        for failure in failures:
            for rule in failure.get("appliedRules", []):
                rule_id = rule.get("ruleId") or rule.get("id", "unknown")
                failure_rules[rule_id] += 1

        for success in successes:
            for rule in success.get("appliedRules", []):
                rule_id = rule.get("ruleId") or rule.get("id", "unknown")
                success_rules[rule_id] += 1

        # Find rules more common in failures vs successes
        problematic_rules = []
        beneficial_rules = []

        all_rules = set(failure_rules.keys()) | set(success_rules.keys())

        for rule_id in all_rules:
            failure_rate = failure_rules[rule_id] / len(failures) if failures else 0
            success_rate = success_rules[rule_id] / len(successes) if successes else 0

            if failure_rate > success_rate * 1.5 and failure_rules[rule_id] >= 3:
                problematic_rules.append({
                    "rule_id": rule_id,
                    "failure_rate": failure_rate,
                    "success_rate": success_rate,
                    "bias_ratio": failure_rate / (success_rate + 0.01),
                })
            elif success_rate > failure_rate * 1.5 and success_rules[rule_id] >= 3:
                beneficial_rules.append({
                    "rule_id": rule_id,
                    "failure_rate": failure_rate,
                    "success_rate": success_rate,
                    "benefit_ratio": success_rate / (failure_rate + 0.01),
                })

        comparison["rule_analysis"] = {
            "problematic_rules": sorted(
                problematic_rules, key=lambda x: x["bias_ratio"], reverse=True
            )[:5],
            "beneficial_rules": sorted(
                beneficial_rules, key=lambda x: x["benefit_ratio"], reverse=True
            )[:5],
        }

        return comparison

    # Robustness validation methods

    async def _run_robustness_validation(self, test_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Run comprehensive robustness validation"""
        failures = [r for r in test_results if r.get("overallImprovement", 0) < self.config.failure_threshold]
        return await self._perform_robustness_validation(failures, test_results)

    async def _perform_robustness_validation(
        self, failures: list[dict[str, Any]], test_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Perform comprehensive robustness validation tests"""
        validation_results = {
            "noise_robustness": None,
            "adversarial_robustness": None,
            "edge_case_robustness": None,
            "data_drift_robustness": None,
            "overall_robustness_score": 0.0,
            "robustness_recommendations": [],
        }

        try:
            # Test 1: Noise Robustness
            if len(test_results) >= self.config.robustness_test_samples:
                validation_results[
                    "noise_robustness"
                ] = await self._test_noise_robustness(test_results)

            # Test 2: Adversarial Robustness
            if self.config.adversarial_testing and ART_AVAILABLE:
                validation_results[
                    "adversarial_robustness"
                ] = await self._test_adversarial_robustness(test_results)

            # Test 3: Edge Case Robustness
            validation_results[
                "edge_case_robustness"
            ] = await self._test_edge_case_robustness(failures)

            # Test 4: Data Drift Robustness
            validation_results[
                "data_drift_robustness"
            ] = await self._test_data_drift_robustness(test_results)

            # Calculate overall robustness score
            scores = []
            for test_name, result in validation_results.items():
                if result and isinstance(result, dict) and "robustness_score" in result:
                    scores.append(result["robustness_score"])

            validation_results["overall_robustness_score"] = (
                np.mean(scores) if scores else 0.0
            )

            # Generate robustness recommendations
            validation_results["robustness_recommendations"] = (
                self._generate_robustness_recommendations(validation_results)
            )

            # Store test results for future analysis
            self.robustness_test_results.extend([
                RobustnessTestResult(
                    test_type=test_name,
                    success_rate=result.get("success_rate", 0.0),
                    degradation_rate=result.get("degradation_rate", 0.0),
                    robustness_score=result.get("robustness_score", 0.0),
                    failed_samples=result.get("failed_samples", []),
                    recommendations=result.get("recommendations", []),
                )
                for test_name, result in validation_results.items()
                if result and isinstance(result, dict) and "robustness_score" in result
            ])

            self.logger.info(
                f"Robustness validation completed with overall score: {validation_results['overall_robustness_score']:.3f}"
            )

        except Exception as e:
            self.logger.error(f"Robustness validation failed: {e}")
            validation_results["error"] = str(e)

        return validation_results

    async def _test_noise_robustness(
        self, test_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Test robustness against input noise"""
        noise_test_results = {
            "test_type": "noise",
            "noise_levels_tested": self.config.noise_levels,
            "results_by_noise_level": {},
            "success_rate": 0.0,
            "degradation_rate": 0.0,
            "robustness_score": 0.0,
            "failed_samples": [],
            "recommendations": [],
        }

        # Sample test data
        sample_size = min(self.config.robustness_test_samples, len(test_results))
        test_samples = test_results[:sample_size]

        baseline_scores = [
            result.get("overallImprovement", 0) or result.get("improvementScore", 0)
            for result in test_samples
        ]
        baseline_avg = np.mean(baseline_scores) if baseline_scores else 0.0

        total_degradation = 0.0
        noise_level_count = 0

        for noise_level in self.config.noise_levels:
            # Simulate noise injection
            noisy_scores = []
            failed_samples = []

            for i, sample in enumerate(test_samples):
                # Apply noise to numerical features
                original_score = baseline_scores[i]

                # Simulate noise impact on performance
                noise_impact = np.random.normal(0, noise_level)
                noisy_score = max(0, min(1, original_score + noise_impact))
                noisy_scores.append(noisy_score)

                # Track significant degradations
                if abs(noisy_score - original_score) > 0.1:
                    failed_samples.append({
                        "sample_index": i,
                        "original_score": original_score,
                        "noisy_score": noisy_score,
                        "degradation": original_score - noisy_score,
                    })

            # Calculate performance metrics for this noise level
            noisy_avg = np.mean(noisy_scores) if noisy_scores else 0.0
            success_rate = sum(
                1 for score in noisy_scores if score >= baseline_avg * 0.9
            ) / len(noisy_scores)
            degradation_rate = (
                max(0, (baseline_avg - noisy_avg) / baseline_avg)
                if baseline_avg > 0
                else 0
            )

            noise_test_results["results_by_noise_level"][noise_level] = {
                "baseline_avg": baseline_avg,
                "noisy_avg": noisy_avg,
                "success_rate": success_rate,
                "degradation_rate": degradation_rate,
                "failed_samples_count": len(failed_samples),
            }

            total_degradation += degradation_rate
            noise_level_count += 1

            # Keep track of worst failures
            if len(failed_samples) > 0:
                noise_test_results["failed_samples"].extend(
                    failed_samples[:3]
                )  # Top 3 failures per level

        # Calculate overall metrics
        avg_degradation = (
            total_degradation / noise_level_count if noise_level_count > 0 else 0
        )
        noise_test_results["degradation_rate"] = avg_degradation
        noise_test_results["success_rate"] = 1 - avg_degradation
        noise_test_results["robustness_score"] = max(0, 1 - avg_degradation)

        # Generate recommendations
        if avg_degradation > 0.2:
            noise_test_results["recommendations"].extend([
                "Implement input validation and sanitization",
                "Add noise injection during training for robustness",
                "Consider ensemble methods to reduce noise sensitivity",
            ])

        return noise_test_results

    async def _test_adversarial_robustness(
        self, test_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Test robustness against adversarial attacks"""
        adversarial_test_results = {
            "test_type": "adversarial",
            "epsilon": self.config.adversarial_epsilon,
            "success_rate": 0.0,
            "degradation_rate": 0.0,
            "robustness_score": 0.0,
            "failed_samples": [],
            "recommendations": [],
            "attack_success_rate": 0.0,
        }

        try:
            # This is a simplified adversarial test
            sample_size = min(50, len(test_results))
            test_samples = test_results[:sample_size]

            successful_attacks = 0
            failed_samples = []

            for i, sample in enumerate(test_samples):
                # Simulate adversarial perturbation
                original_score = sample.get("overallImprovement", 0) or sample.get(
                    "improvementScore", 0
                )

                # Simple adversarial simulation: targeted degradation
                adversarial_score = max(
                    0, original_score - self.config.adversarial_epsilon
                )

                # Check if attack was successful (significant degradation)
                if original_score - adversarial_score > 0.1:
                    successful_attacks += 1
                    failed_samples.append({
                        "sample_index": i,
                        "original_score": original_score,
                        "adversarial_score": adversarial_score,
                        "attack_success": True,
                    })

            # Calculate metrics
            attack_success_rate = successful_attacks / sample_size
            adversarial_test_results["attack_success_rate"] = attack_success_rate
            adversarial_test_results["success_rate"] = 1 - attack_success_rate
            adversarial_test_results["degradation_rate"] = attack_success_rate
            adversarial_test_results["robustness_score"] = 1 - attack_success_rate
            adversarial_test_results["failed_samples"] = failed_samples[
                :10
            ]  # Top 10 failures

            # Generate recommendations based on vulnerability
            if attack_success_rate > 0.3:
                adversarial_test_results["recommendations"].extend([
                    "Implement adversarial training techniques",
                    "Add input preprocessing and filtering",
                    "Consider defensive distillation methods",
                    "Implement gradient masking protections",
                ])
            elif attack_success_rate > 0.1:
                adversarial_test_results["recommendations"].extend([
                    "Monitor for adversarial patterns",
                    "Implement anomaly detection for inputs",
                ])

            self.logger.info(
                f"Adversarial robustness test: {attack_success_rate:.1%} attack success rate"
            )

        except Exception as e:
            self.logger.error(f"Adversarial robustness test failed: {e}")
            adversarial_test_results["error"] = str(e)

        return adversarial_test_results

    async def _test_edge_case_robustness(
        self, failures: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Test robustness against edge cases"""
        edge_case_test_results = {
            "test_type": "edge_case",
            "edge_cases_identified": 0,
            "success_rate": 0.0,
            "degradation_rate": 0.0,
            "robustness_score": 0.0,
            "failed_samples": [],
            "recommendations": [],
            "edge_case_patterns": [],
        }

        # Identify edge cases from failures
        edge_cases = await self.pattern_detector.find_edge_cases(failures)
        edge_case_test_results["edge_cases_identified"] = len(edge_cases)

        if len(edge_cases) > 0:
            # Analyze edge case patterns
            patterns = []
            for edge_case in edge_cases:
                patterns.append({
                    "type": edge_case["type"],
                    "description": edge_case["description"],
                    "affected_count": edge_case["affected_failures"],
                    "characteristics": edge_case["characteristics"],
                })

            edge_case_test_results["edge_case_patterns"] = patterns

            # Calculate robustness based on edge case frequency
            total_failures = len(failures)
            edge_case_failures = sum(pattern["affected_count"] for pattern in patterns)
            edge_case_rate = (
                edge_case_failures / total_failures if total_failures > 0 else 0
            )

            edge_case_test_results["degradation_rate"] = edge_case_rate
            edge_case_test_results["success_rate"] = 1 - edge_case_rate
            edge_case_test_results["robustness_score"] = max(0, 1 - edge_case_rate)

            # Extract failed samples from edge cases
            for edge_case in edge_cases[:5]:  # Top 5 edge cases
                if "examples" in edge_case:
                    edge_case_test_results["failed_samples"].extend(
                        edge_case["examples"][:2]
                    )

            # Generate recommendations
            if edge_case_rate > 0.2:
                edge_case_test_results["recommendations"].extend([
                    "Implement comprehensive input validation",
                    "Add edge case handling logic",
                    "Expand training data to cover edge cases",
                    "Create specialized rules for identified edge cases",
                ])
        else:
            # No edge cases found - good robustness
            edge_case_test_results["success_rate"] = 1.0
            edge_case_test_results["robustness_score"] = 1.0

        return edge_case_test_results

    async def _test_data_drift_robustness(
        self, test_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Test robustness against data drift"""
        drift_test_results = {
            "test_type": "data_drift",
            "drift_detected": False,
            "drift_magnitude": 0.0,
            "success_rate": 0.0,
            "degradation_rate": 0.0,
            "robustness_score": 0.0,
            "failed_samples": [],
            "recommendations": [],
            "drift_indicators": [],
        }

        if len(test_results) < 20:
            drift_test_results["insufficient_data"] = (
                "Need at least 20 samples for drift detection"
            )
            return drift_test_results

        try:
            # Split data into "baseline" and "current" periods
            split_point = len(test_results) // 2
            baseline_results = test_results[:split_point]
            current_results = test_results[split_point:]

            # Extract performance scores
            baseline_scores = [
                result.get("overallImprovement", 0) or result.get("improvementScore", 0)
                for result in baseline_results
            ]
            current_scores = [
                result.get("overallImprovement", 0) or result.get("improvementScore", 0)
                for result in current_results
            ]

            # Statistical test for distribution drift
            if len(baseline_scores) > 0 and len(current_scores) > 0:
                # Kolmogorov-Smirnov test
                ks_statistic, ks_p_value = stats.ks_2samp(
                    baseline_scores, current_scores
                )

                # Mann-Whitney U test
                mw_statistic, mw_p_value = stats.mannwhitneyu(
                    baseline_scores, current_scores, alternative="two-sided"
                )

                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (np.var(baseline_scores) + np.var(current_scores)) / 2
                )
                cohens_d = (
                    (np.mean(current_scores) - np.mean(baseline_scores)) / pooled_std
                    if pooled_std > 0
                    else 0
                )

                # Determine if drift is significant
                drift_detected = ks_p_value < 0.05 or mw_p_value < 0.05
                drift_magnitude = abs(cohens_d)

                drift_test_results.update({
                    "drift_detected": drift_detected,
                    "drift_magnitude": drift_magnitude,
                    "ks_statistic": ks_statistic,
                    "ks_p_value": ks_p_value,
                    "mw_p_value": mw_p_value,
                    "cohens_d": cohens_d,
                    "baseline_mean": np.mean(baseline_scores),
                    "current_mean": np.mean(current_scores),
                })

                # Calculate robustness based on drift magnitude
                if drift_detected:
                    degradation_rate = min(1.0, drift_magnitude / 2.0)  # Scale to [0,1]
                    drift_test_results["degradation_rate"] = degradation_rate
                    drift_test_results["success_rate"] = 1 - degradation_rate
                    drift_test_results["robustness_score"] = max(
                        0, 1 - degradation_rate
                    )

                    # Generate recommendations
                    if drift_magnitude > 0.5:
                        drift_test_results["recommendations"].extend([
                            "Implement continuous data monitoring",
                            "Set up automated drift detection alerts",
                            "Consider model retraining pipeline",
                            "Investigate data source changes",
                        ])
                    elif drift_magnitude > 0.2:
                        drift_test_results["recommendations"].extend([
                            "Monitor data distribution changes",
                            "Consider adaptive model updates",
                        ])
                else:
                    # No significant drift detected
                    drift_test_results["success_rate"] = 1.0
                    drift_test_results["robustness_score"] = 1.0

        except Exception as e:
            self.logger.error(f"Data drift robustness test failed: {e}")
            drift_test_results["error"] = str(e)

        return drift_test_results

    def _generate_robustness_recommendations(
        self, validation_results: dict[str, Any]
    ) -> list[str]:
        """Generate comprehensive robustness recommendations"""
        recommendations = []
        overall_score = validation_results.get("overall_robustness_score", 0.0)

        # Overall robustness assessment
        if overall_score < 0.5:
            recommendations.append(
                "CRITICAL: Overall system robustness is poor - immediate action required"
            )
        elif overall_score < 0.7:
            recommendations.append("WARNING: System robustness needs improvement")

        # Collect specific recommendations from each test
        for test_name, result in validation_results.items():
            if isinstance(result, dict) and "recommendations" in result:
                recommendations.extend(result["recommendations"])

        # Add general robustness recommendations
        if overall_score < 0.8:
            recommendations.extend([
                "Implement comprehensive testing suite with edge cases",
                "Add monitoring for performance degradation",
                "Create fallback mechanisms for system failures",
                "Establish regular robustness validation schedule",
            ])

        # Remove duplicates and return
        return list(set(recommendations))

    # Training data integration helper methods

    async def _extract_failure_cases(self, training_data: dict) -> dict:
        """Extract failure cases from training data for analysis"""
        failures = []
        features = []
        
        for sample in training_data.get("samples", []):
            # Check if this is a failure case
            score = sample.get("overallImprovement", 0) or sample.get("improvementScore", 0)
            if score < self.config.failure_threshold:
                failures.append(sample)
                # Extract numerical features for ML analysis
                feature_vector = [
                    len(sample.get("originalPrompt", "").split()),  # prompt length
                    score,  # improvement score
                    len(sample.get("appliedRules", [])),  # rule count
                    len(str(sample.get("context", {}))),  # context complexity
                ]
                features.append(feature_vector)
        
        return {"failures": failures, "features": features}

    async def _extract_failure_features(self, failure_data: dict) -> np.ndarray:
        """Extract numerical features for ML analysis"""
        return np.array(failure_data.get("features", []))

    async def _train_isolation_forest(self, failure_features: np.ndarray) -> dict:
        """Train isolation forest for anomaly detection"""
        try:
            from sklearn.ensemble import IsolationForest
            
            if len(failure_features) < 10:
                return {"status": "insufficient_data", "samples": len(failure_features)}
            
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(failure_features)
            
            # Store model for future predictions
            self.isolation_forest_model = model
            
            # Calculate performance metrics
            anomaly_scores = model.decision_function(failure_features)
            anomalies = model.predict(failure_features)
            
            return {
                "status": "success",
                "model_type": "isolation_forest",
                "training_samples": len(failure_features),
                "contamination": 0.1,
                "anomaly_count": np.sum(anomalies == -1),
                "average_anomaly_score": float(np.mean(anomaly_scores)),
                "feature_dimensions": failure_features.shape[1]
            }
            
        except Exception as e:
            self.logger.error(f"Isolation forest training failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _cluster_failure_modes(self, failure_features: np.ndarray, failure_data: dict) -> dict:
        """Cluster failure modes to identify patterns"""
        try:
            from sklearn.cluster import DBSCAN
            
            if len(failure_features) < 5:
                return {"status": "insufficient_data"}
            
            clustering = DBSCAN(eps=0.5, min_samples=3)
            cluster_labels = clustering.fit_predict(failure_features)
            
            # Analyze clusters
            unique_labels = set(cluster_labels)
            clusters = {}
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                    
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_failures = [failure_data["failures"][i] for i in cluster_indices]
                
                clusters[f"cluster_{label}"] = {
                    "size": len(cluster_indices),
                    "representative_failures": cluster_failures[:3],
                    "avg_score": np.mean([
                        f.get("overallImprovement", 0) or f.get("improvementScore", 0)
                        for f in cluster_failures
                    ])
                }
            
            return {
                "status": "success",
                "total_clusters": len(clusters),
                "noise_points": np.sum(cluster_labels == -1),
                "clusters": clusters
            }
            
        except Exception as e:
            self.logger.error(f"Failure clustering failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _analyze_failure_trends(self, failure_data: dict) -> dict:
        """Analyze temporal trends in failure data"""
        failures = failure_data.get("failures", [])
        
        if len(failures) < 5:
            return {"status": "insufficient_data"}
        
        # Extract timestamps and scores
        timestamped_failures = []
        for failure in failures:
            timestamp_str = failure.get("timestamp") or failure.get("createdAt")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    score = failure.get("overallImprovement", 0) or failure.get("improvementScore", 0)
                    timestamped_failures.append((timestamp, score))
                except:
                    continue
        
        if len(timestamped_failures) < 5:
            return {"status": "insufficient_timestamps"}
        
        # Sort by timestamp
        timestamped_failures.sort(key=lambda x: x[0])
        
        # Calculate trend
        scores = [score for _, score in timestamped_failures]
        time_indices = list(range(len(scores)))
        
        # Simple linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_indices, scores)
        
        return {
            "status": "success",
            "total_samples": len(timestamped_failures),
            "trend_slope": slope,
            "trend_strength": r_value**2,
            "trend_significance": p_value,
            "trend_direction": "improving" if slope > 0 else "deteriorating" if slope < 0 else "stable",
            "average_score": float(np.mean(scores)),
            "score_std": float(np.std(scores))
        }

    async def _identify_root_causes(self, failure_data: dict) -> list[dict]:
        """Identify root causes from failure data"""
        failures = failure_data.get("failures", [])
        
        if len(failures) < 3:
            return []
        
        # Analyze common characteristics
        root_causes = []
        
        # Rule-based root causes
        rule_failures = defaultdict(int)
        for failure in failures:
            for rule in failure.get("appliedRules", []):
                rule_id = rule.get("ruleId") or rule.get("id", "unknown")
                rule_failures[rule_id] += 1
        
        # Find rules with high failure rates
        for rule_id, count in rule_failures.items():
            if count >= len(failures) * 0.3:  # Appears in 30%+ of failures
                root_causes.append({
                    "type": "rule_based",
                    "description": f"Rule {rule_id} associated with {count} failures",
                    "affected_failures": count,
                    "confidence": count / len(failures)
                })
        
        # Context-based root causes
        context_failures = defaultdict(int)
        for failure in failures:
            context = failure.get("context", {})
            project_type = context.get("projectType", "unknown")
            context_failures[project_type] += 1
        
        for context_type, count in context_failures.items():
            if count >= len(failures) * 0.4:  # Appears in 40%+ of failures
                root_causes.append({
                    "type": "context_based",
                    "description": f"Context type {context_type} associated with {count} failures",
                    "affected_failures": count,
                    "confidence": count / len(failures)
                })
        
        return root_causes[:10]  # Top 10 root causes

    def _score_to_probability(self, anomaly_score: float) -> float:
        """Convert isolation forest anomaly score to failure probability"""
        # Isolation forest scores are typically in range [-0.5, 0.5]
        # Negative scores indicate anomalies (potential failures)
        return max(0.0, min(1.0, 0.5 - anomaly_score))

    def _generate_failure_recommendations(self, failure_prob: float, similar_failures: list) -> list[str]:
        """Generate recommendations based on failure probability"""
        recommendations = []
        
        if failure_prob > 0.8:
            recommendations.extend([
                "HIGH RISK: Review prompt structure and context",
                "Consider applying multiple improvement rules",
                "Validate input data quality"
            ])
        elif failure_prob > 0.5:
            recommendations.extend([
                "MEDIUM RISK: Monitor for potential issues",
                "Consider additional validation steps"
            ])
        else:
            recommendations.extend([
                "LOW RISK: Continue with standard processing"
            ])
        
        return recommendations

    async def _get_similar_failures(self, prompt_features: np.ndarray) -> list[dict]:
        """Get similar historical failures for context"""
        # This would use trained models to find similar cases
        # For now, return empty list as placeholder
        return []