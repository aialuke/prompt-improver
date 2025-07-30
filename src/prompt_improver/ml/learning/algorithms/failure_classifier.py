"""Failure Classification System

Machine Learning-based failure mode analysis using FMEA methodology.
Performs ensemble anomaly detection and classifies failures according to ML system
failure modes from Microsoft Learn guidelines.
"""

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, TYPE_CHECKING

# Import OTelAlert for type checking only
if TYPE_CHECKING:
    from src.prompt_improver.monitoring.opentelemetry.ml_framework import OTelAlert
else:
    OTelAlert = Any

# Check if OTel framework is available at runtime
try:
    from src.prompt_improver.monitoring.opentelemetry.ml_framework import OTelAlert as _OTelAlert
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    _OTelAlert = None

import numpy as np

# Phase 3 enhancement imports for robustness validation and monitoring
try:
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM

    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False
    warnings.warn(
        "Anomaly detection libraries not available. Install with: pip install scikit-learn"
    )

# REMOVED prometheus_client for OpenTelemetry consolidation
PROMETHEUS_AVAILABLE = False

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


@dataclass
class MLFailureMode:
    """Structured representation of ML system failure modes"""

    failure_type: str  # 'data', 'model', 'infrastructure', 'deployment'
    description: str
    severity: int  # 1-10 scale
    occurrence: int  # 1-10 scale
    detection: int  # 1-10 scale
    rpn: int  # Risk Priority Number
    root_causes: list[str]
    detection_methods: list[str]
    mitigation_strategies: list[str]
    affected_components: list[str] = field(default_factory=list)
    impact_areas: list[str] = field(default_factory=list)


# PrometheusAlert removed - using OTelAlert from ML framework instead


class FailureClassifier:
    """Machine Learning-based failure classification and FMEA analysis"""

    def __init__(self, config):
        """Initialize the failure classifier
        
        Args:
            config: FailureConfig instance containing analysis parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize ML FMEA database
        self.ml_failure_modes = self._initialize_ml_fmea_database()
        
        # Initialize anomaly detection components
        self.ensemble_detectors = {}
        self.anomaly_detectors = {}
        
        # Initialize OpenTelemetry monitoring components
        self.prometheus_metrics: dict[str, Any] = {}  # Legacy name for compatibility
        self.alert_definitions: list[OTelAlert] = []
        self.active_alerts: list[OTelAlert] = []
        self.alert_history: list[OTelAlert] = []
        
        # Initialize components
        self._initialize_robustness_components()
        if self.config.enable_prometheus_monitoring:
            self._initialize_otel_monitoring()

    def _initialize_ml_fmea_database(self) -> list:
        """Initialize ML FMEA failure modes database following Microsoft Learn guidelines"""
        
        return [
            # Data-related failure modes
            MLFailureMode(
                failure_type="data",
                description="Data drift - distribution changes from training to production",
                severity=8,
                occurrence=6,
                detection=4,
                rpn=192,
                root_causes=[
                    "Environment changes",
                    "User behavior shifts",
                    "System updates",
                ],
                detection_methods=["KS test", "PSI monitoring", "JS divergence"],
                mitigation_strategies=[
                    "Continuous monitoring",
                    "Model retraining",
                    "Adaptive thresholds",
                ],
                affected_components=["input_validation", "model_inference"],
                impact_areas=["prediction_accuracy", "user_experience"],
            ),
            MLFailureMode(
                failure_type="data",
                description="Data quality - missing validation, inconsistent collection",
                severity=7,
                occurrence=5,
                detection=3,
                rpn=105,
                root_causes=[
                    "Incomplete validation",
                    "Pipeline errors",
                    "Source system changes",
                ],
                detection_methods=[
                    "Data profiling",
                    "Schema validation",
                    "Completeness checks",
                ],
                mitigation_strategies=[
                    "Robust validation",
                    "Pipeline monitoring",
                    "Data contracts",
                ],
                affected_components=["data_preprocessing", "feature_engineering"],
                impact_areas=["model_performance", "system_reliability"],
            ),
            # Model-related failure modes
            MLFailureMode(
                failure_type="model",
                description="Model overfitting - poor generalization to new data",
                severity=7,
                occurrence=4,
                detection=5,
                rpn=140,
                root_causes=[
                    "Insufficient regularization",
                    "Limited training data",
                    "Poor validation",
                ],
                detection_methods=[
                    "Cross-validation",
                    "Holdout testing",
                    "Performance monitoring",
                ],
                mitigation_strategies=[
                    "Regularization techniques",
                    "More diverse data",
                    "Ensemble methods",
                ],
                affected_components=["model_training", "model_evaluation"],
                impact_areas=["prediction_accuracy", "model_robustness"],
            ),
            MLFailureMode(
                failure_type="model",
                description="Adversarial vulnerability - security concerns with robustness",
                severity=9,
                occurrence=3,
                detection=6,
                rpn=162,
                root_causes=[
                    "Lack of adversarial training",
                    "No robustness testing",
                    "Model architecture",
                ],
                detection_methods=[
                    "Adversarial testing",
                    "Robustness metrics",
                    "Security scanning",
                ],
                mitigation_strategies=[
                    "Adversarial training",
                    "Input sanitization",
                    "Model hardening",
                ],
                affected_components=["model_inference", "security_layer"],
                impact_areas=["system_security", "model_reliability"],
            ),
            # Infrastructure failure modes
            MLFailureMode(
                failure_type="infrastructure",
                description="Resource constraints - memory, CPU, or storage limitations",
                severity=6,
                occurrence=7,
                detection=4,
                rpn=168,
                root_causes=[
                    "Insufficient provisioning",
                    "Memory leaks",
                    "Inefficient algorithms",
                ],
                detection_methods=[
                    "Resource monitoring",
                    "Performance profiling",
                    "Load testing",
                ],
                mitigation_strategies=[
                    "Auto-scaling",
                    "Resource optimization",
                    "Efficient algorithms",
                ],
                affected_components=["model_serving", "data_processing"],
                impact_areas=["system_performance", "scalability"],
            ),
            # Deployment failure modes
            MLFailureMode(
                failure_type="deployment",
                description="Model version conflicts - incompatible model/code versions",
                severity=8,
                occurrence=4,
                detection=3,
                rpn=96,
                root_causes=[
                    "Poor version control",
                    "Inadequate testing",
                    "Manual deployment",
                ],
                detection_methods=[
                    "Version tracking",
                    "Integration tests",
                    "Deployment validation",
                ],
                mitigation_strategies=[
                    "Automated deployment",
                    "Version compatibility checks",
                    "Rollback procedures",
                ],
                affected_components=["deployment_pipeline", "model_registry"],
                impact_areas=["system_stability", "deployment_reliability"],
            ),
        ]

    async def perform_ml_fmea_analysis(
        self, failures: list[dict[str, Any]], test_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Perform ML FMEA analysis on failures"""
        try:
            fmea_analysis = {
                "identified_failure_modes": [],
                "risk_matrix": {},
                "critical_paths": [],
                "mitigation_plan": [],
            }

            # Analyze failures against known ML failure modes
            for failure_mode in self.ml_failure_modes:
                affected_failures = self._match_failures_to_mode(failures, failure_mode)

                if len(affected_failures) > 0:
                    # Update occurrence based on actual failure data
                    actual_occurrence = min(10, max(1, len(affected_failures) * 2))
                    updated_rpn = (
                        failure_mode.severity
                        * actual_occurrence
                        * failure_mode.detection
                    )

                    fmea_analysis["identified_failure_modes"].append({
                        "failure_mode": failure_mode.description,
                        "type": failure_mode.failure_type,
                        "severity": failure_mode.severity,
                        "occurrence": actual_occurrence,
                        "detection": failure_mode.detection,
                        "rpn": updated_rpn,
                        "affected_failures_count": len(affected_failures),
                        "root_causes": failure_mode.root_causes,
                        "mitigation_strategies": failure_mode.mitigation_strategies,
                        "priority": "critical"
                        if updated_rpn > 150
                        else "high"
                        if updated_rpn > 100
                        else "medium",
                    })

            # Build risk matrix
            fmea_analysis["risk_matrix"] = self._build_risk_matrix(
                fmea_analysis["identified_failure_modes"]
            )

            # Identify critical failure paths
            fmea_analysis["critical_paths"] = self._identify_critical_failure_paths(
                fmea_analysis["identified_failure_modes"]
            )

            # Generate mitigation plan
            fmea_analysis["mitigation_plan"] = self._generate_mitigation_plan(
                fmea_analysis["identified_failure_modes"]
            )

            return fmea_analysis

        except Exception as e:
            self.logger.error(f"ML FMEA analysis failed: {e}")
            return {"error": str(e)}

    async def perform_ensemble_anomaly_detection(
        self, failures: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Perform ensemble anomaly detection using multiple methods"""
        try:
            if len(failures) < 10:
                return {
                    "insufficient_data": "Need at least 10 failures for anomaly detection"
                }

            # Extract features for anomaly detection
            features = self._extract_anomaly_features(failures)

            if len(features) == 0:
                return {
                    "no_features": "Could not extract features for anomaly detection"
                }

            feature_matrix = np.array(features)

            # Apply ensemble anomaly detection
            anomaly_results = {}

            # Use the class attribute anomaly_detectors directly
            for detector_name, detector in self.anomaly_detectors.items():
                try:
                    # Fit and predict anomalies
                    anomalies = detector.fit_predict(feature_matrix)
                    anomaly_indices = np.where(anomalies == -1)[0]

                    anomaly_results[detector_name] = {
                        "anomaly_count": len(anomaly_indices),
                        "anomaly_percentage": len(anomaly_indices)
                        / len(failures)
                        * 100,
                        "anomaly_indices": anomaly_indices.tolist(),
                        "anomalous_failures": [
                            failures[i] for i in anomaly_indices[:5]
                        ],  # Top 5 examples
                    }

                except Exception as e:
                    self.logger.warning(
                        f"Anomaly detection with {detector_name} failed: {e}"
                    )
                    anomaly_results[detector_name] = {"error": str(e)}

            # Consensus-based anomaly identification
            consensus_anomalies = self._identify_consensus_anomalies(
                anomaly_results, failures
            )

            return {
                "individual_detectors": anomaly_results,
                "consensus_anomalies": consensus_anomalies,
                "anomaly_summary": {
                    "total_failures": len(failures),
                    "consensus_anomalies_count": len(consensus_anomalies),
                    "consensus_anomaly_rate": len(consensus_anomalies)
                    / len(failures)
                    * 100,
                },
            }

        except Exception as e:
            self.logger.error(f"Ensemble anomaly detection failed: {e}")
            return {"error": str(e)}

    async def calculate_risk_priority_numbers(
        self, failures: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate Risk Priority Numbers for identified failure modes"""
        risk_analysis = {
            "high_risk_modes": [],
            "medium_risk_modes": [],
            "low_risk_modes": [],
            "overall_risk_score": 0.0,
        }

        total_rpn = 0
        mode_count = 0

        for failure_mode in self.ml_failure_modes:
            # Calculate actual RPN based on failure data
            affected_count = self._count_affected_failures(failures, failure_mode)

            if affected_count > 0:
                # Adjust occurrence based on actual data
                adjusted_occurrence = min(10, max(1, affected_count))
                rpn = (
                    failure_mode.severity * adjusted_occurrence * failure_mode.detection
                )

                mode_data = {
                    "failure_mode": failure_mode.description,
                    "type": failure_mode.failure_type,
                    "rpn": rpn,
                    "severity": failure_mode.severity,
                    "occurrence": adjusted_occurrence,
                    "detection": failure_mode.detection,
                    "affected_failures": affected_count,
                }

                if rpn >= 150:
                    risk_analysis["high_risk_modes"].append(mode_data)
                elif rpn >= 100:
                    risk_analysis["medium_risk_modes"].append(mode_data)
                else:
                    risk_analysis["low_risk_modes"].append(mode_data)

                total_rpn += rpn
                mode_count += 1

        risk_analysis["overall_risk_score"] = (
            total_rpn / mode_count if mode_count > 0 else 0
        )

        return risk_analysis

    def _match_failures_to_mode(
        self, failures: list[dict[str, Any]], failure_mode
    ) -> list[dict[str, Any]]:
        """Match failures to specific ML failure mode"""
        matched_failures = []

        for failure in failures:
            # Simple keyword matching - could be enhanced with ML classification
            failure_text = f"{failure.get('error', '')} {failure.get('originalPrompt', '')}".lower()

            # Check for keywords related to each failure mode type
            if failure_mode.failure_type == "data":
                data_keywords = [
                    "data",
                    "input",
                    "validation",
                    "schema",
                    "format",
                    "missing",
                    "null",
                ]
                if any(keyword in failure_text for keyword in data_keywords):
                    matched_failures.append(failure)

            elif failure_mode.failure_type == "model":
                model_keywords = [
                    "model",
                    "prediction",
                    "accuracy",
                    "performance",
                    "overfitting",
                ]
                if any(keyword in failure_text for keyword in model_keywords):
                    matched_failures.append(failure)

            elif failure_mode.failure_type == "infrastructure":
                infra_keywords = [
                    "memory",
                    "cpu",
                    "timeout",
                    "resource",
                    "capacity",
                    "performance",
                ]
                if any(keyword in failure_text for keyword in infra_keywords):
                    matched_failures.append(failure)

            elif failure_mode.failure_type == "deployment":
                deploy_keywords = [
                    "version",
                    "deployment",
                    "config",
                    "environment",
                    "compatibility",
                ]
                if any(keyword in failure_text for keyword in deploy_keywords):
                    matched_failures.append(failure)

        return matched_failures

    def _extract_anomaly_features(
        self, failures: list[dict[str, Any]]
    ) -> list[list[float]]:
        """Extract numerical features for anomaly detection"""
        features = []

        for failure in failures:
            feature_vector = []

            # Prompt length
            prompt = failure.get("originalPrompt", "")
            feature_vector.append(len(prompt.split()))

            # Performance score
            score = failure.get("overallImprovement", 0) or failure.get(
                "improvementScore", 0
            )
            feature_vector.append(score)

            # Number of applied rules
            applied_rules = failure.get("appliedRules", [])
            feature_vector.append(len(applied_rules))

            # Execution time if available
            exec_time = failure.get("executionTime", 0) or failure.get(
                "processingTime", 0
            )
            feature_vector.append(exec_time)

            # Context complexity (simplified)
            context = failure.get("context", {})
            feature_vector.append(len(str(context)))

            if len(feature_vector) == 5:  # Ensure consistent feature count
                features.append(feature_vector)

        return features

    def _identify_consensus_anomalies(
        self, anomaly_results: dict[str, Any], failures: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify anomalies that are detected by multiple methods"""
        consensus_anomalies = []

        # Count how many detectors identify each failure as anomalous
        anomaly_counts = defaultdict(int)

        for detector_name, results in anomaly_results.items():
            if "anomaly_indices" in results:
                for idx in results["anomaly_indices"]:
                    anomaly_counts[idx] += 1

        # Require at least 2 detectors to agree for consensus
        consensus_threshold = 2

        for idx, count in anomaly_counts.items():
            if count >= consensus_threshold and idx < len(failures):
                # Get total number of detectors from anomaly_results
                total_detectors = len([
                    r for r in anomaly_results.values() if "error" not in r
                ])
                consensus_anomalies.append({
                    "failure_index": idx,
                    "detector_agreement": count,
                    "failure_data": failures[idx],
                    "anomaly_score": count
                    / max(total_detectors, 1),  # Normalized score
                })

        # Sort by anomaly score
        consensus_anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)

        return consensus_anomalies

    def _count_affected_failures(
        self, failures: list[dict[str, Any]], failure_mode
    ) -> int:
        """Count how many failures are affected by this failure mode"""
        return len(self._match_failures_to_mode(failures, failure_mode))

    def _build_risk_matrix(self, failure_modes: list[dict[str, Any]]) -> dict[str, Any]:
        """Build risk matrix for visualization and prioritization"""
        risk_matrix = {
            "high_severity_high_occurrence": [],
            "high_severity_low_occurrence": [],
            "low_severity_high_occurrence": [],
            "low_severity_low_occurrence": [],
        }

        for mode in failure_modes:
            severity = mode["severity"]
            occurrence = mode["occurrence"]

            if severity >= 7 and occurrence >= 6:
                risk_matrix["high_severity_high_occurrence"].append(mode)
            elif severity >= 7 and occurrence < 6:
                risk_matrix["high_severity_low_occurrence"].append(mode)
            elif severity < 7 and occurrence >= 6:
                risk_matrix["low_severity_high_occurrence"].append(mode)
            else:
                risk_matrix["low_severity_low_occurrence"].append(mode)

        return risk_matrix

    def _identify_critical_failure_paths(
        self, failure_modes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify critical failure paths requiring immediate attention"""
        critical_paths = []

        # Sort by RPN to identify highest risk paths
        sorted_modes = sorted(failure_modes, key=lambda x: x["rpn"], reverse=True)

        for mode in sorted_modes[:5]:  # Top 5 critical paths
            if mode["rpn"] >= 100:  # Only include high-risk modes
                critical_paths.append({
                    "failure_mode": mode["failure_mode"],
                    "rpn": mode["rpn"],
                    "priority": "immediate" if mode["rpn"] >= 150 else "urgent",
                    "affected_failures": mode["affected_failures_count"],
                    "recommended_actions": mode["mitigation_strategies"][
                        :3
                    ],  # Top 3 strategies
                })

        return critical_paths

    def _generate_mitigation_plan(
        self, failure_modes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate comprehensive mitigation plan"""
        mitigation_plan = []

        # Sort by RPN for prioritization
        sorted_modes = sorted(failure_modes, key=lambda x: x["rpn"], reverse=True)

        for i, mode in enumerate(sorted_modes[:10]):  # Top 10 modes
            plan_item = {
                "priority_rank": i + 1,
                "failure_mode": mode["failure_mode"],
                "rpn": mode["rpn"],
                "immediate_actions": mode["mitigation_strategies"][:2],
                "long_term_actions": mode["mitigation_strategies"][2:],
                "success_metrics": [
                    f"Reduce RPN below {mode['rpn'] * 0.5}",
                    f"Decrease occurrence to {max(1, mode['occurrence'] - 2)}",
                    "Improve detection capability",
                ],
                "timeline": "immediate"
                if mode["rpn"] >= 150
                else "1-2 weeks"
                if mode["rpn"] >= 100
                else "1 month",
            }
            mitigation_plan.append(plan_item)

        return mitigation_plan

    def _initialize_robustness_components(self) -> None:
        """Initialize Phase 3 robustness validation components"""
        try:
            # Initialize ensemble anomaly detectors
            if ANOMALY_DETECTION_AVAILABLE:
                self.ensemble_detectors = {
                    "isolation_forest": IsolationForest(
                        contamination=0.1, random_state=42, n_estimators=100
                    ),
                    "elliptic_envelope": EllipticEnvelope(
                        contamination=0.1, random_state=42
                    ),
                    "one_class_svm": OneClassSVM(nu=0.1, kernel="rbf", gamma="scale"),
                }
                # Assign anomaly_detectors for consistent access
                self.anomaly_detectors = self.ensemble_detectors
                self.logger.info("Ensemble anomaly detectors initialized")

            # Initialize adversarial testing components
            if ART_AVAILABLE:
                self.logger.info("Adversarial Robustness Toolbox components available")

            self.logger.info(
                "Robustness validation components initialized successfully"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize robustness components: {e}")
            self.config.enable_robustness_validation = False

    def _initialize_otel_monitoring(self) -> None:
        """Initialize OpenTelemetry monitoring components"""
        try:
            # Import OpenTelemetry metrics
            from src.prompt_improver.monitoring.opentelemetry.metrics import (
                get_ml_metrics, get_ml_alerting_metrics
            )

            # Initialize OpenTelemetry ML metrics
            self.otel_ml_metrics = get_ml_metrics()
            self.otel_alerting_metrics = get_ml_alerting_metrics(self.config.alert_thresholds)

            # Store reference for backward compatibility
            self.prometheus_metrics = {
                "failure_rate": self.otel_ml_metrics,
                "failure_count": self.otel_ml_metrics,
                "response_time": self.otel_ml_metrics,
                "anomaly_score": self.otel_ml_metrics,
                "rpn_score": self.otel_ml_metrics,
            }

            # Initialize alert definitions using OpenTelemetry
            self._initialize_alert_definitions()

            self.logger.info("OpenTelemetry monitoring initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenTelemetry monitoring: {e}")
            self.config.enable_prometheus_monitoring = False

    def _initialize_alert_definitions(self) -> None:
        """Initialize alert definitions for monitoring - now handled by OTel alerting system"""
        # Alert definitions are now managed by MLAlertingMetrics in the OTel framework
        # The alerting system automatically creates default alerts based on thresholds
        self.logger.info("Alert definitions managed by OpenTelemetry alerting system")

    async def _update_otel_metrics(
        self, failure_analysis: dict[str, Any]
    ) -> None:
        """Update OpenTelemetry metrics with analysis results"""
        if not hasattr(self, 'otel_ml_metrics'):
            return

        try:
            metadata = failure_analysis.get("metadata", {})
            summary = failure_analysis.get("summary", {})

            # Extract metrics data
            failure_rate = metadata.get("failure_rate", 0.0)
            severity = summary.get("severity", "unknown")
            total_failures = metadata.get("total_failures", 0)

            # Extract anomaly score
            anomaly_rate = None
            anomaly_detection = failure_analysis.get("anomaly_detection", {})
            if "anomaly_summary" in anomaly_detection:
                anomaly_rate = anomaly_detection["anomaly_summary"].get(
                    "consensus_anomaly_rate", 0
                ) / 100.0

            # Extract RPN score
            rpn_score = None
            risk_assessment = failure_analysis.get("risk_assessment", {})
            if "overall_risk_score" in risk_assessment:
                rpn_score = risk_assessment["overall_risk_score"]

            # Extract response time
            response_time = metadata.get("avg_response_time", 0.1)

            # Update OpenTelemetry metrics using the enhanced record_failure_analysis method
            self.otel_ml_metrics.record_failure_analysis(
                failure_rate=failure_rate,
                failure_type="classification",
                severity=severity,
                total_failures=total_failures,
                anomaly_rate=anomaly_rate,
                rpn_score=rpn_score,
                response_time=response_time
            )

            self.logger.debug("OpenTelemetry metrics updated successfully")

        except Exception as e:
            self.logger.error(f"Failed to update OpenTelemetry metrics: {e}")

    async def _check_and_trigger_alerts(
        self, failure_analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Check thresholds and trigger alerts using OpenTelemetry alerting"""
        if not hasattr(self, 'otel_alerting_metrics'):
            return []

        try:
            # Use OpenTelemetry alerting system
            triggered_alerts = self.otel_alerting_metrics.check_alerts(failure_analysis)

            # Convert to legacy format for compatibility
            legacy_alerts = []
            for alert in triggered_alerts:
                legacy_alert = {
                    "alert_name": alert.alert_name,
                    "metric_name": alert.metric_name,
                    "current_value": self.otel_alerting_metrics._extract_metric_value(
                        failure_analysis, alert.metric_name
                    ),
                    "threshold": alert.threshold,
                    "severity": alert.severity,
                    "description": alert.description,
                    "triggered_at": alert.triggered_at.isoformat() if alert.triggered_at else None,
                    "recommended_actions": self._get_alert_recommendations(alert.alert_name),
                }
                legacy_alerts.append(legacy_alert)

            return legacy_alerts

        except Exception as e:
            self.logger.error(f"Failed to check alerts: {e}")
            return []

    def _extract_metric_value(
        self, failure_analysis: dict[str, Any], metric_name: str
    ) -> float | None:
        """Extract metric value from failure analysis results"""
        metadata = failure_analysis.get("metadata", {})

        if metric_name == "ml_failure_rate":
            return metadata.get("failure_rate", 0.0)
        if metric_name == "ml_response_time_seconds":
            # This would come from actual performance measurements
            return metadata.get("avg_response_time", 0.1)  # Placeholder
        if metric_name == "ml_anomaly_score":
            anomaly_detection = failure_analysis.get("anomaly_detection", {})
            if "anomaly_summary" in anomaly_detection:
                return (
                    anomaly_detection["anomaly_summary"].get(
                        "consensus_anomaly_rate", 0
                    )
                    / 100.0
                )
            return 0.0

        return None

    def _check_threshold(self, value: float, alert_def: OTelAlert) -> bool:
        """Check if metric value exceeds alert threshold"""
        if alert_def.comparison == "gt":
            return value > alert_def.threshold
        if alert_def.comparison == "lt":
            return value < alert_def.threshold
        if alert_def.comparison == "eq":
            return abs(value - alert_def.threshold) < 0.001
        return False

    def _is_alert_ready_to_trigger(
        self, alert_def: OTelAlert, current_time: datetime
    ) -> bool:
        """Check if alert is not in cooldown period"""
        if alert_def.triggered_at is None:
            return True

        time_since_last = (current_time - alert_def.triggered_at).total_seconds()
        return time_since_last >= self.config.alert_cooldown_seconds

    def _get_alert_recommendations(self, alert_name: str) -> list[str]:
        """Get recommended actions for specific alert"""
        recommendations = {
            "HighFailureRate": [
                "Investigate recent system changes",
                "Check for data quality issues",
                "Review model performance metrics",
                "Consider temporary rollback if needed",
            ],
            "SlowResponseTime": [
                "Check system resource utilization",
                "Review database performance",
                "Monitor network latency",
                "Consider scaling resources",
            ],
            "HighAnomalyScore": [
                "Investigate anomalous patterns",
                "Check for data drift",
                "Review recent input changes",
                "Validate model predictions",
            ],
        }

        return recommendations.get(
            alert_name, ["Investigate the issue", "Check system logs"]
        )

    def _update_active_alerts(
        self, failure_analysis: dict[str, Any], current_time: datetime
    ) -> None:
        """Update active alerts - now handled by OpenTelemetry alerting system"""
        # Alert management is now handled by MLAlertingMetrics
        pass