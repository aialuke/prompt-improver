"""Failure Mode Analysis Engine

Understands and addresses failure patterns in prompt improvement.
Analyzes test results to identify systemic failures, edge cases, and root causes
to improve overall system effectiveness.
"""

import logging
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

try:
    import prometheus_client
    from prometheus_client import (
        Counter as PromCounter,
        Gauge,
        Histogram,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    warnings.warn(
        "Prometheus client not available. Install with: pip install prometheus-client"
    )

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


@dataclass
class FailurePattern:
    """Identified failure pattern"""

    pattern_id: str
    description: str
    frequency: int
    severity: float
    affected_contexts: list[str]
    common_characteristics: dict[str, Any]
    example_cases: list[dict[str, Any]]


@dataclass
class RootCause:
    """Root cause of failures"""

    cause_id: str
    description: str
    affected_failures: int
    correlation_strength: float
    evidence: list[str]
    suggested_fixes: list[str]


@dataclass
class EdgeCase:
    """Edge case that causes failures"""

    case_id: str
    description: str
    characteristics: dict[str, Any]
    failure_examples: list[dict[str, Any]]
    suggested_handling: str


@dataclass
class SystematicIssue:
    """Systematic issue affecting multiple areas"""

    issue_id: str
    description: str
    scope: str  # 'global', 'rule_specific', 'context_specific'
    affected_rules: list[str]
    affected_contexts: list[str]
    impact_magnitude: float
    priority: str  # 'critical', 'high', 'medium', 'low'


@dataclass
class FailureRecommendation:
    """Recommendation to address failures"""

    recommendation_id: str
    type: str  # 'rule_fix', 'new_rule', 'data_collection', 'systematic_fix'
    description: str
    expected_impact: float
    implementation_effort: str
    priority: str
    target_failures: list[str]


@dataclass
class MLFailureMode:
    """Structured representation of ML system failure modes (Phase 3)"""

    failure_type: str  # 'data', 'model', 'infrastructure', 'deployment'
    description: str
    severity: int  # 1-10 scale
    occurrence: int  # 1-10 scale
    detection: int  # 1-10 scale
    rpn: int  # Risk Priority Number
    root_causes: list[str]
    detection_methods: list[str]
    mitigation_strategies: list[str]


@dataclass
class RobustnessTestResult:
    """Results from robustness validation tests (Phase 3)"""

    test_type: str  # 'noise', 'adversarial', 'drift', 'edge_case'
    success_rate: float
    degradation_rate: float
    robustness_score: float
    failed_samples: list[dict[str, Any]]
    recommendations: list[str]


@dataclass
class PrometheusAlert:
    """Prometheus alert definition (Phase 3)"""

    alert_name: str
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str  # 'critical', 'warning', 'info'
    description: str
    triggered_at: datetime | None = None


class FailureModeAnalyzer:
    """Failure Mode Analysis Engine for prompt improvement systems"""

    def __init__(self, config: FailureConfig | None = None):
        """Initialize the failure mode analyzer

        Args:
            config: Configuration for failure analysis
        """
        self.config = config or FailureConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Failure tracking
        self.failure_patterns: dict[str, FailurePattern] = {}
        self.root_causes: dict[str, RootCause] = {}
        self.systematic_issues: list[SystematicIssue] = []
        self.edge_cases: list[EdgeCase] = []

        # Initialize text analysis tools
        self.text_vectorizer = TfidfVectorizer(
            max_features=500, stop_words="english", ngram_range=(1, 2)
        )

        # Phase 3 enhancements - Robustness validation state
        self.robustness_test_results: list[RobustnessTestResult] = []
        self.ensemble_detectors: dict[str, Any] = {}
        self.anomaly_detectors: dict[str, Any] = {}
        self.adversarial_attack_results: list[dict[str, Any]] = []

        # Phase 3 enhancements - Prometheus monitoring state
        self.prometheus_metrics: dict[str, Any] = {}
        self.active_alerts: list[PrometheusAlert] = []
        self.alert_history: list[PrometheusAlert] = []

        # Initialize Phase 3 components
        if self.config.enable_robustness_validation:
            self._initialize_robustness_components()

        if self.config.enable_prometheus_monitoring:
            self._initialize_prometheus_monitoring()

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

        # Core failure analysis
        failure_analysis = {
            "summary": self._generate_failure_summary(failures, test_results),
            "patterns": await self._identify_failure_patterns(failures),
            "root_causes": await self._identify_root_causes(failures, successes),
            "rule_gaps": await self._find_missing_rules(failures),
            "edge_cases": await self._find_edge_cases(failures),
            "systematic_issues": await self._find_systematic_issues(
                failures, test_results
            ),
            # ML FMEA Analysis
            "ml_fmea": await self._perform_ml_fmea_analysis(failures, test_results),
            "anomaly_detection": await self._perform_ensemble_anomaly_detection(
                failures
            ),
            "risk_assessment": await self._calculate_risk_priority_numbers(failures),
            # Comparative analysis
            "failure_vs_success": self._compare_failures_with_successes(
                failures, successes
            ),
            # Actionable outputs
            "recommendations": await self._generate_failure_recommendations(failures),
            "prioritized_fixes": await self._prioritize_fixes_by_impact(failures),
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
            await self._update_prometheus_metrics(failure_analysis)
            failure_analysis[
                "prometheus_alerts"
            ] = await self._check_and_trigger_alerts(failure_analysis)

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

    async def _identify_failure_patterns(
        self, failures: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify patterns in failures"""
        patterns = []

        # Pattern 1: Context-based patterns
        context_patterns = await self._find_context_patterns(failures)
        patterns.extend(context_patterns)

        # Pattern 2: Prompt characteristic patterns
        prompt_patterns = await self._find_prompt_patterns(failures)
        patterns.extend(prompt_patterns)

        # Pattern 3: Rule application patterns
        rule_patterns = await self._find_rule_patterns(failures)
        patterns.extend(rule_patterns)

        # Pattern 4: Temporal patterns
        temporal_patterns = await self._find_temporal_patterns(failures)
        patterns.extend(temporal_patterns)

        # Sort by frequency and significance
        patterns.sort(key=lambda p: (p["frequency"], p["severity"]), reverse=True)

        return patterns[: self.config.max_patterns]

    async def _find_context_patterns(
        self, failures: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find context-based failure patterns"""
        patterns = []

        # Group failures by context
        context_failures = defaultdict(list)
        for failure in failures:
            context = failure.get("context", {})
            context_key = self._get_context_key(context)
            context_failures[context_key].append(failure)

        # Identify significant context patterns
        for context_key, context_group in context_failures.items():
            if len(context_group) >= self.config.min_pattern_size:
                avg_score = np.mean([
                    f.get("overallImprovement", 0) or f.get("improvementScore", 0)
                    for f in context_group
                ])

                pattern = {
                    "pattern_id": f"context_{hash(context_key) % 10000}",
                    "type": "context",
                    "description": f"Failures in {context_key} context",
                    "frequency": len(context_group),
                    "severity": 1 - avg_score,  # Lower score = higher severity
                    "characteristics": {
                        "context": context_key,
                        "avg_failure_score": avg_score,
                        "common_issues": self._extract_common_issues(context_group),
                    },
                    "examples": context_group[:3],
                }
                patterns.append(pattern)

        return patterns

    async def _find_prompt_patterns(
        self, failures: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find prompt characteristic patterns"""
        patterns = []

        # Analyze prompt characteristics
        prompt_texts = [
            f.get("originalPrompt", "") for f in failures if f.get("originalPrompt")
        ]

        if len(prompt_texts) < self.config.min_pattern_size:
            return patterns

        # Length-based patterns
        lengths = [len(text.split()) for text in prompt_texts]
        if lengths:
            avg_length = np.mean(lengths)
            std_length = np.std(lengths)

            # Very short prompts
            short_prompts = [i for i, length in enumerate(lengths) if length < 5]
            if len(short_prompts) >= self.config.min_pattern_size:
                patterns.append({
                    "pattern_id": "prompt_too_short",
                    "type": "prompt_characteristic",
                    "description": "Failures with very short prompts (< 5 words)",
                    "frequency": len(short_prompts),
                    "severity": 0.7,
                    "characteristics": {
                        "avg_length": np.mean([lengths[i] for i in short_prompts]),
                        "threshold": "< 5 words",
                    },
                    "examples": [failures[i] for i in short_prompts[:3]],
                })

            # Very long prompts
            long_prompts = [i for i, length in enumerate(lengths) if length > 100]
            if len(long_prompts) >= self.config.min_pattern_size:
                patterns.append({
                    "pattern_id": "prompt_too_long",
                    "type": "prompt_characteristic",
                    "description": "Failures with very long prompts (> 100 words)",
                    "frequency": len(long_prompts),
                    "severity": 0.6,
                    "characteristics": {
                        "avg_length": np.mean([lengths[i] for i in long_prompts]),
                        "threshold": "> 100 words",
                    },
                    "examples": [failures[i] for i in long_prompts[:3]],
                })

        # Content-based patterns using clustering
        if len(prompt_texts) >= 10:
            try:
                # Vectorize prompts
                vectors = self.text_vectorizer.fit_transform(prompt_texts)

                # Cluster similar failing prompts
                clustering = DBSCAN(eps=0.3, min_samples=self.config.min_pattern_size)
                clusters = clustering.fit_predict(vectors.toarray())

                # Analyze each cluster
                for cluster_id in set(clusters):
                    if cluster_id != -1:  # Ignore noise points
                        cluster_indices = np.where(clusters == cluster_id)[0]
                        if len(cluster_indices) >= self.config.min_pattern_size:
                            cluster_prompts = [prompt_texts[i] for i in cluster_indices]

                            # Find common words/phrases
                            common_terms = self._find_common_terms(cluster_prompts)

                            patterns.append({
                                "pattern_id": f"prompt_cluster_{cluster_id}",
                                "type": "prompt_content",
                                "description": f"Failures with similar content: {', '.join(common_terms[:3])}",
                                "frequency": len(cluster_indices),
                                "severity": 0.5,
                                "characteristics": {
                                    "common_terms": common_terms,
                                    "cluster_size": len(cluster_indices),
                                },
                                "examples": [failures[i] for i in cluster_indices[:3]],
                            })
            except Exception as e:
                self.logger.warning(f"Text clustering failed: {e}")

        return patterns

    async def _find_rule_patterns(
        self, failures: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find rule application patterns"""
        patterns = []

        # Analyze rule combinations that fail
        rule_combinations = defaultdict(list)

        for failure in failures:
            applied_rules = failure.get("appliedRules", [])
            rule_ids = [
                rule.get("ruleId") or rule.get("id", "unknown")
                for rule in applied_rules
            ]

            # Single rule failures
            for rule_id in rule_ids:
                rule_combinations[rule_id].append(failure)

            # Rule pair failures (if multiple rules applied)
            if len(rule_ids) > 1:
                for i, rule1 in enumerate(rule_ids):
                    for rule2 in rule_ids[i + 1 :]:
                        combo_key = f"{rule1}+{rule2}"
                        rule_combinations[combo_key].append(failure)

        # Identify significant rule patterns
        for combo_key, combo_failures in rule_combinations.items():
            if len(combo_failures) >= self.config.min_pattern_size:
                avg_score = np.mean([
                    f.get("overallImprovement", 0) or f.get("improvementScore", 0)
                    for f in combo_failures
                ])

                pattern = {
                    "pattern_id": f"rule_{hash(combo_key) % 10000}",
                    "type": "rule_application",
                    "description": f"Failures when applying {combo_key}",
                    "frequency": len(combo_failures),
                    "severity": 1 - avg_score,
                    "characteristics": {
                        "rule_combination": combo_key,
                        "avg_failure_score": avg_score,
                        "is_combination": "+" in combo_key,
                    },
                    "examples": combo_failures[:3],
                }
                patterns.append(pattern)

        return patterns

    async def _find_temporal_patterns(
        self, failures: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find temporal failure patterns"""
        patterns = []

        # Extract timestamps
        timestamps = []
        for failure in failures:
            timestamp_str = failure.get("timestamp") or failure.get("createdAt")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    timestamps.append((timestamp, failure))
                except:
                    continue

        if len(timestamps) < self.config.min_pattern_size:
            return patterns

        # Sort by timestamp
        timestamps.sort(key=lambda x: x[0])

        # Look for time-based clusters
        time_deltas = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i][0] - timestamps[i - 1][0]).total_seconds()
            time_deltas.append(delta)

        if time_deltas:
            # Find clusters of failures in time
            median_delta = np.median(time_deltas)

            # Identify bursts (failures clustered in time)
            burst_threshold = median_delta / 3  # Failures closer than 1/3 median
            bursts = []
            current_burst = [timestamps[0]]

            for i in range(1, len(timestamps)):
                delta = (timestamps[i][0] - timestamps[i - 1][0]).total_seconds()
                if delta <= burst_threshold:
                    current_burst.append(timestamps[i])
                else:
                    if len(current_burst) >= self.config.min_pattern_size:
                        bursts.append(current_burst)
                    current_burst = [timestamps[i]]

            # Add final burst if significant
            if len(current_burst) >= self.config.min_pattern_size:
                bursts.append(current_burst)

            # Create patterns for significant bursts
            for i, burst in enumerate(bursts):
                if len(burst) >= self.config.min_pattern_size:
                    burst_failures = [item[1] for item in burst]
                    start_time = burst[0][0]
                    end_time = burst[-1][0]
                    duration = (end_time - start_time).total_seconds()

                    patterns.append({
                        "pattern_id": f"temporal_burst_{i}",
                        "type": "temporal",
                        "description": f"Burst of {len(burst)} failures over {duration:.1f} seconds",
                        "frequency": len(burst),
                        "severity": 0.6,
                        "characteristics": {
                            "start_time": start_time.isoformat(),
                            "end_time": end_time.isoformat(),
                            "duration_seconds": duration,
                            "failures_per_second": len(burst) / max(duration, 1),
                        },
                        "examples": burst_failures[:3],
                    })

        return patterns

    async def _identify_root_causes(
        self, failures: list[dict[str, Any]], successes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify root causes of failures through comparative analysis"""
        root_causes = []

        # Compare characteristics between failures and successes
        failure_characteristics = self._extract_characteristics(failures)
        success_characteristics = self._extract_characteristics(successes)

        # Statistical comparison
        for characteristic, failure_values in failure_characteristics.items():
            success_values = success_characteristics.get(characteristic, [])

            if len(failure_values) >= 3 and len(success_values) >= 3:
                # Perform statistical test
                try:
                    if self._is_numeric_characteristic(failure_values):
                        # Numeric comparison
                        statistic, p_value = stats.ttest_ind(
                            failure_values, success_values
                        )
                        effect_size = (
                            np.mean(failure_values) - np.mean(success_values)
                        ) / np.sqrt(
                            (np.var(failure_values) + np.var(success_values)) / 2
                        )

                        if p_value < 0.05 and abs(effect_size) > 0.5:
                            root_causes.append({
                                "cause_id": f"numeric_{characteristic}",
                                "type": "statistical_difference",
                                "description": f"Significant difference in {characteristic}",
                                "affected_failures": len(failure_values),
                                "correlation_strength": abs(effect_size),
                                "evidence": [
                                    f"Failure mean: {np.mean(failure_values):.3f}",
                                    f"Success mean: {np.mean(success_values):.3f}",
                                    f"P-value: {p_value:.3f}",
                                    f"Effect size: {effect_size:.3f}",
                                ],
                                "statistical_significance": p_value < 0.05,
                            })
                    else:
                        # Categorical comparison
                        failure_counts = Counter(failure_values)
                        success_counts = Counter(success_values)

                        # Find categories overrepresented in failures
                        total_failures = len(failure_values)
                        total_successes = len(success_values)

                        for category in failure_counts:
                            failure_rate = failure_counts[category] / total_failures
                            success_rate = (
                                success_counts.get(category, 0) / total_successes
                            )

                            if (
                                failure_rate > success_rate * 2
                                and failure_counts[category] >= 3
                            ):
                                root_causes.append({
                                    "cause_id": f"categorical_{characteristic}_{category}",
                                    "type": "overrepresented_category",
                                    "description": f"{category} in {characteristic} overrepresented in failures",
                                    "affected_failures": failure_counts[category],
                                    "correlation_strength": failure_rate
                                    / (success_rate + 0.01),
                                    "evidence": [
                                        f"Failure rate: {failure_rate:.3f}",
                                        f"Success rate: {success_rate:.3f}",
                                        f"Overrepresentation: {failure_rate / (success_rate + 0.01):.1f}x",
                                    ],
                                    "category": category,
                                })
                except Exception as e:
                    self.logger.warning(
                        f"Statistical comparison failed for {characteristic}: {e}"
                    )

        # Sort by correlation strength
        root_causes.sort(key=lambda x: x["correlation_strength"], reverse=True)

        return root_causes[: self.config.max_root_causes]

    async def _find_missing_rules(
        self, failures: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find gaps where new rules might be needed"""
        rule_gaps = []

        # Analyze failure characteristics to identify underserved areas
        characteristics = self._extract_characteristics(failures)

        # Look for common failure contexts without specific rules
        context_gaps = defaultdict(list)

        for failure in failures:
            context = failure.get("context", {})
            applied_rules = failure.get("appliedRules", [])

            # Check if this context type has limited rule coverage
            context_key = self._get_context_key(context)
            rule_count = len(applied_rules)

            if rule_count < 3:  # Threshold for insufficient rule coverage
                context_gaps[context_key].append(failure)

        # Identify significant gaps
        for context_key, gap_failures in context_gaps.items():
            if len(gap_failures) >= self.config.min_pattern_size:
                # Analyze what types of improvements are needed
                common_issues = self._extract_common_issues(gap_failures)

                rule_gaps.append({
                    "gap_id": f"rule_gap_{hash(context_key) % 10000}",
                    "context": context_key,
                    "affected_failures": len(gap_failures),
                    "description": f"Insufficient rule coverage for {context_key}",
                    "common_issues": common_issues,
                    "suggested_rule_types": self._suggest_rule_types(gap_failures),
                    "priority": "high" if len(gap_failures) > 10 else "medium",
                })

        return rule_gaps

    async def _find_edge_cases(
        self, failures: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find edge cases that cause failures"""
        edge_cases = []

        # Use clustering to find unusual/outlier failures
        numeric_features = []
        feature_names = []

        for failure in failures:
            features = []

            # Extract numeric features
            prompt_length = len(failure.get("originalPrompt", "").split())
            features.append(prompt_length)
            if len(feature_names) == 0:
                feature_names.append("prompt_length")

            # Improvement score
            score = failure.get("overallImprovement", 0) or failure.get(
                "improvementScore", 0
            )
            features.append(score)
            if len(feature_names) == 1:
                feature_names.append("improvement_score")

            # Number of applied rules
            rule_count = len(failure.get("appliedRules", []))
            features.append(rule_count)
            if len(feature_names) == 2:
                feature_names.append("rule_count")

            numeric_features.append(features)

        if len(numeric_features) >= 5:
            # Find outliers using statistical methods
            features_array = np.array(numeric_features)

            for i, feature_name in enumerate(feature_names):
                feature_values = features_array[:, i]
                z_scores = np.abs(stats.zscore(feature_values))
                outliers = np.where(z_scores > self.config.outlier_threshold)[0]

                if len(outliers) >= 2:
                    outlier_failures = [failures[idx] for idx in outliers]

                    edge_cases.append({
                        "case_id": f"edge_case_{feature_name}",
                        "type": "statistical_outlier",
                        "description": f"Extreme values in {feature_name}",
                        "affected_failures": len(outliers),
                        "characteristics": {
                            "feature": feature_name,
                            "outlier_values": feature_values[outliers].tolist(),
                            "normal_range": [
                                float(np.percentile(feature_values, 5)),
                                float(np.percentile(feature_values, 95)),
                            ],
                        },
                        "examples": outlier_failures[:3],
                    })

        return edge_cases

    async def _find_systematic_issues(
        self, failures: list[dict[str, Any]], all_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find systematic issues affecting the entire system"""
        systematic_issues = []

        # Issue 1: Rules that consistently perform poorly
        rule_performance = defaultdict(list)
        for result in all_results:
            applied_rules = result.get("appliedRules", [])
            for rule in applied_rules:
                rule_id = rule.get("ruleId") or rule.get("id", "unknown")
                score = result.get("overallImprovement", 0) or result.get(
                    "improvementScore", 0
                )
                rule_performance[rule_id].append(score)

        for rule_id, scores in rule_performance.items():
            if len(scores) >= 10:  # Sufficient sample size
                avg_score = np.mean(scores)
                failure_rate = sum(
                    1 for s in scores if s < self.config.failure_threshold
                ) / len(scores)

                if failure_rate > 0.4:  # More than 40% failure rate
                    systematic_issues.append({
                        "issue_id": f"systematic_rule_{rule_id}",
                        "type": "poor_rule_performance",
                        "scope": "rule_specific",
                        "description": f"Rule {rule_id} has high failure rate",
                        "affected_rules": [rule_id],
                        "impact_magnitude": failure_rate,
                        "evidence": [
                            f"Failure rate: {failure_rate:.3f}",
                            f"Average score: {avg_score:.3f}",
                            f"Sample size: {len(scores)}",
                        ],
                        "priority": "critical" if failure_rate > 0.6 else "high",
                    })

        # Issue 2: Context-wide problems
        context_performance = defaultdict(list)
        for result in all_results:
            context = result.get("context", {})
            context_key = self._get_context_key(context)
            score = result.get("overallImprovement", 0) or result.get(
                "improvementScore", 0
            )
            context_performance[context_key].append(score)

        for context_key, scores in context_performance.items():
            if len(scores) >= 15:  # Sufficient sample size
                failure_rate = sum(
                    1 for s in scores if s < self.config.failure_threshold
                ) / len(scores)

                if failure_rate > 0.3:  # More than 30% failure rate
                    systematic_issues.append({
                        "issue_id": f"systematic_context_{hash(context_key) % 10000}",
                        "type": "context_performance",
                        "scope": "context_specific",
                        "description": f"Context {context_key} has elevated failure rate",
                        "affected_contexts": [context_key],
                        "impact_magnitude": failure_rate,
                        "evidence": [
                            f"Failure rate: {failure_rate:.3f}",
                            f"Sample size: {len(scores)}",
                        ],
                        "priority": "high" if failure_rate > 0.5 else "medium",
                    })

        return systematic_issues

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

    async def _generate_failure_recommendations(
        self, failures: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate actionable recommendations to address failures"""
        recommendations = []

        # Get identified patterns for recommendation generation
        patterns = await self._identify_failure_patterns(failures)

        for pattern in patterns[:5]:  # Top 5 patterns
            if pattern["type"] == "context":
                recommendations.append({
                    "recommendation_id": f"fix_{pattern['pattern_id']}",
                    "type": "context_optimization",
                    "description": f"Optimize rules for {pattern['characteristics']['context']} context",
                    "expected_impact": pattern["frequency"] / len(failures),
                    "implementation_effort": "medium",
                    "priority": "high" if pattern["frequency"] > 5 else "medium",
                    "target_failures": pattern["frequency"],
                })

            elif pattern["type"] == "rule_application":
                recommendations.append({
                    "recommendation_id": f"fix_{pattern['pattern_id']}",
                    "type": "rule_fix",
                    "description": f"Fix or replace rule combination {pattern['characteristics']['rule_combination']}",
                    "expected_impact": pattern["frequency"] / len(failures),
                    "implementation_effort": "low"
                    if not pattern["characteristics"]["is_combination"]
                    else "medium",
                    "priority": "critical" if pattern["severity"] > 0.8 else "high",
                    "target_failures": pattern["frequency"],
                })

            elif pattern["type"] == "prompt_characteristic":
                recommendations.append({
                    "recommendation_id": f"fix_{pattern['pattern_id']}",
                    "type": "input_validation",
                    "description": f"Add handling for {pattern['description'].lower()}",
                    "expected_impact": pattern["frequency"] / len(failures),
                    "implementation_effort": "low",
                    "priority": "medium",
                    "target_failures": pattern["frequency"],
                })

        # Sort by expected impact and priority
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda x: (priority_order.get(x["priority"], 0), x["expected_impact"]),
            reverse=True,
        )

        return recommendations

    async def _prioritize_fixes_by_impact(
        self, failures: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prioritize fixes by potential impact"""
        # This would involve more sophisticated impact modeling
        # For now, return a simplified prioritization

        fixes = []

        # High-impact, low-effort fixes
        fixes.append({
            "fix_id": "input_validation",
            "description": "Add input validation for extreme prompt lengths",
            "impact": "medium",
            "effort": "low",
            "priority_score": 8,
            "estimated_failures_prevented": len([
                f
                for f in failures
                if len(f.get("originalPrompt", "").split()) < 5
                or len(f.get("originalPrompt", "").split()) > 100
            ]),
        })

        # Medium-impact, medium-effort fixes
        fixes.append({
            "fix_id": "rule_optimization",
            "description": "Optimize poorly performing rules",
            "impact": "high",
            "effort": "medium",
            "priority_score": 7,
            "estimated_failures_prevented": len(failures) // 3,  # Estimate
        })

        # High-impact, high-effort fixes
        fixes.append({
            "fix_id": "context_specialization",
            "description": "Create context-specific rule variations",
            "impact": "high",
            "effort": "high",
            "priority_score": 6,
            "estimated_failures_prevented": len(failures) // 2,  # Estimate
        })

        return sorted(fixes, key=lambda x: x["priority_score"], reverse=True)

    # Helper methods

    def _get_context_key(self, context: dict[str, Any]) -> str:
        """Generate context key for grouping"""
        if not context:
            return "unknown"

        project_type = context.get("projectType", "unknown")
        domain = context.get("domain", "unknown")
        return f"{project_type}|{domain}"

    def _extract_characteristics(
        self, results: list[dict[str, Any]]
    ) -> dict[str, list[Any]]:
        """Extract characteristics from results for analysis"""
        characteristics = defaultdict(list)

        for result in results:
            # Prompt characteristics
            prompt = result.get("originalPrompt", "")
            characteristics["prompt_length"].append(len(prompt.split()))
            characteristics["prompt_has_question"].append("?" in prompt)

            # Context characteristics
            context = result.get("context", {})
            characteristics["project_type"].append(
                context.get("projectType", "unknown")
            )
            characteristics["domain"].append(context.get("domain", "unknown"))

            # Rule characteristics
            applied_rules = result.get("appliedRules", [])
            characteristics["rule_count"].append(len(applied_rules))

            # Performance characteristics
            score = result.get("overallImprovement", 0) or result.get(
                "improvementScore", 0
            )
            characteristics["score"].append(score)

        return dict(characteristics)

    def _is_numeric_characteristic(self, values: list[Any]) -> bool:
        """Check if characteristic values are numeric"""
        try:
            [float(v) for v in values[:5]]  # Test first 5 values
            return True
        except:
            return False

    def _extract_common_issues(self, failures: list[dict[str, Any]]) -> list[str]:
        """Extract common issues from a group of failures"""
        issues = []

        # Analyze failure reasons if available
        error_messages = [f.get("error", "") for f in failures if f.get("error")]

        if error_messages:
            # Simple keyword extraction
            common_keywords = Counter()
            for msg in error_messages:
                words = re.findall(r"\w+", msg.lower())
                for word in words:
                    if len(word) > 3:  # Skip short words
                        common_keywords[word] += 1

            # Return most common keywords
            issues.extend([word for word, count in common_keywords.most_common(3)])

        return issues

    def _suggest_rule_types(self, failures: list[dict[str, Any]]) -> list[str]:
        """Suggest what types of rules might address these failures"""
        suggestions = []

        # Analyze common prompt characteristics
        prompt_lengths = [len(f.get("originalPrompt", "").split()) for f in failures]
        avg_length = np.mean(prompt_lengths) if prompt_lengths else 0

        if avg_length < 10:
            suggestions.append("prompt_expansion")
        elif avg_length > 50:
            suggestions.append("prompt_condensation")

        # Check for question patterns
        questions = sum(1 for f in failures if "?" in f.get("originalPrompt", ""))
        if questions > len(failures) * 0.6:
            suggestions.append("question_optimization")

        # Default suggestions
        if not suggestions:
            suggestions.extend(["clarity_improvement", "structure_enhancement"])

        return suggestions

    def _find_common_terms(self, texts: list[str]) -> list[str]:
        """Find common terms in a list of texts"""
        try:
            # Use TF-IDF to find important terms
            vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Get feature names and their average scores
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

            # Sort by importance
            term_scores = list(zip(feature_names, mean_scores, strict=False))
            term_scores.sort(key=lambda x: x[1], reverse=True)

            return [term for term, score in term_scores[:10] if score > 0.1]
        except:
            return []

    def _initialize_ml_fmea_database(self) -> list:
        """Initialize ML FMEA failure modes database following Microsoft Learn guidelines"""
        from dataclasses import dataclass, field

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

    async def _perform_ml_fmea_analysis(
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

    async def _perform_ensemble_anomaly_detection(
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

    async def _calculate_risk_priority_numbers(
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

    # ==================== PHASE 3 ENHANCEMENTS ====================

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

            # Initialize ML FMEA database
            self.ml_failure_modes = self._initialize_ml_fmea_database()

            self.logger.info(
                "Robustness validation components initialized successfully"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize robustness components: {e}")
            self.config.enable_robustness_validation = False

    def _initialize_prometheus_monitoring(self) -> None:
        """Initialize Phase 3 Prometheus monitoring components"""
        try:
            if PROMETHEUS_AVAILABLE:
                # Define Prometheus metrics
                self.prometheus_metrics = {
                    "failure_rate": Gauge(
                        "ml_failure_rate", "Current ML system failure rate"
                    ),
                    "failure_count": PromCounter(
                        "ml_failures_total",
                        "Total number of ML failures",
                        ["failure_type", "severity"],
                    ),
                    "response_time": Histogram(
                        "ml_response_time_seconds",
                        "ML system response time distribution",
                    ),
                    "anomaly_score": Gauge(
                        "ml_anomaly_score", "Current anomaly detection score"
                    ),
                    "rpn_score": Gauge(
                        "ml_risk_priority_number", "ML FMEA Risk Priority Number"
                    ),
                }

                # Start Prometheus HTTP server
                try:
                    start_http_server(self.config.prometheus_port)
                    self.logger.info(
                        f"Prometheus metrics server started on port {self.config.prometheus_port}"
                    )
                except Exception as e:
                    self.logger.warning(f"Could not start Prometheus server: {e}")

                # Initialize alert definitions
                self._initialize_alert_definitions()

                self.logger.info("Prometheus monitoring initialized successfully")
            else:
                self.logger.warning("Prometheus client not available")

        except Exception as e:
            self.logger.error(f"Failed to initialize Prometheus monitoring: {e}")
            self.config.enable_prometheus_monitoring = False

    def _initialize_alert_definitions(self) -> None:
        """Initialize alert definitions for monitoring"""
        alert_definitions = [
            PrometheusAlert(
                alert_name="HighFailureRate",
                metric_name="ml_failure_rate",
                threshold=self.config.alert_thresholds["failure_rate"],
                comparison="gt",
                severity="critical",
                description="ML system failure rate exceeds threshold",
            ),
            PrometheusAlert(
                alert_name="SlowResponseTime",
                metric_name="ml_response_time_seconds",
                threshold=self.config.alert_thresholds["response_time_ms"] / 1000.0,
                comparison="gt",
                severity="warning",
                description="ML system response time exceeds threshold",
            ),
            PrometheusAlert(
                alert_name="HighAnomalyScore",
                metric_name="ml_anomaly_score",
                threshold=self.config.alert_thresholds["anomaly_score"],
                comparison="gt",
                severity="warning",
                description="Anomaly detection score is elevated",
            ),
        ]

        # Store alert definitions for monitoring
        self.alert_definitions = alert_definitions
        self.logger.info(f"Initialized {len(alert_definitions)} alert definitions")

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
            # In practice, you would use ART with actual ML models

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
        edge_cases = await self._find_edge_cases(failures)
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

                    # Identify samples that contribute most to drift
                    score_differences = []
                    for i, (baseline, current) in enumerate(
                        zip(baseline_scores, current_scores, strict=False)
                    ):
                        diff = abs(current - baseline)
                        score_differences.append((i, diff, baseline, current))

                    # Sort by difference and take top contributors
                    score_differences.sort(key=lambda x: x[1], reverse=True)
                    for i, (idx, diff, baseline, current) in enumerate(
                        score_differences[:5]
                    ):
                        drift_test_results["failed_samples"].append({
                            "sample_index": idx,
                            "baseline_score": baseline,
                            "current_score": current,
                            "drift_contribution": diff,
                        })

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

    async def _update_prometheus_metrics(
        self, failure_analysis: dict[str, Any]
    ) -> None:
        """Update Prometheus metrics with analysis results"""
        if not PROMETHEUS_AVAILABLE or not self.prometheus_metrics:
            return

        try:
            metadata = failure_analysis.get("metadata", {})

            # Update failure rate metric
            failure_rate = metadata.get("failure_rate", 0.0)
            self.prometheus_metrics["failure_rate"].set(failure_rate)

            # Update failure count by type and severity
            summary = failure_analysis.get("summary", {})
            severity = summary.get("severity", "unknown")
            total_failures = metadata.get("total_failures", 0)

            self.prometheus_metrics["failure_count"].labels(
                failure_type="general", severity=severity
            ).inc(total_failures)

            # Update anomaly score if available
            anomaly_detection = failure_analysis.get("anomaly_detection", {})
            if "anomaly_summary" in anomaly_detection:
                anomaly_rate = anomaly_detection["anomaly_summary"].get(
                    "consensus_anomaly_rate", 0
                )
                self.prometheus_metrics["anomaly_score"].set(anomaly_rate / 100.0)

            # Update RPN score if available
            risk_assessment = failure_analysis.get("risk_assessment", {})
            if "overall_risk_score" in risk_assessment:
                rpn_score = risk_assessment["overall_risk_score"]
                self.prometheus_metrics["rpn_score"].set(rpn_score)

            self.logger.debug("Prometheus metrics updated successfully")

        except Exception as e:
            self.logger.error(f"Failed to update Prometheus metrics: {e}")

    async def _check_and_trigger_alerts(
        self, failure_analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Check thresholds and trigger alerts if necessary"""
        triggered_alerts = []
        current_time = datetime.now()

        if not hasattr(self, "alert_definitions"):
            return triggered_alerts

        try:
            metadata = failure_analysis.get("metadata", {})

            for alert_def in self.alert_definitions:
                metric_value = self._extract_metric_value(
                    failure_analysis, alert_def.metric_name
                )

                if metric_value is not None and self._check_threshold(
                    metric_value, alert_def
                ):
                    # Check if alert is not in cooldown period
                    if self._is_alert_ready_to_trigger(alert_def, current_time):
                        # Trigger alert
                        alert_def.triggered_at = current_time

                        triggered_alert = {
                            "alert_name": alert_def.alert_name,
                            "metric_name": alert_def.metric_name,
                            "current_value": metric_value,
                            "threshold": alert_def.threshold,
                            "severity": alert_def.severity,
                            "description": alert_def.description,
                            "triggered_at": current_time.isoformat(),
                            "recommended_actions": self._get_alert_recommendations(
                                alert_def.alert_name
                            ),
                        }

                        triggered_alerts.append(triggered_alert)
                        self.active_alerts.append(alert_def)
                        self.alert_history.append(alert_def)

                        self.logger.warning(
                            f"Alert triggered: {alert_def.alert_name} - "
                            f"{alert_def.metric_name}={metric_value} > {alert_def.threshold}"
                        )

            # Update active alerts (remove resolved ones)
            self._update_active_alerts(failure_analysis, current_time)

        except Exception as e:
            self.logger.error(f"Alert checking failed: {e}")

        return triggered_alerts

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

    def _check_threshold(self, value: float, alert_def: PrometheusAlert) -> bool:
        """Check if metric value exceeds alert threshold"""
        if alert_def.comparison == "gt":
            return value > alert_def.threshold
        if alert_def.comparison == "lt":
            return value < alert_def.threshold
        if alert_def.comparison == "eq":
            return abs(value - alert_def.threshold) < 0.001
        return False

    def _is_alert_ready_to_trigger(
        self, alert_def: PrometheusAlert, current_time: datetime
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
        """Update active alerts list, removing resolved alerts"""
        resolved_alerts = []

        for alert in self.active_alerts:
            metric_value = self._extract_metric_value(
                failure_analysis, alert.metric_name
            )

            # Check if alert condition is no longer met
            if metric_value is not None and not self._check_threshold(
                metric_value, alert
            ):
                resolved_alerts.append(alert)
                self.logger.info(f"Alert resolved: {alert.alert_name}")

        # Remove resolved alerts from active list
        for alert in resolved_alerts:
            self.active_alerts.remove(alert)

    def _initialize_ml_fmea_failure_modes(self) -> list[Any]:
        """Initialize comprehensive ML FMEA failure modes database"""
        return [
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
            ),
            MLFailureMode(
                failure_type="data",
                description="Data quality issues - missing validation, inconsistent collection",
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
            ),
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
            ),
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
            ),
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
            ),
        ]
