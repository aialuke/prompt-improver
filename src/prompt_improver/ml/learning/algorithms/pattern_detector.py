"""Pattern Detection Engine

Analyzes failure patterns in prompt improvement test results.
Identifies context-based, prompt characteristic, rule application, and temporal patterns
to understand systemic failure modes.
"""

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class PatternDetector:
    """Pattern Detection Engine for identifying failure patterns in test results"""

    def __init__(self, config):
        """Initialize the pattern detector
        
        Args:
            config: FailureConfig instance containing analysis parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize text processing tools
        self.text_vectorizer = TfidfVectorizer(max_features=100, stop_words="english")

    async def identify_failure_patterns(
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

    async def identify_root_causes_comparative(
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

    async def find_missing_rules(
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

    async def find_edge_cases(
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

    async def find_systematic_issues(
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