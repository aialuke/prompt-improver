"""Optimization Validator

Validates optimization results and ensures improvements are genuine
and statistically significant before deployment.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for optimization validation"""

    min_sample_size: int = 30
    significance_level: float = 0.05
    min_effect_size: float = 0.2
    validation_duration_hours: int = 24


class OptimizationValidator:
    """Validator for optimization results"""

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def validate_optimization(
        self,
        optimization_id: str,
        baseline_data: dict[str, Any],
        optimized_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate an optimization result"""
        self.logger.info(f"Validating optimization {optimization_id}")

        # Extract performance scores
        baseline_scores = baseline_data.get("scores", [])
        optimized_scores = optimized_data.get("scores", [])

        if (
            len(baseline_scores) < self.config.min_sample_size
            or len(optimized_scores) < self.config.min_sample_size
        ):
            return {
                "optimization_id": optimization_id,
                "valid": False,
                "reason": "Insufficient sample size for validation",
                "validation_date": datetime.now().isoformat(),
            }

        # Perform statistical test
        try:
            statistic, p_value = stats.ttest_ind(optimized_scores, baseline_scores)

            # Calculate effect size
            pooled_std = np.sqrt(
                (np.var(baseline_scores) + np.var(optimized_scores)) / 2
            )
            effect_size = (
                np.mean(optimized_scores) - np.mean(baseline_scores)
            ) / pooled_std

            # Validation criteria
            statistically_significant = p_value < self.config.significance_level
            practically_significant = effect_size > self.config.min_effect_size
            improvement_detected = np.mean(optimized_scores) > np.mean(baseline_scores)

            valid = (
                statistically_significant
                and practically_significant
                and improvement_detected
            )

            return {
                "optimization_id": optimization_id,
                "valid": valid,
                "statistical_significance": statistically_significant,
                "practical_significance": practically_significant,
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "baseline_mean": float(np.mean(baseline_scores)),
                "optimized_mean": float(np.mean(optimized_scores)),
                "improvement": float(
                    np.mean(optimized_scores) - np.mean(baseline_scores)
                ),
                "validation_date": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {
                "optimization_id": optimization_id,
                "valid": False,
                "reason": f"Validation error: {e!s}",
                "validation_date": datetime.now().isoformat(),
                "validation_confidence": 0.0,
            }

    def _validate_metrics_realism(
        self, baseline_data: dict[str, Any], optimized_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate that all performance metrics fall within realistic ranges"""
        baseline_scores = baseline_data.get("scores", [])
        optimized_scores = optimized_data.get("scores", [])
        metric_type = baseline_data.get("metadata", {}).get("metric_type", "unknown")

        # Define realistic metric ranges
        realistic_ranges = {
            "response_time_ms": {
                "min": 0.1,
                "max": 500,
                "suspicious_below": 1.0,
                "suspicious_above": 300,
            },
            "memory_usage_mb": {
                "min": 10,
                "max": 1000,
                "suspicious_below": 5,
                "suspicious_above": 500,
            },
            "cpu_usage_percent": {
                "min": 0.1,
                "max": 100,
                "suspicious_below": 0.5,
                "suspicious_above": 90,
            },
            "throughput_rps": {
                "min": 1,
                "max": 10000,
                "suspicious_below": 5,
                "suspicious_above": 5000,
            },
            "error_rate_percent": {
                "min": 0,
                "max": 100,
                "suspicious_below": -0.1,
                "suspicious_above": 10,
            },
            "cache_hit_ratio": {
                "min": 0,
                "max": 100,
                "suspicious_below": 30,
                "suspicious_above": 100.1,
            },
        }

        ranges = realistic_ranges.get(
            metric_type,
            {
                "min": 0,
                "max": float("inf"),
                "suspicious_below": -1,
                "suspicious_above": float("inf"),
            },
        )

        all_scores = baseline_scores + optimized_scores

        # Check realistic bounds
        realistic = all(ranges["min"] <= score <= ranges["max"] for score in all_scores)

        # Check for suspicious values
        suspicious_low = any(score < ranges["suspicious_below"] for score in all_scores)
        suspicious_high = any(
            score > ranges["suspicious_above"] for score in all_scores
        )

        # Validate improvement direction for metric type
        baseline_mean = np.mean(baseline_scores) if baseline_scores else 0
        optimized_mean = np.mean(optimized_scores) if optimized_scores else 0

        # For metrics where lower is better
        lower_is_better = metric_type in [
            "response_time_ms",
            "memory_usage_mb",
            "cpu_usage_percent",
            "error_rate_percent",
        ]

        if lower_is_better:
            improvement_direction_correct = optimized_mean <= baseline_mean
        else:
            improvement_direction_correct = optimized_mean >= baseline_mean

        return {
            "realistic_metrics": realistic,
            "suspicious_values": suspicious_low or suspicious_high,
            "improvement_direction_correct": improvement_direction_correct,
            "min_value": min(all_scores) if all_scores else 0,
            "max_value": max(all_scores) if all_scores else 0,
            "metric_type": metric_type,
            "validation_status": "PASS"
            if realistic
            and not (suspicious_low or suspicious_high)
            and improvement_direction_correct
            else "SUSPICIOUS",
        }

    def _perform_cross_validation(
        self, baseline_scores: list[float], optimized_scores: list[float]
    ) -> dict[str, Any]:
        """Perform cross-validation for robust optimization assessment"""
        from sklearn.model_selection import KFold

        if len(baseline_scores) < 20 or len(optimized_scores) < 20:
            return {
                "robust": False,
                "reason": "Insufficient data for cross-validation",
                "consistency_score": 0.0,
            }

        baseline_array = np.array(baseline_scores)
        optimized_array = np.array(optimized_scores)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        fold_results = []

        for train_idx, test_idx in kf.split(baseline_array):
            # Use test indices for validation fold
            baseline_fold = baseline_array[test_idx]
            optimized_fold = (
                optimized_array[test_idx]
                if len(test_idx) <= len(optimized_array)
                else optimized_array[: len(test_idx)]
            )

            if len(optimized_fold) < len(test_idx):
                # Pad if necessary
                optimized_fold = np.concatenate([
                    optimized_fold,
                    np.random.choice(
                        optimized_fold, len(test_idx) - len(optimized_fold)
                    ),
                ])

            # Perform t-test on this fold
            try:
                t_stat, p_value = stats.ttest_ind(optimized_fold, baseline_fold)

                fold_result = {
                    "fold_size": len(test_idx),
                    "baseline_mean": float(np.mean(baseline_fold)),
                    "optimized_mean": float(np.mean(optimized_fold)),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }
                fold_results.append(fold_result)
            except Exception as e:
                fold_results.append({
                    "fold_size": len(test_idx),
                    "error": str(e),
                    "significant": False,
                })

        # Aggregate results
        significant_folds = sum(
            1 for result in fold_results if result.get("significant", False)
        )
        consistency_score = significant_folds / len(fold_results)

        return {
            "fold_results": fold_results,
            "significant_folds": significant_folds,
            "total_folds": len(fold_results),
            "consistency_score": consistency_score,
            "robust": consistency_score
            >= 0.6,  # At least 60% of folds should be significant
        }
