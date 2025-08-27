"""
Training data generation utilities for ML testing.

This module contains training data generation utilities extracted from conftest.py
to maintain clean architecture and separation of concerns.
"""
from typing import Any

import numpy as np

from prompt_improver.utils.datetime_utils import aware_utc_now


class TrainingDataGenerator:
    """Generate synthetic training data for ML testing."""

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def generate_rule_performance_data(
        self,
        n_samples: int = 100,
        n_rules: int = 5,
        effectiveness_distribution: str = "normal",
    ) -> dict[str, Any]:
        """Generate rule performance training data."""
        rule_ids = [f"rule_{i}" for i in range(n_rules)]
        features = []
        effectiveness_scores = []
        for _ in range(n_samples):
            clarity_score = self.rng.beta(2, 1)
            length = self.rng.lognormal(4, 0.5)
            complexity = self.rng.uniform(1, 10)
            user_rating = self.rng.normal(7, 1.5)
            user_rating = np.clip(user_rating, 1, 10)
            context_match = self.rng.beta(3, 2)
            features.append([
                clarity_score,
                length,
                complexity,
                user_rating,
                context_match,
            ])
            if effectiveness_distribution == "normal":
                base_effectiveness = (
                    0.4
                    + 0.3 * clarity_score
                    + 0.2 * context_match
                    - 0.1 * (complexity / 10)
                )
                effectiveness = self.rng.normal(base_effectiveness, 0.1)
            elif effectiveness_distribution == "bimodal":
                if self.rng.random() < 0.6:
                    effectiveness = self.rng.normal(0.7, 0.1)
                else:
                    effectiveness = self.rng.normal(0.4, 0.1)
            else:
                effectiveness = self.rng.uniform(0.2, 0.9)
            effectiveness = np.clip(effectiveness, 0.0, 1.0)
            effectiveness_scores.append(effectiveness)
        return {
            "features": features,
            "effectiveness_scores": effectiveness_scores,
            "rule_ids": rule_ids,
            "feature_names": [
                "clarity_score",
                "length",
                "complexity",
                "user_rating",
                "context_match",
            ],
            "metadata": {
                "n_samples": n_samples,
                "n_rules": n_rules,
                "distribution": effectiveness_distribution,
                "generated_at": aware_utc_now().isoformat(),
            },
        }

    def generate_ab_test_data(
        self,
        control_samples: int = 150,
        treatment_samples: int = 150,
        effect_size: float = 0.1,
    ) -> dict[str, Any]:
        """Generate A/B test data with specified effect size."""
        control_scores = self.rng.normal(0.65, 0.15, control_samples)
        control_scores = np.clip(control_scores, 0.0, 1.0)
        treatment_scores = self.rng.normal(
            0.65 + effect_size, 0.15, treatment_samples
        )
        treatment_scores = np.clip(treatment_scores, 0.0, 1.0)

        return {
            "control_group": {
                "scores": control_scores.tolist(),
                "n_samples": control_samples,
                "mean_score": float(np.mean(control_scores)),
                "std_score": float(np.std(control_scores)),
            },
            "treatment_group": {
                "scores": treatment_scores.tolist(),
                "n_samples": treatment_samples,
                "mean_score": float(np.mean(treatment_scores)),
                "std_score": float(np.std(treatment_scores)),
            },
            "effect_size": effect_size,
            "metadata": {
                "control_samples": control_samples,
                "treatment_samples": treatment_samples,
                "generated_at": aware_utc_now().isoformat(),
            },
        }
