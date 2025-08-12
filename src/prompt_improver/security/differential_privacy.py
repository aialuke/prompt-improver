"""Real differential privacy service for production use and integration testing."""

import math
import random

import numpy as np


class DifferentialPrivacyService:
    """Real differential privacy service that implements secure privacy-preserving mechanisms."""

    def __init__(self, initial_epsilon: float = 1.0, initial_delta: float = 1e-06):
        self.initial_epsilon = initial_epsilon
        self.initial_delta = initial_delta
        self.epsilon = initial_epsilon
        self.delta = initial_delta
        self.privacy_spent = 0.0
        self.privacy_budget_used = 0.0
        self.query_count = 0
        self.max_queries = 1000
        self.privacy_log: list[dict[str, float | str]] = []

    def check_privacy_budget(self, requested_epsilon: float) -> bool:
        """Check if sufficient privacy budget remains for the requested operation."""
        if self.query_count >= self.max_queries:
            return False
        if self.privacy_spent + requested_epsilon > self.epsilon:
            return False
        return True

    def consume_privacy_budget(
        self, epsilon_used: float, operation: str = "query"
    ) -> bool:
        """Consume privacy budget for an operation."""
        if not self.check_privacy_budget(epsilon_used):
            return False
        self.privacy_spent += epsilon_used
        self.privacy_budget_used = self.privacy_spent / self.epsilon
        self.query_count += 1
        self.privacy_log.append({
            "operation": operation,
            "epsilon_used": epsilon_used,
            "total_spent": self.privacy_spent,
            "budget_remaining": self.epsilon - self.privacy_spent,
            "query_count": self.query_count,
        })
        return True

    def add_laplace_noise(
        self, value: float, sensitivity: float = 1.0, epsilon: float = None
    ) -> float:
        """Add Laplace noise for differential privacy."""
        if epsilon is None:
            epsilon = min(0.1, self.epsilon - self.privacy_spent)
        if not self.consume_privacy_budget(epsilon, "laplace_noise"):
            raise ValueError("Insufficient privacy budget")
        scale = sensitivity / epsilon
        rng = np.random.default_rng()
        noise = rng.laplace(0, scale)
        return float(value + noise)

    def add_gaussian_noise(
        self,
        value: float,
        sensitivity: float = 1.0,
        epsilon: float = None,
        delta: float = None,
    ) -> float:
        """Add Gaussian noise for (epsilon, delta)-differential privacy."""
        if epsilon is None:
            epsilon = min(0.1, self.epsilon - self.privacy_spent)
        if delta is None:
            delta = self.delta
        if not self.consume_privacy_budget(epsilon, "gaussian_noise"):
            raise ValueError("Insufficient privacy budget")
        sigma = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
        rng = np.random.default_rng()
        noise = rng.normal(0, sigma)
        return float(value + noise)

    def add_noise(
        self, value: float, sensitivity: float = 1.0, mechanism: str = "laplace"
    ) -> float:
        """Add noise using specified mechanism (convenience method)."""
        if mechanism == "laplace":
            return self.add_laplace_noise(value, sensitivity)
        if mechanism == "gaussian":
            return self.add_gaussian_noise(value, sensitivity)
        raise ValueError(f"Unknown mechanism: {mechanism}")

    def private_count(
        self, data: list[int | float], threshold: float = None, epsilon: float = None
    ) -> int:
        """Return private count of data points above threshold."""
        if epsilon is None:
            epsilon = min(0.1, self.epsilon - self.privacy_spent)
        if threshold is None:
            threshold = 0
        true_count = sum(1 for x in data if x >= threshold)
        noisy_count = self.add_laplace_noise(
            true_count, sensitivity=1.0, epsilon=epsilon
        )
        return max(0, int(round(noisy_count)))

    def private_sum(
        self,
        data: list[int | float],
        clipping_bound: float = 1.0,
        epsilon: float = None,
    ) -> float:
        """Return private sum with clipping for bounded sensitivity."""
        if epsilon is None:
            epsilon = min(0.1, self.epsilon - self.privacy_spent)
        clipped_data = [max(-clipping_bound, min(clipping_bound, x)) for x in data]
        true_sum = sum(clipped_data)
        return self.add_laplace_noise(
            true_sum, sensitivity=clipping_bound, epsilon=epsilon
        )

    def private_mean(
        self,
        data: list[int | float],
        clipping_bound: float = 1.0,
        epsilon: float = None,
    ) -> float:
        """Return private mean with clipping."""
        if len(data) == 0:
            return 0.0
        if epsilon is None:
            epsilon = min(0.1, self.epsilon - self.privacy_spent)
        epsilon_count = epsilon / 2
        epsilon_sum = epsilon / 2
        private_count_val = self.private_count(
            data, threshold=float("-inf"), epsilon=epsilon_count
        )
        private_sum_val = self.private_sum(data, clipping_bound, epsilon=epsilon_sum)
        if private_count_val == 0:
            return 0.0
        return private_sum_val / private_count_val

    def exponential_mechanism(
        self,
        candidates: list[any],
        utility_scores: list[float],
        sensitivity: float = 1.0,
        epsilon: float = None,
    ) -> any:
        """Select candidate using exponential mechanism."""
        if len(candidates) != len(utility_scores):
            raise ValueError("Candidates and utility scores must have same length")
        if epsilon is None:
            epsilon = min(0.1, self.epsilon - self.privacy_spent)
        if not self.consume_privacy_budget(epsilon, "exponential_mechanism"):
            raise ValueError("Insufficient privacy budget")
        max_utility = max(utility_scores)
        exp_utilities = [
            math.exp(epsilon * (score - max_utility) / (2 * sensitivity))
            for score in utility_scores
        ]
        total_weight = sum(exp_utilities)
        probabilities = [weight / total_weight for weight in exp_utilities]
        rand_val = random.random()
        cumulative_prob = 0.0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return candidates[i]
        return candidates[-1]

    def reset_privacy_budget(self, new_epsilon: float = None, new_delta: float = None):
        """Reset privacy budget (use with caution in production)."""
        if new_epsilon is not None:
            self.epsilon = new_epsilon
        else:
            self.epsilon = self.initial_epsilon
        if new_delta is not None:
            self.delta = new_delta
        else:
            self.delta = self.initial_delta
        self.privacy_spent = 0.0
        self.privacy_budget_used = 0.0
        self.query_count = 0
        self.privacy_log.clear()

    def get_privacy_spent(self) -> dict[str, float | int]:
        """Get current privacy expenditure summary."""
        return {
            "epsilon_spent": self.privacy_spent,
            "epsilon_remaining": self.epsilon - self.privacy_spent,
            "budget_used_percentage": self.privacy_budget_used * 100,
            "total_queries": self.query_count,
            "queries_remaining": self.max_queries - self.query_count,
        }

    def get_privacy_log(self) -> list[dict[str, float | str]]:
        """Get detailed log of privacy expenditures."""
        return self.privacy_log.copy()

    def compose_privacy_parameters(
        self, epsilons: list[float], deltas: list[float] = None
    ) -> dict[str, float]:
        """Compute composed privacy parameters for multiple mechanisms."""
        if deltas is None:
            total_epsilon = sum(epsilons)
            return {"epsilon": total_epsilon, "delta": 0.0}
        if len(epsilons) != len(deltas):
            raise ValueError("Epsilons and deltas must have same length")
        total_epsilon = sum(epsilons)
        total_delta = sum(deltas)
        return {"epsilon": total_epsilon, "delta": total_delta}
