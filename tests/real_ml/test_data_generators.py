"""Test Data Generators for Real ML Testing

Provides deterministic, realistic datasets for ML pipeline testing
without requiring large external datasets.
"""

import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np


class PromptDataGenerator:
    """Generates realistic prompt and rule data for testing."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

        # Realistic prompt templates
        self.prompt_templates = [
            "Write a {adjective} {content_type} about {topic}",
            "Create a {adjective} analysis of {topic}",
            "Explain {topic} in {adjective} terms",
            "Generate a {content_type} that covers {topic}",
            "Develop a {adjective} approach to {topic}",
            "Analyze the {adjective} aspects of {topic}",
            "Provide {adjective} insights on {topic}",
            "Design a {content_type} focused on {topic}"
        ]

        self.adjectives = [
            "comprehensive", "detailed", "concise", "technical", "simple",
            "advanced", "basic", "thorough", "brief", "extensive"
        ]

        self.content_types = [
            "report", "summary", "guide", "tutorial", "analysis",
            "overview", "documentation", "explanation", "description", "review"
        ]

        self.topics = [
            "machine learning", "data science", "software engineering",
            "artificial intelligence", "web development", "database design",
            "system architecture", "cloud computing", "cybersecurity",
            "user experience", "project management", "data visualization"
        ]

        # Rule types and their characteristics
        self.rule_types = [
            {
                "name": "clarity_rule",
                "description": "Improves prompt clarity and specificity",
                "base_effectiveness": 0.8,
                "variability": 0.15
            },
            {
                "name": "specificity_rule",
                "description": "Adds specific constraints and requirements",
                "base_effectiveness": 0.75,
                "variability": 0.2
            },
            {
                "name": "chain_of_thought_rule",
                "description": "Adds step-by-step reasoning structure",
                "base_effectiveness": 0.85,
                "variability": 0.1
            },
            {
                "name": "few_shot_examples_rule",
                "description": "Includes relevant examples for context",
                "base_effectiveness": 0.9,
                "variability": 0.08
            },
            {
                "name": "role_based_prompting_rule",
                "description": "Defines specific role for the AI",
                "base_effectiveness": 0.7,
                "variability": 0.25
            }
        ]

    def generate_prompts(self, count: int) -> list[str]:
        """Generate realistic prompt texts."""
        prompts = []

        for _ in range(count):
            template = random.choice(self.prompt_templates)
            adjective = random.choice(self.adjectives)
            content_type = random.choice(self.content_types)
            topic = random.choice(self.topics)

            prompt = template.format(
                adjective=adjective,
                content_type=content_type,
                topic=topic
            )
            prompts.append(prompt)

        return prompts

    def generate_rules_data(self, count: int) -> list[dict[str, Any]]:
        """Generate realistic rule configuration data."""
        rules = []

        for i in range(count):
            rule_type = random.choice(self.rule_types)

            rule = {
                "id": f"rule_{i:03d}",
                "name": rule_type["name"],
                "description": rule_type["description"],
                "enabled": random.choice([True, True, True, False]),  # 75% enabled
                "priority": random.randint(1, 10),
                "default_parameters": self._generate_rule_parameters(),
                "parameter_constraints": self._generate_parameter_constraints(),
                "category": random.choice(["structure", "clarity", "examples", "formatting"]),
                "version": f"1.{random.randint(0, 5)}"
            }

            rules.append(rule)

        return rules

    def generate_performance_data(
        self,
        rules_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate realistic performance data correlated with rule characteristics."""
        performance_data = []

        for rule in rules_data:
            # Find rule type characteristics
            rule_type = None
            for rt in self.rule_types:
                if rt["name"] == rule["name"]:
                    rule_type = rt
                    break

            if rule_type:
                base_effectiveness = rule_type["base_effectiveness"]
                variability = rule_type["variability"]
            else:
                base_effectiveness = 0.7
                variability = 0.2

            # Apply modifiers based on rule properties
            effectiveness = base_effectiveness

            # Priority affects effectiveness
            effectiveness += (rule["priority"] - 5) * 0.02

            # Enabled rules perform better in tests
            if not rule["enabled"]:
                effectiveness *= 0.7

            # Add random variation
            effectiveness += np.random.normal(0, variability)
            effectiveness = max(0.1, min(0.99, effectiveness))  # Clamp to valid range

            # Generate correlated metrics
            accuracy = effectiveness + np.random.normal(0, 0.05)
            precision = effectiveness + np.random.normal(0, 0.08)
            recall = effectiveness + np.random.normal(0, 0.06)

            performance = {
                "rule_id": rule["id"],
                "effectiveness": effectiveness,
                "accuracy": max(0.1, min(0.99, accuracy)),
                "precision": max(0.1, min(0.99, precision)),
                "recall": max(0.1, min(0.99, recall)),
                "f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.5,
                "sample_count": random.randint(50, 500),
                "training_time": random.uniform(0.5, 10.0),
                "last_updated": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
            }

            performance_data.append(performance)

        return performance_data

    def generate_training_dataset(
        self,
        size: int = 100
    ) -> tuple[list[str], list[int], list[dict[str, Any]]]:
        """Generate complete training dataset with prompts, labels, and metadata."""
        prompts = self.generate_prompts(size)

        # Generate labels based on prompt characteristics
        labels = []
        metadata = []

        for i, prompt in enumerate(prompts):
            # Simple heuristic labeling ensuring both classes are represented
            base_label = 1 if (
                len(prompt.split()) > 8 and
                any(word in prompt.lower() for word in ["detailed", "comprehensive", "thorough", "specific"])
            ) else 0

            # Ensure balanced classes by alternating some labels
            if i % 4 == 0:  # Force some variation
                effectiveness_label = 1 - base_label
            else:
                effectiveness_label = base_label

            labels.append(effectiveness_label)

            # Generate metadata
            meta = {
                "word_count": len(prompt.split()),
                "char_count": len(prompt),
                "has_specific_terms": any(word in prompt.lower() for word in ["specific", "detailed", "comprehensive"]),
                "complexity_score": len(set(prompt.lower().split())) / len(prompt.split()),  # Vocabulary diversity
                "generated_at": datetime.now().isoformat()
            }
            metadata.append(meta)

        return prompts, labels, metadata

    def generate_pattern_data(self, count: int) -> list[dict[str, Any]]:
        """Generate data suitable for pattern discovery."""
        pattern_data = []

        # Create clusters of related data points
        clusters = [
            {"center": [0.8, 0.7, 0.9], "spread": 0.1, "size": count // 3},  # High performance cluster
            {"center": [0.5, 0.6, 0.4], "spread": 0.15, "size": count // 3},  # Medium performance cluster
            {"center": [0.3, 0.4, 0.2], "spread": 0.12, "size": count - 2 * (count // 3)}  # Low performance cluster
        ]

        for cluster in clusters:
            for _ in range(cluster["size"]):
                # Generate point around cluster center
                effectiveness = max(0.1, min(0.99,
                    cluster["center"][0] + np.random.normal(0, cluster["spread"])
                ))
                accuracy = max(0.1, min(0.99,
                    cluster["center"][1] + np.random.normal(0, cluster["spread"])
                ))
                improvement = max(0.0, min(0.8,
                    cluster["center"][2] + np.random.normal(0, cluster["spread"])
                ))

                data_point = {
                    "effectiveness": effectiveness,
                    "accuracy": accuracy,
                    "improvement": improvement,
                    "rule_count": random.randint(1, 10),
                    "sample_size": random.randint(20, 200),
                    "training_epochs": random.randint(5, 50),
                    "feature_count": random.randint(5, 25),
                    "processing_time": random.uniform(0.1, 5.0),
                    "timestamp": (datetime.now() - timedelta(hours=random.randint(0, 168))).isoformat()
                }

                pattern_data.append(data_point)

        return pattern_data

    def _generate_rule_parameters(self) -> dict[str, Any]:
        """Generate realistic rule parameters."""
        parameters = {}

        # Common parameters across rule types
        if random.random() < 0.7:
            parameters["weight"] = round(random.uniform(0.1, 2.0), 2)

        if random.random() < 0.5:
            parameters["min_length"] = random.randint(5, 50)

        if random.random() < 0.4:
            parameters["max_length"] = random.randint(100, 1000)

        if random.random() < 0.3:
            parameters["threshold"] = round(random.uniform(0.1, 0.9), 2)

        if random.random() < 0.6:
            parameters["enabled"] = random.choice([True, False])

        # Rule-specific parameters
        if random.random() < 0.2:
            parameters["temperature"] = round(random.uniform(0.1, 1.0), 2)

        if random.random() < 0.3:
            parameters["max_examples"] = random.randint(1, 10)

        return parameters

    def _generate_parameter_constraints(self) -> dict[str, dict[str, Any]]:
        """Generate parameter constraints."""

        common_constraints = [
            ("weight", {"min": 0.1, "max": 2.0}),
            ("min_length", {"min": 1, "max": 100}),
            ("max_length", {"min": 50, "max": 2000}),
            ("threshold", {"min": 0.0, "max": 1.0}),
            ("temperature", {"min": 0.1, "max": 1.0}),
            ("max_examples", {"min": 1, "max": 20})
        ]

        # Randomly include some constraints
        return {param_name: constraint for param_name, constraint in common_constraints if random.random() < 0.4}


class MLTrainingDataGenerator:
    """Generates realistic ML training scenarios and datasets."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_feature_matrix(
        self,
        n_samples: int,
        n_features: int,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """Generate realistic feature matrix with controlled characteristics."""

        # Create base features with different distributions
        features = np.zeros((n_samples, n_features))

        for i in range(n_features):
            if i % 4 == 0:
                # Normal distribution features
                features[:, i] = np.random.normal(0, 1, n_samples)
            elif i % 4 == 1:
                # Uniform distribution features
                features[:, i] = np.random.uniform(-2, 2, n_samples)
            elif i % 4 == 2:
                # Exponential distribution features
                features[:, i] = np.random.exponential(1, n_samples)
            else:
                # Binary features
                features[:, i] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, features.shape)
            features += noise

        return features

    def generate_regression_targets(
        self,
        features: np.ndarray,
        target_type: str = "linear"
    ) -> np.ndarray:
        """Generate realistic regression targets based on features."""
        n_samples, n_features = features.shape

        if target_type == "linear":
            # Linear combination with some nonlinearity
            weights = np.random.normal(0, 1, n_features)
            targets = features @ weights

            # Add quadratic term for some complexity
            if n_features > 1:
                targets += 0.1 * features[:, 0] * features[:, 1]

        elif target_type == "nonlinear":
            # More complex nonlinear relationship
            weights = np.random.normal(0, 0.5, n_features)
            targets = features @ weights

            # Add sine and polynomial terms
            targets += 0.2 * np.sin(features[:, 0]) if n_features > 0 else 0
            targets += 0.1 * np.square(features[:, min(1, n_features - 1)])

        else:  # random
            targets = np.random.normal(0, 1, n_samples)

        # Add noise
        targets += np.random.normal(0, 0.1, n_samples)

        # Scale to [0, 1] range for effectiveness scores
        return (targets - targets.min()) / (targets.max() - targets.min())

    def generate_classification_targets(
        self,
        features: np.ndarray,
        n_classes: int = 2
    ) -> np.ndarray:
        """Generate realistic classification targets."""
        n_samples = features.shape[0]

        # Create decision boundary based on feature combinations
        if features.shape[1] >= 2:
            decision_score = (
                0.5 * features[:, 0] +
                0.3 * features[:, 1] +
                0.2 * np.sum(features[:, 2:], axis=1) if features.shape[1] > 2 else 0
            )
        else:
            decision_score = features[:, 0] if features.shape[1] > 0 else np.random.normal(0, 1, n_samples)

        # Convert to probabilities and sample classes
        if n_classes == 2:
            probabilities = 1 / (1 + np.exp(-decision_score))  # Sigmoid
            targets = (probabilities > 0.5).astype(int)
        else:
            # Multiclass using softmax-like approach
            scores = np.column_stack([
                decision_score + np.random.normal(0, 0.5, n_samples)
                for _ in range(n_classes)
            ])
            targets = np.argmax(scores, axis=1)

        return targets

    def generate_time_series_data(
        self,
        length: int,
        n_features: int = 1,
        trend: bool = True,
        seasonality: bool = True,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """Generate realistic time series data for temporal ML testing."""
        time_points = np.arange(length)
        data = np.zeros((length, n_features))

        for feature_idx in range(n_features):
            series = np.zeros(length)

            # Base level
            series += 50

            # Trend component
            if trend:
                trend_strength = np.random.uniform(-0.1, 0.1)
                series += trend_strength * time_points

            # Seasonal component
            if seasonality:
                seasonal_period = random.randint(7, 30)  # Weekly to monthly patterns
                seasonal_amplitude = np.random.uniform(5, 15)
                series += seasonal_amplitude * np.sin(2 * np.pi * time_points / seasonal_period)

                # Add secondary seasonality
                secondary_period = random.randint(3, 7)
                secondary_amplitude = np.random.uniform(2, 8)
                series += secondary_amplitude * np.cos(2 * np.pi * time_points / secondary_period)

            # Random walk component
            random_walk = np.cumsum(np.random.normal(0, 0.5, length))
            series += 0.3 * random_walk

            # Noise
            if noise_level > 0:
                series += np.random.normal(0, noise_level * np.std(series), length)

            data[:, feature_idx] = series

        return data
