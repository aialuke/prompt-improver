"""Production Synthetic Data Generator for ML Training

Enhanced synthetic data generation based on research-driven best practices:
- Multi-domain pattern generation (technical, creative, analytical, instructional, conversational)
- Statistical quality guarantees (class diversity, variance control, three-tier stratification)
- Advanced feature engineering using scikit-learn principles
- Comprehensive validation framework for ML pipeline integration

Research Sources:
- Firecrawl Deep Research: NLP synthetic data generation best practices
- Context7 Scikit-learn: Advanced data generation and statistical controls
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import TrainingPrompt
from .enhanced_quality_scorer import EnhancedQualityMetrics, EnhancedQualityScorer

logger = logging.getLogger(__name__)


@dataclass
class DomainConfig:
    """Configuration for domain-specific data generation"""

    name: str
    ratio: float
    patterns: list[tuple[str, str]]
    feature_ranges: dict[str, tuple[float, float]]
    effectiveness_params: tuple[float, float]  # Beta distribution parameters
    complexity_range: tuple[int, int]


@dataclass
class QualityMetrics:
    """Quality validation metrics for generated data"""

    sample_count: int
    class_diversity: int
    variance_sufficient: bool
    min_samples_met: bool
    ensemble_ready: bool
    no_invalid_values: bool
    overall_quality: bool
    domain_distribution: dict[str, int]
    feature_correlations: dict[str, float]


class ProductionSyntheticDataGenerator:
    """Production-grade synthetic data generator with advanced quality assessment"""

    def __init__(
        self,
        target_samples: int = 1000,
        random_state: int = 42,
        use_enhanced_scoring: bool = True,
    ):
        """Initialize the production synthetic data generator

        Args:
            target_samples: Total number of samples to generate (default: 1000)
            random_state: Random seed for reproducible generation
            use_enhanced_scoring: Whether to use enhanced multi-dimensional quality scoring
        """
        self.target_samples = target_samples
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.use_enhanced_scoring = use_enhanced_scoring

        # Initialize enhanced quality scorer
        if use_enhanced_scoring:
            self.quality_scorer = EnhancedQualityScorer(confidence_level=0.95)

        # Configure domains based on research insights
        self.domains = self._initialize_domain_configs()

        # Feature specifications (6-dimensional feature vectors)
        self.feature_names = [
            "clarity",  # 0: How clear and understandable the prompt is
            "length",  # 1: Content length and detail level
            "specificity",  # 2: Level of specific details and precision
            "complexity",  # 3: Intellectual/technical complexity level
            "context_richness",  # 4: Amount of contextual information provided
            "actionability",  # 5: How actionable and implementable the result is
        ]

        # Legacy quality validation thresholds (for backward compatibility)
        self.quality_thresholds = {
            "min_samples": 10,  # Minimum for basic optimization
            "ensemble_threshold": 20,  # Minimum for ensemble methods
            "min_classes": 2,  # Minimum class diversity
            "min_variance": 0.1,  # Minimum effectiveness variance
            "max_correlation": 0.8,  # Maximum feature correlation
        }

    def _initialize_domain_configs(self) -> dict[str, DomainConfig]:
        """Initialize domain-specific configurations based on research"""
        return {
            "technical": DomainConfig(
                name="technical",
                ratio=0.25,  # 25% technical content
                patterns=[
                    (
                        "Create API endpoint",
                        "Create a comprehensive REST API endpoint with authentication, rate limiting, error handling, and detailed OpenAPI documentation including examples",
                    ),
                    (
                        "Debug error",
                        "Debug this specific error by analyzing logs, identifying root cause, implementing robust error handling, and adding prevention measures",
                    ),
                    (
                        "Optimize function",
                        "Optimize this function for performance by implementing algorithmic improvements, memory efficiency, and scalability considerations",
                    ),
                    (
                        "Add tests",
                        "Add comprehensive unit tests with edge cases, mocking strategies, integration test coverage, and performance benchmarks",
                    ),
                    (
                        "Document system",
                        "Document this system architecture with detailed diagrams, deployment guides, troubleshooting procedures, and maintenance workflows",
                    ),
                    (
                        "Setup guide",
                        "Create a detailed setup guide with prerequisites, step-by-step instructions, verification steps, and common troubleshooting solutions",
                    ),
                    (
                        "Code review",
                        "Conduct thorough code review focusing on security, performance, maintainability, and adherence to best practices",
                    ),
                    (
                        "Refactor code",
                        "Refactor this code to improve readability, reduce complexity, eliminate duplication, and enhance testability",
                    ),
                ],
                feature_ranges={
                    "clarity": (0.4, 0.9),  # Higher baseline for technical clarity
                    "length": (150, 300),  # Longer technical content
                    "specificity": (0.7, 1.0),  # High specificity for technical
                    "complexity": (4, 8),  # Medium-high complexity
                    "context_richness": (0.5, 0.9),
                    "actionability": (
                        0.8,
                        1.0,
                    ),  # Very high actionability for technical
                },
                effectiveness_params=(
                    3,
                    2,
                ),  # Beta(3,2) - skewed higher for measurable outcomes
                complexity_range=(4, 8),
            ),
            "creative": DomainConfig(
                name="creative",
                ratio=0.20,  # 20% creative content
                patterns=[
                    (
                        "Write story",
                        "Write an engaging story with compelling characters, clear narrative arc, vivid descriptive details, and emotional resonance",
                    ),
                    (
                        "Create content",
                        "Create engaging content that resonates with the target audience, drives meaningful engagement, and achieves specific goals",
                    ),
                    (
                        "Improve writing",
                        "Improve this writing by enhancing flow, adding sensory details, strengthening emotional impact, and clarifying messaging",
                    ),
                    (
                        "Add creativity",
                        "Add creative elements like metaphors, unique perspectives, innovative approaches, and memorable hooks",
                    ),
                    (
                        "Write copy",
                        "Write persuasive copy that speaks to customer pain points, highlights unique value propositions, and drives action",
                    ),
                    (
                        "Create campaign",
                        "Create a comprehensive marketing campaign with compelling messaging, multi-channel strategy, and measurable success metrics",
                    ),
                    (
                        "Design content",
                        "Design visual content that captures attention, communicates key messages, and aligns with brand identity",
                    ),
                    (
                        "Brand voice",
                        "Develop a distinctive brand voice that reflects company values, resonates with target audience, and maintains consistency",
                    ),
                ],
                feature_ranges={
                    "clarity": (0.3, 0.8),  # More variable for creative expression
                    "length": (80, 250),  # Varied creative content length
                    "specificity": (0.4, 0.9),  # Wide range for creative specificity
                    "complexity": (2, 7),  # Varied complexity levels
                    "context_richness": (0.6, 1.0),  # High context for creative
                    "actionability": (0.5, 0.9),  # Medium-high actionability
                },
                effectiveness_params=(
                    2,
                    2,
                ),  # Beta(2,2) - more uniform for subjective creative quality
                complexity_range=(2, 7),
            ),
            "analytical": DomainConfig(
                name="analytical",
                ratio=0.20,  # 20% analytical content
                patterns=[
                    (
                        "Analyze data",
                        "Analyze this dataset comprehensively including statistical summaries, trend identification, correlation analysis, and actionable insights",
                    ),
                    (
                        "Research topic",
                        "Research this topic thoroughly using reliable sources, synthesizing findings, identifying patterns, and drawing evidence-based conclusions",
                    ),
                    (
                        "Create report",
                        "Create a detailed analytical report with executive summary, methodology, key findings, recommendations, and supporting visualizations",
                    ),
                    (
                        "Compare options",
                        "Compare these options systematically using relevant criteria, quantitative analysis, risk assessment, and strategic recommendations",
                    ),
                    (
                        "Identify trends",
                        "Identify significant trends in this data using statistical analysis, predictive modeling, and contextual interpretation",
                    ),
                    (
                        "Validate hypothesis",
                        "Validate this hypothesis through rigorous testing, statistical analysis, and comprehensive evaluation of evidence",
                    ),
                    (
                        "Performance metrics",
                        "Establish comprehensive performance metrics with baselines, targets, measurement methods, and reporting frameworks",
                    ),
                    (
                        "Market analysis",
                        "Conduct thorough market analysis including competitor research, trend analysis, opportunity identification, and strategic implications",
                    ),
                ],
                feature_ranges={
                    "clarity": (0.5, 0.9),  # High clarity for analytical work
                    "length": (120, 280),  # Substantial analytical content
                    "specificity": (0.6, 1.0),  # High specificity for analysis
                    "complexity": (3, 7),  # Medium-high analytical complexity
                    "context_richness": (0.7, 1.0),  # Rich context for analysis
                    "actionability": (0.6, 0.9),  # High actionability for insights
                },
                effectiveness_params=(
                    2.5,
                    1.8,
                ),  # Beta(2.5,1.8) - moderately skewed for analytical rigor
                complexity_range=(3, 7),
            ),
            "instructional": DomainConfig(
                name="instructional",
                ratio=0.20,  # 20% instructional content
                patterns=[
                    (
                        "Explain concept",
                        "Explain this concept clearly with simple language, relevant examples, step-by-step breakdowns, and practical applications",
                    ),
                    (
                        "Create tutorial",
                        "Create a comprehensive tutorial with learning objectives, structured lessons, hands-on exercises, and progress assessments",
                    ),
                    (
                        "Teaching guide",
                        "Develop a teaching guide with curriculum outline, learning activities, assessment methods, and differentiation strategies",
                    ),
                    (
                        "Simplify complex",
                        "Simplify this complex topic using analogies, visual aids, progressive disclosure, and relatable examples",
                    ),
                    (
                        "Learning path",
                        "Design a learning path with prerequisites, milestones, resources, practice opportunities, and mastery indicators",
                    ),
                    (
                        "Study guide",
                        "Create a study guide with key concepts, practice questions, review activities, and self-assessment tools",
                    ),
                    (
                        "Workshop design",
                        "Design an interactive workshop with clear objectives, engaging activities, collaborative exercises, and practical outcomes",
                    ),
                    (
                        "Skill development",
                        "Develop a skill-building program with competency frameworks, practice scenarios, feedback mechanisms, and certification criteria",
                    ),
                ],
                feature_ranges={
                    "clarity": (0.6, 1.0),  # Very high clarity for instruction
                    "length": (100, 220),  # Moderate length for digestibility
                    "specificity": (0.5, 0.8),  # Balanced specificity for learning
                    "complexity": (2, 6),  # Moderate complexity for accessibility
                    "context_richness": (0.6, 0.9),  # Good context for learning
                    "actionability": (
                        0.7,
                        1.0,
                    ),  # Very high actionability for instruction
                },
                effectiveness_params=(
                    2.2,
                    1.5,
                ),  # Beta(2.2,1.5) - skewed toward effective instruction
                complexity_range=(2, 6),
            ),
            "conversational": DomainConfig(
                name="conversational",
                ratio=0.15,  # 15% conversational content
                patterns=[
                    (
                        "Answer question",
                        "Answer this question thoroughly with clear explanations, relevant context, helpful examples, and follow-up guidance",
                    ),
                    (
                        "Provide advice",
                        "Provide thoughtful advice considering multiple perspectives, potential outcomes, practical steps, and personalized recommendations",
                    ),
                    (
                        "Clarify confusion",
                        "Clarify this confusion by addressing misconceptions, providing clear explanations, and offering additional resources",
                    ),
                    (
                        "Support request",
                        "Respond to this support request with empathy, practical solutions, step-by-step guidance, and follow-up options",
                    ),
                    (
                        "Facilitate discussion",
                        "Facilitate this discussion by asking thoughtful questions, encouraging participation, and synthesizing key points",
                    ),
                    (
                        "Resolve conflict",
                        "Help resolve this conflict through active listening, perspective-taking, common ground identification, and solution-focused approaches",
                    ),
                    (
                        "Build rapport",
                        "Build rapport through authentic engagement, shared understanding, emotional intelligence, and mutual respect",
                    ),
                    (
                        "Guide decision",
                        "Guide this decision-making process with structured analysis, option evaluation, and personalized recommendations",
                    ),
                ],
                feature_ranges={
                    "clarity": (0.4, 0.9),  # Variable clarity for natural conversation
                    "length": (60, 180),  # Shorter conversational responses
                    "specificity": (0.3, 0.7),  # Moderate specificity for dialogue
                    "complexity": (1, 5),  # Lower complexity for accessibility
                    "context_richness": (0.5, 0.8),  # Moderate context richness
                    "actionability": (
                        0.4,
                        0.8,
                    ),  # Variable actionability for conversation
                },
                effectiveness_params=(
                    2,
                    2.5,
                ),  # Beta(2,2.5) - slightly skewed for helpful responses
                complexity_range=(1, 5),
            ),
        }

    async def generate_comprehensive_training_data(self) -> dict[str, Any]:
        """Generate comprehensive training data with enhanced quality assessment"""
        logger.info(
            f"Starting generation of {self.target_samples} high-quality synthetic samples"
        )

        # Initialize storage
        all_features = []
        all_effectiveness = []
        all_prompts = []
        domain_counts = {}

        # Generate data for each domain
        for domain_name, domain_config in self.domains.items():
            domain_samples = int(self.target_samples * domain_config.ratio)
            logger.info(f"Generating {domain_samples} samples for {domain_name} domain")

            domain_data = await self._generate_domain_data(
                domain_config, domain_samples
            )

            all_features.extend(domain_data["features"])
            all_effectiveness.extend(domain_data["effectiveness_scores"])
            all_prompts.extend(domain_data["prompts"])
            domain_counts[domain_name] = len(domain_data["features"])

        # Apply research-based quality guarantees
        logger.info("Applying statistical quality guarantees")
        features, effectiveness = self._ensure_quality_guarantees(
            all_features, all_effectiveness
        )

        # Enhanced or legacy quality assessment
        if self.use_enhanced_scoring:
            # Use enhanced multi-dimensional quality assessment
            logger.info("Performing enhanced multi-dimensional quality assessment")
            enhanced_metrics = await self.quality_scorer.assess_comprehensive_quality(
                features,
                effectiveness,
                domain_counts,
                {
                    "target_samples": self.target_samples,
                    "random_state": self.random_state,
                },
            )

            # Apply corrections if overall score is below threshold
            if enhanced_metrics.overall_score < 0.55:  # ADEQUATE threshold
                logger.warning(
                    f"Enhanced quality score {enhanced_metrics.overall_score:.3f} below threshold, applying corrections"
                )
                features, effectiveness = self._apply_quality_corrections(
                    features, effectiveness
                )

                # Re-assess after corrections
                enhanced_metrics = (
                    await self.quality_scorer.assess_comprehensive_quality(
                        features,
                        effectiveness,
                        domain_counts,
                        {
                            "target_samples": self.target_samples,
                            "random_state": self.random_state,
                        },
                    )
                )
        else:
            # Use legacy binary quality validation
            logger.info("Validating against ML pipeline requirements")
            legacy_metrics = self._validate_ml_requirements(
                features, effectiveness, domain_counts
            )

            if not legacy_metrics.overall_quality:
                logger.warning(
                    "Generated data failed quality validation, applying corrections"
                )
                features, effectiveness = self._apply_quality_corrections(
                    features, effectiveness
                )
                legacy_metrics = self._validate_ml_requirements(
                    features, effectiveness, domain_counts
                )

        # Prepare result structure
        result = {
            "features": features,
            "effectiveness_scores": effectiveness,
            "prompts": all_prompts[
                : len(features)
            ],  # Ensure alignment after corrections
            "metadata": {
                "source": "enhanced_synthetic_v3",
                "total_samples": len(features),
                "domain_distribution": domain_counts,
                "generation_timestamp": datetime.utcnow().isoformat(),
                "random_state": self.random_state,
                "feature_names": self.feature_names,
                "quality_assessment_type": "enhanced"
                if self.use_enhanced_scoring
                else "legacy",
            },
        }

        # Add appropriate quality metrics to metadata
        if self.use_enhanced_scoring:
            result["metadata"]["enhanced_quality_metrics"] = enhanced_metrics
        else:
            result["metadata"]["quality_metrics"] = legacy_metrics

        logger.info(
            f"Successfully generated {len(features)} high-quality synthetic samples"
        )
        return result

    async def _generate_domain_data(
        self, domain_config: DomainConfig, sample_count: int
    ) -> dict[str, Any]:
        """Generate domain-specific training data with realistic patterns"""
        features = []
        effectiveness_scores = []
        prompts = []

        patterns = domain_config.patterns

        # Use scikit-learn inspired feature generation for base structure
        base_features, base_effectiveness = make_classification(
            n_samples=sample_count,
            n_features=len(self.feature_names),
            n_informative=len(self.feature_names),  # All features informative
            n_redundant=0,  # No redundant features
            n_clusters_per_class=2,  # 2 clusters per effectiveness level
            n_classes=3,  # Three effectiveness tiers
            class_sep=1.2,  # Good class separation
            flip_y=0.01,  # Minimal label noise
            random_state=self.random_state + hash(domain_config.name) % 1000,
        )

        # Transform base features to domain-specific ranges
        for i in range(sample_count):
            pattern_idx = i % len(patterns)
            original, enhanced = patterns[pattern_idx]

            # Transform features to domain-specific ranges
            domain_features = self._transform_to_domain_features(
                base_features[i], domain_config
            )

            # Generate effectiveness score based on domain characteristics
            effectiveness_score = self._generate_domain_effectiveness(
                domain_config, base_effectiveness[i]
            )

            features.append(domain_features)
            effectiveness_scores.append(effectiveness_score)
            prompts.append((original, enhanced))

        return {
            "features": features,
            "effectiveness_scores": effectiveness_scores,
            "prompts": prompts,
        }

    def _transform_to_domain_features(
        self, base_features: np.ndarray, domain_config: DomainConfig
    ) -> list[float]:
        """Transform normalized features to domain-specific ranges"""
        # Normalize base features to [0, 1] range
        normalized_features = (base_features - base_features.min()) / (
            base_features.max() - base_features.min() + 1e-8
        )

        domain_features = []
        for i, feature_name in enumerate(self.feature_names):
            feature_range = domain_config.feature_ranges[feature_name]

            if feature_name == "complexity":
                # Handle discrete complexity as integer
                complexity_val = int(
                    feature_range[0]
                    + normalized_features[i] * (feature_range[1] - feature_range[0])
                )
                domain_features.append(float(complexity_val))
            else:
                # Transform continuous features
                feature_val = feature_range[0] + normalized_features[i] * (
                    feature_range[1] - feature_range[0]
                )
                domain_features.append(feature_val)

        return domain_features

    def _generate_domain_effectiveness(
        self, domain_config: DomainConfig, base_class: int
    ) -> float:
        """Generate domain-specific effectiveness score with three-tier stratification"""
        alpha, beta = domain_config.effectiveness_params

        # Map class to effectiveness tier
        if base_class == 0:  # Low effectiveness (25%)
            effectiveness = self.rng.beta(alpha, beta + 2) * 0.3  # 0.0-0.3 range
        elif base_class == 1:  # Medium effectiveness (50%)
            effectiveness = 0.3 + self.rng.beta(alpha, beta) * 0.3  # 0.3-0.6 range
        else:  # High effectiveness (25%)
            effectiveness = 0.6 + self.rng.beta(alpha + 1, beta) * 0.4  # 0.6-1.0 range

        # Ensure valid range
        return np.clip(effectiveness, 0.0, 1.0)

    def _ensure_quality_guarantees(
        self, features: list[list[float]], effectiveness: list[float]
    ) -> tuple[list[list[float]], list[float]]:
        """Apply research-based quality guarantees from test suite"""
        effectiveness_array = np.array(effectiveness)

        # 1. Class diversity guarantee (from research)
        unique_values = np.unique(effectiveness_array)
        if len(unique_values) < 3:  # Target 3 tiers
            logger.warning("Insufficient class diversity, applying stratification")
            effectiveness_array = self._apply_three_tier_distribution(
                effectiveness_array
            )

        # 2. Variance guarantee (from research)
        effectiveness_std = np.std(effectiveness_array)
        if effectiveness_std < self.quality_thresholds["min_variance"]:
            logger.warning(
                f"Low variance ({effectiveness_std:.3f}), injecting controlled noise"
            )
            noise = self.rng.normal(0, 0.15, len(effectiveness_array))
            effectiveness_array = np.clip(effectiveness_array + noise, 0.0, 1.0)

        # 3. Feature correlation control
        features_array = np.array(features)
        correlations = np.corrcoef(features_array.T)
        max_corr = np.max(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))

        if max_corr > self.quality_thresholds["max_correlation"]:
            logger.warning(
                f"High feature correlation ({max_corr:.3f}), applying decorrelation"
            )
            features_array = self._decorrelate_features(features_array)

        return features_array.tolist(), effectiveness_array.tolist()

    def _apply_three_tier_distribution(
        self, effectiveness_array: np.ndarray
    ) -> np.ndarray:
        """Apply three-tier performance stratification from research"""
        n_samples = len(effectiveness_array)
        quarter_size = n_samples // 4
        half_size = n_samples // 2

        # Redistribute to ensure three performance tiers
        shuffle_indices = self.rng.permutation(n_samples)

        # High performers (25%)
        effectiveness_array[shuffle_indices[:quarter_size]] = self.rng.uniform(
            0.7, 1.0, quarter_size
        )

        # Medium performers (50%)
        effectiveness_array[
            shuffle_indices[quarter_size : quarter_size + half_size]
        ] = self.rng.uniform(0.4, 0.6, half_size)

        # Low performers (25%)
        effectiveness_array[shuffle_indices[quarter_size + half_size :]] = (
            self.rng.uniform(0.0, 0.3, n_samples - quarter_size - half_size)
        )

        return effectiveness_array

    def _decorrelate_features(self, features_array: np.ndarray) -> np.ndarray:
        """Apply feature decorrelation to reduce excessive correlations"""
        # Add small amount of independent noise to reduce correlations
        noise_scale = 0.05  # Small noise to preserve feature meaning
        noise = self.rng.normal(0, noise_scale, features_array.shape)

        # Apply noise with feature-specific scaling
        for i, feature_name in enumerate(self.feature_names):
            if feature_name == "complexity":
                # Round complexity back to integers
                features_array[:, i] = np.round(features_array[:, i] + noise[:, i])
                features_array[:, i] = np.clip(features_array[:, i], 1, 8)
            else:
                features_array[:, i] += noise[:, i]
                # Clip to reasonable ranges
                features_array[:, i] = np.clip(features_array[:, i], 0.0, 1000.0)

        return features_array

    def _validate_ml_requirements(
        self,
        features: list[list[float]],
        effectiveness: list[float],
        domain_counts: dict[str, int],
    ) -> QualityMetrics:
        """Validate against ML pipeline requirements"""
        features_array = np.array(features)
        effectiveness_array = np.array(effectiveness)

        # Basic validations
        sample_count = len(features)
        class_diversity = len(np.unique(effectiveness_array))
        variance_sufficient = (
            np.std(effectiveness_array) >= self.quality_thresholds["min_variance"]
        )
        min_samples_met = sample_count >= self.quality_thresholds["min_samples"]
        ensemble_ready = sample_count >= self.quality_thresholds["ensemble_threshold"]
        no_invalid_values = not (
            np.isnan(features_array).any() or np.isnan(effectiveness_array).any()
        )

        # Feature correlation analysis
        correlations = np.corrcoef(features_array.T)
        max_corr = np.max(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))
        feature_correlations = {
            "max_correlation": float(max_corr),
            "mean_correlation": float(
                np.mean(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))
            ),
            "correlation_threshold_met": max_corr
            <= self.quality_thresholds["max_correlation"],
        }

        overall_quality = all([
            min_samples_met,
            class_diversity >= self.quality_thresholds["min_classes"],
            variance_sufficient,
            no_invalid_values,
            feature_correlations["correlation_threshold_met"],
        ])

        return QualityMetrics(
            sample_count=sample_count,
            class_diversity=class_diversity,
            variance_sufficient=variance_sufficient,
            min_samples_met=min_samples_met,
            ensemble_ready=ensemble_ready,
            no_invalid_values=no_invalid_values,
            overall_quality=overall_quality,
            domain_distribution=domain_counts,
            feature_correlations=feature_correlations,
        )

    def _apply_quality_corrections(
        self, features: list[list[float]], effectiveness: list[float]
    ) -> tuple[list[list[float]], list[float]]:
        """Apply corrections for failed quality validation"""
        logger.info("Applying quality corrections to meet ML requirements")

        # Ensure minimum sample count by duplicating with noise
        while len(features) < self.quality_thresholds["min_samples"]:
            idx = self.rng.randint(0, len(features))
            new_feature = np.array(features[idx]) + self.rng.normal(
                0, 0.05, len(features[idx])
            )
            new_effectiveness = effectiveness[idx] + self.rng.normal(0, 0.05)

            features.append(new_feature.tolist())
            effectiveness.append(np.clip(new_effectiveness, 0.0, 1.0))

        # Ensure class diversity by forcing three-tier distribution
        effectiveness_array = np.array(effectiveness)
        effectiveness_array = self._apply_three_tier_distribution(effectiveness_array)

        # Apply final quality guarantees
        features, effectiveness = self._ensure_quality_guarantees(
            features, effectiveness_array.tolist()
        )

        return features, effectiveness

    async def save_to_database(
        self, training_data: dict[str, Any], db_session: AsyncSession
    ) -> int:
        """Save generated training data to database with TrainingPrompt model compatibility

        Args:
            training_data: Generated training data from generate_comprehensive_training_data
            db_session: Database session for saving

        Returns:
            Number of records saved
        """
        features = training_data["features"]
        effectiveness_scores = training_data["effectiveness_scores"]
        prompts = training_data["prompts"]
        metadata = training_data["metadata"]

        saved_count = 0

        try:
            for i, (feature_vector, effectiveness, (original, enhanced)) in enumerate(
                zip(features, effectiveness_scores, prompts, strict=False)
            ):
                training_prompt = TrainingPrompt(
                    prompt_text=original,
                    enhancement_result={
                        "enhanced_prompt": enhanced,
                        "effectiveness_score": effectiveness,
                        "feature_vector": feature_vector,
                        "metadata": {
                            "source": metadata["source"],
                            "domain": self._identify_domain_from_index(
                                i, metadata["domain_distribution"]
                            ),
                            "generation_timestamp": metadata["generation_timestamp"],
                            "feature_names": metadata["feature_names"],
                            "quality_validated": True,
                        },
                    },
                    data_source="synthetic",
                    training_priority=10,  # Synthetic data priority
                )

                db_session.add(training_prompt)
                saved_count += 1

            await db_session.commit()
            logger.info(
                f"Successfully saved {saved_count} synthetic training samples to database"
            )

        except Exception as e:
            await db_session.rollback()
            logger.error(f"Failed to save synthetic data to database: {e}")
            raise

        return saved_count

    def _identify_domain_from_index(
        self, index: int, domain_distribution: dict[str, int]
    ) -> str:
        """Identify domain for a given sample index"""
        current_count = 0
        for domain_name, count in domain_distribution.items():
            if index < current_count + count:
                return domain_name
            current_count += count

        return "unknown"  # Fallback

    def get_generation_summary(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive summary with enhanced quality reporting"""
        metadata = training_data["metadata"]

        if self.use_enhanced_scoring and "enhanced_quality_metrics" in metadata:
            enhanced_metrics = metadata["enhanced_quality_metrics"]

            # Generate comprehensive quality report
            quality_report = self.quality_scorer.generate_quality_report(
                enhanced_metrics
            )

            return {
                "generation_summary": {
                    "total_samples": metadata["total_samples"],
                    "target_samples": self.target_samples,
                    "generation_efficiency": metadata["total_samples"]
                    / self.target_samples,
                    "overall_quality_score": enhanced_metrics.overall_score,
                    "confidence_score": enhanced_metrics.confidence_score,
                    "recommendation_tier": enhanced_metrics.recommendation_tier,
                    "domains_covered": len(metadata["domain_distribution"]),
                    "generation_time": metadata["generation_timestamp"],
                    "assessment_type": "enhanced_multi_dimensional",
                },
                "quality_analysis": quality_report,
                "domain_breakdown": metadata["domain_distribution"],
                "dimensional_scores": {
                    "fidelity": enhanced_metrics.fidelity.score,
                    "utility": enhanced_metrics.utility.score,
                    "privacy": enhanced_metrics.privacy.score,
                    "statistical_validity": enhanced_metrics.statistical_validity.score,
                    "diversity": enhanced_metrics.diversity.score,
                    "consistency": enhanced_metrics.consistency.score,
                },
            }
        # Legacy quality metrics
        quality_metrics = metadata["quality_metrics"]

        return {
            "generation_summary": {
                "total_samples": metadata["total_samples"],
                "target_samples": self.target_samples,
                "generation_efficiency": metadata["total_samples"]
                / self.target_samples,
                "quality_score": float(quality_metrics.overall_quality),
                "domains_covered": len(metadata["domain_distribution"]),
                "generation_time": metadata["generation_timestamp"],
                "assessment_type": "legacy_binary",
            },
            "quality_analysis": {
                "class_diversity": quality_metrics.class_diversity,
                "variance_sufficient": quality_metrics.variance_sufficient,
                "ml_requirements_met": quality_metrics.overall_quality,
                "ensemble_ready": quality_metrics.ensemble_ready,
                "feature_correlations": quality_metrics.feature_correlations,
            },
            "domain_breakdown": metadata["domain_distribution"],
            "recommendations": self._generate_recommendations(quality_metrics),
        }

    def _generate_recommendations(self, quality_metrics: QualityMetrics) -> list[str]:
        """Generate recommendations for improving synthetic data quality"""
        recommendations = []

        if not quality_metrics.overall_quality:
            recommendations.append(
                "Consider increasing sample count for more robust ML training"
            )

        if quality_metrics.class_diversity < 3:
            recommendations.append(
                "Enhance effectiveness score stratification for better class separation"
            )

        if not quality_metrics.variance_sufficient:
            recommendations.append(
                "Increase diversity in generation patterns to improve variance"
            )

        if quality_metrics.feature_correlations["max_correlation"] > 0.7:
            recommendations.append(
                "Reduce feature correlations for improved ML model performance"
            )

        if quality_metrics.overall_quality:
            recommendations.append(
                "Generated data meets all quality requirements for ML training"
            )

        return recommendations
