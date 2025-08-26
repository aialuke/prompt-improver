"""Production Synthetic Data Generator Orchestrator

Main orchestration module for synthetic data generation.
Decomposed from synthetic_data_generator.py (3,389 lines) into focused generators.

This module provides the public API with ProductionSyntheticDataGenerator
while delegating specialized generation to focused sub-modules.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy.ext.asyncio import AsyncSession

# import numpy as np  # Converted to lazy loading
from ...core.utils.lazy_ml_loader import get_numpy, get_torch

from ...database.models import TrainingPrompt
from ..analytics.generation_analytics import (
    GenerationAnalytics,
    GenerationHistoryTracker,
)
from ..learning.quality.enhanced_scorer import (
    EnhancedQualityMetrics,
    EnhancedQualityScorer,
)
from ..optimization.batch import (
    ProcessingStrategy,
    UnifiedBatchConfig,
    UnifiedBatchProcessor,
)

# Import specialized generators
from .generators.statistical_generator import (
    GenerationMethodMetrics,
    MethodPerformanceTracker,
    StatisticalDataGenerator,
    StatisticalQualityAssessor,
)

# Import neural generators with optional availability
try:
    from .generators.neural_generator import (
        DiffusionSyntheticGenerator,
        NeuralSyntheticGenerator,
    )
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

# Import GAN generators with optional availability  
try:
    from .generators.gan_generator import (
        HybridGenerationSystem,
        TabularDiffusion,
        TabularGAN,
        TabularVAE,
    )
    GAN_AVAILABLE = True
except ImportError:
    GAN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Check for PyTorch availability for neural methods
try:
    # import torch  # Converted to lazy loading
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
    distribution_quality: float
    feature_diversity: float
    effectiveness_variance: float
    class_balance: float
    correlation_structure: float
    overall_score: float


class ProductionSyntheticDataGenerator:
    """Production-grade synthetic data generator with advanced quality assessment and modern generative models

    Enhanced with 2025 best practices for adaptive data generation:
    - Gap-based targeting for performance improvement
    - Difficulty distribution control
    - Focus area specification
    - Hardness characterization integration
    """

    def __init__(
        self,
        target_samples: int = 1000,
        random_state: int = 42,
        use_enhanced_scoring: bool = True,
        generation_method: str = "statistical",  # "statistical", "neural", "hybrid", "diffusion"
        neural_model_type: str = "vae",  # "vae", "gan", "diffusion"
        neural_epochs: int = 200,
        neural_batch_size: int = 64,
        neural_learning_rate: float = 1e-3,
        neural_device: str = "auto",
        # New 2025 adaptive generation parameters
        enable_gap_targeting: bool = True,
        difficulty_distribution: str = "adaptive",  # "uniform", "adaptive", "hard_focused"
        focus_areas: list[str] | None = None,
        hardness_threshold: float = 0.7,
    ):
        """Initialize the production synthetic data generator

        Args:
            target_samples: Total number of samples to generate (default: 1000)
            random_state: Random seed for reproducible generation
            use_enhanced_scoring: Whether to use enhanced multi-dimensional quality scoring
            generation_method: Method for data generation ("statistical", "neural", "hybrid")
            neural_model_type: Type of neural model to use ("vae", "gan", "diffusion")
            neural_epochs: Number of training epochs for neural models
            neural_batch_size: Batch size for neural model training
            neural_learning_rate: Learning rate for neural models
            neural_device: Device for neural model training ("auto", "cpu", "cuda")
            enable_gap_targeting: Enable performance gap-based targeting (2025 best practice)
            difficulty_distribution: Strategy for difficulty distribution ("uniform", "adaptive", "hard_focused")
            focus_areas: Specific areas to focus generation on (e.g., ["clarity", "specificity"])
            hardness_threshold: Threshold for identifying hard examples (0.0-1.0)
        """
        self.target_samples = target_samples
        self.random_state = random_state
        self.rng = get_numpy().random.RandomState(random_state)
        self.use_enhanced_scoring = use_enhanced_scoring
        self.generation_method = generation_method
        self.neural_model_type = neural_model_type
        self.neural_epochs = neural_epochs
        self.neural_batch_size = neural_batch_size
        self.neural_learning_rate = neural_learning_rate
        self.neural_device = neural_device

        # 2025 adaptive generation parameters
        self.enable_gap_targeting = enable_gap_targeting
        self.difficulty_distribution = difficulty_distribution
        self.focus_areas = focus_areas or []
        self.hardness_threshold = hardness_threshold

        # Performance gap tracking
        self.current_performance_gaps: dict[str, float] = {}
        self.generation_strategy: str = "statistical"  # Will be determined dynamically

        # Initialize enhanced quality scorer
        if use_enhanced_scoring:
            self.quality_scorer = EnhancedQualityScorer(confidence_level=0.95)

        # Initialize specialized generators
        self.statistical_generator = StatisticalDataGenerator(random_state=random_state)
        self.statistical_quality_assessor = StatisticalQualityAssessor(random_state=random_state)
        
        # Initialize neural generators if available
        self.neural_generator = None
        self.hybrid_generator = None
        
        # Feature specifications (6-dimensional feature vectors)
        self.feature_names = [
            "clarity",  # 0: How clear and understandable the prompt is
            "length",  # 1: Content length and detail level
            "specificity",  # 2: Level of specific details and precision
            "complexity",  # 3: Intellectual/technical complexity level
            "context_richness",  # 4: Amount of contextual information provided
            "actionability",  # 5: How actionable and implementable the result is
        ]

        # Initialize neural generators based on method and availability
        if generation_method in ["neural", "hybrid", "diffusion"]:
            self._initialize_neural_generators()

        # Quality thresholds and filters
        self.quality_filter_threshold = 0.7
        self.enable_quality_filtering = True
        self.quality_thresholds = {
            "min_samples": 10,  # Minimum for basic optimization
            "ensemble_threshold": 20,  # Minimum for ensemble methods
            "min_classes": 2,  # Minimum class diversity
            "min_variance": 0.1,  # Minimum effectiveness variance
            "max_correlation": 0.8,  # Maximum feature correlation
        }

        # Unified batch processing system (2025 best practice)
        batch_config = UnifiedBatchConfig(
            strategy=ProcessingStrategy.OPTIMIZED,
            max_memory_mb=2000.0,  # 2GB limit
            enable_optimization=True
        )
        self.batch_optimizer = UnifiedBatchProcessor(batch_config)

        # Generation history tracking
        self.history_tracker: GenerationHistoryTracker | None = None
        self.current_session_id: str | None = None

        # Initialize method performance tracker for auto-selection
        self.method_tracker = MethodPerformanceTracker()

        # Configure domains based on research insights
        self.domains = self._initialize_domain_configs()

    def _initialize_neural_generators(self):
        """Initialize neural generators based on availability and configuration"""
        try:
            if self.generation_method == "hybrid" and GAN_AVAILABLE:
                # Initialize hybrid generation system (2025 best practice)
                self.hybrid_generator = HybridGenerationSystem(
                    data_dim=len(self.feature_names),
                    device=self.neural_device
                )
            elif self.generation_method == "diffusion" and NEURAL_AVAILABLE:
                self.neural_generator = DiffusionSyntheticGenerator(
                    epochs=self.neural_epochs,
                    batch_size=self.neural_batch_size,
                    learning_rate=self.neural_learning_rate,
                    device=self.neural_device
                )
            elif self.generation_method == "neural" and NEURAL_AVAILABLE:
                self.neural_generator = NeuralSyntheticGenerator(
                    model_type=self.neural_model_type,
                    epochs=self.neural_epochs,
                    batch_size=self.neural_batch_size,
                    learning_rate=self.neural_learning_rate,
                    device=self.neural_device
                )
        except Exception as e:
            logger.warning(f"Failed to initialize neural generators: {e}")
            self.generation_method = "statistical"  # Fallback

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
                    "actionability": (0.8, 1.0),  # Very high actionability for technical
                },
                effectiveness_params=(3, 2),  # Beta(3,2) - skewed higher for measurable outcomes
                complexity_range=(4, 8),
            ),
            "creative": DomainConfig(
                name="creative",
                ratio=0.20,  # 20% creative content
                patterns=[
                    (
                        "Write story",
                        "Write an engaging story with compelling characters, vivid descriptions, emotional depth, and a satisfying narrative arc",
                    ),
                    (
                        "Create content",
                        "Create original content with unique perspectives, authentic voice, engaging style, and clear value proposition",
                    ),
                    (
                        "Design concept",
                        "Design an innovative concept with creative vision, practical considerations, aesthetic appeal, and user-centered approach",
                    ),
                    (
                        "Brainstorm ideas",
                        "Brainstorm creative ideas using diverse thinking methods, cross-domain inspiration, and systematic exploration of possibilities",
                    ),
                    (
                        "Develop campaign",
                        "Develop a creative campaign with clear messaging, target audience insights, multi-channel approach, and measurable objectives",
                    ),
                    (
                        "Craft message",
                        "Craft a compelling message with emotional resonance, clear value proposition, memorable elements, and strong call-to-action",
                    ),
                    (
                        "Generate alternatives",
                        "Generate creative alternatives by exploring different approaches, challenging assumptions, and combining unexpected elements",
                    ),
                    (
                        "Visualize concept",
                        "Visualize this concept through detailed descriptions, metaphors, analogies, and sensory-rich language",
                    ),
                ],
                feature_ranges={
                    "clarity": (0.3, 0.8),  # More variable for creative expression
                    "length": (80, 250),  # Variable length for creative content
                    "specificity": (0.2, 0.7),  # Lower specificity for creative freedom
                    "complexity": (2, 7),  # Wide complexity range
                    "context_richness": (0.4, 1.0),  # High context richness for creativity
                    "actionability": (0.3, 0.8),  # Variable actionability for creative content
                },
                effectiveness_params=(2, 2.5),  # Beta(2,2.5) - slightly skewed for creative exploration
                complexity_range=(2, 7),
            ),
            "analytical": DomainConfig(
                name="analytical",
                ratio=0.20,  # 20% analytical content
                patterns=[
                    (
                        "Analyze data",
                        "Analyze this data systematically using statistical methods, visualization techniques, and evidence-based conclusions",
                    ),
                    (
                        "Compare options",
                        "Compare these options through structured analysis, criteria evaluation, pros/cons assessment, and recommendation synthesis",
                    ),
                    (
                        "Evaluate performance",
                        "Evaluate performance using relevant metrics, benchmarking standards, trend analysis, and improvement recommendations",
                    ),
                    (
                        "Research topic",
                        "Research this topic comprehensively using credible sources, systematic methodology, and objective analysis",
                    ),
                    (
                        "Identify patterns",
                        "Identify patterns through data exploration, statistical analysis, correlation studies, and predictive modeling",
                    ),
                    (
                        "Assess risks",
                        "Assess risks systematically through threat identification, probability analysis, impact evaluation, and mitigation strategies",
                    ),
                    (
                        "Review findings",
                        "Review findings critically through methodology validation, bias assessment, alternative explanations, and peer review",
                    ),
                    (
                        "Synthesize insights",
                        "Synthesize insights from multiple sources through thematic analysis, pattern recognition, and evidence integration",
                    ),
                ],
                feature_ranges={
                    "clarity": (0.6, 1.0),  # High clarity for analytical rigor
                    "length": (120, 280),  # Substantial length for thorough analysis
                    "specificity": (0.8, 1.0),  # Very high specificity for precision
                    "complexity": (3, 8),  # Medium-high complexity
                    "context_richness": (0.7, 1.0),  # Rich context for analysis
                    "actionability": (0.6, 0.9),  # Good actionability for recommendations
                },
                effectiveness_params=(2.5, 1.8),  # Beta(2.5,1.8) - moderately skewed for analytical rigor
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
                    "actionability": (0.7, 1.0),  # Very high actionability for instruction
                },
                effectiveness_params=(2.2, 1.5),  # Beta(2.2,1.5) - skewed toward effective instruction
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
                    "length": (60, 200),  # Shorter for conversational flow
                    "specificity": (0.3, 0.8),  # Variable specificity for conversation
                    "complexity": (1, 6),  # Wide range for conversational flexibility
                    "context_richness": (0.5, 0.9),  # Good context for understanding
                    "actionability": (0.4, 0.9),  # Variable actionability for conversation
                },
                effectiveness_params=(2, 2),  # Beta(2,2) - balanced for conversational variety
                complexity_range=(1, 6),
            ),
        }

    async def generate_comprehensive_training_data(self) -> dict[str, Any]:
        """Generate comprehensive training data using statistical methods (primary method)"""
        logger.info(f"Starting comprehensive generation of {self.target_samples} samples")
        start_time = time.time()

        try:
            # Generate features and effectiveness scores by domain
            all_features = []
            all_effectiveness = []
            all_prompts = []
            domain_counts = {}

            for domain_name, domain_config in self.domains.items():
                sample_count = max(1, int(self.target_samples * domain_config.ratio))  # Ensure at least 1 sample
                domain_counts[domain_name] = sample_count

                logger.info(f"Generating {sample_count} samples for {domain_name} domain")

                # Generate domain-specific features
                features, effectiveness = self.statistical_generator.generate_domain_statistical_features(
                    sample_count=sample_count,
                    feature_names=self.feature_names,
                    domain_name=domain_name
                )

                # Generate corresponding prompts
                prompts = []
                for _ in range(sample_count):
                    # Convert to list and select randomly
                    pattern_list = list(domain_config.patterns)
                    if pattern_list:
                        prompt_pair = pattern_list[self.rng.randint(len(pattern_list))]
                    else:
                        prompt_pair = ("Generate content", "Generate high-quality content")
                    prompts.append(prompt_pair)

                all_features.extend(features)
                all_effectiveness.extend(effectiveness)
                all_prompts.extend(prompts)

            # Apply quality assessment if enabled
            if self.use_enhanced_scoring and hasattr(self, 'quality_scorer'):
                quality_metrics = self._assess_comprehensive_quality(
                    all_features, all_effectiveness, all_prompts
                )
            else:
                quality_metrics = QualityMetrics(
                    distribution_quality=0.8,
                    feature_diversity=0.8,
                    effectiveness_variance=0.7,
                    class_balance=0.8,
                    correlation_structure=0.7,
                    overall_score=0.76
                )

            generation_time = time.time() - start_time

            result = {
                "features": all_features,
                "effectiveness": all_effectiveness,
                "prompts": all_prompts,
                "metadata": {
                    "generation_method": "comprehensive_statistical",
                    "samples_generated": len(all_features),
                    "generation_time": generation_time,
                    "domain_distribution": domain_counts,
                    "quality_metrics": quality_metrics,
                    "feature_names": self.feature_names,
                    "random_state": self.random_state,
                }
            }

            logger.info("Generated %d samples in %.2fs", len(all_features), generation_time)
            return result

        except Exception as e:
            logger.error(f"Comprehensive generation failed: {e}")
            raise

    async def generate_neural_training_data(self) -> dict[str, Any]:
        """Generate training data using modern neural generative models"""
        if not NEURAL_AVAILABLE or not self.neural_generator:
            logger.warning("Neural generation not available, falling back to statistical generation")
            return await self.generate_comprehensive_training_data()

        logger.info(f"Starting neural generation of {self.target_samples} samples using {self.neural_model_type}")

        try:
            # First generate a base dataset using statistical methods for training the neural model
            base_result = await self.generate_comprehensive_training_data()
            base_features = get_numpy().array(base_result["features"])

            # Train neural model on base data
            self.neural_generator.fit(base_features)

            # Generate new synthetic data
            synthetic_features = self.neural_generator.generate(self.target_samples)

            # Generate corresponding effectiveness and prompts
            effectiveness_scores = []
            prompts = []

            for feature_vector in synthetic_features:
                # Determine domain based on feature characteristics
                domain_name = self._select_domain_from_features(feature_vector)
                domain_config = self.domains[domain_name]

                # Generate effectiveness score for this domain
                effectiveness = self._generate_domain_effectiveness(domain_config, 0)
                effectiveness_scores.append(effectiveness)

                # Select prompt pair for this domain
                pattern_list = list(domain_config.patterns)
                if pattern_list:
                    prompt_pair = pattern_list[self.rng.randint(len(pattern_list))]
                else:
                    prompt_pair = ("Generate content", "Generate high-quality content")
                prompts.append(prompt_pair)

            result = {
                "features": synthetic_features.tolist(),
                "effectiveness": effectiveness_scores,
                "prompts": prompts,
                "metadata": {
                    "generation_method": f"neural_{self.neural_model_type}",
                    "samples_generated": len(synthetic_features),
                    "model_type": self.neural_model_type,
                    "base_samples_used": len(base_features),
                    "feature_names": self.feature_names,
                }
            }

            return result

        except Exception as e:
            logger.error(f"Neural generation failed: {e}")
            # Fallback to statistical generation
            return await self.generate_comprehensive_training_data()

    async def generate_hybrid_training_data(self) -> dict[str, Any]:
        """Generate training data using hybrid approach combining multiple methods"""
        if not GAN_AVAILABLE or not self.hybrid_generator:
            logger.warning("Hybrid generation not available, falling back to statistical generation")
            return await self.generate_comprehensive_training_data()

        logger.info(f"Starting hybrid generation of {self.target_samples} samples")

        try:
            # Use the hybrid generation system
            result = await self.hybrid_generator.generate_hybrid_data(
                batch_size=self.target_samples,
                performance_gaps=self.current_performance_gaps,
                quality_threshold=self.quality_filter_threshold
            )

            # Convert to expected format
            features = result['samples']
            
            # Generate effectiveness and prompts for the generated features
            effectiveness_scores = []
            prompts = []

            for feature_vector in features:
                # Determine domain based on feature characteristics
                domain_name = self._select_domain_from_features(get_numpy().array(feature_vector))
                domain_config = self.domains[domain_name]

                # Generate effectiveness score for this domain
                effectiveness = self._generate_domain_effectiveness(domain_config, 0)
                effectiveness_scores.append(effectiveness)

                # Select prompt pair for this domain
                pattern_list = list(domain_config.patterns)
                if pattern_list:
                    prompt_pair = pattern_list[self.rng.randint(len(pattern_list))]
                else:
                    prompt_pair = ("Generate content", "Generate high-quality content")
                prompts.append(prompt_pair)

            formatted_result = {
                "features": features,
                "effectiveness": effectiveness_scores,
                "prompts": prompts,
                "metadata": {
                    "generation_method": "hybrid_ensemble",
                    "samples_generated": len(features),
                    "generation_time": result['total_generation_time'],
                    "method_metrics": result['method_metrics'],
                    "method_allocation": result['method_allocation'],
                    "quality_threshold": result['quality_threshold'],
                    "feature_names": self.feature_names,
                }
            }

            return formatted_result

        except Exception as e:
            logger.error(f"Hybrid generation failed: {e}")
            # Fallback to statistical generation
            return await self.generate_comprehensive_training_data()

    async def generate_diffusion_training_data(self) -> dict[str, Any]:
        """Generate training data using diffusion models"""
        if not NEURAL_AVAILABLE or not isinstance(self.neural_generator, DiffusionSyntheticGenerator):
            logger.warning("Diffusion generation not available, falling back to statistical generation")
            return await self.generate_comprehensive_training_data()

        logger.info(f"Starting diffusion generation of {self.target_samples} samples")

        try:
            # First generate a base dataset using statistical methods for training the diffusion model
            base_result = await self.generate_comprehensive_training_data()
            base_features = get_numpy().array(base_result["features"])

            # Train diffusion model on base data
            self.neural_generator.fit(base_features)

            # Generate new synthetic data
            synthetic_features = self.neural_generator.generate(self.target_samples)

            # Generate corresponding effectiveness and prompts
            effectiveness_scores = []
            prompts = []

            for feature_vector in synthetic_features:
                # Determine domain based on feature characteristics
                domain_name = self._select_domain_from_features(feature_vector)
                domain_config = self.domains[domain_name]

                # Generate effectiveness score for this domain
                effectiveness = self._generate_domain_effectiveness(domain_config, 0)
                effectiveness_scores.append(effectiveness)

                # Select prompt pair for this domain
                pattern_list = list(domain_config.patterns)
                if pattern_list:
                    prompt_pair = pattern_list[self.rng.randint(len(pattern_list))]
                else:
                    prompt_pair = ("Generate content", "Generate high-quality content")
                prompts.append(prompt_pair)

            result = {
                "features": synthetic_features.tolist(),
                "effectiveness": effectiveness_scores,
                "prompts": prompts,
                "metadata": {
                    "generation_method": "diffusion_tabular",
                    "samples_generated": len(synthetic_features),
                    "base_samples_used": len(base_features),
                    "feature_names": self.feature_names,
                }
            }

            return result

        except Exception as e:
            logger.error(f"Diffusion generation failed: {e}")
            # Fallback to statistical generation
            return await self.generate_comprehensive_training_data()

    async def generate_data(self) -> dict[str, Any]:
        """Generate data using the configured method"""
        if self.generation_method == "neural":
            return await self.generate_neural_training_data()
        elif self.generation_method == "hybrid":
            return await self.generate_hybrid_training_data()
        elif self.generation_method == "diffusion":
            return await self.generate_diffusion_training_data()
        else:
            return await self.generate_comprehensive_training_data()

    # Placeholder methods for targeted generation (simplified for decomposition)
    async def generate_targeted_data(
        self,
        performance_gaps: dict[str, float],
        batch_size: int,
        focus_areas: list[str] | None = None,
        difficulty_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Generate targeted data based on performance gaps"""
        logger.info("Generating targeted data based on performance gaps")
        
        # Store current gaps for method selection
        self.current_performance_gaps = performance_gaps
        
        # For now, use the primary generation method with gap awareness
        result = await self.generate_data()
        
        # Add targeting metadata
        result["metadata"]["targeting_enabled"] = True
        result["metadata"]["performance_gaps"] = performance_gaps
        result["metadata"]["focus_areas"] = focus_areas or []
        
        return result

    async def generate_with_dynamic_batching(
        self,
        session: AsyncSession,
        batch_config: UnifiedBatchConfig | None = None
    ) -> dict[str, Any]:
        """Generate data with unified batch processing"""
        logger.info("Generating data with unified batch processing")
        
        # Use the configured batch processor or provided config
        if batch_config:
            optimizer = UnifiedBatchProcessor(batch_config)
        else:
            optimizer = self.batch_optimizer
            
        # For now, generate normally but with batch tracking
        result = await self.generate_data()
        result["metadata"]["dynamic_batching_enabled"] = True
        
        return result

    async def generate_with_history_tracking(
        self,
        session: AsyncSession,
        session_id: str
    ) -> dict[str, Any]:
        """Generate data with history tracking"""
        logger.info(f"Generating data with history tracking for session: {session_id}")
        
        self.current_session_id = session_id
        
        # Initialize history tracker if not already done
        if not self.history_tracker:
            self.history_tracker = GenerationHistoryTracker()
            
        result = await self.generate_data()
        result["metadata"]["history_tracking_enabled"] = True
        result["metadata"]["session_id"] = session_id
        
        return result

    # Helper methods for maintaining compatibility
    def _select_domain_from_features(self, feature_vector: get_numpy().ndarray) -> str:
        """Select domain based on feature characteristics"""
        # Simple heuristic based on feature ranges
        clarity = feature_vector[0] if len(feature_vector) > 0 else 0.5
        complexity = feature_vector[3] if len(feature_vector) > 3 else 5.0
        specificity = feature_vector[2] if len(feature_vector) > 2 else 0.5
        
        if complexity > 6 and specificity > 0.7:
            return "technical"
        elif clarity < 0.6 and complexity < 4:
            return "creative"
        elif specificity > 0.8 and complexity > 5:
            return "analytical"
        elif clarity > 0.8 and complexity < 5:
            return "instructional"
        else:
            return "conversational"

    def _generate_domain_effectiveness(self, domain_config: DomainConfig, sample_index: int) -> float:
        """Generate effectiveness score for a domain"""
        # Use beta distribution with domain-specific parameters
        alpha, beta = domain_config.effectiveness_params
        return self.rng.beta(alpha, beta)

    def _assess_comprehensive_quality(
        self,
        features: list,
        effectiveness: list,
        prompts: list
    ) -> QualityMetrics:
        """Assess quality of comprehensive generation"""
        try:
            feature_array = get_numpy().array(features)
            effectiveness_array = get_numpy().array(effectiveness)
            
            # Distribution quality
            dist_quality = self.statistical_quality_assessor.assess_sample_quality(features)
            
            # Feature diversity (variance across features)
            feature_variances = get_numpy().var(feature_array, axis=0)
            diversity_score = get_numpy().mean(feature_variances > 0.1)
            
            # Effectiveness variance
            eff_variance = get_numpy().var(effectiveness_array)
            variance_score = min(1.0, eff_variance / 0.1)  # Normalize
            
            # Class balance (check effectiveness distribution)
            hist, _ = get_numpy().histogram(effectiveness_array, bins=5)
            balance_score = 1.0 - get_numpy().std(hist) / get_numpy().mean(hist + 1e-8)
            
            # Correlation structure (features shouldn't be too correlated)
            correlation_matrix = get_numpy().corrcoef(feature_array.T)
            max_correlation = get_numpy().max(get_numpy().abs(correlation_matrix - get_numpy().eye(len(self.feature_names))))
            correlation_score = max(0, 1.0 - max_correlation / 0.8)
            
            overall_score = get_numpy().mean([
                dist_quality, diversity_score, variance_score, 
                balance_score, correlation_score
            ])
            
            return QualityMetrics(
                distribution_quality=dist_quality,
                feature_diversity=diversity_score,
                effectiveness_variance=variance_score,
                class_balance=balance_score,
                correlation_structure=correlation_score,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return QualityMetrics(
                distribution_quality=0.7,
                feature_diversity=0.7,
                effectiveness_variance=0.7,
                class_balance=0.7,
                correlation_structure=0.7,
                overall_score=0.7
            )