"""Context-Specific Learning Engine

Optimizes prompt improvement strategies for different project types and contexts.
Learns from historical data to identify context-specific patterns and specialization
opportunities for more effective rule application.
"""

import hashlib
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

# Phase 2 enhancement imports
try:
    import hdbscan
    import umap

    ADVANCED_CLUSTERING_AVAILABLE = True
except ImportError:
    ADVANCED_CLUSTERING_AVAILABLE = False
    warnings.warn(
        "HDBSCAN and UMAP not available. Install with: pip install hdbscan umap-learn"
    )

# Phase 3 enhancement imports for advanced privacy-preserving ML
try:
    import opacus
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    warnings.warn(
        "Opacus not available for differential privacy. Install with: pip install opacus"
    )

try:
    import cryptography
    from cryptography.fernet import Fernet

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    warnings.warn(
        "Cryptography not available for secure storage. Install with: pip install cryptography"
    )

# Linguistic analysis integration
try:
    from ..analysis.linguistic_analyzer import (
        LinguisticAnalyzer,
        LinguisticConfig,
        get_lightweight_config,
        get_memory_optimized_config,
        get_ultra_lightweight_config,
    )

    LINGUISTIC_ANALYSIS_AVAILABLE = True
except ImportError:
    LINGUISTIC_ANALYSIS_AVAILABLE = False
    warnings.warn(
        "LinguisticAnalyzer not available. Linguistic features will be disabled."
    )

# Domain-specific feature extraction integration
try:
    from ..analysis.domain_detector import PromptDomain
    from ..analysis.domain_feature_extractor import (
        DomainFeatureExtractor,
        DomainFeatures,
    )

    DOMAIN_ANALYSIS_AVAILABLE = True
except ImportError:
    DOMAIN_ANALYSIS_AVAILABLE = False
    warnings.warn(
        "Domain feature extraction not available. Domain-specific features will be disabled."
    )

# Context-aware feature weighting integration
try:
    from .context_aware_weighter import (
        ContextAwareFeatureWeighter,
        WeightingConfig,
        WeightingStrategy,
    )

    CONTEXT_WEIGHTING_AVAILABLE = True
except ImportError:
    CONTEXT_WEIGHTING_AVAILABLE = False
    warnings.warn(
        "Context-aware feature weighting not available. Feature weighting will be disabled."
    )

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context-specific learning"""

    # Learning parameters
    significance_threshold: float = 0.15
    min_sample_size: int = 10
    similarity_threshold: float = 0.8

    # Performance tracking
    improvement_threshold: float = 0.1
    consistency_threshold: float = 0.7

    # Specialization parameters
    max_specializations: int = 5
    confidence_threshold: float = 0.8

    # Context analysis
    max_context_groups: int = 20
    enable_semantic_clustering: bool = True

    # Phase 2 enhancements - In-Context Learning
    enable_in_context_learning: bool = True
    icl_demonstrations: int = 5
    icl_similarity_threshold: float = 0.7
    privacy_preserving: bool = True
    differential_privacy_epsilon: float = 1.0

    # Phase 3 enhancements - Advanced Privacy-Preserving ML
    enable_federated_learning: bool = True
    federated_rounds: int = 3
    federated_min_clients: int = 2
    secure_aggregation: bool = True
    homomorphic_encryption: bool = False
    privacy_budget_tracking: bool = True
    max_privacy_budget: float = 10.0
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0

    # Phase 2 enhancements - Advanced Clustering
    use_advanced_clustering: bool = True
    umap_n_components: int = 10
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    clustering_quality_threshold: float = 0.5

    # Linguistic Analysis Integration
    enable_linguistic_features: bool = True
    linguistic_feature_weight: float = (
        0.3  # Weight for linguistic features vs traditional features
    )
    cache_linguistic_analysis: bool = True  # Cache results for performance

    # Domain-Specific Feature Extraction
    enable_domain_features: bool = True
    domain_feature_weight: float = 0.4  # Weight for domain-specific features
    cache_domain_analysis: bool = True  # Cache domain analysis results
    adaptive_domain_weighting: bool = True  # Adjust weights based on domain confidence

    # Context-Aware Feature Weighting
    enable_context_aware_weighting: bool = True
    weighting_strategy: str = "adaptive"  # static, adaptive, dynamic, hybrid
    confidence_boost_factor: float = 0.3
    min_weight_threshold: float = 0.1
    max_weight_threshold: float = 2.0
    secondary_domain_weight_factor: float = 0.6
    normalize_feature_weights: bool = True

    # Memory and Performance Optimization
    use_lightweight_models: bool = False  # Use lightweight models for testing
    use_ultra_lightweight_models: bool = (
        False  # Use ultra-lightweight models for extreme memory constraints
    )
    enable_model_quantization: bool = True  # Enable model quantization
    enable_4bit_quantization: bool = False  # Enable aggressive 4-bit quantization
    max_memory_threshold_mb: int = 200  # Maximum memory threshold
    force_cpu_only: bool = False  # Force CPU-only processing

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.differential_privacy_epsilon < 0:
            raise ValueError("Differential privacy epsilon must be non-negative")
        if self.privacy_preserving and self.differential_privacy_epsilon == 0:
            raise ValueError("Epsilon cannot be zero when privacy is enabled")


@dataclass
class ContextInsight:
    """Insights for a specific context"""

    context_key: str
    sample_size: int
    avg_performance: float
    consistency_score: float
    top_performing_rules: list[dict[str, Any]]
    poor_performing_rules: list[dict[str, Any]]
    specialization_potential: float
    unique_patterns: list[str]


@dataclass
class SpecializationOpportunity:
    """Opportunity for rule specialization"""

    context_key: str
    rule_id: str
    current_performance: float
    specialized_performance: float
    improvement_potential: float
    confidence: float
    required_modifications: list[str]
    cost_benefit_ratio: float


@dataclass
class LearningRecommendation:
    """Learning-based recommendation"""

    type: str  # 'specialize', 'generalize', 'create_new', 'deprecate'
    priority: str  # 'high', 'medium', 'low'
    context: str
    description: str
    expected_impact: float
    implementation_effort: str
    supporting_evidence: list[str]


class ContextSpecificLearner:
    """Context-Specific Learning Engine for prompt improvement optimization"""

    def __init__(self, config: ContextConfig | None = None):
        """Initialize the context-specific learner

        Args:
            config: Configuration for context learning
        """
        self.config = config or ContextConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Learning state
        self.context_groups: dict[str, list[dict[str, Any]]] = {}
        self.context_patterns: dict[str, dict[str, Any]] = {}
        self.specialization_opportunities: list[SpecializationOpportunity] = []
        self.performance_baseline: dict[str, float] | None = None

        # Phase 2 enhancements - In-Context Learning state
        self.demonstration_cache: dict[str, list[dict[str, Any]]] = {}
        self.context_embeddings: np.ndarray | None = None
        self.icl_model: Any | None = None

        # Phase 3 enhancements - Advanced Privacy-Preserving ML state
        self.privacy_engine: Any | None = None
        self.privacy_budget_used: float = 0.0
        self.federated_models: dict[str, Any] = {}
        self.encryption_key: bytes | None = None
        self.secure_aggregator: Any | None = None

        # Initialize vectorizer for semantic context analysis
        if self.config.enable_semantic_clustering:
            self.context_vectorizer = TfidfVectorizer(
                max_features=1000, stop_words="english", ngram_range=(1, 2)
            )
        else:
            self.context_vectorizer = None

        # Initialize linguistic analyzer for advanced features
        self.linguistic_analyzer = None
        self.linguistic_cache = {}  # Cache for linguistic analysis results
        if self.config.enable_linguistic_features and LINGUISTIC_ANALYSIS_AVAILABLE:
            try:
                # Configure linguistic analyzer for performance in ML pipeline
                linguistic_config = LinguisticConfig(
                    max_workers=2,  # Conservative for ML pipeline
                    enable_caching=self.config.cache_linguistic_analysis,
                    cache_size=1000,  # Reasonable cache for ML pipeline
                    timeout_seconds=15,  # Faster timeout for ML processing
                    # Memory optimization settings
                    use_lightweight_models=self.config.use_lightweight_models,
                    use_ultra_lightweight_models=self.config.use_ultra_lightweight_models,
                    enable_model_quantization=self.config.enable_model_quantization,
                    enable_4bit_quantization=self.config.enable_4bit_quantization,
                    max_memory_threshold_mb=self.config.max_memory_threshold_mb,
                    force_cpu_only=self.config.force_cpu_only,
                    # Auto-download NLTK resources
                    auto_download_nltk=True,
                    nltk_fallback_enabled=True,
                )
                self.linguistic_analyzer = LinguisticAnalyzer(linguistic_config)
                self.logger.info(
                    "Linguistic analysis enabled for ML feature extraction with optimized configuration"
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize linguistic analyzer: {e}")
                self.config.enable_linguistic_features = False
        elif self.config.enable_linguistic_features:
            self.logger.warning(
                "Linguistic features requested but LinguisticAnalyzer not available"
            )
            self.config.enable_linguistic_features = False

        # Initialize domain feature extractor for domain-specific analysis
        self.domain_feature_extractor = None
        self.domain_cache = {}  # Cache for domain analysis results
        if self.config.enable_domain_features and DOMAIN_ANALYSIS_AVAILABLE:
            try:
                # Configure domain feature extractor with spaCy if available
                enable_spacy = True  # Default to enabling spaCy for better analysis
                self.domain_feature_extractor = DomainFeatureExtractor(
                    enable_spacy=enable_spacy
                )
                self.logger.info(
                    "Domain-specific feature extraction enabled for ML pipeline"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize domain feature extractor: {e}"
                )
                self.config.enable_domain_features = False
        elif self.config.enable_domain_features:
            self.logger.warning(
                "Domain features requested but DomainFeatureExtractor not available"
            )
            self.config.enable_domain_features = False

        # Initialize context-aware feature weighter for adaptive feature importance
        self.context_aware_weighter = None
        if self.config.enable_context_aware_weighting and CONTEXT_WEIGHTING_AVAILABLE:
            try:
                # Create weighting configuration from context config
                weighting_config = WeightingConfig(
                    enable_context_aware_weighting=self.config.enable_context_aware_weighting,
                    weighting_strategy=WeightingStrategy(
                        self.config.weighting_strategy
                    ),
                    confidence_boost_factor=self.config.confidence_boost_factor,
                    min_weight_threshold=self.config.min_weight_threshold,
                    max_weight_threshold=self.config.max_weight_threshold,
                    secondary_domain_weight_factor=self.config.secondary_domain_weight_factor,
                    normalize_weights=self.config.normalize_feature_weights,
                )
                self.context_aware_weighter = ContextAwareFeatureWeighter(
                    weighting_config
                )
                self.logger.info(
                    "Context-aware feature weighting enabled for ML pipeline"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize context-aware feature weighter: {e}"
                )
                self.config.enable_context_aware_weighting = False
        elif self.config.enable_context_aware_weighting:
            self.logger.warning(
                "Context-aware weighting requested but ContextAwareFeatureWeighter not available"
            )
            self.config.enable_context_aware_weighting = False

        # Phase 2 enhancements - Advanced clustering components
        self.umap_reducer = None
        self.hdbscan_clusterer = None
        self.clustering_quality_score = 0.0

        # Check for advanced clustering availability
        if self.config.use_advanced_clustering and not ADVANCED_CLUSTERING_AVAILABLE:
            self.logger.warning(
                "Advanced clustering requested but HDBSCAN/UMAP not available. Falling back to K-means."
            )
            self.config.use_advanced_clustering = False

        # Phase 3: Initialize privacy-preserving components
        if self.config.privacy_preserving:
            self._initialize_privacy_components()

    async def analyze_context_effectiveness(
        self, results: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze context effectiveness across test results

        Args:
            results: Context-grouped data with historical_data fields
                    Format: {"context_name": {"historical_data": [...], "sample_size": N, ...}}

        Returns:
            Context-specific learning analysis
        """
        # Convert context-grouped data to flat results list for internal processing
        flat_results = []
        for context_key, context_data in results.items():
            if "historical_data" in context_data:
                historical_data = context_data["historical_data"]
                if isinstance(historical_data, list):
                    flat_results.extend(historical_data)

        self.logger.info(
            f"Starting context effectiveness analysis with {len(flat_results)} results from {len(results)} contexts"
        )

        # Phase 2 enhancement: Apply advanced clustering if enabled
        clustering_result = None
        if self.config.use_advanced_clustering and flat_results:
            clustering_result = self._perform_advanced_clustering(flat_results)
            if clustering_result.get("status") == "success":
                context_groups = clustering_result["clusters"]
                self.logger.info(
                    f"Advanced clustering identified {clustering_result['n_clusters']} context groups with quality {clustering_result['quality_score']:.3f}"
                )
            # Fallback to using pre-grouped data if available
            elif all(
                "historical_data" in context_data for context_data in results.values()
            ):
                context_groups = {}
                for context_key, context_data in results.items():
                    context_groups[context_key] = context_data.get(
                        "historical_data", []
                    )
                self.logger.info(
                    f"Advanced clustering failed, using pre-grouped data: {len(context_groups)} context groups"
                )
            else:
                # Fallback to traditional context grouping
                context_groups = self._group_results_by_context(flat_results)
                self.logger.info(
                    f"Fallback: Identified {len(context_groups)} context groups"
                )
        elif all(
            "historical_data" in context_data for context_data in results.values()
        ):
            # Use pre-grouped data when advanced clustering is disabled
            context_groups = {}
            for context_key, context_data in results.items():
                context_groups[context_key] = context_data.get("historical_data", [])
            self.logger.info(
                f"Using pre-grouped data: {len(context_groups)} context groups"
            )
        else:
            # Traditional context grouping
            context_groups = self._group_results_by_context(flat_results)
            self.logger.info(f"Identified {len(context_groups)} context groups")

        # Analyze each context group
        context_insights = {}
        for context_key, context_results in context_groups.items():
            context_insights[context_key] = await self._analyze_context_group(
                context_key, context_results
            )

        # Cross-context analysis
        cross_context_analysis = self._perform_cross_context_analysis(context_insights)

        # Identify specialization opportunities
        specialization_opportunities = self._identify_specialization_opportunities(
            context_insights
        )

        # Generate learning recommendations
        learning_recommendations = self._generate_learning_recommendations(
            context_insights, specialization_opportunities
        )

        # Phase 2 enhancement: Apply in-context learning if enabled
        icl_result = None
        if self.config.enable_in_context_learning:
            # Extract context data and user preferences for ICL
            context_data = []
            for result in flat_results[:20]:  # Sample for ICL analysis
                if "context" in result:
                    context_data.append(result["context"])

            user_preferences = {}  # Could be populated from user settings
            task_examples = flat_results[:50]  # Sample task examples

            icl_result = self._implement_in_context_learning(
                context_data, user_preferences, task_examples
            )

        analysis_result = {
            "context_insights": context_insights,
            "cross_context_comparisons": cross_context_analysis["comparisons"],
            "universal_patterns": cross_context_analysis["universal_patterns"],
            "context_specific_patterns": cross_context_analysis[
                "context_specific_patterns"
            ],
            "specialization_opportunities": [
                op.__dict__ for op in specialization_opportunities
            ],
            "learning_recommendations": [
                rec.__dict__ for rec in learning_recommendations
            ],
            "clustering_quality_score": self.clustering_quality_score,
            "metadata": {
                "total_contexts": len(context_insights),
                "total_results": len(results),
                "analysis_date": datetime.now().isoformat(),
                "config": self.config.__dict__,
                # Phase 2 metadata
                "phase2_enhancements": {
                    "in_context_learning_enabled": self.config.enable_in_context_learning,
                    "advanced_clustering_enabled": self.config.use_advanced_clustering,
                    "privacy_preserving_enabled": self.config.privacy_preserving,
                    "clustering_method": "umap_hdbscan"
                    if ADVANCED_CLUSTERING_AVAILABLE
                    and self.config.use_advanced_clustering
                    else "kmeans",
                },
            },
        }

        # Only add in-context learning result if ICL is enabled
        if icl_result is not None:
            analysis_result["in_context_learning"] = icl_result

        # Add advanced clustering result only if clustering was performed
        if clustering_result is not None:
            analysis_result["advanced_clustering"] = clustering_result

        self.logger.info(
            f"Context analysis completed: {len(context_insights)} contexts, "
            f"{len(specialization_opportunities)} specialization opportunities"
        )

        return analysis_result

    def _group_results_by_context(
        self, results: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group results by project context"""
        groups = defaultdict(list)

        for result in results:
            context_key = self._generate_context_key(result.get("context", {}))
            groups[context_key].append(result)

        # Filter out groups with insufficient samples
        filtered_groups = {
            key: group
            for key, group in groups.items()
            if len(group) >= self.config.min_sample_size
        }

        self.logger.info(
            f"Filtered {len(groups)} groups to {len(filtered_groups)} with sufficient samples"
        )

        return filtered_groups

    def _generate_context_key(self, context: dict[str, Any]) -> str:
        """Generate a context key for grouping similar contexts"""
        if not context:
            return "unknown"

        # Extract key context features
        project_type = context.get("projectType", "unknown")
        domain = context.get("domain", "unknown")
        complexity = context.get("complexity", "unknown")
        team_size = context.get("teamSize", "unknown")

        # Create a normalized context key
        key_components = [project_type, domain, complexity, str(team_size)]
        context_key = "|".join(key_components).lower()

        return context_key

    async def _analyze_context_group(
        self, context_key: str, results: list[dict[str, Any]]
    ) -> ContextInsight:
        """Analyze a specific context group"""
        # Extract performance metrics
        performance_scores = []
        rule_performance = defaultdict(list)

        for result in results:
            score = result.get("overallScore") or result.get("improvementScore", 0)
            performance_scores.append(score)

            applied_rules = result.get("appliedRules", [])
            for rule in applied_rules:
                rule_id = rule.get("ruleId") or rule.get("id", "unknown")
                rule_score = rule.get("improvementScore", score)
                rule_performance[rule_id].append(rule_score)

        if not performance_scores:
            return ContextInsight(
                context_key=context_key,
                sample_size=len(results),
                avg_performance=0.0,
                consistency_score=0.0,
                top_performing_rules=[],
                poor_performing_rules=[],
                specialization_potential=0.0,
                unique_patterns=[],
            )

        # Calculate basic statistics
        avg_performance = float(np.mean(performance_scores))
        consistency_score = (
            1 - (np.std(performance_scores) / avg_performance)
            if avg_performance > 0
            else 0
        )

        # Analyze rule performance
        rule_stats = {}
        for rule_id, scores in rule_performance.items():
            if len(scores) >= 3:  # Minimum for reliable statistics
                rule_stats[rule_id] = {
                    "avg_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "sample_size": len(scores),
                    "consistency": 1 - (np.std(scores) / np.mean(scores))
                    if np.mean(scores) > 0
                    else 0,
                }

        # Identify top and poor performing rules
        sorted_rules = sorted(
            rule_stats.items(), key=lambda x: x[1]["avg_score"], reverse=True
        )

        top_performing_rules = [
            {"rule_id": rule_id, **stats} for rule_id, stats in sorted_rules[:5]
        ]

        poor_performing_rules = [
            {"rule_id": rule_id, **stats}
            for rule_id, stats in sorted_rules[-3:]
            if stats["avg_score"] < avg_performance * 0.8
        ]

        # Calculate specialization potential
        specialization_potential = self._calculate_specialization_potential(
            context_key, rule_stats, avg_performance
        )

        # Identify unique patterns
        unique_patterns = self._identify_unique_patterns(results, context_key)

        return ContextInsight(
            context_key=context_key,
            sample_size=len(results),
            avg_performance=avg_performance,
            consistency_score=float(max(0, consistency_score)),
            top_performing_rules=top_performing_rules,
            poor_performing_rules=poor_performing_rules,
            specialization_potential=specialization_potential,
            unique_patterns=unique_patterns,
        )

    def _calculate_specialization_potential(
        self,
        context_key: str,
        rule_stats: dict[str, dict[str, Any]],
        context_avg: float,
    ) -> float:
        """Calculate the potential for rule specialization in this context"""
        if not rule_stats or not self.performance_baseline:
            return 0.0

        # Compare context performance to baseline
        baseline_avg = self.performance_baseline.get("overall", context_avg)
        context_deviation = (
            abs(context_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else 0
        )

        # Look for rules with significantly different performance
        rule_deviations = []
        for rule_id, stats in rule_stats.items():
            baseline_rule_avg = self.performance_baseline.get(
                rule_id, stats["avg_score"]
            )
            if baseline_rule_avg > 0:
                deviation = (
                    abs(stats["avg_score"] - baseline_rule_avg) / baseline_rule_avg
                )
                rule_deviations.append(deviation)

        if not rule_deviations:
            return context_deviation

        # Combine context and rule deviations
        avg_rule_deviation = np.mean(rule_deviations)
        specialization_potential = (context_deviation + avg_rule_deviation) / 2

        return min(1.0, specialization_potential)

    def _identify_unique_patterns(
        self, results: list[dict[str, Any]], context_key: str
    ) -> list[str]:
        """Identify unique patterns in this context"""
        patterns = []

        # Analyze prompt characteristics
        prompt_lengths = []
        prompt_types = defaultdict(int)
        improvement_types = defaultdict(int)

        for result in results:
            original_prompt = result.get("originalPrompt", "")
            prompt_lengths.append(len(original_prompt.split()))

            # Extract prompt type patterns
            if "question" in original_prompt.lower():
                prompt_types["question"] += 1
            elif "instruction" in original_prompt.lower():
                prompt_types["instruction"] += 1
            elif "describe" in original_prompt.lower():
                prompt_types["description"] += 1
            else:
                prompt_types["other"] += 1

            # Extract improvement patterns
            improvements = result.get("appliedImprovements", [])
            for improvement in improvements:
                improvement_types[improvement.get("type", "unknown")] += 1

        # Identify significant patterns
        if prompt_lengths:
            avg_length = np.mean(prompt_lengths)
            if avg_length > 50:
                patterns.append(f"long_prompts_avg_{avg_length:.1f}_words")
            elif avg_length < 10:
                patterns.append(f"short_prompts_avg_{avg_length:.1f}_words")

        # Most common prompt type
        if prompt_types:
            most_common_type = max(prompt_types.items(), key=lambda x: x[1])
            if most_common_type[1] / len(results) > 0.6:  # 60% threshold
                patterns.append(f"predominantly_{most_common_type[0]}_prompts")

        # Most common improvement type
        if improvement_types:
            most_common_improvement = max(improvement_types.items(), key=lambda x: x[1])
            if most_common_improvement[1] / len(results) > 0.4:  # 40% threshold
                patterns.append(f"frequent_{most_common_improvement[0]}_improvements")

        return patterns[:5]  # Limit to top 5 patterns

    def _perform_cross_context_analysis(
        self, context_insights: dict[str, ContextInsight]
    ) -> dict[str, Any]:
        """Perform cross-context analysis to identify universal vs context-specific patterns"""
        if len(context_insights) < 2:
            return {
                "comparisons": [],
                "universal_patterns": [],
                "context_specific_patterns": [],
            }

        comparisons = []
        universal_patterns = []
        context_specific_patterns = []

        # Compare contexts pairwise
        context_keys = list(context_insights.keys())
        for i, context1 in enumerate(context_keys):
            for context2 in context_keys[i + 1 :]:
                insight1 = context_insights[context1]
                insight2 = context_insights[context2]

                # Performance comparison
                perf_diff = abs(insight1.avg_performance - insight2.avg_performance)
                significant_diff = perf_diff > self.config.significance_threshold

                # Rule overlap analysis
                rules1 = {rule["rule_id"] for rule in insight1.top_performing_rules}
                rules2 = {rule["rule_id"] for rule in insight2.top_performing_rules}

                overlap = len(rules1.intersection(rules2))
                total_unique = len(rules1.union(rules2))
                overlap_ratio = overlap / total_unique if total_unique > 0 else 0

                comparison = {
                    "context_pair": [context1, context2],
                    "performance_difference": float(perf_diff),
                    "significant_difference": significant_diff,
                    "rule_overlap_ratio": float(overlap_ratio),
                    "shared_rules": list(rules1.intersection(rules2)),
                    "context1_unique_rules": list(rules1 - rules2),
                    "context2_unique_rules": list(rules2 - rules1),
                }
                comparisons.append(comparison)

        # Identify universal patterns (rules that work well across contexts)
        rule_performance_across_contexts = defaultdict(list)
        for insight in context_insights.values():
            for rule in insight.top_performing_rules:
                rule_performance_across_contexts[rule["rule_id"]].append({
                    "context": insight.context_key,
                    "score": rule["avg_score"],
                })

        for rule_id, performances in rule_performance_across_contexts.items():
            if (
                len(performances) >= len(context_insights) * 0.7
            ):  # Appears in 70% of contexts
                avg_score = np.mean([p["score"] for p in performances])
                consistency = (
                    1 - (np.std([p["score"] for p in performances]) / avg_score)
                    if avg_score > 0
                    else 0
                )

                if consistency > self.config.consistency_threshold:
                    universal_patterns.append({
                        "rule_id": rule_id,
                        "avg_score_across_contexts": float(avg_score),
                        "consistency": float(consistency),
                        "present_in_contexts": len(performances),
                        "contexts": [p["context"] for p in performances],
                    })

        # Identify context-specific patterns
        for context_key, insight in context_insights.items():
            for rule in insight.top_performing_rules:
                rule_id = rule["rule_id"]

                # Check if this rule is NOT universal
                is_universal = any(
                    up["rule_id"] == rule_id for up in universal_patterns
                )

                if (
                    not is_universal
                    and rule["avg_score"] > insight.avg_performance * 1.1
                ):
                    context_specific_patterns.append({
                        "context": context_key,
                        "rule_id": rule_id,
                        "context_score": rule["avg_score"],
                        "context_avg": insight.avg_performance,
                        "relative_performance": rule["avg_score"]
                        / insight.avg_performance,
                        "sample_size": rule["sample_size"],
                    })

        return {
            "comparisons": comparisons,
            "universal_patterns": universal_patterns,
            "context_specific_patterns": context_specific_patterns,
        }

    def _identify_specialization_opportunities(
        self, context_insights: dict[str, ContextInsight]
    ) -> list[SpecializationOpportunity]:
        """Identify opportunities for rule specialization"""
        opportunities = []

        for context_key, insight in context_insights.items():
            # Look for rules with poor performance that could be specialized
            for poor_rule in insight.poor_performing_rules:
                rule_id = poor_rule["rule_id"]
                current_performance = poor_rule["avg_score"]

                # Estimate potential improvement through specialization
                context_potential = insight.specialization_potential
                if context_potential > 0.3:  # Threshold for worthwhile specialization
                    # Calculate potential specialized performance
                    # This is a heuristic - in practice, you'd use more sophisticated modeling
                    specialized_performance = min(
                        1.0,
                        current_performance
                        + (context_potential * self.config.improvement_threshold),
                    )

                    improvement_potential = (
                        specialized_performance - current_performance
                    )

                    if improvement_potential > self.config.improvement_threshold:
                        # Estimate implementation cost
                        cost_benefit_ratio = improvement_potential / (
                            context_potential + 0.1
                        )

                        opportunity = SpecializationOpportunity(
                            context_key=context_key,
                            rule_id=rule_id,
                            current_performance=current_performance,
                            specialized_performance=specialized_performance,
                            improvement_potential=improvement_potential,
                            confidence=min(
                                1.0, context_potential * insight.consistency_score
                            ),
                            required_modifications=self._suggest_modifications(
                                insight, rule_id
                            ),
                            cost_benefit_ratio=cost_benefit_ratio,
                        )

                        if opportunity.confidence > self.config.confidence_threshold:
                            opportunities.append(opportunity)

        # Sort by improvement potential
        opportunities.sort(key=lambda x: x.improvement_potential, reverse=True)

        return opportunities[: self.config.max_specializations]

    def _suggest_modifications(
        self, insight: ContextInsight, rule_id: str
    ) -> list[str]:
        """Suggest modifications for rule specialization"""
        modifications = []

        # Based on unique patterns in the context
        for pattern in insight.unique_patterns:
            if "long_prompts" in pattern:
                modifications.append("Add length-aware optimizations")
            elif "short_prompts" in pattern:
                modifications.append("Focus on conciseness improvements")
            elif "question" in pattern:
                modifications.append("Optimize for question-answering format")
            elif "instruction" in pattern:
                modifications.append("Enhance instruction clarity")
            elif "description" in pattern:
                modifications.append("Improve descriptive language")

        # Based on performance characteristics
        if insight.consistency_score < 0.5:
            modifications.append("Add consistency checks")

        if insight.avg_performance < 0.7:
            modifications.append("Strengthen core improvement logic")

        return modifications[:3]  # Limit to top 3 suggestions

    def _generate_learning_recommendations(
        self,
        context_insights: dict[str, ContextInsight],
        specialization_opportunities: list[SpecializationOpportunity],
    ) -> list[LearningRecommendation]:
        """Generate learning-based recommendations"""
        recommendations = []

        # Specialization recommendations
        for opportunity in specialization_opportunities[:3]:  # Top 3
            rec = LearningRecommendation(
                type="specialize",
                priority="high"
                if opportunity.improvement_potential > 0.2
                else "medium",
                context=opportunity.context_key,
                description=f"Specialize rule {opportunity.rule_id} for context {opportunity.context_key}",
                expected_impact=opportunity.improvement_potential,
                implementation_effort="medium",
                supporting_evidence=[
                    f"Current performance: {opportunity.current_performance:.3f}",
                    f"Potential improvement: {opportunity.improvement_potential:.3f}",
                    f"Confidence: {opportunity.confidence:.3f}",
                ],
            )
            recommendations.append(rec)

        # Generalization recommendations
        poor_contexts = [
            (key, insight)
            for key, insight in context_insights.items()
            if insight.avg_performance < 0.6
            and insight.sample_size >= self.config.min_sample_size * 2
        ]

        for context_key, insight in poor_contexts[:2]:  # Top 2 poor performers
            rec = LearningRecommendation(
                type="generalize",
                priority="medium",
                context=context_key,
                description=f"Improve general rule effectiveness for context {context_key}",
                expected_impact=0.8 - insight.avg_performance,  # Target 0.8 performance
                implementation_effort="low",
                supporting_evidence=[
                    f"Below-average performance: {insight.avg_performance:.3f}",
                    f"Sufficient sample size: {insight.sample_size}",
                    "Consistency issues"
                    if insight.consistency_score < 0.6
                    else "Consistent poor performance",
                ],
            )
            recommendations.append(rec)

        # New rule creation recommendations
        high_potential_contexts = [
            (key, insight)
            for key, insight in context_insights.items()
            if insight.specialization_potential > 0.6
            and len(insight.top_performing_rules) < 3
        ]

        for context_key, insight in high_potential_contexts[:2]:
            rec = LearningRecommendation(
                type="create_new",
                priority="low",
                context=context_key,
                description=f"Create context-specific rules for {context_key}",
                expected_impact=insight.specialization_potential * 0.3,
                implementation_effort="high",
                supporting_evidence=[
                    f"High specialization potential: {insight.specialization_potential:.3f}",
                    f"Limited effective rules: {len(insight.top_performing_rules)}",
                    f"Unique patterns: {', '.join(insight.unique_patterns[:3])}",
                ],
            )
            recommendations.append(rec)

        # Sort by priority and expected impact
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda x: (priority_order[x.priority], x.expected_impact), reverse=True
        )

        return recommendations[:10]  # Limit to top 10 recommendations

    # ==================== PHASE 2 ENHANCEMENTS ====================

    def _implement_in_context_learning(
        self,
        context_data: list[dict[str, Any]],
        user_preferences: dict[str, Any],
        task_examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """2025 In-Context Learning framework for personalized prompt optimization.

        Key insights from research:
        - ICL operates without parameter updates
        - Privacy-preserving through federated approaches
        - Adaptive few-shot learning based on user patterns
        """
        self.logger.info("Implementing in-context learning framework")

        if not self.config.enable_in_context_learning:
            return {"status": "disabled", "demonstrations": []}

        # Context-aware demonstration selection
        demonstrations = self._select_contextual_demonstrations(
            context_data, task_examples, user_preferences
        )

        # Contextual bandit for personalization
        bandit_recommendations = self._apply_contextual_bandit(
            demonstrations, user_preferences, context_data
        )

        # Privacy-preserving adaptation with differential privacy
        privacy_metrics = None
        if self.config.privacy_preserving:
            original_count = len(demonstrations)
            demonstrations = self._apply_differential_privacy(
                demonstrations, self.config.differential_privacy_epsilon
            )
            privacy_metrics = {
                "epsilon_spent": self.config.differential_privacy_epsilon,
                "noise_added": True,
                "original_demonstrations": original_count,
                "protected_demonstrations": len(demonstrations),
                "privacy_method": "differential_privacy",
            }

        # Generate personalization improvements based on bandit recommendations
        personalization_improvements = self._generate_personalization_improvements(
            bandit_recommendations, demonstrations, user_preferences
        )

        # Generate context-specific recommendations
        context_specific_recommendations = self._generate_context_recommendations(
            context_data, demonstrations
        )

        result = {
            "status": "success",
            "demonstrations": demonstrations,
            "bandit_recommendations": bandit_recommendations,
            "privacy_applied": self.config.privacy_preserving,
            "demonstration_count": len(demonstrations),
            "personalization_improvements": personalization_improvements,
            "context_specific_recommendations": context_specific_recommendations,
        }

        if privacy_metrics:
            result["privacy_metrics"] = privacy_metrics

        return result

    def _generate_personalization_improvements(
        self,
        bandit_recommendations: dict[str, Any],
        demonstrations: list[dict[str, Any]],
        user_preferences: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate personalization improvements based on contextual bandit analysis"""
        improvements = {
            "adaptive_strategies": [],
            "user_specific_patterns": {},
            "confidence_score": 0.0,
            "exploration_benefits": [],
        }

        if not bandit_recommendations or not bandit_recommendations.get(
            "recommendations"
        ):
            return improvements

        # Extract top recommendations
        top_recs = bandit_recommendations["recommendations"][:3]

        # Generate adaptive strategies
        for i, rec in enumerate(top_recs):
            strategy = {
                "strategy_id": f"adaptive_{i}",
                "expected_reward": rec.get("expected_reward", 0.5),
                "confidence": rec.get("confidence", 0.5),
                "description": f"Personalized approach based on demonstration {rec.get('demonstration_index', i)}",
            }
            improvements["adaptive_strategies"].append(strategy)

        # User-specific patterns
        if user_preferences:
            improvements["user_specific_patterns"] = {
                "preference_alignment": min(1.0, len(user_preferences) / 5.0),
                "personalization_depth": "medium" if len(demonstrations) > 3 else "low",
                "adaptation_potential": bandit_recommendations.get(
                    "exploration_score", 0.5
                ),
            }

        # Overall confidence
        if top_recs:
            improvements["confidence_score"] = np.mean([
                rec.get("confidence", 0.5) for rec in top_recs
            ])

        # Exploration benefits
        exploration_score = bandit_recommendations.get("exploration_score", 0.0)
        if exploration_score > 0.3:
            improvements["exploration_benefits"].append(
                "High exploration potential identified"
            )
        if len(top_recs) > 1:
            improvements["exploration_benefits"].append(
                "Multiple promising approaches available"
            )

        return improvements

    def _generate_context_recommendations(
        self, context_data: list[dict[str, Any]], demonstrations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate context-specific recommendations"""
        recommendations = []

        if not context_data or not demonstrations:
            return recommendations

        # Extract unique contexts
        contexts = set()
        for ctx in context_data:
            if isinstance(ctx, dict) and "context" in ctx:
                contexts.add(ctx["context"])
            elif isinstance(ctx, str):
                contexts.add(ctx)

        # Generate recommendations for each context
        for context in list(contexts)[:3]:  # Limit to top 3 contexts
            rec = {
                "context": context,
                "recommendation_type": "optimization",
                "priority": "medium",
                "description": f"Optimize prompts for {context} context",
                "estimated_impact": min(0.8, 0.5 + len(demonstrations) * 0.1),
            }
            recommendations.append(rec)

        return recommendations

    def _select_contextual_demonstrations(
        self,
        context_data: list[dict[str, Any]],
        task_examples: list[dict[str, Any]],
        user_preferences: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Select most relevant demonstrations using cosine similarity"""
        if not task_examples:
            return []

        # Extract context features for similarity computation
        context_texts = []
        target_contexts = set()
        for ctx in context_data:
            text_features = []
            # Handle both dict and string context formats
            if isinstance(ctx, dict):
                if "context" in ctx:
                    target_contexts.add(ctx["context"])
                    text_features.append(ctx["context"])
                if "projectType" in ctx:
                    text_features.append(f"project_{ctx['projectType']}")
                if "domain" in ctx:
                    text_features.append(f"domain_{ctx['domain']}")
                if "complexity" in ctx:
                    text_features.append(f"complexity_{ctx['complexity']}")
            elif isinstance(ctx, str):
                target_contexts.add(ctx)
                text_features.append(ctx)
            context_texts.append(" ".join(text_features))

        # First filter by exact context match if we have target contexts
        if target_contexts:
            filtered_examples = [
                ex for ex in task_examples if ex.get("context") in target_contexts
            ]
            if filtered_examples:
                task_examples = filtered_examples

        if not context_texts:
            return task_examples[: self.config.icl_demonstrations]

        # Vectorize contexts and examples
        try:
            if self.context_vectorizer is None:
                self.context_vectorizer = TfidfVectorizer(
                    max_features=500, stop_words="english"
                )

            # Combine context and example texts for vectorization
            example_texts = []
            for example in task_examples:
                example_text = (
                    example.get("originalPrompt", "")
                    + " "
                    + example.get("improvedPrompt", "")
                )
                example_texts.append(example_text)

            all_texts = context_texts + example_texts
            if len(all_texts) > 1:
                vectors = self.context_vectorizer.fit_transform(all_texts)

                # Compute similarities between contexts and examples
                context_vectors = vectors[: len(context_texts)]
                example_vectors = vectors[len(context_texts) :]

                if context_vectors.shape[0] > 0 and example_vectors.shape[0] > 0:
                    similarities = cosine_similarity(context_vectors, example_vectors)

                    # Select top demonstrations based on similarity
                    avg_similarities = np.mean(similarities, axis=0)
                    top_indices = np.argsort(avg_similarities)[
                        -self.config.icl_demonstrations :
                    ]

                    selected_demonstrations = [
                        task_examples[i]
                        for i in top_indices
                        if avg_similarities[i] > self.config.icl_similarity_threshold
                    ]

                    self.logger.info(
                        f"Selected {len(selected_demonstrations)} contextual demonstrations"
                    )
                    return selected_demonstrations

        except Exception as e:
            self.logger.warning(f"Error in demonstration selection: {e}")

        # Fallback to top examples
        return task_examples[: self.config.icl_demonstrations]

    def _apply_contextual_bandit(
        self,
        demonstrations: list[dict[str, Any]],
        user_preferences: dict[str, Any],
        context_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply Thompson Sampling for exploration-exploitation balance"""
        if not demonstrations:
            return {
                "recommendations": [],
                "exploration_score": 0.0,
                "selected_action": 0,
                "confidence_interval": {"lower": 0.0, "upper": 1.0},
                "exploration_bonus": 1.0,
                "method": "random_fallback",
            }

        # Thompson Sampling implementation for contextual bandits
        # This is a simplified version - in production, you'd use a more sophisticated approach

        # Extract performance scores from demonstrations
        performance_scores = []
        for demo in demonstrations:
            score = demo.get("overallScore", demo.get("improvementScore", 0.5))
            performance_scores.append(score)

        if not performance_scores:
            return {"recommendations": [], "exploration_score": 0.0}

        # Beta distribution parameters for Thompson Sampling
        successes = [max(1, int(score * 10)) for score in performance_scores]
        failures = [max(1, int((1 - score) * 10)) for score in performance_scores]

        # Sample from Beta distributions
        sampled_rewards = []
        for i in range(len(demonstrations)):
            sampled_reward = np.random.beta(successes[i], failures[i])
            sampled_rewards.append(sampled_reward)

        # Rank demonstrations by sampled rewards
        ranked_indices = np.argsort(sampled_rewards)[::-1]

        recommendations = []
        for i in ranked_indices:
            recommendations.append({
                "demonstration_index": i,
                "expected_reward": sampled_rewards[i],
                "confidence": successes[i] / (successes[i] + failures[i]),
                "exploration_value": 1.0 / (successes[i] + failures[i]),
            })

        exploration_score = np.mean([
            rec["exploration_value"] for rec in recommendations
        ])

        # Select best action and compute confidence interval
        if recommendations:
            best_rec = recommendations[0]
            best_index = best_rec["demonstration_index"]
            confidence = best_rec["confidence"]

            # Compute confidence interval
            margin = 0.1 * (1 - confidence)  # Larger margin for lower confidence
            lower_bound = max(0.0, best_rec["expected_reward"] - margin)
            upper_bound = min(1.0, best_rec["expected_reward"] + margin)

            selected_action = best_index
            confidence_interval = {"lower": lower_bound, "upper": upper_bound}
            exploration_bonus = best_rec["exploration_value"]
        else:
            selected_action = 0
            confidence_interval = {"lower": 0.0, "upper": 1.0}
            exploration_bonus = 1.0

        return {
            "recommendations": recommendations,
            "exploration_score": exploration_score,
            "method": "thompson_sampling",
            "selected_action": selected_action,
            "confidence_interval": confidence_interval,
            "exploration_bonus": exploration_bonus,
        }

    def _apply_differential_privacy(
        self, data, epsilon: float | None = None
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Apply differential privacy using Laplace mechanism

        Args:
            data: Either List[Dict] (demonstrations) or List[float] (scores)
            epsilon: Privacy parameter (uses config default if None)

        Returns:
            For List[Dict]: Modified demonstrations (backward compatibility)
            For List[float]: Dict with privacy metrics (test interface)
        """
        if epsilon is None:
            epsilon = getattr(self.config, "differential_privacy_epsilon", 1.0)

        # Handle disabled privacy
        if not getattr(self.config, "privacy_preserving", True):
            if isinstance(data, list) and data and isinstance(data[0], (int, float)):
                return {
                    "noisy_scores": data,
                    "privacy_budget_used": 0.0,
                    "privacy_guarantee": "disabled",
                }
            return data

        # Handle score list (test interface) - follows PyDP/diffprivlib patterns
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            noisy_scores = []
            for score in data:
                # Apply Laplace noise (standard DP mechanism)
                noise = np.random.laplace(0, 1.0 / epsilon)
                noisy_score = np.clip(float(score) + noise, 0.0, 1.0)
                noisy_scores.append(noisy_score)

            self.logger.info(
                f"Applied differential privacy to {len(data)} scores with epsilon={epsilon}"
            )
            return {
                "noisy_scores": noisy_scores,
                "privacy_budget_used": epsilon,
                "privacy_guarantee": f"({epsilon})-differential-privacy",
            }

        # Handle demonstration objects (backward compatibility)
        demonstrations = data if isinstance(data, list) else []
        if not demonstrations:
            return demonstrations

        # Apply Laplace noise to sensitive numerical values in demonstrations
        for demo in demonstrations:
            if "overallScore" in demo:
                noise = np.random.laplace(0, 1.0 / epsilon)
                demo["overallScore"] = np.clip(demo["overallScore"] + noise, 0, 1)

            if "improvementScore" in demo:
                noise = np.random.laplace(0, 1.0 / epsilon)
                demo["improvementScore"] = np.clip(
                    demo["improvementScore"] + noise, 0, 1
                )

            # Also handle score field if present
            if "score" in demo and isinstance(demo["score"], (int, float)):
                noise = np.random.laplace(0, 1.0 / epsilon)
                demo["score"] = np.clip(demo["score"] + noise, 0, 1)

        self.logger.info(
            f"Applied differential privacy to {len(demonstrations)} demonstrations with epsilon={epsilon}"
        )
        return demonstrations

    # ==================== PHASE 3 ENHANCEMENTS ====================

    def _initialize_privacy_components(self) -> None:
        """Initialize Phase 3 privacy-preserving ML components"""
        try:
            # Initialize privacy budget tracking
            self.privacy_budget_used = 0.0

            # Initialize secure aggregation if enabled
            if self.config.secure_aggregation and CRYPTO_AVAILABLE:
                self.encryption_key = Fernet.generate_key()
                self.logger.info("Secure aggregation initialized with encryption")

            # Initialize Opacus privacy engine if available
            if OPACUS_AVAILABLE and self.config.enable_federated_learning:
                self.privacy_engine = PrivacyEngine()
                self.logger.info("Opacus privacy engine initialized")

            self.logger.info(
                "Privacy-preserving ML components initialized successfully"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize privacy components: {e}")
            self.config.privacy_preserving = False

    def _implement_federated_learning(
        self,
        context_data: list[dict[str, Any]],
        client_data: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Implement federated learning for privacy-preserving personalization"""
        if not self.config.enable_federated_learning:
            return {"status": "disabled", "reason": "Federated learning not enabled"}

        if len(client_data) < self.config.federated_min_clients:
            return {
                "status": "insufficient_clients",
                "clients_available": len(client_data),
            }

        self.logger.info(f"Starting federated learning with {len(client_data)} clients")

        federated_results = {
            "status": "success",
            "rounds_completed": 0,
            "participating_clients": len(client_data),
            "privacy_budget_used": 0.0,
            "global_model_updates": [],
            "convergence_metrics": [],
        }

        try:
            # Federated learning rounds
            for round_num in range(self.config.federated_rounds):
                round_results = self._execute_federated_round(
                    round_num, client_data, context_data
                )

                if round_results["status"] == "success":
                    federated_results["rounds_completed"] += 1
                    federated_results["global_model_updates"].append(
                        round_results["global_update"]
                    )
                    federated_results["convergence_metrics"].append(
                        round_results["convergence_metric"]
                    )
                    federated_results["privacy_budget_used"] += round_results[
                        "privacy_cost"
                    ]

                    # Check privacy budget
                    if (
                        federated_results["privacy_budget_used"]
                        >= self.config.max_privacy_budget
                    ):
                        self.logger.warning(
                            f"Privacy budget exhausted at round {round_num}"
                        )
                        break
                else:
                    self.logger.error(
                        f"Federated round {round_num} failed: {round_results['error']}"
                    )
                    break

            # Update global privacy budget
            self.privacy_budget_used += federated_results["privacy_budget_used"]

            self.logger.info(
                f"Federated learning completed: {federated_results['rounds_completed']} rounds"
            )
            return federated_results

        except Exception as e:
            self.logger.error(f"Federated learning failed: {e}")
            return {"status": "error", "error": str(e)}

    def _execute_federated_round(
        self,
        round_num: int,
        client_data: dict[str, list[dict[str, Any]]],
        context_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute a single federated learning round"""
        client_updates = []
        privacy_costs = []

        # Client local training
        for client_id, client_examples in client_data.items():
            if len(client_examples) < self.config.min_sample_size:
                continue

            # Local model update with differential privacy
            local_update = self._compute_local_update(client_examples, context_data)

            # Apply differential privacy to local update
            if self.config.privacy_preserving:
                private_update, privacy_cost = (
                    self._apply_advanced_differential_privacy(local_update)
                )
                client_updates.append(private_update)
                privacy_costs.append(privacy_cost)
            else:
                client_updates.append(local_update)
                privacy_costs.append(0.0)

        if not client_updates:
            return {
                "status": "no_valid_clients",
                "error": "No clients with sufficient data",
            }

        # Secure aggregation
        if self.config.secure_aggregation:
            global_update = self._secure_aggregate_updates(client_updates)
        else:
            global_update = self._simple_aggregate_updates(client_updates)

        # Compute convergence metric
        convergence_metric = self._compute_convergence_metric(client_updates)

        return {
            "status": "success",
            "round": round_num,
            "participating_clients": len(client_updates),
            "global_update": global_update,
            "convergence_metric": convergence_metric,
            "privacy_cost": sum(privacy_costs),
        }

    def _compute_local_update(
        self, client_examples: list[dict[str, Any]], context_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute local model update for a client"""
        # Extract features and performance scores
        features = []
        scores = []

        for example in client_examples:
            if "overallScore" in example or "improvementScore" in example:
                # Use context features for local learning
                context_features = self._extract_context_features(
                    example.get("context", {})
                )
                features.append(context_features)
                scores.append(
                    example.get("overallScore", example.get("improvementScore", 0.5))
                )

        if not features:
            return {"gradient": np.zeros(10), "sample_count": 0}

        features_array = np.array(features)
        scores_array = np.array(scores)

        # Simple gradient computation (placeholder for actual ML model)
        # In practice, this would be gradients from a neural network or other ML model
        gradient = np.mean(features_array * scores_array.reshape(-1, 1), axis=0)

        return {
            "gradient": gradient,
            "sample_count": len(features),
            "mean_score": np.mean(scores_array),
        }

    def _extract_context_features(self, context: dict[str, Any]) -> np.ndarray:
        """Extract numerical features from context"""
        # Simple feature extraction - in practice would be more sophisticated
        features = []

        # Project type encoding
        project_types = ["web", "mobile", "desktop", "api", "ml", "data"]
        project_type = context.get("projectType", "web").lower()
        for pt in project_types:
            features.append(1.0 if pt == project_type else 0.0)

        # Add complexity and team size
        complexity_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "very_high": 1.0}
        features.append(complexity_map.get(context.get("complexity", "medium"), 0.5))

        team_size = context.get("teamSize", 5)
        if isinstance(team_size, str):
            team_size = {"small": 3, "medium": 8, "large": 15}.get(team_size.lower(), 8)
        features.append(min(team_size / 20.0, 1.0))

        # Pad to fixed size
        while len(features) < 10:
            features.append(0.0)

        return np.array(features[:10])

    def _apply_advanced_differential_privacy(
        self, local_update: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """Apply advanced differential privacy with Gaussian noise"""
        if not OPACUS_AVAILABLE:
            # Fallback to simple Laplace noise
            return self._apply_laplace_noise(local_update)

        # Use Opacus-style Gaussian noise for better privacy guarantees
        gradient = local_update["gradient"]
        sample_count = local_update["sample_count"]

        # Clip gradient norm
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > self.config.max_grad_norm:
            gradient = gradient * (self.config.max_grad_norm / gradient_norm)

        # Add Gaussian noise
        noise_scale = self.config.noise_multiplier * self.config.max_grad_norm
        noise = np.random.normal(0, noise_scale, gradient.shape)

        private_gradient = gradient + noise

        # Compute privacy cost (simplified)
        privacy_cost = (sample_count * self.config.differential_privacy_epsilon) / 100.0

        private_update = {
            "gradient": private_gradient,
            "sample_count": sample_count,
            "mean_score": local_update["mean_score"],
            "privacy_applied": True,
        }

        return private_update, privacy_cost

    def _apply_laplace_noise(
        self, local_update: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """Fallback Laplace noise application"""
        gradient = local_update["gradient"]

        # Apply Laplace noise to gradient
        noise = np.random.laplace(
            0, 1.0 / self.config.differential_privacy_epsilon, gradient.shape
        )
        private_gradient = gradient + noise

        private_update = {
            "gradient": private_gradient,
            "sample_count": local_update["sample_count"],
            "mean_score": local_update["mean_score"],
            "privacy_applied": True,
        }

        privacy_cost = self.config.differential_privacy_epsilon / 10.0
        return private_update, privacy_cost

    def _secure_aggregate_updates(
        self, client_updates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Securely aggregate client updates using encryption"""
        if not self.config.secure_aggregation or not CRYPTO_AVAILABLE:
            return self._simple_aggregate_updates(client_updates)

        try:
            # Encrypt individual updates
            fernet = Fernet(self.encryption_key)
            encrypted_updates = []

            for update in client_updates:
                # Serialize and encrypt gradient
                gradient_bytes = update["gradient"].tobytes()
                encrypted_gradient = fernet.encrypt(gradient_bytes)

                encrypted_updates.append({
                    "encrypted_gradient": encrypted_gradient,
                    "sample_count": update["sample_count"],
                    "mean_score": update["mean_score"],
                })

            # Decrypt and aggregate (in practice, this would use secure multi-party computation)
            decrypted_gradients = []
            total_samples = 0
            total_weighted_score = 0.0

            for enc_update in encrypted_updates:
                decrypted_gradient_bytes = fernet.decrypt(
                    enc_update["encrypted_gradient"]
                )
                gradient = np.frombuffer(decrypted_gradient_bytes, dtype=np.float64)

                decrypted_gradients.append(gradient)
                total_samples += enc_update["sample_count"]
                total_weighted_score += (
                    enc_update["mean_score"] * enc_update["sample_count"]
                )

            # Aggregate gradients
            aggregated_gradient = np.mean(decrypted_gradients, axis=0)
            mean_score = (
                total_weighted_score / total_samples if total_samples > 0 else 0.0
            )

            return {
                "aggregated_gradient": aggregated_gradient,
                "total_samples": total_samples,
                "mean_score": mean_score,
                "aggregation_method": "secure",
            }

        except Exception as e:
            self.logger.error(f"Secure aggregation failed: {e}")
            return self._simple_aggregate_updates(client_updates)

    def _simple_aggregate_updates(
        self, client_updates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Simple aggregation of client updates"""
        if not client_updates:
            return {
                "aggregated_gradient": np.zeros(10),
                "total_samples": 0,
                "mean_score": 0.0,
            }

        gradients = [update["gradient"] for update in client_updates]
        sample_counts = [update["sample_count"] for update in client_updates]
        scores = [update["mean_score"] for update in client_updates]

        # Weighted average by sample count
        total_samples = sum(sample_counts)
        if total_samples == 0:
            return {
                "aggregated_gradient": np.zeros(10),
                "total_samples": 0,
                "mean_score": 0.0,
            }

        weighted_gradient = np.zeros_like(gradients[0])
        weighted_score = 0.0

        for gradient, count, score in zip(
            gradients, sample_counts, scores, strict=False
        ):
            weight = count / total_samples
            weighted_gradient += weight * gradient
            weighted_score += weight * score

        return {
            "aggregated_gradient": weighted_gradient,
            "total_samples": total_samples,
            "mean_score": weighted_score,
            "aggregation_method": "simple",
        }

    def _compute_convergence_metric(
        self, client_updates: list[dict[str, Any]]
    ) -> float:
        """Compute convergence metric for federated learning"""
        if len(client_updates) < 2:
            return 1.0  # Single client - assume converged

        gradients = [update["gradient"] for update in client_updates]

        # Compute variance of gradients as convergence metric
        # Lower variance indicates better convergence
        gradient_matrix = np.array(gradients)
        gradient_variance = np.var(gradient_matrix, axis=0)
        mean_variance = np.mean(gradient_variance)

        # Convert to convergence score (0 = not converged, 1 = fully converged)
        convergence_score = 1.0 / (1.0 + mean_variance)

        return convergence_score

    def _perform_advanced_clustering(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Replace K-means with UMAP + HDBSCAN for better context clustering.

        Key advantages:
        - HDBSCAN: Density-based clustering handling variable density
        - UMAP: Preserves both local and global structure
        - Quality assessment: Silhouette score and Calinski-Harabasz index
        """
        if not self.config.use_advanced_clustering or not ADVANCED_CLUSTERING_AVAILABLE:
            return self._fallback_to_kmeans(results)

        self.logger.info("Performing advanced clustering with UMAP + HDBSCAN")

        # Extract features for clustering
        features = self._extract_clustering_features(results)
        if features is None or len(features) < self.config.min_sample_size:
            self.logger.warning(
                f"Insufficient data for clustering: {len(features) if features is not None else 0} samples"
            )
            return {"status": "insufficient_data", "clusters": {}}

        try:
            # Validate and adjust UMAP parameters based on dataset size
            n_samples, n_features = features.shape

            # Best practice: Ensure n_neighbors >= n_components and reasonable for dataset size
            adjusted_n_neighbors = min(
                self.config.umap_n_neighbors, max(2, n_samples // 3)
            )
            adjusted_n_components = min(
                self.config.umap_n_components, n_features, adjusted_n_neighbors - 1
            )

            # Ensure min_dist is appropriate for the data scale
            adjusted_min_dist = max(0.0, min(self.config.umap_min_dist, 0.99))

            self.logger.info(
                f"UMAP parameters: n_components={adjusted_n_components}, n_neighbors={adjusted_n_neighbors}, min_dist={adjusted_min_dist}"
            )

            # Step 1: UMAP dimensionality reduction with validated parameters
            self.umap_reducer = umap.UMAP(
                n_components=adjusted_n_components,
                n_neighbors=adjusted_n_neighbors,
                min_dist=adjusted_min_dist,
                metric="euclidean",  # Consistent with HDBSCAN
                random_state=42,
                verbose=False,  # Reduce noise in logs
            )

            reduced_features = self.umap_reducer.fit_transform(features)

            # Validate HDBSCAN parameters based on dataset size
            # Best practice: min_cluster_size should be reasonable relative to dataset
            adjusted_min_cluster_size = min(
                self.config.hdbscan_min_cluster_size, max(2, n_samples // 10)
            )
            adjusted_min_samples = min(
                self.config.hdbscan_min_samples, adjusted_min_cluster_size
            )

            self.logger.info(
                f"HDBSCAN parameters: min_cluster_size={adjusted_min_cluster_size}, min_samples={adjusted_min_samples}"
            )

            # Step 2: HDBSCAN clustering with validated parameters
            self.hdbscan_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=adjusted_min_cluster_size,
                min_samples=adjusted_min_samples,
                metric="euclidean",
                cluster_selection_method="eom",  # Excess of Mass method
                algorithm="best",  # Auto-select best algorithm
                leaf_size=40,  # Balanced performance
                core_dist_n_jobs=1,  # Consistent execution
            )

            cluster_labels = self.hdbscan_clusterer.fit_predict(reduced_features)

            # Enhanced quality assessment with multiple metrics
            quality_score = self._assess_clustering_quality(
                reduced_features, cluster_labels
            )
            noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

            self.clustering_quality_score = quality_score

            # Additional quality checks based on research best practices
            if n_clusters == 0:
                self.logger.warning(
                    "HDBSCAN found no clusters, falling back to K-means"
                )
                return self._fallback_to_kmeans(results)

            if noise_ratio > 0.5:  # More than 50% noise points
                self.logger.warning(
                    f"High noise ratio ({noise_ratio:.2f}), clustering may be poor quality"
                )
                if (
                    quality_score < self.config.clustering_quality_threshold * 0.8
                ):  # More lenient threshold
                    return self._fallback_to_kmeans(results)

            # Group results by clusters
            clusters = self._group_by_clusters(results, cluster_labels)

            clustering_result = {
                "status": "success",
                "method": "umap_hdbscan",
                "n_clusters": n_clusters,
                "n_noise_points": int(np.sum(cluster_labels == -1)),
                "noise_ratio": noise_ratio,
                "quality_score": quality_score,
                "clusters": clusters,
                "reduced_dimensions": reduced_features.shape[1],
                "cluster_sizes": [
                    np.sum(cluster_labels == i) for i in range(n_clusters)
                ],
                "parameters": {
                    "umap_n_components": adjusted_n_components,
                    "umap_n_neighbors": adjusted_n_neighbors,
                    "umap_min_dist": adjusted_min_dist,
                    "hdbscan_min_cluster_size": adjusted_min_cluster_size,
                    "hdbscan_min_samples": adjusted_min_samples,
                },
            }

            if quality_score < self.config.clustering_quality_threshold:
                self.logger.warning(
                    f"Clustering quality {quality_score:.3f} below threshold {self.config.clustering_quality_threshold}"
                )
                return self._fallback_to_kmeans(results)

            self.logger.info(
                f"Advanced clustering completed: {n_clusters} clusters, "
                f"quality={quality_score:.3f}, noise_ratio={noise_ratio:.3f}"
            )
            return clustering_result

        except Exception as e:
            self.logger.error(f"Advanced clustering failed: {e}", exc_info=True)
            return self._fallback_to_kmeans(results)

    def _extract_clustering_features(
        self, results: list[dict[str, Any]]
    ) -> np.ndarray | None:
        """Extract numerical features for clustering with enhanced linguistic and domain-specific analysis

        Feature vector composition:
        - Performance metrics: 5 features
        - Linguistic features: 10 features (when enabled)
        - Domain-specific features: 15 features (when enabled)
        - Context features: ~16 features (project type, team size, complexity, etc.)
        Total: ~46 features when all enhancements are enabled
        """
        if not results:
            return None

        features = []
        for result in results:
            feature_vector = []

            # Performance metrics (5 features)
            feature_vector.append(result.get("overallScore", 0.5))
            feature_vector.append(result.get("clarity", 0.5))
            feature_vector.append(result.get("completeness", 0.5))
            feature_vector.append(result.get("actionability", 0.5))
            feature_vector.append(result.get("effectiveness", 0.5))

            # Extract linguistic features if enabled (10 additional features)
            if self.config.enable_linguistic_features and self.linguistic_analyzer:
                linguistic_features = self._extract_linguistic_features(result)
                feature_vector.extend(linguistic_features)
            else:
                # Add placeholder zeros for linguistic features to maintain consistent vector size
                feature_vector.extend([0.0] * 10)

            # Extract domain-specific features if enabled (15 additional features)
            if self.config.enable_domain_features and self.domain_feature_extractor:
                domain_features = self._extract_domain_features(result)
                feature_vector.extend(domain_features)
            else:
                # Add placeholder zeros for domain features to maintain consistent vector size
                feature_vector.extend([0.0] * 15)

            # Context features (encoded)
            context = result.get("context", {})

            # Project type encoding
            project_types = ["web", "mobile", "desktop", "api", "ml", "data", "other"]

            # Handle both string and dict context formats
            if isinstance(context, str):
                # Map string context to project type categories
                context_str = context.lower()
                if any(
                    term in context_str
                    for term in ["technical", "documentation", "api", "code"]
                ):
                    project_type = "api"
                elif any(
                    term in context_str for term in ["creative", "writing", "content"]
                ):
                    project_type = "web"
                elif any(
                    term in context_str
                    for term in ["business", "communication", "meeting"]
                ):
                    project_type = "other"
                else:
                    project_type = "other"
            else:
                # Dictionary context format
                project_type = context.get("projectType", "other").lower()
            project_encoding = [
                1.0 if pt == project_type else 0.0 for pt in project_types
            ]
            feature_vector.extend(project_encoding)

            # Domain encoding
            domains = [
                "finance",
                "healthcare",
                "education",
                "ecommerce",
                "gaming",
                "social",
                "other",
            ]
            if isinstance(context, str):
                # Default domain mapping for string contexts
                context_str = context.lower()
                if any(
                    term in context_str for term in ["business", "finance", "meeting"]
                ):
                    domain = "finance"
                elif any(
                    term in context_str
                    for term in ["education", "learning", "documentation"]
                ):
                    domain = "education"
                else:
                    domain = "other"
            else:
                domain = context.get("domain", "other").lower()
            domain_encoding = [1.0 if d == domain else 0.0 for d in domains]
            feature_vector.extend(domain_encoding)

            # Complexity encoding
            complexity_map = {
                "low": 0.25,
                "medium": 0.5,
                "high": 0.75,
                "very_high": 1.0,
            }
            if isinstance(context, str):
                # Default complexity for string contexts
                if any(term in context_str for term in ["technical", "documentation"]):
                    complexity = 0.75  # high
                elif any(term in context_str for term in ["creative", "writing"]):
                    complexity = 0.5  # medium
                else:
                    complexity = 0.5  # medium
            else:
                complexity = complexity_map.get(
                    context.get("complexity", "medium"), 0.5
                )
            feature_vector.append(complexity)

            # Team size (normalized)
            if isinstance(context, str):
                team_size = 5  # Default team size for string contexts
            else:
                team_size = context.get("teamSize", 5)
                if isinstance(team_size, str):
                    team_size = {"small": 3, "medium": 8, "large": 15}.get(
                        team_size.lower(), 8
                    )
            feature_vector.append(min(team_size / 20.0, 1.0))  # Normalize to [0,1]

            # Apply context-aware feature weighting if enabled
            if (
                self.config.enable_context_aware_weighting
                and self.context_aware_weighter
                and self.config.enable_domain_features
                and self.domain_feature_extractor
            ):
                try:
                    # Get prompt text for domain analysis
                    prompt_text = result.get("originalPrompt", "") or result.get(
                        "prompt", ""
                    )
                    if prompt_text and isinstance(prompt_text, str):
                        # Extract domain features for weighting calculation
                        domain_features = (
                            self.domain_feature_extractor.extract_domain_features(
                                prompt_text
                            )
                        )

                        # Create feature names list (matching the order of feature_vector)
                        feature_names = self._get_feature_names()

                        # Calculate context-aware weights
                        weights = self.context_aware_weighter.calculate_feature_weights(
                            domain_features, tuple(feature_names)
                        )

                        # Apply weights to feature vector
                        if len(weights) == len(feature_vector):
                            feature_vector = [
                                f * w
                                for f, w in zip(feature_vector, weights, strict=False)
                            ]
                        else:
                            # Fallback: pad or trim weights to match feature vector size
                            if len(weights) < len(feature_vector):
                                weights = np.pad(
                                    weights,
                                    (0, len(feature_vector) - len(weights)),
                                    constant_values=1.0,
                                )
                            else:
                                weights = weights[: len(feature_vector)]
                            feature_vector = [
                                f * w
                                for f, w in zip(feature_vector, weights, strict=False)
                            ]

                        self.logger.debug(
                            f"Applied context-aware weighting for {domain_features.domain.value} domain"
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Context-aware weighting failed, using unweighted features: {e}"
                    )

            features.append(feature_vector)

        return np.array(features)

    def _get_feature_names(self) -> list[str]:
        """Get the names of features in the order they appear in the feature vector."""
        feature_names = []

        # Performance metrics (5 features)
        feature_names.extend([
            "overall_score",
            "clarity",
            "completeness",
            "actionability",
            "effectiveness",
        ])

        # Linguistic features (10 features)
        if self.config.enable_linguistic_features:
            feature_names.extend([
                "linguistic_readability",
                "linguistic_lexical_diversity",
                "linguistic_entity_density",
                "linguistic_syntactic_complexity",
                "linguistic_sentence_structure",
                "linguistic_technical_ratio",
                "linguistic_avg_sentence_length",
                "linguistic_instruction_clarity",
                "linguistic_has_examples",
                "linguistic_overall_quality",
            ])
        else:
            feature_names.extend([f"linguistic_placeholder_{i}" for i in range(10)])

        # Domain-specific features (15 features)
        if self.config.enable_domain_features:
            feature_names.extend([
                "domain_confidence",
                "domain_complexity",
                "domain_specificity",
                "technical_domain_indicator",
                "creative_domain_indicator",
                "academic_domain_indicator",
                "business_domain_indicator",
                "technical_feature_density",
                "creative_feature_density",
                "academic_feature_density",
                "conversational_politeness",
                "urgency_indicator",
                "question_density",
                "instruction_clarity",
                "domain_hybrid_indicator",
            ])
        else:
            feature_names.extend([f"domain_placeholder_{i}" for i in range(15)])

        # Context features
        project_types = ["web", "mobile", "desktop", "api", "ml", "data", "other"]
        feature_names.extend([f"project_type_{pt}" for pt in project_types])

        domains = [
            "finance",
            "healthcare",
            "education",
            "ecommerce",
            "gaming",
            "social",
            "other",
        ]
        feature_names.extend([f"domain_{d}" for d in domains])

        feature_names.extend(["complexity", "team_size"])

        return feature_names

    def _extract_linguistic_features(self, result: dict[str, Any]) -> list[float]:
        """Extract linguistic features from prompt text for ML analysis

        Returns 10 normalized linguistic features:
        1. Readability score (0-1)
        2. Lexical diversity (0-1)
        3. Entity density (0-1)
        4. Syntactic complexity (0-1)
        5. Sentence structure quality (0-1)
        6. Technical term ratio (0-1)
        7. Average sentence length (normalized)
        8. Instruction clarity (0-1)
        9. Has examples (0/1)
        10. Overall linguistic quality (0-1)
        """
        # Get the prompt text for analysis
        prompt_text = result.get("originalPrompt", "") or result.get("prompt", "")
        if not prompt_text or not isinstance(prompt_text, str):
            # Return default feature vector if no text
            return [0.5] * 10

        # Create deterministic cache key including config for consistency
        cache_content = f"{prompt_text}|weight:{self.config.linguistic_feature_weight}"
        cache_key = hashlib.md5(cache_content.encode()).hexdigest()

        if self.config.cache_linguistic_analysis and cache_key in self.linguistic_cache:
            return self.linguistic_cache[cache_key]

        try:
            # Set random seed before analysis for deterministic results
            import random

            import numpy as np

            random.seed(42)
            np.random.seed(42)

            # Perform linguistic analysis
            linguistic_features = self.linguistic_analyzer.analyze(prompt_text)

            # Extract and normalize features
            features = []

            # 1. Readability score (already 0-1)
            features.append(min(1.0, max(0.0, linguistic_features.readability_score)))

            # 2. Lexical diversity (already 0-1)
            features.append(min(1.0, max(0.0, linguistic_features.lexical_diversity)))

            # 3. Entity density (normalize by text length)
            entity_density = len(linguistic_features.entities) / max(
                len(prompt_text.split()), 1
            )
            features.append(min(1.0, entity_density))

            # 4. Syntactic complexity (already 0-1)
            features.append(
                min(1.0, max(0.0, linguistic_features.syntactic_complexity))
            )

            # 5. Sentence structure quality (already 0-1)
            features.append(
                min(1.0, max(0.0, linguistic_features.sentence_structure_quality))
            )

            # 6. Technical term ratio
            technical_ratio = len(linguistic_features.technical_terms) / max(
                len(prompt_text.split()), 1
            )
            features.append(min(1.0, technical_ratio))

            # 7. Average sentence length (normalize by dividing by 50 - typical max)
            avg_sent_length = min(1.0, linguistic_features.avg_sentence_length / 50.0)
            features.append(avg_sent_length)

            # 8. Instruction clarity (already 0-1)
            features.append(
                min(1.0, max(0.0, linguistic_features.instruction_clarity_score))
            )

            # 9. Has examples (binary feature)
            features.append(1.0 if linguistic_features.has_examples else 0.0)

            # 10. Overall linguistic quality (already 0-1)
            features.append(
                min(1.0, max(0.0, linguistic_features.overall_linguistic_quality))
            )

            # Cache the result for future use
            if self.config.cache_linguistic_analysis:
                self.linguistic_cache[cache_key] = features

            # Apply linguistic feature weight for balanced integration
            weight = self.config.linguistic_feature_weight
            weighted_features = [f * weight + (1 - weight) * 0.5 for f in features]

            return weighted_features

        except Exception as e:
            # Log error and return neutral features if analysis fails
            self.logger.warning(f"Linguistic feature extraction failed: {e}")
            return [0.5] * 10

    def _extract_domain_features(self, result: dict[str, Any]) -> list[float]:
        """Extract domain-specific features from prompt text for ML analysis

        Returns 15 normalized domain-specific features:
        1. Domain confidence (0-1)
        2. Domain complexity score (0-1)
        3. Domain specificity score (0-1)
        4. Technical domain indicator (0/1)
        5. Creative domain indicator (0/1)
        6. Academic domain indicator (0/1)
        7. Business domain indicator (0/1)
        8. Technical feature density (0-1)
        9. Creative feature density (0-1)
        10. Academic feature density (0-1)
        11. Conversational politeness score (0-1)
        12. Urgency indicator (0/1)
        13. Question density (0-1)
        14. Instruction clarity (0-1)
        15. Domain hybrid indicator (0/1)
        """
        # Get the prompt text for analysis
        prompt_text = result.get("originalPrompt", "") or result.get("prompt", "")
        if not prompt_text or not isinstance(prompt_text, str):
            # Return default feature vector if no text
            return [0.5] * 15

        # Create deterministic cache key including config for consistency
        cache_content = (
            f"{prompt_text}|domain_weight:{self.config.domain_feature_weight}"
        )
        cache_key = hashlib.md5(cache_content.encode()).hexdigest()

        if self.config.cache_domain_analysis and cache_key in self.domain_cache:
            return self.domain_cache[cache_key]

        try:
            # Set random seed before analysis for deterministic results
            import random

            import numpy as np

            random.seed(42)
            np.random.seed(42)

            # Perform domain-specific feature extraction
            domain_features = self.domain_feature_extractor.extract_domain_features(
                prompt_text
            )

            # Extract and normalize features
            features = []

            # 1. Domain confidence (already 0-1)
            features.append(min(1.0, max(0.0, domain_features.confidence)))

            # 2. Domain complexity score (already 0-1)
            features.append(min(1.0, max(0.0, domain_features.complexity_score)))

            # 3. Domain specificity score (already 0-1)
            features.append(min(1.0, max(0.0, domain_features.specificity_score)))

            # 4-7. Domain type indicators (binary features)
            from ..analysis.domain_detector import PromptDomain

            # Technical domains
            technical_domains = {
                PromptDomain.SOFTWARE_DEVELOPMENT,
                PromptDomain.DATA_SCIENCE,
                PromptDomain.AI_ML,
                PromptDomain.WEB_DEVELOPMENT,
                PromptDomain.SYSTEM_ADMIN,
                PromptDomain.API_DOCUMENTATION,
            }
            features.append(1.0 if domain_features.domain in technical_domains else 0.0)

            # Creative domains
            creative_domains = {
                PromptDomain.CREATIVE_WRITING,
                PromptDomain.CONTENT_CREATION,
                PromptDomain.MARKETING,
                PromptDomain.STORYTELLING,
            }
            features.append(1.0 if domain_features.domain in creative_domains else 0.0)

            # Academic domains
            academic_domains = {
                PromptDomain.RESEARCH,
                PromptDomain.EDUCATION,
                PromptDomain.ACADEMIC_WRITING,
                PromptDomain.SCIENTIFIC,
            }
            features.append(1.0 if domain_features.domain in academic_domains else 0.0)

            # Business domains
            business_domains = {
                PromptDomain.BUSINESS_ANALYSIS,
                PromptDomain.PROJECT_MANAGEMENT,
                PromptDomain.CUSTOMER_SERVICE,
                PromptDomain.SALES,
            }
            features.append(1.0 if domain_features.domain in business_domains else 0.0)

            # 8-10. Feature density scores (normalize by feature vector length)
            if len(domain_features.feature_vector) > 0:
                # Technical feature density (count of non-zero technical features)
                tech_features = [
                    f
                    for f in domain_features.technical_features.values()
                    if isinstance(f, (int, float))
                ]
                tech_density = (
                    sum(1 for f in tech_features if f > 0) / max(len(tech_features), 1)
                    if tech_features
                    else 0.0
                )
                features.append(min(1.0, tech_density))

                # Creative feature density
                creative_features = [
                    f
                    for f in domain_features.creative_features.values()
                    if isinstance(f, (int, float))
                ]
                creative_density = (
                    sum(1 for f in creative_features if f > 0)
                    / max(len(creative_features), 1)
                    if creative_features
                    else 0.0
                )
                features.append(min(1.0, creative_density))

                # Academic feature density
                academic_features = [
                    f
                    for f in domain_features.academic_features.values()
                    if isinstance(f, (int, float))
                ]
                academic_density = (
                    sum(1 for f in academic_features if f > 0)
                    / max(len(academic_features), 1)
                    if academic_features
                    else 0.0
                )
                features.append(min(1.0, academic_density))
            else:
                features.extend([0.0, 0.0, 0.0])

            # 11-15. Conversational and structural features
            conv_features = domain_features.conversational_features

            # 11. Conversational politeness (based on polite requests)
            politeness_score = min(1.0, conv_features.get("has_polite_requests", 0))
            features.append(politeness_score)

            # 12. Urgency indicator (binary)
            features.append(1.0 if conv_features.get("has_urgency", False) else 0.0)

            # 13. Question density (normalize by text length)
            question_count = conv_features.get("question_count", 0)
            word_count = len(prompt_text.split())
            question_density = min(
                1.0, question_count / max(word_count, 1) * 10
            )  # Scale up for visibility
            features.append(question_density)

            # 14. Instruction clarity (based on instruction indicators)
            instruction_count = conv_features.get("instruction_indicators", 0)
            instruction_clarity = min(
                1.0, instruction_count / max(word_count, 1) * 20
            )  # Scale up for visibility
            features.append(instruction_clarity)

            # 15. Domain hybrid indicator (based on multiple strong domain signals)
            hybrid_indicator = (
                1.0 if getattr(domain_features, "hybrid_domain", False) else 0.0
            )
            features.append(hybrid_indicator)

            # Cache the result for future use
            if self.config.cache_domain_analysis:
                self.domain_cache[cache_key] = features

            # Apply domain feature weight with adaptive weighting based on confidence
            base_weight = self.config.domain_feature_weight
            if self.config.adaptive_domain_weighting:
                # Adjust weight based on domain confidence
                confidence_boost = domain_features.confidence * 0.2  # Up to 20% boost
                adjusted_weight = min(1.0, base_weight + confidence_boost)
            else:
                adjusted_weight = base_weight

            # Apply weighting for balanced integration
            weighted_features = [
                f * adjusted_weight + (1 - adjusted_weight) * 0.5 for f in features
            ]

            return weighted_features

        except Exception as e:
            # Log error and return neutral features if analysis fails
            self.logger.warning(f"Domain feature extraction failed: {e}")
            return [0.5] * 15

    def _assess_clustering_quality(
        self, features: np.ndarray, labels: np.ndarray
    ) -> float:
        """Assess clustering quality using multiple metrics with enhanced validation"""
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        # Handle edge cases
        if n_clusters <= 1:
            return 0.0  # No meaningful clusters

        # Filter out noise points for quality assessment
        mask = labels != -1
        noise_points = int(np.sum(~mask))

        if np.sum(mask) < 2:
            return 0.0  # Not enough non-noise points

        filtered_features = features[mask]
        filtered_labels = labels[mask]

        if len(set(filtered_labels)) <= 1:
            return 0.0  # Only one cluster remains

        try:
            # Silhouette score (higher is better, range [-1, 1])
            silhouette = silhouette_score(
                filtered_features, filtered_labels, metric="euclidean"
            )

            # Calinski-Harabasz score (higher is better, unbounded)
            calinski = calinski_harabasz_score(filtered_features, filtered_labels)

            # Davies-Bouldin score (lower is better, unbounded, minimum 0)
            davies_bouldin = davies_bouldin_score(filtered_features, filtered_labels)

            # Normalize scores to [0, 1] range
            silhouette_normalized = (
                silhouette + 1
            ) / 2  # Convert from [-1, 1] to [0, 1]

            # More robust Calinski-Harabasz normalization
            # Use sigmoid-like transformation to handle unbounded nature
            calinski_normalized = calinski / (calinski + 100.0)  # Asymptotic to 1

            # Davies-Bouldin normalization (invert since lower is better)
            # Use exponential decay transformation
            davies_bouldin_normalized = np.exp(-davies_bouldin / 2.0)

            # Noise penalty: penalize high noise ratios
            noise_ratio = float(noise_points) / len(labels)
            noise_penalty = max(
                0.0, 1.0 - 2.0 * noise_ratio
            )  # Linear penalty, 0 at 50% noise

            # Cluster balance penalty: penalize highly imbalanced clusters
            cluster_sizes = np.bincount(filtered_labels)
            if len(cluster_sizes) > 0:
                cluster_balance = float(np.std(cluster_sizes)) / (
                    float(np.mean(cluster_sizes)) + 1e-8
                )
                balance_penalty = max(
                    0.1, 1.0 - cluster_balance / 3.0
                )  # Normalize by reasonable imbalance
            else:
                balance_penalty = 0.1  # Default penalty if no clusters

            # Weighted combination with research-validated weights
            quality_score = (
                0.40
                * float(silhouette_normalized)  # Primary metric for cluster separation
                + 0.25
                * float(calinski_normalized)  # Secondary metric for cluster compactness
                + 0.20
                * float(
                    davies_bouldin_normalized
                )  # Tertiary metric for cluster quality
                + 0.10 * float(noise_penalty)  # Penalty for excessive noise
                + 0.05 * float(balance_penalty)  # Penalty for cluster imbalance
            )

            # Log detailed metrics for debugging
            self.logger.debug(
                f"Clustering quality metrics: "
                f"silhouette={silhouette:.3f}, "
                f"calinski={calinski:.1f}, "
                f"davies_bouldin={davies_bouldin:.3f}, "
                f"noise_ratio={noise_ratio:.3f}, "
                f"cluster_balance={cluster_balance:.3f}, "
                f"final_score={quality_score:.3f}"
            )

            return float(max(0.0, min(1.0, quality_score)))  # Ensure [0, 1] range

        except Exception as e:
            self.logger.warning(f"Error assessing clustering quality: {e}")
            return 0.0

    def _group_by_clusters(
        self, results: list[dict[str, Any]], labels: np.ndarray
    ) -> dict[str, list[dict[str, Any]]]:
        """Group results by cluster labels"""
        clusters = {}

        for i, result in enumerate(results):
            cluster_id = int(labels[i])
            cluster_key = f"cluster_{cluster_id}" if cluster_id != -1 else "noise"

            if cluster_key not in clusters:
                clusters[cluster_key] = []

            clusters[cluster_key].append(result)

        return clusters

    def _fallback_to_kmeans(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Fallback to K-means clustering when advanced methods fail"""
        self.logger.info("Using K-means clustering fallback")

        features = self._extract_clustering_features(results)
        if features is None or len(features) < self.config.min_sample_size:
            return {"status": "insufficient_data", "clusters": {}}

        try:
            # Determine optimal number of clusters (simple heuristic)
            n_samples = len(features)
            max_clusters = min(self.config.max_context_groups, max(2, n_samples // 10))
            n_clusters = min(max_clusters, max(2, int(np.sqrt(n_samples))))

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)

            clusters = self._group_by_clusters(results, cluster_labels)

            return {
                "status": "success",
                "method": "kmeans_fallback",
                "n_clusters": n_clusters,
                "clusters": clusters,
                "quality_score": 0.5,  # Neutral score for fallback
            }

        except Exception as e:
            self.logger.error(f"K-means fallback failed: {e}")
            return {"status": "error", "clusters": {}}

    def update_baseline_performance(self, performance_data: dict[str, float]):
        """Update baseline performance metrics"""
        self.performance_baseline = performance_data.copy()
        self.logger.info(
            f"Updated baseline performance with {len(performance_data)} metrics"
        )

    def get_context_patterns(self) -> dict[str, dict[str, Any]]:
        """Get learned context patterns"""
        return self.context_patterns.copy()

    def clear_learning_state(self):
        """Clear accumulated learning state"""
        self.context_groups.clear()
        self.context_patterns.clear()
        self.specialization_opportunities.clear()
        self.performance_baseline = None
        self.logger.info("Cleared learning state")
