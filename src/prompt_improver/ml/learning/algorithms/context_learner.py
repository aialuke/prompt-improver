"""Context-Specific Learning Engine

Context learning engine that uses specialized components
for feature extraction and clustering analysis.
"""
import logging
from typing import Any, Dict, List, Optional
from sqlmodel import SQLModel, Field
from pydantic import BaseModel
import numpy as np
from ....security.owasp_input_validator import OWASP2025InputValidator
from ....security.input_validator import ValidationError
from ....security.memory_guard import MemoryGuard, get_memory_guard
from ....utils.datetime_utils import aware_utc_now
from ...clustering.services.clustering_optimizer_facade import ClusteringOptimizerFacade as ClusteringOptimizer
from ...clustering.services import ClusteringResult
# Import config from clustering services
from ..clustering import ClusteringConfig
from ..features import CompositeFeatureExtractor, FeatureExtractionConfig, FeatureExtractorFactory
logger = logging.getLogger(__name__)

class ContextConfig(BaseModel):
    """Configuration for context learner."""
    enable_linguistic_features: bool = Field(default=True, description='Enable linguistic feature extraction')
    enable_domain_features: bool = Field(default=True, description='Enable domain-specific feature extraction')
    enable_context_features: bool = Field(default=True, description='Enable context feature extraction')
    linguistic_weight: float = Field(default=1.0, ge=0.0, le=10.0, description='Weight for linguistic features')
    domain_weight: float = Field(default=1.0, ge=0.0, le=10.0, description='Weight for domain features')
    context_weight: float = Field(default=1.0, ge=0.0, le=10.0, description='Weight for context features')
    use_advanced_clustering: bool = Field(default=True, description='Use advanced clustering algorithms')
    min_cluster_size: int = Field(default=5, ge=2, description='Minimum cluster size')
    max_clusters: int = Field(default=8, ge=2, le=50, description='Maximum number of clusters')
    cache_enabled: bool = Field(default=True, description='Enable result caching')
    deterministic: bool = Field(default=True, description='Use deterministic algorithms')
    min_sample_size: int = Field(default=20, ge=1, description='Minimum sample size for learning')
    min_silhouette_score: float = Field(default=0.3, ge=0.0, le=1.0, description='Minimum silhouette score threshold')

class ContextLearningResult(BaseModel):
    """Result of context learning operation."""
    clusters_found: int = Field(ge=0, description='Number of clusters found')
    features_extracted: int = Field(ge=0, description='Number of features extracted')
    silhouette_score: float = Field(ge=-1.0, le=1.0, description='Silhouette score for clustering quality')
    processing_time: float = Field(ge=0.0, description='Processing time in seconds')
    quality_metrics: dict[str, float] = Field(default_factory=dict, description='Quality assessment metrics')
    recommendations: list[str] = Field(default_factory=list, description='Learning recommendations')

class ContextLearner:
    """Context-specific learning engine using specialized components.

    This implementation provides:
    - Specialized feature extractors for different data types
    - Dedicated clustering engine for pattern analysis
    - Clear orchestration of learning workflow
    - Separation of concerns for maintainability
    """

    def __init__(self, config: ContextConfig | None=None, training_loader=None):
        """Initialize context learner.

        Args:
            config: Configuration for context learning
            training_loader: Training data loader for ML pipeline integration
        """
        self.config = config or ContextConfig()
        self.input_validator = OWASP2025InputValidator()
        self.memory_guard = get_memory_guard()
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._initialize_feature_extractor()
        self._initialize_clustering_engine()
        self.training_loader = training_loader

    async def run_orchestrated_analysis(self, config: dict[str, Any]) -> dict[str, Any]:
        """Orchestrator-compatible interface for context learning (2025 pattern)

        Args:
            config: Orchestrator configuration containing:
                - context_data: Context data to analyze and learn from
                - output_path: Local path for output files (optional)
                - learning_mode: Mode of learning ('adaptive', 'batch', 'incremental')
                - feature_types: Types of features to extract (optional)

        Returns:
            Orchestrator-compatible result with learning results and metadata
        """
        start_time = aware_utc_now()
        try:
            context_data = config.get('context_data', [])
            output_path = config.get('output_path', './outputs/context_learning')
            learning_mode = config.get('learning_mode', 'adaptive')
            feature_types = config.get('feature_types', ['linguistic', 'domain', 'context'])
            if not context_data:
                raise ValidationError('No context data provided for learning')
            learning_result = await self.learn_context_patterns(context_data)
            result = {'clusters_discovered': learning_result.clusters_found, 'feature_analysis': {'features_extracted': learning_result.features_extracted, 'feature_types_used': feature_types, 'quality_score': learning_result.silhouette_score}, 'learning_insights': {'patterns_identified': learning_result.clusters_found, 'quality_metrics': learning_result.quality_metrics, 'recommendations': learning_result.recommendations}, 'context_clusters': [{'cluster_id': i, 'size': cluster_info.get('size', 0), 'characteristics': cluster_info.get('characteristics', []), 'quality_score': cluster_info.get('quality', 0.0)} for i, cluster_info in enumerate(learning_result.quality_metrics.get('cluster_details', []))]}
            end_time = aware_utc_now()
            execution_time = (end_time - start_time).total_seconds()
            return {'orchestrator_compatible': True, 'component_result': result, 'local_metadata': {'output_path': output_path, 'execution_time': execution_time, 'learning_mode': learning_mode, 'samples_processed': len(context_data), 'memory_usage_mb': self.memory_guard.get_current_usage() if self.memory_guard else 0, 'cache_enabled': self.config.cache_enabled, 'component_version': '1.0.0'}}
        except ValidationError as e:
            self.logger.error('Validation error in orchestrated context learning: %s', e)
            return {'orchestrator_compatible': True, 'component_result': {'error': f'Validation error: {str(e)}', 'clusters_discovered': 0}, 'local_metadata': {'execution_time': (aware_utc_now() - start_time).total_seconds(), 'error': True, 'error_type': 'validation', 'component_version': '1.0.0'}}
        except Exception as e:
            self.logger.error('Orchestrated context learning failed: %s', e)
            return {'orchestrator_compatible': True, 'component_result': {'error': str(e), 'clusters_discovered': 0}, 'local_metadata': {'execution_time': (aware_utc_now() - start_time).total_seconds(), 'error': True, 'component_version': '1.0.0'}}
        self.context_patterns: dict[str, Any] = {}
        self.cluster_results: ClusteringResult | None = None
        logger.info('ContextLearner initialized with specialized components')

    def _initialize_feature_extractor(self) -> None:
        """Initialize composite feature extractor."""
        feature_config = FeatureExtractionConfig(enable_linguistic=self.config.enable_linguistic_features, enable_domain=self.config.enable_domain_features, enable_context=self.config.enable_context_features, linguistic_weight=self.config.linguistic_weight, domain_weight=self.config.domain_weight, context_weight=self.config.context_weight, cache_enabled=self.config.cache_enabled)
        self.feature_extractor = CompositeFeatureExtractor(feature_config)
        logger.info('Feature extractor initialized with %s features', self.feature_extractor.get_feature_count())

    def _initialize_clustering_engine(self) -> None:
        """Initialize clustering engine."""
        clustering_config = ClusteringConfig(hdbscan_min_cluster_size=self.config.min_cluster_size, min_silhouette_score=self.config.min_silhouette_score, auto_dim_reduction=self.config.use_advanced_clustering, enable_caching=self.config.cache_enabled)
        self.clustering_engine = ClusteringOptimizer(clustering_config)
        logger.info('Clustering engine initialized')

    async def learn_from_data(self, training_data: list[dict[str, Any]]) -> ContextLearningResult:
        """Learn context patterns from training data.
        
        Args:
            training_data: List of training examples with text and context
            
        Returns:
            ContextLearningResult with learning outcomes
        """
        import time
        start_time = time.time()
        try:
            validated_data = self._validate_training_data(training_data)
            if not validated_data:
                return self._get_default_result()
            features_matrix = await self._extract_features_batch(validated_data)
            if features_matrix is None or features_matrix.size == 0:
                return self._get_default_result()
            cluster_result = await self.clustering_engine.cluster_contexts(features_matrix)
            self.cluster_results = cluster_result
            patterns = self._analyze_cluster_patterns(cluster_result, validated_data)
            self.context_patterns = patterns
            recommendations = self._generate_recommendations(cluster_result, patterns)
            processing_time = time.time() - start_time
            return ContextLearningResult(clusters_found=cluster_result.n_clusters, features_extracted=features_matrix.shape[1], silhouette_score=cluster_result.silhouette_score, processing_time=processing_time, quality_metrics=cluster_result.quality_metrics, recommendations=recommendations)
        except Exception as e:
            logger.error('Context learning failed: %s', e)
            return self._get_default_result()

    def _validate_training_data(self, training_data: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Validate and sanitize training data."""
        try:
            if not training_data or not isinstance(training_data, list):
                logger.warning('Invalid training data format')
                return None
            if len(training_data) < self.config.min_sample_size:
                logger.warning('Insufficient training data: {len(training_data)} < %s', self.config.min_sample_size)
                return None
            validated_examples = []
            for i, example in enumerate(training_data):
                try:
                    if not isinstance(example, dict):
                        continue
                    text = example.get('text') or example.get('prompt') or example.get('originalPrompt')
                    if not text or not isinstance(text, str):
                        continue
                    sanitized_example = self.input_validator.sanitize_json_input(example)
                    validated_examples.append(sanitized_example)
                except Exception as e:
                    logger.warning('Skipping invalid example {i}: %s', e)
                    continue
            if len(validated_examples) < self.config.min_sample_size:
                logger.warning('Too few valid examples after validation: %s', len(validated_examples))
                return None
            logger.info('Validated %s training examples', len(validated_examples))
            return validated_examples
        except Exception as e:
            logger.error('Training data validation failed: %s', e)
            return None

    async def _extract_features_batch(self, training_data: list[dict[str, Any]]) -> np.ndarray | None:
        """Extract features from batch of training examples."""
        try:
            features_list = []
            for example in training_data:
                text = example.get('text') or example.get('prompt') or example.get('originalPrompt', '')
                context_data = {'performance': example.get('performance', {}), 'user_id': example.get('user_id'), 'session_id': example.get('session_id'), 'project_type': example.get('project_type'), 'interaction': example.get('interaction', {}), 'temporal': example.get('temporal', {})}
                feature_result = self.feature_extractor.extract_features(text, context_data)
                features_list.append(feature_result['features'])
            if not features_list:
                return None
            features_matrix = np.vstack(features_list)
            logger.info('Extracted features matrix: %s', features_matrix.shape)
            return features_matrix
        except Exception as e:
            logger.error('Batch feature extraction failed: %s', e)
            return None

    def _analyze_cluster_patterns(self, cluster_result: ClusteringResult, training_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze patterns within clusters."""
        try:
            patterns = {}
            for cluster_id in range(cluster_result.n_clusters):
                cluster_mask = cluster_result.cluster_labels == cluster_id
                cluster_examples = [training_data[i] for i, mask in enumerate(cluster_mask) if mask]
                if not cluster_examples:
                    continue
                cluster_pattern = {'size': len(cluster_examples), 'avg_performance': self._calculate_avg_performance(cluster_examples), 'common_project_types': self._find_common_project_types(cluster_examples), 'typical_features': self._identify_typical_features(cluster_examples), 'recommendations': self._generate_cluster_recommendations(cluster_examples)}
                patterns[f'cluster_{cluster_id}'] = cluster_pattern
            return patterns
        except Exception as e:
            logger.error('Pattern analysis failed: %s', e)
            return {}

    def _calculate_avg_performance(self, examples: list[dict[str, Any]]) -> float:
        """Calculate average performance for cluster examples."""
        try:
            scores = []
            for example in examples:
                performance = example.get('performance', {})
                score = performance.get('improvement_score', 0.5)
                scores.append(float(score))
            return np.mean(scores) if scores else 0.5
        except Exception:
            return 0.5

    def _find_common_project_types(self, examples: list[dict[str, Any]]) -> list[str]:
        """Find most common project types in cluster."""
        try:
            project_types = [example.get('project_type', 'unknown') for example in examples]
            from collections import Counter
            common_types = Counter(project_types).most_common(3)
            return [ptype for ptype, count in common_types]
        except Exception:
            return ['unknown']

    def _identify_typical_features(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Identify typical features for cluster."""
        return {'avg_text_length': np.mean([len(str(ex.get('text', ''))) for ex in examples]), 'complexity_level': 'medium', 'domain_focus': 'general'}

    def _generate_cluster_recommendations(self, examples: list[dict[str, Any]]) -> list[str]:
        """Generate recommendations for cluster."""
        avg_performance = self._calculate_avg_performance(examples)
        recommendations = []
        if avg_performance < 0.6:
            recommendations.append('Consider more specific prompting strategies')
        if len(examples) > 20:
            recommendations.append('Large cluster - consider sub-clustering')
        return recommendations

    def _generate_recommendations(self, cluster_result: ClusteringResult, patterns: dict[str, Any]) -> list[str]:
        """Generate overall recommendations based on clustering results."""
        recommendations = []
        if cluster_result.silhouette_score < self.config.min_silhouette_score:
            recommendations.append('Low clustering quality - consider more training data')
        if cluster_result.n_clusters < 2:
            recommendations.append('No distinct patterns found - data may be too homogeneous')
        if cluster_result.n_clusters > 6:
            recommendations.append('Many clusters found - consider pattern consolidation')
        for cluster_id, pattern in patterns.items():
            if pattern['avg_performance'] > 0.8:
                recommendations.append(f'High-performing pattern in {cluster_id}')
        return recommendations

    def _get_default_result(self) -> ContextLearningResult:
        """Get default result when learning fails."""
        return ContextLearningResult(clusters_found=0, features_extracted=0, silhouette_score=0.0, processing_time=0.0, quality_metrics={}, recommendations=['Learning failed - check input data quality'])

    def get_context_patterns(self) -> dict[str, Any]:
        """Get learned context patterns."""
        return self.context_patterns.copy()

    def get_cluster_info(self) -> dict[str, Any] | None:
        """Get information about current clusters."""
        if self.cluster_results is None:
            return None
        return {'n_clusters': self.cluster_results.n_clusters, 'silhouette_score': self.cluster_results.silhouette_score, 'algorithm_used': self.cluster_results.algorithm_used, 'quality_metrics': self.cluster_results.quality_metrics}

    def clear_learning_state(self) -> None:
        """Clear accumulated learning state."""
        self.context_patterns.clear()
        self.cluster_results = None
        self.feature_extractor.clear_all_caches()
        logger.info('Learning state cleared')
