"""ML Testing Container for ML Intelligence Services.

Provides isolated ML testing environment with model simulation,
feature extraction testing, and ML pipeline validation.
"""

import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


@dataclass
class MLTestDataset:
    """ML test dataset for comprehensive testing."""

    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    size: str = "small"  # small, medium, large
    domains: list[str] = field(default_factory=lambda: ["general"])

    # Dataset content
    prompts: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    contexts: list[dict[str, Any]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    # Feature matrices (computed)
    feature_matrix: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)

    # Metadata
    created_at: str = ""
    sample_count: int = 0
    feature_count: int = 0


@dataclass
class MLModelResult:
    """Result from ML model operations."""

    success: bool = False
    model_type: str = ""
    processing_time_ms: float = 0.0

    # Results data
    predictions: list[Any] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)

    # Performance metrics
    accuracy: float | None = None
    silhouette_score: float | None = None
    cluster_count: int | None = None

    # Error information
    error_message: str | None = None


class MLTestContainer:
    """ML Testing Container for ML Intelligence Services.

    Provides isolated ML testing environment without requiring external ML services.
    Simulates ML operations for testing service boundaries and integration patterns.
    """

    def __init__(
        self,
        models_path: str | None = None,
        enable_gpu: bool = False,
        memory_limit_mb: int = 512,
        enable_caching: bool = True,
    ):
        """Initialize ML test container.

        Args:
            models_path: Path to store test models (temp dir if None)
            enable_gpu: Enable GPU support (CPU-only for CI/CD)
            memory_limit_mb: Memory limit for ML operations
            enable_caching: Enable model and feature caching
        """
        self.models_path = Path(models_path) if models_path else Path(tempfile.mkdtemp(prefix="ml_test_"))
        self.enable_gpu = enable_gpu
        self.memory_limit_mb = memory_limit_mb
        self.enable_caching = enable_caching

        self._container_id = str(uuid.uuid4())[:8]
        self._models: dict[str, Any] = {}
        self._vectorizers: dict[str, TfidfVectorizer] = {}
        self._cached_features: dict[str, np.ndarray] = {}
        self._test_datasets: dict[str, MLTestDataset] = {}

        # Performance tracking
        self._operation_times: dict[str, list[float]] = {
            "feature_extraction": [],
            "clustering": [],
            "prediction": [],
            "training": [],
        }

        # Ensure models directory exists
        self.models_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"MLTestContainer initialized: {self._container_id}")

    async def start(self) -> "MLTestContainer":
        """Start ML container and initialize models."""
        try:
            # Initialize default models for testing
            await self._initialize_test_models()

            # Create default test datasets
            await self._create_default_datasets()

            logger.info(f"ML test container started: {self._container_id}")
            return self

        except Exception as e:
            logger.exception(f"Failed to start ML test container: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop ML container and clean up resources."""
        try:
            # Clear cached data
            self._models.clear()
            self._vectorizers.clear()
            self._cached_features.clear()
            self._test_datasets.clear()

            # Clean up temporary files if using temp directory
            if "tmp" in str(self.models_path):
                import shutil
                shutil.rmtree(self.models_path, ignore_errors=True)

            logger.info(f"ML test container stopped: {self._container_id}")

        except Exception as e:
            logger.warning(f"Error stopping ML test container: {e}")

    async def _initialize_test_models(self):
        """Initialize test models for ML operations."""
        try:
            # Initialize clustering model
            self._models["hdbscan_clustering"] = HDBSCAN(
                min_cluster_size=5,
                min_samples=3,
                metric='euclidean',
                cluster_selection_method='eom'
            )

            # Initialize dimensionality reduction
            self._models["pca_reduction"] = PCA(
                n_components=50,  # Reduced for testing
                random_state=42
            )

            # Initialize text vectorizer
            self._vectorizers["tfidf"] = TfidfVectorizer(
                max_features=1000,  # Limited for testing
                stop_words='english',
                lowercase=True,
                max_df=0.95,
                min_df=2
            )

            logger.debug("Test models initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize test models: {e}")
            raise

    async def _create_default_datasets(self):
        """Create default test datasets for various testing scenarios."""
        try:
            # Small dataset for fast tests
            small_dataset = await self.create_test_dataset(
                size="small",
                domains=["general", "technical"],
                sample_count=100
            )
            self._test_datasets["small"] = small_dataset

            # Medium dataset for integration tests
            medium_dataset = await self.create_test_dataset(
                size="medium",
                domains=["general", "technical", "creative"],
                sample_count=500
            )
            self._test_datasets["medium"] = medium_dataset

            logger.debug("Default test datasets created")

        except Exception as e:
            logger.exception(f"Failed to create default datasets: {e}")
            raise

    async def create_test_dataset(
        self,
        size: str = "small",
        domains: list[str] | None = None,
        sample_count: int | None = None
    ) -> MLTestDataset:
        """Create ML test dataset for testing scenarios.

        Args:
            size: Dataset size ("small", "medium", "large")
            domains: List of domains for dataset
            sample_count: Override sample count

        Returns:
            MLTestDataset with generated data
        """
        domains = domains or ["general"]

        # Determine sample count based on size
        size_config = {
            "small": 100,
            "medium": 500,
            "large": 2000,
        }
        count = sample_count or size_config.get(size, 100)

        dataset = MLTestDataset(
            size=size,
            domains=domains,
            sample_count=count,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )

        # Generate synthetic prompt/improvement pairs
        prompts, improvements, contexts, labels = await self._generate_synthetic_data(
            count, domains
        )

        dataset.prompts = prompts
        dataset.improvements = improvements
        dataset.contexts = contexts
        dataset.labels = labels

        return dataset

    async def _generate_synthetic_data(
        self,
        count: int,
        domains: list[str]
    ) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
        """Generate synthetic ML training data.

        Args:
            count: Number of samples to generate
            domains: List of domains to generate data for

        Returns:
            Tuple of (prompts, improvements, contexts, labels)
        """
        prompts = []
        improvements = []
        contexts = []
        labels = []

        # Domain-specific templates
        domain_templates = {
            "general": {
                "prompts": [
                    "Write a summary of {}",
                    "Explain the concept of {}",
                    "Provide details about {}",
                    "Describe how {} works",
                ],
                "topics": ["machine learning", "data science", "programming", "technology", "business"]
            },
            "technical": {
                "prompts": [
                    "Implement {} algorithm",
                    "Debug the {} issue",
                    "Optimize {} performance",
                    "Design {} architecture",
                ],
                "topics": ["sorting", "database query", "API endpoint", "caching system", "microservice"]
            },
            "creative": {
                "prompts": [
                    "Write a story about {}",
                    "Create a poem featuring {}",
                    "Design a creative solution for {}",
                    "Brainstorm ideas for {}",
                ],
                "topics": ["adventure", "mystery", "innovation", "collaboration", "inspiration"]
            }
        }

        for i in range(count):
            # Select domain
            domain = domains[i % len(domains)]
            templates = domain_templates.get(domain, domain_templates["general"])

            # Generate prompt
            prompt_template = templates["prompts"][i % len(templates["prompts"])]
            topic = templates["topics"][i % len(templates["topics"])]
            prompt = prompt_template.format(topic)

            # Generate improvement (simulated)
            improvement = f"Enhanced {prompt.lower()} with better structure and clarity"

            # Generate context
            context = {
                "domain": domain,
                "topic": topic,
                "complexity": "basic" if i % 3 == 0 else "intermediate" if i % 3 == 1 else "advanced",
                "length": "short" if len(prompt) < 50 else "medium" if len(prompt) < 100 else "long",
                "generated_id": i,
            }

            # Generate label
            label = f"{domain}_{context['complexity']}"

            prompts.append(prompt)
            improvements.append(improvement)
            contexts.append(context)
            labels.append(label)

        return prompts, improvements, contexts, labels

    async def extract_features(
        self,
        texts: list[str],
        feature_type: str = "tfidf"
    ) -> MLModelResult:
        """Extract features from text data.

        Args:
            texts: List of text samples
            feature_type: Type of feature extraction ("tfidf", "basic")

        Returns:
            MLModelResult with extracted features
        """
        start_time = time.perf_counter()

        try:
            if feature_type == "tfidf":
                vectorizer = self._vectorizers.get("tfidf")
                if vectorizer is None:
                    raise ValueError("TF-IDF vectorizer not initialized")

                # Fit and transform texts
                feature_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out().tolist()

                # Cache if enabled
                if self.enable_caching:
                    cache_key = f"tfidf_{hash(tuple(texts))}"
                    self._cached_features[cache_key] = feature_matrix

                processing_time = (time.perf_counter() - start_time) * 1000
                self._operation_times["feature_extraction"].append(processing_time)

                return MLModelResult(
                    success=True,
                    model_type="tfidf_vectorizer",
                    processing_time_ms=processing_time,
                    predictions=feature_matrix.toarray().tolist(),
                    feature_importance=dict(zip(feature_names[:10], [1.0] * 10, strict=False)),  # Top 10 features
                )

            raise ValueError(f"Unsupported feature type: {feature_type}")

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Feature extraction failed: {e}")

            return MLModelResult(
                success=False,
                model_type=feature_type,
                processing_time_ms=processing_time,
                error_message=str(e)
            )

    async def perform_clustering(
        self,
        feature_matrix: np.ndarray,
        algorithm: str = "hdbscan"
    ) -> MLModelResult:
        """Perform clustering on feature matrix.

        Args:
            feature_matrix: Features to cluster
            algorithm: Clustering algorithm to use

        Returns:
            MLModelResult with clustering results
        """
        start_time = time.perf_counter()

        try:
            if algorithm == "hdbscan":
                clustering_model = self._models.get("hdbscan_clustering")
                if clustering_model is None:
                    raise ValueError("HDBSCAN clustering model not initialized")

                # Perform clustering
                if hasattr(feature_matrix, 'toarray'):
                    # Convert sparse matrix to dense
                    dense_features = feature_matrix.toarray()
                else:
                    dense_features = feature_matrix

                # Reduce dimensions if feature count is too high
                if dense_features.shape[1] > 100:
                    pca = self._models.get("pca_reduction")
                    if pca:
                        dense_features = pca.fit_transform(dense_features)

                cluster_labels = clustering_model.fit_predict(dense_features)

                # Calculate performance metrics
                unique_labels = np.unique(cluster_labels)
                cluster_count = len([label for label in unique_labels if label != -1])  # Exclude noise (-1)

                silhouette = None
                if cluster_count > 1 and len(dense_features) > cluster_count:
                    try:
                        silhouette = silhouette_score(dense_features, cluster_labels)
                    except Exception:
                        silhouette = 0.0  # Default for testing

                processing_time = (time.perf_counter() - start_time) * 1000
                self._operation_times["clustering"].append(processing_time)

                return MLModelResult(
                    success=True,
                    model_type="hdbscan_clustering",
                    processing_time_ms=processing_time,
                    predictions=cluster_labels.tolist(),
                    cluster_count=cluster_count,
                    silhouette_score=silhouette,
                )

            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Clustering failed: {e}")

            return MLModelResult(
                success=False,
                model_type=algorithm,
                processing_time_ms=processing_time,
                error_message=str(e)
            )

    async def generate_predictions(
        self,
        input_data: dict[str, Any],
        model_type: str = "pattern_prediction"
    ) -> MLModelResult:
        """Generate ML predictions for testing.

        Args:
            input_data: Input data for predictions
            model_type: Type of prediction model

        Returns:
            MLModelResult with predictions
        """
        start_time = time.perf_counter()

        try:
            # Simulate prediction logic based on input characteristics
            characteristics = input_data.get("characteristics", {})
            context = input_data.get("context", {})

            # Generate synthetic predictions
            predictions = []
            confidence_scores = []

            # Simulate rule effectiveness predictions
            if "rule_effectiveness" in characteristics:
                effectiveness_data = characteristics["rule_effectiveness"]
                for rule_id, effectiveness in effectiveness_data.items():
                    # Predict improvement potential based on current effectiveness
                    improvement_potential = max(0.1, 1.0 - effectiveness)
                    confidence = min(0.95, 0.5 + (improvement_potential * 0.4))

                    predictions.append({
                        "rule_id": rule_id,
                        "improvement_potential": improvement_potential,
                        "recommended_action": "optimize" if improvement_potential > 0.3 else "maintain"
                    })
                    confidence_scores.append(confidence)

            # Simulate pattern-based predictions
            if "pattern_insights" in context:
                pattern_count = len(context["pattern_insights"])
                overall_confidence = min(0.9, 0.6 + (pattern_count * 0.05))

                predictions.append({
                    "pattern_strength": "high" if pattern_count > 10 else "medium" if pattern_count > 5 else "low",
                    "optimization_opportunities": min(pattern_count, 5),
                    "expected_improvement": 0.1 + (pattern_count * 0.02)
                })
                confidence_scores.append(overall_confidence)

            processing_time = (time.perf_counter() - start_time) * 1000
            self._operation_times["prediction"].append(processing_time)

            return MLModelResult(
                success=True,
                model_type=model_type,
                processing_time_ms=processing_time,
                predictions=predictions,
                confidence_scores=confidence_scores,
                accuracy=0.85,  # Simulated accuracy
            )

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Prediction generation failed: {e}")

            return MLModelResult(
                success=False,
                model_type=model_type,
                processing_time_ms=processing_time,
                error_message=str(e)
            )

    def get_test_dataset(self, size: str = "small") -> MLTestDataset | None:
        """Get test dataset by size.

        Args:
            size: Dataset size to retrieve

        Returns:
            MLTestDataset if available
        """
        return self._test_datasets.get(size)

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get ML container performance metrics.

        Returns:
            Performance metrics for all operations
        """
        metrics = {
            "container_id": self._container_id,
            "operation_counts": {
                op_type: len(times) for op_type, times in self._operation_times.items()
            },
            "average_times_ms": {},
            "performance_targets": {
                "feature_extraction_ms": 50.0,  # Target <50ms
                "clustering_ms": 100.0,         # Target <100ms
                "prediction_ms": 20.0,          # Target <20ms
                "training_ms": 200.0,           # Target <200ms
            }
        }

        # Calculate average times and check performance targets
        performance_results = {}
        for op_type, times in self._operation_times.items():
            if times:
                avg_time = sum(times) / len(times)
                metrics["average_times_ms"][op_type] = avg_time

                target = metrics["performance_targets"].get(f"{op_type}_ms", float('inf'))
                performance_results[op_type] = {
                    "average_ms": avg_time,
                    "target_ms": target,
                    "performance_met": avg_time < target,
                    "sample_count": len(times)
                }

        metrics["performance_results"] = performance_results
        metrics["overall_performance"] = all(
            result["performance_met"] for result in performance_results.values()
        )

        return metrics

    def get_memory_usage(self) -> dict[str, Any]:
        """Get memory usage information."""
        return {
            "memory_limit_mb": self.memory_limit_mb,
            "cached_features_count": len(self._cached_features),
            "loaded_models_count": len(self._models),
            "test_datasets_count": len(self._test_datasets),
            "enable_caching": self.enable_caching,
        }

    async def cleanup_cache(self):
        """Clean up cached data."""
        self._cached_features.clear()
        logger.debug("ML container cache cleaned up")

    async def __aenter__(self) -> "MLTestContainer":
        """Async context manager entry."""
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class MLTestFixture:
    """Test fixture helper for ML testcontainers."""

    def __init__(self, container: MLTestContainer):
        self.container = container

    async def validate_ml_performance_targets(self) -> dict[str, Any]:
        """Validate ML operations against performance targets."""
        # Get sample dataset
        dataset = self.container.get_test_dataset("small")
        if not dataset:
            return {"error": "No test dataset available"}

        # Test feature extraction performance
        feature_result = await self.container.extract_features(
            dataset.prompts[:50],  # Small sample for testing
            feature_type="tfidf"
        )

        # Test clustering performance if features were extracted successfully
        clustering_result = None
        if feature_result.success and feature_result.predictions:
            feature_matrix = np.array(feature_result.predictions)
            clustering_result = await self.container.perform_clustering(
                feature_matrix,
                algorithm="hdbscan"
            )

        # Test prediction performance
        prediction_input = {
            "characteristics": {"rule_effectiveness": {"rule_1": 0.7, "rule_2": 0.8}},
            "context": {"pattern_insights": [{"pattern": "test"} for _ in range(8)]}
        }
        prediction_result = await self.container.generate_predictions(
            prediction_input,
            model_type="pattern_prediction"
        )

        # Get comprehensive metrics
        performance_metrics = self.container.get_performance_metrics()

        return {
            "feature_extraction": {
                "success": feature_result.success,
                "processing_time_ms": feature_result.processing_time_ms,
                "target_ms": 50.0,
                "performance_met": feature_result.processing_time_ms < 50.0,
            },
            "clustering": {
                "success": clustering_result.success if clustering_result else False,
                "processing_time_ms": clustering_result.processing_time_ms if clustering_result else 0,
                "target_ms": 100.0,
                "performance_met": clustering_result.processing_time_ms < 100.0 if clustering_result else False,
                "cluster_count": clustering_result.cluster_count if clustering_result else 0,
            },
            "prediction": {
                "success": prediction_result.success,
                "processing_time_ms": prediction_result.processing_time_ms,
                "target_ms": 20.0,
                "performance_met": prediction_result.processing_time_ms < 20.0,
                "prediction_count": len(prediction_result.predictions),
            },
            "overall_metrics": performance_metrics,
        }

    async def create_comprehensive_test_scenario(self) -> dict[str, Any]:
        """Create comprehensive test scenario with multiple datasets and operations."""
        scenarios = {}

        # Test different dataset sizes
        for size in ["small", "medium"]:
            dataset = await self.container.create_test_dataset(
                size=size,
                domains=["general", "technical", "creative"]
            )

            # Extract features
            feature_result = await self.container.extract_features(
                dataset.prompts,
                feature_type="tfidf"
            )

            scenarios[size] = {
                "dataset_size": dataset.sample_count,
                "feature_extraction": {
                    "success": feature_result.success,
                    "processing_time_ms": feature_result.processing_time_ms,
                    "feature_count": len(feature_result.feature_importance) if feature_result.success else 0,
                }
            }

        return {
            "scenarios": scenarios,
            "container_metrics": self.container.get_performance_metrics(),
            "memory_usage": self.container.get_memory_usage(),
        }
