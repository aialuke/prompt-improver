"""Advanced Pattern Discovery Service for Phase 4 ML Enhancement & Discovery
Modern 2025 implementation with HDBSCAN, FP-Growth, and ensemble pattern mining
Enhanced with Apriori Algorithm integration for association rule discovery
"""

import logging
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd

# Modern clustering and pattern mining imports with performance optimization
try:
    from concurrent.futures import ThreadPoolExecutor

    import hdbscan

    # Import joblib for parallel processing optimization
    import joblib

    HDBSCAN_AVAILABLE = True
    JOBLIB_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    JOBLIB_AVAILABLE = False

from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# FP-Growth pattern mining (using apyori as fallback)
try:
    from mlxtend.frequent_patterns import association_rules, fpgrowth
    from mlxtend.preprocessing import TransactionEncoder

    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ....database import get_unified_manager, ManagerMode
from ....database.models import RuleMetadata, RulePerformance
from .apriori_analyzer import AprioriAnalyzer, AprioriConfig
# Caching functionality moved to AppConfig

logger = logging.getLogger(__name__)

@dataclass
class PatternCluster:
    """Advanced pattern cluster with metadata"""

    cluster_id: int
    pattern_type: str  # 'parameter', 'sequence', 'performance', 'semantic', 'apriori'
    patterns: list[dict[str, Any]]
    centroid: dict[str, Any] | None
    cluster_score: float
    outlier_score: float
    density: float
    samples_count: int
    effectiveness_range: tuple[float, float]

@dataclass
class FrequentPattern:
    """Frequent pattern from FP-Growth mining"""

    itemset: set[str]
    support: float
    confidence: float | None
    lift: float | None
    conviction: float | None
    effectiveness_impact: float
    rule_context: str

@dataclass
class AprioriPattern:
    """Association pattern from Apriori algorithm"""

    antecedents: list[str]
    consequents: list[str]
    support: float
    confidence: float
    lift: float
    conviction: float
    rule_strength: float
    business_insight: str
    pattern_category: str

class AdvancedPatternDiscovery:
    """Advanced Pattern Discovery using modern 2025 techniques:

    - HDBSCAN for varying density cluster discovery
    - FP-Growth for frequent pattern mining (faster than Apriori)
    - Apriori Algorithm for association rule mining and business insights
    - Ensemble pattern mining with multiple algorithms
    - Semantic pattern analysis for rule parameter relationships
    - Outlier-based discovery for unique high-performing patterns
    - Statistical validation with confidence intervals
    """

    def __init__(self, db_manager = None):
        """Initialize AdvancedPatternDiscovery with lazy initialization support.
        
        Args:
            db_manager: Optional DatabaseManager instance for dependency injection.
                       If None, will be created lazily when first needed.
        """
        self.scaler = StandardScaler()
        self.min_cluster_size = 5
        self.min_support = 0.1
        self.min_confidence = 0.5
        self.min_effectiveness = 0.7

        # HDBSCAN Performance Optimization Settings (based on Context7 research)
        self._configure_hdbscan_performance()

        # Lazy initialization attributes
        self._db_manager = db_manager or get_unified_manager(ManagerMode.ML_TRAINING)
        self._apriori_analyzer = None
        self._training_loader = None
        self._db_manager_lock = threading.Lock()
        self._apriori_analyzer_lock = threading.Lock()
        self._training_loader_lock = threading.Lock()

        logger.info("AdvancedPatternDiscovery initialized with lazy loading strategy")

    @property
    def db_manager(self):
        """Get DatabaseManager instance with lazy initialization."""
        if self._db_manager is None:
            with self._db_manager_lock:
                if self._db_manager is None:
                    try:
                        self._db_manager = get_unified_manager(ManagerMode.ML_TRAINING)
                        logger.info("DatabaseManager created via lazy initialization")
                    except Exception as e:
                        logger.error(f"Failed to create DatabaseManager: {e}")
                        return None
        return self._db_manager

    @property
    def apriori_analyzer(self) -> AprioriAnalyzer | None:
        """Get AprioriAnalyzer instance with lazy initialization."""
        if self._apriori_analyzer is None:
            with self._apriori_analyzer_lock:
                if self._apriori_analyzer is None:
                    db_mgr = self.db_manager
                    if db_mgr is not None:
                        try:
                            self._apriori_analyzer = AprioriAnalyzer(
                                db_manager=db_mgr,
                                config=AprioriConfig(
                                    min_support=0.08,  # Slightly lower for more patterns
                                    min_confidence=0.6,
                                    min_lift=1.2,
                                    max_itemset_length=4,
                                    verbose=True,
                                ),
                            )
                            logger.info("AprioriAnalyzer created via lazy initialization")
                        except Exception as e:
                            logger.error(f"Failed to create AprioriAnalyzer: {e}")
                            return None
                    else:
                        logger.warning("Cannot create AprioriAnalyzer - DatabaseManager not available")
                        return None
        return self._apriori_analyzer

    @property
    def training_loader(self):
        """Get TrainingDataLoader instance with lazy initialization."""
        if self._training_loader is None:
            with self._training_loader_lock:
                if self._training_loader is None:
                    try:
                        # Import here to avoid circular imports
                        from ...core.training_data_loader import TrainingDataLoader
                        self._training_loader = TrainingDataLoader()
                        logger.info("TrainingDataLoader created via lazy initialization")
                    except Exception as e:
                        logger.error(f"Failed to create TrainingDataLoader: {e}")
                        return None
        return self._training_loader

    def _ensure_database_connection(self) -> bool:
        """Ensure database connection is available with enhanced error handling."""
        try:
            if self.db_manager is None:
                logger.error("DatabaseManager not available - ensure database configuration is correct")
                return False
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

    def _ensure_apriori_analyzer(self) -> bool:
        """Ensure AprioriAnalyzer is available with enhanced error handling."""
        try:
            if self.apriori_analyzer is None:
                logger.error("AprioriAnalyzer not available - database connection required")
                return False
            return True
        except Exception as e:
            logger.error(f"AprioriAnalyzer check failed: {e}")
            return False

    @cached(ttl=3600, key_func=lambda *a, **kw: f"pattern_discovery:{kw.get('min_effectiveness', 0.7)}:{kw.get('min_support', 5)}:{':'.join(kw.get('pattern_types', ['all']))}:{kw.get('use_ensemble', True)}:{kw.get('include_apriori', True)}")
    async def discover_advanced_patterns(
        self,
        db_session: AsyncSession,
        min_effectiveness: float = 0.7,
        min_support: int = 5,
        pattern_types: list[str] | None = None,
        use_ensemble: bool = True,
        include_apriori: bool = True,
    ) -> dict[str, Any]:
        """Discover advanced patterns using ensemble ML techniques and Apriori analysis

        Enhanced with association rule mining for comprehensive pattern discovery

        Args:
            db_session: Database session for data access
            min_effectiveness: Minimum effectiveness threshold
            min_support: Minimum support for pattern validation
            pattern_types: Types of patterns to discover
            use_ensemble: Use ensemble validation
            include_apriori: Include Apriori association rule mining

        Returns:
            Comprehensive pattern discovery results with Apriori insights
        """
        start_time = time.time()

        if pattern_types is None:
            pattern_types = ["parameter", "sequence", "performance", "semantic"]
            if include_apriori and self.apriori_analyzer:
                pattern_types.append("apriori")

        logger.info(
            f"Starting advanced pattern discovery with types: {pattern_types}, "
            f"including Apriori: {include_apriori}"
        )

        # Get performance data
        performance_data = await self._get_performance_data(
            db_session, min_effectiveness
        )

        if len(performance_data) < min_support:
            logger.warning(
                f"Insufficient data points ({len(performance_data)}) for pattern discovery"
            )
            return {"error": "Insufficient data", "data_points": len(performance_data)}

        # Discover patterns by type
        results: dict[str, Any] = {
            "discovery_metadata": {
                "start_time": start_time,
                "data_points": len(performance_data),
            }
        }

        # Execute pattern discovery in parallel where possible
        discovery_tasks = []

        if "parameter" in pattern_types:
            results["parameter_patterns"] = await self._discover_parameter_patterns(
                performance_data
            )

        if "sequence" in pattern_types:
            results["sequence_patterns"] = await self._discover_sequence_patterns(
                performance_data
            )

        if "performance" in pattern_types:
            results["performance_patterns"] = await self._discover_performance_patterns(
                performance_data
            )

        if "semantic" in pattern_types:
            results["semantic_patterns"] = await self._discover_semantic_patterns(
                performance_data
            )

        # NEW: Apriori Association Rule Discovery
        if "apriori" in pattern_types and self.apriori_analyzer:
            results["apriori_patterns"] = await self._discover_apriori_patterns(
                performance_data
            )

        # Ensemble validation combining traditional ML with Apriori insights
        if use_ensemble:
            results["ensemble_analysis"] = self._analyze_pattern_ensemble(results)

        # Enhanced statistical validation including Apriori metrics
        results["statistical_validation"] = self._validate_patterns_statistically(
            results
        )

        # Generate comprehensive recommendations including Apriori business insights
        recommendations = self._generate_pattern_recommendations(results)
        results["recommendations"] = recommendations

        # Add execution metadata
        results["discovery_metadata"].update({
            "execution_time": time.time() - start_time,
            "algorithms_used": pattern_types,
            "apriori_enabled": include_apriori and self.apriori_analyzer is not None,
            "ensemble_validation": use_ensemble,
            "timestamp": time.time(),
        })

        logger.info(
            f"Pattern discovery completed in {results['discovery_metadata']['execution_time']:.2f}s "
            f"with {len(pattern_types)} algorithms"
        )

        return results

    async def _get_performance_data(
        self, db_session: AsyncSession, min_effectiveness: float
    ) -> list[dict[str, Any]]:
        """Get comprehensive performance data for pattern analysis"""
        try:
            # Use proper SQL query following established patterns in codebase
            query = text("""
                SELECT 
                    rp.rule_id,
                    rp.improvement_score,
                    rp.execution_time_ms,
                    rp.confidence_level,
                    rp.parameters_used,
                    rp.created_at,
                    rm.default_parameters as rule_metadata_params,
                    rm.category,
                    rm.priority
                FROM rule_performance rp
                LEFT JOIN rule_metadata rm ON rp.rule_id = rm.rule_id
                WHERE rp.improvement_score >= :min_effectiveness
                ORDER BY rp.created_at DESC
                LIMIT 1000
            """)

            result = await db_session.execute(
                query, {"min_effectiveness": min_effectiveness}
            )
            rows = result.fetchall()

            performance_data = []
            for row in rows:
                # Convert to comprehensive feature dictionary using correct attributes
                data_point = {
                    "rule_id": row.rule_id,
                    "rule_name": row.rule_id,  # Use rule_id as name since rule_name doesn't exist
                    "rule_category": row.category
                    or "general",  # Use category not rule_category
                    "improvement_score": row.improvement_score or 0.0,
                    "execution_time_ms": row.execution_time_ms or 0,
                    "user_satisfaction": 0.8,  # Default since user_satisfaction_score doesn't exist
                    "confidence_level": row.confidence_level or 0.0,
                    "weight": 1.0,  # Default since weight doesn't exist
                    "priority": row.priority or 5,
                    "parameters": row.parameters_used
                    or {},  # Use parameters_used not rule_parameters
                    "metadata_params": row.rule_metadata_params
                    or {},  # Use default_parameters
                    "before_metrics": {},  # Default since before_metrics doesn't exist
                    "after_metrics": {},  # Default since after_metrics doesn't exist
                    "prompt_characteristics": {},  # Default since prompt_characteristics doesn't exist
                    "created_at": row.created_at,
                    "timestamp": row.created_at.timestamp() if row.created_at else 0,
                }
                performance_data.append(data_point)

            return performance_data

        except Exception as e:
            logger.error(f"Failed to get performance data: {e}")
            return []

    async def _discover_parameter_patterns(
        self, performance_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Discover parameter patterns using HDBSCAN clustering"""
        try:
            # Extract parameter features
            param_features = []
            data_indices = []

            for i, data_point in enumerate(performance_data):
                params = data_point.get("parameters", {})
                metadata_params = data_point.get("metadata_params", {})

                # Combine parameters from both sources
                all_params = {**params, **metadata_params}

                if all_params:
                    # Convert parameters to numerical features
                    feature_vector = self._params_to_features(all_params, data_point)
                    if feature_vector:
                        param_features.append(feature_vector)
                        data_indices.append(i)

            if len(param_features) < self.min_cluster_size:
                return {
                    "clusters": [],
                    "outliers": [],
                    "message": "Insufficient parameter data",
                }

            # Normalize features
            param_features = np.array(param_features)
            param_features_scaled = self.scaler.fit_transform(param_features)

            # Apply optimized HDBSCAN for parameter clustering
            if HDBSCAN_AVAILABLE:
                # Optimized HDBSCAN configuration based on performance research
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=max(
                        3, self.min_cluster_size // 3
                    ),  # Adaptive min_samples
                    metric="euclidean",
                    algorithm="boruvka_kdtree",  # Most efficient for moderate datasets
                    cluster_selection_method="eom",  # Excess of Mass for stability
                    alpha=1.0,  # Conservative clustering
                    core_dist_n_jobs=self._get_optimal_n_jobs(),  # Parallel processing
                )
                cluster_labels = clusterer.fit_predict(param_features_scaled)
                outlier_scores = clusterer.outlier_scores_
                probabilities = clusterer.probabilities_
            else:
                # Fallback to DBSCAN
                clusterer = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)
                cluster_labels = clusterer.fit_predict(param_features_scaled)
                outlier_scores = np.zeros(len(cluster_labels))
                probabilities = np.ones(len(cluster_labels))

            # Process clusters
            clusters = self._process_parameter_clusters(
                cluster_labels,
                performance_data,
                data_indices,
                outlier_scores,
                probabilities,
            )

            # Identify high-value outliers
            outliers = self._identify_parameter_outliers(
                cluster_labels, performance_data, data_indices, outlier_scores
            )

            return {
                "clusters": clusters,
                "outliers": outliers,
                "algorithm": "HDBSCAN" if HDBSCAN_AVAILABLE else "DBSCAN",
                "total_patterns": len(clusters),
                "total_outliers": len(outliers),
                "feature_dimensions": param_features.shape[1],
            }

        except Exception as e:
            logger.error(f"Parameter pattern discovery failed: {e}")
            return {"error": str(e)}

    async def _discover_sequence_patterns(
        self, performance_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Discover sequence patterns using FP-Growth algorithm"""
        try:
            if not MLXTEND_AVAILABLE:
                return self._fallback_sequence_patterns(performance_data)

            # Convert data to transaction format for FP-Growth
            transactions = []
            effectiveness_map = {}

            for data_point in performance_data:
                # Create transaction from rule characteristics
                transaction = self._create_transaction(data_point)
                if transaction:
                    transactions.append(transaction)

                    # Map transaction to effectiveness
                    transaction_key = tuple(sorted(transaction))
                    if transaction_key not in effectiveness_map:
                        effectiveness_map[transaction_key] = []
                    effectiveness_map[transaction_key].append(
                        data_point["improvement_score"]
                    )

            if len(transactions) < 5:
                return {
                    "frequent_patterns": [],
                    "association_rules": [],
                    "message": "Insufficient sequence data",
                }

            # Apply Transaction Encoder with proper pandas compatibility
            te = TransactionEncoder()
            te_data = te.fit(transactions).transform(transactions)

            # Handle mlxtend-pandas compatibility with proper type checking
            try:
                if hasattr(te, "columns_") and te.columns_ is not None:
                    # Proper type handling for pandas DataFrame creation
                    column_names = [str(col) for col in te.columns_]
                    # Create DataFrame with explicit column specification - use Any to handle type checker limitations
                    if column_names:
                        df = pd.DataFrame(te_data, columns=column_names)  # type: ignore[arg-type]  # mlxtend-pandas compatibility
                    else:
                        df = pd.DataFrame(te_data)
                else:
                    # Fallback if columns_ is not available
                    df = pd.DataFrame(te_data)
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"DataFrame creation fallback due to: {e}")
                # Ultimate fallback for compatibility
                df = pd.DataFrame(te_data)

            # Apply FP-Growth algorithm with type safety
            try:
                frequent_itemsets = fpgrowth(
                    df, min_support=self.min_support, use_colnames=True
                )

                # Ensure we have a proper DataFrame for type safety
                if not isinstance(frequent_itemsets, pd.DataFrame):
                    frequent_itemsets = pd.DataFrame()

            except Exception:
                frequent_itemsets = pd.DataFrame()  # Empty fallback

            if frequent_itemsets.empty:
                return {
                    "frequent_patterns": [],
                    "association_rules": [],
                    "message": "No frequent patterns found",
                }

            # Generate association rules with defensive handling
            try:
                # Ensure we have a proper DataFrame before processing
                if (
                    isinstance(frequent_itemsets, pd.DataFrame)
                    and not frequent_itemsets.empty
                ):
                    rules = association_rules(
                        frequent_itemsets,
                        metric="confidence",
                        min_threshold=self.min_confidence,
                    )
                else:
                    rules = pd.DataFrame()

            except (ValueError, TypeError, AttributeError):
                rules = pd.DataFrame()  # Empty rules if generation fails

            # Process patterns with effectiveness analysis - enhanced type safety
            patterns = self._process_frequent_patterns(
                frequent_itemsets, effectiveness_map
            )
            rule_patterns = self._process_association_rules(rules, effectiveness_map)

            return {
                "frequent_patterns": patterns,
                "association_rules": rule_patterns,
                "algorithm": "FP-Growth",
                "total_transactions": len(transactions),
                "total_patterns": len(patterns),
                "total_rules": len(rule_patterns),
            }

        except Exception as e:
            logger.error(f"Sequence pattern discovery failed: {e}")
            return {"error": str(e)}

    async def _discover_performance_patterns(
        self, performance_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Discover performance patterns using density-based clustering"""
        try:
            # Extract performance features
            perf_features = []
            for data_point in performance_data:
                features = [
                    data_point["improvement_score"],
                    data_point["execution_time_ms"] / 1000.0,  # Normalize to seconds
                    data_point["user_satisfaction"],
                    data_point["confidence_level"],
                    data_point["weight"],
                    data_point["priority"] / 10.0,  # Normalize priority
                ]
                perf_features.append(features)

            if len(perf_features) < self.min_cluster_size:
                return {"clusters": [], "message": "Insufficient performance data"}

            # Normalize features
            perf_features = np.array(perf_features)
            perf_features_scaled = self.scaler.fit_transform(perf_features)

            # Apply optimized clustering for performance patterns
            if HDBSCAN_AVAILABLE:
                # Performance-optimized HDBSCAN for larger datasets
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=max(3, self.min_cluster_size // 3),
                    algorithm="boruvka_kdtree",  # Best performance for mixed data
                    cluster_selection_method="eom",
                    cluster_selection_epsilon=0.0,  # No epsilon filtering initially
                    alpha=1.0,  # Conservative merging
                    core_dist_n_jobs=self._get_optimal_n_jobs(),
                )
                cluster_labels = clusterer.fit_predict(perf_features_scaled)
                outlier_scores = clusterer.outlier_scores_
            else:
                clusterer = DBSCAN(eps=0.3, min_samples=self.min_cluster_size)
                cluster_labels = clusterer.fit_predict(perf_features_scaled)
                outlier_scores = np.zeros(len(cluster_labels))

            # Calculate cluster quality metrics
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(perf_features_scaled, cluster_labels)
                calinski_score = calinski_harabasz_score(
                    perf_features_scaled, cluster_labels
                )
            else:
                silhouette_avg = 0.0
                calinski_score = 0.0

            # Process performance clusters
            clusters = self._process_performance_clusters(
                cluster_labels, performance_data, perf_features, outlier_scores
            )

            return {
                "clusters": clusters,
                "algorithm": "HDBSCAN" if HDBSCAN_AVAILABLE else "DBSCAN",
                "cluster_quality": {
                    "silhouette_score": silhouette_avg,
                    "calinski_harabasz_score": calinski_score,
                },
                "total_clusters": len([c for c in clusters if c["cluster_id"] != -1]),
                "noise_points": len([c for c in clusters if c["cluster_id"] == -1]),
            }

        except Exception as e:
            logger.error(f"Performance pattern discovery failed: {e}")
            return {"error": str(e)}

    async def _discover_semantic_patterns(
        self, performance_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Discover semantic patterns in rule categories and characteristics"""
        try:
            # Group by semantic categories
            category_groups = defaultdict(list)
            for data_point in performance_data:
                category = data_point.get("rule_category", "general")
                category_groups[category].append(data_point)

            semantic_patterns = []

            for category, data_points in category_groups.items():
                if len(data_points) >= 3:  # Minimum for semantic analysis
                    pattern = self._analyze_semantic_category(category, data_points)
                    if pattern:
                        semantic_patterns.append(pattern)

            # Cross-category analysis
            cross_patterns = self._analyze_cross_category_patterns(category_groups)

            return {
                "category_patterns": semantic_patterns,
                "cross_category_patterns": cross_patterns,
                "total_categories": len(category_groups),
                "total_patterns": len(semantic_patterns) + len(cross_patterns),
            }

        except Exception as e:
            logger.error(f"Semantic pattern discovery failed: {e}")
            return {"error": str(e)}

    def _params_to_features(
        self, params: dict[str, Any], data_point: dict[str, Any]
    ) -> list[float] | None:
        """Convert parameters to numerical feature vector"""
        try:
            features = []

            # Extract numerical parameters
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, bool):
                    features.append(1.0 if value else 0.0)
                elif isinstance(value, str):
                    # Convert string to hash-based feature
                    features.append(float(hash(value) % 1000) / 1000.0)

            # Add contextual features
            features.extend([
                data_point["improvement_score"],
                data_point["execution_time_ms"] / 1000.0,
                data_point["user_satisfaction"],
                data_point["confidence_level"],
            ])

            return features if features else None

        except Exception:
            return None

    def _create_transaction(self, data_point: dict[str, Any]) -> list[str] | None:
        """Create transaction for sequence pattern mining"""
        try:
            transaction = []

            # Add rule characteristics
            transaction.append(f"category_{data_point['rule_category']}")

            # Add performance ranges
            score = data_point["improvement_score"]
            if score >= 0.9:
                transaction.append("performance_excellent")
            elif score >= 0.7:
                transaction.append("performance_good")
            else:
                transaction.append("performance_moderate")

            # Add execution time ranges
            exec_time = data_point["execution_time_ms"]
            if exec_time <= 10:
                transaction.append("speed_fast")
            elif exec_time <= 50:
                transaction.append("speed_medium")
            else:
                transaction.append("speed_slow")

            # Add satisfaction levels
            satisfaction = data_point["user_satisfaction"]
            if satisfaction >= 0.8:
                transaction.append("satisfaction_high")
            elif satisfaction >= 0.6:
                transaction.append("satisfaction_medium")
            else:
                transaction.append("satisfaction_low")

            return transaction if len(transaction) > 1 else None

        except Exception:
            return None

    def _fallback_sequence_patterns(
        self, performance_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Fallback sequence pattern discovery without mlxtend"""
        # Simple frequency-based pattern discovery
        pattern_counts = Counter()

        for data_point in performance_data:
            transaction = self._create_transaction(data_point)
            if transaction:
                # Count individual items and pairs
                for item in transaction:
                    pattern_counts[item] += 1

                # Count pairs
                for pair in combinations(transaction, 2):
                    pattern_counts[tuple(sorted(pair))] += 1

        # Filter by minimum support
        min_count = max(1, int(len(performance_data) * self.min_support))
        frequent_patterns = [
            {"pattern": pattern, "support": count / len(performance_data)}
            for pattern, count in pattern_counts.items()
            if count >= min_count
        ]

        return {
            "frequent_patterns": frequent_patterns,
            "association_rules": [],
            "algorithm": "Simple Frequency",
            "total_patterns": len(frequent_patterns),
        }

    def _process_parameter_clusters(
        self,
        cluster_labels: np.ndarray,
        performance_data: list[dict[str, Any]],
        data_indices: list[int],
        outlier_scores: np.ndarray,
        probabilities: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Process parameter clusters into structured results"""
        clusters = []
        unique_labels = set(cluster_labels)

        for label in unique_labels:
            if label == -1:  # Skip noise points for clusters
                continue

            # Get cluster members
            cluster_mask = cluster_labels == label
            cluster_indices = [
                data_indices[i] for i, mask in enumerate(cluster_mask) if mask
            ]
            cluster_data = [performance_data[i] for i in cluster_indices]

            # Calculate cluster statistics
            effectiveness_scores = [d["improvement_score"] for d in cluster_data]
            avg_effectiveness = np.mean(effectiveness_scores)
            std_effectiveness = np.std(effectiveness_scores)

            # Extract common parameters
            common_params = self._extract_common_parameters(cluster_data)

            cluster_info = {
                "cluster_id": int(label),
                "size": len(cluster_data),
                "avg_effectiveness": float(avg_effectiveness),
                "std_effectiveness": float(std_effectiveness),
                "min_effectiveness": float(min(effectiveness_scores)),
                "max_effectiveness": float(max(effectiveness_scores)),
                "common_parameters": common_params,
                "avg_outlier_score": float(np.mean(outlier_scores[cluster_mask])),
                "avg_probability": float(np.mean(probabilities[cluster_mask])),
                "sample_rules": [
                    d["rule_name"] for d in cluster_data[:3]
                ],  # Sample rules
            }
            clusters.append(cluster_info)

        return sorted(clusters, key=lambda x: x["avg_effectiveness"], reverse=True)

    def _identify_parameter_outliers(
        self,
        cluster_labels: np.ndarray,
        performance_data: list[dict[str, Any]],
        data_indices: list[int],
        outlier_scores: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Identify high-value parameter outliers"""
        outliers = []

        # Find outliers (cluster label -1 or high outlier score)
        for i, (label, score) in enumerate(
            zip(cluster_labels, outlier_scores, strict=False)
        ):
            if label == -1 or score > 0.7:  # High outlier score threshold
                data_idx = data_indices[i]
                data_point = performance_data[data_idx]

                # Only include high-performing outliers
                if data_point["improvement_score"] >= self.min_effectiveness:
                    outlier_info = {
                        "rule_id": data_point["rule_id"],
                        "rule_name": data_point["rule_name"],
                        "effectiveness": data_point["improvement_score"],
                        "outlier_score": float(score),
                        "parameters": data_point["parameters"],
                        "unique_characteristics": self._analyze_outlier_uniqueness(
                            data_point
                        ),
                    }
                    outliers.append(outlier_info)

        return sorted(outliers, key=lambda x: x["effectiveness"], reverse=True)

    def _process_frequent_patterns(
        self,
        frequent_itemsets: pd.DataFrame,
        effectiveness_map: dict[tuple, list[float]],
    ) -> list[dict[str, Any]]:
        """Process frequent patterns with effectiveness analysis"""
        patterns = []

        for _, row in frequent_itemsets.iterrows():
            itemset = set(row["itemsets"])
            support = row["support"]

            # Calculate effectiveness for this pattern
            pattern_key = tuple(sorted(itemset))
            effectiveness_scores = effectiveness_map.get(pattern_key, [])

            if effectiveness_scores:
                avg_effectiveness = np.mean(effectiveness_scores)
                if avg_effectiveness >= self.min_effectiveness:
                    pattern_info = {
                        "itemset": list(itemset),
                        "support": float(support),
                        "avg_effectiveness": float(avg_effectiveness),
                        "effectiveness_count": len(effectiveness_scores),
                        "effectiveness_std": float(np.std(effectiveness_scores)),
                    }
                    patterns.append(pattern_info)

        return sorted(patterns, key=lambda x: x["avg_effectiveness"], reverse=True)

    def _process_association_rules(
        self, rules: pd.DataFrame, effectiveness_map: dict[tuple, list[float]]
    ) -> list[dict[str, Any]]:
        """Process association rules with effectiveness analysis"""
        rule_patterns = []

        for _, row in rules.iterrows():
            antecedents = set(row["antecedents"])
            consequents = set(row["consequents"])
            confidence = row["confidence"]
            lift = row["lift"]

            # Calculate effectiveness impact
            ant_key = tuple(sorted(antecedents))
            cons_key = tuple(sorted(consequents))

            ant_effectiveness = effectiveness_map.get(ant_key, [])
            cons_effectiveness = effectiveness_map.get(cons_key, [])

            if ant_effectiveness and cons_effectiveness:
                effectiveness_impact = np.mean(cons_effectiveness) - np.mean(
                    ant_effectiveness
                )

                rule_info = {
                    "antecedents": list(antecedents),
                    "consequents": list(consequents),
                    "confidence": float(confidence),
                    "lift": float(lift),
                    "effectiveness_impact": float(effectiveness_impact),
                    "support_antecedent": len(ant_effectiveness),
                    "support_consequent": len(cons_effectiveness),
                }
                rule_patterns.append(rule_info)

        return sorted(
            rule_patterns, key=lambda x: x["effectiveness_impact"], reverse=True
        )

    def _process_performance_clusters(
        self,
        cluster_labels: np.ndarray,
        performance_data: list[dict[str, Any]],
        features: np.ndarray,
        outlier_scores: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Process performance clusters"""
        clusters = []
        unique_labels = set(cluster_labels)

        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_data = [
                performance_data[i] for i, mask in enumerate(cluster_mask) if mask
            ]
            cluster_features = features[cluster_mask]

            if len(cluster_data) == 0:
                continue

            # Calculate cluster characteristics
            effectiveness_scores = [d["improvement_score"] for d in cluster_data]
            execution_times = [d["execution_time_ms"] for d in cluster_data]
            satisfaction_scores = [d["user_satisfaction"] for d in cluster_data]

            cluster_info = {
                "cluster_id": int(label),
                "cluster_type": "noise" if label == -1 else "performance_cluster",
                "size": len(cluster_data),
                "characteristics": {
                    "avg_effectiveness": float(np.mean(effectiveness_scores)),
                    "avg_execution_time": float(np.mean(execution_times)),
                    "avg_satisfaction": float(np.mean(satisfaction_scores)),
                    "effectiveness_range": [
                        float(min(effectiveness_scores)),
                        float(max(effectiveness_scores)),
                    ],
                    "execution_time_range": [
                        float(min(execution_times)),
                        float(max(execution_times)),
                    ],
                },
                "centroid": [float(x) for x in np.mean(cluster_features, axis=0)],
                "avg_outlier_score": float(np.mean(outlier_scores[cluster_mask])),
                "representative_rules": [d["rule_name"] for d in cluster_data[:3]],
            }
            clusters.append(cluster_info)

        return sorted(
            clusters,
            key=lambda x: x["characteristics"]["avg_effectiveness"],
            reverse=True,
        )

    def _analyze_semantic_category(
        self, category: str, data_points: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Analyze semantic patterns within a category"""
        try:
            # Calculate category statistics
            effectiveness_scores = [d["improvement_score"] for d in data_points]
            execution_times = [d["execution_time_ms"] for d in data_points]

            # Extract common characteristics
            common_chars = self._extract_common_characteristics(data_points)

            pattern = {
                "category": category,
                "sample_size": len(data_points),
                "avg_effectiveness": float(np.mean(effectiveness_scores)),
                "std_effectiveness": float(np.std(effectiveness_scores)),
                "avg_execution_time": float(np.mean(execution_times)),
                "common_characteristics": common_chars,
                "top_performers": sorted(
                    data_points, key=lambda x: x["improvement_score"], reverse=True
                )[:3],
            }

            return pattern

        except Exception:
            return None

    def _analyze_cross_category_patterns(
        self, category_groups: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Analyze patterns across categories"""
        cross_patterns = []

        # Compare categories
        categories = list(category_groups.keys())
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i + 1 :]:
                if len(category_groups[cat1]) >= 3 and len(category_groups[cat2]) >= 3:
                    pattern = self._compare_categories(
                        cat1, category_groups[cat1], cat2, category_groups[cat2]
                    )
                    if pattern:
                        cross_patterns.append(pattern)

        return cross_patterns

    def _extract_common_parameters(
        self, cluster_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Extract common parameters from cluster data"""
        param_counts = defaultdict(Counter)

        for data_point in cluster_data:
            params = data_point.get("parameters", {})
            metadata_params = data_point.get("metadata_params", {})
            all_params = {**params, **metadata_params}

            for key, value in all_params.items():
                param_counts[key][value] += 1

        # Find most common values for each parameter
        common_params = {}
        for param, value_counts in param_counts.items():
            if value_counts:
                most_common_value, count = value_counts.most_common(1)[0]
                if count / len(cluster_data) >= 0.5:  # At least 50% share this value
                    common_params[param] = {
                        "value": most_common_value,
                        "frequency": count / len(cluster_data),
                    }

        return common_params

    def _analyze_outlier_uniqueness(self, data_point: dict[str, Any]) -> list[str]:
        """Analyze what makes an outlier unique"""
        unique_characteristics = []

        # High performance characteristics
        if data_point["improvement_score"] >= 0.95:
            unique_characteristics.append("exceptional_performance")

        if data_point["execution_time_ms"] <= 5:
            unique_characteristics.append("ultra_fast_execution")

        if data_point["user_satisfaction"] >= 0.9:
            unique_characteristics.append("high_user_satisfaction")

        # Parameter uniqueness
        params = data_point.get("parameters", {})
        if len(params) > 10:
            unique_characteristics.append("complex_parameter_set")

        return unique_characteristics

    def _extract_common_characteristics(
        self, data_points: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Extract common characteristics from data points"""
        characteristics = {}

        # Performance characteristics
        effectiveness_scores = [d["improvement_score"] for d in data_points]
        characteristics["effectiveness_profile"] = {
            "min": float(min(effectiveness_scores)),
            "max": float(max(effectiveness_scores)),
            "avg": float(np.mean(effectiveness_scores)),
            "std": float(np.std(effectiveness_scores)),
        }

        # Common rule names pattern
        rule_names = [d["rule_name"] for d in data_points]
        characteristics["common_rule_patterns"] = Counter(rule_names).most_common(3)

        return characteristics

    def _compare_categories(
        self,
        cat1: str,
        data1: list[dict[str, Any]],
        cat2: str,
        data2: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Compare two categories for cross-patterns"""
        try:
            eff1 = [d["improvement_score"] for d in data1]
            eff2 = [d["improvement_score"] for d in data2]

            # Statistical comparison with proper scipy.stats return type handling
            from scipy.stats import ttest_ind

            ttest_result = ttest_ind(eff1, eff2)
            # Proper handling of scipy TtestResult object (2025 best practice)
            stat_value = float(ttest_result.statistic)  # type: ignore[attr-defined]  # scipy TtestResult
            p_value = float(ttest_result.pvalue)  # type: ignore[attr-defined]  # scipy TtestResult

            pattern = {
                "category_1": cat1,
                "category_2": cat2,
                "avg_effectiveness_1": float(np.mean(eff1)),
                "avg_effectiveness_2": float(np.mean(eff2)),
                "effectiveness_difference": float(np.mean(eff1) - np.mean(eff2)),
                "statistical_significance": p_value,
                "test_statistic": stat_value,
                "sample_sizes": [len(data1), len(data2)],
            }

            return pattern if abs(pattern["effectiveness_difference"]) > 0.1 else None

        except Exception:
            return None

    def _analyze_pattern_ensemble(
        self, pattern_discovery: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze patterns across different discovery methods"""
        ensemble_results = {
            "cross_validation": {},
            "consensus_patterns": [],
            "algorithm_comparison": {},
        }

        # Cross-validate patterns between methods
        if (
            "parameter_patterns" in pattern_discovery
            and "performance_patterns" in pattern_discovery
        ):
            param_clusters = pattern_discovery["parameter_patterns"].get("clusters", [])
            perf_clusters = pattern_discovery["performance_patterns"].get(
                "clusters", []
            )

            # Find overlapping high-performance patterns
            consensus = self._find_consensus_patterns(param_clusters, perf_clusters)
            ensemble_results["consensus_patterns"] = consensus

        return ensemble_results

    def _find_consensus_patterns(
        self, param_clusters: list[dict[str, Any]], perf_clusters: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find consensus patterns between different clustering methods"""
        consensus_patterns = []

        # Find parameter clusters that align with performance clusters
        for param_cluster in param_clusters:
            if param_cluster["avg_effectiveness"] >= 0.8:
                # Look for corresponding performance cluster
                for perf_cluster in perf_clusters:
                    if (
                        abs(
                            param_cluster["avg_effectiveness"]
                            - perf_cluster["characteristics"]["avg_effectiveness"]
                        )
                        < 0.1
                    ):
                        consensus_pattern = {
                            "consensus_type": "high_performance",
                            "parameter_cluster_id": param_cluster["cluster_id"],
                            "performance_cluster_id": perf_cluster["cluster_id"],
                            "avg_effectiveness": (
                                param_cluster["avg_effectiveness"]
                                + perf_cluster["characteristics"]["avg_effectiveness"]
                            )
                            / 2,
                            "confidence": "high",
                        }
                        consensus_patterns.append(consensus_pattern)

        return consensus_patterns

    def _validate_patterns_statistically(
        self, pattern_discovery: dict[str, Any]
    ) -> dict[str, Any]:
        """Statistical validation of discovered patterns"""
        validation_results = {
            "pattern_significance": {},
            "confidence_intervals": {},
            "effect_sizes": {},
        }

        # Validate parameter patterns
        if "parameter_patterns" in pattern_discovery:
            param_validation = self._validate_parameter_patterns(
                pattern_discovery["parameter_patterns"]
            )
            validation_results["parameter_patterns"] = param_validation

        # Validate sequence patterns
        if "sequence_patterns" in pattern_discovery:
            seq_validation = self._validate_sequence_patterns(
                pattern_discovery["sequence_patterns"]
            )
            validation_results["sequence_patterns"] = seq_validation

        return validation_results

    def _validate_parameter_patterns(
        self, param_patterns: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate parameter patterns statistically"""
        validation = {"cluster_stability": {}, "statistical_significance": {}}

        clusters = param_patterns.get("clusters", [])

        for cluster in clusters:
            if cluster["size"] >= 5:  # Minimum for statistical validation
                # Calculate confidence interval for effectiveness
                mean_eff = cluster["avg_effectiveness"]
                std_eff = cluster["std_effectiveness"]
                n = cluster["size"]

                # 95% confidence interval
                margin_error = 1.96 * (std_eff / np.sqrt(n))
                ci_lower = mean_eff - margin_error
                ci_upper = mean_eff + margin_error

                validation["cluster_stability"][cluster["cluster_id"]] = {
                    "confidence_interval": [float(ci_lower), float(ci_upper)],
                    "margin_of_error": float(margin_error),
                    "sample_size": n,
                }

        return validation

    def _validate_sequence_patterns(
        self, seq_patterns: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate sequence patterns statistically"""
        validation = {"pattern_reliability": {}, "support_confidence": {}}

        patterns = seq_patterns.get("frequent_patterns", [])

        for i, pattern in enumerate(patterns):
            if pattern.get("effectiveness_count", 0) >= 3:
                validation["pattern_reliability"][i] = {
                    "support_level": pattern["support"],
                    "effectiveness_reliability": pattern.get("effectiveness_std", 0)
                    < 0.2,
                    "sample_adequacy": pattern.get("effectiveness_count", 0) >= 5,
                }

        return validation

    def _generate_pattern_recommendations(
        self, results: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate actionable recommendations from pattern analysis"""
        recommendations = []

        # Parameter pattern recommendations
        param_patterns = results.get("parameter_patterns", {})
        if param_patterns.get("clusters"):
            best_cluster = max(
                param_patterns["clusters"], key=lambda x: x["avg_effectiveness"]
            )
            if best_cluster["avg_effectiveness"] > 0.8:
                recommendations.append({
                    "type": "parameter_optimization",
                    "priority": "high",
                    "action": "Deploy parameter configuration from top-performing cluster",
                    "details": {
                        "cluster_id": best_cluster["cluster_id"],
                        "expected_improvement": best_cluster["avg_effectiveness"],
                        "common_parameters": best_cluster["common_parameters"],
                    },
                })

        # Sequence pattern recommendations
        seq_patterns = results.get("sequence_patterns", {})
        if seq_patterns.get("frequent_patterns"):
            top_pattern = max(
                seq_patterns["frequent_patterns"], key=lambda x: x["avg_effectiveness"]
            )
            recommendations.append({
                "type": "sequence_optimization",
                "priority": "medium",
                "action": "Implement high-performing rule sequence pattern",
                "details": {
                    "pattern": top_pattern["itemset"],
                    "expected_effectiveness": top_pattern["avg_effectiveness"],
                    "support": top_pattern["support"],
                },
            })

        # Outlier recommendations
        if param_patterns.get("outliers"):
            top_outlier = max(
                param_patterns["outliers"], key=lambda x: x["effectiveness"]
            )
            recommendations.append({
                "type": "outlier_investigation",
                "priority": "medium",
                "action": "Investigate high-performing outlier for unique insights",
                "details": {
                    "rule_name": top_outlier["rule_name"],
                    "effectiveness": top_outlier["effectiveness"],
                    "unique_characteristics": top_outlier["unique_characteristics"],
                },
            })

        return recommendations

    def _configure_hdbscan_performance(self):
        """Configure HDBSCAN performance settings based on system capabilities."""
        try:
            # Set optimal number of parallel jobs based on CPU count
            import os

            self._optimal_n_jobs = min(os.cpu_count() or 1, 4)  # Cap at 4 for stability

            # Configure BLAS/LAPACK threads for NumPy optimization
            os.environ.setdefault("OMP_NUM_THREADS", str(self._optimal_n_jobs))
            os.environ.setdefault("OPENBLAS_NUM_THREADS", str(self._optimal_n_jobs))
            os.environ.setdefault("MKL_NUM_THREADS", str(self._optimal_n_jobs))

            logger.info(
                f"HDBSCAN performance configured: {self._optimal_n_jobs} parallel jobs"
            )

        except Exception as e:
            logger.warning(f"Failed to configure HDBSCAN performance: {e}")
            self._optimal_n_jobs = 1

    def _get_optimal_n_jobs(self) -> int:
        """Get optimal number of parallel jobs for HDBSCAN."""
        return getattr(self, "_optimal_n_jobs", 1)

    async def benchmark_clustering_performance(
        self,
        dataset_sizes: list[int] | None = None,
        max_time: int = 45,
        sample_size: int = 2,
    ) -> dict[str, Any]:
        """Benchmark HDBSCAN performance on different dataset sizes.

        Based on Context7 research for performance optimization.

        Args:
            dataset_sizes: List of dataset sizes to benchmark
            max_time: Maximum time per benchmark (seconds)
            sample_size: Number of samples per dataset size

        Returns:
            Performance benchmark results
        """
        import time

        from sklearn.datasets import make_blobs

        if dataset_sizes is None:
            dataset_sizes = [100, 500, 1000, 2000, 5000, 10000]

        results = {}

        for size in dataset_sizes:
            size_results = []

            for sample in range(sample_size):
                try:
                    # Generate synthetic data - fix tuple unpacking
                    blobs_result = make_blobs(n_samples=size, n_features=10, centers=5)
                    data = blobs_result[0]  # First element is the data

                    # Benchmark HDBSCAN
                    start_time = time.time()

                    if HDBSCAN_AVAILABLE:
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=max(5, size // 100),
                            algorithm="boruvka_kdtree",
                            core_dist_n_jobs=self._get_optimal_n_jobs(),
                        )
                        clusterer.fit(data)

                    execution_time = time.time() - start_time

                    if execution_time > max_time:
                        logger.warning(f"Benchmark timeout at size {size}")
                        break

                    size_results.append(execution_time)

                except Exception as e:
                    logger.error(f"Benchmark failed for size {size}: {e}")

            if size_results:
                results[size] = {
                    "avg_time": np.mean(size_results),
                    "min_time": min(size_results),
                    "max_time": max(size_results),
                    "samples": len(size_results),
                }

        return {
            "status": "success",
            "algorithm": "HDBSCAN-Boruvka",
            "parallel_jobs": self._get_optimal_n_jobs(),
            "results": results,
            "performance_summary": self._analyze_performance_results(results),
        }

    def _analyze_performance_results(self, results: dict) -> dict[str, Any]:
        """Analyze performance benchmark results."""
        if not results:
            return {"message": "No performance data available"}

        sizes = list(results.keys())
        times = [results[size]["avg_time"] for size in sizes]

        # Estimate complexity
        if len(times) >= 3:
            # Simple linear regression to estimate scaling
            from scipy.stats import linregress

            log_sizes = np.log(sizes)
            log_times = np.log(times)
            slope, intercept, r_value, p_value, std_err = linregress(
                log_sizes, log_times
            )

            complexity_estimate = f"O(n^{slope:.2f})"
        else:
            complexity_estimate = "Insufficient data"

        return {
            "max_dataset_size": max(sizes),
            "fastest_time": min(times),
            "slowest_time": max(times),
            "complexity_estimate": complexity_estimate,
            "scalability_rating": "Good"
            if max(times) < 10
            else "Moderate"
            if max(times) < 60
            else "Poor",
        }

    async def _discover_apriori_patterns(
        self, performance_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Discover association patterns using Apriori algorithm.

        This method converts performance data into transactions and mines
        association rules to discover relationships between:
        - Rule combinations and their effectiveness
        - Prompt characteristics and success patterns
        - Performance indicators and user satisfaction

        Args:
            performance_data: Rule performance data

        Returns:
            Dictionary with Apriori analysis results and business insights
        """
        logger.info("Starting Apriori pattern discovery")

        if not self._ensure_apriori_analyzer():
            logger.warning("AprioriAnalyzer not available - skipping Apriori analysis")
            return {"error": "AprioriAnalyzer not initialized"}

        try:
            # Perform comprehensive Apriori analysis
            apriori_results = self.apriori_analyzer.analyze_patterns(
                window_days=30, save_to_database=True
            )

            if "error" in apriori_results:
                logger.warning(f"Apriori analysis error: {apriori_results['error']}")
                return apriori_results

            # Convert results to standardized pattern format
            apriori_patterns = []

            # Process top association rules
            for rule in apriori_results.get("top_rules", []):
                pattern = AprioriPattern(
                    antecedents=rule["antecedents"],
                    consequents=rule["consequents"],
                    support=rule["support"],
                    confidence=rule["confidence"],
                    lift=rule["lift"],
                    conviction=rule.get("conviction", 0),
                    rule_strength=rule["rule_strength"],
                    business_insight=self._generate_business_insight(rule),
                    pattern_category=self._categorize_apriori_pattern(rule),
                )
                apriori_patterns.append(pattern)

            # Analyze performance data for additional context
            performance_context = self._analyze_apriori_performance_context(
                performance_data, apriori_results
            )

            return {
                "patterns": [
                    self._apriori_pattern_to_dict(p) for p in apriori_patterns
                ],
                "transaction_count": apriori_results.get("transaction_count", 0),
                "frequent_itemsets_count": apriori_results.get(
                    "frequent_itemsets_count", 0
                ),
                "association_rules_count": apriori_results.get(
                    "association_rules_count", 0
                ),
                "top_itemsets": apriori_results.get("top_itemsets", []),
                "pattern_insights": apriori_results.get("pattern_insights", {}),
                "performance_context": performance_context,
                "config": apriori_results.get("config", {}),
                "discovery_type": "apriori_association_rules",
                "algorithm": "mlxtend_apriori",
                "timestamp": apriori_results.get("timestamp"),
            }

        except Exception as e:
            logger.error(f"Error in Apriori pattern discovery: {e}")
            return {"error": f"Apriori analysis failed: {e!s}"}

    def _generate_business_insight(self, rule: dict[str, Any]) -> str:
        """Generate human-readable business insight from association rule."""
        antecedents = rule["antecedents"]
        consequents = rule["consequents"]
        confidence = rule["confidence"]
        lift = rule["lift"]

        # Analyze rule components for insight generation
        rule_insights = []

        # Check for rule combination patterns
        rules_in_antecedent = [item for item in antecedents if item.startswith("rule_")]
        rules_in_consequent = [item for item in consequents if item.startswith("rule_")]

        # Fix any() function calls - check string content directly
        consequents_str = str(consequents)
        if rules_in_antecedent and (
            "performance_high" in consequents_str or "quality_high" in consequents_str
        ):
            rule_names = [r.replace("rule_", "") for r in rules_in_antecedent]
            insight = f"Applying {', '.join(rule_names)} leads to high performance with {confidence:.1%} confidence"
            rule_insights.append(insight)

        # Check for prompt characteristic patterns
        antecedents_str = str(antecedents)
        if "domain_" in antecedents_str and "quality_" in consequents_str:
            insight = (
                f"Specific prompt domains show quality improvements (lift: {lift:.2f}x)"
            )
            rule_insights.append(insight)

        # Check for user satisfaction patterns
        if "feedback_positive" in consequents_str:
            insight = f"These patterns lead to positive user feedback with {confidence:.1%} certainty"
            rule_insights.append(insight)

        return (
            "; ".join(rule_insights)
            if rule_insights
            else f"Pattern shows {lift:.2f}x lift with {confidence:.1%} confidence"
        )

    def _categorize_apriori_pattern(self, rule: dict[str, Any]) -> str:
        """Categorize Apriori patterns for better organization."""
        antecedents = str(rule["antecedents"])
        consequents = str(rule["consequents"])

        if "rule_" in antecedents and "performance_" in consequents:
            return "rule_performance"
        if "domain_" in antecedents and "quality_" in consequents:
            return "domain_quality"
        if "complexity_" in antecedents:
            return "complexity_patterns"
        if "feedback_" in consequents:
            return "user_satisfaction"
        if "length_" in antecedents:
            return "prompt_structure"
        return "general_association"

    def _analyze_apriori_performance_context(
        self, performance_data: list[dict[str, Any]], apriori_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze how Apriori patterns relate to performance data."""
        context = {
            "performance_data_points": len(performance_data),
            "average_effectiveness": np.mean([
                d.get("effectiveness", 0) for d in performance_data
            ]),
            "rule_coverage": self._calculate_rule_coverage(
                performance_data, apriori_results
            ),
            "pattern_validation": self._validate_apriori_against_performance(
                performance_data, apriori_results
            ),
        }

        return context

    def _calculate_rule_coverage(
        self, performance_data: list[dict[str, Any]], apriori_results: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate how well Apriori patterns cover the performance data."""
        total_sessions = len(performance_data)
        if total_sessions == 0:
            return {"coverage": 0.0}

        # Extract rules from performance data
        rules_in_performance = set()
        for data_point in performance_data:
            rule_name = data_point.get("rule_name")
            if rule_name:
                rules_in_performance.add(f"rule_{rule_name}")

        # Extract rules from Apriori patterns
        rules_in_apriori = set()
        for pattern in apriori_results.get("pattern_insights", {}).get(
            "rule_performance_patterns", []
        ):
            if "Rule " in pattern:
                # Extract rule name from pattern text
                rule_start = pattern.find("Rule ") + 5
                rule_end = pattern.find(" →")
                if rule_end > rule_start:
                    rule_name = pattern[rule_start:rule_end]
                    rules_in_apriori.add(f"rule_{rule_name}")

        # Calculate coverage
        if rules_in_performance:
            coverage = len(rules_in_apriori.intersection(rules_in_performance)) / len(
                rules_in_performance
            )
        else:
            coverage = 0.0

        return {
            "coverage": coverage,
            "rules_in_performance": len(rules_in_performance),
            "rules_in_apriori": len(rules_in_apriori),
            "overlap": len(rules_in_apriori.intersection(rules_in_performance)),
        }

    def _validate_apriori_against_performance(
        self, performance_data: list[dict[str, Any]], apriori_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate Apriori patterns against actual performance data."""
        validation_results = {
            "pattern_accuracy": 0.0,
            "prediction_confidence": 0.0,
            "business_value_score": 0.0,
        }

        # Extract high-confidence patterns for validation
        high_conf_patterns = []
        for rule in apriori_results.get("top_rules", []):
            if rule["confidence"] >= 0.7 and rule["lift"] >= 1.5:
                high_conf_patterns.append(rule)

        if high_conf_patterns:
            # Calculate average confidence and lift for validation
            avg_confidence = np.mean([p["confidence"] for p in high_conf_patterns])
            avg_lift = np.mean([p["lift"] for p in high_conf_patterns])

            # Fix dict.update() and min() type issues
            validation_results["pattern_accuracy"] = float(avg_confidence)
            validation_results["prediction_confidence"] = float(
                min(float(avg_confidence * avg_lift / 2), 1.0)
            )
            validation_results["business_value_score"] = float(
                len(high_conf_patterns)
                / max(len(apriori_results.get("top_rules", [])), 1)
            )
            validation_results["high_confidence_patterns"] = len(high_conf_patterns)

        return validation_results

    def _apriori_pattern_to_dict(self, pattern: AprioriPattern) -> dict[str, Any]:
        """Convert AprioriPattern dataclass to dictionary."""
        return {
            "antecedents": pattern.antecedents,
            "consequents": pattern.consequents,
            "support": pattern.support,
            "confidence": pattern.confidence,
            "lift": pattern.lift,
            "conviction": pattern.conviction,
            "rule_strength": pattern.rule_strength,
            "business_insight": pattern.business_insight,
            "pattern_category": pattern.pattern_category,
            "type": "apriori_association_rule",
        }

    async def get_contextualized_patterns(
        self,
        context_items: list[str],
        db_session: AsyncSession,
        min_confidence: float = 0.6,
    ) -> dict[str, Any]:
        """Get patterns relevant to a specific context using Apriori analysis.

        This method leverages association rules to find patterns relevant
        to the current prompt improvement context.

        Args:
            context_items: Items representing current context (rules, characteristics)
            db_session: Database session
            min_confidence: Minimum confidence for returned patterns

        Returns:
            Dictionary with contextualized patterns and recommendations
        """
        if not self._ensure_apriori_analyzer():
            logger.warning("AprioriAnalyzer not available for contextualized patterns")
            return {"error": "AprioriAnalyzer not initialized"}

        try:
            # Get relevant association rules
            relevant_rules = self.apriori_analyzer.get_rules_for_context(
                context_items, min_confidence
            )

            # Get performance data for additional context
            performance_data = await self._get_performance_data(db_session, 0.5)

            # Generate contextualized recommendations
            recommendations = self._generate_contextualized_recommendations(
                relevant_rules, context_items, performance_data
            )

            return {
                "context_items": context_items,
                "relevant_rules": relevant_rules,
                "recommendations": recommendations,
                "rule_count": len(relevant_rules),
                "context_coverage": self._calculate_context_coverage(
                    relevant_rules, context_items
                ),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error getting contextualized patterns: {e}")
            return {"error": f"Contextualized pattern analysis failed: {e!s}"}

    def _generate_contextualized_recommendations(
        self,
        relevant_rules: list[dict[str, Any]],
        context_items: list[str],
        performance_data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on context and association rules."""
        recommendations = []

        for rule in relevant_rules:
            # Analyze rule for actionable recommendations
            consequents = rule["consequents"]
            confidence = rule["confidence"]
            lift = rule["lift"]

            # Generate specific recommendations
            if any("quality_high" in consequent for consequent in consequents):
                recommendations.append({
                    "type": "quality_improvement",
                    "action": f"Apply pattern {rule['antecedents']} for quality improvement",
                    "confidence": confidence,
                    "expected_lift": lift,
                    "priority": "high" if confidence > 0.8 else "medium",
                })

            if any("performance_high" in consequent for consequent in consequents):
                recommendations.append({
                    "type": "performance_optimization",
                    "action": f"Use combination {rule['antecedents']} for performance gains",
                    "confidence": confidence,
                    "expected_lift": lift,
                    "priority": "high" if lift > 2.0 else "medium",
                })

        # Sort by confidence and lift
        recommendations.sort(
            key=lambda x: (x["confidence"], x["expected_lift"]), reverse=True
        )

        return recommendations[:10]  # Return top 10 recommendations

    def _calculate_context_coverage(
        self, relevant_rules: list[dict[str, Any]], context_items: list[str]
    ) -> float:
        """Calculate how well the rules cover the provided context."""
        if not relevant_rules or not context_items:
            return 0.0

        context_set = set(context_items)
        covered_items = set()

        for rule in relevant_rules:
            antecedents = set(rule["antecedents"])
            covered_items.update(antecedents.intersection(context_set))

        return len(covered_items) / len(context_set) if context_items else 0.0

    # Phase 2 Training Data Integration Methods
    
    async def discover_training_data_patterns(
        self,
        db_session: AsyncSession,
        pattern_types: list[str] | None = None,
        min_effectiveness: float = 0.7,
        use_clustering: bool = True,
        include_feature_patterns: bool = True
    ) -> dict[str, Any]:
        """Discover patterns in training data using advanced ML techniques
        
        Phase 2 Enhancement: Analyzes training data to discover patterns
        that lead to improved rule effectiveness and optimization opportunities.
        
        Args:
            db_session: Database session for training data access
            pattern_types: Types of patterns to discover
            min_effectiveness: Minimum effectiveness threshold
            use_clustering: Enable clustering-based pattern discovery
            include_feature_patterns: Include feature-level pattern analysis
            
        Returns:
            Comprehensive training data pattern analysis
        """
        try:
            logger.info("Starting training data pattern discovery")
            
            # Load training data from pipeline
            training_data = await self.training_loader.load_training_data(db_session)
            
            # Check if training data validation passed
            if not training_data.get("validation", {}).get("is_valid", False):
                logger.warning("Insufficient training data for pattern discovery")
                return {
                    "status": "insufficient_data",
                    "samples": training_data["metadata"]["total_samples"],
                    "message": "Insufficient training data for reliable pattern discovery"
                }
            
            results = {
                "status": "success",
                "training_metadata": training_data["metadata"],
                "patterns": {},
                "insights": [],
                "recommendations": []
            }
            
            # Extract features and labels
            features = np.array(training_data["features"])
            labels = np.array(training_data["labels"])
            
            if pattern_types is None:
                pattern_types = ["clustering", "feature_importance", "effectiveness"]
            
            # Clustering-based pattern discovery
            if "clustering" in pattern_types and use_clustering:
                clustering_patterns = await self._discover_training_clustering_patterns(
                    features, labels, min_effectiveness
                )
                results["patterns"]["clustering"] = clustering_patterns
                
                if clustering_patterns["status"] == "success":
                    results["insights"].append(
                        f"Discovered {clustering_patterns['n_clusters']} distinct pattern clusters"
                    )
            
            # Feature importance patterns
            if "feature_importance" in pattern_types and include_feature_patterns:
                feature_patterns = await self._discover_feature_importance_patterns(
                    features, labels
                )
                results["patterns"]["feature_importance"] = feature_patterns
                
                if feature_patterns["status"] == "success":
                    top_features = feature_patterns["top_features"][:3]
                    results["insights"].append(
                        f"Top influential features: {', '.join([f'Feature {i}' for i in top_features])}"
                    )
            
            # Effectiveness patterns
            if "effectiveness" in pattern_types:
                effectiveness_patterns = await self._discover_effectiveness_patterns(
                    features, labels, min_effectiveness
                )
                results["patterns"]["effectiveness"] = effectiveness_patterns
                
                if effectiveness_patterns["status"] == "success":
                    high_eff_ratio = effectiveness_patterns.get("high_effectiveness_ratio", 0.0)
                    results["insights"].append(
                        f"High effectiveness patterns found in {high_eff_ratio:.1%} of training data"
                    )
            
            # Generate recommendations
            results["recommendations"] = self._generate_training_pattern_recommendations(
                results["patterns"], training_data["metadata"]
            )
            
            logger.info(f"Training data pattern discovery completed: {len(results['patterns'])} pattern types found")
            return results
            
        except Exception as e:
            logger.error(f"Error in training data pattern discovery: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to discover patterns in training data"
            }
    
    async def _discover_training_clustering_patterns(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        min_effectiveness: float
    ) -> dict[str, Any]:
        """Discover patterns using clustering analysis on training data"""
        try:
            if len(features) < 10:
                return {"status": "insufficient_data", "samples": len(features)}
            
            # Standardize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Use HDBSCAN if available, otherwise DBSCAN
            if HDBSCAN_AVAILABLE:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=max(5, len(features) // 20),
                    min_samples=3,
                    cluster_selection_epsilon=0.1
                )
            else:
                clusterer = DBSCAN(eps=0.5, min_samples=5)
            
            cluster_labels = clusterer.fit_predict(features_scaled)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            if n_clusters < 2:
                return {"status": "no_clusters_found", "n_clusters": n_clusters}
            
            # Analyze cluster effectiveness
            cluster_analysis = []
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_mask = cluster_labels == cluster_id
                cluster_effectiveness = labels[cluster_mask]
                
                cluster_info = {
                    "cluster_id": int(cluster_id),
                    "size": int(np.sum(cluster_mask)),
                    "avg_effectiveness": float(np.mean(cluster_effectiveness)),
                    "std_effectiveness": float(np.std(cluster_effectiveness)),
                    "min_effectiveness": float(np.min(cluster_effectiveness)),
                    "max_effectiveness": float(np.max(cluster_effectiveness)),
                    "high_performance": float(np.mean(cluster_effectiveness >= min_effectiveness))
                }
                cluster_analysis.append(cluster_info)
            
            # Sort by average effectiveness
            cluster_analysis.sort(key=lambda x: x["avg_effectiveness"], reverse=True)
            
            return {
                "status": "success",
                "n_clusters": n_clusters,
                "noise_points": int(np.sum(cluster_labels == -1)),
                "cluster_analysis": cluster_analysis,
                "best_cluster": cluster_analysis[0] if cluster_analysis else None
            }
            
        except Exception as e:
            logger.error(f"Error in training clustering pattern discovery: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _discover_feature_importance_patterns(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> dict[str, Any]:
        """Discover feature importance patterns in training data"""
        try:
            if len(features) < 10 or features.shape[1] == 0:
                return {"status": "insufficient_data", "samples": len(features)}
            
            # Calculate correlation-based feature importance
            feature_importance = []
            for i in range(features.shape[1]):
                correlation = np.corrcoef(features[:, i], labels)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                feature_importance.append({
                    "feature_index": i,
                    "correlation": float(correlation),
                    "abs_correlation": float(abs(correlation)),
                    "feature_std": float(np.std(features[:, i])),
                    "feature_mean": float(np.mean(features[:, i]))
                })
            
            # Sort by absolute correlation
            feature_importance.sort(key=lambda x: x["abs_correlation"], reverse=True)
            
            # Identify top features
            top_features = [f["feature_index"] for f in feature_importance[:10]]
            
            return {
                "status": "success",
                "feature_importance": feature_importance,
                "top_features": top_features,
                "max_correlation": feature_importance[0]["abs_correlation"] if feature_importance else 0.0,
                "n_significant_features": sum(1 for f in feature_importance if f["abs_correlation"] > 0.1)
            }
            
        except Exception as e:
            logger.error(f"Error in feature importance pattern discovery: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _discover_effectiveness_patterns(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        min_effectiveness: float
    ) -> dict[str, Any]:
        """Discover effectiveness-based patterns in training data"""
        try:
            if len(labels) < 10:
                return {"status": "insufficient_data", "samples": len(labels)}
            
            # Analyze effectiveness distribution
            high_effectiveness_mask = labels >= min_effectiveness
            n_high_effectiveness = np.sum(high_effectiveness_mask)
            
            effectiveness_stats = {
                "mean_effectiveness": float(np.mean(labels)),
                "std_effectiveness": float(np.std(labels)),
                "min_effectiveness": float(np.min(labels)),
                "max_effectiveness": float(np.max(labels)),
                "median_effectiveness": float(np.median(labels)),
                "high_effectiveness_count": int(n_high_effectiveness),
                "high_effectiveness_ratio": float(n_high_effectiveness / len(labels))
            }
            
            # Identify effectiveness patterns
            percentiles = [25, 50, 75, 90, 95]
            effectiveness_percentiles = {
                f"p{p}": float(np.percentile(labels, p)) for p in percentiles
            }
            
            return {
                "status": "success",
                "effectiveness_stats": effectiveness_stats,
                "effectiveness_percentiles": effectiveness_percentiles,
                "high_effectiveness_threshold": min_effectiveness
            }
            
        except Exception as e:
            logger.error(f"Error in effectiveness pattern discovery: {e}")
            return {"status": "error", "error": str(e)}
    
    def _generate_training_pattern_recommendations(
        self,
        patterns: dict[str, Any],
        training_metadata: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on training pattern analysis"""
        recommendations = []
        
        try:
            # Clustering recommendations
            if "clustering" in patterns and patterns["clustering"].get("status") == "success":
                n_clusters = patterns["clustering"]["n_clusters"]
                best_cluster = patterns["clustering"].get("best_cluster")
                
                recommendations.append(f"🔍 Pattern Discovery: Found {n_clusters} distinct performance clusters")
                
                if best_cluster and best_cluster["avg_effectiveness"] > 0.8:
                    recommendations.append(
                        f"🎆 High-Performance Cluster: Cluster {best_cluster['cluster_id']} shows {best_cluster['avg_effectiveness']:.2f} avg effectiveness"
                    )
            
            # Feature importance recommendations
            if "feature_importance" in patterns and patterns["feature_importance"].get("status") == "success":
                max_correlation = patterns["feature_importance"]["max_correlation"]
                n_significant = patterns["feature_importance"]["n_significant_features"]
                
                if max_correlation > 0.3:
                    recommendations.append(
                        f"📊 Strong features: {n_significant} features show significant correlation (max: {max_correlation:.3f})"
                    )
                    recommendations.append(
                        "⚙️ Optimization Focus: Prioritize top correlated features for rule tuning"
                    )
            
            # Effectiveness recommendations
            if "effectiveness" in patterns and patterns["effectiveness"].get("status") == "success":
                stats = patterns["effectiveness"].get("effectiveness_stats", {})
                high_ratio = stats.get("high_effectiveness_ratio", 0.0)
                
                if high_ratio > 0.7:
                    recommendations.append(
                        f"✅ Strong Performance: {high_ratio:.1%} of training data shows high effectiveness"
                    )
                elif high_ratio < 0.3:
                    recommendations.append(
                        f"⚠️ Performance Opportunity: Only {high_ratio:.1%} of data shows high effectiveness - room for improvement"
                    )
            
            # Training data quality recommendations
            total_samples = training_metadata.get("total_samples", 0)
            synthetic_ratio = training_metadata.get("synthetic_ratio", 0)
            
            if total_samples > 1000:
                recommendations.append(
                    f"📈 Rich Dataset: {total_samples} samples provide strong pattern discovery foundation"
                )
            
            if synthetic_ratio > 0.3:
                recommendations.append(
                    f"🤖 Synthetic Data: {synthetic_ratio:.1%} synthetic data - consider increasing real data collection"
                )
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            recommendations.append("⚠️ Recommendation Generation: Some recommendations could not be generated")
        
        return recommendations

# Singleton instance for easy access
_advanced_pattern_discovery = None

async def get_advanced_pattern_discovery(db_manager = None) -> AdvancedPatternDiscovery:
    """Get singleton AdvancedPatternDiscovery instance with lazy initialization support.
    
    Args:
        db_manager: Optional DatabaseManager instance for dependency injection.
                   If None, will be created lazily when first needed.
    
    Returns:
        AdvancedPatternDiscovery instance with proper dependency injection
    """
    global _advanced_pattern_discovery
    if _advanced_pattern_discovery is None:
        _advanced_pattern_discovery = AdvancedPatternDiscovery(db_manager=db_manager)
    return _advanced_pattern_discovery
