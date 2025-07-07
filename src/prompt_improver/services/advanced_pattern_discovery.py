"""Advanced Pattern Discovery Service for Phase 4 ML Enhancement & Discovery
Modern 2025 implementation with HDBSCAN, FP-Growth, and ensemble pattern mining
"""

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import RuleMetadata, RulePerformance

logger = logging.getLogger(__name__)


@dataclass
class PatternCluster:
    """Advanced pattern cluster with metadata"""

    cluster_id: int
    pattern_type: str  # 'parameter', 'sequence', 'performance', 'semantic'
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


class AdvancedPatternDiscovery:
    """Advanced Pattern Discovery using modern 2025 techniques:

    - HDBSCAN for varying density cluster discovery
    - FP-Growth for frequent pattern mining (faster than Apriori)
    - Ensemble pattern mining with multiple algorithms
    - Semantic pattern analysis for rule parameter relationships
    - Outlier-based discovery for unique high-performing patterns
    - Statistical validation with confidence intervals
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.min_cluster_size = 5
        self.min_support = 0.1
        self.min_confidence = 0.5
        self.min_effectiveness = 0.7

        # HDBSCAN Performance Optimization Settings (based on Context7 research)
        self._configure_hdbscan_performance()

    async def discover_advanced_patterns(
        self,
        db_session: AsyncSession,
        min_effectiveness: float = 0.7,
        min_support: int = 5,
        pattern_types: list[str] = None,
        use_ensemble: bool = True,
    ) -> dict[str, Any]:
        """Comprehensive pattern discovery using ensemble of modern algorithms.

        Args:
            db_session: Database session
            min_effectiveness: Minimum effectiveness threshold
            min_support: Minimum support count
            pattern_types: Types of patterns to discover ['parameter', 'sequence', 'performance', 'semantic']
            use_ensemble: Whether to use ensemble of algorithms

        Returns:
            Comprehensive pattern discovery results
        """
        start_time = time.time()

        if pattern_types is None:
            pattern_types = ["parameter", "sequence", "performance", "semantic"]

        try:
            # Get performance data for analysis
            performance_data = await self._get_performance_data(
                db_session, min_effectiveness
            )

            if len(performance_data) < min_support:
                return {
                    "status": "insufficient_data",
                    "message": f"Only {len(performance_data)} samples found (minimum: {min_support})",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                }

            results = {
                "status": "success",
                "total_samples": len(performance_data),
                "pattern_discovery": {},
                "ensemble_results": {},
                "statistical_validation": {},
                "recommendations": [],
            }

            # 1. Parameter Pattern Discovery (HDBSCAN-based)
            if "parameter" in pattern_types:
                param_patterns = await self._discover_parameter_patterns(
                    performance_data
                )
                results["pattern_discovery"]["parameter_patterns"] = param_patterns

            # 2. Sequence Pattern Discovery (FP-Growth)
            if "sequence" in pattern_types:
                sequence_patterns = await self._discover_sequence_patterns(
                    performance_data
                )
                results["pattern_discovery"]["sequence_patterns"] = sequence_patterns

            # 3. Performance Pattern Discovery (Density-based clustering)
            if "performance" in pattern_types:
                perf_patterns = await self._discover_performance_patterns(
                    performance_data
                )
                results["pattern_discovery"]["performance_patterns"] = perf_patterns

            # 4. Semantic Pattern Discovery (Advanced clustering)
            if "semantic" in pattern_types:
                semantic_patterns = await self._discover_semantic_patterns(
                    performance_data
                )
                results["pattern_discovery"]["semantic_patterns"] = semantic_patterns

            # 5. Ensemble Pattern Analysis
            if use_ensemble and len(pattern_types) > 1:
                ensemble_results = self._analyze_pattern_ensemble(
                    results["pattern_discovery"]
                )
                results["ensemble_results"] = ensemble_results

            # 6. Statistical Validation
            validation_results = self._validate_patterns_statistically(
                results["pattern_discovery"]
            )
            results["statistical_validation"] = validation_results

            # 7. Generate Actionable Recommendations
            recommendations = self._generate_pattern_recommendations(results)
            results["recommendations"] = recommendations

            results["processing_time_ms"] = (time.time() - start_time) * 1000

            logger.info(
                f"Advanced pattern discovery completed: {len(results['pattern_discovery'])} pattern types analyzed"
            )

            return results

        except Exception as e:
            logger.error(f"Advanced pattern discovery failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    async def _get_performance_data(
        self, db_session: AsyncSession, min_effectiveness: float
    ) -> list[dict[str, Any]]:
        """Get comprehensive performance data for pattern analysis"""
        try:
            # Get rule performance with metadata
            stmt = (
                select(
                    RulePerformance.rule_id,
                    RulePerformance.rule_name,
                    RulePerformance.improvement_score,
                    RulePerformance.execution_time_ms,
                    RulePerformance.user_satisfaction_score,
                    RulePerformance.confidence_level,
                    RulePerformance.rule_parameters,
                    RulePerformance.before_metrics,
                    RulePerformance.after_metrics,
                    RulePerformance.prompt_characteristics,
                    RulePerformance.created_at,
                    RuleMetadata.parameters.label("rule_metadata_params"),
                    RuleMetadata.rule_category,
                    RuleMetadata.weight,
                    RuleMetadata.priority,
                )
                .join(
                    RuleMetadata,
                    RulePerformance.rule_id == RuleMetadata.rule_id,
                    isouter=True,
                )
                .where(RulePerformance.improvement_score >= min_effectiveness)
                .order_by(RulePerformance.created_at.desc())
                .limit(1000)
            )  # Limit for performance

            result = await db_session.execute(stmt)
            rows = result.fetchall()

            performance_data = []
            for row in rows:
                # Convert to comprehensive feature dictionary
                data_point = {
                    "rule_id": row.rule_id,
                    "rule_name": row.rule_name,
                    "rule_category": row.rule_category or "general",
                    "improvement_score": row.improvement_score or 0.0,
                    "execution_time_ms": row.execution_time_ms or 0,
                    "user_satisfaction": row.user_satisfaction_score or 0.0,
                    "confidence_level": row.confidence_level or 0.0,
                    "weight": row.weight or 1.0,
                    "priority": row.priority or 5,
                    "parameters": row.rule_parameters or {},
                    "metadata_params": row.rule_metadata_params or {},
                    "before_metrics": row.before_metrics or {},
                    "after_metrics": row.after_metrics or {},
                    "prompt_characteristics": row.prompt_characteristics or {},
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

            # Apply Transaction Encoder
            te = TransactionEncoder()
            te_data = te.fit(transactions).transform(transactions)
            df = pd.DataFrame(te_data, columns=te.columns_)

            # Apply FP-Growth algorithm
            frequent_itemsets = fpgrowth(
                df, min_support=self.min_support, use_colnames=True
            )

            if frequent_itemsets.empty:
                return {
                    "frequent_patterns": [],
                    "association_rules": [],
                    "message": "No frequent patterns found",
                }

            # Generate association rules
            try:
                rules = association_rules(
                    frequent_itemsets,
                    metric="confidence",
                    min_threshold=self.min_confidence,
                )
            except Exception:
                rules = pd.DataFrame()  # Empty rules if generation fails

            # Process patterns with effectiveness analysis
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

            # Statistical comparison
            from scipy.stats import ttest_ind

            stat, p_value = ttest_ind(eff1, eff2)

            pattern = {
                "category_1": cat1,
                "category_2": cat2,
                "avg_effectiveness_1": float(np.mean(eff1)),
                "avg_effectiveness_2": float(np.mean(eff2)),
                "effectiveness_difference": float(np.mean(eff1) - np.mean(eff2)),
                "statistical_significance": float(p_value),
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
        param_patterns = results.get("pattern_discovery", {}).get(
            "parameter_patterns", {}
        )
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
        seq_patterns = results.get("pattern_discovery", {}).get("sequence_patterns", {})
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
        self, dataset_sizes: list[int] = None, max_time: int = 45, sample_size: int = 2
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
                    # Generate synthetic data
                    data, _ = make_blobs(n_samples=size, n_features=10, centers=5)

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


# Singleton instance for easy access
_advanced_pattern_discovery = None


async def get_advanced_pattern_discovery() -> AdvancedPatternDiscovery:
    """Get singleton AdvancedPatternDiscovery instance"""
    global _advanced_pattern_discovery
    if _advanced_pattern_discovery is None:
        _advanced_pattern_discovery = AdvancedPatternDiscovery()
    return _advanced_pattern_discovery
