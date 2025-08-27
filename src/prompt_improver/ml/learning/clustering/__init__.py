"""Clustering Components

Specialized clustering engines for context analysis.
Enhanced ClusteringOptimizer with merged ContextClusteringEngine features.
"""
from ...clustering.services.clustering_optimizer_facade import ClusteringOptimizerFacade as ContextClusteringEngine
from ...clustering.services import ClusteringResult
# Define config for clustering operations
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ClusteringConfig:
    """Configuration for clustering operations."""
    algorithm: str = "hdbscan"
    min_cluster_size: int = 5
    min_samples: int = None
    metric: str = "euclidean"
    cluster_selection_epsilon: float = 0.0
    alpha: float = 1.0
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = False
    max_cluster_size: Optional[int] = None
    prediction_data: bool = False
    memory_optimization: bool = True
__all__ = ['ContextClusteringEngine', 'ClusteringConfig', 'ClusteringResult']
