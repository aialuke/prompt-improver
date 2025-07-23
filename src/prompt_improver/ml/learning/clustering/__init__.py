"""Clustering Components

Specialized clustering engines for context analysis.
Enhanced ClusteringOptimizer with merged ContextClusteringEngine features.
"""

# Import enhanced clustering components from optimization module
from ...optimization.algorithms.clustering_optimizer import (
    ClusteringOptimizer as ContextClusteringEngine,
    ClusteringConfig,
    ClusteringResult
)

__all__ = [
    "ContextClusteringEngine",  # Now points to enhanced ClusteringOptimizer
    "ClusteringConfig", 
    "ClusteringResult"
]
