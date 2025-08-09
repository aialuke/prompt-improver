"""Clustering Components

Specialized clustering engines for context analysis.
Enhanced ClusteringOptimizer with merged ContextClusteringEngine features.
"""
from ...optimization.algorithms.clustering_optimizer import ClusteringConfig, ClusteringOptimizer as ContextClusteringEngine, ClusteringResult
__all__ = ['ContextClusteringEngine', 'ClusteringConfig', 'ClusteringResult']
