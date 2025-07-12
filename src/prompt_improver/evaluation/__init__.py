"""Evaluation and Statistical Analysis Module

This module provides comprehensive evaluation tools and statistical analysis
for prompt improvement systems, including:

- Statistical validation and hypothesis testing
- Structural analysis of prompts
- Quality assessment and metrics
- Performance evaluation frameworks
"""

from .statistical_analyzer import StatisticalAnalyzer
from .structural_analyzer import StructuralAnalyzer

__all__ = [
    "StatisticalAnalyzer",
    "StructuralAnalyzer",
]