"""Data models for rule engine components.

This module contains shared data structures used across the rule engine
to avoid circular imports between modules.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PromptCharacteristics:
    """Comprehensive prompt characteristics for rule selection.

    Contains both basic characteristics and Phase 4 ML-enhanced features.
    """

    prompt_type: str
    complexity_level: float
    domain: str
    length_category: str
    reasoning_required: bool
    specificity_level: float
    context_richness: float
    task_type: str
    language_style: str
    custom_attributes: dict[str, Any]
    semantic_complexity: float | None = None
    domain_confidence: float | None = None
    reasoning_depth: int | None = None
    context_dependencies: list[str] | None = None
    linguistic_features: dict[str, Any] | None = None
    pattern_signatures: list[str] | None = None

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt_type": self.prompt_type,
            "complexity_level": self.complexity_level,
            "domain": self.domain,
            "length_category": self.length_category,
            "reasoning_required": self.reasoning_required,
            "specificity_level": self.specificity_level,
            "context_richness": self.context_richness,
            "task_type": self.task_type,
            "language_style": self.language_style,
            "custom_attributes": self.custom_attributes,
            "semantic_complexity": self.semantic_complexity,
            "domain_confidence": self.domain_confidence,
            "reasoning_depth": self.reasoning_depth,
            "context_dependencies": self.context_dependencies,
            "linguistic_features": self.linguistic_features,
            "pattern_signatures": self.pattern_signatures,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptCharacteristics":
        """Create from dictionary."""
        return cls(**data)
