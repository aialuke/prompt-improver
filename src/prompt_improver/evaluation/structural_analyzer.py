"""Structural Analyzer

Analyzes the structural properties of prompts and their improvements.
Provides metrics on prompt structure, format, and organization.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StructuralConfig:
    """Configuration for structural analysis"""

    min_section_length: int = 10
    max_section_length: int = 500
    structure_patterns: list[str] = None

    def __post_init__(self):
        if self.structure_patterns is None:
            self.structure_patterns = [
                r"^\d+\.",  # Numbered lists
                r"^[-*]\s",  # Bullet points
                r"^#{1,6}\s",  # Headers
                r"```",  # Code blocks
            ]


class StructuralAnalyzer:
    """Analyzer for prompt structural properties"""

    def __init__(self, config: StructuralConfig | None = None):
        self.config = config or StructuralConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def analyze_structure(self, text: str) -> dict[str, Any]:
        """Analyze structural properties of text"""
        if not text:
            return self._empty_analysis()

        lines = text.split("\n")

        analysis = {
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "avg_line_length": sum(len(line) for line in lines) / len(lines)
            if lines
            else 0,
            "has_headers": self._detect_headers(text),
            "has_lists": self._detect_lists(text),
            "has_code_blocks": self._detect_code_blocks(text),
            "structure_score": 0.0,
            "organization_score": 0.0,
        }

        # Calculate structure score
        analysis["structure_score"] = self._calculate_structure_score(analysis)
        analysis["organization_score"] = self._calculate_organization_score(
            text, analysis
        )

        return analysis

    def _empty_analysis(self) -> dict[str, Any]:
        """Return empty analysis for empty text"""
        return {
            "total_lines": 0,
            "non_empty_lines": 0,
            "avg_line_length": 0,
            "has_headers": False,
            "has_lists": False,
            "has_code_blocks": False,
            "structure_score": 0.0,
            "organization_score": 0.0,
        }

    def _detect_headers(self, text: str) -> bool:
        """Detect if text has headers"""
        return bool(re.search(r"^#{1,6}\s", text, re.MULTILINE))

    def _detect_lists(self, text: str) -> bool:
        """Detect if text has lists"""
        patterns = [r"^\d+\.", r"^[-*]\s"]
        return any(re.search(pattern, text, re.MULTILINE) for pattern in patterns)

    def _detect_code_blocks(self, text: str) -> bool:
        """Detect if text has code blocks"""
        return "```" in text or text.count("`") >= 2

    def _calculate_structure_score(self, analysis: dict[str, Any]) -> float:
        """Calculate overall structure score"""
        score = 0.0

        # Points for having structural elements
        if analysis["has_headers"]:
            score += 0.3
        if analysis["has_lists"]:
            score += 0.3
        if analysis["has_code_blocks"]:
            score += 0.2

        # Points for reasonable line length
        if 20 <= analysis["avg_line_length"] <= 100:
            score += 0.2

        return min(1.0, score)

    def _calculate_organization_score(
        self, text: str, analysis: dict[str, Any]
    ) -> float:
        """Calculate organization score"""
        score = 0.5  # Base score

        # Points for consistent formatting
        if analysis["non_empty_lines"] / max(analysis["total_lines"], 1) > 0.7:
            score += 0.2

        # Points for logical flow (simple heuristic)
        if len(text) > 100 and analysis["has_headers"]:
            score += 0.3

        return min(1.0, score)
