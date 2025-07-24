"""Named Entity Recognition (NER) extractor for prompt analysis.

This module provides NER capabilities using NLTK's built-in NER tools
and optional transformers models for more accurate entity recognition.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

@dataclass
class EntityResult:
    """Result container for a named entity."""

    text: str
    label: str
    confidence: float
    start: int
    end: int
    context: str | None = None

class NERExtractor:
    """Named Entity Recognition extractor using NLTK and transformers.

    This class provides comprehensive NER capabilities including:
    - Standard named entity recognition (PERSON, ORGANIZATION, LOCATION, etc.)
    - Technical term identification
    - Domain-specific entity recognition
    - Confidence scoring
    """

    def __init__(self, use_transformers: bool = True, model_name: str | None = None):
        """Initialize the NER extractor.

        Args:
            use_transformers: Whether to use transformers models
            model_name: Name of the transformers model to use
        """
        self.use_transformers = use_transformers
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Standard NER labels mapping
        self.ner_labels = {
            "PERSON": "Person",
            "ORGANIZATION": "Organization",
            "GPE": "Geopolitical Entity",
            "LOCATION": "Location",
            "FACILITY": "Facility",
            "MONEY": "Money",
            "PERCENT": "Percentage",
            "DATE": "Date",
            "TIME": "Time",
            "CARDINAL": "Cardinal Number",
            "ORDINAL": "Ordinal Number",
        }

        # Technical domain patterns
        self.technical_patterns = {
            "API_ENDPOINT": re.compile(
                r"\b(?:GET|POST|PUT|DELETE|PATCH)\s+/[\w/\-\?&=]+", re.IGNORECASE
            ),
            "HTTP_STATUS": re.compile(r"\b(?:200|201|400|401|403|404|500)\b"),
            "JSON_OBJECT": re.compile(r"\{[^{}]*\}"),
            "CODE_BLOCK": re.compile(r"```[\s\S]*?```"),
            "FUNCTION_CALL": re.compile(r"\b\w+\([^)]*\)"),
            "FILE_PATH": re.compile(
                r"(?:[./][\w/\-\.]+|[\w\-\.]+\.(?:py|js|html|css|json|xml|txt))"
            ),
            "URL": re.compile(r"https?://[\w\-\.]+(?:/[\w\-\.]*)*"),
            "EMAIL": re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
            "VERSION": re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:-[\w\.]+)?\b"),
            "VARIABLE": re.compile(r"\b[a-zA-Z_]\w*\b(?=\s*[=:])"),
        }

        # Programming language keywords
        self.programming_keywords = {
            "python",
            "javascript",
            "java",
            "cpp",
            "csharp",
            "go",
            "rust",
            "sql",
            "html",
            "css",
            "json",
            "xml",
            "yaml",
            "markdown",
            "bash",
            "shell",
            "powershell",
            "dockerfile",
            "kubernetes",
        }

        # AI/ML specific terms
        self.ai_ml_terms = {
            "neural",
            "network",
            "deep",
            "learning",
            "machine",
            "artificial",
            "intelligence",
            "model",
            "training",
            "inference",
            "prediction",
            "classification",
            "regression",
            "clustering",
            "supervised",
            "unsupervised",
            "reinforcement",
            "transformer",
            "attention",
            "embedding",
            "tokenization",
            "fine-tuning",
            "pretraining",
            "llm",
            "gpt",
            "bert",
            "roberta",
            "t5",
            "llama",
            "claude",
        }

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from text.

        Args:
            text: The text to analyze

        Returns:
            List of entity dictionaries with text, label, confidence, etc.
        """
        entities = []

        try:
            # Extract standard named entities using NLTK
            nltk_entities = self._extract_nltk_entities(text)
            entities.extend(nltk_entities)

            # Extract technical entities
            technical_entities = self._extract_technical_entities(text)
            entities.extend(technical_entities)

            # Extract domain-specific entities
            domain_entities = self._extract_domain_entities(text)
            entities.extend(domain_entities)

            # Remove duplicates and sort by position
            entities = self._deduplicate_entities(entities)
            entities.sort(key=lambda x: x.get("start", 0))

            self.logger.debug(f"Extracted {len(entities)} entities from text")

        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")

        return entities

    def _extract_nltk_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract entities using NLTK's built-in NER."""
        entities = []

        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)

            # Named entity recognition
            tree = ne_chunk(pos_tags, binary=False)

            current_pos = 0
            for element in tree:
                if isinstance(element, Tree):
                    # This is a named entity
                    entity_text = " ".join([token for token, pos in element.leaves()])
                    entity_label = element.label()

                    # Find position in original text
                    start_pos = text.lower().find(entity_text.lower(), current_pos)
                    end_pos = start_pos + len(entity_text) if start_pos != -1 else -1

                    if start_pos != -1:
                        entities.append({
                            "text": entity_text,
                            "label": self.ner_labels.get(entity_label, entity_label),
                            "confidence": 0.8,  # NLTK doesn't provide confidence scores
                            "start": start_pos,
                            "end": end_pos,
                            "method": "nltk",
                        })
                        current_pos = end_pos
                else:
                    # Regular token
                    token, pos = element
                    token_pos = text.find(token, current_pos)
                    if token_pos != -1:
                        current_pos = token_pos + len(token)

        except Exception as e:
            self.logger.warning(f"NLTK entity extraction failed: {e}")

        return entities

    def _extract_technical_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract technical entities using regex patterns."""
        entities = []

        for entity_type, pattern in self.technical_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "label": entity_type,
                    "confidence": 0.9,
                    "start": match.start(),
                    "end": match.end(),
                    "method": "regex",
                })

        return entities

    def _extract_domain_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract domain-specific entities (programming, AI/ML terms)."""
        entities = []
        words = re.findall(r"\b\w+\b", text.lower())

        # Check for programming language mentions
        for word in words:
            if word in self.programming_keywords:
                # Find actual position in text
                pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
                matches = pattern.finditer(text)
                for match in matches:
                    entities.append({
                        "text": match.group(),
                        "label": "PROGRAMMING_LANGUAGE",
                        "confidence": 0.95,
                        "start": match.start(),
                        "end": match.end(),
                        "method": "domain",
                    })

        # Check for AI/ML terms
        for word in words:
            if word in self.ai_ml_terms:
                pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
                matches = pattern.finditer(text)
                for match in matches:
                    entities.append({
                        "text": match.group(),
                        "label": "AI_ML_TERM",
                        "confidence": 0.85,
                        "start": match.start(),
                        "end": match.end(),
                        "method": "domain",
                    })

        return entities

    def _deduplicate_entities(
        self, entities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicate entities and resolve overlaps."""
        if not entities:
            return entities

        # Sort by start position
        entities.sort(key=lambda x: (x.get("start", 0), x.get("end", 0)))

        deduplicated = []
        for entity in entities:
            # Check for overlaps with existing entities
            overlap_found = False

            for existing in deduplicated:
                if self._entities_overlap(entity, existing):
                    # Keep the entity with higher confidence
                    if entity.get("confidence", 0) > existing.get("confidence", 0):
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlap_found = True
                    break

            if not overlap_found:
                deduplicated.append(entity)

        return deduplicated

    def _entities_overlap(
        self, entity1: dict[str, Any], entity2: dict[str, Any]
    ) -> bool:
        """Check if two entities overlap in position."""
        start1, end1 = entity1.get("start", 0), entity1.get("end", 0)
        start2, end2 = entity2.get("start", 0), entity2.get("end", 0)

        return not (end1 <= start2 or end2 <= start1)

    def extract_technical_terms(
        self, text: str, custom_keywords: set[str] | None = None
    ) -> list[str]:
        """Extract technical terms from text.

        Args:
            text: The text to analyze
            custom_keywords: Additional keywords to look for

        Returns:
            List of technical terms found
        """
        terms = set()

        # Use default technical keywords
        all_keywords = self.programming_keywords | self.ai_ml_terms

        # Add custom keywords if provided
        if custom_keywords:
            all_keywords |= custom_keywords

        # Extract terms
        words = set(re.findall(r"\b\w+\b", text.lower()))
        terms = words.intersection(all_keywords)

        return list(terms)

    def get_entity_context(
        self, text: str, entity_start: int, entity_end: int, context_window: int = 50
    ) -> str:
        """Get context around an entity.

        Args:
            text: The full text
            entity_start: Start position of entity
            entity_end: End position of entity
            context_window: Number of characters to include on each side

        Returns:
            Context string around the entity
        """
        start = max(0, entity_start - context_window)
        end = min(len(text), entity_end + context_window)

        context = text[start:end]

        # Mark the entity in context
        entity_text = text[entity_start:entity_end]
        relative_start = entity_start - start
        relative_end = entity_end - start

        marked_context = (
            context[:relative_start] + f"**{entity_text}**" + context[relative_end:]
        )

        return marked_context

    def analyze_entity_distribution(
        self, entities: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze the distribution of entities in the text.

        Args:
            entities: List of extracted entities

        Returns:
            Dictionary with distribution statistics
        """
        if not entities:
            return {
                "total_entities": 0,
                "unique_types": 0,
                "type_distribution": {},
                "confidence_stats": {},
            }

        # Count by type
        type_counts = {}
        confidences = []

        for entity in entities:
            label = entity.get("label", "UNKNOWN")
            type_counts[label] = type_counts.get(label, 0) + 1

            confidence = entity.get("confidence", 0)
            confidences.append(confidence)

        # Calculate confidence statistics
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        min_confidence = min(confidences) if confidences else 0
        max_confidence = max(confidences) if confidences else 0

        return {
            "total_entities": len(entities),
            "unique_types": len(type_counts),
            "type_distribution": type_counts,
            "confidence_stats": {
                "average": avg_confidence,
                "minimum": min_confidence,
                "maximum": max_confidence,
            },
        }
