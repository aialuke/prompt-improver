"""Dependency parsing module for syntactic analysis of prompts.

This module provides dependency parsing capabilities using NLTK
for analyzing grammatical structure, sentence complexity,
and syntactic quality of prompts.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.chunk import RegexpParser
from nltk.tree import Tree


@dataclass
class DependencyRelation:
    """Container for a dependency relation."""

    head: str
    head_pos: str
    head_index: int
    dependent: str
    dependent_pos: str
    dependent_index: int
    relation: str
    confidence: float = 1.0


@dataclass
class SyntacticFeatures:
    """Container for syntactic analysis features."""

    clause_count: int = 0
    phrase_count: int = 0
    dependency_depth: int = 0
    coordination_count: int = 0
    subordination_count: int = 0
    passive_voice_count: int = 0
    complexity_score: float = 0.0


class DependencyParser:
    """Dependency parser for syntactic analysis using NLTK.

    This class provides syntactic analysis capabilities including:
    - Basic dependency parsing using chunk grammar
    - Syntactic complexity assessment
    - Sentence structure quality evaluation
    - Grammatical pattern recognition
    """

    def __init__(self):
        """Initialize the dependency parser."""
        self.logger = logging.getLogger(__name__)

        # Define chunk grammar for basic parsing
        self.chunk_grammar = r"""
            NP: {<DT|PP\$>?<JJ>*<NN.*>+}          # Noun phrase
            PP: {<IN><NP>}                         # Prepositional phrase
            VP: {<VB.*><NP|PP|CLAUSE>+$}          # Verb phrase
            CLAUSE: {<NP><VP>}                     # Simple clause
            ADJP: {<RB>?<JJ><PP>?}                # Adjective phrase
            ADVP: {<RB|RBR|RBS>+}                 # Adverb phrase
        """

        # Initialize chunk parser
        self.chunk_parser = RegexpParser(self.chunk_grammar)

        # Define dependency relation patterns
        self.dependency_patterns = {
            # Subject relations
            "nsubj": [  # nominal subject
                (r"<NP><VP>", "NP", "VP", "subject"),
            ],
            "nsubjpass": [  # passive nominal subject
                (r"<NP><VP.*pass>", "NP", "VP", "passive_subject"),
            ],
            # Object relations
            "dobj": [  # direct object
                (r"<VP><NP>", "VP", "NP", "direct_object"),
            ],
            "iobj": [  # indirect object
                (r"<VP><NP><NP>", "VP", "NP", "indirect_object"),
            ],
            # Modifier relations
            "amod": [  # adjectival modifier
                (r"<JJ><NN.*>", "NN", "JJ", "adjective_modifier"),
            ],
            "advmod": [  # adverbial modifier
                (r"<RB.*><VB.*>", "VB", "RB", "adverb_modifier"),
                (r"<RB.*><JJ>", "JJ", "RB", "adverb_modifier"),
            ],
            # Prepositional relations
            "prep": [  # prepositional modifier
                (r"<PP>", "head", "IN", "preposition"),
            ],
            "pobj": [  # object of preposition
                (r"<IN><NP>", "IN", "NP", "prep_object"),
            ],
            # Coordination
            "conj": [  # conjunct
                (r"<CC>", "left", "right", "conjunction"),
            ],
            # Complementizer
            "comp": [  # complement
                (r"<IN|TO><CLAUSE>", "main", "CLAUSE", "complement"),
            ],
        }

        # Passive voice indicators
        self.passive_indicators = {
            "be_verbs": {"is", "are", "was", "were", "being", "been", "be"},
            "past_participle_tags": {"VBN"},
        }

        # Complex structure indicators
        self.complexity_indicators = {
            "subordinating_conjunctions": {
                "because",
                "since",
                "although",
                "though",
                "while",
                "whereas",
                "if",
                "unless",
                "when",
                "whenever",
                "where",
                "wherever",
                "before",
                "after",
                "until",
                "as",
                "that",
                "which",
                "who",
                "whom",
            },
            "coordinating_conjunctions": {
                "and",
                "but",
                "or",
                "nor",
                "for",
                "so",
                "yet",
            },
            "complex_verbs": {"would", "could", "should", "might", "must", "ought"},
        }

    def parse(self, text: str) -> list[dict[str, Any]]:
        """Parse text and extract dependency relationships.

        Args:
            text: The text to parse

        Returns:
            List of dependency relation dictionaries
        """
        dependencies = []

        try:
            # Split into sentences
            sentences = sent_tokenize(text)

            for sent_idx, sentence in enumerate(sentences):
                # Parse individual sentence
                sent_dependencies = self._parse_sentence(sentence, sent_idx)
                dependencies.extend(sent_dependencies)

        except Exception as e:
            self.logger.error(f"Dependency parsing failed: {e}")

        return dependencies

    def _parse_sentence(self, sentence: str, sent_idx: int) -> list[dict[str, Any]]:
        """Parse a single sentence and extract dependencies."""
        dependencies = []

        try:
            # Tokenize and POS tag
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)

            # Create basic dependency structure
            for i, (token, pos) in enumerate(pos_tags):
                # Find head (simplified approach)
                head_idx = self._find_head(pos_tags, i)

                # Determine relation type
                relation = self._determine_relation(pos_tags, i, head_idx)

                if head_idx != -1 and head_idx != i:
                    dependencies.append({
                        "sentence_id": sent_idx,
                        "head_token": pos_tags[head_idx][0],
                        "head_pos": pos_tags[head_idx][1],
                        "head_index": head_idx,
                        "dependent_token": token,
                        "dependent_pos": pos,
                        "dependent_index": i,
                        "relation": relation,
                        "confidence": 0.7,  # Simplified confidence
                        "depth": 1,  # Simplified depth
                    })

            # Extract additional features
            chunk_tree = self.chunk_parser.parse(pos_tags)
            chunk_dependencies = self._extract_chunk_dependencies(chunk_tree, sent_idx)
            dependencies.extend(chunk_dependencies)

        except Exception as e:
            self.logger.warning(f"Sentence parsing failed: {e}")

        return dependencies

    def _find_head(self, pos_tags: list[tuple[str, str]], token_idx: int) -> int:
        """Find the head of a token (simplified heuristic approach)."""
        token, pos = pos_tags[token_idx]

        # Simple heuristics for finding heads
        if pos.startswith("NN"):  # Noun
            # Look for governing verb
            for i, (t, p) in enumerate(pos_tags):
                if p.startswith("VB") and i != token_idx:
                    return i
            return -1

        if pos.startswith("VB"):  # Verb
            # Verbs are often heads or depend on ROOT
            return -1

        if pos.startswith("JJ"):  # Adjective
            # Look for modified noun
            for i in range(token_idx + 1, len(pos_tags)):
                if pos_tags[i][1].startswith("NN"):
                    return i
            # Look backwards
            for i in range(token_idx - 1, -1, -1):
                if pos_tags[i][1].startswith("NN"):
                    return i
            return -1

        if pos.startswith("RB"):  # Adverb
            # Look for modified verb or adjective
            for i, (t, p) in enumerate(pos_tags):
                if (p.startswith("VB") or p.startswith("JJ")) and i != token_idx:
                    return i
            return -1

        if pos in ["DT", "PRP$"]:  # Determiner
            # Look for following noun
            for i in range(token_idx + 1, len(pos_tags)):
                if pos_tags[i][1].startswith("NN"):
                    return i
            return -1

        if pos.startswith("IN"):  # Preposition
            # Look for governing verb or noun
            for i, (t, p) in enumerate(pos_tags):
                if (p.startswith("VB") or p.startswith("NN")) and i < token_idx:
                    return i
            return -1

        return -1

    def _determine_relation(
        self, pos_tags: list[tuple[str, str]], dependent_idx: int, head_idx: int
    ) -> str:
        """Determine the dependency relation type."""
        if head_idx == -1:
            return "ROOT"

        dep_pos = pos_tags[dependent_idx][1]
        head_pos = pos_tags[head_idx][1]

        # Simple relation mapping based on POS tags
        if dep_pos.startswith("NN") and head_pos.startswith("VB"):
            # Check position to determine subject vs object
            if dependent_idx < head_idx:
                return "nsubj"  # subject
            return "dobj"  # direct object

        if dep_pos.startswith("JJ") and head_pos.startswith("NN"):
            return "amod"  # adjectival modifier

        if (dep_pos.startswith("RB") and head_pos.startswith("VB")) or (
            dep_pos.startswith("RB") and head_pos.startswith("JJ")
        ):
            return "advmod"  # adverbial modifier

        if dep_pos in ["DT", "PRP$"] and head_pos.startswith("NN"):
            return "det"  # determiner

        if dep_pos.startswith("IN"):
            return "prep"  # preposition

        if dep_pos == "CC":
            return "cc"  # coordinating conjunction

        if dep_pos.startswith("WP") or dep_pos.startswith("WDT"):
            return "wh"  # wh-word

        return "dep"  # generic dependency

    def _extract_chunk_dependencies(
        self, tree: Tree, sent_idx: int
    ) -> list[dict[str, Any]]:
        """Extract dependencies from chunk parse tree."""
        dependencies = []

        def traverse_tree(subtree, parent_label=None, depth=0):
            if isinstance(subtree, Tree):
                label = subtree.label()

                # Add relation to parent if exists
                if parent_label:
                    dependencies.append({
                        "sentence_id": sent_idx,
                        "head_token": parent_label,
                        "head_pos": parent_label,
                        "head_index": -1,
                        "dependent_token": label,
                        "dependent_pos": label,
                        "dependent_index": -1,
                        "relation": f"chunk_{parent_label.lower()}_{label.lower()}",
                        "confidence": 0.6,
                        "depth": depth,
                    })

                # Recursively process children
                for child in subtree:
                    traverse_tree(child, label, depth + 1)

        traverse_tree(tree)
        return dependencies

    def analyze_syntactic_complexity(self, text: str) -> SyntacticFeatures:
        """Analyze syntactic complexity of text.

        Args:
            text: The text to analyze

        Returns:
            SyntacticFeatures object with complexity metrics
        """
        features = SyntacticFeatures()

        try:
            sentences = sent_tokenize(text)

            for sentence in sentences:
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)

                # Count clauses (simplified)
                verb_count = sum(1 for _, pos in pos_tags if pos.startswith("VB"))
                features.clause_count += max(1, verb_count)

                # Count phrases using chunking
                chunks = self.chunk_parser.parse(pos_tags)
                phrase_count = sum(1 for subtree in chunks if isinstance(subtree, Tree))
                features.phrase_count += phrase_count

                # Check for coordination
                coord_count = sum(
                    1
                    for token, _ in pos_tags
                    if token.lower()
                    in self.complexity_indicators["coordinating_conjunctions"]
                )
                features.coordination_count += coord_count

                # Check for subordination
                subord_count = sum(
                    1
                    for token, _ in pos_tags
                    if token.lower()
                    in self.complexity_indicators["subordinating_conjunctions"]
                )
                features.subordination_count += subord_count

                # Check for passive voice
                passive_count = self._count_passive_voice(pos_tags)
                features.passive_voice_count += passive_count

            # Calculate complexity score
            features.complexity_score = self._calculate_complexity_score(
                features, len(sentences)
            )

        except Exception as e:
            self.logger.error(f"Syntactic complexity analysis failed: {e}")

        return features

    def _count_passive_voice(self, pos_tags: list[tuple[str, str]]) -> int:
        """Count passive voice constructions in POS tags."""
        passive_count = 0

        for i in range(len(pos_tags) - 1):
            token, pos = pos_tags[i]
            next_token, next_pos = pos_tags[i + 1]

            # Look for "be + past participle" pattern
            if (
                token.lower() in self.passive_indicators["be_verbs"]
                and next_pos in self.passive_indicators["past_participle_tags"]
            ):
                passive_count += 1

        return passive_count

    def _calculate_complexity_score(
        self, features: SyntacticFeatures, sentence_count: int
    ) -> float:
        """Calculate overall syntactic complexity score."""
        if sentence_count == 0:
            return 0.0

        # Normalize by sentence count
        avg_clauses = features.clause_count / sentence_count
        avg_phrases = features.phrase_count / sentence_count
        avg_coordination = features.coordination_count / sentence_count
        avg_subordination = features.subordination_count / sentence_count
        avg_passive = features.passive_voice_count / sentence_count

        # Weighted complexity score
        complexity = (
            avg_clauses * 0.2
            + avg_phrases * 0.15
            + avg_coordination * 0.15
            + avg_subordination * 0.25
            + avg_passive * 0.25
        )

        # Normalize to 0-1 scale
        return min(complexity / 3.0, 1.0)

    def extract_grammatical_patterns(self, text: str) -> dict[str, Any]:
        """Extract grammatical patterns from text.

        Args:
            text: The text to analyze

        Returns:
            Dictionary with grammatical pattern statistics
        """
        patterns = {
            "sentence_types": defaultdict(int),
            "phrase_types": defaultdict(int),
            "pos_distribution": defaultdict(int),
            "dependency_types": defaultdict(int),
        }

        try:
            sentences = sent_tokenize(text)

            for sentence in sentences:
                # Classify sentence type
                sentence_type = self._classify_sentence_type(sentence)
                patterns["sentence_types"][sentence_type] += 1

                # Analyze POS distribution
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)

                for token, pos in pos_tags:
                    patterns["pos_distribution"][pos] += 1

                # Extract phrase types
                chunks = self.chunk_parser.parse(pos_tags)
                for subtree in chunks:
                    if isinstance(subtree, Tree):
                        patterns["phrase_types"][subtree.label()] += 1

                # Extract dependency types
                dependencies = self._parse_sentence(sentence, 0)
                for dep in dependencies:
                    patterns["dependency_types"][dep["relation"]] += 1

        except Exception as e:
            self.logger.error(f"Pattern extraction failed: {e}")

        return dict(patterns)

    def _classify_sentence_type(self, sentence: str) -> str:
        """Classify sentence type based on structure and punctuation."""
        sentence = sentence.strip()

        if sentence.endswith("?"):
            return "interrogative"
        if sentence.endswith("!"):
            return "exclamatory"
        if any(
            word in sentence.lower()
            for word in ["please", "let", "make", "do", "don't"]
        ):
            return "imperative"
        return "declarative"

    def assess_sentence_structure_quality(
        self, dependencies: list[dict[str, Any]]
    ) -> float:
        """Assess the quality of sentence structure based on dependencies.

        Args:
            dependencies: List of dependency relations

        Returns:
            Quality score between 0 and 1
        """
        if not dependencies:
            return 0.0

        quality_score = 0.0
        total_weights = 0.0

        # Check for essential relations
        relations = [dep["relation"] for dep in dependencies]

        # Subject-verb-object structure
        if "nsubj" in relations:
            quality_score += 0.3
            total_weights += 0.3

        if "dobj" in relations or "prep" in relations:
            quality_score += 0.2
            total_weights += 0.2

        # Modifier richness
        modifier_relations = ["amod", "advmod", "det"]
        modifier_count = sum(1 for rel in relations if rel in modifier_relations)
        if modifier_count > 0:
            modifier_score = min(modifier_count / len(dependencies), 0.3)
            quality_score += modifier_score
            total_weights += 0.3

        # Balanced complexity
        complex_relations = ["prep", "conj", "comp"]
        complex_count = sum(1 for rel in relations if rel in complex_relations)
        if (
            0 < complex_count <= len(dependencies) * 0.3
        ):  # Not too many complex relations
            quality_score += 0.2
            total_weights += 0.2

        return quality_score / max(total_weights, 1.0) if total_weights > 0 else 0.0
