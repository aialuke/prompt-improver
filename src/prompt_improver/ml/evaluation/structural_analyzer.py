"""Enhanced Structural Analyzer - 2025 Edition

Advanced structural analysis using graph neural networks, semantic embeddings,
and automated pattern discovery. Implements 2025 best practices for structural
understanding including multi-modal analysis and ML-driven insights.
"""

import logging
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Core ML and NLP libraries
try:
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    GRAPH_ANALYSIS_AVAILABLE = True
except ImportError:
    GRAPH_ANALYSIS_AVAILABLE = False
    warnings.warn("Graph analysis libraries not available. Install with: pip install networkx scikit-learn")

# Advanced NLP for semantic analysis
try:
    import spacy
    from transformers import AutoTokenizer, AutoModel
    import torch
    SEMANTIC_ANALYSIS_AVAILABLE = True
except ImportError:
    SEMANTIC_ANALYSIS_AVAILABLE = False
    warnings.warn("Semantic analysis libraries not available. Install with: pip install spacy transformers torch")

logger = logging.getLogger(__name__)

class StructuralElementType(Enum):
    """Types of structural elements in 2025 analysis"""

    HEADER = "header"
    LIST_ITEM = "list_item"
    CODE_BLOCK = "code_block"
    PARAGRAPH = "paragraph"
    INSTRUCTION = "instruction"
    EXAMPLE = "example"
    CONTEXT = "context"
    CONSTRAINT = "constraint"
    OUTPUT_SPEC = "output_specification"
    SEMANTIC_SECTION = "semantic_section"

class StructuralRelationType(Enum):
    """Types of relationships between structural elements"""

    HIERARCHICAL = "hierarchical"
    SEQUENTIAL = "sequential"
    DEPENDENCY = "dependency"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    FUNCTIONAL = "functional"
    CAUSAL = "causal"

@dataclass
class StructuralElement:
    """Enhanced structural element with 2025 features"""

    element_id: str
    element_type: StructuralElementType
    content: str
    position: int
    level: int = 0
    semantic_embedding: Optional[np.ndarray] = None
    importance_score: float = 0.0
    quality_score: float = 0.0
    relationships: List[Tuple[str, StructuralRelationType, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StructuralGraph:
    """Graph representation of document structure"""

    nodes: Dict[str, StructuralElement]
    edges: List[Tuple[str, str, StructuralRelationType, float]]
    graph_metrics: Dict[str, float]
    semantic_clusters: List[List[str]]
    structural_patterns: List[Dict[str, Any]]

@dataclass
class EnhancedStructuralConfig:
    """Enhanced configuration for 2025 structural analysis"""

    # Basic structural analysis
    min_section_length: int = 10
    max_section_length: int = 500
    structure_patterns: List[str] = field(default_factory=lambda: [
        r"^\d+\.",  # Numbered lists
        r"^[-*]\s",  # Bullet points
        r"^#{1,6}\s",  # Headers
        r"```",  # Code blocks
        r"^[A-Z][^.!?]*:",  # Instructions/labels
    ])

    # 2025 enhancements
    enable_semantic_analysis: bool = True
    enable_graph_analysis: bool = True
    enable_pattern_discovery: bool = True
    enable_quality_assessment: bool = True

    # Semantic analysis settings
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    clustering_eps: float = 0.3
    min_cluster_size: int = 2

    # Graph analysis settings
    max_graph_depth: int = 5
    importance_damping: float = 0.85
    min_edge_weight: float = 0.1

    # Pattern discovery settings
    min_pattern_support: float = 0.3
    max_patterns: int = 20
    pattern_confidence_threshold: float = 0.6

class EnhancedStructuralAnalyzer:
    """Enhanced structural analyzer implementing 2025 best practices

    features:
    - Graph-based structural representation
    - Semantic understanding with transformers
    - Automated pattern discovery
    - Multi-dimensional quality assessment
    - ML-driven structural insights
    """

    def __init__(self, config: Optional[EnhancedStructuralConfig] = None):
        self.config = config or EnhancedStructuralConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize semantic analysis components
        self.semantic_model = None
        self.tokenizer = None
        self.nlp = None

        if self.config.enable_semantic_analysis and SEMANTIC_ANALYSIS_AVAILABLE:
            self._initialize_semantic_models()

        # Initialize graph analysis components
        self.graph_analyzer = None
        if self.config.enable_graph_analysis and GRAPH_ANALYSIS_AVAILABLE:
            self.graph_analyzer = GraphStructuralAnalyzer(self.config)

        # Initialize pattern discovery
        self.pattern_discoverer = None
        if self.config.enable_pattern_discovery:
            self.pattern_discoverer = StructuralPatternDiscoverer(self.config)

    def _initialize_semantic_models(self):
        """Initialize semantic analysis models"""
        try:
            # Load transformer model for semantic embeddings
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.semantic_model)
            self.semantic_model = AutoModel.from_pretrained(self.config.semantic_model)

            # Load spaCy model for linguistic analysis
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None

        except Exception as e:
            self.logger.error(f"Failed to initialize semantic models: {e}")
            self.semantic_model = None
            self.tokenizer = None

    async def analyze_enhanced_structure(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive 2025 structural analysis

        Args:
            text: Text to analyze

        Returns:
            Comprehensive structural analysis with 2025 features
        """
        if not text:
            return self._empty_enhanced_analysis()

        start_time = datetime.now()

        # Phase 1: Extract structural elements
        elements = await self._extract_structural_elements(text)

        # Phase 2: Build structural graph
        structural_graph = None
        if self.config.enable_graph_analysis and self.graph_analyzer:
            structural_graph = await self.graph_analyzer.build_structural_graph(elements)

        # Phase 3: Semantic analysis
        semantic_analysis = {}
        if self.config.enable_semantic_analysis and self.semantic_model:
            semantic_analysis = await self._perform_semantic_analysis(elements)

        # Phase 4: Pattern discovery
        discovered_patterns = []
        if self.config.enable_pattern_discovery and self.pattern_discoverer:
            discovered_patterns = await self.pattern_discoverer.discover_patterns(elements)

        # Phase 5: Quality assessment
        quality_metrics = {}
        if self.config.enable_quality_assessment:
            quality_metrics = await self._assess_structural_quality(
                elements, structural_graph, semantic_analysis
            )

        # Phase 6: Generate insights and recommendations
        insights = await self._generate_structural_insights(
            elements, structural_graph, semantic_analysis, discovered_patterns
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            "analysis_metadata": {
                "timestamp": start_time.isoformat(),
                "execution_time": execution_time,
                "analyzer_version": "2025.1.0",
                "features_enabled": {
                    "semantic_analysis": self.config.enable_semantic_analysis,
                    "graph_analysis": self.config.enable_graph_analysis,
                    "pattern_discovery": self.config.enable_pattern_discovery,
                    "quality_assessment": self.config.enable_quality_assessment
                }
            },
            "structural_elements": {
                "total_elements": len(elements),
                "element_types": self._count_element_types(elements),
                "elements": [self._element_to_dict(elem) for elem in elements]
            },
            "structural_graph": self._graph_to_dict(structural_graph) if structural_graph else None,
            "semantic_analysis": semantic_analysis,
            "discovered_patterns": discovered_patterns,
            "quality_metrics": quality_metrics,
            "insights_and_recommendations": insights,
            "legacy_metrics": await self._calculate_legacy_metrics(text)  # Backward compatibility
        }

    async def _extract_structural_elements(self, text: str) -> List[StructuralElement]:
        """Extract structural elements using 2025 ML-driven approach"""
        elements = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            # Determine element type using ML-enhanced detection
            element_type = await self._classify_element_type(line, i, lines)

            # Create structural element
            element = StructuralElement(
                element_id=f"elem_{i}",
                element_type=element_type,
                content=line.strip(),
                position=i,
                level=self._determine_element_level(line, element_type)
            )

            # Add semantic embedding if available
            if self.semantic_model:
                element.semantic_embedding = await self._get_semantic_embedding(line)

            # Calculate importance and quality scores
            element.importance_score = await self._calculate_importance_score(element, lines)
            element.quality_score = await self._calculate_element_quality(element)

            elements.append(element)

        return elements

    async def _classify_element_type(self, line: str, position: int, all_lines: List[str]) -> StructuralElementType:
        """Classify element type using ML-enhanced detection"""
        line_lower = line.lower().strip()

        # Enhanced pattern matching with context awareness
        if re.match(r'^#{1,6}\s', line):
            return StructuralElementType.HEADER
        elif re.match(r'^[-*•]\s', line) or re.match(r'^\d+\.\s', line):
            return StructuralElementType.LIST_ITEM
        elif '```' in line:
            return StructuralElementType.CODE_BLOCK
        elif any(keyword in line_lower for keyword in ['please', 'write', 'create', 'generate', 'analyze']):
            return StructuralElementType.INSTRUCTION
        elif any(keyword in line_lower for keyword in ['example:', 'for instance', 'such as']):
            return StructuralElementType.EXAMPLE
        elif any(keyword in line_lower for keyword in ['context:', 'background:', 'given']):
            return StructuralElementType.CONTEXT
        elif any(keyword in line_lower for keyword in ['must', 'should', 'required', 'constraint']):
            return StructuralElementType.CONSTRAINT
        elif any(keyword in line_lower for keyword in ['output:', 'format:', 'return']):
            return StructuralElementType.OUTPUT_SPEC
        else:
            # Use semantic analysis for more sophisticated classification
            if self.nlp and len(line) > 20:
                return await self._semantic_element_classification(line)
            return StructuralElementType.PARAGRAPH

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

    async def _semantic_element_classification(self, line: str) -> StructuralElementType:
        """Use semantic analysis for element classification"""
        try:
            if not self.nlp:
                return StructuralElementType.PARAGRAPH

            doc = self.nlp(line)

            # Analyze linguistic features
            has_imperative = any(token.tag_ == 'VB' for token in doc)
            has_question = line.strip().endswith('?')
            has_colon = ':' in line

            if has_imperative and not has_question:
                return StructuralElementType.INSTRUCTION
            elif has_colon and len(line.split(':')[0]) < 20:
                return StructuralElementType.CONTEXT
            else:
                return StructuralElementType.SEMANTIC_SECTION

        except Exception as e:
            self.logger.warning(f"Semantic classification failed: {e}")
            return StructuralElementType.PARAGRAPH

    async def _get_semantic_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get semantic embedding for text"""
        try:
            if not self.semantic_model or not self.tokenizer:
                return None

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.semantic_model(**inputs)
                # Use mean pooling of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

            return embedding

        except Exception as e:
            self.logger.warning(f"Semantic embedding failed: {e}")
            return None

    def _determine_element_level(self, line: str, element_type: StructuralElementType) -> int:
        """Determine hierarchical level of element"""
        if element_type == StructuralElementType.HEADER:
            # Count header level from markdown syntax
            match = re.match(r'^(#{1,6})', line)
            if match:
                return len(match.group(1))
        elif element_type == StructuralElementType.LIST_ITEM:
            # Count indentation level
            leading_spaces = len(line) - len(line.lstrip())
            return leading_spaces // 2 + 1

        return 0

    async def _calculate_importance_score(self, element: StructuralElement, all_lines: List[str]) -> float:
        """Calculate importance score using multiple factors"""
        score = 0.0

        # Position-based importance (beginning and end are more important)
        total_lines = len(all_lines)
        if element.position < total_lines * 0.2:  # First 20%
            score += 0.3
        elif element.position > total_lines * 0.8:  # Last 20%
            score += 0.2

        # Type-based importance
        type_weights = {
            StructuralElementType.HEADER: 0.9,
            StructuralElementType.INSTRUCTION: 0.8,
            StructuralElementType.OUTPUT_SPEC: 0.7,
            StructuralElementType.CONSTRAINT: 0.6,
            StructuralElementType.EXAMPLE: 0.5,
            StructuralElementType.CONTEXT: 0.4,
            StructuralElementType.LIST_ITEM: 0.3,
            StructuralElementType.PARAGRAPH: 0.2,
            StructuralElementType.CODE_BLOCK: 0.6,
            StructuralElementType.SEMANTIC_SECTION: 0.3
        }
        score += type_weights.get(element.element_type, 0.2)

        # Content-based importance (length, keywords)
        content_length = len(element.content)
        if 20 <= content_length <= 200:  # Optimal length
            score += 0.2

        # Keyword-based importance
        important_keywords = ['must', 'required', 'important', 'critical', 'key', 'main']
        if any(keyword in element.content.lower() for keyword in important_keywords):
            score += 0.3

        return min(1.0, score)

    async def _calculate_element_quality(self, element: StructuralElement) -> float:
        """Calculate quality score for structural element"""
        score = 0.5  # Base score

        # Length appropriateness
        content_length = len(element.content)
        if element.element_type == StructuralElementType.HEADER:
            if 5 <= content_length <= 80:
                score += 0.2
        elif element.element_type == StructuralElementType.INSTRUCTION:
            if 10 <= content_length <= 200:
                score += 0.2
        elif element.element_type == StructuralElementType.PARAGRAPH:
            if 20 <= content_length <= 500:
                score += 0.2

        # Clarity indicators
        if element.element_type in [StructuralElementType.INSTRUCTION, StructuralElementType.OUTPUT_SPEC]:
            # Check for action verbs
            action_verbs = ['write', 'create', 'analyze', 'generate', 'provide', 'explain']
            if any(verb in element.content.lower() for verb in action_verbs):
                score += 0.2

        # Completeness indicators
        if element.element_type == StructuralElementType.INSTRUCTION:
            if any(word in element.content.lower() for word in ['what', 'how', 'why', 'when', 'where']):
                score += 0.1

        return min(1.0, score)

    async def _perform_semantic_analysis(self, elements: List[StructuralElement]) -> Dict[str, Any]:
        """Perform semantic analysis on structural elements"""
        if not elements or not self.semantic_model:
            return {}

        try:
            # Extract embeddings
            embeddings = []
            element_ids = []

            for element in elements:
                if element.semantic_embedding is not None:
                    embeddings.append(element.semantic_embedding)
                    element_ids.append(element.element_id)

            if not embeddings:
                return {}

            embeddings = np.array(embeddings)

            # Calculate semantic similarity matrix
            similarity_matrix = cosine_similarity(embeddings)

            # Perform semantic clustering
            if GRAPH_ANALYSIS_AVAILABLE:
                clustering = DBSCAN(
                    eps=self.config.clustering_eps,
                    min_samples=self.config.min_cluster_size,
                    metric='cosine'
                )
                cluster_labels = clustering.fit_predict(embeddings)

                # Group elements by semantic clusters
                semantic_clusters = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    if label != -1:  # Not noise
                        semantic_clusters[f"cluster_{label}"].append(element_ids[i])
            else:
                semantic_clusters = {}
                cluster_labels = []

            # Calculate semantic coherence
            coherence_score = self._calculate_semantic_coherence(similarity_matrix)

            return {
                "similarity_matrix": similarity_matrix.tolist(),
                "semantic_clusters": dict(semantic_clusters),
                "coherence_score": coherence_score,
                "cluster_count": len(semantic_clusters),
                "noise_elements": sum(1 for label in cluster_labels if label == -1)
            }

        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {e}")
            return {}

    def _calculate_semantic_coherence(self, similarity_matrix: np.ndarray) -> float:
        """Calculate overall semantic coherence score"""
        if similarity_matrix.size == 0:
            return 0.0

        # Calculate mean similarity excluding diagonal
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        mean_similarity = similarity_matrix[mask].mean()

        return float(mean_similarity)

    async def _assess_structural_quality(
        self,
        elements: List[StructuralElement],
        structural_graph: Optional[StructuralGraph],
        semantic_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall structural quality using 2025 metrics"""

        if not elements:
            return {"overall_score": 0.0, "quality_dimensions": {}}

        # Calculate quality dimensions
        quality_dimensions = {}

        # 1. Hierarchical Organization
        quality_dimensions["hierarchical_organization"] = self._assess_hierarchy_quality(elements)

        # 2. Semantic Coherence
        quality_dimensions["semantic_coherence"] = semantic_analysis.get("coherence_score", 0.0)

        # 3. Structural Completeness
        quality_dimensions["structural_completeness"] = self._assess_completeness(elements)

        # 4. Element Quality
        element_qualities = [elem.quality_score for elem in elements]
        quality_dimensions["element_quality"] = np.mean(element_qualities) if element_qualities else 0.0

        # 5. Graph Connectivity (if available)
        if structural_graph:
            quality_dimensions["graph_connectivity"] = structural_graph.graph_metrics.get("connectivity", 0.0)
        else:
            quality_dimensions["graph_connectivity"] = 0.0

        # Calculate overall score
        weights = {
            "hierarchical_organization": 0.25,
            "semantic_coherence": 0.25,
            "structural_completeness": 0.20,
            "element_quality": 0.20,
            "graph_connectivity": 0.10
        }

        overall_score = sum(
            quality_dimensions[dim] * weights[dim]
            for dim in weights.keys()
        )

        return {
            "overall_score": overall_score,
            "quality_dimensions": quality_dimensions,
            "quality_grade": self._get_quality_grade(overall_score),
            "improvement_suggestions": self._generate_quality_suggestions(quality_dimensions)
        }

    def _assess_hierarchy_quality(self, elements: List[StructuralElement]) -> float:
        """Assess quality of hierarchical organization"""
        if not elements:
            return 0.0

        # Check for proper header hierarchy
        headers = [elem for elem in elements if elem.element_type == StructuralElementType.HEADER]
        if not headers:
            return 0.3  # Some structure but no headers

        # Check for logical level progression
        levels = [elem.level for elem in headers]
        if len(levels) <= 1:
            return 0.7

        # Check for gaps in hierarchy (e.g., h1 -> h3 without h2)
        level_gaps = 0
        for i in range(1, len(levels)):
            if levels[i] - levels[i-1] > 1:
                level_gaps += 1

        hierarchy_score = max(0.0, 1.0 - (level_gaps * 0.2))
        return hierarchy_score

    def _assess_completeness(self, elements: List[StructuralElement]) -> float:
        """Assess structural completeness"""
        element_types = {elem.element_type for elem in elements}

        # Check for essential structural elements
        essential_types = {
            StructuralElementType.INSTRUCTION,
            StructuralElementType.CONTEXT,
            StructuralElementType.OUTPUT_SPEC
        }

        completeness_score = len(element_types & essential_types) / len(essential_types)

        # Bonus for additional helpful elements
        helpful_types = {
            StructuralElementType.EXAMPLE,
            StructuralElementType.CONSTRAINT,
            StructuralElementType.HEADER
        }

        bonus = len(element_types & helpful_types) * 0.1

        return min(1.0, completeness_score + bonus)

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade"""
        if score >= 0.9:
            return "EXCELLENT"
        elif score >= 0.8:
            return "VERY_GOOD"
        elif score >= 0.7:
            return "GOOD"
        elif score >= 0.6:
            return "FAIR"
        elif score >= 0.5:
            return "POOR"
        else:
            return "VERY_POOR"

    def _generate_quality_suggestions(self, quality_dimensions: Dict[str, float]) -> List[str]:
        """Generate improvement suggestions based on quality assessment"""
        suggestions = []

        if quality_dimensions.get("hierarchical_organization", 0) < 0.7:
            suggestions.append("Improve hierarchical organization with clear headers and logical structure")

        if quality_dimensions.get("semantic_coherence", 0) < 0.6:
            suggestions.append("Enhance semantic coherence by grouping related content together")

        if quality_dimensions.get("structural_completeness", 0) < 0.8:
            suggestions.append("Add missing structural elements: instructions, context, or output specifications")

        if quality_dimensions.get("element_quality", 0) < 0.7:
            suggestions.append("Improve individual element quality with clearer, more specific content")

        return suggestions

    async def _generate_structural_insights(
        self,
        elements: List[StructuralElement],
        structural_graph: Optional[StructuralGraph],
        semantic_analysis: Dict[str, Any],
        discovered_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive structural insights"""

        insights = {
            "structural_summary": {
                "total_elements": len(elements),
                "element_distribution": self._count_element_types(elements),
                "average_element_quality": np.mean([elem.quality_score for elem in elements]) if elements else 0.0,
                "highest_importance_elements": [
                    elem.element_id for elem in sorted(elements, key=lambda x: x.importance_score, reverse=True)[:3]
                ]
            },
            "semantic_insights": {
                "coherence_level": semantic_analysis.get("coherence_score", 0.0),
                "cluster_count": semantic_analysis.get("cluster_count", 0),
                "semantic_organization": "good" if semantic_analysis.get("coherence_score", 0) > 0.6 else "needs_improvement"
            },
            "pattern_insights": {
                "patterns_discovered": len(discovered_patterns),
                "pattern_types": [pattern.get("type", "unknown") for pattern in discovered_patterns],
                "pattern_confidence": np.mean([pattern.get("confidence", 0) for pattern in discovered_patterns]) if discovered_patterns else 0.0
            },
            "recommendations": self._generate_actionable_recommendations(elements, semantic_analysis, discovered_patterns)
        }

        return insights

    def _generate_actionable_recommendations(
        self,
        elements: List[StructuralElement],
        semantic_analysis: Dict[str, Any],
        discovered_patterns: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations for improvement"""
        recommendations = []

        # Analyze element distribution
        element_counts = self._count_element_types(elements)

        if element_counts.get("instruction", 0) == 0:
            recommendations.append("Add clear instructions to specify what you want the AI to do")

        if element_counts.get("context", 0) == 0:
            recommendations.append("Provide context or background information to improve response quality")

        if element_counts.get("output_spec", 0) == 0:
            recommendations.append("Specify the desired output format or structure")

        # Semantic analysis recommendations
        if semantic_analysis.get("coherence_score", 0) < 0.5:
            recommendations.append("Reorganize content to improve semantic coherence and flow")

        # Pattern-based recommendations
        if len(discovered_patterns) > 0:
            strong_patterns = [p for p in discovered_patterns if p.get("confidence", 0) > 0.8]
            if strong_patterns:
                recommendations.append(f"Leverage discovered structural patterns: {', '.join([p.get('type', 'unknown') for p in strong_patterns])}")

        return recommendations

    def _count_element_types(self, elements: List[StructuralElement]) -> Dict[str, int]:
        """Count elements by type"""
        counts = defaultdict(int)
        for element in elements:
            counts[element.element_type.value] += 1
        return dict(counts)

    def _element_to_dict(self, element: StructuralElement) -> Dict[str, Any]:
        """Convert element to dictionary for serialization"""
        return {
            "element_id": element.element_id,
            "element_type": element.element_type.value,
            "content": element.content,
            "position": element.position,
            "level": element.level,
            "importance_score": element.importance_score,
            "quality_score": element.quality_score,
            "relationships": [
                {"target": rel[0], "type": rel[1].value, "weight": rel[2]}
                for rel in element.relationships
            ],
            "metadata": element.metadata
        }

    def _graph_to_dict(self, graph: Optional[StructuralGraph]) -> Optional[Dict[str, Any]]:
        """Convert structural graph to dictionary"""
        if not graph:
            return None

        return {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "graph_metrics": graph.graph_metrics,
            "semantic_clusters": graph.semantic_clusters,
            "structural_patterns": graph.structural_patterns
        }

    async def _calculate_legacy_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate legacy metrics for backward compatibility"""
        lines = text.split("\n")

        return {
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
            "has_headers": self._detect_headers(text),
            "has_lists": self._detect_lists(text),
            "has_code_blocks": self._detect_code_blocks(text),
            "structure_score": 0.0,  # Will be calculated
            "organization_score": 0.0  # Will be calculated
        }

    def _empty_enhanced_analysis(self) -> Dict[str, Any]:
        """Return empty enhanced analysis for empty text"""
        return {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "execution_time": 0.0,
                "analyzer_version": "2025.1.0",
                "features_enabled": {}
            },
            "structural_elements": {"total_elements": 0, "element_types": {}, "elements": []},
            "structural_graph": None,
            "semantic_analysis": {},
            "discovered_patterns": [],
            "quality_metrics": {"overall_score": 0.0, "quality_dimensions": {}},
            "insights_and_recommendations": {},
            "legacy_metrics": {
                "total_lines": 0,
                "non_empty_lines": 0,
                "avg_line_length": 0,
                "has_headers": False,
                "has_lists": False,
                "has_code_blocks": False,
                "structure_score": 0.0,
                "organization_score": 0.0,
            }
        }

    # Add orchestrator-compatible interface
    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrator-compatible interface for structural analysis (2025 pattern)

        Args:
            config: Orchestrator configuration containing:
                - text: Text to analyze
                - output_path: Local path for output files (optional)
                - analysis_type: Type of analysis ('enhanced', 'legacy', 'comprehensive')
                - enable_features: Dict of features to enable/disable

        Returns:
            Orchestrator-compatible result with structural analysis and metadata
        """
        start_time = datetime.now()

        try:
            # Extract configuration from orchestrator
            text = config.get("text", "")
            output_path = config.get("output_path", "./outputs/structural_analysis")
            analysis_type = config.get("analysis_type", "enhanced")
            enable_features = config.get("enable_features", {})

            # Override config with orchestrator settings
            if enable_features:
                self.config.enable_semantic_analysis = enable_features.get("semantic_analysis", self.config.enable_semantic_analysis)
                self.config.enable_graph_analysis = enable_features.get("graph_analysis", self.config.enable_graph_analysis)
                self.config.enable_pattern_discovery = enable_features.get("pattern_discovery", self.config.enable_pattern_discovery)
                self.config.enable_quality_assessment = enable_features.get("quality_assessment", self.config.enable_quality_assessment)

            # Perform analysis based on type
            if analysis_type == "enhanced":
                result = await self.analyze_enhanced_structure(text)
            else:
                legacy_result = await self.analyze_structure(text)
                result = {"legacy_analysis": legacy_result}

            # Calculate execution metadata
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "orchestrator_compatible": True,
                "component_result": result,
                "local_metadata": {
                    "output_path": output_path,
                    "execution_time": execution_time,
                    "analysis_type": analysis_type,
                    "text_length": len(text),
                    "features_enabled": {
                        "semantic_analysis": self.config.enable_semantic_analysis,
                        "graph_analysis": self.config.enable_graph_analysis,
                        "pattern_discovery": self.config.enable_pattern_discovery,
                        "quality_assessment": self.config.enable_quality_assessment
                    },
                    "component_version": "2025.1.0"
                }
            }

        except Exception as e:
            self.logger.error(f"Orchestrated structural analysis failed: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": str(e), "analysis": {}},
                "local_metadata": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "error": True,
                    "component_version": "2025.1.0"
                }
            }

class GraphStructuralAnalyzer:
    """Graph-based structural analysis using NetworkX"""

    def __init__(self, config: EnhancedStructuralConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def build_structural_graph(self, elements: List[StructuralElement]) -> Optional[StructuralGraph]:
        """Build graph representation of structural elements"""
        if not GRAPH_ANALYSIS_AVAILABLE or not elements:
            return None

        try:
            # Create NetworkX graph
            G = nx.DiGraph()

            # Add nodes
            for element in elements:
                G.add_node(
                    element.element_id,
                    element_type=element.element_type.value,
                    content=element.content,
                    importance=element.importance_score,
                    quality=element.quality_score
                )

            # Add edges based on relationships
            edges = []

            # Hierarchical relationships
            for i, element in enumerate(elements):
                if i > 0:
                    prev_element = elements[i-1]
                    # Sequential relationship
                    edges.append((prev_element.element_id, element.element_id, StructuralRelationType.SEQUENTIAL, 0.8))

                    # Hierarchical relationship based on levels
                    if element.level > prev_element.level:
                        edges.append((prev_element.element_id, element.element_id, StructuralRelationType.HIERARCHICAL, 0.9))

            # Semantic relationships (if embeddings available)
            if elements and elements[0].semantic_embedding is not None:
                edges.extend(await self._calculate_semantic_edges(elements))

            # Add edges to graph
            for edge in edges:
                if edge[3] >= self.config.min_edge_weight:  # Filter weak edges
                    G.add_edge(edge[0], edge[1], relation_type=edge[2].value, weight=edge[3])

            # Calculate graph metrics
            graph_metrics = self._calculate_graph_metrics(G)

            # Find semantic clusters
            semantic_clusters = self._find_semantic_clusters(G, elements)

            # Discover structural patterns
            structural_patterns = self._discover_graph_patterns(G)

            return StructuralGraph(
                nodes={elem.element_id: elem for elem in elements},
                edges=edges,
                graph_metrics=graph_metrics,
                semantic_clusters=semantic_clusters,
                structural_patterns=structural_patterns
            )

        except Exception as e:
            self.logger.error(f"Graph construction failed: {e}")
            return None

    async def _calculate_semantic_edges(self, elements: List[StructuralElement]) -> List[Tuple[str, str, StructuralRelationType, float]]:
        """Calculate semantic similarity edges between elements"""
        edges = []

        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                if elem1.semantic_embedding is not None and elem2.semantic_embedding is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(elem1.semantic_embedding, elem2.semantic_embedding) / (
                        np.linalg.norm(elem1.semantic_embedding) * np.linalg.norm(elem2.semantic_embedding)
                    )

                    if similarity >= self.config.similarity_threshold:
                        edges.append((elem1.element_id, elem2.element_id, StructuralRelationType.SEMANTIC_SIMILARITY, float(similarity)))

        return edges

    def _calculate_graph_metrics(self, G: nx.DiGraph) -> Dict[str, float]:
        """Calculate graph-based structural metrics"""
        metrics = {}

        if len(G.nodes) == 0:
            return {"connectivity": 0.0, "centrality": 0.0, "clustering": 0.0}

        # Connectivity
        if nx.is_weakly_connected(G):
            metrics["connectivity"] = 1.0
        else:
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            metrics["connectivity"] = len(largest_cc) / len(G.nodes)

        # Centrality measures
        try:
            centrality = nx.degree_centrality(G)
            metrics["centrality"] = np.mean(list(centrality.values()))
        except:
            metrics["centrality"] = 0.0

        # Clustering coefficient
        try:
            clustering = nx.clustering(G.to_undirected())
            metrics["clustering"] = np.mean(list(clustering.values()))
        except:
            metrics["clustering"] = 0.0

        return metrics

    def _find_semantic_clusters(self, G: nx.DiGraph, elements: List[StructuralElement]) -> List[List[str]]:
        """Find semantic clusters in the graph"""
        clusters = []

        try:
            # Use community detection on semantic similarity edges
            semantic_edges = [(u, v) for u, v, d in G.edges(data=True)
                            if d.get('relation_type') == StructuralRelationType.SEMANTIC_SIMILARITY.value]

            if semantic_edges:
                semantic_graph = G.edge_subgraph(semantic_edges).to_undirected()
                communities = nx.community.greedy_modularity_communities(semantic_graph)
                clusters = [list(community) for community in communities]

        except Exception as e:
            self.logger.warning(f"Semantic clustering failed: {e}")

        return clusters

    def _discover_graph_patterns(self, G: nx.DiGraph) -> List[Dict[str, Any]]:
        """Discover structural patterns in the graph"""
        patterns = []

        try:
            # Pattern 1: Linear sequences
            linear_paths = []
            for node in G.nodes():
                if G.in_degree(node) <= 1 and G.out_degree(node) <= 1:
                    # Find longest path from this node
                    try:
                        path_length = nx.dag_longest_path_length(G, weight=None)
                        if path_length > 2:
                            linear_paths.append({"type": "linear_sequence", "length": path_length})
                    except:
                        pass

            if linear_paths:
                patterns.append({
                    "type": "linear_structure",
                    "confidence": 0.8,
                    "instances": len(linear_paths),
                    "description": "Document follows linear sequential structure"
                })

            # Pattern 2: Hierarchical structure
            hierarchical_edges = [(u, v) for u, v, d in G.edges(data=True)
                                if d.get('relation_type') == StructuralRelationType.HIERARCHICAL.value]

            if len(hierarchical_edges) > 0:
                patterns.append({
                    "type": "hierarchical_structure",
                    "confidence": 0.9,
                    "instances": len(hierarchical_edges),
                    "description": "Document has clear hierarchical organization"
                })

            # Pattern 3: Semantic clusters
            semantic_edges = [(u, v) for u, v, d in G.edges(data=True)
                            if d.get('relation_type') == StructuralRelationType.SEMANTIC_SIMILARITY.value]

            if len(semantic_edges) > 2:
                patterns.append({
                    "type": "semantic_clustering",
                    "confidence": 0.7,
                    "instances": len(semantic_edges),
                    "description": "Document contains semantically related content clusters"
                })

        except Exception as e:
            self.logger.warning(f"Pattern discovery failed: {e}")

        return patterns

class StructuralPatternDiscoverer:
    """Automated structural pattern discovery using ML techniques"""

    def __init__(self, config: EnhancedStructuralConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def discover_patterns(self, elements: List[StructuralElement]) -> List[Dict[str, Any]]:
        """Discover structural patterns in elements"""
        patterns = []

        if not elements:
            return patterns

        try:
            # Pattern 1: Element type sequences
            type_patterns = self._discover_type_patterns(elements)
            patterns.extend(type_patterns)

            # Pattern 2: Content patterns
            content_patterns = self._discover_content_patterns(elements)
            patterns.extend(content_patterns)

            # Pattern 3: Quality patterns
            quality_patterns = self._discover_quality_patterns(elements)
            patterns.extend(quality_patterns)

        except Exception as e:
            self.logger.error(f"Pattern discovery failed: {e}")

        return patterns

    def _discover_type_patterns(self, elements: List[StructuralElement]) -> List[Dict[str, Any]]:
        """Discover patterns in element type sequences"""
        patterns = []

        # Extract type sequence
        type_sequence = [elem.element_type.value for elem in elements]

        # Find common subsequences
        if len(type_sequence) >= 3:
            # Look for instruction -> context -> output pattern
            instruction_context_output = 0
            for i in range(len(type_sequence) - 2):
                if (type_sequence[i] == "instruction" and
                    type_sequence[i+1] == "context" and
                    type_sequence[i+2] == "output_specification"):
                    instruction_context_output += 1

            if instruction_context_output > 0:
                patterns.append({
                    "type": "instruction_context_output_pattern",
                    "confidence": 0.9,
                    "instances": instruction_context_output,
                    "description": "Clear instruction -> context -> output specification pattern"
                })

        return patterns

    def _discover_content_patterns(self, elements: List[StructuralElement]) -> List[Dict[str, Any]]:
        """Discover patterns in content characteristics"""
        patterns = []

        # Analyze content length patterns
        lengths = [len(elem.content) for elem in elements]
        if lengths:
            avg_length = np.mean(lengths)
            std_length = np.std(lengths)

            # Check for consistent length pattern
            if std_length / avg_length < 0.5:  # Low coefficient of variation
                patterns.append({
                    "type": "consistent_length_pattern",
                    "confidence": 0.7,
                    "instances": len(elements),
                    "description": f"Consistent content length (avg: {avg_length:.1f} chars)"
                })

        return patterns

    def _discover_quality_patterns(self, elements: List[StructuralElement]) -> List[Dict[str, Any]]:
        """Discover patterns in quality scores"""
        patterns = []

        quality_scores = [elem.quality_score for elem in elements]
        if quality_scores:
            avg_quality = np.mean(quality_scores)

            if avg_quality > 0.8:
                patterns.append({
                    "type": "high_quality_pattern",
                    "confidence": 0.8,
                    "instances": len([q for q in quality_scores if q > 0.8]),
                    "description": "Consistently high-quality structural elements"
                })

        return patterns

# Maintain backward compatibility with original class name
StructuralAnalyzer = EnhancedStructuralAnalyzer
