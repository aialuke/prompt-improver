"""
Domain-Specific Feature Extraction System

This module provides specialized feature extraction based on detected prompt domains,
enabling more targeted and relevant analysis for different types of prompts.
"""

import logging
import re
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import json

try:
    import spacy
    from spacy.language import Language
    from spacy.matcher import Matcher, PhraseMatcher
    from spacy.tokens import Doc, Span, Token
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    Language = None
    Matcher = None
    PhraseMatcher = None
    Doc = None
    Span = None
    Token = None
    SPACY_AVAILABLE = False

from .domain_detector import DomainDetector, PromptDomain, DomainClassificationResult


@dataclass
class DomainFeatures:
    """Container for domain-specific features."""
    
    # Common features across all domains
    domain: PromptDomain
    confidence: float
    complexity_score: float = 0.0
    specificity_score: float = 0.0
    
    # Multi-domain support
    secondary_domains: List[Tuple['PromptDomain', float]] = field(default_factory=list)
    hybrid_domain: bool = False
    
    # Domain-specific feature sets
    technical_features: Dict[str, Any] = field(default_factory=dict)
    creative_features: Dict[str, Any] = field(default_factory=dict)
    academic_features: Dict[str, Any] = field(default_factory=dict)
    business_features: Dict[str, Any] = field(default_factory=dict)
    conversational_features: Dict[str, Any] = field(default_factory=dict)
    
    # Meta-features
    feature_vector: List[float] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)


class BaseDomainExtractor(ABC):
    """Abstract base class for domain-specific feature extractors."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def extract_features(self, text: str, domain_result: DomainClassificationResult) -> Dict[str, Any]:
        """Extract domain-specific features from text."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of features extracted by this extractor."""
        pass


class TechnicalDomainExtractor(BaseDomainExtractor):
    """Feature extractor for technical domains (software, data science, AI/ML, etc.)."""
    
    def __init__(self):
        super().__init__("technical")
        self._compile_technical_patterns()
    
    def _compile_technical_patterns(self):
        """Compile regex patterns for technical feature detection."""
        self.patterns = {
            # Code patterns
            'code_snippets': [
                r'```[\s\S]*?```',  # Code blocks
                r'`[^`\n]+`',       # Inline code
                r'\b(?:def|function|class|import|from|if|else|for|while)\s+\w+',
            ],
            
            # API patterns
            'api_references': [
                r'\b(?:GET|POST|PUT|DELETE|PATCH)\s+/',
                r'\b(?:api|endpoint|route)\b.*?/',
                r'\{[\w\s,:"\']*\}',  # JSON-like structures
            ],
            
            # Technical concepts
            'algorithms': [
                r'\b(?:algorithm|sorting|searching|optimization|recursion)\b',
                r'\b(?:big o|time complexity|space complexity)\b',
                r'\b(?:binary search|quicksort|mergesort|hash table)\b',
            ],
            
            # Data structures
            'data_structures': [
                r'\b(?:array|list|dictionary|hash|tree|graph|queue|stack)\b',
                r'\b(?:linked list|binary tree|heap|trie)\b',
            ],
            
            # ML/AI specific
            'ml_concepts': [
                r'\b(?:neural network|deep learning|machine learning)\b',
                r'\b(?:training|validation|test)\s+(?:set|data)\b',
                r'\b(?:accuracy|precision|recall|f1[-_]score|auc|roc)\b',
                r'\b(?:supervised|unsupervised|reinforcement)\s+learning\b',
            ],
            
            # System architecture
            'architecture': [
                r'\b(?:microservice|api gateway|load balancer)\b',
                r'\b(?:docker|kubernetes|container|deployment)\b',
                r'\b(?:scalability|high availability|fault tolerance)\b',
            ]
        }
        
        self.compiled_patterns = {}
        for category, patterns in self.patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                for pattern in patterns
            ]
    
    def extract_features(self, text: str, domain_result: DomainClassificationResult) -> Dict[str, Any]:
        """Extract technical domain features."""
        features = {}
        
        # Pattern-based features
        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))
            
            features[f'{category}_count'] = len(matches)
            features[f'{category}_density'] = len(matches) / max(len(text.split()), 1)
            features[f'has_{category}'] = len(matches) > 0
        
        # Technical vocabulary analysis
        features.update(self._analyze_technical_vocabulary(text, domain_result))
        
        # Code quality indicators
        features.update(self._analyze_code_quality_indicators(text))
        
        # Documentation patterns
        features.update(self._analyze_documentation_patterns(text))
        
        return features
    
    def _analyze_technical_vocabulary(self, text: str, domain_result: DomainClassificationResult) -> Dict[str, Any]:
        """Analyze technical vocabulary usage."""
        text_lower = text.lower()
        
        # Count technical terms
        technical_terms = []
        for keywords in domain_result.domain_keywords_found.values():
            technical_terms.extend(keywords)
        
        # Calculate technical density
        word_count = len(text.split())
        technical_density = len(technical_terms) / max(word_count, 1)
        
        # Analyze technical term distribution
        unique_technical_terms = len(set(technical_terms))
        technical_diversity = unique_technical_terms / max(len(technical_terms), 1)
        
        return {
            'technical_term_count': len(technical_terms),
            'unique_technical_terms': unique_technical_terms,
            'technical_density': technical_density,
            'technical_diversity': technical_diversity,
            'avg_technical_term_length': sum(len(term) for term in technical_terms) / max(len(technical_terms), 1)
        }
    
    def _analyze_code_quality_indicators(self, text: str) -> Dict[str, Any]:
        """Analyze indicators of code quality discussion."""
        quality_indicators = [
            r'\b(?:clean code|best practice|code review|refactor)\b',
            r'\b(?:maintainable|readable|scalable|testable)\b',
            r'\b(?:design pattern|solid principle|dry principle)\b',
            r'\b(?:unit test|integration test|test coverage)\b',
            r'\b(?:performance|optimization|efficiency)\b',
            r'\b(?:security|vulnerability|authentication|authorization)\b'
        ]
        
        quality_matches = 0
        for pattern in quality_indicators:
            quality_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        return {
            'code_quality_mentions': quality_matches,
            'has_quality_focus': quality_matches > 0,
            'quality_density': quality_matches / max(len(text.split()), 1)
        }
    
    def _analyze_documentation_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze documentation and explanation patterns."""
        doc_patterns = [
            r'\b(?:explain|describe|how to|step by step)\b',
            r'\b(?:example|sample|demo|tutorial)\b',
            r'\b(?:documentation|readme|guide|manual)\b',
            r'\b(?:usage|installation|setup|configuration)\b'
        ]
        
        doc_matches = 0
        for pattern in doc_patterns:
            doc_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Check for structured documentation
        has_headings = bool(re.search(r'^#+\s+', text, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[-*+]\s+|^\s*\d+\.\s+', text, re.MULTILINE))
        has_code_blocks = bool(re.search(r'```[\s\S]*?```', text))
        
        return {
            'documentation_mentions': doc_matches,
            'has_documentation_focus': doc_matches > 0,
            'has_structured_format': has_headings or has_lists,
            'has_code_examples': has_code_blocks,
            'documentation_completeness': sum([has_headings, has_lists, has_code_blocks]) / 3
        }
    
    def get_feature_names(self) -> List[str]:
        """Get names of technical features."""
        base_features = []
        
        # Pattern-based feature names
        for category in self.patterns.keys():
            base_features.extend([
                f'{category}_count',
                f'{category}_density',
                f'has_{category}'
            ])
        
        # Technical vocabulary features
        base_features.extend([
            'technical_term_count',
            'unique_technical_terms',
            'technical_density',
            'technical_diversity',
            'avg_technical_term_length'
        ])
        
        # Code quality features
        base_features.extend([
            'code_quality_mentions',
            'has_quality_focus',
            'quality_density'
        ])
        
        # Documentation features
        base_features.extend([
            'documentation_mentions',
            'has_documentation_focus',
            'has_structured_format',
            'has_code_examples',
            'documentation_completeness'
        ])
        
        return base_features


class CreativeDomainExtractor(BaseDomainExtractor):
    """Feature extractor for creative domains (writing, content creation, marketing)."""
    
    def __init__(self):
        super().__init__("creative")
        self._compile_creative_patterns()
    
    def _compile_creative_patterns(self):
        """Compile patterns for creative feature detection."""
        self.patterns = {
            'narrative_elements': [
                r'\b(?:character|protagonist|antagonist|hero|villain)\b',
                r'\b(?:plot|story|narrative|tale|saga)\b',
                r'\b(?:setting|scene|atmosphere|mood)\b',
                r'\b(?:conflict|tension|climax|resolution)\b'
            ],
            
            'literary_devices': [
                r'\b(?:metaphor|simile|symbolism|irony)\b',
                r'\b(?:alliteration|imagery|personification)\b',
                r'\b(?:foreshadowing|flashback|perspective)\b'
            ],
            
            'emotional_language': [
                r'\b(?:passionate|inspiring|emotional|heartfelt|overwhelming)\b',
                r'\b(?:compelling|engaging|captivating|thrilling|exciting)\b',
                r'\b(?:dramatic|intense|powerful|moving|beautiful|stunning)\b',
                r'\b(?:joy|excitement|hope|delight|magnificent)\b',
                r'\b(?:feel|felt|filled with|heart)\b'
            ],
            
            'creative_process': [
                r'\b(?:brainstorm|ideate|conceptualize|imagine)\b',
                r'\b(?:creative|innovative|original|unique)\b',
                r'\b(?:inspiration|muse|vision|artistic)\b'
            ]
        }
        
        self.compiled_patterns = {}
        for category, patterns in self.patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def extract_features(self, text: str, domain_result: DomainClassificationResult) -> Dict[str, Any]:
        """Extract creative domain features."""
        features = {}
        
        # Pattern-based features
        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))
            
            features[f'{category}_count'] = len(matches)
            features[f'{category}_presence'] = len(matches) > 0
        
        # Creative language analysis
        features.update(self._analyze_creative_language(text))
        
        # Narrative structure analysis
        features.update(self._analyze_narrative_structure(text))
        
        # Emotional tone analysis
        features.update(self._analyze_emotional_tone(text))
        
        return features
    
    def _analyze_creative_language(self, text: str) -> Dict[str, Any]:
        """Analyze creative language usage."""
        # Descriptive language indicators
        descriptive_patterns = [
            r'\b\w+ly\b',  # Adverbs
            r'\b(?:vivid|rich|detailed|elaborate|intricate)\b',
            r'\b(?:beautiful|stunning|magnificent|breathtaking)\b'
        ]
        
        descriptive_count = 0
        for pattern in descriptive_patterns:
            descriptive_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Sensory language
        sensory_patterns = [
            r'\b(?:see|look|watch|observe|glimpse)\b',
            r'\b(?:hear|listen|sound|echo|whisper)\b',
            r'\b(?:feel|touch|texture|smooth|rough)\b',
            r'\b(?:smell|scent|aroma|fragrance)\b',
            r'\b(?:taste|flavor|sweet|bitter|sour)\b'
        ]
        
        sensory_count = 0
        for pattern in sensory_patterns:
            sensory_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        return {
            'descriptive_language_count': descriptive_count,
            'sensory_language_count': sensory_count,
            'has_rich_description': descriptive_count > 0,
            'uses_sensory_details': sensory_count > 0
        }
    
    def _analyze_narrative_structure(self, text: str) -> Dict[str, Any]:
        """Analyze narrative structure elements."""
        # Story progression indicators
        progression_patterns = [
            r'\b(?:first|initially|beginning|start)\b',
            r'\b(?:then|next|subsequently|after)\b',
            r'\b(?:finally|eventually|conclusion|end)\b'
        ]
        
        progression_matches = 0
        for pattern in progression_patterns:
            progression_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Dialogue indicators
        has_dialogue = bool(re.search(r'"[^"]*"', text)) or bool(re.search(r"'[^']*'", text))
        dialogue_count = len(re.findall(r'"[^"]*"', text)) + len(re.findall(r"'[^']*'", text))
        
        return {
            'narrative_progression_indicators': progression_matches,
            'has_clear_progression': progression_matches >= 2,
            'has_dialogue': has_dialogue,
            'dialogue_count': dialogue_count
        }
    
    def _analyze_emotional_tone(self, text: str) -> Dict[str, Any]:
        """Analyze emotional tone and impact."""
        # Positive emotions
        positive_patterns = [
            r'\b(?:joy|happiness|excitement|delight|pleasure)\b',
            r'\b(?:love|affection|warmth|comfort|peace)\b',
            r'\b(?:hope|optimism|confidence|pride|satisfaction)\b'
        ]
        
        # Negative emotions
        negative_patterns = [
            r'\b(?:sadness|grief|sorrow|despair|melancholy)\b',
            r'\b(?:anger|rage|fury|irritation|frustration)\b',
            r'\b(?:fear|anxiety|worry|dread|terror)\b'
        ]
        
        positive_count = 0
        for pattern in positive_patterns:
            positive_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        negative_count = 0
        for pattern in negative_patterns:
            negative_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        total_emotional = positive_count + negative_count
        emotional_polarity = (positive_count - negative_count) / max(total_emotional, 1)
        
        return {
            'positive_emotion_count': positive_count,
            'negative_emotion_count': negative_count,
            'total_emotional_language': total_emotional,
            'emotional_polarity': emotional_polarity,
            'has_emotional_content': total_emotional > 0
        }
    
    def get_feature_names(self) -> List[str]:
        """Get names of creative features."""
        base_features = []
        
        # Pattern-based features
        for category in self.patterns.keys():
            base_features.extend([
                f'{category}_count',
                f'{category}_presence'
            ])
        
        # Creative language features
        base_features.extend([
            'descriptive_language_count',
            'sensory_language_count',
            'has_rich_description',
            'uses_sensory_details'
        ])
        
        # Narrative structure features
        base_features.extend([
            'narrative_progression_indicators',
            'has_clear_progression',
            'has_dialogue',
            'dialogue_count'
        ])
        
        # Emotional tone features
        base_features.extend([
            'positive_emotion_count',
            'negative_emotion_count',
            'total_emotional_language',
            'emotional_polarity',
            'has_emotional_content'
        ])
        
        return base_features


class AcademicDomainExtractor(BaseDomainExtractor):
    """Feature extractor for academic domains (research, education, scientific writing)."""
    
    def __init__(self):
        super().__init__("academic")
        self._compile_academic_patterns()
    
    def _compile_academic_patterns(self):
        """Compile patterns for academic feature detection."""
        self.patterns = {
            'research_methodology': [
                r'\b(?:methodology|method|approach|technique)\b',
                r'\b(?:quantitative|qualitative|mixed methods)\b',
                r'\b(?:survey|interview|experiment|observation)\b',
                r'\b(?:sample|population|participant|subject)\b'
            ],
            
            'academic_structure': [
                r'\b(?:abstract|introduction|literature review)\b',
                r'\b(?:methodology|results|discussion|conclusion)\b',
                r'\b(?:hypothesis|research question|objective)\b'
            ],
            
            'citation_patterns': [
                r'\([^)]*\d{4}[^)]*\)',  # Year citations
                r'\b(?:according to|as stated by|cited in)\b',
                r'\b(?:reference|citation|bibliography)\b'
            ],
            
            'academic_language': [
                r'\b(?:furthermore|moreover|however|nevertheless)\b',
                r'\b(?:consequently|therefore|thus|hence)\b',
                r'\b(?:investigate|examine|analyze|evaluate)\b'
            ]
        }
        
        self.compiled_patterns = {}
        for category, patterns in self.patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def extract_features(self, text: str, domain_result: DomainClassificationResult) -> Dict[str, Any]:
        """Extract academic domain features."""
        features = {}
        
        # Pattern-based features
        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))
            
            features[f'{category}_count'] = len(matches)
            features[f'has_{category}'] = len(matches) > 0
        
        # Academic rigor analysis
        features.update(self._analyze_academic_rigor(text))
        
        # Objectivity analysis
        features.update(self._analyze_objectivity(text))
        
        # Evidence-based reasoning
        features.update(self._analyze_evidence_based_reasoning(text))
        
        return features
    
    def _analyze_academic_rigor(self, text: str) -> Dict[str, Any]:
        """Analyze academic rigor indicators."""
        rigor_indicators = [
            r'\b(?:peer.reviewed|journal|publication)\b',
            r'\b(?:statistical|significant|analysis)\b',
            r'\b(?:evidence|data|findings|results)\b',
            r'\b(?:objective|systematic|rigorous)\b'
        ]
        
        rigor_count = 0
        for pattern in rigor_indicators:
            rigor_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Check for statistical terms
        statistical_terms = [
            r'\b(?:p.value|significance|correlation|regression)\b',
            r'\b(?:mean|median|standard deviation|variance)\b',
            r'\b(?:confidence interval|effect size)\b'
        ]
        
        statistical_count = 0
        for pattern in statistical_terms:
            statistical_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        return {
            'academic_rigor_indicators': rigor_count,
            'statistical_terms_count': statistical_count,
            'has_statistical_focus': statistical_count > 0,
            'rigor_density': rigor_count / max(len(text.split()), 1)
        }
    
    def _analyze_objectivity(self, text: str) -> Dict[str, Any]:
        """Analyze objectivity and formal tone."""
        # Subjective language indicators (lower objectivity)
        subjective_patterns = [
            r'\bi\s+(?:think|believe|feel|assume)\b',
            r'\b(?:obviously|clearly|definitely|certainly)\b',
            r'\b(?:amazing|terrible|wonderful|awful)\b'
        ]
        
        subjective_count = 0
        for pattern in subjective_patterns:
            subjective_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Objective language indicators
        objective_patterns = [
            r'\b(?:the study|research indicates|findings suggest)\b',
            r'\b(?:it was found|results show|data indicate)\b',
            r'\b(?:according to|based on|evidence suggests)\b'
        ]
        
        objective_count = 0
        for pattern in objective_patterns:
            objective_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        total_tone_indicators = subjective_count + objective_count
        objectivity_ratio = objective_count / max(total_tone_indicators, 1)
        
        return {
            'subjective_language_count': subjective_count,
            'objective_language_count': objective_count,
            'objectivity_ratio': objectivity_ratio,
            'has_formal_tone': objectivity_ratio > 0.5
        }
    
    def _analyze_evidence_based_reasoning(self, text: str) -> Dict[str, Any]:
        """Analyze evidence-based reasoning patterns."""
        evidence_patterns = [
            r'\b(?:evidence|proof|data|study|research)\b',
            r'\b(?:supports|demonstrates|indicates|reveals)\b',
            r'\b(?:based on|according to|as shown by)\b'
        ]
        
        evidence_count = 0
        for pattern in evidence_patterns:
            evidence_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Reasoning connectors
        reasoning_patterns = [
            r'\b(?:because|since|due to|as a result)\b',
            r'\b(?:therefore|thus|consequently|hence)\b',
            r'\b(?:leads to|results in|causes|explains)\b'
        ]
        
        reasoning_count = 0
        for pattern in reasoning_patterns:
            reasoning_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        return {
            'evidence_references': evidence_count,
            'reasoning_connectors': reasoning_count,
            'has_evidence_base': evidence_count > 0,
            'has_logical_structure': reasoning_count >= 2
        }
    
    def get_feature_names(self) -> List[str]:
        """Get names of academic features."""
        base_features = []
        
        # Pattern-based features
        for category in self.patterns.keys():
            base_features.extend([
                f'{category}_count',
                f'has_{category}'
            ])
        
        # Academic rigor features
        base_features.extend([
            'academic_rigor_indicators',
            'statistical_terms_count',
            'has_statistical_focus',
            'rigor_density'
        ])
        
        # Objectivity features
        base_features.extend([
            'subjective_language_count',
            'objective_language_count',
            'objectivity_ratio',
            'has_formal_tone'
        ])
        
        # Evidence-based reasoning features
        base_features.extend([
            'evidence_references',
            'reasoning_connectors',
            'has_evidence_base',
            'has_logical_structure'
        ])
        
        return base_features


class DomainFeatureExtractor:
    """Main coordinator for domain-specific feature extraction."""
    
    def __init__(self, enable_spacy: bool = True):
        """Initialize the domain feature extractor."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize domain detector
        self.domain_detector = DomainDetector(use_spacy=enable_spacy)
        
        # Initialize domain-specific extractors
        self.extractors = {
            'technical': TechnicalDomainExtractor(),
            'creative': CreativeDomainExtractor(),
            'academic': AcademicDomainExtractor(),
        }
        
        # Cache for feature extraction results
        self._feature_cache = {}
    
    @lru_cache(maxsize=500)
    def extract_domain_features(self, text: str) -> DomainFeatures:
        """
        Extract comprehensive domain-specific features from text.
        
        Args:
            text: Input prompt text
            
        Returns:
            DomainFeatures object with extracted features
        """
        if not text or not text.strip():
            return DomainFeatures(
                domain=PromptDomain.GENERAL,
                confidence=0.0
            )
        
        # Detect domain
        domain_result = self.domain_detector.detect_domain(text)
        
        # Initialize feature container
        features = DomainFeatures(
            domain=domain_result.primary_domain,
            confidence=domain_result.confidence,
            complexity_score=domain_result.technical_complexity,
            specificity_score=domain_result.domain_specificity,
            secondary_domains=domain_result.secondary_domains,
            hybrid_domain=domain_result.hybrid_domain
        )
        
        # Extract features based on domain type (including secondary domains for hybrid prompts)
        all_domains = [domain_result.primary_domain] + [d[0] for d in domain_result.secondary_domains]
        
        # Check if any domain is technical
        if any(self._is_technical_domain(d) for d in all_domains):
            features.technical_features = self.extractors['technical'].extract_features(text, domain_result)
        
        # Check if any domain is creative  
        if any(self._is_creative_domain(d) for d in all_domains):
            features.creative_features = self.extractors['creative'].extract_features(text, domain_result)
        
        # Check if any domain is academic
        if any(self._is_academic_domain(d) for d in all_domains):
            features.academic_features = self.extractors['academic'].extract_features(text, domain_result)
        
        # Extract general conversational features
        features.conversational_features = self._extract_conversational_features(text)
        
        # Create unified feature vector
        features.feature_vector, features.feature_names = self._create_feature_vector(features)
        
        return features
    
    def _is_technical_domain(self, domain: PromptDomain) -> bool:
        """Check if domain is technical."""
        technical_domains = {
            PromptDomain.SOFTWARE_DEVELOPMENT,
            PromptDomain.DATA_SCIENCE,
            PromptDomain.AI_ML,
            PromptDomain.WEB_DEVELOPMENT,
            PromptDomain.SYSTEM_ADMIN,
            PromptDomain.API_DOCUMENTATION,
        }
        return domain in technical_domains
    
    def _is_creative_domain(self, domain: PromptDomain) -> bool:
        """Check if domain is creative."""
        creative_domains = {
            PromptDomain.CREATIVE_WRITING,
            PromptDomain.CONTENT_CREATION,
            PromptDomain.MARKETING,
            PromptDomain.STORYTELLING,
        }
        return domain in creative_domains
    
    def _is_academic_domain(self, domain: PromptDomain) -> bool:
        """Check if domain is academic."""
        academic_domains = {
            PromptDomain.RESEARCH,
            PromptDomain.EDUCATION,
            PromptDomain.ACADEMIC_WRITING,
            PromptDomain.SCIENTIFIC,
        }
        return domain in academic_domains
    
    def _extract_conversational_features(self, text: str) -> Dict[str, Any]:
        """Extract general conversational features."""
        features = {}
        
        # Question patterns
        question_count = len(re.findall(r'\?', text))
        features['question_count'] = question_count
        features['has_questions'] = question_count > 0
        
        # Instruction patterns
        instruction_patterns = [
            r'\b(?:please|could you|can you|would you)\b',
            r'\b(?:explain|describe|tell me|show me)\b',
            r'\b(?:help|assist|guide|walk through)\b'
        ]
        
        instruction_count = 0
        for pattern in instruction_patterns:
            instruction_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        features['instruction_indicators'] = instruction_count
        features['has_polite_requests'] = instruction_count > 0
        
        # Urgency indicators
        urgency_patterns = [
            r'\b(?:urgent|asap|quickly|immediately|now)\b',
            r'\b(?:deadline|due|time.sensitive)\b'
        ]
        
        urgency_count = 0
        for pattern in urgency_patterns:
            urgency_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        features['urgency_indicators'] = urgency_count
        features['has_urgency'] = urgency_count > 0
        
        # Sentiment indicators
        positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        positive_count = sum(text.lower().count(word) for word in positive_words)
        negative_count = sum(text.lower().count(word) for word in negative_words)
        
        features['positive_sentiment_count'] = positive_count
        features['negative_sentiment_count'] = negative_count
        features['sentiment_polarity'] = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        
        return features
    
    def _create_feature_vector(self, features: DomainFeatures) -> Tuple[List[float], List[str]]:
        """Create unified feature vector from domain features."""
        feature_vector = []
        feature_names = []
        
        # Core domain features
        feature_vector.extend([
            features.confidence,
            features.complexity_score,
            features.specificity_score
        ])
        feature_names.extend([
            'domain_confidence',
            'domain_complexity',
            'domain_specificity'
        ])
        
        # Technical features
        for key, value in features.technical_features.items():
            feature_vector.append(float(value) if isinstance(value, (int, float, bool)) else 0.0)
            feature_names.append(f'tech_{key}')
        
        # Creative features
        for key, value in features.creative_features.items():
            feature_vector.append(float(value) if isinstance(value, (int, float, bool)) else 0.0)
            feature_names.append(f'creative_{key}')
        
        # Academic features
        for key, value in features.academic_features.items():
            feature_vector.append(float(value) if isinstance(value, (int, float, bool)) else 0.0)
            feature_names.append(f'academic_{key}')
        
        # Conversational features
        for key, value in features.conversational_features.items():
            feature_vector.append(float(value) if isinstance(value, (int, float, bool)) else 0.0)
            feature_names.append(f'conv_{key}')
        
        return feature_vector, feature_names
    
    async def extract_domain_features_async(self, text: str) -> DomainFeatures:
        """Async version of domain feature extraction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_domain_features, text)
    
    def get_all_feature_names(self) -> List[str]:
        """Get all possible feature names."""
        feature_names = [
            'domain_confidence',
            'domain_complexity', 
            'domain_specificity'
        ]
        
        # Add extractor feature names
        for extractor_name, extractor in self.extractors.items():
            extractor_features = extractor.get_feature_names()
            feature_names.extend([f'{extractor_name}_{name}' for name in extractor_features])
        
        # Add conversational feature names
        conv_features = [
            'question_count', 'has_questions', 'instruction_indicators',
            'has_polite_requests', 'urgency_indicators', 'has_urgency',
            'positive_sentiment_count', 'negative_sentiment_count', 'sentiment_polarity'
        ]
        feature_names.extend([f'conv_{name}' for name in conv_features])
        
        return feature_names
    
    def analyze_prompt_domain_suitability(self, text: str, target_domain: PromptDomain) -> Dict[str, Any]:
        """
        Analyze how well a prompt fits a target domain.
        
        Args:
            text: Input prompt text
            target_domain: Target domain to analyze against
            
        Returns:
            Analysis results with suitability score and recommendations
        """
        features = self.extract_domain_features(text)
        
        # Calculate domain alignment
        primary_match = features.domain == target_domain
        secondary_match = any(domain == target_domain for domain, _ in features.secondary_domains)
        
        # Get domain-specific keywords
        target_keywords = self.domain_detector.get_domain_keywords(target_domain)
        found_keywords = sum(1 for keyword in target_keywords if keyword.lower() in text.lower())
        keyword_coverage = found_keywords / max(len(target_keywords), 1)
        
        # Calculate suitability score
        if primary_match:
            suitability_score = features.confidence
        elif secondary_match:
            # Find the secondary domain score
            secondary_score = next((score for domain, score in features.secondary_domains if domain == target_domain), 0.0)
            suitability_score = secondary_score * 0.7  # Discount for secondary match
        else:
            suitability_score = keyword_coverage * 0.3  # Low score for keyword-only match
        
        # Generate recommendations
        recommendations = []
        if suitability_score < 0.5:
            recommendations.append(f"Consider adding more {target_domain.value}-specific terminology")
            recommendations.append(f"Include domain-specific examples or use cases")
        
        if keyword_coverage < 0.1:
            sample_keywords = list(target_keywords)[:5]
            recommendations.append(f"Consider including keywords like: {', '.join(sample_keywords)}")
        
        return {
            'target_domain': target_domain.value,
            'suitability_score': suitability_score,
            'primary_domain_match': primary_match,
            'secondary_domain_match': secondary_match,
            'keyword_coverage': keyword_coverage,
            'found_keywords_count': found_keywords,
            'recommendations': recommendations,
            'domain_confidence': features.confidence,
            'detected_domain': features.domain.value
        }