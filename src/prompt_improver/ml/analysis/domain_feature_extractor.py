"""Domain-Specific Feature Extraction System

This module provides specialized feature extraction based on detected prompt domains,
enabling more targeted and relevant analysis for different types of prompts.
"""
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
from functools import lru_cache
import logging
import re
from typing import Any
from .domain_detector import DomainClassificationResult, DomainDetector, PromptDomain
try:
    import spacy
    from spacy.language import language
    from spacy.matcher import matcher, phrase_matcher
    from spacy.tokens import doc, span, token
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    language = None
    matcher = None
    phrase_matcher = None
    doc = None
    span = None
    token = None
    SPACY_AVAILABLE = False

@dataclass
class DomainFeatures:
    """Container for domain-specific features."""
    domain: PromptDomain
    confidence: float
    complexity_score: float = 0.0
    specificity_score: float = 0.0
    secondary_domains: list[tuple['PromptDomain', float]] = field(default_factory=list)
    hybrid_domain: bool = False
    technical_features: dict[str, Any] = field(default_factory=dict)
    creative_features: dict[str, Any] = field(default_factory=dict)
    academic_features: dict[str, Any] = field(default_factory=dict)
    business_features: dict[str, Any] = field(default_factory=dict)
    conversational_features: dict[str, Any] = field(default_factory=dict)
    feature_vector: list[float] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)

class BaseDomainExtractor(ABC):
    """Abstract base class for domain-specific feature extractors."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

    @abstractmethod
    def extract_features(self, text: str, domain_result: DomainClassificationResult) -> dict[str, Any]:
        """Extract domain-specific features from text."""

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Get names of features extracted by this extractor."""

class TechnicalDomainExtractor(BaseDomainExtractor):
    """Feature extractor for technical domains (software, data science, AI/ML, etc.)."""

    def __init__(self):
        super().__init__('technical')
        self._compile_technical_patterns()

    def _compile_technical_patterns(self):
        """Compile regex patterns for technical feature detection."""
        self.patterns = {'code_snippets': ['```[\\s\\S]*?```', '`[^`\\n]+`', '\\b(?:def|function|class|import|from|if|else|for|while)\\s+\\w+'], 'api_references': ['\\b(?:GET|POST|PUT|DELETE|PATCH)\\s+/', '\\b(?:api|endpoint|route)\\b.*?/', '\\{[\\w\\s,:"\\\']*\\}'], 'algorithms': ['\\b(?:algorithm|sorting|searching|optimization|recursion)\\b', '\\b(?:big o|time complexity|space complexity)\\b', '\\b(?:binary search|quicksort|mergesort|hash table)\\b'], 'data_structures': ['\\b(?:array|list|dictionary|hash|tree|graph|queue|stack)\\b', '\\b(?:linked list|binary tree|heap|trie)\\b'], 'ml_concepts': ['\\b(?:neural network|deep learning|machine learning)\\b', '\\b(?:training|validation|test)\\s+(?:set|data)\\b', '\\b(?:accuracy|precision|recall|f1[-_]score|auc|roc)\\b', '\\b(?:supervised|unsupervised|reinforcement)\\s+learning\\b'], 'architecture': ['\\b(?:microservice|api gateway|load balancer)\\b', '\\b(?:docker|kubernetes|container|deployment)\\b', '\\b(?:scalability|high availability|fault tolerance)\\b']}
        self.compiled_patterns = {}
        for category, patterns in self.patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]

    def extract_features(self, text: str, domain_result: DomainClassificationResult) -> dict[str, Any]:
        """Extract technical domain features."""
        features = {}
        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))
            features[f'{category}_count'] = len(matches)
            features[f'{category}_density'] = len(matches) / max(len(text.split()), 1)
            features[f'has_{category}'] = len(matches) > 0
        features.update(self._analyze_technical_vocabulary(text, domain_result))
        features.update(self._analyze_code_quality_indicators(text))
        features.update(self._analyze_documentation_patterns(text))
        return features

    def _analyze_technical_vocabulary(self, text: str, domain_result: DomainClassificationResult) -> dict[str, Any]:
        """Analyze technical vocabulary usage."""
        technical_terms = []
        for keywords in domain_result.domain_keywords_found.values():
            technical_terms.extend(keywords)
        word_count = len(text.split())
        technical_density = len(technical_terms) / max(word_count, 1)
        unique_technical_terms = len(set(technical_terms))
        technical_diversity = unique_technical_terms / max(len(technical_terms), 1)
        return {'technical_term_count': len(technical_terms), 'unique_technical_terms': unique_technical_terms, 'technical_density': technical_density, 'technical_diversity': technical_diversity, 'avg_technical_term_length': sum((len(term) for term in technical_terms)) / max(len(technical_terms), 1)}

    def _analyze_code_quality_indicators(self, text: str) -> dict[str, Any]:
        """Analyze indicators of code quality discussion."""
        quality_indicators = ['\\b(?:clean code|best practice|code review|refactor)\\b', '\\b(?:maintainable|readable|scalable|testable)\\b', '\\b(?:design pattern|solid principle|dry principle)\\b', '\\b(?:unit test|integration test|test coverage)\\b', '\\b(?:performance|optimization|efficiency)\\b', '\\b(?:security|vulnerability|authentication|authorization)\\b']
        quality_matches = 0
        for pattern in quality_indicators:
            quality_matches += len(re.findall(pattern, text, re.IGNORECASE))
        return {'code_quality_mentions': quality_matches, 'has_quality_focus': quality_matches > 0, 'quality_density': quality_matches / max(len(text.split()), 1)}

    def _analyze_documentation_patterns(self, text: str) -> dict[str, Any]:
        """Analyze documentation and explanation patterns."""
        doc_patterns = ['\\b(?:explain|describe|how to|step by step)\\b', '\\b(?:example|sample|demo|tutorial)\\b', '\\b(?:documentation|readme|guide|manual)\\b', '\\b(?:usage|installation|setup|configuration)\\b']
        doc_matches = 0
        for pattern in doc_patterns:
            doc_matches += len(re.findall(pattern, text, re.IGNORECASE))
        has_headings = bool(re.search('^#+\\s+', text, re.MULTILINE))
        has_lists = bool(re.search('^\\s*[-*+]\\s+|^\\s*\\d+\\.\\s+', text, re.MULTILINE))
        has_code_blocks = bool(re.search('```[\\s\\S]*?```', text))
        return {'documentation_mentions': doc_matches, 'has_documentation_focus': doc_matches > 0, 'has_structured_format': has_headings or has_lists, 'has_code_examples': has_code_blocks, 'documentation_completeness': sum([has_headings, has_lists, has_code_blocks]) / 3}

    def get_feature_names(self) -> list[str]:
        """Get names of technical features."""
        base_features = []
        for category in self.patterns.keys():
            base_features.extend([f'{category}_count', f'{category}_density', f'has_{category}'])
        base_features.extend(['technical_term_count', 'unique_technical_terms', 'technical_density', 'technical_diversity', 'avg_technical_term_length'])
        base_features.extend(['code_quality_mentions', 'has_quality_focus', 'quality_density'])
        base_features.extend(['documentation_mentions', 'has_documentation_focus', 'has_structured_format', 'has_code_examples', 'documentation_completeness'])
        return base_features

class CreativeDomainExtractor(BaseDomainExtractor):
    """Feature extractor for creative domains (writing, content creation, marketing)."""

    def __init__(self):
        super().__init__('creative')
        self._compile_creative_patterns()

    def _compile_creative_patterns(self):
        """Compile patterns for creative feature detection."""
        self.patterns = {'narrative_elements': ['\\b(?:character|protagonist|antagonist|hero|villain)\\b', '\\b(?:plot|story|narrative|tale|saga)\\b', '\\b(?:setting|scene|atmosphere|mood)\\b', '\\b(?:conflict|tension|climax|resolution)\\b'], 'literary_devices': ['\\b(?:metaphor|simile|symbolism|irony)\\b', '\\b(?:alliteration|imagery|personification)\\b', '\\b(?:foreshadowing|flashback|perspective)\\b'], 'emotional_language': ['\\b(?:passionate|inspiring|emotional|heartfelt|overwhelming)\\b', '\\b(?:compelling|engaging|captivating|thrilling|exciting)\\b', '\\b(?:dramatic|intense|powerful|moving|beautiful|stunning)\\b', '\\b(?:joy|excitement|hope|delight|magnificent)\\b', '\\b(?:feel|felt|filled with|heart)\\b'], 'creative_process': ['\\b(?:brainstorm|ideate|conceptualize|imagine)\\b', '\\b(?:creative|innovative|original|unique)\\b', '\\b(?:inspiration|muse|vision|artistic)\\b']}
        self.compiled_patterns = {}
        for category, patterns in self.patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def extract_features(self, text: str, domain_result: DomainClassificationResult) -> dict[str, Any]:
        """Extract creative domain features."""
        features = {}
        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))
            features[f'{category}_count'] = len(matches)
            features[f'{category}_presence'] = len(matches) > 0
        features.update(self._analyze_creative_language(text))
        features.update(self._analyze_narrative_structure(text))
        features.update(self._analyze_emotional_tone(text))
        return features

    def _analyze_creative_language(self, text: str) -> dict[str, Any]:
        """Analyze creative language usage."""
        descriptive_patterns = ['\\b\\w+ly\\b', '\\b(?:vivid|rich|detailed|elaborate|intricate)\\b', '\\b(?:beautiful|stunning|magnificent|breathtaking)\\b']
        descriptive_count = 0
        for pattern in descriptive_patterns:
            descriptive_count += len(re.findall(pattern, text, re.IGNORECASE))
        sensory_patterns = ['\\b(?:see|look|watch|observe|glimpse)\\b', '\\b(?:hear|listen|sound|echo|whisper)\\b', '\\b(?:feel|touch|texture|smooth|rough)\\b', '\\b(?:smell|scent|aroma|fragrance)\\b', '\\b(?:taste|flavor|sweet|bitter|sour)\\b']
        sensory_count = 0
        for pattern in sensory_patterns:
            sensory_count += len(re.findall(pattern, text, re.IGNORECASE))
        return {'descriptive_language_count': descriptive_count, 'sensory_language_count': sensory_count, 'has_rich_description': descriptive_count > 0, 'uses_sensory_details': sensory_count > 0}

    def _analyze_narrative_structure(self, text: str) -> dict[str, Any]:
        """Analyze narrative structure elements."""
        progression_patterns = ['\\b(?:first|initially|beginning|start)\\b', '\\b(?:then|next|subsequently|after)\\b', '\\b(?:finally|eventually|conclusion|end)\\b']
        progression_matches = 0
        for pattern in progression_patterns:
            progression_matches += len(re.findall(pattern, text, re.IGNORECASE))
        has_dialogue = bool(re.search('"[^"]*"', text)) or bool(re.search("'[^']*'", text))
        dialogue_count = len(re.findall('"[^"]*"', text)) + len(re.findall("'[^']*'", text))
        return {'narrative_progression_indicators': progression_matches, 'has_clear_progression': progression_matches >= 2, 'has_dialogue': has_dialogue, 'dialogue_count': dialogue_count}

    def _analyze_emotional_tone(self, text: str) -> dict[str, Any]:
        """Analyze emotional tone and impact."""
        positive_patterns = ['\\b(?:joy|happiness|excitement|delight|pleasure)\\b', '\\b(?:love|affection|warmth|comfort|peace)\\b', '\\b(?:hope|optimism|confidence|pride|satisfaction)\\b']
        negative_patterns = ['\\b(?:sadness|grief|sorrow|despair|melancholy)\\b', '\\b(?:anger|rage|fury|irritation|frustration)\\b', '\\b(?:fear|anxiety|worry|dread|terror)\\b']
        positive_count = 0
        for pattern in positive_patterns:
            positive_count += len(re.findall(pattern, text, re.IGNORECASE))
        negative_count = 0
        for pattern in negative_patterns:
            negative_count += len(re.findall(pattern, text, re.IGNORECASE))
        total_emotional = positive_count + negative_count
        emotional_polarity = (positive_count - negative_count) / max(total_emotional, 1)
        return {'positive_emotion_count': positive_count, 'negative_emotion_count': negative_count, 'total_emotional_language': total_emotional, 'emotional_polarity': emotional_polarity, 'has_emotional_content': total_emotional > 0}

    def get_feature_names(self) -> list[str]:
        """Get names of creative features."""
        base_features = []
        for category in self.patterns.keys():
            base_features.extend([f'{category}_count', f'{category}_presence'])
        base_features.extend(['descriptive_language_count', 'sensory_language_count', 'has_rich_description', 'uses_sensory_details'])
        base_features.extend(['narrative_progression_indicators', 'has_clear_progression', 'has_dialogue', 'dialogue_count'])
        base_features.extend(['positive_emotion_count', 'negative_emotion_count', 'total_emotional_language', 'emotional_polarity', 'has_emotional_content'])
        return base_features

class AcademicDomainExtractor(BaseDomainExtractor):
    """Feature extractor for academic domains (research, education, scientific writing)."""

    def __init__(self):
        super().__init__('academic')
        self._compile_academic_patterns()

    def _compile_academic_patterns(self):
        """Compile patterns for academic feature detection."""
        self.patterns = {'research_methodology': ['\\b(?:methodology|method|approach|technique)\\b', '\\b(?:quantitative|qualitative|mixed methods)\\b', '\\b(?:survey|interview|experiment|observation)\\b', '\\b(?:sample|population|participant|subject)\\b'], 'academic_structure': ['\\b(?:abstract|introduction|literature review)\\b', '\\b(?:methodology|results|discussion|conclusion)\\b', '\\b(?:hypothesis|research question|objective)\\b'], 'citation_patterns': ['\\([^)]*\\d{4}[^)]*\\)', '\\b(?:according to|as stated by|cited in)\\b', '\\b(?:reference|citation|bibliography)\\b'], 'academic_language': ['\\b(?:furthermore|moreover|however|nevertheless)\\b', '\\b(?:consequently|therefore|thus|hence)\\b', '\\b(?:investigate|examine|analyze|evaluate)\\b']}
        self.compiled_patterns = {}
        for category, patterns in self.patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def extract_features(self, text: str, domain_result: DomainClassificationResult) -> dict[str, Any]:
        """Extract academic domain features."""
        features = {}
        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))
            features[f'{category}_count'] = len(matches)
            features[f'has_{category}'] = len(matches) > 0
        features.update(self._analyze_academic_rigor(text))
        features.update(self._analyze_objectivity(text))
        features.update(self._analyze_evidence_based_reasoning(text))
        return features

    def _analyze_academic_rigor(self, text: str) -> dict[str, Any]:
        """Analyze academic rigor indicators."""
        rigor_indicators = ['\\b(?:peer.reviewed|journal|publication)\\b', '\\b(?:statistical|significant|analysis)\\b', '\\b(?:evidence|data|findings|results)\\b', '\\b(?:objective|systematic|rigorous)\\b']
        rigor_count = 0
        for pattern in rigor_indicators:
            rigor_count += len(re.findall(pattern, text, re.IGNORECASE))
        statistical_terms = ['\\b(?:p.value|significance|correlation|regression)\\b', '\\b(?:mean|median|standard deviation|variance)\\b', '\\b(?:confidence interval|effect size)\\b']
        statistical_count = 0
        for pattern in statistical_terms:
            statistical_count += len(re.findall(pattern, text, re.IGNORECASE))
        return {'academic_rigor_indicators': rigor_count, 'statistical_terms_count': statistical_count, 'has_statistical_focus': statistical_count > 0, 'rigor_density': rigor_count / max(len(text.split()), 1)}

    def _analyze_objectivity(self, text: str) -> dict[str, Any]:
        """Analyze objectivity and formal tone."""
        subjective_patterns = ['\\bi\\s+(?:think|believe|feel|assume)\\b', '\\b(?:obviously|clearly|definitely|certainly)\\b', '\\b(?:amazing|terrible|wonderful|awful)\\b']
        subjective_count = 0
        for pattern in subjective_patterns:
            subjective_count += len(re.findall(pattern, text, re.IGNORECASE))
        objective_patterns = ['\\b(?:the study|research indicates|findings suggest)\\b', '\\b(?:it was found|results show|data indicate)\\b', '\\b(?:according to|based on|evidence suggests)\\b']
        objective_count = 0
        for pattern in objective_patterns:
            objective_count += len(re.findall(pattern, text, re.IGNORECASE))
        total_tone_indicators = subjective_count + objective_count
        objectivity_ratio = objective_count / max(total_tone_indicators, 1)
        return {'subjective_language_count': subjective_count, 'objective_language_count': objective_count, 'objectivity_ratio': objectivity_ratio, 'has_formal_tone': objectivity_ratio > 0.5}

    def _analyze_evidence_based_reasoning(self, text: str) -> dict[str, Any]:
        """Analyze evidence-based reasoning patterns."""
        evidence_patterns = ['\\b(?:evidence|proof|data|study|research)\\b', '\\b(?:supports|demonstrates|indicates|reveals)\\b', '\\b(?:based on|according to|as shown by)\\b']
        evidence_count = 0
        for pattern in evidence_patterns:
            evidence_count += len(re.findall(pattern, text, re.IGNORECASE))
        reasoning_patterns = ['\\b(?:because|since|due to|as a result)\\b', '\\b(?:therefore|thus|consequently|hence)\\b', '\\b(?:leads to|results in|causes|explains)\\b']
        reasoning_count = 0
        for pattern in reasoning_patterns:
            reasoning_count += len(re.findall(pattern, text, re.IGNORECASE))
        return {'evidence_references': evidence_count, 'reasoning_connectors': reasoning_count, 'has_evidence_base': evidence_count > 0, 'has_logical_structure': reasoning_count >= 2}

    def get_feature_names(self) -> list[str]:
        """Get names of academic features."""
        base_features = []
        for category in self.patterns.keys():
            base_features.extend([f'{category}_count', f'has_{category}'])
        base_features.extend(['academic_rigor_indicators', 'statistical_terms_count', 'has_statistical_focus', 'rigor_density'])
        base_features.extend(['subjective_language_count', 'objective_language_count', 'objectivity_ratio', 'has_formal_tone'])
        base_features.extend(['evidence_references', 'reasoning_connectors', 'has_evidence_base', 'has_logical_structure'])
        return base_features

class DomainFeatureExtractor:
    """Main coordinator for domain-specific feature extraction."""

    def __init__(self, enable_spacy: bool=True):
        """Initialize the domain feature extractor."""
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self.domain_detector = DomainDetector(use_spacy=enable_spacy)
        self.extractors = {'technical': TechnicalDomainExtractor(), 'creative': CreativeDomainExtractor(), 'academic': AcademicDomainExtractor()}
        self._feature_cache = {}

    @lru_cache(maxsize=500)
    def extract_domain_features(self, text: str) -> DomainFeatures:
        """Extract comprehensive domain-specific features from text.

        Args:
            text: Input prompt text

        Returns:
            DomainFeatures object with extracted features
        """
        if not text or not text.strip():
            return DomainFeatures(domain=PromptDomain.GENERAL, confidence=0.0)
        domain_result = self.domain_detector.detect_domain(text)
        features = DomainFeatures(domain=domain_result.primary_domain, confidence=domain_result.confidence, complexity_score=domain_result.technical_complexity, specificity_score=domain_result.domain_specificity, secondary_domains=domain_result.secondary_domains, hybrid_domain=domain_result.hybrid_domain)
        all_domains = [domain_result.primary_domain] + [d[0] for d in domain_result.secondary_domains]
        if any((self._is_technical_domain(d) for d in all_domains)):
            features.technical_features = self.extractors['technical'].extract_features(text, domain_result)
        if any((self._is_creative_domain(d) for d in all_domains)):
            features.creative_features = self.extractors['creative'].extract_features(text, domain_result)
        if any((self._is_academic_domain(d) for d in all_domains)):
            features.academic_features = self.extractors['academic'].extract_features(text, domain_result)
        features.conversational_features = self._extract_conversational_features(text)
        features.feature_vector, features.feature_names = self._create_feature_vector(features)
        return features

    def _is_technical_domain(self, domain: PromptDomain) -> bool:
        """Check if domain is technical."""
        technical_domains = {PromptDomain.SOFTWARE_DEVELOPMENT, PromptDomain.DATA_SCIENCE, PromptDomain.AI_ML, PromptDomain.WEB_DEVELOPMENT, PromptDomain.SYSTEM_ADMIN, PromptDomain.API_DOCUMENTATION}
        return domain in technical_domains

    def _is_creative_domain(self, domain: PromptDomain) -> bool:
        """Check if domain is creative."""
        creative_domains = {PromptDomain.CREATIVE_WRITING, PromptDomain.CONTENT_CREATION, PromptDomain.MARKETING, PromptDomain.STORYTELLING}
        return domain in creative_domains

    def _is_academic_domain(self, domain: PromptDomain) -> bool:
        """Check if domain is academic."""
        academic_domains = {PromptDomain.RESEARCH, PromptDomain.EDUCATION, PromptDomain.ACADEMIC_WRITING, PromptDomain.SCIENTIFIC}
        return domain in academic_domains

    def _extract_conversational_features(self, text: str) -> dict[str, Any]:
        """Extract general conversational features."""
        features = {}
        question_count = len(re.findall('\\?', text))
        features['question_count'] = question_count
        features['has_questions'] = question_count > 0
        instruction_patterns = ['\\b(?:please|could you|can you|would you)\\b', '\\b(?:explain|describe|tell me|show me)\\b', '\\b(?:help|assist|guide|walk through)\\b']
        instruction_count = 0
        for pattern in instruction_patterns:
            instruction_count += len(re.findall(pattern, text, re.IGNORECASE))
        features['instruction_indicators'] = instruction_count
        features['has_polite_requests'] = instruction_count > 0
        urgency_patterns = ['\\b(?:urgent|asap|quickly|immediately|now)\\b', '\\b(?:deadline|due|time.sensitive)\\b']
        urgency_count = 0
        for pattern in urgency_patterns:
            urgency_count += len(re.findall(pattern, text, re.IGNORECASE))
        features['urgency_indicators'] = urgency_count
        features['has_urgency'] = urgency_count > 0
        positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        positive_count = sum((text.lower().count(word) for word in positive_words))
        negative_count = sum((text.lower().count(word) for word in negative_words))
        features['positive_sentiment_count'] = positive_count
        features['negative_sentiment_count'] = negative_count
        features['sentiment_polarity'] = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        return features

    def _create_feature_vector(self, features: DomainFeatures) -> tuple[list[float], list[str]]:
        """Create unified feature vector from domain features."""
        feature_vector = []
        feature_names = []
        feature_vector.extend([features.confidence, features.complexity_score, features.specificity_score])
        feature_names.extend(['domain_confidence', 'domain_complexity', 'domain_specificity'])
        for key, value in features.technical_features.items():
            feature_vector.append(float(value) if isinstance(value, (int, float, bool)) else 0.0)
            feature_names.append(f'tech_{key}')
        for key, value in features.creative_features.items():
            feature_vector.append(float(value) if isinstance(value, (int, float, bool)) else 0.0)
            feature_names.append(f'creative_{key}')
        for key, value in features.academic_features.items():
            feature_vector.append(float(value) if isinstance(value, (int, float, bool)) else 0.0)
            feature_names.append(f'academic_{key}')
        for key, value in features.conversational_features.items():
            feature_vector.append(float(value) if isinstance(value, (int, float, bool)) else 0.0)
            feature_names.append(f'conv_{key}')
        return (feature_vector, feature_names)

    async def extract_domain_features_async(self, text: str) -> DomainFeatures:
        """Async version of domain feature extraction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_domain_features, text)

    def get_all_feature_names(self) -> list[str]:
        """Get all possible feature names."""
        feature_names = ['domain_confidence', 'domain_complexity', 'domain_specificity']
        for extractor_name, extractor in self.extractors.items():
            extractor_features = extractor.get_feature_names()
            feature_names.extend([f'{extractor_name}_{name}' for name in extractor_features])
        conv_features = ['question_count', 'has_questions', 'instruction_indicators', 'has_polite_requests', 'urgency_indicators', 'has_urgency', 'positive_sentiment_count', 'negative_sentiment_count', 'sentiment_polarity']
        feature_names.extend([f'conv_{name}' for name in conv_features])
        return feature_names

    def analyze_prompt_domain_suitability(self, text: str, target_domain: PromptDomain) -> dict[str, Any]:
        """Analyze how well a prompt fits a target domain.

        Args:
            text: Input prompt text
            target_domain: Target domain to analyze against

        Returns:
            Analysis results with suitability score and recommendations
        """
        features = self.extract_domain_features(text)
        primary_match = features.domain == target_domain
        secondary_match = any((domain == target_domain for domain, _ in features.secondary_domains))
        target_keywords = self.domain_detector.get_domain_keywords(target_domain)
        found_keywords = sum((1 for keyword in target_keywords if keyword.lower() in text.lower()))
        keyword_coverage = found_keywords / max(len(target_keywords), 1)
        if primary_match:
            suitability_score = features.confidence
        elif secondary_match:
            secondary_score = next((score for domain, score in features.secondary_domains if domain == target_domain), 0.0)
            suitability_score = secondary_score * 0.7
        else:
            suitability_score = keyword_coverage * 0.3
        recommendations = []
        if suitability_score < 0.5:
            recommendations.append(f'Consider adding more {target_domain.value}-specific terminology')
            recommendations.append('Include domain-specific examples or use cases')
        if keyword_coverage < 0.1:
            sample_keywords = list(target_keywords)[:5]
            recommendations.append(f"Consider including keywords like: {', '.join(sample_keywords)}")
        return {'target_domain': target_domain.value, 'suitability_score': suitability_score, 'primary_domain_match': primary_match, 'secondary_domain_match': secondary_match, 'keyword_coverage': keyword_coverage, 'found_keywords_count': found_keywords, 'recommendations': recommendations, 'domain_confidence': features.confidence, 'detected_domain': features.domain.value}
