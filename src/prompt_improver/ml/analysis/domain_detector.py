"""Domain Detection and Classification System

This module provides intelligent domain detection for prompts to enable
domain-specific feature extraction and optimization strategies.
"""
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import hashlib
import json
try:
    import spacy
    from spacy.lang.en import english
    from spacy.language import Language
    from spacy.matcher import Matcher, PhraseMatcher
    from spacy.tokens import Doc, Span, Token
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    english = None
    Language = None
    PhraseMatcher = None
    Matcher = None
    Doc = None
    Span = None
    Token = None
    SPACY_AVAILABLE = False

# Import unified cache facade for ML-specific caching
from prompt_improver.services.cache.cache_facade import CacheFacade

class PromptDomain(Enum):
    """Enumeration of prompt domains for classification."""
    SOFTWARE_DEVELOPMENT = 'software_development'
    DATA_SCIENCE = 'data_science'
    AI_ML = 'ai_ml'
    WEB_DEVELOPMENT = 'web_development'
    SYSTEM_ADMIN = 'system_admin'
    API_DOCUMENTATION = 'api_documentation'
    CREATIVE_WRITING = 'creative_writing'
    CONTENT_CREATION = 'content_creation'
    MARKETING = 'marketing'
    STORYTELLING = 'storytelling'
    RESEARCH = 'research'
    EDUCATION = 'education'
    ACADEMIC_WRITING = 'academic_writing'
    SCIENTIFIC = 'scientific'
    BUSINESS_ANALYSIS = 'business_analysis'
    PROJECT_MANAGEMENT = 'project_management'
    CUSTOMER_SERVICE = 'customer_service'
    SALES = 'sales'
    MEDICAL = 'medical'
    LEGAL = 'legal'
    HEALTHCARE = 'healthcare'
    CONVERSATIONAL = 'conversational'
    INSTRUCTIONAL = 'instructional'
    ANALYTICAL = 'analytical'
    GENERAL = 'general'

@dataclass
class DomainKeywords:
    """Container for domain-specific keyword sets."""
    software_development: set[str] = field(default_factory=lambda: {'python', 'javascript', 'java', 'c++', 'code', 'function', 'class', 'method', 'variable', 'algorithm', 'debug', 'test', 'unit test', 'framework', 'library', 'git', 'github', 'repository', 'commit', 'pull request', 'merge', 'branch', 'api', 'rest', 'graphql', 'database', 'sql', 'nosql', 'microservice', 'docker', 'kubernetes', 'deployment', 'ci/cd', 'devops', 'agile', 'scrum', 'flask', 'django', 'fastapi', 'clean code', 'documentation', 'error handling', 'input validation', 'binary search'})
    data_science: set[str] = field(default_factory=lambda: {'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'scikit-learn', 'data analysis', 'visualization', 'statistics', 'hypothesis', 'correlation', 'regression', 'classification', 'clustering', 'feature engineering', 'data cleaning', 'etl', 'pipeline', 'jupyter', 'notebook', 'dataset', 'csv', 'json', 'dataframe', 'series', 'plot', 'chart', 'graph'})
    ai_ml: set[str] = field(default_factory=lambda: {'machine learning', 'deep learning', 'neural network', 'cnn', 'rnn', 'lstm', 'transformer', 'bert', 'gpt', 'llm', 'nlp', 'computer vision', 'cv', 'tensorflow', 'pytorch', 'keras', 'model', 'training', 'inference', 'supervised', 'unsupervised', 'reinforcement learning', 'gradient', 'backpropagation', 'epoch', 'batch', 'learning rate', 'loss function', 'accuracy', 'precision', 'recall', 'f1-score', 'auc', 'roc'})
    web_development: set[str] = field(default_factory=lambda: {'html', 'css', 'react', 'vue', 'angular', 'node.js', 'express', 'frontend', 'backend', 'fullstack', 'responsive', 'bootstrap', 'webpack', 'babel', 'npm', 'yarn', 'dom', 'ajax', 'fetch', 'authentication', 'authorization', 'session', 'cookie', 'cors', 'seo', 'accessibility', 'performance', 'optimization', 'cdn'})
    creative_writing: set[str] = field(default_factory=lambda: {'story', 'narrative', 'character', 'plot', 'setting', 'dialogue', 'theme', 'conflict', 'resolution', 'protagonist', 'antagonist', 'fiction', 'non-fiction', 'novel', 'short story', 'poetry', 'prose', 'creative', 'imagination', 'descriptive', 'metaphor', 'simile', 'tone', 'voice', 'style', 'genre', 'literary', 'publish'})
    content_creation: set[str] = field(default_factory=lambda: {'blog', 'article', 'post', 'content', 'social media', 'instagram', 'twitter', 'facebook', 'linkedin', 'youtube', 'video', 'podcast', 'engagement', 'audience', 'viral', 'trending', 'hashtag', 'caption', 'copywriting', 'headline', 'call to action', 'conversion', 'brand', 'voice', 'messaging', 'campaign', 'influencer', 'reach'})
    research: set[str] = field(default_factory=lambda: {'research', 'study', 'analysis', 'methodology', 'hypothesis', 'experiment', 'data collection', 'survey', 'interview', 'observation', 'literature review', 'citation', 'reference', 'peer review', 'journal', 'publication', 'academic', 'scholarly', 'thesis', 'dissertation', 'quantitative', 'qualitative', 'statistical significance', 'p-value', 'confidence interval'})
    education: set[str] = field(default_factory=lambda: {'lesson', 'curriculum', 'learning', 'teaching', 'student', 'teacher', 'instructor', 'course', 'assignment', 'homework', 'quiz', 'exam', 'grade', 'assessment', 'objective', 'outcome', 'skill', 'knowledge', 'understanding', 'concept', 'explanation', 'example', 'practice', 'exercise', 'pedagogy', 'educational', 'academic', 'classroom'})
    business_analysis: set[str] = field(default_factory=lambda: {'business', 'analysis', 'strategy', 'market', 'competitor', 'swot', 'roi', 'kpi', 'metric', 'performance', 'revenue', 'profit', 'cost', 'budget', 'forecast', 'trend', 'opportunity', 'risk', 'stakeholder', 'requirement', 'process', 'workflow', 'efficiency', 'optimization', 'growth', 'scalability', 'sustainability', 'compliance'})
    medical: set[str] = field(default_factory=lambda: {'patient', 'diagnosis', 'treatment', 'symptom', 'disease', 'condition', 'medication', 'therapy', 'clinical', 'medical', 'health', 'healthcare', 'doctor', 'physician', 'nurse', 'hospital', 'clinic', 'surgery', 'procedure', 'examination', 'test', 'result', 'prognosis', 'recovery', 'prevention', 'risk factor', 'epidemiology', 'pathology'})
    legal: set[str] = field(default_factory=lambda: {'law', 'legal', 'court', 'judge', 'lawyer', 'attorney', 'contract', 'agreement', 'clause', 'liability', 'compliance', 'regulation', 'statute', 'precedent', 'case law', 'jurisdiction', 'litigation', 'settlement', 'plaintiff', 'defendant', 'evidence', 'testimony', 'verdict', 'appeal', 'motion', 'brief', 'due process', 'rights'})

@dataclass
class DomainClassificationResult:
    """Result of domain classification analysis."""
    primary_domain: PromptDomain
    confidence: float
    secondary_domains: list[tuple[PromptDomain, float]] = field(default_factory=list)
    domain_keywords_found: dict[str, list[str]] = field(default_factory=dict)
    technical_complexity: float = 0.0
    domain_specificity: float = 0.0
    hybrid_domain: bool = False

class DomainDetector:
    """Intelligent domain detection system for prompts."""

    def __init__(self, use_spacy: bool=True, cache_facade: CacheFacade | None=None):
        """Initialize the domain detector."""
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.domain_keywords = DomainKeywords()
        self.nlp = None
        self.phrase_matcher = None
        self.pattern_matcher = None
        # Initialize ML-specific cache with 4-hour TTL for stable domain classification
        self._cache = cache_facade or CacheFacade(
            l1_max_size=1000,  # Preserve original cache size
            l2_default_ttl=14400,  # 4 hours for domain classification results
            enable_l2=True,  # Use L2 for persistence across restarts
        )
        if self.use_spacy:
            self._initialize_spacy_components()
        self._compile_domain_patterns()

    def _initialize_spacy_components(self):
        """Initialize spaCy components for domain detection."""
        try:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                self.nlp = english()
                self.logger.info('Using blank english pipeline for domain detection')
            if not self.nlp.has_pipe('domain_detector'):
                self.nlp.add_pipe('domain_detector', component=self._create_domain_component(), last=True)
        except Exception as e:
            self.logger.warning('Failed to initialize spaCy components: %s', e)
            self.use_spacy = False

    def _create_domain_component(self):
        """Create custom spaCy component for domain detection."""
        if not SPACY_AVAILABLE:
            return None

        @Language.component('domain_detector')
        def domain_detector_component(doc):
            """Custom spaCy component for domain-specific processing."""
            domain_entities = []
            for token in doc:
                if token.text.lower() in self._get_all_technical_terms():
                    domain_entities.append(Span(doc, token.i, token.i + 1, label='TECH_TERM'))
            doc._.set('domain_entities', domain_entities)
            doc._.set('technical_density', len(domain_entities) / len(doc))
            return doc
        if not Doc.has_extension('domain_entities'):
            Doc.set_extension('domain_entities', default=[])
        if not Doc.has_extension('technical_density'):
            Doc.set_extension('technical_density', default=0.0)
        return domain_detector_component

    def _compile_domain_patterns(self):
        """Compile regex patterns for domain detection."""
        self.domain_patterns = {PromptDomain.SOFTWARE_DEVELOPMENT: ['\\b(?:def|function|class|import|from|if|else|for|while)\\b', '\\b(?:\\.py|\\.js|\\.java|\\.cpp|\\.html|\\.css)\\b', '\\b(?:github\\.com|stackoverflow\\.com)\\b', '\\b(?:version control|pull request|code review)\\b'], PromptDomain.DATA_SCIENCE: ['\\bpandas\\.(?:DataFrame|Series|read_csv)\\b', '\\bnumpy\\.(?:array|mean|std)\\b', '\\bmatplotlib\\.pyplot\\b', '\\b(?:correlation|regression|hypothesis testing)\\b'], PromptDomain.AI_ML: ['\\b(?:neural network|deep learning|machine learning)\\b', '\\b(?:tensorflow|pytorch|scikit-learn)\\b', '\\b(?:training|validation|test) set\\b', '\\b(?:accuracy|precision|recall|f1[-_]score)\\b'], PromptDomain.WEB_DEVELOPMENT: ['\\b(?:react|vue|angular|node\\.js)\\b', '\\b(?:frontend|backend|fullstack)\\b', '\\b(?:html|css|javascript|typescript)\\b', '\\b(?:responsive design|api endpoint)\\b'], PromptDomain.CREATIVE_WRITING: ['\\b(?:character development|plot structure)\\b', '\\b(?:narrative|storytelling|creative writing)\\b', '\\b(?:protagonist|antagonist|conflict|resolution)\\b', '\\b(?:fiction|non-fiction|poetry|prose)\\b'], PromptDomain.ACADEMIC_WRITING: ['\\b(?:thesis|dissertation|research paper)\\b', '\\b(?:literature review|methodology|hypothesis)\\b', '\\b(?:citation|reference|bibliography)\\b', '\\b(?:peer review|journal|publication)\\b'], PromptDomain.BUSINESS_ANALYSIS: ['\\b(?:market analysis|competitive analysis)\\b', '\\b(?:roi|kpi|revenue|profit)\\b', '\\b(?:stakeholder|requirement|process)\\b', '\\b(?:strategy|optimization|efficiency)\\b'], PromptDomain.medical: ['\\b(?:patient|diagnosis|treatment|symptom)\\b', '\\b(?:clinical trial|medical procedure)\\b', '\\b(?:healthcare|medical record|prescription)\\b', '\\b(?:pathology|epidemiology|prognosis)\\b'], PromptDomain.legal: ['\\b(?:contract|agreement|clause|liability)\\b', '\\b(?:litigation|settlement|jurisdiction)\\b', '\\b(?:plaintiff|defendant|evidence|testimony)\\b', '\\b(?:regulation|compliance|statute)\\b']}
        self.compiled_patterns = {}
        for domain, patterns in self.domain_patterns.items():
            self.compiled_patterns[domain] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def _get_all_technical_terms(self) -> set[str]:
        """Get all technical terms across domains."""
        all_terms = set()
        for attr_name in dir(self.domain_keywords):
            if not attr_name.startswith('_'):
                terms = getattr(self.domain_keywords, attr_name)
                if isinstance(terms, set):
                    all_terms.update(terms)
        return all_terms

    def _generate_ml_cache_key(self, text: str) -> str:
        """Generate ML-specific cache key for domain detection.
        
        Cache key pattern: ml:domain:{text_hash}:{keywords_hash}
        """
        # Hash text content for cache key
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
        
        # Include domain keywords configuration in key (for when keywords change)
        keywords_config = str(len(self.domain_keywords.software_development)) + \
                         str(len(self.domain_keywords.data_science)) + \
                         str(len(self.domain_keywords.ai_ml))
        keywords_hash = hashlib.md5(keywords_config.encode()).hexdigest()[:8]
        
        return f"ml:domain:{text_hash}:{keywords_hash}"
    
    async def detect_domain_async(self, text: str) -> DomainClassificationResult:
        """Async domain detection using unified cache infrastructure."""
        if not text or not text.strip():
            return DomainClassificationResult(primary_domain=PromptDomain.GENERAL, confidence=0.0)
        
        cache_key = self._generate_ml_cache_key(text)
        
        # Try to get from unified cache first
        cached_result = await self._cache.get(cache_key)
        if cached_result is not None:
            self.logger.debug(f"Domain detection cache hit for key: {cache_key}")
            return cached_result
        
        # Cache miss - perform domain detection and cache result
        result = self._perform_domain_detection(text)
        
        # Cache with ML-specific TTL (4 hours)
        await self._cache.set(
            key=cache_key,
            value=result,
            l2_ttl=14400,  # 4 hours
            l1_ttl=3600    # 1 hour for L1
        )
        
        self.logger.debug(f"Domain detection cached with key: {cache_key}")
        return result
    
    def detect_domain(self, text: str) -> DomainClassificationResult:
        """Detect the primary domain of a prompt text using unified cache infrastructure.

        Args:
            text: Input prompt text

        Returns:
            DomainClassificationResult with classification details
        """
        try:
            loop = asyncio.get_running_loop()
            return asyncio.run_coroutine_threadsafe(self.detect_domain_async(text), loop).result()
        except RuntimeError:
            return asyncio.run(self.detect_domain_async(text))
    
    def _perform_domain_detection(self, text: str) -> DomainClassificationResult:
        """Core domain detection logic extracted from original cached method."""
        text_lower = text.lower()
        domain_scores = {}
        keywords_found = {}
        for attr_name in dir(self.domain_keywords):
            if attr_name.startswith('_'):
                continue
            domain_keywords = getattr(self.domain_keywords, attr_name)
            if not isinstance(domain_keywords, set):
                continue
            found_keywords = [kw for kw in domain_keywords if kw in text_lower]
            keywords_found[attr_name] = found_keywords
            if found_keywords:
                keyword_score = len(found_keywords) / len(domain_keywords)
                specificity_boost = sum(len(kw.split()) for kw in found_keywords) / len(found_keywords)
                domain_scores[attr_name] = keyword_score * (1 + specificity_boost * 0.1)
        pattern_scores = {}
        for domain, patterns in self.compiled_patterns.items():
            pattern_matches = sum(1 for pattern in patterns if pattern.search(text))
            if pattern_matches > 0:
                pattern_scores[domain] = pattern_matches / len(patterns)
        final_scores = {}
        domain_mapping = {'software_development': PromptDomain.SOFTWARE_DEVELOPMENT, 'data_science': PromptDomain.DATA_SCIENCE, 'ai_ml': PromptDomain.AI_ML, 'web_development': PromptDomain.WEB_DEVELOPMENT, 'creative_writing': PromptDomain.CREATIVE_WRITING, 'content_creation': PromptDomain.CONTENT_CREATION, 'research': PromptDomain.RESEARCH, 'education': PromptDomain.EDUCATION, 'academic_writing': PromptDomain.ACADEMIC_WRITING, 'business_analysis': PromptDomain.BUSINESS_ANALYSIS, 'medical': PromptDomain.MEDICAL, 'legal': PromptDomain.LEGAL}
        for domain_name, score in domain_scores.items():
            if domain_name in domain_mapping:
                domain_enum = domain_mapping[domain_name]
                final_scores[domain_enum] = score
                if domain_enum in pattern_scores:
                    final_scores[domain_enum] = (score + pattern_scores[domain_enum]) / 2
        for domain_enum, score in pattern_scores.items():
            if domain_enum not in final_scores:
                final_scores[domain_enum] = score
        if not final_scores:
            primary_domain = PromptDomain.GENERAL
            confidence = 0.0
            secondary_domains = []
        else:
            sorted_domains = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            primary_domain = sorted_domains[0][0]
            confidence = min(sorted_domains[0][1], 1.0)
            secondary_domains = [(domain, score) for domain, score in sorted_domains[1:4]]
        technical_complexity = self._calculate_technical_complexity(text, keywords_found)
        domain_specificity = self._calculate_domain_specificity(final_scores)
        hybrid_domain = len([s for s in final_scores.values() if s > 0.2]) > 1
        return DomainClassificationResult(primary_domain=primary_domain, confidence=confidence, secondary_domains=secondary_domains, domain_keywords_found=keywords_found, technical_complexity=technical_complexity, domain_specificity=domain_specificity, hybrid_domain=hybrid_domain)

    def _calculate_technical_complexity(self, text: str, keywords_found: dict) -> float:
        """Calculate technical complexity score."""
        technical_domains = ['software_development', 'data_science', 'ai_ml', 'web_development']
        total_technical_keywords = sum(len(keywords_found.get(domain, [])) for domain in technical_domains)
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        return min(total_technical_keywords / word_count, 1.0)

    def _calculate_domain_specificity(self, domain_scores: dict) -> float:
        """Calculate how domain-specific the text is."""
        if not domain_scores:
            return 0.0
        max_score = max(domain_scores.values())
        score_variance = sum((score - max_score) ** 2 for score in domain_scores.values())
        return max_score * (1 - score_variance / len(domain_scores))

    async def detect_domain_async_old(self, text: str) -> DomainClassificationResult:
        """Legacy async version of domain detection."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.detect_domain, text)

    def get_domain_keywords(self, domain: PromptDomain) -> set[str]:
        """Get keywords for a specific domain."""
        domain_attr_map = {PromptDomain.SOFTWARE_DEVELOPMENT: 'software_development', PromptDomain.DATA_SCIENCE: 'data_science', PromptDomain.AI_ML: 'ai_ml', PromptDomain.WEB_DEVELOPMENT: 'web_development', PromptDomain.CREATIVE_WRITING: 'creative_writing', PromptDomain.CONTENT_CREATION: 'content_creation', PromptDomain.RESEARCH: 'research', PromptDomain.EDUCATION: 'education', PromptDomain.ACADEMIC_WRITING: 'academic_writing', PromptDomain.BUSINESS_ANALYSIS: 'business_analysis', PromptDomain.MEDICAL: 'medical', PromptDomain.LEGAL: 'legal'}
        attr_name = domain_attr_map.get(domain)
        if attr_name and hasattr(self.domain_keywords, attr_name):
            return getattr(self.domain_keywords, attr_name)
        return set()

    def add_domain_keywords(self, domain: PromptDomain, keywords: set[str]):
        """Add new keywords to a domain."""
        existing_keywords = self.get_domain_keywords(domain)
        existing_keywords.update(keywords)
        self.logger.info('Added {len(keywords)} keywords to %s', domain.value)

    def is_technical_domain(self, domain: PromptDomain) -> bool:
        """Check if a domain is considered technical."""
        technical_domains = {PromptDomain.SOFTWARE_DEVELOPMENT, PromptDomain.DATA_SCIENCE, PromptDomain.AI_ML, PromptDomain.WEB_DEVELOPMENT, PromptDomain.SYSTEM_ADMIN, PromptDomain.API_DOCUMENTATION}
        return domain in technical_domains
