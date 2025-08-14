"""Comprehensive linguistic analysis engine for prompt quality assessment.

This module provides the main LinguisticAnalyzer class that coordinates
various linguistic analysis tasks including NER, dependency parsing,
complexity metrics, and readability assessment.
"""
import asyncio
from dataclasses import dataclass, field
from functools import lru_cache
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import nltk
from prompt_improver.core.config.textstat import get_textstat_wrapper
from ..models.model_manager import ModelManager, get_lightweight_ner_pipeline, get_memory_optimized_config, get_ultra_lightweight_ner_pipeline, model_config
from ..learning.features.english_nltk_manager import get_english_nltk_manager
from .dependency_parser import DependencyParser
from .ner_extractor import NERExtractor
try:
    from transformers import auto_model_for_token_classification, auto_tokenizer, pipeline, set_seed
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pipeline = None
    auto_tokenizer = None
    auto_model_for_token_classification = None
    set_seed = None
    TRANSFORMERS_AVAILABLE = False

@dataclass
class LinguisticConfig:
    """Configuration for linguistic analysis."""
    enable_ner: bool = True
    enable_dependency_parsing: bool = True
    enable_readability: bool = True
    enable_complexity_metrics: bool = True
    enable_prompt_segmentation: bool = True
    ner_model: str = 'dbmdz/bert-large-cased-finetuned-conll03-english'
    use_transformers_ner: bool = True
    use_lightweight_models: bool = False
    use_ultra_lightweight_models: bool = False
    enable_model_quantization: bool = True
    quantization_bits: int = 8
    enable_4bit_quantization: bool = False
    max_memory_threshold_mb: int = 200
    force_cpu_only: bool = False
    auto_download_nltk: bool = True
    nltk_fallback_enabled: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    max_workers: int = 4
    timeout_seconds: int = 30
    technical_keywords: set[str] = field(default_factory=lambda: {'llm', 'ai', 'ml', 'nlp', 'gpt', 'bert', 'transformer', 'neural', 'model', 'prompt', 'token', 'embedding', 'fine-tune', 'training', 'inference', 'api', 'json', 'xml', 'python', 'javascript', 'sql', 'database'})

@dataclass
class LinguisticFeatures:
    """Container for linguistic analysis results."""
    entities: list[dict[str, Any]] = field(default_factory=list)
    entity_types: set[str] = field(default_factory=set)
    entity_density: float = 0.0
    technical_terms: list[str] = field(default_factory=list)
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    syntactic_complexity: float = 0.0
    sentence_structure_quality: float = 0.0
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    smog_index: float = 0.0
    coleman_liau_index: float = 0.0
    readability_score: float = 0.0
    lexical_diversity: float = 0.0
    avg_sentence_length: float = 0.0
    avg_word_length: float = 0.0
    syllable_count: int = 0
    has_clear_instructions: bool = False
    has_examples: bool = False
    has_context: bool = False
    instruction_clarity_score: float = 0.0
    overall_linguistic_quality: float = 0.0
    confidence: float = 0.0

class LinguisticAnalyzer:
    """Advanced linguistic analysis for prompt quality assessment."""

    def __init__(self, config: LinguisticConfig | None=None):
        """Initialize the linguistic analyzer with configuration."""
        self.config = config or LinguisticConfig()
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        if TRANSFORMERS_AVAILABLE and set_seed:
            set_seed(42)
        import numpy as np
        np.random.seed(42)
        self.ner_extractor = None
        self.dependency_parser = None
        self.transformers_pipeline = None
        self.model_manager = None
        self.nltk_manager = None
        # Remove ThreadPoolExecutor - use asyncio.to_thread instead
        self._setup_resource_managers()
        self._initialize_components()

    def _setup_resource_managers(self):
        """Setup NLTK and model resource managers."""
        try:
            self.nltk_manager = get_english_nltk_manager()
            # English NLTK manager handles setup automatically
            if not self.nltk_manager._resources_checked:
                self.logger.warning('NLTK resources not fully available, some features may be limited')
            if self.config.use_transformers_ner:
                if self.config.use_ultra_lightweight_models or self.config.use_lightweight_models:
                    self.model_manager = None
                    mode = 'ultra-lightweight' if self.config.use_ultra_lightweight_models else 'lightweight'
                    self.logger.info('Using %s model configuration', mode)
                else:
                    if self.config.enable_4bit_quantization:
                        model_config = get_memory_optimized_config(self.config.max_memory_threshold_mb)
                        self.logger.info('Using memory-optimized config for %sMB target', self.config.max_memory_threshold_mb)
                    else:
                        model_config = model_config(model_name=self.config.ner_model, task='ner', use_quantization=self.config.enable_model_quantization, quantization_bits=self.config.quantization_bits, max_memory_threshold_mb=self.config.max_memory_threshold_mb, device_map='cpu' if self.config.force_cpu_only else 'auto', auto_select_model=True)
                    self.model_manager = ModelManager(model_config)
                    self.logger.info('Initialized optimized model manager')
        except Exception as e:
            self.logger.error('Failed to setup resource managers: %s', e)
            self.model_manager = None
            self.nltk_manager = None

    def _initialize_components(self):
        """Initialize analysis components based on configuration."""
        try:
            if self.config.enable_ner:
                self.ner_extractor = NERExtractor(use_transformers=self.config.use_transformers_ner, model_name=self.config.ner_model)
                if self.config.use_transformers_ner:
                    self._initialize_transformers_pipeline()
            if self.config.enable_dependency_parsing:
                self.dependency_parser = DependencyParser()
        except Exception as e:
            self.logger.error('Failed to initialize components: %s', e)

    def _initialize_transformers_pipeline(self):
        """Initialize transformers pipeline with memory optimization."""
        try:
            if self.config.use_ultra_lightweight_models:
                self.transformers_pipeline = get_ultra_lightweight_ner_pipeline()
                if self.transformers_pipeline:
                    self.logger.info('Initialized ultra-lightweight transformers NER pipeline (<30MB)')
                else:
                    self.logger.warning('Failed to initialize ultra-lightweight pipeline')
                    self.config.use_transformers_ner = False
            elif self.config.use_lightweight_models:
                self.transformers_pipeline = get_lightweight_ner_pipeline()
                if self.transformers_pipeline:
                    self.logger.info('Initialized lightweight transformers NER pipeline')
                else:
                    self.logger.warning('Failed to initialize lightweight pipeline')
                    self.config.use_transformers_ner = False
            elif self.model_manager:
                self.transformers_pipeline = self.model_manager.get_pipeline()
                if self.transformers_pipeline:
                    memory_usage = self.model_manager.get_memory_usage()
                    self.logger.info('Initialized optimized transformers NER pipeline (Memory: %sMB)', format(memory_usage, '.1f'))
                else:
                    self.logger.warning('Failed to initialize optimized pipeline')
                    self.config.use_transformers_ner = False
            else:
                try:
                    self.transformers_pipeline = pipeline('ner', model=self.config.ner_model, tokenizer=self.config.ner_model, aggregation_strategy='simple', device_map='cpu')
                    self.logger.info('Initialized basic transformers NER pipeline')
                except Exception as e:
                    self.logger.warning('Failed to initialize basic transformers pipeline: %s', e)
                    self.config.use_transformers_ner = False
        except Exception as e:
            self.logger.error('Failed to initialize transformers pipeline: %s', e)
            self.config.use_transformers_ner = False

    async def analyze_async(self, text: str) -> LinguisticFeatures:
        """Perform comprehensive linguistic analysis asynchronously.

        Args:
            text: The text to analyze

        Returns:
            LinguisticFeatures object containing all analysis results
        """
        try:
            tasks = []
            if self.config.enable_ner and self.ner_extractor:
                tasks.append(self._analyze_entities(text))
            if self.config.enable_dependency_parsing and self.dependency_parser:
                tasks.append(self._analyze_dependencies(text))
            if self.config.enable_readability:
                tasks.append(self._analyze_readability(text))
            if self.config.enable_complexity_metrics:
                tasks.append(self._analyze_complexity(text))
            if self.config.enable_prompt_segmentation:
                tasks.append(self._analyze_prompt_structure(text))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            features = LinguisticFeatures()
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error('Analysis task {i} failed: %s', result)
                    continue
                if isinstance(result, dict):
                    for key, value in result.items():
                        if hasattr(features, key):
                            setattr(features, key, value)
            features.overall_linguistic_quality = self._calculate_overall_quality(features)
            features.confidence = self._calculate_confidence(features)
            return features
        except Exception as e:
            self.logger.error('Linguistic analysis failed: %s', e)
            return LinguisticFeatures()

    def analyze(self, text: str) -> LinguisticFeatures:
        """Perform comprehensive linguistic analysis synchronously.

        Args:
            text: The text to analyze

        Returns:
            LinguisticFeatures object containing all analysis results
        """
        try:
            loop = asyncio.get_running_loop()
            return asyncio.run_coroutine_threadsafe(self.analyze_async(text), loop).result()
        except RuntimeError:
            return asyncio.run(self.analyze_async(text))

    async def _analyze_entities(self, text: str) -> dict[str, Any]:
        """Analyze named entities and technical terms."""
        try:
            entities = []
            entity_types = set()
            technical_terms = []
            if self.transformers_pipeline:
                try:
                    ner_results = self.transformers_pipeline(text)
                    for entity in ner_results:
                        entities.append({'text': entity['word'], 'label': entity['entity_group'], 'confidence': entity['score'], 'start': entity.get('start', 0), 'end': entity.get('end', 0)})
                        entity_types.add(entity['entity_group'])
                except Exception as e:
                    self.logger.warning('Transformers NER failed: %s', e)
            if not entities and self.ner_extractor:
                entities = await asyncio.to_thread(self.ner_extractor.extract_entities, text)
                entity_types = {e['label'] for e in entities}
            technical_terms = self._extract_technical_terms(text)
            words = len(text.split())
            entity_density = len(entities) / max(words, 1)
            return {'entities': entities, 'entity_types': entity_types, 'entity_density': entity_density, 'technical_terms': technical_terms}
        except Exception as e:
            self.logger.error('Entity analysis failed: %s', e)
            return {}

    async def _analyze_dependencies(self, text: str) -> dict[str, Any]:
        """Analyze dependency parsing and syntactic structure."""
        try:
            if not self.dependency_parser:
                return {}
            dependencies = await asyncio.to_thread(self.dependency_parser.parse, text)
            complexity = self._calculate_syntactic_complexity(dependencies)
            structure_quality = self._assess_sentence_structure(dependencies)
            return {'dependencies': dependencies, 'syntactic_complexity': complexity, 'sentence_structure_quality': structure_quality}
        except Exception as e:
            self.logger.error('Dependency analysis failed: %s', e)
            return {}

    async def _analyze_readability(self, text: str) -> dict[str, Any]:
        """Analyze readability metrics using TextStat wrapper."""
        try:
            # Use optimized wrapper with warning suppression and caching
            textstat_wrapper = get_textstat_wrapper()
            
            # Use asyncio.to_thread for thread safety
            analysis_result = await asyncio.to_thread(
                textstat_wrapper.comprehensive_analysis, 
                text
            )
            
            # Extract specific metrics and calculate readability score
            flesch_ease = analysis_result.get('flesch_reading_ease', 50.0)
            flesch_kincaid = analysis_result.get('flesch_kincaid_grade', 8.0)
            gunning_fog = analysis_result.get('gunning_fog', 8.0)
            
            readability_score = self._calculate_readability_score(flesch_ease, flesch_kincaid, gunning_fog)
            
            return {
                'flesch_reading_ease': flesch_ease,
                'flesch_kincaid_grade': flesch_kincaid,
                'gunning_fog': gunning_fog,
                'smog_index': analysis_result.get('smog_index', 8.0),
                'coleman_liau_index': analysis_result.get('coleman_liau_index', 8.0),
                'readability_score': readability_score
            }
        except Exception as e:
            self.logger.error('Readability analysis failed: %s', e)
            return {}

    async def _analyze_complexity(self, text: str) -> dict[str, Any]:
        """Analyze text complexity metrics."""
        try:
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            lexical_diversity = len(set(words)) / max(len(words), 1)
            avg_sentence_length = len(words) / max(len(sentences), 1)
            avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
            # Use TextStat wrapper for syllable counting
            textstat_wrapper = get_textstat_wrapper()
            syllable_count = textstat_wrapper.syllable_count(text)
            return {'lexical_diversity': lexical_diversity, 'avg_sentence_length': avg_sentence_length, 'avg_word_length': avg_word_length, 'syllable_count': syllable_count}
        except Exception as e:
            self.logger.error('Complexity analysis failed: %s', e)
            return {}

    async def _analyze_prompt_structure(self, text: str) -> dict[str, Any]:
        """Analyze prompt structure and components."""
        try:
            instruction_patterns = ['\\b(please|write|create|generate|explain|describe|analyze|summarize)\\b', '\\b(you should|you need to|your task is|your goal is)\\b', '\\b(step by step|follow these|instructions|guidelines)\\b']
            example_patterns = ['\\b(for example|e\\.g\\.|such as|instance|example)\\b', "\\b(here\\'s an example|consider this|like this)\\b", '```.*?```', '\\".*?\\"']
            context_patterns = ['\\b(context|background|scenario|situation)\\b', '\\b(given that|assuming|in this case)\\b', '\\b(remember that|keep in mind|note that)\\b']
            has_clear_instructions = any(re.search(pattern, text, re.IGNORECASE) for pattern in instruction_patterns)
            has_examples = any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in example_patterns)
            has_context = any(re.search(pattern, text, re.IGNORECASE) for pattern in context_patterns)
            clarity_score = self._calculate_instruction_clarity(text)
            return {'has_clear_instructions': has_clear_instructions, 'has_examples': has_examples, 'has_context': has_context, 'instruction_clarity_score': clarity_score}
        except Exception as e:
            self.logger.error('Prompt structure analysis failed: %s', e)
            return {}

    def _extract_technical_terms(self, text: str) -> list[str]:
        """Extract technical terms from text."""
        words = set(re.findall('\\b\\w+\\b', text.lower()))
        return list(words.intersection(self.config.technical_keywords))

    def _calculate_syntactic_complexity(self, dependencies: list[dict]) -> float:
        """Calculate syntactic complexity score."""
        if not dependencies:
            return 0.0
        dep_types = {dep.get('relation', '') for dep in dependencies}
        complexity_score = len(dep_types) / max(len(dependencies), 1)
        nested_count = sum(1 for dep in dependencies if dep.get('depth', 0) > 2)
        complexity_score += nested_count / max(len(dependencies), 1)
        return min(complexity_score, 1.0)

    def _assess_sentence_structure(self, dependencies: list[dict]) -> float:
        """Assess sentence structure quality."""
        if not dependencies:
            return 0.0
        has_subject = any(dep.get('relation') == 'nsubj' for dep in dependencies)
        has_predicate = any(dep.get('relation') == 'ROOT' for dep in dependencies)
        has_object = any(dep.get('relation') == 'dobj' for dep in dependencies)
        structure_score = sum([has_subject, has_predicate, has_object]) / 3.0
        return structure_score

    def _calculate_readability_score(self, flesch_ease: float, flesch_kincaid: float, gunning_fog: float) -> float:
        """Calculate composite readability score."""
        flesch_normalized = max(0, min(100, flesch_ease)) / 100.0
        kincaid_normalized = max(0, 1 - flesch_kincaid / 20.0)
        gunning_normalized = max(0, 1 - gunning_fog / 20.0)
        return flesch_normalized * 0.4 + kincaid_normalized * 0.3 + gunning_normalized * 0.3

    def _calculate_instruction_clarity(self, text: str) -> float:
        """Calculate instruction clarity score."""
        imperative_verbs = ['write', 'create', 'generate', 'explain', 'describe', 'analyze', 'summarize', 'list', 'identify', 'compare', 'evaluate', 'discuss']
        imperative_count = sum(len(re.findall(f'\\b{verb}\\b', text, re.IGNORECASE)) for verb in imperative_verbs)
        structure_indicators = ['first', 'second', 'third', 'next', 'then', 'finally', 'step 1', 'step 2', 'part a', 'part b']
        structure_count = sum(len(re.findall(f'\\b{indicator}\\b', text, re.IGNORECASE)) for indicator in structure_indicators)
        words = len(text.split())
        clarity_score = (imperative_count + structure_count) / max(words / 10, 1)
        return min(clarity_score, 1.0)

    def _calculate_overall_quality(self, features: LinguisticFeatures) -> float:
        """Calculate overall linguistic quality score with improved normalization."""
        readability_weight = 0.25
        structure_weight = 0.25
        clarity_weight = 0.3
        richness_weight = 0.2
        readability_component = min(features.readability_score, 1.0) * readability_weight
        structure_component = features.sentence_structure_quality * structure_weight
        clarity_component = features.instruction_clarity_score * clarity_weight
        entity_richness = min(features.entity_density * 3.0, 1.0)
        lexical_richness = features.lexical_diversity
        technical_richness = min(len(features.technical_terms) / 10.0, 1.0)
        richness_score = (entity_richness + lexical_richness + technical_richness) / 3.0
        richness_component = richness_score * richness_weight
        total_score = readability_component + structure_component + clarity_component + richness_component
        return min(max(total_score, 0.0), 1.0)

    def _calculate_confidence(self, features: LinguisticFeatures) -> float:
        """Calculate confidence in the analysis results."""
        analysis_count = 0
        if features.entities:
            analysis_count += 1
        if features.dependencies:
            analysis_count += 1
        if features.readability_score > 0:
            analysis_count += 1
        if features.lexical_diversity > 0:
            analysis_count += 1
        max_analyses = 4
        confidence = analysis_count / max_analyses
        return confidence

    @lru_cache(maxsize=1000)
    def analyze_cached(self, text: str) -> LinguisticFeatures:
        """Cached version of analyze method."""
        if self.config.enable_caching:
            return self.analyze(text)
        return self.analyze(text)

    def cleanup(self):
        """Explicitly cleanup resources."""
        try:
            if hasattr(self, 'model_manager') and self.model_manager:
                self.model_manager.cleanup()
            if hasattr(self, 'transformers_pipeline'):
                self.transformers_pipeline = None
            # No more executor to clean up
            self.logger.info('LinguisticAnalyzer resources cleaned up')
        except Exception as e:
            self.logger.error('Error during cleanup: %s', e)

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage information."""
        memory_info = {'total_mb': 0.0, 'model_mb': 0.0}
        try:
            if self.model_manager:
                memory_info['model_mb'] = self.model_manager.get_memory_usage()
            import psutil
            process = psutil.process()
            memory_info['total_mb'] = process.memory_info().rss / 1024 / 1024
        except Exception as e:
            self.logger.debug('Memory usage calculation failed: %s', e)
        return memory_info

    def __del__(self):
        """Cleanup resources."""
        try:
            self.cleanup()
        except:
            pass

def get_lightweight_config() -> LinguisticConfig:
    """Get configuration optimized for testing and resource-constrained environments.

    Returns:
        LinguisticConfig with lightweight settings
    """
    return LinguisticConfig(use_lightweight_models=True, enable_model_quantization=False, force_cpu_only=True, max_memory_threshold_mb=50, enable_ner=True, enable_dependency_parsing=False, enable_readability=True, enable_complexity_metrics=True, enable_prompt_segmentation=True, max_workers=2, timeout_seconds=10, cache_size=100)

def get_ultra_lightweight_config() -> LinguisticConfig:
    """Get configuration optimized for extreme memory constraints (<30MB).

    Uses 4-bit quantization and tiny models for minimal memory usage.

    Returns:
        LinguisticConfig with ultra-lightweight settings
    """
    return LinguisticConfig(use_ultra_lightweight_models=True, enable_model_quantization=True, enable_4bit_quantization=True, quantization_bits=4, force_cpu_only=True, max_memory_threshold_mb=30, enable_ner=True, enable_dependency_parsing=False, enable_readability=True, enable_complexity_metrics=True, enable_prompt_segmentation=True, max_workers=1, timeout_seconds=10, cache_size=50)

def get_production_config() -> LinguisticConfig:
    """Get configuration optimized for production use with memory efficiency.

    Returns:
        LinguisticConfig with production-optimized settings
    """
    return LinguisticConfig(use_lightweight_models=False, enable_model_quantization=True, enable_4bit_quantization=True, quantization_bits=8, force_cpu_only=False, max_memory_threshold_mb=100, enable_ner=True, enable_dependency_parsing=True, enable_readability=True, enable_complexity_metrics=True, enable_prompt_segmentation=True, auto_download_nltk=True, nltk_fallback_enabled=True, max_workers=4, timeout_seconds=30, cache_size=1000)

def get_memory_optimized_config(target_memory_mb: int) -> LinguisticConfig:
    """Get configuration optimized for a specific memory target.

    Args:
        target_memory_mb: Target memory usage in MB

    Returns:
        LinguisticConfig optimized for the target memory
    """
    if target_memory_mb < 30:
        return get_ultra_lightweight_config()
    if target_memory_mb < 50:
        config = get_lightweight_config()
        config.enable_4bit_quantization = True
        config.max_memory_threshold_mb = target_memory_mb
        return config
    config = get_production_config()
    config.max_memory_threshold_mb = target_memory_mb
    return config

def create_test_analyzer() -> LinguisticAnalyzer:
    """Create a LinguisticAnalyzer instance optimized for testing.

    Returns:
        LinguisticAnalyzer with lightweight configuration
    """
    config = get_lightweight_config()
    return LinguisticAnalyzer(config)
