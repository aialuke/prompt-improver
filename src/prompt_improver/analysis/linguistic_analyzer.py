"""
Comprehensive linguistic analysis engine for prompt quality assessment.

This module provides the main LinguisticAnalyzer class that coordinates
various linguistic analysis tasks including NER, dependency parsing,
complexity metrics, and readability assessment.
"""

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

import nltk
import textstat

# Handle transformers import gracefully
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, set_seed
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModelForTokenClassification = None
    set_seed = None
    TRANSFORMERS_AVAILABLE = False

from .ner_extractor import NERExtractor
from .dependency_parser import DependencyParser
from ..utils.nltk_manager import get_nltk_manager, setup_nltk_for_production
from ..utils.model_manager import (
    ModelManager, ModelConfig, get_lightweight_ner_pipeline,
    get_ultra_lightweight_ner_pipeline, get_memory_optimized_config
)


@dataclass
class LinguisticConfig:
    """Configuration for linguistic analysis."""
    
    # Analysis features to enable
    enable_ner: bool = True
    enable_dependency_parsing: bool = True
    enable_readability: bool = True
    enable_complexity_metrics: bool = True
    enable_prompt_segmentation: bool = True
    
    # Model configurations
    ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    use_transformers_ner: bool = True
    
    # Memory optimization settings
    use_lightweight_models: bool = False  # Use for testing/resource-constrained environments
    use_ultra_lightweight_models: bool = False  # Use for extreme memory constraints (<30MB)
    enable_model_quantization: bool = True
    quantization_bits: int = 8  # 4, 8, or 16
    enable_4bit_quantization: bool = False  # Enable aggressive 4-bit quantization
    max_memory_threshold_mb: int = 200
    force_cpu_only: bool = False
    
    # NLTK resource management
    auto_download_nltk: bool = True
    nltk_fallback_enabled: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Performance
    max_workers: int = 4
    timeout_seconds: int = 30
    
    # Domain-specific keywords for technical term recognition
    technical_keywords: Set[str] = field(default_factory=lambda: {
        "llm", "ai", "ml", "nlp", "gpt", "bert", "transformer", "neural", "model",
        "prompt", "token", "embedding", "fine-tune", "training", "inference",
        "api", "json", "xml", "python", "javascript", "sql", "database"
    })


@dataclass
class LinguisticFeatures:
    """Container for linguistic analysis results."""
    
    # Named entities
    entities: List[Dict[str, Any]] = field(default_factory=list)
    entity_types: Set[str] = field(default_factory=set)
    entity_density: float = 0.0
    technical_terms: List[str] = field(default_factory=list)
    
    # Dependency parsing results
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    syntactic_complexity: float = 0.0
    sentence_structure_quality: float = 0.0
    
    # Readability metrics
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    smog_index: float = 0.0
    coleman_liau_index: float = 0.0
    readability_score: float = 0.0
    
    # Text complexity
    lexical_diversity: float = 0.0
    avg_sentence_length: float = 0.0
    avg_word_length: float = 0.0
    syllable_count: int = 0
    
    # Prompt structure
    has_clear_instructions: bool = False
    has_examples: bool = False
    has_context: bool = False
    instruction_clarity_score: float = 0.0
    
    # Overall metrics
    overall_linguistic_quality: float = 0.0
    confidence: float = 0.0


class LinguisticAnalyzer:
    """Advanced linguistic analysis for prompt quality assessment."""

    def __init__(self, config: Optional[LinguisticConfig] = None):
        """Initialize the linguistic analyzer with configuration."""
        self.config = config or LinguisticConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Set random seed for deterministic behavior (especially important for testing)
        if TRANSFORMERS_AVAILABLE and set_seed:
            set_seed(42)  # Fixed seed for reproducibility
        
        # Set numpy seed for additional determinism
        import numpy as np
        np.random.seed(42)
        
        # Initialize components
        self.ner_extractor = None
        self.dependency_parser = None
        self.transformers_pipeline = None
        self.model_manager = None
        self.nltk_manager = None
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Initialize NLTK and model managers
        self._setup_resource_managers()
        
        # Initialize components based on config
        self._initialize_components()
    
    def _setup_resource_managers(self):
        """Setup NLTK and model resource managers."""
        try:
            # Initialize NLTK manager
            self.nltk_manager = get_nltk_manager()
            
            if self.config.auto_download_nltk:
                setup_success = self.nltk_manager.setup_for_production()
                if not setup_success:
                    self.logger.warning("NLTK setup incomplete, some features may be limited")
            
            # Initialize model manager if using transformers
            if self.config.use_transformers_ner:
                if self.config.use_ultra_lightweight_models or self.config.use_lightweight_models:
                    # Use lightweight configuration for testing
                    self.model_manager = None  # Will use lightweight pipeline function
                    mode = "ultra-lightweight" if self.config.use_ultra_lightweight_models else "lightweight"
                    self.logger.info(f"Using {mode} model configuration")
                else:
                    # Create optimized model configuration
                    if self.config.enable_4bit_quantization:
                        # Use memory-optimized configuration for target memory
                        model_config = get_memory_optimized_config(self.config.max_memory_threshold_mb)
                        self.logger.info(f"Using memory-optimized config for {self.config.max_memory_threshold_mb}MB target")
                    else:
                        # Standard configuration
                        model_config = ModelConfig(
                            model_name=self.config.ner_model,
                            task="ner",
                            use_quantization=self.config.enable_model_quantization,
                            quantization_bits=self.config.quantization_bits,
                            max_memory_threshold_mb=self.config.max_memory_threshold_mb,
                            device_map="cpu" if self.config.force_cpu_only else "auto",
                            auto_select_model=True  # Enable auto-selection
                        )
                    self.model_manager = ModelManager(model_config)
                    self.logger.info("Initialized optimized model manager")
                    
        except Exception as e:
            self.logger.error(f"Failed to setup resource managers: {e}")
            self.model_manager = None
            self.nltk_manager = None
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded (legacy method)."""
        # This method is now handled by NLTK manager, but kept for compatibility
        if self.nltk_manager:
            return self.nltk_manager.setup_for_production()
        else:
            # Fallback to old method
            required_data = [
                'punkt', 'stopwords', 'averaged_perceptron_tagger',
                'wordnet', 'vader_lexicon'
            ]
            
            for data_name in required_data:
                try:
                    nltk.data.find(f'tokenizers/{data_name}')
                except LookupError:
                    try:
                        nltk.download(data_name, quiet=True)
                        self.logger.info(f"Downloaded NLTK data: {data_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to download {data_name}: {e}")
    
    def _initialize_components(self):
        """Initialize analysis components based on configuration."""
        try:
            if self.config.enable_ner:
                self.ner_extractor = NERExtractor(
                    use_transformers=self.config.use_transformers_ner,
                    model_name=self.config.ner_model
                )
                
                # Initialize transformers pipeline using model manager
                if self.config.use_transformers_ner:
                    self._initialize_transformers_pipeline()
            
            if self.config.enable_dependency_parsing:
                self.dependency_parser = DependencyParser()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
    
    def _initialize_transformers_pipeline(self):
        """Initialize transformers pipeline with memory optimization."""
        try:
            if self.config.use_ultra_lightweight_models:
                # Use ultra-lightweight pipeline for extreme memory constraints
                self.transformers_pipeline = get_ultra_lightweight_ner_pipeline()
                if self.transformers_pipeline:
                    self.logger.info("Initialized ultra-lightweight transformers NER pipeline (<30MB)")
                else:
                    self.logger.warning("Failed to initialize ultra-lightweight pipeline")
                    self.config.use_transformers_ner = False
            elif self.config.use_lightweight_models:
                # Use lightweight pipeline for testing
                self.transformers_pipeline = get_lightweight_ner_pipeline()
                if self.transformers_pipeline:
                    self.logger.info("Initialized lightweight transformers NER pipeline")
                else:
                    self.logger.warning("Failed to initialize lightweight pipeline")
                    self.config.use_transformers_ner = False
            elif self.model_manager:
                # Use optimized model manager
                self.transformers_pipeline = self.model_manager.get_pipeline()
                if self.transformers_pipeline:
                    memory_usage = self.model_manager.get_memory_usage()
                    self.logger.info(f"Initialized optimized transformers NER pipeline (Memory: {memory_usage:.1f}MB)")
                else:
                    self.logger.warning("Failed to initialize optimized pipeline")
                    self.config.use_transformers_ner = False
            else:
                # Fallback to basic pipeline
                try:
                    self.transformers_pipeline = pipeline(
                        "ner",
                        model=self.config.ner_model,
                        tokenizer=self.config.ner_model,
                        aggregation_strategy="simple",
                        device_map="cpu"  # Force CPU for fallback
                    )
                    self.logger.info("Initialized basic transformers NER pipeline")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize basic transformers pipeline: {e}")
                    self.config.use_transformers_ner = False
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize transformers pipeline: {e}")
            self.config.use_transformers_ner = False
    
    async def analyze_async(self, text: str) -> LinguisticFeatures:
        """
        Perform comprehensive linguistic analysis asynchronously.
        
        Args:
            text: The text to analyze
            
        Returns:
            LinguisticFeatures object containing all analysis results
        """
        try:
            # Run analysis tasks in parallel
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
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            features = LinguisticFeatures()
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Analysis task {i} failed: {result}")
                    continue
                
                if isinstance(result, dict):
                    for key, value in result.items():
                        if hasattr(features, key):
                            setattr(features, key, value)
            
            # Calculate overall quality score
            features.overall_linguistic_quality = self._calculate_overall_quality(features)
            features.confidence = self._calculate_confidence(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Linguistic analysis failed: {e}")
            return LinguisticFeatures()
    
    def analyze(self, text: str) -> LinguisticFeatures:
        """
        Perform comprehensive linguistic analysis synchronously.
        
        Args:
            text: The text to analyze
            
        Returns:
            LinguisticFeatures object containing all analysis results
        """
        return asyncio.run(self.analyze_async(text))
    
    async def _analyze_entities(self, text: str) -> Dict[str, Any]:
        """Analyze named entities and technical terms."""
        try:
            entities = []
            entity_types = set()
            technical_terms = []
            
            # Use transformers pipeline if available
            if self.transformers_pipeline:
                try:
                    ner_results = self.transformers_pipeline(text)
                    for entity in ner_results:
                        entities.append({
                            "text": entity["word"],
                            "label": entity["entity_group"],
                            "confidence": entity["score"],
                            "start": entity.get("start", 0),
                            "end": entity.get("end", 0)
                        })
                        entity_types.add(entity["entity_group"])
                except Exception as e:
                    self.logger.warning(f"Transformers NER failed: {e}")
            
            # Fallback to NLTK-based NER
            if not entities and self.ner_extractor:
                entities = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.ner_extractor.extract_entities, text
                )
                entity_types = {e["label"] for e in entities}
            
            # Extract technical terms
            technical_terms = self._extract_technical_terms(text)
            
            # Calculate entity density
            words = len(text.split())
            entity_density = len(entities) / max(words, 1)
            
            return {
                "entities": entities,
                "entity_types": entity_types,
                "entity_density": entity_density,
                "technical_terms": technical_terms
            }
            
        except Exception as e:
            self.logger.error(f"Entity analysis failed: {e}")
            return {}
    
    async def _analyze_dependencies(self, text: str) -> Dict[str, Any]:
        """Analyze dependency parsing and syntactic structure."""
        try:
            if not self.dependency_parser:
                return {}
            
            dependencies = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.dependency_parser.parse, text
            )
            
            # Calculate syntactic complexity
            complexity = self._calculate_syntactic_complexity(dependencies)
            structure_quality = self._assess_sentence_structure(dependencies)
            
            return {
                "dependencies": dependencies,
                "syntactic_complexity": complexity,
                "sentence_structure_quality": structure_quality
            }
            
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {e}")
            return {}
    
    async def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze readability metrics."""
        try:
            loop = asyncio.get_event_loop()
            
            # Calculate various readability metrics in parallel
            tasks = [
                loop.run_in_executor(self.executor, textstat.flesch_reading_ease, text),
                loop.run_in_executor(self.executor, textstat.flesch_kincaid_grade, text),
                loop.run_in_executor(self.executor, textstat.gunning_fog, text),
                loop.run_in_executor(self.executor, textstat.smog_index, text),
                loop.run_in_executor(self.executor, textstat.coleman_liau_index, text),
            ]
            
            results = await asyncio.gather(*tasks)
            
            flesch_ease, flesch_kincaid, gunning_fog, smog, coleman_liau = results
            
            # Calculate composite readability score
            readability_score = self._calculate_readability_score(
                flesch_ease, flesch_kincaid, gunning_fog
            )
            
            return {
                "flesch_reading_ease": flesch_ease,
                "flesch_kincaid_grade": flesch_kincaid,
                "gunning_fog": gunning_fog,
                "smog_index": smog,
                "coleman_liau_index": coleman_liau,
                "readability_score": readability_score
            }
            
        except Exception as e:
            self.logger.error(f"Readability analysis failed: {e}")
            return {}
    
    async def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity metrics."""
        try:
            # Tokenize text
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            
            # Calculate metrics
            lexical_diversity = len(set(words)) / max(len(words), 1)
            avg_sentence_length = len(words) / max(len(sentences), 1)
            avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
            syllable_count = textstat.syllable_count(text)
            
            return {
                "lexical_diversity": lexical_diversity,
                "avg_sentence_length": avg_sentence_length,
                "avg_word_length": avg_word_length,
                "syllable_count": syllable_count
            }
            
        except Exception as e:
            self.logger.error(f"Complexity analysis failed: {e}")
            return {}
    
    async def _analyze_prompt_structure(self, text: str) -> Dict[str, Any]:
        """Analyze prompt structure and components."""
        try:
            # Patterns for identifying prompt components
            instruction_patterns = [
                r'\b(please|write|create|generate|explain|describe|analyze|summarize)\b',
                r'\b(you should|you need to|your task is|your goal is)\b',
                r'\b(step by step|follow these|instructions|guidelines)\b'
            ]
            
            example_patterns = [
                r'\b(for example|e\.g\.|such as|instance|example)\b',
                r'\b(here\'s an example|consider this|like this)\b',
                r'```.*?```',  # Code blocks
                r'\".*?\"'     # Quoted examples
            ]
            
            context_patterns = [
                r'\b(context|background|scenario|situation)\b',
                r'\b(given that|assuming|in this case)\b',
                r'\b(remember that|keep in mind|note that)\b'
            ]
            
            # Check for instruction clarity
            has_clear_instructions = any(
                re.search(pattern, text, re.IGNORECASE) 
                for pattern in instruction_patterns
            )
            
            # Check for examples
            has_examples = any(
                re.search(pattern, text, re.IGNORECASE | re.DOTALL) 
                for pattern in example_patterns
            )
            
            # Check for context
            has_context = any(
                re.search(pattern, text, re.IGNORECASE) 
                for pattern in context_patterns
            )
            
            # Calculate instruction clarity score
            clarity_score = self._calculate_instruction_clarity(text)
            
            return {
                "has_clear_instructions": has_clear_instructions,
                "has_examples": has_examples,
                "has_context": has_context,
                "instruction_clarity_score": clarity_score
            }
            
        except Exception as e:
            self.logger.error(f"Prompt structure analysis failed: {e}")
            return {}
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return list(words.intersection(self.config.technical_keywords))
    
    def _calculate_syntactic_complexity(self, dependencies: List[Dict]) -> float:
        """Calculate syntactic complexity score."""
        if not dependencies:
            return 0.0
        
        # Count different types of dependencies
        dep_types = set(dep.get("relation", "") for dep in dependencies)
        complexity_score = len(dep_types) / max(len(dependencies), 1)
        
        # Adjust for nested structures
        nested_count = sum(1 for dep in dependencies if dep.get("depth", 0) > 2)
        complexity_score += nested_count / max(len(dependencies), 1)
        
        return min(complexity_score, 1.0)
    
    def _assess_sentence_structure(self, dependencies: List[Dict]) -> float:
        """Assess sentence structure quality."""
        if not dependencies:
            return 0.0
        
        # Look for complete sentence structures
        has_subject = any(dep.get("relation") == "nsubj" for dep in dependencies)
        has_predicate = any(dep.get("relation") == "ROOT" for dep in dependencies)
        has_object = any(dep.get("relation") == "dobj" for dep in dependencies)
        
        structure_score = sum([has_subject, has_predicate, has_object]) / 3.0
        return structure_score
    
    def _calculate_readability_score(self, flesch_ease: float, flesch_kincaid: float, 
                                   gunning_fog: float) -> float:
        """Calculate composite readability score."""
        # Normalize Flesch Reading Ease (0-100 scale)
        flesch_normalized = max(0, min(100, flesch_ease)) / 100.0
        
        # Normalize grade levels (assume reasonable range 0-20)
        kincaid_normalized = max(0, 1 - (flesch_kincaid / 20.0))
        gunning_normalized = max(0, 1 - (gunning_fog / 20.0))
        
        # Weighted average
        return (flesch_normalized * 0.4 + kincaid_normalized * 0.3 + gunning_normalized * 0.3)
    
    def _calculate_instruction_clarity(self, text: str) -> float:
        """Calculate instruction clarity score."""
        # Look for imperative verbs
        imperative_verbs = [
            "write", "create", "generate", "explain", "describe", "analyze",
            "summarize", "list", "identify", "compare", "evaluate", "discuss"
        ]
        
        imperative_count = sum(
            len(re.findall(rf'\b{verb}\b', text, re.IGNORECASE))
            for verb in imperative_verbs
        )
        
        # Look for clear structure indicators
        structure_indicators = [
            "first", "second", "third", "next", "then", "finally",
            "step 1", "step 2", "part a", "part b"
        ]
        
        structure_count = sum(
            len(re.findall(rf'\b{indicator}\b', text, re.IGNORECASE))
            for indicator in structure_indicators
        )
        
        # Calculate score based on presence of clear instructions
        words = len(text.split())
        clarity_score = (imperative_count + structure_count) / max(words / 10, 1)
        
        return min(clarity_score, 1.0)
    
    def _calculate_overall_quality(self, features: LinguisticFeatures) -> float:
        """Calculate overall linguistic quality score with improved normalization."""
        # Use weighted average with proper normalization
        # All components should contribute to final score
        
        # Core quality components (weighted average)
        readability_weight = 0.25
        structure_weight = 0.25
        clarity_weight = 0.30
        richness_weight = 0.20
        
        # Normalize readability score (higher is better, but flesch_reading_ease can be > 100)
        readability_component = min(features.readability_score, 1.0) * readability_weight
        
        # Sentence structure quality (0-1, higher is better)
        structure_component = features.sentence_structure_quality * structure_weight
        
        # Instruction clarity (0-1, higher is better)
        clarity_component = features.instruction_clarity_score * clarity_weight
        
        # Content richness: combination of entities, lexical diversity, and technical terms
        entity_richness = min(features.entity_density * 3.0, 1.0)  # Scale entity density
        lexical_richness = features.lexical_diversity
        technical_richness = min(len(features.technical_terms) / 10.0, 1.0)  # Scale technical terms
        
        # Average richness components
        richness_score = (entity_richness + lexical_richness + technical_richness) / 3.0
        richness_component = richness_score * richness_weight
        
        # Calculate final weighted score
        total_score = readability_component + structure_component + clarity_component + richness_component
        
        # Ensure score is in [0, 1] range
        return min(max(total_score, 0.0), 1.0)
    
    def _calculate_confidence(self, features: LinguisticFeatures) -> float:
        """Calculate confidence in the analysis results."""
        # Base confidence on number of successful analyses
        analysis_count = 0
        
        if features.entities:
            analysis_count += 1
        if features.dependencies:
            analysis_count += 1
        if features.readability_score > 0:
            analysis_count += 1
        if features.lexical_diversity > 0:
            analysis_count += 1
        
        # Maximum possible analyses
        max_analyses = 4
        confidence = analysis_count / max_analyses
        
        return confidence
    
    @lru_cache(maxsize=1000)
    def analyze_cached(self, text: str) -> LinguisticFeatures:
        """Cached version of analyze method."""
        if self.config.enable_caching:
            return self.analyze(text)
        else:
            # Disable caching
            return self.analyze(text)
    
    def cleanup(self):
        """Explicitly cleanup resources."""
        try:
            if hasattr(self, 'model_manager') and self.model_manager:
                self.model_manager.cleanup()
                
            if hasattr(self, 'transformers_pipeline'):
                self.transformers_pipeline = None
                
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=False)
                
            self.logger.info("LinguisticAnalyzer resources cleaned up")
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        memory_info = {"total_mb": 0.0, "model_mb": 0.0}
        
        try:
            if self.model_manager:
                memory_info["model_mb"] = self.model_manager.get_memory_usage()
                
            # Get process memory
            import psutil
            process = psutil.Process()
            memory_info["total_mb"] = process.memory_info().rss / 1024 / 1024
            
        except Exception as e:
            self.logger.debug(f"Memory usage calculation failed: {e}")
            
        return memory_info
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction


# Configuration helpers for testing and production

def get_lightweight_config() -> LinguisticConfig:
    """
    Get configuration optimized for testing and resource-constrained environments.
    
    Returns:
        LinguisticConfig with lightweight settings
    """
    return LinguisticConfig(
        use_lightweight_models=True,
        enable_model_quantization=False,
        force_cpu_only=True,
        max_memory_threshold_mb=50,
        enable_ner=True,
        enable_dependency_parsing=False,  # Disable heavy parsing for tests
        enable_readability=True,
        enable_complexity_metrics=True,
        enable_prompt_segmentation=True,
        max_workers=2,  # Reduce concurrency for tests
        timeout_seconds=10,
        cache_size=100  # Smaller cache for tests
    )


def get_ultra_lightweight_config() -> LinguisticConfig:
    """
    Get configuration optimized for extreme memory constraints (<30MB).
    
    Uses 4-bit quantization and tiny models for minimal memory usage.
    
    Returns:
        LinguisticConfig with ultra-lightweight settings
    """
    return LinguisticConfig(
        use_ultra_lightweight_models=True,
        enable_model_quantization=True,
        enable_4bit_quantization=True,
        quantization_bits=4,
        force_cpu_only=True,
        max_memory_threshold_mb=30,
        enable_ner=True,
        enable_dependency_parsing=False,  # Disable to save memory
        enable_readability=True,
        enable_complexity_metrics=True,
        enable_prompt_segmentation=True,
        max_workers=1,  # Minimal concurrency
        timeout_seconds=10,
        cache_size=50  # Minimal cache
    )


def get_production_config() -> LinguisticConfig:
    """
    Get configuration optimized for production use with memory efficiency.
    
    Returns:
        LinguisticConfig with production-optimized settings
    """
    return LinguisticConfig(
        use_lightweight_models=False,
        enable_model_quantization=True,
        enable_4bit_quantization=True,  # Enable 4-bit for production efficiency
        quantization_bits=8,
        force_cpu_only=False,
        max_memory_threshold_mb=100,  # Reduced from 200MB for better efficiency
        enable_ner=True,
        enable_dependency_parsing=True,
        enable_readability=True,
        enable_complexity_metrics=True,
        enable_prompt_segmentation=True,
        auto_download_nltk=True,
        nltk_fallback_enabled=True,
        max_workers=4,
        timeout_seconds=30,
        cache_size=1000
    )


def get_memory_optimized_config(target_memory_mb: int) -> LinguisticConfig:
    """
    Get configuration optimized for a specific memory target.
    
    Args:
        target_memory_mb: Target memory usage in MB
        
    Returns:
        LinguisticConfig optimized for the target memory
    """
    if target_memory_mb < 30:
        return get_ultra_lightweight_config()
    elif target_memory_mb < 50:
        config = get_lightweight_config()
        config.enable_4bit_quantization = True
        config.max_memory_threshold_mb = target_memory_mb
        return config
    else:
        config = get_production_config()
        config.max_memory_threshold_mb = target_memory_mb
        return config


def create_test_analyzer() -> LinguisticAnalyzer:
    """
    Create a LinguisticAnalyzer instance optimized for testing.
    
    Returns:
        LinguisticAnalyzer with lightweight configuration
    """
    config = get_lightweight_config()
    return LinguisticAnalyzer(config) 