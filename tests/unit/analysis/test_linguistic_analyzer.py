"""
Unit tests for LinguisticAnalyzer class.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch

from prompt_improver.analysis.linguistic_analyzer import (
    LinguisticAnalyzer, LinguisticConfig, LinguisticFeatures
)


class TestLinguisticAnalyzer:
    """Test cases for LinguisticAnalyzer."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LinguisticConfig(
            enable_ner=True,
            enable_dependency_parsing=True,
            enable_readability=True,
            enable_complexity_metrics=True,
            enable_prompt_segmentation=True,
            use_transformers_ner=False,  # Disable transformers for testing
            max_workers=2
        )
    
    @pytest.fixture
    def analyzer(self, config):
        """Create LinguisticAnalyzer instance."""
        return LinguisticAnalyzer(config)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        Please write a Python function that calculates the factorial of a number.
        The function should handle edge cases like negative numbers.
        For example, factorial(5) should return 120.
        Remember to include proper error handling and documentation.
        """
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config is not None
        assert analyzer.ner_extractor is not None
        assert analyzer.dependency_parser is not None
        assert analyzer.executor is not None
    
    def test_analyze_sync(self, analyzer, sample_text):
        """Test synchronous analysis."""
        features = analyzer.analyze(sample_text)
        
        assert isinstance(features, LinguisticFeatures)
        assert features.overall_linguistic_quality >= 0.0
        assert features.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_async(self, analyzer, sample_text):
        """Test asynchronous analysis."""
        features = await analyzer.analyze_async(sample_text)
        
        assert isinstance(features, LinguisticFeatures)
        assert features.overall_linguistic_quality >= 0.0
        assert features.confidence >= 0.0
    
    def test_readability_analysis(self, analyzer, sample_text):
        """Test readability analysis."""
        features = analyzer.analyze(sample_text)
        
        # Should have readability scores
        assert features.flesch_reading_ease >= 0
        assert features.flesch_kincaid_grade >= 0
        assert features.readability_score >= 0
    
    def test_complexity_analysis(self, analyzer, sample_text):
        """Test complexity analysis."""
        features = analyzer.analyze(sample_text)
        
        # Should have complexity metrics
        assert features.lexical_diversity >= 0
        assert features.avg_sentence_length > 0
        assert features.avg_word_length > 0
        assert features.syllable_count >= 0
    
    def test_prompt_structure_analysis(self, analyzer, sample_text):
        """Test prompt structure analysis."""
        features = analyzer.analyze(sample_text)
        
        # Should identify prompt components
        assert features.has_clear_instructions is True  # "Please write"
        assert features.has_examples is True           # "For example"
        assert features.has_context is True            # "Remember"
        assert features.instruction_clarity_score > 0
    
    def test_entity_extraction(self, analyzer):
        """Test entity extraction."""
        text = "Use Python and JavaScript to build an API with JSON responses."
        features = analyzer.analyze(text)
        
        # Should extract technical terms
        assert len(features.technical_terms) > 0
        assert features.entity_density >= 0
    
    def test_dependency_parsing(self, analyzer, sample_text):
        """Test dependency parsing."""
        features = analyzer.analyze(sample_text)
        
        # Should have dependency analysis results
        assert features.syntactic_complexity >= 0
        assert features.sentence_structure_quality >= 0
    
    def test_empty_text(self, analyzer):
        """Test analysis of empty text."""
        features = analyzer.analyze("")
        
        assert isinstance(features, LinguisticFeatures)
        assert features.overall_linguistic_quality == 0.0
    
    def test_short_text(self, analyzer):
        """Test analysis of very short text."""
        features = analyzer.analyze("Hello.")
        
        assert isinstance(features, LinguisticFeatures)
        assert features.confidence >= 0.0
    
    def test_technical_text(self, analyzer):
        """Test analysis of technical text."""
        text = """
        Configure the neural network with transformer architecture.
        Use BERT embeddings for token classification.
        Fine-tune the model on your dataset using PyTorch.
        """
        
        features = analyzer.analyze(text)
        
        # Should identify many technical terms
        assert len(features.technical_terms) >= 3
        assert "neural" in features.technical_terms
        assert "transformer" in features.technical_terms
        assert "bert" in features.technical_terms
    
    def test_confidence_calculation(self, analyzer, sample_text):
        """Test confidence calculation."""
        features = analyzer.analyze(sample_text)
        
        # Confidence should be reasonable for well-structured text
        assert 0.0 <= features.confidence <= 1.0
        assert features.confidence > 0.5  # Should be confident with good analysis
    
    def test_overall_quality_calculation(self, analyzer, sample_text):
        """Test overall quality calculation."""
        features = analyzer.analyze(sample_text)
        
        # Quality should be reasonable for well-structured prompt
        assert 0.0 <= features.overall_linguistic_quality <= 1.0
        assert features.overall_linguistic_quality > 0.3  # Should have decent quality
    
    def test_caching_functionality(self, config):
        """Test caching functionality."""
        config.enable_caching = True
        analyzer = LinguisticAnalyzer(config)
        
        text = "Test caching with this text."
        
        # First analysis
        features1 = analyzer.analyze_cached(text)
        
        # Second analysis (should use cache)
        features2 = analyzer.analyze_cached(text)
        
        # Results should be identical
        assert features1.overall_linguistic_quality == features2.overall_linguistic_quality
    
    def test_disabled_features(self):
        """Test analyzer with disabled features."""
        config = LinguisticConfig(
            enable_ner=False,
            enable_dependency_parsing=False,
            enable_readability=True,
            enable_complexity_metrics=False,
            enable_prompt_segmentation=False
        )
        
        analyzer = LinguisticAnalyzer(config)
        features = analyzer.analyze("Test text for disabled features.")
        
        # Should still work but with limited features
        assert isinstance(features, LinguisticFeatures)
        assert len(features.entities) == 0  # NER disabled
        assert len(features.dependencies) == 0  # Dependency parsing disabled
        assert features.readability_score > 0  # Readability enabled
    
    def test_malformed_text(self, analyzer):
        """Test analysis of malformed text."""
        malformed_text = "This is... incomplete and has weird... formatting!!!"
        
        features = analyzer.analyze(malformed_text)
        
        # Should handle gracefully
        assert isinstance(features, LinguisticFeatures)
        assert features.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, analyzer):
        """Test concurrent analysis of multiple texts."""
        texts = [
            "Write a function to sort numbers.",
            "Create a machine learning model for classification.",
            "Build a web API using REST principles."
        ]
        
        # Analyze concurrently
        tasks = [analyzer.analyze_async(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 3
        for features in results:
            assert isinstance(features, LinguisticFeatures)
            assert features.confidence >= 0.0
    
    def test_linguistic_features_dataclass(self):
        """Test LinguisticFeatures dataclass."""
        features = LinguisticFeatures()
        
        # Check default values
        assert features.entities == []
        assert features.entity_types == set()
        assert features.entity_density == 0.0
        assert features.technical_terms == []
        assert features.dependencies == []
        assert features.overall_linguistic_quality == 0.0
        assert features.confidence == 0.0
    
    def test_config_dataclass(self):
        """Test LinguisticConfig dataclass."""
        config = LinguisticConfig()
        
        # Check default values
        assert config.enable_ner is True
        assert config.enable_dependency_parsing is True
        assert config.enable_readability is True
        assert config.enable_complexity_metrics is True
        assert config.enable_prompt_segmentation is True
        assert config.enable_caching is True
        assert config.max_workers == 4
        assert len(config.technical_keywords) > 0 