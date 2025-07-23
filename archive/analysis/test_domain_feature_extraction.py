"""
Tests for Domain-Specific Feature Extraction System

This test suite validates the domain detection and domain-specific
feature extraction functionality.
"""

from typing import Any, Dict

import numpy as np
import pytest
from src.prompt_improver.analysis.domain_detector import (
    DomainClassificationResult,
    DomainDetector,
    PromptDomain,
)
from src.prompt_improver.analysis.domain_feature_extractor import (
    AcademicDomainExtractor,
    CreativeDomainExtractor,
    DomainFeatureExtractor,
    DomainFeatures,
    TechnicalDomainExtractor,
)


class TestDomainDetector:
    """Test domain detection functionality."""

    @pytest.fixture
    def detector(self):
        """Create domain detector for testing."""
        return DomainDetector(
            use_spacy=False
        )  # Use False to avoid spaCy dependency in tests

    def test_software_development_detection(self, detector):
        """Test detection of software development domain."""
        text = """
        I need help writing a Python function that implements a binary search algorithm.
        The function should take a sorted list and a target value, returning the index
        if found or -1 if not found. Please include unit tests and error handling.
        """

        result = detector.detect_domain(text)

        assert result.primary_domain == PromptDomain.SOFTWARE_DEVELOPMENT
        assert result.confidence > 0.15
        assert result.technical_complexity > 0.1
        assert "software_development" in result.domain_keywords_found

    def test_data_science_detection(self, detector):
        """Test detection of data science domain."""
        text = """
        Help me analyze this dataset using pandas and numpy. I need to perform
        exploratory data analysis, create visualizations with matplotlib,
        and build a machine learning model to predict customer churn.
        """

        result = detector.detect_domain(text)

        assert result.primary_domain in [PromptDomain.DATA_SCIENCE, PromptDomain.AI_ML]
        assert result.confidence > 0.15
        assert result.technical_complexity > 0.2

    def test_creative_writing_detection(self, detector):
        """Test detection of creative writing domain."""
        text = """
        Write a compelling short story about a protagonist who discovers
        they have the ability to see people's emotions as colors. Focus on
        character development, vivid imagery, and create a dramatic climax
        that resolves the internal conflict.
        """

        result = detector.detect_domain(text)

        assert result.primary_domain == PromptDomain.CREATIVE_WRITING
        assert result.confidence > 0.15
        assert "creative_writing" in result.domain_keywords_found

    def test_academic_writing_detection(self, detector):
        """Test detection of academic writing domain."""
        text = """
        I need to write a research paper analyzing the methodology used in
        recent studies on climate change. Include a literature review,
        hypothesis, and statistical analysis of the data with proper citations.
        """

        result = detector.detect_domain(text)

        assert result.primary_domain in [
            PromptDomain.RESEARCH,
            PromptDomain.ACADEMIC_WRITING,
        ]
        assert result.confidence > 0.15
        assert (
            "research" in result.domain_keywords_found
            or "academic_writing" in result.domain_keywords_found
        )

    def test_business_analysis_detection(self, detector):
        """Test detection of business analysis domain."""
        text = """
        Perform a SWOT analysis for our company's market position.
        Include ROI calculations, KPI tracking, and competitive analysis
        with recommendations for improving efficiency and profitability.
        """

        result = detector.detect_domain(text)

        assert result.primary_domain == PromptDomain.BUSINESS_ANALYSIS
        assert result.confidence > 0.15
        assert "business_analysis" in result.domain_keywords_found

    def test_general_domain_fallback(self, detector):
        """Test fallback to general domain for unclear text."""
        text = "Hello, how are you today?"

        result = detector.detect_domain(text)

        assert result.primary_domain == PromptDomain.GENERAL
        assert result.confidence < 0.5

    def test_hybrid_domain_detection(self, detector):
        """Test detection of hybrid domains."""
        text = """
        Create a machine learning model to predict story engagement.
        Use Python and scikit-learn to analyze narrative elements,
        character development, and plot structure from creative writing samples.
        """

        result = detector.detect_domain(text)

        # Should detect multiple strong domains
        assert len(result.secondary_domains) > 0
        assert (
            result.hybrid_domain is True
            or len([s for s in result.secondary_domains if s[1] > 0.2]) > 0
        )

    def test_keyword_addition(self, detector):
        """Test adding new keywords to domains."""
        new_keywords = {"neural network", "deep learning", "tensorflow"}
        detector.add_domain_keywords(PromptDomain.AI_ML, new_keywords)

        text = "Build a neural network using tensorflow for deep learning applications."
        result = detector.detect_domain(text)

        assert result.primary_domain == PromptDomain.AI_ML
        assert result.confidence > 0.15


class TestTechnicalDomainExtractor:
    """Test technical domain feature extraction."""

    @pytest.fixture
    def extractor(self):
        """Create technical domain extractor for testing."""
        return TechnicalDomainExtractor()

    def test_code_detection(self, extractor):
        """Test detection of code snippets and patterns."""
        text = """
        Here's a Python function:
        ```python
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        ```
        """

        domain_result = DomainClassificationResult(
            primary_domain=PromptDomain.SOFTWARE_DEVELOPMENT, confidence=0.8
        )

        features = extractor.extract_features(text, domain_result)

        assert features["code_snippets_count"] > 0
        assert features["has_code_snippets"] is True
        assert features["code_snippets_density"] > 0

    def test_api_reference_detection(self, extractor):
        """Test detection of API references."""
        text = """
        Send a GET request to /api/users/{id} to retrieve user data.
        The response will be a JSON object with user details.
        """

        domain_result = DomainClassificationResult(
            primary_domain=PromptDomain.API_DOCUMENTATION, confidence=0.8
        )

        features = extractor.extract_features(text, domain_result)

        assert features["api_references_count"] > 0
        assert features["has_api_references"] is True

    def test_algorithm_concepts(self, extractor):
        """Test detection of algorithmic concepts."""
        text = """
        Implement a sorting algorithm with O(n log n) time complexity.
        Consider using quicksort or mergesort for optimal performance.
        """

        domain_result = DomainClassificationResult(
            primary_domain=PromptDomain.SOFTWARE_DEVELOPMENT, confidence=0.8
        )

        features = extractor.extract_features(text, domain_result)

        assert features["algorithms_count"] > 0
        assert features["has_algorithms"] is True


class TestCreativeDomainExtractor:
    """Test creative domain feature extraction."""

    @pytest.fixture
    def extractor(self):
        """Create creative domain extractor for testing."""
        return CreativeDomainExtractor()

    def test_narrative_elements(self, extractor):
        """Test detection of narrative elements."""
        text = """
        The protagonist faced a terrible conflict when the antagonist
        revealed the truth about her past. The dramatic climax reached
        its peak as she had to choose between revenge and forgiveness.
        """

        domain_result = DomainClassificationResult(
            primary_domain=PromptDomain.CREATIVE_WRITING, confidence=0.8
        )

        features = extractor.extract_features(text, domain_result)

        assert features["narrative_elements_count"] > 0
        assert features["narrative_elements_presence"] is True

    def test_emotional_language(self, extractor):
        """Test detection of emotional language."""
        text = """
        She felt overwhelming joy and excitement as she discovered
        the beautiful, stunning landscape. Her heart filled with
        hope and delight at the magnificent view.
        """

        domain_result = DomainClassificationResult(
            primary_domain=PromptDomain.CREATIVE_WRITING, confidence=0.8
        )

        features = extractor.extract_features(text, domain_result)

        assert features["emotional_language_count"] > 0
        assert features["emotional_language_presence"] is True
        assert features["positive_emotion_count"] > 0
        assert features["has_emotional_content"] is True

    def test_dialogue_detection(self, extractor):
        """Test detection of dialogue."""
        text = """
        "Hello there," she said with a smile.
        "How are you feeling today?" he replied nervously.
        """

        domain_result = DomainClassificationResult(
            primary_domain=PromptDomain.CREATIVE_WRITING, confidence=0.8
        )

        features = extractor.extract_features(text, domain_result)

        assert features["has_dialogue"] is True
        assert features["dialogue_count"] > 0


class TestAcademicDomainExtractor:
    """Test academic domain feature extraction."""

    @pytest.fixture
    def extractor(self):
        """Create academic domain extractor for testing."""
        return AcademicDomainExtractor()

    def test_research_methodology(self, extractor):
        """Test detection of research methodology terms."""
        text = """
        This study uses a quantitative methodology with a sample of 200 participants.
        We conducted surveys and interviews to collect data for statistical analysis.
        """

        domain_result = DomainClassificationResult(
            primary_domain=PromptDomain.RESEARCH, confidence=0.8
        )

        features = extractor.extract_features(text, domain_result)

        assert features["research_methodology_count"] > 0
        assert features["has_research_methodology"] is True

    def test_academic_structure(self, extractor):
        """Test detection of academic structure elements."""
        text = """
        The abstract provides an overview of the methodology and results.
        The introduction establishes the research question and hypothesis.
        """

        domain_result = DomainClassificationResult(
            primary_domain=PromptDomain.ACADEMIC_WRITING, confidence=0.8
        )

        features = extractor.extract_features(text, domain_result)

        assert features["academic_structure_count"] > 0
        assert features["has_academic_structure"] is True

    def test_objectivity_analysis(self, extractor):
        """Test analysis of objective vs subjective language."""
        text = """
        The study found that participants showed significant improvement.
        Results indicate a strong correlation between variables.
        Data suggest that the hypothesis is supported.
        """

        domain_result = DomainClassificationResult(
            primary_domain=PromptDomain.RESEARCH, confidence=0.8
        )

        features = extractor.extract_features(text, domain_result)

        assert features["objective_language_count"] > 0
        assert features["objectivity_ratio"] > 0.5
        assert features["has_formal_tone"] is True


class TestDomainFeatureExtractor:
    """Test the main domain feature extractor."""

    @pytest.fixture
    def extractor(self):
        """Create domain feature extractor for testing."""
        return DomainFeatureExtractor(enable_spacy=False)

    def test_software_development_features(self, extractor):
        """Test comprehensive feature extraction for software development."""
        text = """
        Write a Python function that implements a REST API endpoint
        using Flask. Include error handling, input validation, and
        comprehensive unit tests. The function should follow clean code
        principles and include proper documentation.
        """

        features = extractor.extract_domain_features(text)

        assert features.domain in [
            PromptDomain.SOFTWARE_DEVELOPMENT,
            PromptDomain.WEB_DEVELOPMENT,
        ]
        assert features.confidence > 0.15
        assert len(features.technical_features) > 0
        assert len(features.feature_vector) > 0
        assert len(features.feature_names) == len(features.feature_vector)

    def test_creative_writing_features(self, extractor):
        """Test comprehensive feature extraction for creative writing."""
        text = """
        Write a captivating short story about a young woman who discovers
        she can see people's memories as vivid, colorful dreams. Focus on
        rich character development, emotional depth, and create a powerful
        climax that explores themes of identity and belonging.
        """

        features = extractor.extract_domain_features(text)

        assert features.domain == PromptDomain.CREATIVE_WRITING
        assert features.confidence > 0.15
        assert len(features.creative_features) > 0
        assert features.creative_features.get("narrative_elements_count", 0) > 0

    def test_academic_research_features(self, extractor):
        """Test comprehensive feature extraction for academic research."""
        text = """
        Conduct a systematic literature review on machine learning applications
        in healthcare. Include methodology, statistical analysis of findings,
        and evidence-based conclusions with proper citations and references.
        """

        features = extractor.extract_domain_features(text)

        assert features.domain in [PromptDomain.RESEARCH, PromptDomain.ACADEMIC_WRITING]
        assert features.confidence > 0.15
        assert len(features.academic_features) > 0

    def test_conversational_features(self, extractor):
        """Test extraction of conversational features."""
        text = """
        Could you please help me understand how machine learning works?
        I'm really curious about this topic and would appreciate a clear
        explanation with some examples. This is quite urgent for my project.
        """

        features = extractor.extract_domain_features(text)

        conv_features = features.conversational_features
        assert conv_features["question_count"] > 0
        assert conv_features["has_questions"] is True
        assert conv_features["has_polite_requests"] is True
        assert conv_features["has_urgency"] is True

    def test_empty_text_handling(self, extractor):
        """Test handling of empty or invalid text."""
        features = extractor.extract_domain_features("")

        assert features.domain == PromptDomain.GENERAL
        assert features.confidence == 0.0

        features = extractor.extract_domain_features(None)
        assert features.domain == PromptDomain.GENERAL

    def test_domain_suitability_analysis(self, extractor):
        """Test domain suitability analysis."""
        text = """
        Build a neural network using Python and TensorFlow to classify
        images. Include data preprocessing, model training, and evaluation
        with proper metrics and visualization.
        """

        # Analyze suitability for AI/ML domain
        analysis = extractor.analyze_prompt_domain_suitability(text, PromptDomain.AI_ML)

        assert analysis["target_domain"] == PromptDomain.AI_ML.value
        assert analysis["suitability_score"] > 0.3
        assert analysis["keyword_coverage"] > 0
        assert len(analysis["recommendations"]) >= 0

    def test_feature_vector_consistency(self, extractor):
        """Test that feature vectors are consistent across calls."""
        text = """
        Develop a web application using React and Node.js with
        a PostgreSQL database and RESTful API architecture.
        """

        features1 = extractor.extract_domain_features(text)
        features2 = extractor.extract_domain_features(text)

        assert features1.domain == features2.domain
        assert features1.confidence == features2.confidence
        assert len(features1.feature_vector) == len(features2.feature_vector)
        assert features1.feature_vector == features2.feature_vector

    def test_all_feature_names(self, extractor):
        """Test that all feature names are properly defined."""
        feature_names = extractor.get_all_feature_names()

        assert len(feature_names) > 0
        assert "domain_confidence" in feature_names
        assert "domain_complexity" in feature_names
        assert "domain_specificity" in feature_names

        # Check for technical features
        tech_feature_found = any("technical_" in name for name in feature_names)
        assert tech_feature_found

        # Check for creative features
        creative_feature_found = any("creative_" in name for name in feature_names)
        assert creative_feature_found

        # Check for academic features
        academic_feature_found = any("academic_" in name for name in feature_names)
        assert academic_feature_found


class TestDomainFeatureIntegration:
    """Test integration scenarios and edge cases."""

    @pytest.fixture
    def extractor(self):
        """Create domain feature extractor for testing."""
        return DomainFeatureExtractor(enable_spacy=False)

    def test_multi_domain_prompt(self, extractor):
        """Test prompts that span multiple domains."""
        text = """
        Create a data science project that analyzes creative writing samples
        to predict story engagement. Use Python, pandas, and machine learning
        to identify narrative patterns and emotional themes in fiction.
        """

        features = extractor.extract_domain_features(text)

        # Should detect multiple domain signals
        assert features.confidence > 0.15
        # Since this combines data science/technical and creative domains,
        # at least one of these should have features
        has_domain_features = (
            len(features.technical_features) > 0
            or len(features.creative_features) > 0
            or len(features.secondary_domains) > 0
        )

        assert has_domain_features

    def test_domain_confidence_scaling(self, extractor):
        """Test that domain confidence affects feature weighting."""
        # High confidence technical prompt
        high_conf_text = """
        Implement a binary search tree in Python with insert, delete, and search
        operations. Include comprehensive unit tests using pytest and proper
        documentation with type hints and docstrings.
        """

        # Lower confidence technical prompt
        low_conf_text = """
        I need some help with coding. Maybe something with Python?
        """

        high_conf_features = extractor.extract_domain_features(high_conf_text)
        low_conf_features = extractor.extract_domain_features(low_conf_text)

        assert high_conf_features.confidence > low_conf_features.confidence
        assert high_conf_features.complexity_score > low_conf_features.complexity_score

    def test_performance_with_long_text(self, extractor):
        """Test performance with longer text inputs."""
        # Create a longer text by repeating content
        base_text = """
        Write a Python function that implements machine learning algorithms
        for data analysis and visualization. Include proper error handling,
        documentation, and unit tests for all functionality.
        """
        long_text = base_text * 10  # Repeat 10 times

        # Should still process efficiently
        features = extractor.extract_domain_features(long_text)

        assert (
            features.domain == PromptDomain.SOFTWARE_DEVELOPMENT
            or features.domain == PromptDomain.AI_ML
        )
        assert features.confidence > 0.15
        assert len(features.feature_vector) > 0

    @pytest.mark.asyncio
    async def test_async_extraction(self, extractor):
        """Test asynchronous feature extraction."""
        text = """
        Build a machine learning model using scikit-learn to classify
        customer feedback sentiment. Include data preprocessing, feature
        engineering, and model evaluation with cross-validation.
        """

        features = await extractor.extract_domain_features_async(text)

        assert features.domain in [PromptDomain.AI_ML, PromptDomain.DATA_SCIENCE]
        assert features.confidence > 0.15
        assert len(features.feature_vector) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
