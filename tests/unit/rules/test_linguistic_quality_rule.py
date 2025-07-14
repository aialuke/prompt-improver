"""
Unit tests for LinguisticQualityRule.
"""

import pytest
from unittest.mock import MagicMock, patch

from prompt_improver.rules.linguistic_quality_rule import LinguisticQualityRule
from prompt_improver.analysis.linguistic_analyzer import LinguisticFeatures


class TestLinguisticQualityRule:
    """Test cases for LinguisticQualityRule."""
    
    @pytest.fixture
    def rule(self):
        """Create LinguisticQualityRule instance."""
        return LinguisticQualityRule()
    
    @pytest.fixture
    def good_prompt(self):
        """Sample good prompt for testing."""
        return """
        Please write a Python function that calculates the factorial of a number.
        The function should handle edge cases like negative numbers.
        For example, factorial(5) should return 120.
        Remember to include proper error handling and documentation.
        """
    
    @pytest.fixture
    def poor_prompt(self):
        """Sample poor prompt for testing."""
        return "do something with numbers"
    
    def test_rule_initialization(self, rule):
        """Test rule initialization."""
        assert rule.name == "Linguistic Quality Rule"
        assert rule.linguistic_analyzer is not None
        assert rule.thresholds is not None
        assert len(rule.thresholds) > 0
    
    def test_evaluate_good_prompt(self, rule, good_prompt):
        """Test evaluation of a good prompt."""
        result = rule.evaluate(good_prompt)
        
        assert isinstance(result, dict)
        assert "score" in result
        assert "confidence" in result
        assert "passed" in result
        assert "component_scores" in result
        assert "linguistic_features" in result
        assert "suggestions" in result
        assert "explanation" in result
        assert "metadata" in result
        
        # Should pass with reasonable score
        assert result["score"] > 0.5
        assert result["confidence"] >= 0.0
        assert result["passed"] is True
    
    def test_evaluate_poor_prompt(self, rule, poor_prompt):
        """Test evaluation of a poor prompt."""
        result = rule.evaluate(poor_prompt)
        
        assert isinstance(result, dict)
        assert result["score"] < 0.6  # Should score lower
        assert len(result["suggestions"]) > 0  # Should have suggestions
    
    def test_evaluate_empty_prompt(self, rule):
        """Test evaluation of empty prompt."""
        result = rule.evaluate("")
        
        assert result["score"] == 0.0
        assert result["passed"] is False
        assert "empty" in result["explanation"].lower()
    
    def test_component_scores(self, rule, good_prompt):
        """Test component score calculation."""
        result = rule.evaluate(good_prompt)
        
        component_scores = result["component_scores"]
        
        assert "readability" in component_scores
        assert "syntactic_complexity" in component_scores
        assert "structure_quality" in component_scores
        assert "entity_richness" in component_scores
        assert "instruction_clarity" in component_scores
        
        # All scores should be between 0 and 1
        for score in component_scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_linguistic_features_extraction(self, rule, good_prompt):
        """Test linguistic features extraction."""
        result = rule.evaluate(good_prompt)
        
        features = result["linguistic_features"]
        
        assert "flesch_reading_ease" in features
        assert "avg_sentence_length" in features
        assert "lexical_diversity" in features
        assert "entity_count" in features
        assert "technical_terms" in features
        assert "has_clear_instructions" in features
        assert "has_examples" in features
        assert "has_context" in features
        
        # Should detect prompt structure
        assert features["has_clear_instructions"] is True  # "Please write"
        assert features["has_examples"] is True           # "For example"
        assert features["has_context"] is True            # "Remember"
    
    def test_technical_prompt_analysis(self, rule):
        """Test analysis of technical prompt."""
        technical_prompt = """
        Create a machine learning model using Python and scikit-learn.
        Use neural networks for classification with JSON data input.
        The API should return predictions in XML format.
        """
        
        result = rule.evaluate(technical_prompt)
        
        features = result["linguistic_features"]
        technical_terms = features["technical_terms"]
        
        # Should identify technical terms
        assert len(technical_terms) > 0
        assert any(term in technical_terms for term in ["python", "machine", "learning", "neural", "json", "xml"])
    
    def test_suggestions_generation(self, rule):
        """Test suggestion generation for different prompt types."""
        # Test prompt with clarity issues
        unclear_prompt = "Make it work better somehow"
        result = rule.evaluate(unclear_prompt)
        
        suggestions = result["suggestions"]
        assert len(suggestions) > 0
        
        # Should suggest adding clear instructions
        suggestion_text = " ".join(suggestions).lower()
        assert any(word in suggestion_text for word in ["clear", "specific", "action", "examples"])
    
    def test_explanation_generation(self, rule, good_prompt, poor_prompt):
        """Test explanation generation."""
        good_result = rule.evaluate(good_prompt)
        poor_result = rule.evaluate(poor_prompt)
        
        # Good prompt should have positive explanation
        good_explanation = good_result["explanation"].lower()
        assert any(word in good_explanation for word in ["good", "excellent", "quality"])
        
        # Poor prompt should have improvement explanation
        poor_explanation = poor_result["explanation"].lower()
        assert any(word in poor_explanation for word in ["improvement", "needs", "focus"])
    
    def test_readability_assessment(self, rule):
        """Test readability assessment."""
        # Complex prompt
        complex_prompt = """
        Utilizing sophisticated methodologies, endeavor to construct an implementation 
        that facilitates the computational determination of mathematical relationships 
        existing between multidimensional datasets through algorithmic procedures.
        """
        
        # Simple prompt
        simple_prompt = "Write a function to add two numbers."
        
        complex_result = rule.evaluate(complex_prompt)
        simple_result = rule.evaluate(simple_prompt)
        
        # Simple prompt should have better readability
        complex_readability = complex_result["component_scores"]["readability"]
        simple_readability = simple_result["component_scores"]["readability"]
        
        assert simple_readability >= complex_readability
    
    def test_entity_richness_assessment(self, rule):
        """Test entity richness assessment."""
        # Rich prompt with entities
        rich_prompt = """
        Use Python and TensorFlow to build a neural network for analyzing 
        customer data from the e-commerce database. Deploy on AWS.
        """
        
        # Sparse prompt
        sparse_prompt = "Do something."
        
        rich_result = rule.evaluate(rich_prompt)
        sparse_result = rule.evaluate(sparse_prompt)
        
        # Rich prompt should have better entity score
        rich_entity_score = rich_result["component_scores"]["entity_richness"]
        sparse_entity_score = sparse_result["component_scores"]["entity_richness"]
        
        assert rich_entity_score > sparse_entity_score
    
    def test_error_handling(self, rule):
        """Test error handling."""
        # Mock analyzer to raise exception
        with patch.object(rule.linguistic_analyzer, 'analyze', side_effect=Exception("Test error")):
            result = rule.evaluate("test prompt")
            
            assert result["score"] == 0.0
            assert result["passed"] is False
            assert "error" in result["explanation"].lower()
            assert result["metadata"]["error"] is True
    
    def test_metadata_information(self, rule, good_prompt):
        """Test metadata information."""
        result = rule.evaluate(good_prompt)
        
        metadata = result["metadata"]
        
        assert metadata["rule_name"] == "Linguistic Quality Rule"
        assert metadata["analysis_method"] == "advanced_linguistic"
        assert "features_analyzed" in metadata
        assert len(metadata["features_analyzed"]) > 0
    
    def test_weighted_scoring(self, rule):
        """Test weighted scoring calculation."""
        # Create mock features with known values
        mock_features = LinguisticFeatures()
        mock_features.readability_score = 0.8
        mock_features.syntactic_complexity = 0.5
        mock_features.sentence_structure_quality = 0.7
        mock_features.entity_density = 0.2
        mock_features.technical_terms = ["python", "api"]
        mock_features.entity_types = {"PROGRAMMING_LANGUAGE", "API_ENDPOINT"}
        mock_features.entities = [{"label": "PROGRAMMING_LANGUAGE", "text": "python"}]
        mock_features.instruction_clarity_score = 0.6
        mock_features.has_clear_instructions = True
        mock_features.has_examples = False
        mock_features.has_context = False
        
        # Calculate component scores
        readability_score = rule._assess_readability(mock_features)
        complexity_score = rule._assess_complexity(mock_features)
        structure_score = rule._assess_structure(mock_features)
        entity_score = rule._assess_entity_richness(mock_features)
        clarity_score = rule._assess_clarity(mock_features)
        
        # Calculate overall score
        component_scores = {
            "readability": readability_score,
            "complexity": complexity_score,
            "structure": structure_score,
            "entity_richness": entity_score,
            "clarity": clarity_score
        }
        
        overall_score = rule._calculate_overall_score(component_scores)
        
        # Verify score is reasonable
        assert 0.0 <= overall_score <= 1.0
        assert isinstance(overall_score, float)
    
    def test_threshold_configuration(self, rule):
        """Test threshold configuration."""
        thresholds = rule.thresholds
        
        # Verify all required thresholds exist
        required_thresholds = [
            "min_readability", "max_complexity", "min_instruction_clarity",
            "min_entity_diversity", "optimal_sentence_length", "max_sentence_length"
        ]
        
        for threshold in required_thresholds:
            assert threshold in thresholds
            assert isinstance(thresholds[threshold], (int, float))
            assert thresholds[threshold] >= 0 