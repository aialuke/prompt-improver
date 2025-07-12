import pytest
from src.prompt_improver.services.test_generator import IntelligentTestGenerator


def test_generate_test_suite():
    generator = IntelligentTestGenerator()
    context = {"project": "example_project"}
    result = generator.generate_test_suite(context)
    assert len(result["testCases"]) == 100  # Default setting
    assert all("id" in test and "score" in test for test in result["testCases"])


def test_generate_test_suite_custom_count():
    generator = IntelligentTestGenerator()
    context = {"project": "custom_project"}
    options = {"testCount": 50}
    result = generator.generate_test_suite(context, options)
    assert len(result["testCases"]) == 50
    assert result["metadata"]["totalGenerated"] == 50


def test_quality_score_within_threshold():
    generator = IntelligentTestGenerator()
    context = {"project": "quality_test"}
    result = generator.generate_test_suite(context)
    
    # Check all scores meet quality threshold
    for test_case in result["testCases"]:
        assert test_case["score"] >= generator.config["qualityThreshold"]
        assert test_case["score"] <= 1.0


def test_config_customization():
    custom_config = {
        "defaultTestCount": 25,
        "qualityThreshold": 0.8
    }
    generator = IntelligentTestGenerator(custom_config)
    context = {"project": "config_test"}
    result = generator.generate_test_suite(context)
    
    assert len(result["testCases"]) == 25
    for test_case in result["testCases"]:
        assert test_case["score"] >= 0.8
