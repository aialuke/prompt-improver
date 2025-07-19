"""
Unit tests for the ClarityRule.
"""

from prompt_improver.rule_engine.base import TransformationResult
from prompt_improver.rule_engine.rules.clarity import ClarityRule


def test_clarity_rule_check_applicable():
    """
    Test that the ClarityRule's check method correctly identifies a prompt with vague words.
    """
    rule = ClarityRule()
    prompt = "Can you make this thing better?"
    result = rule.check(prompt)
    assert result.applies is True
    assert "thing" in result.metadata["vague_words"]
    assert "better" in result.metadata["vague_words"]


def test_clarity_rule_check_not_applicable():
    """
    Test that the ClarityRule's check method correctly identifies a prompt without vague words.
    """
    rule = ClarityRule()
    prompt = "Rewrite the following paragraph to be suitable for a fifth-grade reading level."
    result = rule.check(prompt)
    assert result.applies is False


def test_clarity_rule_apply():
    """
    Test that the ClarityRule's apply method correctly transforms a prompt
    and returns a valid TransformationResult.
    """
    rule = ClarityRule()
    prompt = "Summarize this."
    result = rule.apply(prompt)

    assert isinstance(result, TransformationResult)
    assert result.success is True
    assert result.improved_prompt != prompt  # Should be different
    assert len(result.transformations) >= 1


def test_clarity_rule_metadata():
    """
    Test that the rule's metadata is correctly defined.
    """
    rule = ClarityRule()
    metadata = rule.metadata
    assert metadata["name"] == "Clarity Enhancement Rule"
    assert metadata["type"] == "Fundamental"
