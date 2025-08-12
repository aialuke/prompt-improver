"""Rule modules for prompt improvement system."""

from prompt_improver.rule_engine.rules.chain_of_thought import ChainOfThoughtRule
from prompt_improver.rule_engine.rules.clarity import ClarityRule
from prompt_improver.rule_engine.rules.few_shot_examples import FewShotExampleRule
from prompt_improver.rule_engine.rules.linguistic_quality_rule import (
    LinguisticQualityRule,
)
from prompt_improver.rule_engine.rules.role_based_prompting import (
    RoleBasedPromptingRule,
)
from prompt_improver.rule_engine.rules.specificity import SpecificityRule
from prompt_improver.rule_engine.rules.xml_structure_enhancement import XMLStructureRule

__all__ = [
    "ChainOfThoughtRule",
    "ClarityRule",
    "FewShotExampleRule",
    "LinguisticQualityRule",
    "RoleBasedPromptingRule",
    "SpecificityRule",
    "XMLStructureRule",
]
