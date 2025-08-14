"""Prompt services package for prompt improvement operations.

This package implements the PromptServiceFacade pattern, decomposing the original
1,544-line prompt_improvement.py god object into focused services:

1. PromptAnalysisService - Analysis and improvement logic
2. RuleApplicationService - Rule execution and validation
3. ValidationService - Input validation and business rules
4. PromptServiceFacade - Unified coordinator

All services follow Clean Architecture principles with protocol-based dependency injection.
"""

from .facade import PromptServiceFacade
from .prompt_analysis_service import PromptAnalysisService
from .rule_application_service import RuleApplicationService
from .validation_service import ValidationService

__all__ = [
    "PromptServiceFacade",
    "PromptAnalysisService",
    "RuleApplicationService",
    "ValidationService",
]