"""
Service layer for APES business logic
"""

# Use lazy imports to avoid circular dependency issues
__all__ = ["PromptImprovementService", "AnalyticsService"]

def __getattr__(name):
    if name == "PromptImprovementService":
        from .prompt_improvement import PromptImprovementService
        return PromptImprovementService
    elif name == "AnalyticsService":
        from .analytics import AnalyticsService
        return AnalyticsService
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
