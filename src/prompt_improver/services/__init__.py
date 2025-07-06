"""Service layer for APES business logic"""

# Use lazy imports to avoid circular dependency issues
__all__ = ["AnalyticsService", "PromptImprovementService"]


def __getattr__(name):
    if name == "PromptImprovementService":
        from .prompt_improvement import PromptImprovementService

        return PromptImprovementService
    if name == "AnalyticsService":
        from .analytics import AnalyticsService

        return AnalyticsService
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
