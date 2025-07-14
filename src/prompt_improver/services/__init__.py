"""Service layer for APES business logic"""

# Use lazy imports to avoid circular dependency issues
__all__ = ["AnalyticsService", "PromptImprovementService", "AprioriAnalyzer", "AdvancedPatternDiscovery", "AutoMLOrchestrator"]


def __getattr__(name):
    if name == "PromptImprovementService":
        from .prompt_improvement import PromptImprovementService

        return PromptImprovementService
    if name == "AnalyticsService":
        from .analytics import AnalyticsService

        return AnalyticsService
    if name == "AprioriAnalyzer":
        from .apriori_analyzer import AprioriAnalyzer

        return AprioriAnalyzer
    if name == "AdvancedPatternDiscovery":
        from .advanced_pattern_discovery import AdvancedPatternDiscovery

        return AdvancedPatternDiscovery
    if name == "AutoMLOrchestrator":
        from ..automl.orchestrator import AutoMLOrchestrator

        return AutoMLOrchestrator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
