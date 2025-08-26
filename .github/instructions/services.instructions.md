---
applyTo: "**/services/**/*.py"
---

## Service Layer Development Guidelines

### Error Detection & Code Quality
- **Primary Error Detection**: Use `get_errors` tool to identify issues before implementing fixes
- **Type Safety**: Ensure proper type annotations and use pyright for validation
- **Code Quality**: Apply ruff formatting and linting standards
- **Testing**: Validate service logic with comprehensive test coverage

### Service Architecture Patterns
- Implement dependency injection for all external dependencies
- Use async/await patterns for all I/O operations
- Design services to be stateless and horizontally scalable
- Separate business logic from data access and external integrations

### Required Service Pattern
```python
class PromptAnalysisService:
    """Service for analyzing and improving prompts using ML models."""
    
    def __init__(
        self,
        database_repository: PromptRepository,
        cache_service: CacheService,
        ml_model_service: MLModelService,
        metrics_collector: MetricsCollector,
        logger: Logger
    ):
        self.repository = database_repository
        self.cache = cache_service
        self.ml_models = ml_model_service
        self.metrics = metrics_collector
        self.logger = logger
    
    async def analyze_prompt(
        self,
        prompt_text: str,
        user_context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> PromptAnalysisResult:
        """
        Analyze prompt and provide improvement suggestions.
        
        Args:
            prompt_text: The prompt text to analyze
            user_context: Optional user context for personalization
            correlation_id: Request correlation ID for tracing
            
        Returns:
            Comprehensive analysis with scores and suggestions
            
        Raises:
            ServiceError: For business logic violations
            ValidationError: For invalid input data
        """
```

### Error Handling Strategy
- Create custom exception classes for different error types
- Always include context in error messages
- Log errors with appropriate levels (ERROR for exceptions, WARNING for business rule violations)
- Include correlation IDs in all log messages

### Caching Implementation
- Cache expensive computations and external API calls
- Use appropriate TTL values based on data volatility
- Implement cache invalidation strategies
- Handle cache failures gracefully with database fallback

### Database Operations
- Use repository pattern for data access
- Always use async database operations
- Wrap related operations in transactions
- Implement proper connection management and cleanup

### Observability Requirements
- Add OpenTelemetry spans for all major operations
- Include relevant attributes in spans (operation type, input size, etc.)
- Collect custom metrics for business operations
- Log performance metrics for optimization

### ML Integration Patterns
- Use MLflow for model loading and version management
- Implement model prediction caching
- Handle model failures gracefully
- Monitor model performance and drift

### Required Imports
```python
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from src.prompt_improver.utils.exceptions import ServiceError, ValidationError
from src.prompt_improver.utils.logging import get_logger
from src.prompt_improver.utils.metrics import MetricsCollector
```

### Performance Guidelines
- Implement connection pooling for external services
- Use batch operations where possible
- Monitor service response times
- Implement circuit breaker patterns for external dependencies

### Testing Requirements
- Mock all external dependencies in unit tests
- Test both success and failure scenarios
- Use pytest fixtures for test data setup
- Achieve >95% test coverage for service logic
