---
applyTo: "**/api/**/*.py"
---

## API Route Development Guidelines

### FastAPI Best Practices
- Use dependency injection for database sessions, authentication, and shared services
- Implement proper HTTP status codes (200, 201, 400, 401, 403, 404, 500)
- Include comprehensive request/response models using Pydantic
- Add proper OpenAPI documentation with examples and descriptions

### Required Response Pattern
```python
@router.post("/prompts/analyze", response_model=PromptAnalysisResponse)
async def analyze_prompt(
    request: PromptAnalysisRequest,
    db: AsyncSession = Depends(get_database_session),
    current_user: User = Depends(get_current_user),
    correlation_id: str = Depends(get_correlation_id)
) -> PromptAnalysisResponse:
    """
    Analyze prompt quality and provide improvement suggestions.
    
    Args:
        request: Prompt analysis request with text and optional context
        db: Database session for data persistence
        current_user: Authenticated user making the request
        correlation_id: Request correlation ID for tracing
        
    Returns:
        Detailed analysis with scores, suggestions, and metadata
        
    Raises:
        HTTPException: 400 for invalid input, 401 for authentication failure
    """
```

### Error Handling Requirements
- Always wrap operations in try-catch blocks
- Use appropriate HTTPException status codes
- Include correlation IDs in error logging
- Provide meaningful error messages without exposing internal details

### Authentication & Authorization
- All endpoints except health checks require authentication
- Use role-based authorization where appropriate
- Log authentication attempts and failures
- Implement rate limiting for sensitive operations

### Validation & Security
- Validate all inputs using Pydantic models
- Sanitize user inputs to prevent injection attacks
- Use parameterized database queries
- Implement proper CORS headers

### Performance Considerations
- Use async patterns throughout
- Implement connection pooling for database operations
- Add appropriate caching for frequently accessed data
- Monitor and log response times

### Required Dependencies
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, validator
from src.prompt_improver.dependencies import (
    get_database_session,
    get_current_user,
    get_correlation_id
)
from src.prompt_improver.utils.logging import get_logger
```
