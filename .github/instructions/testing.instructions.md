---
applyTo: "**/tests/**/*.py"
---

## Testing Guidelines

### Error Detection & Code Quality
- **Primary Error Detection**: Use `get_errors` tool for VS Code-matching error analysis
- **Type Checking**: Use `run_task(id="shell: Type Check (pyright)")` for comprehensive type analysis
- **Linting**: Use `run_task(id="shell: Lint Code")` for code quality checks
- **Test Execution**: Use `run_task(id="shell: Run Tests with Coverage")` for test validation

### Test Structure & Organization
- Follow AAA pattern: Arrange, Act, Assert
- Use descriptive test names that explain the scenario being tested
- Group related tests in classes with clear naming conventions
- Separate unit tests, integration tests, and performance tests

### Required Test Pattern
```python
import pytest
from unittest.mock import AsyncMock, Mock, patch
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from src.prompt_improver.services.prompt_analysis import PromptAnalysisService
from src.prompt_improver.models.prompt import PromptAnalysisResult
from tests.fixtures.database import async_session, test_data
from tests.fixtures.auth import authenticated_user

class TestPromptAnalysisService:
    """Test suite for PromptAnalysisService."""
    
    @pytest.mark.asyncio
    async def test_analyze_prompt_success(
        self,
        prompt_analysis_service: PromptAnalysisService,
        sample_prompt_data: dict,
        mock_ml_model: AsyncMock
    ):
        """
        Test successful prompt analysis with valid input.
        
        Given: A valid prompt text and user context
        When: The analyze_prompt method is called
        Then: It should return a comprehensive analysis result
        """
        # Arrange
        prompt_text = "Create a responsive web component"
        expected_score = 0.85
        mock_ml_model.predict.return_value = {"quality_score": expected_score}
        
        # Act
        result = await prompt_analysis_service.analyze_prompt(
            prompt_text=prompt_text,
            user_context={"domain": "web_development"}
        )
        
        # Assert
        assert result.quality_score == expected_score
        assert result.prompt_text == prompt_text
        assert len(result.improvement_suggestions) > 0
        mock_ml_model.predict.assert_called_once()
```

### Async Testing Requirements
- Use `@pytest.mark.asyncio` for all async test functions
- Use `AsyncMock` for mocking async dependencies
- Properly await all async operations in tests
- Use `pytest-asyncio` for async test fixtures

### Fixture Usage Patterns
```python
@pytest.fixture
async def prompt_analysis_service(
    mock_database_repository: AsyncMock,
    mock_cache_service: AsyncMock,
    mock_ml_model_service: AsyncMock
) -> PromptAnalysisService:
    """Create PromptAnalysisService with mocked dependencies."""
    return PromptAnalysisService(
        database_repository=mock_database_repository,
        cache_service=mock_cache_service,
        ml_model_service=mock_ml_model_service,
        metrics_collector=Mock(),
        logger=Mock()
    )

@pytest.fixture
async def authenticated_client(
    async_client: AsyncClient,
    test_user: User
) -> AsyncClient:
    """Create an authenticated HTTP client."""
    token = create_access_token(data={"sub": str(test_user.id)})
    async_client.headers.update({"Authorization": f"Bearer {token}"})
    return async_client
```

### Database Testing Patterns
- Use transaction rollback for test isolation
- Create test data using factories or fixtures
- Test both successful operations and constraint violations
- Use separate test database for integration tests

### API Testing Requirements
```python
@pytest.mark.asyncio
async def test_analyze_prompt_endpoint_success(
    authenticated_client: AsyncClient,
    sample_prompt_request: dict
):
    """Test successful prompt analysis via API endpoint."""
    # Act
    response = await authenticated_client.post(
        "/api/v1/prompts/analyze",
        json=sample_prompt_request
    )
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "quality_score" in data
    assert "improvement_suggestions" in data
    assert data["quality_score"] >= 0.0
    assert data["quality_score"] <= 1.0
```

### Error Testing Patterns
- Test all expected error conditions
- Verify proper HTTP status codes for API errors
- Test error message content and structure
- Ensure sensitive information is not exposed in errors

### Performance Testing
```python
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_prompt_analysis_performance(
    benchmark,
    prompt_analysis_service: PromptAnalysisService
):
    """Benchmark prompt analysis performance."""
    
    async def analyze_prompt():
        return await prompt_analysis_service.analyze_prompt(
            prompt_text="Test prompt for performance analysis"
        )
    
    result = await benchmark.pedantic(analyze_prompt, rounds=10)
    assert result.quality_score is not None
```

### Mock Usage Guidelines
- Mock external dependencies (databases, APIs, ML models)
- Use `patch` for patching specific methods or functions
- Verify mock calls with proper arguments
- Reset mocks between tests to avoid state leakage

### Coverage Requirements
- Aim for >90% test coverage for all modules
- Test both happy path and error scenarios
- Include edge cases and boundary conditions
- Use coverage reports to identify untested code paths

### Integration Test Patterns
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_prompt_analysis_workflow(
    async_session: AsyncSession,
    authenticated_client: AsyncClient
):
    """Test complete prompt analysis workflow end-to-end."""
    # Create test data
    prompt_data = {
        "prompt_text": "Create a machine learning model",
        "context": {"domain": "data_science"}
    }
    
    # Submit analysis request
    response = await authenticated_client.post(
        "/api/v1/prompts/analyze",
        json=prompt_data
    )
    
    # Verify response
    assert response.status_code == 200
    analysis_id = response.json()["id"]
    
    # Verify data persistence
    result = await async_session.get(PromptAnalysisResult, analysis_id)
    assert result is not None
    assert result.prompt_text == prompt_data["prompt_text"]
```

### Test Data Management
- Use factories for creating test objects
- Keep test data minimal and focused
- Use parameterized tests for testing multiple scenarios
- Clean up test data after tests complete

### Required Test Imports
```python
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from uuid import uuid4
```
