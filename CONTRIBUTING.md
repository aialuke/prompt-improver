# Contributing to APES

Thank you for your interest in contributing to the APES project! This guide outlines our development process, testing standards, and contribution guidelines.

## 🎯 Development Philosophy

We follow **2025 Integration Testing Best Practices** which emphasize:

- Real services over mocks for authentic behavior
- Comprehensive integration testing coverage
- Network isolation through lightweight patches
- Contract testing for API validation
- Performance-oriented test design

## 🧪 Testing Standards (2025)

### Core Principles

#### 1. **No Mocks Policy**

- Use real services in sandboxed environments
- Mock only external APIs and services outside our control
- Prefer in-memory databases over file-based mocks
- **Rationale**: Mocks drift from reality and miss integration issues

#### 2. **Integration-First Testing**

Following the modern test pyramid (2025):

```
Integration Tests (60-70%) - Primary focus
Unit Tests (20-30%) - Business logic validation
E2E Tests (5-10%) - Critical user journeys
```

#### 3. **Network Isolation Lightweight Patches**

For external dependencies, implement minimal patches:

- **Redis**: Graceful fallback to in-memory for CI
- **Timeouts**: Reduce for faster test execution
- **Storage**: Use PostgreSQL containers for test isolation
- **Rate Limits**: Bypass for testing environments

**Example Implementation**:

```python
# === 2025 Network Isolation Lightweight Patch ===
# This implements graceful fallback to maintain test isolation
redis_client = None
try:
    redis_client = redis.from_url(redis_url)
    await redis_client.ping()
except Exception:
    # Graceful fallback to in-memory for CI environments
    redis_client = None
```

## 📋 Contribution Process

### 1. **Setup Development Environment**

```bash
# Clone and setup
git clone https://github.com/your-username/prompt-improver.git
cd prompt-improver

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup database
createdb prompt_improver_dev
python -m alembic upgrade head

# Verify installation
pytest tests/integration/ -v
```

### 2. **Create Feature Branch**

```bash
git checkout -b feature/your-feature-name
```

### 3. **Development Guidelines**

#### Code Quality Standards

- **Type Hints**: All functions must have type hints
- **Docstrings**: Use Google-style docstrings
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging with context
- **Security**: Input validation and sanitization

#### Testing Requirements

- **Integration Tests**: Required for all new features
- **Unit Tests**: Required for business logic
- **Performance Tests**: Required for time-critical features
- **Documentation**: Update relevant documentation

### 4. **Testing Your Changes**

#### Run Integration Tests

```bash
# All integration tests
pytest tests/integration/ -v

# Specific component tests
pytest tests/integration/automl/ -v
pytest tests/integration/services/ -v
pytest tests/integration/cli/ -v
```

#### Run Performance Tests

```bash
# Performance validation
pytest -m performance -v

# Check test execution time
pytest --durations=10
```

#### Coverage Requirements

```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-fail-under=85

# View coverage
open htmlcov/index.html
```

## 🔧 Writing Tests

### Integration Test Template

```python
"""
Integration test following 2025 standards.

=== 2025 Integration Testing Standards Compliance ===

1. **No Mocks Policy**: Uses real services in sandboxed environments
2. **Realistic Database Fixtures**: PostgreSQL with production-like data
3. **Network Isolation Patches**: Graceful fallback for external dependencies
4. **Contract Testing**: Validates API interactions
5. **Performance Requirements**: < 5 seconds execution time
"""

import pytest
from src.your_module import YourService

class TestYourServiceIntegration:
    """Integration tests for YourService following 2025 standards."""

    @pytest.fixture
    async def service(self, test_db_session):
        """Real service instance with database connection."""
        return YourService(test_db_session)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_workflow(self, service):
        """Test complete workflow with real components."""
        # === 2025 Standard: Test real behavior, not mocks ===
        result = await service.process_data(test_input)

        # Validate real database persistence
        assert result.id is not None
        assert result.status == "processed"

        # Verify database state
        stored_result = await service.get_result(result.id)
        assert stored_result == result
```

### Network Isolation Patch Template

```python
# === 2025 Network Isolation Lightweight Patch ===
# Rationale: Provides test isolation without compromising authenticity
async def setup_external_dependency():
    """Setup external dependency with graceful fallback."""
    try:
        # Attempt real connection
        client = await ExternalService.connect()
        await client.ping()
        return client
    except Exception:
        # Fallback to in-memory for CI environments
        return MockExternalService()  # Minimal mock, real behavior
```

## 🚀 Performance Guidelines

### Performance Requirements

- **Unit Tests**: < 100ms each
- **Integration Tests**: < 5 seconds each
- **Full Test Suite**: < 5 minutes total
- **API Endpoints**: < 200ms response time

### Performance Testing

```python
@pytest.mark.performance
@pytest.mark.asyncio
async def test_api_response_time():
    """Validate API response time requirement."""
    start_time = time.time()

    result = await api_endpoint.process_request(data)

    response_time = (time.time() - start_time) * 1000
    assert response_time < 200  # < 200ms requirement
    assert result.status == "success"
```

## 📊 Code Review Process

### Review Checklist

#### Testing Standards

- [ ] Integration tests cover main functionality
- [ ] Network isolation patches documented
- [ ] Performance requirements validated
- [ ] 2025 standards compliance verified
- [ ] Test coverage > 85%

#### Code Quality

- [ ] Type hints on all functions
- [ ] Comprehensive error handling
- [ ] Security best practices followed
- [ ] Performance optimizations applied
- [ ] Documentation updated

#### Integration Compliance

- [ ] Real services used where possible
- [ ] External dependencies isolated
- [ ] Database transactions for test isolation
- [ ] Contract testing for API changes
- [ ] Graceful degradation tested

## 🔍 Debugging Tests

### Common Issues

#### 1. **Test Isolation Problems**

```bash
# Run tests in isolation
pytest tests/integration/test_specific.py::TestClass::test_method -v

# Clear test database
pytest --create-db --database-url=test_db_url
```

#### 2. **Performance Issues**

```bash
# Profile test execution
pytest --profile --profile-svg

# Check slow tests
pytest --durations=0 | grep -E "^[0-9].*s"
```

#### 3. **Database Connection Issues**

```bash
# Check database connection
pytest tests/integration/test_db_connection.py -v

# Reset test database
dropdb prompt_improver_test
createdb prompt_improver_test
python -m alembic upgrade head
```

## 📚 Documentation Standards

### Code Documentation

- **Docstrings**: Google-style with examples
- **Type Hints**: Complete type annotations
- **Comments**: Explain complex logic and 2025 standards compliance
- **README Updates**: Document new features and changes

### Test Documentation

- **Test Names**: Descriptive and behavior-focused
- **Comments**: Explain 2025 standards compliance
- **Fixtures**: Document purpose and scope
- **Performance**: Document timing requirements

### Example Documentation

```python
def process_user_data(
    user_id: str,
    data: Dict[str, Any],
    session: AsyncSession
) -> ProcessingResult:
    """Process user data with validation and persistence.

    Args:
        user_id: Unique identifier for the user
        data: Raw data dictionary to process
        session: Database session for persistence

    Returns:
        ProcessingResult with status and processed data

    Raises:
        ValidationError: If data validation fails
        DatabaseError: If persistence fails

    Example:
        >>> result = await process_user_data(
        ...     user_id="user123",
        ...     data={"name": "John", "age": 30},
        ...     session=db_session
        ... )
        >>> assert result.status == "success"
    """
```

## 🤝 Community Guidelines

### Communication

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions
- **Pull Requests**: Follow PR template and checklist
- **Code Reviews**: Be constructive and helpful

### Mentorship

- **New Contributors**: Pair with experienced developers
- **Documentation**: Maintain comprehensive guides
- **Best Practices**: Share knowledge through examples
- **Standards**: Help others understand 2025 testing standards

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

## 🆘 Getting Help

- **Documentation**: Check [docs/](docs/) directory
- **Issues**: Search existing issues first
- **Testing**: Review [tests/README.md](tests/README.md)
- **Standards**: Review [2025_integration_testing_guidelines.md](2025_integration_testing_guidelines.md)
- **Performance**: Check performance testing examples

Thank you for contributing to APES and helping maintain our high testing standards!

---

_Built with 2025 Integration Testing Standards - Real services, authentic behavior, comprehensive coverage._
