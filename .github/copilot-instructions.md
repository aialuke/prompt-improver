# GitHub Copilot Instructions for Prompt Improver Project

## Project Overview

This is the **Adaptive Prompt Enhancement System (APES)** - an intelligent prompt optimization system with ML-driven rule learning, built as a Python-based web service with FastAPI, PostgreSQL, Redis, and advanced observability.

### COPILOT_OPTIMIZATION: Claude Sonnet 4 Integration Patterns

**Deep Reasoning Activation**: Use `ARCHITECTURAL_CONTEXT` comments for complex design decisions
**Agent Mode Triggers**: Implement `TOOL_COORDINATION` patterns for MCP server integration  
**Vision Capabilities**: Leverage diagram analysis for system architecture understanding
**Performance Context**: Reference 343x CacheFactory performance gains and 114x AnalyticsServiceFacade improvements

### Standard Development Flow

1. **Development Environment**: Use containerized development with Docker Compose for services
2. **Testing Strategy**: Comprehensive testing with pytest, coverage reporting, and performance benchmarks
3. **Database Operations**: All database operations use async PostgreSQL with proper connection pooling
4. **Caching Strategy**: Redis-based caching with intelligent TTL management and cache invalidation
5. **Observability**: Full OpenTelemetry integration with metrics, tracing, and structured logging
6. **Code Quality**: Automated formatting with Ruff, type checking with Pyright, and pre-commit hooks

## Technology Stack & Architecture

### Core Framework
- **Backend**: FastAPI with async/await patterns
- **Database**: PostgreSQL 15+ with asyncpg driver and SQLModel ORM
- **Caching**: Redis 7+ with coredis async client
- **Authentication**: JWT-based with secure token management
- **API**: RESTful with OpenAPI documentation and WebSocket support

### ML & AI Stack
- **ML Libraries**: scikit-learn, pandas, numpy, scipy, optuna for hyperparameter optimization
- **NLP Processing**: NLTK, textstat for text analysis and metrics
- **ML Tracking**: MLflow for experiment tracking and model versioning
- **Advanced ML**: HDBSCAN for clustering, mlxtend for ML extensions

### Observability & Monitoring
- **Tracing**: OpenTelemetry with OTLP exporters
- **Metrics**: Prometheus-compatible metrics collection
- **Logging**: Structured logging with correlation IDs
- **Health Checks**: Comprehensive health monitoring for all services

### Development Tools
- **Code Quality**: Ruff (formatting & linting), Pyright (type checking)
- **Testing**: pytest with async support, coverage reporting, benchmark testing
- **Containerization**: Docker with multi-stage builds and development containers
- **CI/CD**: GitHub Actions with automated testing and deployment

## Project Structure

```
src/prompt_improver/           # Main application code
├── api/                       # FastAPI routes and middleware
├── services/                  # Business logic services
│   ├── cache/                 # Redis caching implementations
│   ├── database/              # Database services and repositories
│   └── ml/                    # Machine learning components
├── models/                    # SQLModel database models
├── mcp_server/               # MCP server for tool integration
└── utils/                    # Shared utilities and helpers

tests/                        # Test suite
├── unit/                     # Unit tests
├── integration/              # Integration tests
└── performance/              # Performance benchmarks

docs/                         # Documentation
├── architecture/             # Architecture decisions and patterns
├── user/                     # User guides and setup instructions
└── api/                      # API documentation

config/                       # Configuration files
database/                     # Database schema and migrations
scripts/                      # Development and deployment scripts
monitoring/                   # Observability configurations
```

## Coding Guidelines & Standards

### Python Code Standards
- **Python Version**: 3.11+ required
- **Type Hints**: Mandatory for all functions, methods, and class attributes
- **Async Patterns**: Use async/await for all I/O operations
- **Error Handling**: Comprehensive exception handling with proper error types
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Import Organization**: Follow isort standards with proper grouping

### Database Guidelines
- **Connection Management**: Always use connection pooling with proper cleanup
- **Query Patterns**: Use parameterized queries to prevent SQL injection
- **Transactions**: Wrap related operations in database transactions
- **Migration Strategy**: Use Alembic for schema migrations
- **Performance**: Include proper indexing and query optimization

### API Development Standards
- **Response Format**: Consistent JSON responses with proper HTTP status codes
- **Validation**: Use Pydantic models for request/response validation
- **Error Responses**: Standardized error response format with correlation IDs
- **Rate Limiting**: Implement appropriate rate limiting for API endpoints
- **Documentation**: Comprehensive OpenAPI documentation with examples

### Testing Requirements
- **Coverage**: Maintain >90% test coverage for all modules
- **Test Types**: Unit tests for business logic, integration tests for APIs
- **Performance Tests**: Benchmark critical paths and database operations
- **Fixtures**: Use pytest fixtures for test data and mock objects
- **Async Testing**: Use pytest-asyncio for testing async code

### Security Practices
- **Authentication**: JWT tokens with proper expiration and refresh handling
- **Input Validation**: Validate and sanitize all user inputs
- **Secret Management**: Use environment variables for sensitive configuration
- **CORS**: Proper CORS configuration for web API access
- **Rate Limiting**: Protection against abuse and DDoS attacks

## Error Detection & Code Quality

### Primary Error Detection Strategy
- **VS Code Native Tools**: Use `get_errors` tool as primary error detection method - shows exactly what developers see in VS Code
- **Task-Based Analysis**: Leverage configured tasks for comprehensive checking (pyright, ruff)
- **Real-Time Validation**: Get immediate feedback matching the developer's IDE experience

### Error Detection Workflow
```python
# Primary error checking - matches VS Code IDE exactly
get_errors(filePaths=["/path/to/file.py"])

# Use configured tasks for comprehensive analysis
run_task(id="shell: Type Check (pyright)", workspaceFolder=workspace_path)
run_task(id="shell: Lint Code", workspaceFolder=workspace_path)
run_task(id="shell: Run Tests with Coverage", workspaceFolder=workspace_path)
```

### Tool Integration Patterns
- **Error Analysis**: Always use `get_errors` before suggesting fixes
- **Performance Tools**: Leverage observability tools for performance monitoring
- **GitHub Tools**: Automate repository operations and issue management

## Development Workflow

### Environment Setup
```bash
# Use development container or manual setup
docker-compose up -d postgres redis  # Start services
python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,test,docs,security]"
pre-commit install
```

### Code Quality Checks
```bash
ruff format .                 # Format code
ruff check . --fix           # Lint and fix issues
pyright                      # Type checking
pytest --cov=src            # Run tests with coverage
```

### Database Operations
```bash
# Database migrations
alembic upgrade head         # Apply migrations
alembic revision --autogenerate -m "description"  # Create migration

# Development data
psql -d prompt_improver -f database/rule_seeds.sql  # Load test data
```

### Performance Testing
```bash
pytest tests/performance/ --benchmark-only  # Run benchmarks
k6 run tests/load/api_load_test.js          # Load testing
```

## File-Specific Guidelines

### API Routes (`src/prompt_improver/api/`)
- Always include proper HTTP status codes and error handling
- Use dependency injection for database connections and authentication
- Implement proper request/response validation with Pydantic models
- Include comprehensive logging with correlation IDs

### Database Models (`src/prompt_improver/models/`)
- Use SQLModel for ORM models with proper type annotations
- Include proper indexes for performance-critical queries
- Implement proper relationships with foreign key constraints
- Add validation constraints at the database level

### Services (`src/prompt_improver/services/`)
- Follow dependency injection patterns for testability
- Implement proper async patterns for I/O operations
- Include comprehensive error handling and logging
- Design for horizontal scalability and statelessness

### Cache Implementation (`src/prompt_improver/services/cache/`)
- Use Redis with proper TTL management and cache invalidation
- Implement cache warming strategies for critical data
- Handle cache failures gracefully with fallback to database
- Monitor cache hit rates and performance metrics

### ML Components (`src/prompt_improver/services/ml/`)
- Use MLflow for experiment tracking and model versioning
- Implement proper feature engineering with validation
- Include model performance monitoring and drift detection
- Design for model retraining and deployment workflows

## Common Patterns & Best Practices

### Async Database Operations
```python
async with database.get_session() as session:
    async with session.begin():
        result = await session.execute(query)
        return result.mappings().all()
```

### Error Handling Pattern
```python
try:
    result = await service_operation()
    return {"status": "success", "data": result}
except ServiceError as e:
    logger.error(f"Service error: {e}", extra={"correlation_id": correlation_id})
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.exception("Unexpected error", extra={"correlation_id": correlation_id})
    raise HTTPException(status_code=500, detail="Internal server error")
```

### Cache Pattern
```python
cache_key = f"prompt_analysis:{prompt_hash}"
cached_result = await cache.get(cache_key)
if cached_result:
    return cached_result

result = await expensive_operation()
await cache.set(cache_key, result, ttl=3600)
return result
```

### Observability Pattern
```python
with tracer.start_as_current_span("operation_name") as span:
    span.set_attribute("operation.type", "data_processing")
    span.set_attribute("input.size", len(data))
    
    try:
        result = await process_data(data)
        span.set_attribute("result.count", len(result))
        return result
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
```

## Performance Considerations

### Database Performance
- Use connection pooling with appropriate pool sizes
- Implement proper indexing strategies for query patterns
- Use EXPLAIN ANALYZE for query optimization
- Monitor slow queries and optimize regularly

### Cache Performance
- Implement cache warming for frequently accessed data
- Use appropriate TTL values based on data volatility
- Monitor cache hit rates and adjust strategies accordingly
- Implement cache invalidation strategies for data consistency

### API Performance
- Use async patterns throughout the application stack
- Implement proper rate limiting and request throttling
- Use connection pooling for external service calls
- Monitor response times and implement appropriate timeouts

## Security Considerations

### Authentication & Authorization
- Use JWT tokens with appropriate expiration times
- Implement proper token refresh mechanisms
- Validate all authentication tokens and permissions
- Log authentication attempts and failures

### Input Validation
- Validate all user inputs using Pydantic models
- Sanitize inputs to prevent injection attacks
- Implement proper rate limiting to prevent abuse
- Use parameterized queries for database operations

### Configuration Security
- Store sensitive configuration in environment variables
- Use secrets management for production deployments
- Implement proper CORS policies for web access
- Regular security audits and dependency updates

## Testing Guidelines

### Unit Testing
- Test business logic in isolation using mocks
- Achieve >90% coverage for all service modules
- Use pytest fixtures for consistent test data
- Test both success and failure scenarios

### Integration Testing
- Test API endpoints with real database connections
- Verify cache behavior and invalidation strategies
- Test authentication and authorization flows
- Validate database migrations and schema changes

### Performance Testing
- Benchmark critical code paths and database queries
- Load test API endpoints under realistic conditions
- Monitor memory usage and resource consumption
- Test cache performance under various load conditions

## Deployment & Operations

### Container Configuration
- Use multi-stage Docker builds for optimization
- Implement proper health checks for all services
- Configure appropriate resource limits and requests
- Use non-root users for security

### Environment Configuration
- Separate configuration for development, staging, and production
- Use environment-specific configuration files
- Implement proper logging levels and formats
- Configure observability and monitoring tools

### Monitoring & Alerting
- Monitor application metrics, database performance, and cache hit rates
- Set up alerts for error rates, response times, and resource usage
- Implement comprehensive health checks for all services
- Use distributed tracing for request flow analysis

This project emphasizes production-ready code with comprehensive testing, observability, and security. Always consider performance, scalability, and maintainability in your implementations.
