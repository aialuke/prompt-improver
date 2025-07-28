# APES - Universal Prompt Testing Framework

A comprehensive, autonomous testing framework that improves prompt engineering by analyzing any codebase, generating relevant tests, and optimizing based on results.

## 🎯 Overview

Transform any prompt improvement tool into an intelligent system that continuously learns and improves across any project type, tech stack, or domain.

### Core Value Proposition

- **Universal Context Analysis** - Understands any codebase/tech stack
- **Adaptive Test Generation** - Creates relevant prompts for detected context
- **Multi-Dimensional Evaluation** - Measures improvement quality comprehensively
- **Pattern Learning Engine** - Extracts insights and optimizes rules
- **Autonomous Orchestration** - Manages entire workflow without human intervention

## 🧪 Integration Testing Standards (2025)

This project follows **2025 Integration Testing Best Practices** with comprehensive real behavior testing and modern test pyramid approaches.

### Key Principles Implemented

#### 1. **No Mocks Policy**

- Uses real services in sandboxed environments for authentic behavior
- Mocks drift from reality and miss integration issues that occur in production
- Real-environment testing provides higher confidence than mock-based approaches
- **Reference**: [Signadot 2025 Guidelines](https://www.signadot.com/blog/why-mocks-fail-real-environment-testing-for-microservices)

#### 2. **Realistic Database Fixtures**

- PostgreSQL with production-like data volumes and patterns
- Proper database isolation through transactions
- Real database constraints and relationships testing
- **Reference**: [Opkey Integration Testing Guide](https://www.opkey.com/blog/integration-testing-a-comprehensive-guide-with-best-practices)

#### 3. **Modern Test Pyramid (2025)**

```
    /\      E2E Tests (5-10%)
   /  \     Focus on critical user journeys
  /____\
 /      \   Integration Tests (60-70%)
/        \  Contract & API testing
\________/
\        /  Unit Tests (20-30%)
 \______/   Business logic validation
```

Modern distributed systems require more integration testing than traditional pyramids suggested.
**Reference**: [Full Scale Modern Test Pyramid](https://fullscale.io/blog/modern-test-pyramid-guide/)

#### 4. **Network Isolation Lightweight Patches**

For external dependencies, we implement minimal patches that maintain test authenticity:

- **Redis**: Graceful fallback to in-memory for CI environments
- **Timeouts**: Optimized for testing (10s vs production 300s)
- **Trial Counts**: Reduced for faster execution (3 vs production 100+)
- **Storage**: PostgreSQL containers for test isolation

**Rationale**: These patches provide network isolation without compromising test authenticity. They maintain real behavior while preventing external dependencies from causing test failures in CI environments.

#### 5. **Contract Testing**

- API contracts between services are validated
- Breaking changes detected early in development
- Consumer-driven contract testing with Pact.io
- **Reference**: [Ambassador Contract Testing](https://www.getambassador.io/blog/contract-testing-microservices-strategy)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 13+
- Redis 6+ (optional, graceful fallback available)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd prompt-improver

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up database
createdb prompt_improver
python -m alembic upgrade head

# Run tests to verify installation
pytest tests/ -v
```

### Basic Usage

```bash
# Run integration tests (recommended for development)
pytest tests/integration/ -v

# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/integration/automl/ -v  # AutoML integration tests
pytest tests/integration/services/ -v  # Service layer tests
pytest tests/unit/ -v  # Unit tests

# Performance testing
pytest -m performance -v
```

## 🏗️ Architecture

The framework consists of 6 core phases:

1. **Context Analysis** - Universal codebase understanding
2. **Test Generation** - Adaptive prompt creation
3. **Test Execution** - Automated improvement testing
4. **Multi-Dimensional Evaluation** - Comprehensive quality assessment
5. **Pattern Analysis & Learning** - Insight extraction
6. **Rule Optimization** - Automated rule improvements

### Key Components

- **AutoML Orchestrator** - Manages optimization experiments
- **Real-Time Analytics** - WebSocket-based monitoring
- **A/B Testing Service** - Experiment management
- **Database Layer** - PostgreSQL with async support
- **MCP Integration** - Model Context Protocol support

## 📊 Testing Strategy

### Integration Test Categories

#### AutoML Integration Tests

- End-to-end optimization workflows
- Real service interactions
- Database persistence validation
- Performance benchmarking

#### Service Layer Tests

- Business logic validation
- Database integration
- External API interactions
- Error handling scenarios

#### CLI Tests

- Command-line interface functionality
- Argument parsing validation
- Output formatting verification
- Configuration management

### Performance Requirements

- **Unit Tests**: < 100ms each
- **Integration Tests**: < 5 seconds each
- **Full Test Suite**: < 5 minutes total
- **API Response Time**: < 200ms

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql+psycopg://user:pass@localhost/prompt_improver

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# Testing
PYTEST_TIMEOUT=300
INTEGRATION_TEST_TIMEOUT=30
```

### Test Configuration

```yaml
# pytest.ini
[tool:pytest]
markers =
    integration: Integration tests with real components
    performance: Performance validation tests
    slow: Tests taking > 1 second
    unit: Pure unit tests
    asyncio: Async test cases

timeout = 300
testpaths = tests/
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

## 📈 Monitoring & Observability

### Real-Time Analytics

- WebSocket-based experiment monitoring
- Metrics calculation and visualization
- Alert generation for significant results
- Performance tracking and optimization

### Health Checks

- Database connection monitoring
- Redis availability checking
- Service discovery validation
- Graceful degradation testing

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Implement changes following 2025 standards
4. Add comprehensive tests (integration focus)
5. Ensure all tests pass
6. Submit pull request

### Test Requirements

- All new features must have integration tests
- Maintain minimum 85% code coverage
- Follow 2025 integration testing standards
- Document network isolation patches
- Validate performance requirements

### Code Quality

- Type hints for all functions
- Comprehensive docstrings
- Error handling and logging
- Security best practices
- Performance optimization

## 📚 Documentation

- [Test Suite Documentation](tests/README.md) - Complete 2025 testing standards
- [Test Suite Documentation](tests/README.md) - Detailed testing guide
- [API Documentation](docs/api/) - Service interfaces
- [Architecture Guide](docs/architecture/) - System design
- [Performance Guide](docs/performance/) - Optimization strategies

## 🆘 Support

- **Issues**: Report bugs and feature requests on GitHub Issues
- **Documentation**: Complete guides available in the `docs/` directory
- **Testing**: Comprehensive test suite with 2025 standards compliance
- **Performance**: Benchmarking and optimization guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

_Built with 2025 Integration Testing Standards - Real services, authentic behavior, comprehensive coverage._
