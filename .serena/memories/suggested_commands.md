# Suggested Commands for APES Development

## Testing Commands
```bash
# Full test suite with integration focus
pytest tests/ -v

# Integration tests (preferred for development)
pytest tests/integration/ -v

# Real behavior testing
pytest tests/real_behavior/ -v

# Performance validation
pytest -m performance -v

# Specific categories
pytest tests/integration/automl/ -v  # AutoML tests
pytest tests/database/ -v           # Database tests
pytest -m unit -v                   # Pure unit tests
```

## Quality Assurance
```bash
# Code quality checks
ruff check src/
ruff format src/

# Type checking
pyright src/

# Import linting (architecture enforcement)
import-linter --config pyproject.toml

# Performance benchmarks
pytest tests/benchmarks/ -v
```

## Development Workflow
```bash
# Database setup via Docker
docker-compose up -d postgres

# Run MCP server (if needed)
./start_mcp_server.sh

# CLI interface
apes --help

# Development server
uvicorn prompt_improver.api.app:app --reload
```

## Architecture Validation
```bash
# Validate clean architecture boundaries
import-linter --config pyproject.toml

# Check for circular imports
python scripts/analyze_dependencies.py

# Performance validation
python scripts/run_performance_validation.py
```