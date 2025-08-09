# AGENT.md - APES Universal Prompt Testing Framework

## Build/Test/Lint Commands
- **Run tests**: `scripts/run_tests.sh` (unified test runner), `pytest tests/` (direct)
- **Single test**: `pytest tests/path/to/test_file.py::test_function_name -v`
- **Performance tests**: `pytest -m performance -v`
- **Coverage**: `pytest --cov=src --cov-report=html`
- **Lint**: `ruff check src tests`, `ruff format src tests`
- **Type check**: `pyright`
- **CLI tool**: `apes` (train/status/why5/stop commands)

## Scripts Directory (Cleaned 2025)
### 📊 Testing & Validation
- `scripts/run_tests.sh` - Unified test runner (replaces 7 redundant test scripts)
- `scripts/validate_mcp_protocol.py` - MCP protocol validation
- `scripts/validate_ml_contracts.py` - ML contract validation

### 🏗️ Development & Setup  
- `scripts/setup_development.sh` - Complete dev environment setup
- `scripts/setup_test_infrastructure.sh` - Test infrastructure setup
- `scripts/dev-server.sh` - Enhanced development server with HMR

### 🔍 Architecture & Analysis
- `scripts/analyze_dependencies.py` - Dependency analysis and visualization
- `scripts/architectural_compliance.py` - Architecture compliance checking
- `scripts/circular_dependency_analyzer.py` - Circular dependency detection
- `scripts/import_analyzer.py` - Import usage analysis

### ⚡ Performance & Monitoring
- `scripts/capture_baselines.py` - Performance baseline measurement
- `scripts/compare_baselines.py` - Baseline comparison and analysis
- `scripts/check_performance_regression.py` - Performance regression detection
- `scripts/mcp_health_monitor.py` - MCP server health monitoring
- `scripts/k6_load_test.js` - K6 load testing configuration

### 🛠️ Utilities & Tools
- `scripts/generate_docs.py` - Automated documentation generation
- `scripts/gradual_tightening.py` - Gradual code quality improvement
- `scripts/install_production_tools.py` - Production tools installation
- `scripts/integrate_business_metrics.py` - Business metrics integration
- `scripts/cleanup_nltk_resources.py` - NLTK resource optimization
- `scripts/create_feature_flags.sh` - Feature flag setup
- `scripts/debug_ml_components.py` - ML component debugging
- `scripts/init_memory.sh` - MCP memory initialization
- `scripts/technical_debt_monitor.py` - Technical debt tracking
- `scripts/verify_unused_dependencies.py` - Dependency cleanup

## Architecture & Structure
- **Clean Architecture**: Core → Domain/ML/RuleEngine → Application → Infrastructure → Presentation
- **Main entry**: CLI via `apes` command (pyproject.toml script), module via `src/prompt_improver/__main__.py`
- **Database**: PostgreSQL (AsyncPG), schema in `database/schema.sql`, auto-initialized via Docker
- **Cache**: Redis with graceful fallback, real-time analytics via WebSockets
- **ML Stack**: AutoML orchestrator, A/B testing service, MLflow experiments, Optuna optimization
- **Testing**: 2025 best practices - No mocks, real behavior testing, 60-70% integration tests

## Code Style Guidelines
- **Imports**: First-party (`prompt_improver`), ML section (`numpy`, `pandas`, `scikit-learn`), absolute imports only
- **Types**: Python 3.11+, comprehensive type hints, Pyright strict mode
- **Formatting**: Ruff (88 chars), Google docstrings, f-strings preferred
- **Async**: AsyncPG for DB, asyncio patterns, EnhancedBackgroundTaskManager for services
- **ML conventions**: Higher complexity limits (max-args=8, max-statements=60), print debugging allowed
- **Errors**: Comprehensive error handling, structured logging, no bare except clauses

## Special Rules (from CLAUDE.md)
- Search existing solutions before creating new code: `rg "pattern" --type py`
- Apply YAGNI/KISS/DRY, extend existing when ≤3 params/≤50 lines/same domain
- Use parallel tool execution for 3-10x performance improvement
- Delegate to specialized agents for domain expertise (code-reviewer, ml-orchestrator, etc.)
- Use thinking blocks for complex reasoning (54% performance improvement)
