# AGENTS.md - Development Guidelines

## Build/Test/Lint Commands
- **Run all tests**: `pytest tests/`
- **Run single test**: `pytest tests/path/to/test_file.py::test_function_name`
- **Run by marker**: `pytest -m integration` (markers: unit, integration, slow, performance, ml_contracts)
- **Lint**: `ruff check src tests`
- **Format**: `ruff format src tests`
- **Type check**: `mypy src/`
- **Setup dev environment**: `scripts/setup-dev-environment.sh`

## Code Style Guidelines
- **Python version**: 3.11+
- **Line length**: 88 characters
- **Import style**: Absolute imports only (`from prompt_improver.module import Class`)
- **Import order**: stdlib → third-party → first-party (prompt_improver)
- **Type hints**: Required for all public functions/methods
- **Docstrings**: Google style for public APIs
- **Error handling**: Use specific exceptions, avoid bare `except:`
- **Async**: Use `async`/`await` patterns, not callbacks

## Project Structure
- Source code: `src/prompt_improver/`
- Tests: `tests/` (mirrors src structure)
- Config: `config/` (YAML files)
- Scripts: `scripts/` (automation)
