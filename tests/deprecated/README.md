# Deprecated Tests

This directory contains legacy and migration tests that are temporary in nature.

## Structure
- `test_[original_name].py` - Legacy tests being phased out
- Migration tests from development phases
- One-time validation tests
- Tests that will be removed or refactored

## Guidelines
- These tests are temporary and should be removed once:
  - Migration is complete
  - Functionality is covered by proper tests
  - Development phase is finished
- Document why each test is deprecated
- Include timeline for removal
- Don't add new tests to this directory

## Note
⚠️ **These tests are not part of the regular test suite and may be removed at any time.**

## Run Tests (if needed)
```bash
pytest tests/deprecated/ -v
```
