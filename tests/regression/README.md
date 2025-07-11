# Regression Tests

This directory contains regression tests to prevent previously fixed bugs from reoccurring.

## Structure
- `test_[feature]_regression.py` - Tests for specific features that previously had bugs
- Focus on ensuring fixed bugs don't regress
- Test scenarios that previously failed
- Include performance regression tests

## Guidelines
- Document the original bug/issue in test docstrings
- Include issue/PR numbers for traceability
- Test the specific scenario that was problematic
- Use realistic data that triggered the original bug
- Include both positive and negative test cases

## Run Tests
```bash
pytest tests/regression/ -v
```
