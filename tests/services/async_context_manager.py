"""
Async context manager utility for testing.

This module contains utilities for testing async context managers,
extracted from conftest.py to maintain clean architecture.
"""


class AsyncContextManager:
    """Helper class for testing async context managers."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
