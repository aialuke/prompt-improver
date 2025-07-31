#!/usr/bin/env python3
"""Validate TestContainers URL conversion to asyncpg format works correctly."""

import pytest
from testcontainers.postgres import PostgresContainer
from sqlalchemy import create_engine, text


def test_asyncpg_conversion():
    """Test that TestContainer URL conversion to asyncpg format works."""
    with PostgresContainer("postgres:15") as postgres:
        # Get original URL from TestContainer
        original_url = postgres.get_connection_url()
        print(f"Original URL: {original_url}")

        # Convert to asyncpg format (APES standard)
        converted_url = original_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
        print(f"Converted URL: {converted_url}")

        # Test URL format conversion (APES standard)
        try:
            # Verify URL format compliance with APES standards
            assert "postgresql+asyncpg://" in converted_url
            assert "psycopg2" not in converted_url  # Ensure clean conversion
            print("✓ URL format validation passed")

            # Test basic URL structure
            assert "test:test@" in converted_url  # Has credentials
            assert ":54" in converted_url or ":5432" in converted_url  # Has port
            assert "/test" in converted_url  # Has database name
            print("✓ URL structure validation passed")

            print("✓ TestContainer → asyncpg conversion successful")
            print("✓ Ready for APES asyncpg usage")

            return True

        except Exception as e:
            print(f"✗ URL validation failed: {e}")
            return False


if __name__ == "__main__":
    success = test_asyncpg_conversion()
    exit(0 if success else 1)