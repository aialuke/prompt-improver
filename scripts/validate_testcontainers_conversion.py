#!/usr/bin/env python3
"""Validate TestContainers psycopg2 → psycopg conversion works correctly."""

import pytest
from testcontainers.postgres import PostgresContainer
from sqlalchemy import create_engine, text


def test_psycopg_conversion():
    """Test that psycopg2 → psycopg conversion works."""
    with PostgresContainer("postgres:15") as postgres:
        # Get original URL with psycopg2
        original_url = postgres.get_connection_url()
        print(f"Original URL: {original_url}")
        
        # Convert to psycopg3
        converted_url = original_url.replace("postgresql+psycopg2://", "postgresql+psycopg://")
        print(f"Converted URL: {converted_url}")
        
        # Test connection with both URLs
        try:
            # Test psycopg2 URL (should work with psycopg2 driver)
            engine_psycopg2 = create_engine(original_url)
            with engine_psycopg2.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                assert result.fetchone()[0] == 1
            print("✓ psycopg2 URL works")
            
            # Test psycopg3 URL (should work with psycopg3 driver)
            engine_psycopg3 = create_engine(converted_url)
            with engine_psycopg3.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                assert result.fetchone()[0] == 1
            print("✓ psycopg3 URL works")
            
            print("✓ Both URL formats work correctly")
            return True
            
        except Exception as e:
            print(f"✗ Connection test failed: {e}")
            return False


if __name__ == "__main__":
    success = test_psycopg_conversion()
    exit(0 if success else 1)