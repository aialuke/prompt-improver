"""Comprehensive Centralized Pydantic Configuration System
Following 2025 best practices for production-ready configuration management.

This module provides a unified configuration system with environment-based settings,
validation, and secure credential management.
"""
import os
from typing import Any, Dict, List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings

class DatabaseConfig(BaseSettings):
    """Database configuration with secure defaults."""
    database_url: str = Field(default='postgresql+asyncpg://postgres:postgres@localhost:5432/prompt_improver', description='PostgreSQL database URL (localhost for container setup)')
    database_pool_size: int = Field(default=10, description='Database connection pool size')
    database_max_overflow: int = Field(default=20, description='Database max overflow connections')
    database_pool_timeout: int = Field(default=30, description='Database pool timeout in seconds')

    class Config:
        env_prefix = 'DB_'
        case_sensitive = False

class RedisConfig(BaseSettings):
    """Redis configuration for external Redis service."""
    redis_url: str = Field(default='redis://redis.external.service:6379/0', description='Redis connection URL (external service)')
    redis_max_connections: int = Field(default=100, description='Redis max connections')
    redis_timeout: int = Field(default=5, description='Redis timeout in seconds')

    class Config:
        env_prefix = 'REDIS_'
        case_sensitive = False

class MCPConfig(BaseSettings):
    """MCP Server configuration."""
    mcp_host: str = Field(default='0.0.0.0', description='MCP server host (0.0.0.0 for external access)')
    mcp_port: int = Field(default=8080, description='MCP server port')
    mcp_timeout: int = Field(default=30, description='MCP timeout in seconds')
    mcp_batch_size: int = Field(default=100, description='MCP batch processing size')
    mcp_session_maxsize: int = Field(default=1000, description='MCP session cache max size')
    mcp_session_ttl: int = Field(default=3600, description='MCP session TTL in seconds')
    mcp_session_cleanup_interval: int = Field(default=300, description='MCP session cleanup interval in seconds')

    class Config:
        env_prefix = 'MCP_'
        case_sensitive = False

class SecurityConfig(BaseSettings):
    """Security configuration."""
    secret_key: str = Field(default='dev-secret-key-change-in-production', description='Application secret key')
    encryption_key: str | None = Field(default=None, description='Encryption key for sensitive data')

    class Config:
        env_prefix = 'SECURITY_'
        case_sensitive = False

class AppConfig(BaseSettings):
    """Main application configuration."""
    environment: str = Field(default='development', description='Application environment')
    debug: bool = Field(default=False, description='Debug mode')
    log_level: str = Field(default='INFO', description='Logging level')
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @validator('environment')
    def validate_environment(cls, v):
        allowed = ['development', 'testing', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f'Environment must be one of {allowed}')
        return v

    @validator('log_level')
    def validate_log_level(cls, v):
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed:
            raise ValueError(f'Log level must be one of {allowed}')
        return v.upper()

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False
_config: AppConfig | None = None

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config

def reload_config() -> AppConfig:
    """Reload the configuration from environment."""
    global _config
    _config = AppConfig()
    return _config
