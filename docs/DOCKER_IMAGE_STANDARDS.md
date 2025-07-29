# Docker Image Standards

This document defines the standardized Docker images used across the APES project to prevent unnecessary image accumulation and maintain consistency.

## Standardized Images

### PostgreSQL
- **Production**: `postgres:15`
- **Testing**: `postgres:15` (aligned with production)
- **Rationale**: Single version across all environments for consistency

### Python
- **All Environments**: `python:3.11-slim-bookworm`
- **Rationale**: Project requires Python >=3.11, mypy configured for 3.11

### Redis (Testing Only)
- **Testing**: `redis:7-alpine`
- **Rationale**: Specific version, alpine for smaller size

## File Locations and Standards

### docker-compose.yml (Production)
- PostgreSQL: `postgres:15`
- No Redis (single-database architecture)

### docker-compose.test.yml (Testing)
- PostgreSQL: `postgres:15` (aligned with production)
- Redis: `redis:7-alpine`

### Dockerfile.mcp (MCP Server)
- Base: `python:3.11-slim-bookworm`
- Multi-stage build using same base image

### .devcontainer/Dockerfile (Development)
- Base: `python:3.11-slim-bookworm`
- ARG PYTHON_VERSION=3.11

### .github/workflows/ci.yml (CI/CD)
- Python matrix: `['3.11']` (standardized)
- PostgreSQL service: `postgres:15`
- Redis service: `redis:7`

## Prevention Guidelines

1. **Always use specific versions** - avoid `latest` tags
2. **Align test and production images** - use same PostgreSQL version
3. **Standardize Python version** - use 3.11 across all environments
4. **Review before adding new images** - ensure necessity and alignment
5. **Regular cleanup** - run `docker system prune` periodically

## Cleanup Commands

```bash
# Remove unused images
docker image prune -f

# Remove unused build cache
docker builder prune -f

# Check current usage
docker system df

# List all images
docker images
```

## Monitoring

- Only `postgres:15` should be actively used in production
- Test environments may temporarily pull additional images
- CI workflows should not leave persistent images on local systems
- Development containers should use standardized Python 3.11

## Last Updated
2025-07-29 - Initial standardization and cleanup completed
