# Database Setup Guide - Simple Docker PostgreSQL Setup

This guide explains the simplified PostgreSQL setup using Docker as the primary database.

## The Solution

We use a clean, simple approach:

- **Docker Container**: Uses port 5432 directly (no conflicts)
- **No Local PostgreSQL**: Homebrew PostgreSQL has been removed to eliminate conflicts
- **Configuration**: Automatically handled by `database/config.py`

### 2. **Automated Setup Script**

Run the setup script to ensure proper configuration:

```bash
./scripts/setup-dev-environment.sh
```

This script:

- ✅ Checks for port conflicts and warns if detected
- ✅ Configures Docker containers with non-conflicting ports
- ✅ Tests database connectivity
- ✅ Creates proper environment configuration

### 3. **Environment Configuration**

Copy the example environment file:

```bash
cp .env.example .env
```

Key settings:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432  # Direct port access
POSTGRES_DATABASE=apes_production
POSTGRES_USERNAME=apes_user
POSTGRES_PASSWORD=apes_secure_password_2024
```

## Manual Setup (Alternative)

If you prefer manual setup:

### Step 1: Start Docker PostgreSQL

```bash
docker-compose up -d postgres
```

### Step 2: Verify Connection

```bash
# Test connection to Docker PostgreSQL on port 5432
docker exec -it apes_postgres psql -U apes_user -d apes_production
```

### Step 3: Run Tests

```bash
python -m pytest tests/integration/ -v
```

## Troubleshooting

### Port Already in Use

If you see "port already in use" errors:

```bash
# Check what's using port 5432 (should only be Docker)
lsof -i :5432

# If you have conflicts, remove Homebrew PostgreSQL completely:
brew uninstall postgresql@15
```

### Connection Refused

If tests fail with connection refused:

```bash
# Check Docker container status
docker-compose ps

# View PostgreSQL logs
docker-compose logs postgres

# Restart containers
docker-compose restart postgres
```

### Wrong Port in Configuration

If you see connection errors, verify configuration:

```python
# Check current configuration
from src.prompt_improver.database.config import get_database_config
config = get_database_config()
print(f"Database URL: {config.database_url}")
```

## Best Practices

### 1. **Use the Setup Script**

Always run `./scripts/setup-dev-environment.sh` when setting up the project.

### 2. **Check Environment Variables**

Verify your `.env` file has the correct port:

```env
POSTGRES_PORT=5432
```

### 3. **Use Docker for Development**

- Docker provides consistent, isolated database environment
- No conflicts since local PostgreSQL is removed
- Ensures team members have identical setups

### 4. **Monitor Port Usage**

Before starting development:

```bash
# Check if any PostgreSQL is running (should only be Docker)
ps aux | grep postgres

# Check port usage (should only show Docker)
netstat -tlnp | grep 5432
```

## Integration with CI/CD

### GitHub Actions / CI

Our configuration automatically handles CI environments:

- Uses PostgreSQL service containers
- No port conflicts in isolated CI environments
- Consistent with local development setup

### Local Development

- Port 5432 used directly (no conflicts)
- Docker containers provide isolation
- Setup script ensures consistent environment

## Migration Notes

If you're migrating from an existing setup:

1. **Remove Homebrew PostgreSQL**: `brew uninstall postgresql@15`
2. **Update existing .env files** to use port 5432
3. **Restart Docker containers** after configuration changes
4. **Run the setup script** to verify everything works

## Support

If you encounter persistent issues:

1. Run the setup script: `./scripts/setup-dev-environment.sh`
2. Check Docker logs: `docker-compose logs postgres`
3. Verify port configuration in `.env` file
4. Test connection manually: `docker exec -it apes_postgres psql -U apes_user -d apes_production`

---

_This guide was simplified after removing Homebrew PostgreSQL conflicts during the migration to real behavior testing in 2025._

## Configuration Standards

### Database URL Formats

**Async (asyncpg)**: `postgresql+asyncpg://user:pass@host:port/db`
**Sync (asyncpg)**: `postgresql://user:pass@host:port/db`

### Environment Variables

All database configuration should use the following pattern:
- `POSTGRES_HOST`: Database host (default: localhost)
- `POSTGRES_PORT`: Database port (default: 5432)
- `POSTGRES_DATABASE`: Database name
- `POSTGRES_USERNAME`: Database username
- `POSTGRES_PASSWORD`: Database password

### TestContainers

TestContainers URLs are automatically converted from psycopg2 to asyncpg format:
```python
db_url = postgres.get_connection_url().replace("postgresql+psycopg2://", "postgresql+asyncpg://")
```

### SQLite Migration Notes

The project has been fully migrated from SQLite to PostgreSQL:
- **aiosqlite dependency removed**: No SQLite-related dependencies remain
- **Package metadata cleaned**: All stale references removed
- **Environment standardized**: All configurations use asyncpg format
- **Validation scripts added**: Monitoring prevents unwanted dependencies

### Dependency Monitoring

Use the provided scripts to ensure clean environment:
```bash
# Check for unwanted dependencies
python scripts/monitor_dependencies.py

# Validate environment configuration
python scripts/validate_environment_config.py

# Validate TestContainers conversion
python scripts/validate_testcontainers_conversion.py
```
