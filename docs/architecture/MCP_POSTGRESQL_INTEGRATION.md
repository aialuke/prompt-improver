# PostgreSQL MCP Server Integration

## Overview

The PostgreSQL MCP server enhances the `database-specialist` agent with direct database operations, schema analysis, and query optimization capabilities. This integration provides real-time database access for advanced database administration tasks.

## Configuration

### MCP Server Configuration
The PostgreSQL MCP server is configured in `.claude/mcp_servers.json`:

```json
{
  "mcpServers": {
    "postgresql-database": {
      "command": "python",
      "args": ["-m", "mcp_server_postgresql"],
      "env": {
        "DATABASE_URL": "${DATABASE_URL}",
        "POSTGRES_HOST": "${POSTGRES_HOST:-localhost}",
        "POSTGRES_PORT": "${POSTGRES_PORT:-5432}",
        "POSTGRES_USERNAME": "${POSTGRES_USERNAME}",
        "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD}",
        "POSTGRES_DATABASE": "${POSTGRES_DATABASE}"
      },
      "description": "PostgreSQL database operations for schema analysis, query optimization, and database administration.",
      "capabilities": [
        "schema_analysis",
        "query_execution", 
        "performance_monitoring",
        "migration_support"
      ]
    }
  }
}
```

### Environment Variables

Based on the project's existing configuration, set these environment variables:

```bash
# Production Database
export DATABASE_URL="postgresql://apes_user:apes_secure_password_2024@localhost:5432/apes_production"
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_USERNAME="apes_user"
export POSTGRES_PASSWORD="apes_secure_password_2024"
export POSTGRES_DATABASE="apes_production"

# Test Database (for development)
export TEST_DATABASE_URL="postgresql://apes_user:apes_secure_password_2024@localhost:5432/apes_test"
```

## Installation

### Step 1: Install PostgreSQL MCP Server

Choose one of these PostgreSQL MCP server implementations:

#### Option A: Universal Database MCP (Recommended)
```bash
# Install the universal database MCP server
pip install mcp-server-postgresql

# Or install from source
git clone https://github.com/bytebase/mcp-server-postgresql
cd mcp-server-postgresql
pip install -e .
```

#### Option B: SQLAlchemy-based MCP  
```bash
# Install SQLAlchemy MCP server
pip install mcp-alchemy

# Ensure SQLAlchemy compatibility with existing project
pip install sqlalchemy[postgresql]
```

### Step 2: Verify Database Connection

Test the database connection before activating the MCP server:

```bash
# Test connection with project credentials
psql -h localhost -U apes_user -d apes_production -c "SELECT version();"

# Verify test database access
psql -h localhost -U apes_user -d apes_test -c "SELECT version();"
```

### Step 3: Activate MCP Server

Add the MCP server to Claude Code:

```bash
# Add the PostgreSQL MCP server
claude mcp add postgresql-database

# Verify MCP server is active
claude mcp list
```

## Enhanced Agent Capabilities

### database-specialist Agent Enhancement

With PostgreSQL MCP integration, the `database-specialist` agent gains:

#### Real-time Schema Analysis
- Live schema inspection and documentation
- Table relationship mapping
- Index analysis and optimization recommendations
- Constraint validation and integrity checks

#### Query Performance Optimization
- Real-time query execution and analysis
- EXPLAIN plan analysis with recommendations
- Index usage optimization
- Query rewriting suggestions

#### Migration Support
- Schema diff analysis
- Migration script generation and validation
- Rollback planning and testing
- Zero-downtime deployment strategies

#### Database Administration
- Connection pool monitoring and optimization
- Database health checks and diagnostics
- Performance metrics collection
- Backup and recovery planning

## Usage Examples

### Example 1: Schema Analysis
```
User: "Analyze the user authentication tables for optimization opportunities"
database-specialist agent → PostgreSQL MCP → Live schema analysis → Optimization recommendations
```

### Example 2: Query Optimization
```
User: "This analytics query is slow: SELECT * FROM user_sessions WHERE created_at > '2024-01-01'"
database-specialist agent → PostgreSQL MCP → EXPLAIN analysis → Index recommendations
```

### Example 3: Migration Planning
```
User: "I need to add a new column to the users table safely"
database-specialist agent → PostgreSQL MCP → Schema analysis → Migration strategy
```

## Security Considerations

### Connection Security
- Use SSL/TLS connections for production: `sslmode=require`
- Implement connection pooling limits
- Regular credential rotation
- Network access restrictions

### Permission Management
- Grant minimal required permissions to MCP database user
- Separate read-only and read-write access levels
- Audit logging for all MCP operations
- Rate limiting for query execution

### Recommended Database Permissions
```sql
-- Create dedicated MCP user with limited permissions
CREATE USER mcp_agent WITH PASSWORD 'secure_password';

-- Grant schema read access
GRANT USAGE ON SCHEMA public TO mcp_agent;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_agent;

-- Grant specific write permissions as needed
GRANT INSERT, UPDATE, DELETE ON specific_tables TO mcp_agent;

-- Grant execution permissions for analysis
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO mcp_agent;
```

## Monitoring & Observability

### MCP Server Health Checks
- Connection pool status monitoring
- Query execution time tracking
- Error rate monitoring
- Resource utilization tracking

### Integration with Performance Monitoring
The PostgreSQL MCP server integrates with the project's existing monitoring:
- OpenTelemetry tracing for database operations
- Metrics collection for query performance
- SLO monitoring for database response times
- Alert integration for connection issues

## Troubleshooting

### Common Issues

#### Connection Failures
```bash
# Check database availability
pg_isready -h localhost -p 5432 -U apes_user

# Verify credentials
psql -h localhost -U apes_user -d apes_production -c "SELECT 1;"
```

#### Permission Errors
```bash
# Check user permissions
psql -h localhost -U apes_user -d apes_production -c "\\du"

# Verify table access
psql -h localhost -U apes_user -d apes_production -c "\\dt"
```

#### MCP Server Issues
```bash
# Check MCP server status
claude mcp status postgresql-database

# Restart MCP server
claude mcp restart postgresql-database

# Debug MCP server logs
claude mcp logs postgresql-database
```

## Next Steps

1. **Install and configure** PostgreSQL MCP server
2. **Test integration** with database-specialist agent
3. **Validate security** permissions and access controls
4. **Monitor performance** and optimize as needed
5. **Document usage patterns** and best practices

## Integration with Project Architecture

This PostgreSQL MCP integration aligns with:
- **Clean Architecture**: Repository patterns with direct database access
- **Real Behavior Testing**: Live database operations in testcontainers
- **Performance Monitoring**: Integration with existing SLO monitoring
- **Security**: Unified security component integration

---

*Generated as part of Claude Code Agent Enhancement Project - Phase 3*