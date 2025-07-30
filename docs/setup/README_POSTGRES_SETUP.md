# PostgreSQL Setup for APES

This guide walks you through setting up PostgreSQL with Docker for the Adaptive Prompt Enhancement System (APES).

## Quick Start

1. **Start the database:**
   ```bash
   ./scripts/start_database.sh start
   ```

2. **Check status:**
   ```bash
   ./scripts/start_database.sh status
   ```

3. **View connection info:**
   ```bash
   ./scripts/start_database.sh info
   ```

That's it! The database is ready to use.

## What Was Set Up For You

### 1. Docker Configuration (`docker-compose.yml`)
- PostgreSQL 15 container with persistent data
- Database: `apes_production`
- User: `apes_user`
- Password: `apes_secure_password_2024`
- Port: `5432`
- Optional pgAdmin web interface on port `8080`

### 2. Database Schema (`database/schema.sql`)
Complete schema with tables for:
- **Rule Performance Tracking**: `rule_performance`, `rule_combinations`
- **User Feedback**: `user_feedback`, `improvement_sessions` 
- **ML Analytics**: `ml_model_performance`, `discovered_patterns`
- **Configuration**: `rule_metadata`, `ab_experiments`
- **Optimized indexes** and **views** for common queries

### 3. Management Script (`scripts/start_database.sh`)
Convenient commands for database management:
```bash
./scripts/start_database.sh start      # Start database
./scripts/start_database.sh stop       # Stop database
./scripts/start_database.sh status     # Check status
./scripts/start_database.sh logs       # View logs
./scripts/start_database.sh connect    # Open psql
./scripts/start_database.sh admin      # Start pgAdmin UI
./scripts/start_database.sh backup     # Create backup
```

### 4. Configuration Files
- `config/database_config.yaml` - Database settings
- `.env.example` - Environment variables template

## Connection Information

**For your Python code:**
```python
DATABASE_URL = "postgresql+psycopg://apes_user:apes_secure_password_2024@localhost:5432/apes_production"
```

**For MCP Server configuration:**
```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": [
        "-y", 
        # Use APES unified MCP server instead
        "postgresql://apes_user:apes_secure_password_2024@localhost:5432/apes_production"
      ]
    }
  }
}
```

## Database Schema Overview

### Core Tables

1. **`rule_performance`** - Track how well each rule performs
   - Rule ID, improvement scores, execution times
   - Prompt characteristics and before/after metrics
   - Used for ML optimization and rule selection

2. **`user_feedback`** - Store user ratings and feedback
   - Original vs improved prompts
   - User ratings (1-5 scale)
   - Applied rules and improvement areas

3. **`ml_model_performance`** - ML model metrics and artifacts
   - Model versions, accuracy scores
   - Hyperparameters and feature importance
   - Links to MLflow runs

4. **`discovered_patterns`** - ML-discovered rule patterns
   - New rule patterns found by ML algorithms
   - Effectiveness scores and validation status

### Key Features

- **Automatic timestamps** with triggers
- **JSONB fields** for flexible metadata storage
- **Optimized indexes** for fast queries
- **Views** for common analytics queries
- **UUID support** for unique identifiers

## Usage Examples

### Start Database and Connect
```bash
# Start the database
./scripts/start_database.sh start

# Connect with psql
./scripts/start_database.sh connect

# In psql, try some queries:
\dt                              # List tables
SELECT * FROM rule_metadata;     # View rule metadata
SELECT * FROM rule_effectiveness_summary;  # View rule performance
```

### Access pgAdmin Web Interface
```bash
# Start pgAdmin
./scripts/start_database.sh admin

# Open http://localhost:8080
# Email: admin@apes.local
# Password: admin_password_2024
```

### Backup Database
```bash
./scripts/start_database.sh backup
```

## Integration with Your APES Code

The database is ready to integrate with your existing APES components:

### 1. **Rule Engine Integration**
Store rule effectiveness in real-time:
```python
# After applying a rule
await db.execute(
    "INSERT INTO rule_performance (rule_id, improvement_score, prompt_type) VALUES ($1, $2, $3)",
    [rule_id, 0.85, "technical"]
)
```

### 2. **ML Optimizer Integration**
Query best rules for prompt characteristics:
```python
# Get top performing rules for a prompt type
rules = await db.fetch(
    "SELECT rule_id FROM rule_effectiveness_summary WHERE prompt_types_count > 5 ORDER BY avg_improvement DESC LIMIT 3"
)
```

### 3. **User Feedback Collection**
Store user ratings:
```python
# After user rates an improved prompt
await db.execute(
    "INSERT INTO user_feedback (original_prompt, improved_prompt, user_rating, applied_rules) VALUES ($1, $2, $3, $4)",
    [original, improved, rating, json.dumps(rules_used)]
)
```

## Next Steps

1. **Use APES Unified MCP Server:**
   The APES project includes a built-in MCP server with database integration.
   No external installation required.

2. **Add to your Claude CLI configuration:**
   ```json
   {
     "mcpServers": {
       "apes-mcp": {
         "command": "python",
         "args": ["-m", "prompt_improver.mcp_server.server"],
         "cwd": "/path/to/prompt-improver",
         "env": {
           "PYTHONPATH": "/path/to/prompt-improver/src"
         }
       }
     }
   }
   ```

3. **Update your Python code** to use the database connection

4. **Start storing rule performance data** as rules are applied

5. **Implement the ML feedback loop** using the stored data

## Environment Variables

Copy `.env.example` to `.env` and customize:
```bash
cp .env.example .env
# Edit .env with your preferences
```

## Troubleshooting

### Database won't start:
```bash
# Check Docker is running
docker --version

# Check container logs
./scripts/start_database.sh logs

# Restart everything
./scripts/start_database.sh restart
```

### Connection issues:
```bash
# Verify database is ready
./scripts/start_database.sh status

# Test connection
./scripts/start_database.sh connect
```

### Reset everything:
```bash
# Stop and remove containers/volumes
docker-compose down -v

# Start fresh
./scripts/start_database.sh start
```

The database setup is complete and ready to transform your static rule system into a dynamic, learning system! 🚀