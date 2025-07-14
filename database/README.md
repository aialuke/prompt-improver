# Database Initialization Files

This directory contains the main database initialization files for the APES (Adaptive Prompt Enhancement System) project.

## Files

### `init.sql`
- Database initialization script that runs first
- Sets up database roles, permissions, and environment
- Creates application-specific roles and grants permissions
- Sets timezone and other database-level configurations

### `schema.sql`
- Complete database schema definition
- Creates all tables, indexes, and constraints
- Defines the data model for rule performance tracking, ML optimization, and user feedback

## Docker Integration

These files are automatically mounted into the PostgreSQL container via Docker Compose:
- `init.sql` → `/docker-entrypoint-initdb.d/init.sql`
- `schema.sql` → `/docker-entrypoint-initdb.d/schema.sql`

The files execute in alphabetical order during container initialization.

## Usage

To start the database with these initialization files:

```bash
docker-compose up postgres -d
```

The database will be accessible at `localhost:5432` with credentials defined in `docker-compose.yml`.

## Schema Overview

The database supports:
- **Rule Performance Tracking**: Store and analyze rule effectiveness
- **ML Model Performance**: Track machine learning model metrics
- **User Feedback**: Collect and analyze user satisfaction data
- **A/B Testing**: Support experimental rule combinations
- **Pattern Discovery**: Store ML-discovered optimization patterns

## Development

For local development, the database can be accessed using:
- **Host**: localhost
- **Port**: 5432
- **Database**: apes_production
- **User**: apes_user
- **Password**: (see docker-compose.yml)