-- APES Database Initialization Script
-- This script runs before schema.sql and sets up the database environment

-- Create any additional users or permissions if needed
-- Note: The main database and user are created via environment variables in docker-compose.yml

-- Set timezone
SET timezone = 'UTC';

-- Create application-specific role (optional, for more granular permissions)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'apes_app_role') THEN
        CREATE ROLE apes_app_role;
    END IF;
END
$$;

-- Grant permissions to the application role
GRANT CONNECT ON DATABASE apes_production TO apes_app_role;
GRANT USAGE ON SCHEMA public TO apes_app_role;
GRANT CREATE ON SCHEMA public TO apes_app_role;

-- Grant the role to the main user
GRANT apes_app_role TO apes_user;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'APES Database initialization completed at %', NOW();
END
$$;