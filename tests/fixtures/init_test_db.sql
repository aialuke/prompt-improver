-- PostgreSQL Test Database Initialization
-- Based on Context7 pytest-postgresql best practices

-- Create test user with necessary permissions
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_user
      WHERE usename = 'test_user') THEN
      
      CREATE USER test_user WITH PASSWORD 'test_password';
   END IF;
END
$do$;

-- Grant necessary permissions to test user
ALTER USER test_user CREATEDB;

-- Create test database if it doesn't exist
SELECT 'CREATE DATABASE apes_test OWNER test_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'apes_test')\gexec

-- Grant all privileges on test database
GRANT ALL PRIVILEGES ON DATABASE apes_test TO test_user;

-- Connect to test database and set up permissions
\c apes_test

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO test_user;
GRANT CREATE ON SCHEMA public TO test_user;

-- Ensure test user can create extensions if needed
ALTER USER test_user WITH SUPERUSER;