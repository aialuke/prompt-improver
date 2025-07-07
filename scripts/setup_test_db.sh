#!/bin/bash
# PostgreSQL Test Database Setup Script
# Based on Context7 pytest-postgresql best practices

set -e

echo "🚀 Setting up PostgreSQL test database and user..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
TEST_DB_NAME="apes_test"
TEST_USER="test_user"
TEST_PASSWORD="test_password"

# Function to check if PostgreSQL is running
check_postgres() {
    echo "🔍 Checking PostgreSQL connection..."
    if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -c '\q' 2>/dev/null; then
        echo -e "${GREEN}✅ PostgreSQL is running${NC}"
        return 0
    else
        echo -e "${RED}❌ PostgreSQL is not accessible${NC}"
        echo "Please ensure PostgreSQL is running and accessible at $POSTGRES_HOST:$POSTGRES_PORT"
        return 1
    fi
}

# Function to create test user
create_test_user() {
    echo "👤 Creating test user '$TEST_USER'..."
    
    # Check if user exists
    USER_EXISTS=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -tAc "SELECT 1 FROM pg_user WHERE usename='$TEST_USER'" 2>/dev/null || echo "0")
    
    if [ "$USER_EXISTS" = "1" ]; then
        echo -e "${YELLOW}⚠️  User '$TEST_USER' already exists${NC}"
        # Update password just in case
        PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -c "ALTER USER $TEST_USER PASSWORD '$TEST_PASSWORD';" 2>/dev/null
    else
        # Create user with password
        PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -c "CREATE USER $TEST_USER WITH PASSWORD '$TEST_PASSWORD';" 2>/dev/null
        echo -e "${GREEN}✅ User '$TEST_USER' created${NC}"
    fi
    
    # Grant necessary permissions
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -c "ALTER USER $TEST_USER CREATEDB;" 2>/dev/null
    echo -e "${GREEN}✅ Granted CREATEDB permission to '$TEST_USER'${NC}"
}

# Function to create test database
create_test_database() {
    echo "🗄️  Creating test database '$TEST_DB_NAME'..."
    
    # Check if database exists
    DB_EXISTS=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -tAc "SELECT 1 FROM pg_database WHERE datname='$TEST_DB_NAME'" 2>/dev/null || echo "0")
    
    if [ "$DB_EXISTS" = "1" ]; then
        echo -e "${YELLOW}⚠️  Database '$TEST_DB_NAME' already exists${NC}"
        # Drop and recreate for clean state
        echo "🧹 Dropping existing test database..."
        PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -c "DROP DATABASE IF EXISTS $TEST_DB_NAME;" 2>/dev/null
    fi
    
    # Create database owned by test user
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -c "CREATE DATABASE $TEST_DB_NAME OWNER $TEST_USER;" 2>/dev/null
    echo -e "${GREEN}✅ Database '$TEST_DB_NAME' created${NC}"
    
    # Grant all privileges on test database to test user
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -c "GRANT ALL PRIVILEGES ON DATABASE $TEST_DB_NAME TO $TEST_USER;" 2>/dev/null
    echo -e "${GREEN}✅ Granted all privileges on '$TEST_DB_NAME' to '$TEST_USER'${NC}"
}

# Function to verify setup
verify_setup() {
    echo "🔐 Verifying test user can connect..."
    
    if PGPASSWORD="$TEST_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$TEST_USER" -d "$TEST_DB_NAME" -c '\q' 2>/dev/null; then
        echo -e "${GREEN}✅ Test user can connect successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Test user connection failed${NC}"
        return 1
    fi
}

# Main execution
main() {
    echo "================================================"
    echo "PostgreSQL Test Environment Setup"
    echo "================================================"
    echo "Host: $POSTGRES_HOST:$POSTGRES_PORT"
    echo "Admin User: $POSTGRES_USER"
    echo "Test Database: $TEST_DB_NAME"
    echo "Test User: $TEST_USER"
    echo "================================================"
    
    # Check if PostgreSQL is running
    if ! check_postgres; then
        echo -e "${YELLOW}Attempting to start PostgreSQL with Docker...${NC}"
        # Try to start with docker-compose if available
        if command -v docker-compose &> /dev/null; then
            docker-compose up -d postgres
            sleep 5
            if ! check_postgres; then
                echo -e "${RED}Failed to start PostgreSQL${NC}"
                exit 1
            fi
        else
            echo -e "${RED}Docker Compose not found. Please start PostgreSQL manually.${NC}"
            exit 1
        fi
    fi
    
    # Create test user
    create_test_user
    
    # Create test database
    create_test_database
    
    # Verify setup
    if verify_setup; then
        echo ""
        echo -e "${GREEN}🎉 PostgreSQL test environment setup complete!${NC}"
        echo ""
        echo "Test connection string:"
        echo "postgresql://$TEST_USER:$TEST_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$TEST_DB_NAME"
        echo ""
        echo "Environment variables for pytest:"
        echo "export POSTGRES_USERNAME=$TEST_USER"
        echo "export POSTGRES_PASSWORD=$TEST_PASSWORD"
        echo "export POSTGRES_HOST=$POSTGRES_HOST"
        echo "export POSTGRES_PORT=$POSTGRES_PORT"
        echo ""
    else
        echo -e "${RED}❌ Setup verification failed${NC}"
        exit 1
    fi
}

# Run main function
main