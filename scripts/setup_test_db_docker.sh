#!/bin/bash
# PostgreSQL Test Database Setup Script using Docker
# Alternative script when psql client is not installed locally

set -e

echo "🚀 Setting up PostgreSQL test database and user using Docker..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="apes_postgres"
TEST_DB_NAME="apes_test"
TEST_USER="test_user"
TEST_PASSWORD="test_password"

# Function to check if container is running
check_container() {
    echo "🔍 Checking PostgreSQL container..."
    if docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${GREEN}✅ PostgreSQL container is running${NC}"
        return 0
    else
        echo -e "${RED}❌ PostgreSQL container is not running${NC}"
        echo "Starting PostgreSQL container..."
        docker-compose up -d postgres
        sleep 5
        return 0
    fi
}

# Function to execute SQL in container
exec_sql() {
    local sql="$1"
    docker exec -i "$CONTAINER_NAME" psql -U apes_user -d apes_production -c "$sql" 2>/dev/null || return 1
}

# Function to create test user
create_test_user() {
    echo "👤 Creating test user '$TEST_USER'..."
    
    # Check if user exists
    if exec_sql "SELECT 1 FROM pg_user WHERE usename='$TEST_USER'" | grep -q "1"; then
        echo -e "${YELLOW}⚠️  User '$TEST_USER' already exists${NC}"
        # Update password
        exec_sql "ALTER USER $TEST_USER PASSWORD '$TEST_PASSWORD';"
    else
        # Create user
        exec_sql "CREATE USER $TEST_USER WITH PASSWORD '$TEST_PASSWORD';"
        echo -e "${GREEN}✅ User '$TEST_USER' created${NC}"
    fi
    
    # Grant permissions
    exec_sql "ALTER USER $TEST_USER CREATEDB;"
    echo -e "${GREEN}✅ Granted CREATEDB permission to '$TEST_USER'${NC}"
}

# Function to create test database
create_test_database() {
    echo "🗄️  Creating test database '$TEST_DB_NAME'..."
    
    # Check if database exists
    if exec_sql "SELECT 1 FROM pg_database WHERE datname='$TEST_DB_NAME'" | grep -q "1"; then
        echo -e "${YELLOW}⚠️  Database '$TEST_DB_NAME' already exists${NC}"
        # Drop and recreate
        echo "🧹 Dropping existing test database..."
        exec_sql "DROP DATABASE IF EXISTS $TEST_DB_NAME;"
    fi
    
    # Create database
    exec_sql "CREATE DATABASE $TEST_DB_NAME OWNER $TEST_USER;"
    echo -e "${GREEN}✅ Database '$TEST_DB_NAME' created${NC}"
    
    # Grant privileges
    exec_sql "GRANT ALL PRIVILEGES ON DATABASE $TEST_DB_NAME TO $TEST_USER;"
    echo -e "${GREEN}✅ Granted all privileges on '$TEST_DB_NAME' to '$TEST_USER'${NC}"
}

# Function to verify setup
verify_setup() {
    echo "🔐 Verifying test user can connect..."
    
    # Test connection using docker exec
    if docker exec -i "$CONTAINER_NAME" psql -U "$TEST_USER" -d "$TEST_DB_NAME" -c '\q' 2>/dev/null; then
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
    echo "PostgreSQL Test Environment Setup (Docker)"
    echo "================================================"
    echo "Container: $CONTAINER_NAME"
    echo "Test Database: $TEST_DB_NAME"
    echo "Test User: $TEST_USER"
    echo "================================================"
    
    # Check container
    check_container
    
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
        echo "postgresql://$TEST_USER:$TEST_PASSWORD@localhost:5432/$TEST_DB_NAME"
        echo ""
        echo "Environment variables for pytest:"
        echo "export POSTGRES_USERNAME=$TEST_USER"
        echo "export POSTGRES_PASSWORD=$TEST_PASSWORD"
        echo "export POSTGRES_HOST=localhost"
        echo "export POSTGRES_PORT=5432"
        echo ""
        echo "To run tests:"
        echo "export POSTGRES_USERNAME=$TEST_USER && export POSTGRES_PASSWORD=$TEST_PASSWORD && export POSTGRES_HOST=localhost && export POSTGRES_PORT=5432 && python3 -m pytest tests/"
        echo ""
    else
        echo -e "${RED}❌ Setup verification failed${NC}"
        exit 1
    fi
}

# Run main function
main