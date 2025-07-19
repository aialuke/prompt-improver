#!/bin/bash

# Development Environment Setup Script
# Prevents PostgreSQL port conflicts and ensures proper Docker setup

set -e

echo "🚀 Setting up development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if port 5432 is in use (should be Docker PostgreSQL only)
check_port_conflict() {
    echo -e "${YELLOW}Checking for port conflicts...${NC}"
    
    if lsof -i :5432 >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Port 5432 is in use (Docker PostgreSQL)${NC}"
        return 0
    else
        echo -e "${GREEN}✅ Port 5432 is available for Docker PostgreSQL${NC}"
        return 1
    fi
}

# Stop any existing containers
stop_existing_containers() {
    echo -e "${YELLOW}Stopping any existing APES containers...${NC}"
    docker-compose down --remove-orphans 2>/dev/null || true
}

# Start PostgreSQL container
start_postgres() {
    echo -e "${YELLOW}Starting PostgreSQL container...${NC}"
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
    timeout=30
    while ! docker exec apes_postgres pg_isready -U apes_user -d apes_production >/dev/null 2>&1; do
        sleep 1
        timeout=$((timeout - 1))
        if [ $timeout -eq 0 ]; then
            echo -e "${RED}❌ PostgreSQL failed to start within 30 seconds${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}✅ PostgreSQL is ready${NC}"
}

# Test database connection
test_connection() {
    echo -e "${YELLOW}Testing database connection...${NC}"
    
    # Test with Python
    if python -c "
import asyncio
import asyncpg
async def test():
    try:
        conn = await asyncpg.connect('postgresql://apes_user:apes_secure_password_2024@localhost:5432/apes_production')
        await conn.close()
        print('✅ Database connection successful')
    except Exception as e:
        print(f'❌ Database connection failed: {e}')
        exit(1)
asyncio.run(test())
    " 2>/dev/null; then
        echo -e "${GREEN}✅ Database connection test passed${NC}"
    else
        echo -e "${RED}❌ Database connection test failed${NC}"
        echo -e "${YELLOW}💡 Try running: docker-compose logs postgres${NC}"
        exit 1
    fi
}

# Create .env file if it doesn't exist
setup_env_file() {
    if [ ! -f .env ]; then
        echo -e "${YELLOW}Creating .env file from template...${NC}"
        cp .env.example .env
        echo -e "${GREEN}✅ .env file created${NC}"
    else
        echo -e "${GREEN}✅ .env file already exists${NC}"
    fi
}

# Main execution
main() {
    echo -e "${GREEN}🔧 APES Development Environment Setup${NC}"
    echo ""
    
    # Check if we're in the right directory
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}❌ Error: docker-compose.yml not found. Please run from project root.${NC}"
        exit 1
    fi
    
    check_port_conflict
    setup_env_file
    stop_existing_containers
    start_postgres
    test_connection
    
    echo ""
    echo -e "${GREEN}🎉 Development environment setup complete!${NC}"
    echo ""
    echo -e "${GREEN}📋 Next steps:${NC}"
    echo "   1. Run tests: ${YELLOW}python -m pytest tests/integration/ -v${NC}"
    echo "   2. Start development: ${YELLOW}python src/prompt_improver/main.py${NC}"
    echo "   3. View logs: ${YELLOW}docker-compose logs -f postgres${NC}"
    echo ""
    echo -e "${GREEN}🔧 Useful commands:${NC}"
    echo "   • Stop containers: ${YELLOW}docker-compose down${NC}"
    echo "   • View PostgreSQL logs: ${YELLOW}docker-compose logs postgres${NC}"
    echo "   • Connect to database: ${YELLOW}docker exec -it apes_postgres psql -U apes_user -d apes_production${NC}"
    echo ""
}

# Run main function
main "$@"