#!/bin/bash
# Test Infrastructure Setup Script
# Sets up PostgreSQL and Redis for integration testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Docker installation
check_docker() {
    if ! command_exists docker; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "Docker is available and running"
}

# Setup PostgreSQL container
setup_postgresql() {
    log_info "Setting up PostgreSQL for integration tests..."
    
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^apes_postgres_test$"; then
        log_warning "PostgreSQL test container already exists"
        
        # Check if it's running
        if docker ps --format '{{.Names}}' | grep -q "^apes_postgres_test$"; then
            log_success "PostgreSQL test container is already running"
            return 0
        else
            log_info "Starting existing PostgreSQL test container..."
            docker start apes_postgres_test
        fi
    else
        log_info "Creating new PostgreSQL test container..."
        docker run -d \
            --name apes_postgres_test \
            -e POSTGRES_DB=apes_production \
            -e POSTGRES_USER=apes_user \
            -e POSTGRES_PASSWORD=apes_secure_password_2024 \
            -e POSTGRES_INITDB_ARGS="--auth-host=md5" \
            -p 5432:5432 \
            postgres:15-alpine
    fi
    
    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec apes_postgres_test pg_isready -U apes_user -d apes_production >/dev/null 2>&1; then
            log_success "PostgreSQL is ready!"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "PostgreSQL failed to start within timeout"
            exit 1
        fi
        
        log_info "Attempt $attempt/$max_attempts - waiting..."
        sleep 2
        ((attempt++))
    done
    
    # Create test database if it doesn't exist
    log_info "Creating test database if needed..."
    docker exec apes_postgres_test psql -U apes_user -d apes_production -c "CREATE DATABASE apes_test;" 2>/dev/null || true
    
    log_success "PostgreSQL test container is ready"
}

# Setup Redis container
setup_redis() {
    log_info "Setting up Redis for integration tests..."
    
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^apes_redis_test$"; then
        log_warning "Redis test container already exists"
        
        # Check if it's running
        if docker ps --format '{{.Names}}' | grep -q "^apes_redis_test$"; then
            log_success "Redis test container is already running"
            return 0
        else
            log_info "Starting existing Redis test container..."
            docker start apes_redis_test
        fi
    else
        log_info "Creating new Redis test container..."
        docker run -d \
            --name apes_redis_test \
            -p 6379:6379 \
            redis:7-alpine redis-server --appendonly yes
    fi
    
    # Wait for Redis to be ready
    log_info "Waiting for Redis to be ready..."
    max_attempts=15
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec apes_redis_test redis-cli ping >/dev/null 2>&1; then
            log_success "Redis is ready!"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "Redis failed to start within timeout"
            exit 1
        fi
        
        log_info "Attempt $attempt/$max_attempts - waiting..."
        sleep 1
        ((attempt++))
    done
    
    log_success "Redis test container is ready"
}

# Setup test directories
setup_directories() {
    log_info "Creating test directories..."
    
    directories=(
        "logs"
        "test_models"
        "tests/fixtures"
        "/tmp/test_secrets"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    # Set permissions
    chmod 755 logs test_models tests/fixtures 2>/dev/null || true
    chmod 700 /tmp/test_secrets 2>/dev/null || true
    
    log_success "Test directories are ready"
}

# Copy environment file
setup_env_file() {
    log_info "Setting up environment file..."
    
    if [ ! -f ".env.test.local" ]; then
        if [ -f ".env.test" ]; then
            cp .env.test .env.test.local
            log_success "Copied .env.test to .env.test.local"
            log_warning "Please review and customize .env.test.local for your environment"
        else
            log_error ".env.test file not found. Please create it first."
            exit 1
        fi
    else
        log_success ".env.test.local already exists"
    fi
}

# Test connections
test_connections() {
    log_info "Testing database and Redis connections..."
    
    # Test PostgreSQL
    if docker exec apes_postgres_test psql -U apes_user -d apes_production -c "SELECT version();" >/dev/null 2>&1; then
        log_success "PostgreSQL connection test passed"
    else
        log_error "PostgreSQL connection test failed"
        exit 1
    fi
    
    # Test Redis
    if docker exec apes_redis_test redis-cli ping >/dev/null 2>&1; then
        log_success "Redis connection test passed"
    else
        log_error "Redis connection test failed"
        exit 1
    fi
}

# Clean up function
cleanup() {
    log_info "Cleaning up test infrastructure..."
    
    # Stop and remove containers
    docker stop apes_postgres_test apes_redis_test 2>/dev/null || true
    docker rm apes_postgres_test apes_redis_test 2>/dev/null || true
    
    log_success "Test infrastructure cleaned up"
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     Set up test infrastructure (default)"
    echo "  cleanup   Clean up test infrastructure"
    echo "  restart   Restart test infrastructure"
    echo "  status    Show status of test infrastructure"
    echo "  test      Test connections to infrastructure"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                # Set up infrastructure"
    echo "  $0 setup          # Set up infrastructure"
    echo "  $0 cleanup        # Clean up infrastructure"
    echo "  $0 restart        # Restart infrastructure"
}

# Show status
show_status() {
    log_info "Test infrastructure status:"
    echo ""
    
    # PostgreSQL status
    if docker ps --format '{{.Names}}' | grep -q "^apes_postgres_test$"; then
        log_success "PostgreSQL container: Running"
        echo "  - Port: 5432"
        echo "  - Database: apes_production"
        echo "  - User: apes_user"
    elif docker ps -a --format '{{.Names}}' | grep -q "^apes_postgres_test$"; then
        log_warning "PostgreSQL container: Stopped"
    else
        log_error "PostgreSQL container: Not found"
    fi
    
    echo ""
    
    # Redis status
    if docker ps --format '{{.Names}}' | grep -q "^apes_redis_test$"; then
        log_success "Redis container: Running"
        echo "  - Port: 6379"
    elif docker ps -a --format '{{.Names}}' | grep -q "^apes_redis_test$"; then
        log_warning "Redis container: Stopped"
    else
        log_error "Redis container: Not found"
    fi
    
    echo ""
    
    # Environment file status
    if [ -f ".env.test.local" ]; then
        log_success "Environment file: .env.test.local exists"
    elif [ -f ".env.test" ]; then
        log_warning "Environment file: Only .env.test exists (create .env.test.local)"
    else
        log_error "Environment file: Not found"
    fi
}

# Main function
main() {
    local command="${1:-setup}"
    
    case "$command" in
        "setup")
            check_docker
            setup_postgresql
            setup_redis
            setup_directories
            setup_env_file
            test_connections
            echo ""
            log_success "Test infrastructure setup complete!"
            log_info "Next steps:"
            echo "  1. Review and customize .env.test.local"
            echo "  2. Run: python scripts/validate_test_environment.py"
            echo "  3. Run integration tests: pytest tests/integration/"
            ;;
        "cleanup")
            cleanup
            ;;
        "restart")
            cleanup
            sleep 2
            check_docker
            setup_postgresql
            setup_redis
            test_connections
            log_success "Test infrastructure restarted"
            ;;
        "status")
            show_status
            ;;
        "test")
            test_connections
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"