#!/bin/bash

# APES Database Management Script
# Manages Docker PostgreSQL setup for the Adaptive Prompt Enhancement System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
DB_NAME="apes_production"
DB_USER="apes_user"
DB_HOST="localhost"
DB_PORT="5432"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
}

check_compose_file() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_error "docker-compose.yml not found at $COMPOSE_FILE"
        exit 1
    fi
}

start_database() {
    print_status "Starting APES PostgreSQL database..."
    
    cd "$PROJECT_ROOT"
    
    # Start only the postgres service
    if docker compose version &> /dev/null; then
        docker compose up -d postgres
    else
        docker-compose up -d postgres
    fi
    
    print_status "Waiting for database to be ready..."
    
    # Wait for PostgreSQL to be ready
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec apes_postgres pg_isready -U "$DB_USER" -d "$DB_NAME" &> /dev/null; then
            print_success "Database is ready!"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_error "Database failed to start after $max_attempts attempts"
            exit 1
        fi
        
        print_status "Attempt $attempt/$max_attempts - waiting for database..."
        sleep 2
        ((attempt++))
    done
}

stop_database() {
    print_status "Stopping APES PostgreSQL database..."
    
    cd "$PROJECT_ROOT"
    
    if docker compose version &> /dev/null; then
        docker compose down
    else
        docker-compose down
    fi
    
    print_success "Database stopped"
}

show_status() {
    print_status "Checking database status..."
    
    if docker ps | grep -q apes_postgres; then
        print_success "PostgreSQL container is running"
        
        # Test database connectivity
        if docker exec apes_postgres pg_isready -U "$DB_USER" -d "$DB_NAME" &> /dev/null; then
            print_success "Database is accepting connections"
        else
            print_warning "Database container is running but not ready"
        fi
        
        # Show container info
        echo ""
        echo "Container Information:"
        docker ps --filter "name=apes_postgres" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        
    else
        print_warning "PostgreSQL container is not running"
    fi
}

show_logs() {
    print_status "Showing database logs..."
    docker logs apes_postgres --tail 50 -f
}

connect_to_db() {
    print_status "Connecting to database..."
    docker exec -it apes_postgres psql -U "$DB_USER" -d "$DB_NAME"
}

start_pgadmin() {
    print_status "Starting pgAdmin (database management UI)..."
    
    cd "$PROJECT_ROOT"
    
    if docker compose version &> /dev/null; then
        docker compose --profile admin up -d pgadmin
    else
        docker-compose --profile admin up -d pgadmin
    fi
    
    print_success "pgAdmin started!"
    print_status "Access pgAdmin at: http://localhost:8080"
    print_status "Email: admin@apes.local"
    print_status "Password: ${PGADMIN_PASSWORD:-admin_password_2024}"
}

show_connection_info() {
    echo ""
    echo "=== APES Database Connection Information ==="
    echo "Host: localhost"
    echo "Port: 5432"
    echo "Database: $DB_NAME"
    echo "Username: $DB_USER"
    echo "Password: ${POSTGRES_PASSWORD:-apes_secure_password_2024}"
    echo ""
    echo "Connection String:"
    echo "postgresql://$DB_USER:${POSTGRES_PASSWORD:-apes_secure_password_2024}@localhost:5432/$DB_NAME"
    echo ""
    echo "For APES MCP Server configuration:"
    echo '  "apes-mcp": {'
    echo '    "command": "python",'
    echo '    "args": ["-m", "prompt_improver.mcp_server.server"],'
    echo '    "cwd": "/path/to/prompt-improver",'
    echo '    "env": {"PYTHONPATH": "/path/to/prompt-improver/src"}'
    echo '  }'
    echo ""
}

backup_database() {
    local backup_file="$PROJECT_ROOT/database/backup_$(date +%Y%m%d_%H%M%S).sql"
    print_status "Creating database backup..."
    
    docker exec apes_postgres pg_dump -U "$DB_USER" -d "$DB_NAME" > "$backup_file"
    print_success "Backup created: $backup_file"
}

show_help() {
    echo "APES Database Management Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start       Start the PostgreSQL database"
    echo "  stop        Stop the PostgreSQL database"  
    echo "  restart     Restart the PostgreSQL database"
    echo "  status      Show database status"
    echo "  logs        Show database logs (follow mode)"
    echo "  connect     Connect to database with psql"
    echo "  admin       Start pgAdmin web interface"
    echo "  info        Show connection information"
    echo "  backup      Create database backup"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start              # Start the database"
    echo "  $0 status             # Check if database is running"
    echo "  $0 connect            # Open psql connection"
    echo "  $0 admin              # Start pgAdmin UI"
}

# Main script logic
main() {
    case "${1:-help}" in
        start)
            check_docker
            check_compose_file
            start_database
            show_connection_info
            ;;
        stop)
            check_docker
            stop_database
            ;;
        restart)
            check_docker
            check_compose_file
            stop_database
            sleep 2
            start_database
            show_connection_info
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        connect)
            connect_to_db
            ;;
        admin)
            check_docker
            check_compose_file
            start_pgadmin
            ;;
        info)
            show_connection_info
            ;;
        backup)
            backup_database
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"