#!/bin/bash
set -e

echo "Starting development services..."

# Check if docker-compose.yml exists
if [ -f "/workspace/docker-compose.yml" ]; then
    echo "Starting Docker services..."
    cd /workspace
    
    # Start PostgreSQL and Redis
    docker-compose up -d postgres redis
    
    # Wait for services to be ready
    echo "Waiting for PostgreSQL to be ready..."
    until docker-compose exec -T postgres pg_isready -U postgres; do
        sleep 1
    done
    
    echo "Waiting for Redis to be ready..."
    until docker-compose exec -T redis redis-cli ping | grep -q PONG; do
        sleep 1
    done
    
    # Run database migrations
    echo "Running database migrations..."
    source /workspace/.venv/bin/activate
    cd /workspace
    # Database initialized automatically via Docker
    
    echo "Services started successfully!"
else
    echo "No docker-compose.yml found, skipping service startup"
fi

# Start file watcher for hot reloading
if [ -f "/workspace/scripts/dev-server.sh" ]; then
    echo "Starting development file watcher..."
    /workspace/scripts/dev-server.sh &
fi

echo "Development services startup complete!"