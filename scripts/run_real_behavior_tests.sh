#!/bin/bash
set -e

# Real-behavior database testing runner following 2025 best practices
# This script demonstrates how to run authentic database error tests without mocks

echo "🔧 Real-behavior Database Error Testing (2025 Best Practices)"
echo "================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check dependencies
print_status "Checking dependencies..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    print_status "On macOS: Start Docker Desktop"
    print_status "On Linux: sudo systemctl start docker"
    exit 1
fi

print_success "Docker is running"

# Check Python version
if ! python3 --version | grep -q "Python 3.[8-9]\|Python 3.1[0-9]"; then
    print_warning "Python 3.8+ recommended for async testing features"
fi

# Install test dependencies if needed
if [ ! -f "venv/bin/activate" ] && [ ! -f ".venv/bin/activate" ]; then
    print_status "Setting up virtual environment for real-behavior testing..."
    python3 -m venv venv
    source venv/bin/activate
    print_success "Virtual environment created"
else
    print_status "Activating existing virtual environment..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        source .venv/bin/activate
    fi
fi

# Install or update dependencies from pyproject.toml
print_status "Installing real-behavior testing dependencies from pyproject.toml..."
if ! command -v uv &> /dev/null; then
    pip install uv
fi
uv pip install -e ".[test]"
print_success "Dependencies installed from pyproject.toml"

# Set environment variables for testing
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LOG_LEVEL="INFO"

# Configuration for testcontainers
print_status "Configuring testcontainers..."
export TESTCONTAINERS_RYUK_DISABLED=false  # Enable cleanup
export TESTCONTAINERS_CHECKS_DISABLE=false  # Enable Docker checks

# Run different types of real-behavior tests
echo ""
print_status "Starting Real-Behavior Database Error Tests..."
echo "==========================================="

# Basic error classification tests
print_status "1. Testing Real Error Classification..."
pytest -xvs tests/integration/test_psycopg_real_error_behavior.py::test_real_connection_error_classification \
           tests/integration/test_psycopg_real_error_behavior.py::test_real_unique_violation_classification \
           tests/integration/test_psycopg_real_error_behavior.py::test_real_foreign_key_violation_classification \
           tests/integration/test_psycopg_real_error_behavior.py::test_real_syntax_error_classification \
    --tb=short --disable-warnings

if [ $? -eq 0 ]; then
    print_success "✅ Real error classification tests passed"
else
    print_error "❌ Real error classification tests failed"
    exit 1
fi

# Retry mechanism tests with real errors
print_status "2. Testing Retry Mechanisms with Real Errors..."
pytest -xvs tests/integration/test_psycopg_real_error_behavior.py::test_retry_manager_with_real_transient_errors \
    --tb=short --disable-warnings

if [ $? -eq 0 ]; then
    print_success "✅ Real retry mechanism tests passed"
else
    print_error "❌ Real retry mechanism tests failed"
    exit 1
fi

# Circuit breaker tests with real failures
print_status "3. Testing Circuit Breaker with Real Failures..."
pytest -xvs tests/integration/test_psycopg_real_error_behavior.py::test_circuit_breaker_with_real_errors \
    --tb=short --disable-warnings

if [ $? -eq 0 ]; then
    print_success "✅ Real circuit breaker tests passed"
else
    print_error "❌ Real circuit breaker tests failed"
    exit 1
fi

# Error metrics collection tests
print_status "4. Testing Error Metrics Collection..."
pytest -xvs tests/integration/test_psycopg_real_error_behavior.py::test_error_metrics_with_real_errors \
    --tb=short --disable-warnings

if [ $? -eq 0 ]; then
    print_success "✅ Real error metrics tests passed"
else
    print_error "❌ Real error metrics tests failed"
    exit 1
fi

# Comprehensive resilience patterns test
print_status "5. Testing Complete Resilience Patterns..."
pytest -xvs tests/integration/test_psycopg_real_error_behavior.py::test_real_database_resilience_patterns \
    --tb=short --disable-warnings

if [ $? -eq 0 ]; then
    print_success "✅ Real resilience patterns tests passed"
else
    print_error "❌ Real resilience patterns tests failed"
    exit 1
fi

# Load and stress testing
print_status "6. Testing Under Real Database Load..."
pytest -xvs tests/integration/test_psycopg_real_error_behavior.py::test_real_database_load_and_error_rates \
    --tb=short --disable-warnings

if [ $? -eq 0 ]; then
    print_success "✅ Real load testing passed"
else
    print_error "❌ Real load testing failed"
    exit 1
fi

# Run all tests together for full validation
print_status "7. Running Full Test Suite..."
pytest tests/integration/test_psycopg_real_error_behavior.py \
    --tb=short \
    --disable-warnings \
    --verbose \
    --durations=10 \
    --color=yes

if [ $? -eq 0 ]; then
    print_success "🎉 All real-behavior database error tests passed!"
else
    print_error "❌ Some real-behavior tests failed"
    exit 1
fi

# Optional: Run with coverage
print_status "8. Running with Coverage Analysis..."
pytest tests/integration/test_psycopg_real_error_behavior.py \
    --cov=src/prompt_improver/database/error_handling \
    --cov-report=html \
    --cov-report=term-missing \
    --tb=short \
    --disable-warnings

print_success "Coverage report generated in htmlcov/"

# Optional: Run performance benchmarks
print_status "9. Running Performance Benchmarks..."
pytest tests/integration/test_psycopg_real_error_behavior.py \
    --benchmark-only \
    --benchmark-sort=mean \
    --benchmark-min-rounds=5 \
    || print_warning "Benchmark tests skipped (no benchmarks defined)"

echo ""
print_success "🚀 Real-Behavior Database Testing Complete!"
echo "=============================================="
print_status "Key Benefits of This Approach (vs Mocks):"
echo "  ✅ Tests actual database behavior and real errors"
echo "  ✅ Catches schema constraint violations"
echo "  ✅ Validates real retry and resilience patterns"
echo "  ✅ Provides confidence in production behavior"
echo "  ✅ No brittle mock maintenance"
echo "  ✅ Tests actual PostgreSQL error conditions"
echo ""
print_status "Test containers automatically cleaned up by Ryuk"
print_status "View detailed test output above for any issues"

# Optional: Show Docker containers (for debugging)
if [ "$1" = "--show-containers" ]; then
    print_status "Active test containers:"
    docker ps --filter "label=org.testcontainers=true"
fi

exit 0 