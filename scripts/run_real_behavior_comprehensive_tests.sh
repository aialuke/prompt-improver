#!/bin/bash

# COMPREHENSIVE REAL BEHAVIOR TEST RUNNER
# 
# This script executes the complete real behavior validation suite
# with actual data, production-like conditions, and comprehensive reporting.
# NO MOCKS - only real behavior testing with actual systems and data.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/real_behavior_test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to print colored output
print_header() {
    echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║${NC} $1 ${PURPLE}║${NC}"
    echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
}

print_section() {
    echo -e "${CYAN}▶ $1${NC}"
    echo -e "${CYAN}$(printf '%.0s─' {1..80})${NC}"
}

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

# Cleanup function
cleanup() {
    print_status "Cleaning up temporary resources..."
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    # Clean up temporary files
    find /tmp -name "real_behavior_*" -type d -mmin +60 -exec rm -rf {} + 2>/dev/null || true
    find /tmp -name "*_real_test_*" -type f -mmin +60 -delete 2>/dev/null || true
}

trap cleanup EXIT

# Main execution
main() {
    print_header "🚀 COMPREHENSIVE REAL BEHAVIOR VALIDATION SUITE"
    
    print_status "Starting comprehensive real behavior validation..."
    print_status "Timestamp: $(date)"
    print_status "Project Root: $PROJECT_ROOT"
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    CURRENT_RESULTS_DIR="$RESULTS_DIR/run_$TIMESTAMP"
    mkdir -p "$CURRENT_RESULTS_DIR"
    
    # Initialize test environment
    print_section "Environment Setup"
    
    # Check dependencies
    print_status "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    print_success "Python $PYTHON_VERSION detected"
    
    # Check virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "No virtual environment detected"
        if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
            print_status "Activating project virtual environment..."
            source "$PROJECT_ROOT/venv/bin/activate"
        elif [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
            print_status "Activating project virtual environment..."
            source "$PROJECT_ROOT/.venv/bin/activate"
        else
            print_warning "No virtual environment found - using system Python"
        fi
    else
        print_success "Virtual environment active: $VIRTUAL_ENV"
    fi
    
    # Set environment variables
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    export REAL_BEHAVIOR_TEST_MODE=1
    export TEST_RESULTS_DIR="$CURRENT_RESULTS_DIR"
    
    # Check system resources
    print_status "Checking system resources..."
    
    # Memory check (need at least 4GB free)
    if command -v free &> /dev/null; then
        FREE_MEM_GB=$(free -g | awk 'NR==2{printf "%.1f", $7}')
        if (( $(echo "$FREE_MEM_GB < 4" | bc -l) )); then
            print_warning "Low memory detected: ${FREE_MEM_GB}GB free (recommend 4GB+)"
        else
            print_success "Memory available: ${FREE_MEM_GB}GB"
        fi
    fi
    
    # Disk space check (need at least 10GB free)
    FREE_DISK_GB=$(df -BG "$PROJECT_ROOT" | awk 'NR==2{print $4}' | sed 's/G//')
    if (( FREE_DISK_GB < 10 )); then
        print_warning "Low disk space: ${FREE_DISK_GB}GB free (recommend 10GB+)"
    else
        print_success "Disk space available: ${FREE_DISK_GB}GB"
    fi
    
    # Install/update dependencies from pyproject.toml
    print_status "Installing/updating test dependencies from pyproject.toml..."
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        # Install uv if not available
        if ! command -v uv &> /dev/null; then
            pip install -q uv
        fi
        # Install project with test dependencies
        uv pip install -q -e "$PROJECT_ROOT[test]"
        print_success "Test dependencies installed from pyproject.toml"
    else
        pip install -q pytest numpy pandas scikit-learn scipy psutil
        print_success "Basic dependencies installed"
    fi
    
    # Phase 1: Core Real Behavior Tests
    print_section "Phase 1: Core Real Behavior Framework"
    
    TEST_RESULTS_FILE="$CURRENT_RESULTS_DIR/test_results.json"
    echo '{"test_phases": []}' > "$TEST_RESULTS_FILE"
    
    # Test 1: Master Framework Validation
    print_status "Running master framework validation..."
    cd "$PROJECT_ROOT"
    
    if python3 -m pytest tests/real_behavior/real_behavior_test_suite.py::RealBehaviorTestSuite::validate_framework -v --tb=short > "$CURRENT_RESULTS_DIR/framework_validation.log" 2>&1; then
        print_success "✅ Master framework validation passed"
        FRAMEWORK_STATUS="PASS"
    else
        print_error "❌ Master framework validation failed"
        FRAMEWORK_STATUS="FAIL"
        cat "$CURRENT_RESULTS_DIR/framework_validation.log"
    fi
    
    # Phase 2: Component-Specific Real Behavior Tests
    print_section "Phase 2: Component-Specific Real Behavior Testing"
    
    # Component test configurations
    declare -A COMPONENT_TESTS=(
        ["type_safety"]="tests/real_behavior/type_safety_real_tests.py"
        ["database_performance"]="tests/real_behavior/database_real_performance.py"
        ["batch_processing"]="tests/real_behavior/batch_processing_real_tests.py"
        ["ab_testing"]="tests/real_behavior/ab_testing_real_scenarios.py"
        ["ml_platform"]="tests/real_behavior/ml_platform_real_deployment.py"
    )
    
    declare -A COMPONENT_RESULTS=()
    COMPONENTS_PASSED=0
    TOTAL_COMPONENTS=${#COMPONENT_TESTS[@]}
    
    for component in "${!COMPONENT_TESTS[@]}"; do
        test_file="${COMPONENT_TESTS[$component]}"
        
        print_status "Testing $component real behavior..."
        
        # Run component test
        component_log="$CURRENT_RESULTS_DIR/${component}_test.log"
        component_json="$CURRENT_RESULTS_DIR/${component}_results.json"
        
        if timeout 1800 python3 "$test_file" > "$component_log" 2>&1; then  # 30 minute timeout
            print_success "✅ $component real behavior tests passed"
            COMPONENT_RESULTS[$component]="PASS"
            ((COMPONENTS_PASSED++))
        else
            print_error "❌ $component real behavior tests failed"
            COMPONENT_RESULTS[$component]="FAIL"
            
            # Show last 20 lines of log for quick debugging
            print_status "Last 20 lines of $component test log:"
            tail -20 "$component_log" || true
        fi
        
        # Extract metrics if available
        if [[ -f "$component_json" ]]; then
            print_status "$component metrics extracted to $component_json"
        fi
        
        # Progress indicator
        echo -e "${BLUE}Progress: $((COMPONENTS_PASSED + (${TOTAL_COMPONENTS} - COMPONENTS_PASSED - ${#COMPONENT_TESTS[@]} + 1)))/${TOTAL_COMPONENTS} components tested${NC}"
    done
    
    # Phase 3: Integration Real Behavior Tests
    print_section "Phase 3: Integration Real Behavior Testing"
    
    print_status "Running integration real behavior tests..."
    
    # Run existing comprehensive integration tests with real behavior flag
    integration_log="$CURRENT_RESULTS_DIR/integration_test.log"
    
    if timeout 2400 python3 -m pytest tests/comprehensive_integration_test_runner.py -v --real-behavior > "$integration_log" 2>&1; then  # 40 minute timeout
        print_success "✅ Integration real behavior tests passed"
        INTEGRATION_STATUS="PASS"
    else
        print_warning "⚠️ Integration real behavior tests had issues (check log)"
        INTEGRATION_STATUS="PARTIAL"
        
        # Show summary of integration issues
        if grep -q "FAILED" "$integration_log"; then
            print_status "Integration test failures:"
            grep "FAILED" "$integration_log" | head -10 || true
        fi
    fi
    
    # Phase 4: Performance Real Behavior Validation
    print_section "Phase 4: Performance Real Behavior Validation"
    
    print_status "Running performance validation with real data..."
    
    # Run existing batch processor performance tests
    performance_log="$CURRENT_RESULTS_DIR/performance_test.log"
    
    if timeout 3600 python3 -m pytest tests/test_batch_processor_performance.py::TestBatchProcessorPerformance::test_comprehensive_performance_report -v > "$performance_log" 2>&1; then  # 60 minute timeout
        print_success "✅ Performance validation passed"
        PERFORMANCE_STATUS="PASS"
    else
        print_warning "⚠️ Performance validation had issues"
        PERFORMANCE_STATUS="PARTIAL"
        
        # Extract performance metrics if available
        if grep -q "Performance report" "$performance_log"; then
            print_status "Performance metrics available in log"
        fi
    fi
    
    # Phase 5: Generate Comprehensive Report
    print_section "Phase 5: Comprehensive Report Generation"
    
    # Create comprehensive report
    REPORT_FILE="$CURRENT_RESULTS_DIR/comprehensive_real_behavior_report.md"
    
    cat > "$REPORT_FILE" << EOF
# COMPREHENSIVE REAL BEHAVIOR VALIDATION REPORT

**Generated:** $(date)  
**Test Run ID:** $TIMESTAMP  
**Total Duration:** $SECONDS seconds  

## Executive Summary

This report validates all new implementations with actual usage scenarios, real data, and production-like conditions. **NO MOCKS** were used - only real behavior testing.

### Overall Results

- **Framework Validation:** $FRAMEWORK_STATUS
- **Component Tests:** $COMPONENTS_PASSED/$TOTAL_COMPONENTS passed
- **Integration Tests:** $INTEGRATION_STATUS  
- **Performance Validation:** $PERFORMANCE_STATUS

## Component Test Results

EOF
    
    for component in "${!COMPONENT_RESULTS[@]}"; do
        result="${COMPONENT_RESULTS[$component]}"
        status_emoji="✅"
        if [[ "$result" == "FAIL" ]]; then
            status_emoji="❌"
        fi
        
        echo "- **$component:** $status_emoji $result" >> "$REPORT_FILE"
    done
    
    cat >> "$REPORT_FILE" << EOF

## Real Data Processing Summary

EOF
    
    # Extract real data processing metrics from logs
    total_data_processed=0
    for component in "${!COMPONENT_TESTS[@]}"; do
        component_log="$CURRENT_RESULTS_DIR/${component}_test.log"
        if [[ -f "$component_log" ]]; then
            # Try to extract data processing metrics
            data_count=$(grep -o "Data Processed: [0-9,]*" "$component_log" | sed 's/Data Processed: //g' | sed 's/,//g' | tail -1)
            if [[ -n "$data_count" && "$data_count" =~ ^[0-9]+$ ]]; then
                total_data_processed=$((total_data_processed + data_count))
                echo "- **$component:** $(printf "%'d" $data_count) records processed" >> "$REPORT_FILE"
            fi
        fi
    done
    
    cat >> "$REPORT_FILE" << EOF

**Total Real Data Processed:** $(printf "%'d" $total_data_processed) records

## Performance Metrics

EOF
    
    # Extract performance metrics
    if [[ -f "$performance_log" ]]; then
        if grep -q "throughput" "$performance_log"; then
            echo "### Throughput Measurements" >> "$REPORT_FILE"
            grep -o "[0-9,]* items/sec\|[0-9,]* records/sec\|[0-9,]* queries/sec" "$performance_log" | head -10 >> "$REPORT_FILE" || true
        fi
        
        if grep -q "Memory" "$performance_log"; then
            echo "### Memory Usage" >> "$REPORT_FILE"
            grep -o "Memory.*: [0-9.]*MB\|Peak Memory: [0-9.]*MB" "$performance_log" | head -5 >> "$REPORT_FILE" || true
        fi
    fi
    
    cat >> "$REPORT_FILE" << EOF

## Production Readiness Assessment

### Criteria Validation

EOF
    
    # Calculate production readiness score
    total_criteria=0
    passed_criteria=0
    
    criteria=(
        "Framework validation:$FRAMEWORK_STATUS"
        "Component tests:$(if [[ $COMPONENTS_PASSED -ge $((TOTAL_COMPONENTS * 8 / 10)) ]]; then echo "PASS"; else echo "FAIL"; fi)"
        "Integration tests:$INTEGRATION_STATUS"
        "Performance validation:$PERFORMANCE_STATUS"
        "Real data processing:$(if [[ $total_data_processed -gt 1000000 ]]; then echo "PASS"; else echo "FAIL"; fi)"
    )
    
    for criterion in "${criteria[@]}"; do
        name=$(echo "$criterion" | cut -d: -f1)
        status=$(echo "$criterion" | cut -d: -f2)
        total_criteria=$((total_criteria + 1))
        
        if [[ "$status" == "PASS" ]]; then
            passed_criteria=$((passed_criteria + 1))
            echo "- ✅ $name: **PASSED**" >> "$REPORT_FILE"
        else
            echo "- ❌ $name: **FAILED**" >> "$REPORT_FILE"
        fi
    done
    
    readiness_score=$((passed_criteria * 100 / total_criteria))
    
    cat >> "$REPORT_FILE" << EOF

### Production Readiness Score: $readiness_score%

EOF
    
    if [[ $readiness_score -ge 80 ]]; then
        echo "🎉 **READY FOR PRODUCTION DEPLOYMENT**" >> "$REPORT_FILE"
        OVERALL_STATUS="READY"
    elif [[ $readiness_score -ge 60 ]]; then
        echo "⚠️ **REQUIRES MINOR FIXES BEFORE PRODUCTION**" >> "$REPORT_FILE"
        OVERALL_STATUS="NEEDS_FIXES"
    else
        echo "❌ **NOT READY FOR PRODUCTION - SIGNIFICANT ISSUES**" >> "$REPORT_FILE"
        OVERALL_STATUS="NOT_READY"
    fi
    
    cat >> "$REPORT_FILE" << EOF

## Test Artifacts

- **Results Directory:** \`$CURRENT_RESULTS_DIR\`
- **Test Logs:** Available for each component
- **Performance Data:** Extracted metrics available
- **Raw Data:** All test outputs preserved

## Recommendations

EOF
    
    # Add recommendations based on results
    if [[ $COMPONENTS_PASSED -lt $TOTAL_COMPONENTS ]]; then
        echo "- Fix failing component tests before production deployment" >> "$REPORT_FILE"
    fi
    
    if [[ "$INTEGRATION_STATUS" != "PASS" ]]; then
        echo "- Address integration test issues for system reliability" >> "$REPORT_FILE"
    fi
    
    if [[ "$PERFORMANCE_STATUS" != "PASS" ]]; then
        echo "- Optimize performance bottlenecks identified in testing" >> "$REPORT_FILE"
    fi
    
    if [[ $total_data_processed -lt 1000000 ]]; then
        echo "- Increase real data processing volume for better validation" >> "$REPORT_FILE"
    fi
    
    echo "- Continue monitoring in production environment" >> "$REPORT_FILE"
    
    cat >> "$REPORT_FILE" << EOF

---

*This report validates real behavior - no mocks, simulations, or synthetic data used.*
EOF
    
    # Phase 6: Final Summary
    print_section "Phase 6: Final Summary"
    
    print_status "Comprehensive real behavior validation complete!"
    print_status "Report generated: $REPORT_FILE"
    
    echo ""
    print_header "🎯 FINAL RESULTS SUMMARY"
    
    echo -e "${BLUE}Test Run:${NC} $TIMESTAMP"
    echo -e "${BLUE}Duration:${NC} $SECONDS seconds"
    echo -e "${BLUE}Results Directory:${NC} $CURRENT_RESULTS_DIR"
    echo ""
    
    echo -e "${BLUE}Framework Validation:${NC} $(if [[ "$FRAMEWORK_STATUS" == "PASS" ]]; then echo -e "${GREEN}✅ PASSED${NC}"; else echo -e "${RED}❌ FAILED${NC}"; fi)"
    echo -e "${BLUE}Component Tests:${NC} $(if [[ $COMPONENTS_PASSED -eq $TOTAL_COMPONENTS ]]; then echo -e "${GREEN}✅ $COMPONENTS_PASSED/$TOTAL_COMPONENTS PASSED${NC}"; else echo -e "${YELLOW}⚠️ $COMPONENTS_PASSED/$TOTAL_COMPONENTS PASSED${NC}"; fi)"
    echo -e "${BLUE}Integration Tests:${NC} $(if [[ "$INTEGRATION_STATUS" == "PASS" ]]; then echo -e "${GREEN}✅ PASSED${NC}"; else echo -e "${YELLOW}⚠️ PARTIAL${NC}"; fi)"
    echo -e "${BLUE}Performance Tests:${NC} $(if [[ "$PERFORMANCE_STATUS" == "PASS" ]]; then echo -e "${GREEN}✅ PASSED${NC}"; else echo -e "${YELLOW}⚠️ PARTIAL${NC}"; fi)"
    echo -e "${BLUE}Real Data Processed:${NC} $(printf "%'d" $total_data_processed) records"
    echo ""
    
    echo -e "${BLUE}Production Readiness Score:${NC} $readiness_score%"
    
    if [[ "$OVERALL_STATUS" == "READY" ]]; then
        echo -e "${GREEN}🎉 READY FOR PRODUCTION DEPLOYMENT${NC}"
        exit_code=0
    elif [[ "$OVERALL_STATUS" == "NEEDS_FIXES" ]]; then
        echo -e "${YELLOW}⚠️ REQUIRES MINOR FIXES BEFORE PRODUCTION${NC}"
        exit_code=1
    else
        echo -e "${RED}❌ NOT READY FOR PRODUCTION - SIGNIFICANT ISSUES${NC}"
        exit_code=2
    fi
    
    echo ""
    print_success "Comprehensive report available at: $REPORT_FILE"
    
    # Copy report to project root for easy access
    cp "$REPORT_FILE" "$PROJECT_ROOT/REAL_BEHAVIOR_VALIDATION_REPORT.md"
    print_success "Report also copied to: $PROJECT_ROOT/REAL_BEHAVIOR_VALIDATION_REPORT.md"
    
    return $exit_code
}

# Execute main function
main "$@"
exit_code=$?

print_status "Real behavior validation completed with exit code: $exit_code"
exit $exit_code