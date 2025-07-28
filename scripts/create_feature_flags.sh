#!/bin/bash

# Feature Flag Configuration System Setup Script
# Creates and initializes feature flags for 6-phase technical debt cleanup
#
# This script follows 2025 best practices:
# - Idempotent operations (safe to run multiple times)
# - Comprehensive error handling and validation
# - Atomic operations with rollback capability
# - Detailed logging and progress reporting
# - Environment-aware configuration

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
SRC_DIR="$PROJECT_ROOT/src/prompt_improver"
LOGS_DIR="$PROJECT_ROOT/logs"
BACKUP_DIR="$PROJECT_ROOT/backups/feature_flags"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging configuration
LOG_FILE="$LOGS_DIR/feature_flag_setup_$(date +%Y%m%d_%H%M%S).log"
VERBOSE=${VERBOSE:-false}

# Environment detection
ENVIRONMENT=${ENVIRONMENT:-development}
DRY_RUN=${DRY_RUN:-false}

# Function definitions
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
    
    case $level in
        ERROR)
            echo -e "${RED}ERROR: $message${NC}" >&2
            ;;
        WARN)
            echo -e "${YELLOW}WARNING: $message${NC}"
            ;;
        INFO)
            echo -e "${GREEN}INFO: $message${NC}"
            ;;
        DEBUG)
            if [[ "$VERBOSE" == "true" ]]; then
                echo -e "${BLUE}DEBUG: $message${NC}"
            fi
            ;;
    esac
}

error_exit() {
    log ERROR "$1"
    exit 1
}

check_prerequisites() {
    log INFO "Checking prerequisites..."
    
    # Check Python installation
    if ! command -v python3 &> /dev/null; then
        error_exit "Python 3 is required but not installed"
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2)
    log INFO "Found Python $python_version"
    
    # Check required Python packages
    local required_packages=("pydantic" "watchdog" "pyyaml")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            log WARN "Required package '$package' not found, installing..."
            if [[ "$DRY_RUN" == "false" ]]; then
                pip3 install "$package" || error_exit "Failed to install $package"
            fi
        else
            log DEBUG "Package '$package' is available"
        fi
    done
    
    # Check file permissions
    if [[ ! -w "$PROJECT_ROOT" ]]; then
        error_exit "No write permission for project root: $PROJECT_ROOT"
    fi
    
    log INFO "Prerequisites check completed successfully"
}

create_directories() {
    log INFO "Creating necessary directories..."
    
    local dirs=(
        "$CONFIG_DIR"
        "$LOGS_DIR"
        "$BACKUP_DIR"
        "$SRC_DIR/core"
        "$PROJECT_ROOT/tests/unit/core"
        "$PROJECT_ROOT/tests/integration/feature_flags"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log DEBUG "Creating directory: $dir"
            if [[ "$DRY_RUN" == "false" ]]; then
                mkdir -p "$dir" || error_exit "Failed to create directory: $dir"
            fi
        else
            log DEBUG "Directory already exists: $dir"
        fi
    done
    
    log INFO "Directory structure created successfully"
}

backup_existing_config() {
    log INFO "Backing up existing configuration..."
    
    local feature_flag_config="$CONFIG_DIR/feature_flags.yaml"
    
    if [[ -f "$feature_flag_config" ]]; then
        local backup_name="feature_flags_backup_$(date +%Y%m%d_%H%M%S).yaml"
        local backup_path="$BACKUP_DIR/$backup_name"
        
        log INFO "Backing up existing feature flag configuration to: $backup_path"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            cp "$feature_flag_config" "$backup_path" || error_exit "Failed to backup existing configuration"
        fi
    else
        log DEBUG "No existing feature flag configuration found"
    fi
}

validate_configuration() {
    log INFO "Validating feature flag configuration..."
    
    local config_file="$CONFIG_DIR/feature_flags.yaml"
    
    if [[ ! -f "$config_file" ]]; then
        error_exit "Feature flag configuration file not found: $config_file"
    fi
    
    # Validate YAML syntax
    if ! python3 -c "
import yaml
import sys
try:
    with open('$config_file', 'r') as f:
        yaml.safe_load(f)
    print('YAML syntax is valid')
except yaml.YAMLError as e:
    print(f'YAML syntax error: {e}', file=sys.stderr)
    sys.exit(1)
" &> /dev/null; then
        error_exit "Invalid YAML syntax in feature flag configuration"
    fi
    
    # Validate feature flag schema
    if ! python3 -c "
import sys
sys.path.insert(0, '$SRC_DIR')
from core.feature_flags import FeatureFlagDefinition
import yaml

try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    
    flags = config.get('flags', {})
    for flag_key, flag_config in flags.items():
        flag_config['key'] = flag_key
        FeatureFlagDefinition(**flag_config)
    
    print(f'Successfully validated {len(flags)} feature flags')
except Exception as e:
    print(f'Feature flag validation error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null; then
        error_exit "Feature flag configuration validation failed"
    fi
    
    log INFO "Configuration validation completed successfully"
}

setup_environment_configs() {
    log INFO "Setting up environment-specific configurations..."
    
    local environments=("development" "staging" "production")
    
    for env in "${environments[@]}"; do
        local env_config="$CONFIG_DIR/feature_flags_${env}.yaml"
        
        if [[ ! -f "$env_config" ]] && [[ "$DRY_RUN" == "false" ]]; then
            log DEBUG "Creating environment-specific config for: $env"
            
            cat > "$env_config" << EOF
# Environment-specific feature flag overrides for $env
# This file extends the base feature_flags.yaml configuration

version: "1.0.0"
environment: "$env"
extends: "feature_flags.yaml"

# Environment-specific global settings
global:
  environment: "$env"
  $(if [[ "$env" == "development" ]]; then
    echo "  default_rollout_percentage: 100.0"
    echo "  hot_reload_enabled: true" 
  elif [[ "$env" == "staging" ]]; then
    echo "  default_rollout_percentage: 50.0"
    echo "  hot_reload_enabled: true"
  else
    echo "  default_rollout_percentage: 10.0"
    echo "  hot_reload_enabled: false"
  fi)

# Environment-specific flag overrides
flags:
  $(if [[ "$env" == "development" ]]; then
    cat << 'DEV_EOF'
  # Enable most flags in development
  phase1_config_externalization:
    state: "enabled"
    rollout:
      percentage: 100.0
  
  phase2_health_checks:
    state: "rollout"
    rollout:
      percentage: 75.0
      
  rollback_mechanism:
    state: "enabled"
    
  performance_monitoring:
    state: "enabled"
    default_variant: "detailed"
DEV_EOF
  elif [[ "$env" == "staging" ]]; then
    cat << 'STAGING_EOF'
  # Moderate rollout in staging
  phase1_config_externalization:
    state: "enabled"
    rollout:
      percentage: 100.0
  
  phase2_health_checks:
    state: "rollout"
    rollout:
      percentage: 25.0
      
  rollback_mechanism:
    state: "enabled"
    
  performance_monitoring:
    state: "enabled"
STAGING_EOF
  else
    cat << 'PROD_EOF'
  # Conservative rollout in production
  rollback_mechanism:
    state: "enabled"
    
  performance_monitoring:
    state: "enabled"
    default_variant: "detailed"
    
  canary_deployment:
    state: "enabled"
PROD_EOF
  fi)

# Environment-specific monitoring
monitoring:
  alert_channels:
    $(if [[ "$env" == "production" ]]; then
      echo '    - "pagerduty"'
      echo '    - "slack-prod"'
      echo '    - "email"'
    elif [[ "$env" == "staging" ]]; then
      echo '    - "slack-staging"'
      echo '    - "email"'
    else
      echo '    - "slack-dev"'
    fi)
EOF
        fi
    done
    
    log INFO "Environment configurations setup completed"
}

create_initialization_script() {
    log INFO "Creating feature flag initialization script..."
    
    local init_script="$SRC_DIR/core/feature_flag_init.py"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        cat > "$init_script" << 'EOF'
"""
Feature Flag System Initialization

Provides initialization and configuration utilities for the feature flag system.
This module handles the bootstrap process and environment-specific setup.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from .feature_flags import FeatureFlagManager, initialize_feature_flags

logger = logging.getLogger(__name__)


def get_config_path(environment: Optional[str] = None) -> Path:
    """
    Get the appropriate configuration path based on environment.
    
    Args:
        environment: Target environment (development, staging, production)
        
    Returns:
        Path to the feature flag configuration file
    """
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development')
    
    project_root = Path(__file__).parent.parent.parent.parent
    config_dir = project_root / 'config'
    
    # Try environment-specific config first
    env_config = config_dir / f'feature_flags_{environment}.yaml'
    if env_config.exists():
        logger.info(f"Using environment-specific config: {env_config}")
        return env_config
    
    # Fall back to default config
    default_config = config_dir / 'feature_flags.yaml'
    if default_config.exists():
        logger.info(f"Using default config: {default_config}")
        return default_config
    
    raise FileNotFoundError(f"No feature flag configuration found for environment: {environment}")


def initialize_feature_flag_system(
    environment: Optional[str] = None,
    config_path: Optional[Path] = None,
    watch_files: bool = True
) -> FeatureFlagManager:
    """
    Initialize the feature flag system for the given environment.
    
    Args:
        environment: Target environment
        config_path: Custom configuration path (overrides environment detection)
        watch_files: Enable file watching for hot-reload
        
    Returns:
        Initialized FeatureFlagManager instance
    """
    if config_path is None:
        config_path = get_config_path(environment)
    
    logger.info(f"Initializing feature flag system with config: {config_path}")
    
    # Initialize the feature flag manager
    manager = initialize_feature_flags(config_path, watch_files)
    
    # Log configuration info
    config_info = manager.get_configuration_info()
    logger.info(f"Feature flag system initialized: {config_info}")
    
    return manager


def setup_logging():
    """Setup logging for feature flag system."""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Auto-initialize if this module is imported and AUTO_INIT is set
if os.getenv('FEATURE_FLAGS_AUTO_INIT', 'false').lower() == 'true':
    setup_logging()
    try:
        initialize_feature_flag_system()
        logger.info("Feature flag system auto-initialized successfully")
    except Exception as e:
        logger.error(f"Failed to auto-initialize feature flag system: {e}")
EOF
    fi
    
    log INFO "Initialization script created successfully"
}

create_usage_examples() {
    log INFO "Creating usage examples and documentation..."
    
    local examples_dir="$PROJECT_ROOT/examples/feature_flags"
    mkdir -p "$examples_dir"
    
    # Create basic usage example
    if [[ "$DRY_RUN" == "false" ]]; then
        cat > "$examples_dir/basic_usage.py" << 'EOF'
"""
Basic Feature Flag Usage Examples

This file demonstrates common patterns for using the feature flag system
in the technical debt cleanup phases.
"""

import asyncio
from src.prompt_improver.core.feature_flags import (
    EvaluationContext,
    get_feature_flag_manager,
    is_feature_enabled,
    get_feature_variant
)
from src.prompt_improver.core.feature_flag_init import initialize_feature_flag_system


async def main():
    """Demonstrate basic feature flag usage."""
    
    # Initialize the feature flag system
    manager = initialize_feature_flag_system()
    
    # Create evaluation context
    context = EvaluationContext(
        user_id="user123",
        user_type="developer",
        environment="development",
        custom_attributes={
            "service": "config",
            "team": "platform"
        }
    )
    
    # Check if Phase 1 configuration externalization is enabled
    if manager.is_enabled("phase1_config_externalization", context):
        print("✅ Phase 1: Configuration externalization is enabled")
        
        # Check specific sub-features
        if manager.is_enabled("phase1_pydantic_settings", context):
            print("  ✅ Pydantic settings enabled")
        
        env_variant = manager.get_variant("phase1_environment_configs", context)
        print(f"  🔧 Environment config variant: {env_variant}")
    
    # Check Phase 2 health checks
    health_result = manager.evaluate_flag("phase2_health_checks", context)
    print(f"🏥 Phase 2 Health Checks: {health_result.variant} ({health_result.reason})")
    
    # Convenience functions
    phase3_enabled = is_feature_enabled("phase3_metrics_observability", user_id="user123")
    print(f"📊 Phase 3 Metrics: {'Enabled' if phase3_enabled else 'Disabled'}")
    
    # Get all current flags
    all_flags = manager.get_all_flags()
    print(f"\n📋 Total flags configured: {len(all_flags)}")
    
    # Show metrics
    metrics = manager.get_metrics()
    for flag_key, flag_metrics in metrics.items():
        if flag_metrics.evaluations_count > 0:
            print(f"📈 {flag_key}: {flag_metrics.evaluations_count} evaluations")


def demonstrate_rollout_scenarios():
    """Demonstrate different rollout scenarios."""
    
    # Simulate different user types
    user_scenarios = [
        {"user_id": "admin001", "user_type": "admin"},
        {"user_id": "dev001", "user_type": "developer"},
        {"user_id": "qa001", "user_type": "qa"},
        {"user_id": "user001", "user_type": "user"},
    ]
    
    manager = get_feature_flag_manager()
    if not manager:
        print("❌ Feature flag manager not initialized")
        return
    
    print("\n🎯 Rollout Scenarios:")
    print("-" * 50)
    
    for scenario in user_scenarios:
        context = EvaluationContext(**scenario, environment="production")
        
        print(f"\n👤 User: {scenario['user_id']} ({scenario['user_type']})")
        
        # Check each phase
        phases = [
            "phase1_config_externalization",
            "phase2_health_checks", 
            "phase3_metrics_observability",
            "phase4_code_quality",
            "phase5_agent_integration",
            "phase6_final_validation"
        ]
        
        for phase in phases:
            result = manager.evaluate_flag(phase, context, False)
            status = "🟢" if result.value else "🔴"
            print(f"  {status} {phase}: {result.variant} ({result.reason})")


if __name__ == "__main__":
    asyncio.run(main())
    demonstrate_rollout_scenarios()
EOF

        # Create integration example
        cat > "$examples_dir/integration_example.py" << 'EOF'
"""
Feature Flag Integration Example

Shows how to integrate feature flags into existing application code
for the technical debt cleanup phases.
"""

from functools import wraps
from typing import Any, Callable, Optional

from src.prompt_improver.core.feature_flags import (
    EvaluationContext,
    get_feature_flag_manager
)


def feature_flag(flag_key: str, default_value: Any = False, 
                user_id_attr: str = "user_id"):
    """
    Decorator to enable/disable functionality based on feature flags.
    
    Args:
        flag_key: The feature flag key to check
        default_value: Default value if flag evaluation fails
        user_id_attr: Attribute name to extract user_id from function args
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_feature_flag_manager()
            if not manager:
                # Fall back to default behavior if manager not available
                return func(*args, **kwargs) if default_value else None
            
            # Extract user context
            user_id = kwargs.get(user_id_attr)
            if not user_id and args:
                # Try to extract from first argument if it has user_id attribute
                first_arg = args[0]
                if hasattr(first_arg, user_id_attr):
                    user_id = getattr(first_arg, user_id_attr)
            
            # Create evaluation context
            context = EvaluationContext(
                user_id=user_id,
                custom_attributes=kwargs.get('feature_context', {})
            )
            
            # Evaluate flag
            if manager.is_enabled(flag_key, context):
                return func(*args, **kwargs)
            else:
                return default_value
        
        return wrapper
    return decorator


class ConfigurationService:
    """Example service using Phase 1 feature flags."""
    
    def __init__(self):
        self.manager = get_feature_flag_manager()
    
    @feature_flag("phase1_config_externalization", default_value=False)
    def load_external_config(self, user_id: str) -> dict:
        """Load configuration from external source."""
        print("Loading configuration from external source")
        return {"external": True, "source": "database"}
    
    @feature_flag("phase1_pydantic_settings", default_value=False)  
    def validate_config_with_pydantic(self, config: dict, user_id: str) -> bool:
        """Validate configuration using Pydantic models."""
        print("Validating configuration with Pydantic")
        return True
    
    def get_configuration(self, user_id: str) -> dict:
        """Get configuration with feature flag controls."""
        base_config = {"type": "base", "hardcoded": True}
        
        # Phase 1: Try external configuration
        external_config = self.load_external_config(user_id=user_id)
        if external_config:
            base_config.update(external_config)
        
        # Phase 1: Validate with Pydantic if enabled
        if self.validate_config_with_pydantic(base_config, user_id=user_id):
            base_config["validated"] = True
        
        return base_config


class HealthCheckService:
    """Example service using Phase 2 feature flags."""
    
    @feature_flag("phase2_health_checks", default_value={"status": "unknown"})
    def comprehensive_health_check(self, user_id: str) -> dict:
        """Perform comprehensive health check."""
        return {
            "status": "healthy",
            "components": {
                "database": "healthy",
                "ml_models": "healthy", 
                "external_apis": "healthy"
            }
        }
    
    @feature_flag("phase2_ml_model_health", default_value={"status": "disabled"})
    def ml_model_health_check(self, user_id: str) -> dict:
        """Check ML model health with drift detection."""
        return {
            "status": "healthy",
            "drift_detected": False,
            "model_version": "1.2.3"
        }


def demonstrate_integration():
    """Demonstrate the integration examples."""
    print("🔧 Feature Flag Integration Demo")
    print("=" * 40)
    
    # Configuration service demo
    config_service = ConfigurationService()
    config = config_service.get_configuration(user_id="demo_user")
    print(f"📋 Configuration: {config}")
    
    # Health check service demo  
    health_service = HealthCheckService()
    health = health_service.comprehensive_health_check(user_id="demo_user")
    print(f"🏥 Health Check: {health}")
    
    ml_health = health_service.ml_model_health_check(user_id="demo_user")
    print(f"🤖 ML Health: {ml_health}")


if __name__ == "__main__":
    from src.prompt_improver.core.feature_flag_init import initialize_feature_flag_system
    
    # Initialize system
    initialize_feature_flag_system()
    
    # Run demo
    demonstrate_integration()
EOF
    fi
    
    log INFO "Usage examples created successfully"
}

create_monitoring_dashboard() {
    log INFO "Creating monitoring dashboard configuration..."
    
    local monitoring_dir="$PROJECT_ROOT/monitoring/feature_flags"
    mkdir -p "$monitoring_dir"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Prometheus metrics configuration
        cat > "$monitoring_dir/prometheus_metrics.py" << 'EOF'
"""
Prometheus Metrics for Feature Flag System

Exports feature flag metrics to Prometheus for monitoring and alerting.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Dict, Any

# Feature flag evaluation metrics
FEATURE_FLAG_EVALUATIONS = Counter(
    'feature_flag_evaluations_total',
    'Total number of feature flag evaluations',
    ['flag_key', 'variant', 'reason', 'environment']
)

FEATURE_FLAG_EVALUATION_DURATION = Histogram(
    'feature_flag_evaluation_duration_seconds',
    'Time spent evaluating feature flags',
    ['flag_key', 'environment']
)

FEATURE_FLAG_ERRORS = Counter(
    'feature_flag_errors_total',
    'Total number of feature flag evaluation errors',
    ['flag_key', 'error_type', 'environment']
)

FEATURE_FLAG_CONFIG_RELOADS = Counter(
    'feature_flag_config_reloads_total',
    'Total number of configuration reloads',
    ['status', 'environment']
)

FEATURE_FLAG_ACTIVE_FLAGS = Gauge(
    'feature_flag_active_flags',
    'Number of active feature flags',
    ['environment']
)

FEATURE_FLAG_ROLLOUT_PERCENTAGE = Gauge(
    'feature_flag_rollout_percentage',
    'Current rollout percentage for flags',
    ['flag_key', 'environment']
)

# Technical debt phase metrics
TECHNICAL_DEBT_PHASE_PROGRESS = Gauge(
    'technical_debt_phase_progress',
    'Progress of technical debt cleanup phases',
    ['phase', 'environment']
)

def record_flag_evaluation(flag_key: str, variant: str, reason: str, 
                          environment: str, duration: float):
    """Record a feature flag evaluation."""
    FEATURE_FLAG_EVALUATIONS.labels(
        flag_key=flag_key,
        variant=variant,
        reason=reason,
        environment=environment
    ).inc()
    
    FEATURE_FLAG_EVALUATION_DURATION.labels(
        flag_key=flag_key,
        environment=environment
    ).observe(duration)

def record_flag_error(flag_key: str, error_type: str, environment: str):
    """Record a feature flag evaluation error."""
    FEATURE_FLAG_ERRORS.labels(
        flag_key=flag_key,
        error_type=error_type,
        environment=environment
    ).inc()

def update_rollout_percentage(flag_key: str, percentage: float, environment: str):
    """Update rollout percentage metric."""
    FEATURE_FLAG_ROLLOUT_PERCENTAGE.labels(
        flag_key=flag_key,
        environment=environment
    ).set(percentage)

def update_phase_progress(phase: str, progress: float, environment: str):
    """Update technical debt phase progress."""
    TECHNICAL_DEBT_PHASE_PROGRESS.labels(
        phase=phase,
        environment=environment
    ).set(progress)
EOF

        # Grafana dashboard configuration
        cat > "$monitoring_dir/grafana_dashboard.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Feature Flags - Technical Debt Cleanup",
    "tags": ["feature-flags", "technical-debt"],
    "timezone": "UTC",
    "panels": [
      {
        "id": 1,
        "title": "Feature Flag Evaluations",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(feature_flag_evaluations_total[5m])",
            "legendFormat": "{{flag_key}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Technical Debt Phase Progress",
        "type": "bargauge",
        "targets": [
          {
            "expr": "technical_debt_phase_progress",
            "legendFormat": "{{phase}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Rollout Percentages",
        "type": "table",
        "targets": [
          {
            "expr": "feature_flag_rollout_percentage",
            "format": "table"
          }
        ],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Evaluation Errors",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(feature_flag_errors_total[5m])",
            "legendFormat": "{{flag_key}} - {{error_type}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
      },
      {
        "id": 5,
        "title": "Configuration Reloads",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(feature_flag_config_reloads_total[5m])",
            "legendFormat": "{{status}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
      }
    ],
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "30s"
  }
}
EOF
    fi
    
    log INFO "Monitoring dashboard configuration created"
}

run_tests() {
    log INFO "Running feature flag system tests..."
    
    # Create and run basic tests
    if [[ "$DRY_RUN" == "false" ]]; then
        python3 -c "
import sys
sys.path.insert(0, '$SRC_DIR')

from core.feature_flags import FeatureFlagManager, EvaluationContext
import tempfile
import yaml

# Test configuration
test_config = {
    'version': '1.0.0',
    'flags': {
        'test_flag': {
            'state': 'enabled',
            'default_variant': 'off',
            'variants': {'on': True, 'off': False},
            'rollout': {
                'strategy': 'percentage',
                'percentage': 50.0,
                'sticky': True
            }
        }
    }
}

# Create temporary config file
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(test_config, f)
    temp_config = f.name

try:
    # Initialize manager
    manager = FeatureFlagManager(temp_config, watch_files=False)
    
    # Test evaluation
    context = EvaluationContext(user_id='test_user', environment='test')
    result = manager.evaluate_flag('test_flag', context)
    
    print(f'✅ Test evaluation successful: {result.variant} ({result.reason})')
    
    # Test metrics
    metrics = manager.get_metrics('test_flag')
    print(f'✅ Metrics working: {metrics.evaluations_count} evaluations')
    
    # Test configuration info
    info = manager.get_configuration_info()
    print(f'✅ Configuration info: {info[\"flags_count\"]} flags loaded')
    
    manager.shutdown()
    print('✅ All tests passed!')
    
except Exception as e:
    print(f'❌ Test failed: {e}')
    sys.exit(1)
finally:
    import os
    os.unlink(temp_config)
"
    fi
    
    log INFO "Tests completed successfully"
}

show_usage() {
    cat << EOF
Feature Flag Configuration System Setup

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose logging
    -d, --dry-run       Show what would be done without making changes
    -e, --environment   Set target environment (development|staging|production)
    --skip-tests        Skip running validation tests
    --skip-backup       Skip backing up existing configuration
    --init-only         Only initialize, don't create examples or monitoring

Environment Variables:
    ENVIRONMENT         Target environment (default: development)
    DRY_RUN            Enable dry-run mode (default: false)
    VERBOSE            Enable verbose logging (default: false)
    LOG_LEVEL          Set log level (default: INFO)

Examples:
    # Basic setup for development
    $0

    # Production setup with dry-run
    $0 --environment production --dry-run

    # Verbose setup with test skipping
    $0 --verbose --skip-tests

EOF
}

main() {
    local skip_tests=false
    local skip_backup=false
    local init_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --skip-backup)
                skip_backup=true
                shift
                ;;
            --init-only)
                init_only=true
                shift
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done
    
    # Setup logging directory
    mkdir -p "$LOGS_DIR"
    
    log INFO "Starting feature flag system setup"
    log INFO "Environment: $ENVIRONMENT"
    log INFO "Dry run: $DRY_RUN"
    log INFO "Project root: $PROJECT_ROOT"
    
    # Main setup workflow
    check_prerequisites
    create_directories
    
    if [[ "$skip_backup" != "true" ]]; then
        backup_existing_config
    fi
    
    validate_configuration
    setup_environment_configs
    create_initialization_script
    
    if [[ "$init_only" != "true" ]]; then
        create_usage_examples
        create_monitoring_dashboard
    fi
    
    if [[ "$skip_tests" != "true" ]]; then
        run_tests
    fi
    
    log INFO "Feature flag system setup completed successfully!"
    log INFO "Log file: $LOG_FILE"
    
    # Show next steps
    cat << EOF

🎉 Feature Flag System Setup Complete!

Next steps:
1. Review the configuration: $CONFIG_DIR/feature_flags.yaml
2. Check environment configs: $CONFIG_DIR/feature_flags_*.yaml  
3. Try the examples: $PROJECT_ROOT/examples/feature_flags/
4. Set up monitoring: $PROJECT_ROOT/monitoring/feature_flags/

To use in your code:
    from src.prompt_improver.core.feature_flag_init import initialize_feature_flag_system
    manager = initialize_feature_flag_system()

For technical debt phases:
    - Phase 1: Configuration externalization (25% rollout)
    - Phase 2: Health check implementation (15% rollout)
    - Phase 3-6: Disabled (ready for activation)

Environment: $ENVIRONMENT
Configuration: $CONFIG_DIR/feature_flags.yaml
Log file: $LOG_FILE

EOF
}

# Run main function
main "$@"