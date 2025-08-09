#!/bin/bash
#
# SystemD Service Installation Script
# Installs and configures MCP Server systemd services
#
# Usage:
#   sudo ./systemd/install-systemd-services.sh [OPTIONS]
#
# Options:
#   --mode <stdio|http|both>  Install specific service mode (default: stdio)
#   --user <username>         Service user (default: mcp-server)
#   --install-path <path>     Installation path (default: /opt/prompt-improver)
#   --dry-run                 Show what would be done without making changes
#   --uninstall               Remove installed services
#

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default configuration
MODE="stdio"
SERVICE_USER="mcp-server"  
INSTALL_PATH="/opt/prompt-improver"
DRY_RUN=false
UNINSTALL=false

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

error_exit() {
    log_error "$1"
    exit 1
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root (use sudo)"
    fi
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                MODE="$2"
                if [[ "$MODE" != "stdio" && "$MODE" != "http" && "$MODE" != "both" ]]; then
                    error_exit "Invalid mode: $MODE. Must be 'stdio', 'http', or 'both'"
                fi
                shift 2
                ;;
            --user)
                SERVICE_USER="$2"
                shift 2
                ;;
            --install-path)
                INSTALL_PATH="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --uninstall)
                UNINSTALL=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done
}

show_usage() {
    cat << EOF
SystemD Service Installation Script

USAGE:
    sudo $0 [OPTIONS]

OPTIONS:
    --mode <stdio|http|both>  Install specific service mode (default: stdio)
    --user <username>         Service user (default: mcp-server)
    --install-path <path>     Installation path (default: /opt/prompt-improver)
    --dry-run                 Show what would be done without making changes
    --uninstall               Remove installed services
    --help                    Show this help message

EXAMPLES:
    sudo $0                                    # Install stdio service
    sudo $0 --mode http                        # Install HTTP service
    sudo $0 --mode both                        # Install both services
    sudo $0 --uninstall                        # Remove all services
    sudo $0 --dry-run --mode both              # Preview installation

SYSTEMD COMMANDS AFTER INSTALLATION:
    sudo systemctl start mcp-server           # Start stdio service
    sudo systemctl start mcp-server-http      # Start HTTP service
    sudo systemctl status mcp-server          # Check status
    sudo journalctl -u mcp-server -f          # View logs
    sudo systemctl enable mcp-server          # Auto-start on boot
EOF
}

# Create service user if it doesn't exist
create_service_user() {
    if ! id "$SERVICE_USER" &>/dev/null; then
        log_info "Creating service user: $SERVICE_USER"
        
        if [[ "$DRY_RUN" == true ]]; then
            log_info "DRY RUN: Would create user $SERVICE_USER"
            return
        fi
        
        useradd --system --no-create-home --shell /bin/false "$SERVICE_USER"
        log_success "Created service user: $SERVICE_USER"
    else
        log_info "Service user already exists: $SERVICE_USER"
    fi
}

# Create directory structure
create_directories() {
    local dirs=(
        "$INSTALL_PATH"
        "$INSTALL_PATH/logs"
        "$INSTALL_PATH/tmp"
        "/etc/mcp-server"
        "/var/log/mcp-server"
    )
    
    for dir in "${dirs[@]}"; do
        log_info "Creating directory: $dir"
        
        if [[ "$DRY_RUN" == true ]]; then
            log_info "DRY RUN: Would create directory $dir"
            continue
        fi
        
        mkdir -p "$dir"
        chown "$SERVICE_USER:$SERVICE_USER" "$dir"
        chmod 755 "$dir"
    done
    
    log_success "Directory structure created"
}

# Install application files
install_application() {
    log_info "Installing application files to $INSTALL_PATH"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would copy application files"
        return
    fi
    
    # Copy application files (excluding development files)
    rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
          --exclude='.pytest_cache' --exclude='tests/' --exclude='docs/' \
          "$PROJECT_ROOT/" "$INSTALL_PATH/"
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_PATH"
    chmod +x "$INSTALL_PATH/scripts/start_mcp_native.sh"
    
    log_success "Application files installed"
}

# Install systemd service files
install_service_files() {
    log_info "Installing systemd service files..."
    
    local services=()
    if [[ "$MODE" == "stdio" || "$MODE" == "both" ]]; then
        services+=("mcp-server.service")
    fi
    if [[ "$MODE" == "http" || "$MODE" == "both" ]]; then
        services+=("mcp-server-http.service")
    fi
    services+=("prompt-enhancement.target")
    
    for service in "${services[@]}"; do
        log_info "Installing $service"
        
        if [[ "$DRY_RUN" == true ]]; then
            log_info "DRY RUN: Would install $service"
            continue
        fi
        
        # Update service file with actual paths
        sed -e "s|/opt/prompt-improver|$INSTALL_PATH|g" \
            -e "s|User=mcp-server|User=$SERVICE_USER|g" \
            -e "s|Group=mcp-server|Group=$SERVICE_USER|g" \
            "$SCRIPT_DIR/$service" > "/etc/systemd/system/$service"
            
        chmod 644 "/etc/systemd/system/$service"
    done
    
    log_success "SystemD service files installed"
}

# Create environment configuration
create_environment_config() {
    local env_file="/etc/mcp-server/environment"
    
    log_info "Creating environment configuration: $env_file"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would create environment file"
        return
    fi
    
    cat > "$env_file" << EOF
# MCP Server Environment Configuration
# Edit this file to customize your deployment

# Database Configuration (Required)
# DATABASE_URL=postgresql://mcp_user:secure_password@localhost:5432/prompt_improver_db

# Redis Configuration (Required) 
# REDIS_URL=redis://localhost:6379/0

# Logging Configuration
MCP_LOG_LEVEL=INFO
FASTMCP_LOG_LEVEL=INFO

# Performance Settings
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
MALLOC_ARENA_MAX=2

# Security Settings (Optional)
# MCP_RATE_LIMIT_ENABLED=true
# MCP_AUTH_ENABLED=false

# Monitoring Settings
PERFORMANCE_MONITORING=true
HEALTH_CHECK_ENABLED=true
EOF
    
    chmod 640 "$env_file"
    chown root:"$SERVICE_USER" "$env_file"
    
    log_success "Environment configuration created"
}

# Reload systemd and enable services
enable_services() {
    log_info "Reloading systemd daemon..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would reload systemd and enable services"
        return
    fi
    
    systemctl daemon-reload
    
    # Enable services based on mode
    if [[ "$MODE" == "stdio" || "$MODE" == "both" ]]; then
        log_info "Enabling mcp-server.service"
        systemctl enable mcp-server.service
    fi
    
    if [[ "$MODE" == "http" || "$MODE" == "both" ]]; then
        log_info "Enabling mcp-server-http.service"
        systemctl enable mcp-server-http.service
    fi
    
    log_info "Enabling prompt-enhancement.target"
    systemctl enable prompt-enhancement.target
    
    log_success "Services enabled for auto-start"
}

# Uninstall services
uninstall_services() {
    log_info "Uninstalling MCP Server systemd services..."
    
    local services=(
        "mcp-server.service"
        "mcp-server-http.service"  
        "prompt-enhancement.target"
    )
    
    for service in "${services[@]}"; do
        if systemctl is-enabled "$service" &>/dev/null; then
            log_info "Stopping and disabling $service"
            
            if [[ "$DRY_RUN" == false ]]; then
                systemctl stop "$service" || true
                systemctl disable "$service"
                rm -f "/etc/systemd/system/$service"
            fi
        fi
    done
    
    if [[ "$DRY_RUN" == false ]]; then
        systemctl daemon-reload
    fi
    
    log_success "Services uninstalled"
}

# Validate installation
validate_installation() {
    log_info "Validating installation..."
    
    local services=()
    if [[ "$MODE" == "stdio" || "$MODE" == "both" ]]; then
        services+=("mcp-server.service")
    fi
    if [[ "$MODE" == "http" || "$MODE" == "both" ]]; then
        services+=("mcp-server-http.service")
    fi
    
    local validation_failed=false
    
    for service in "${services[@]}"; do
        if systemctl is-enabled "$service" &>/dev/null; then
            log_success "$service is enabled"
        else
            log_error "$service is not enabled"
            validation_failed=true
        fi
        
        # Test service file syntax
        if systemctl cat "$service" &>/dev/null; then
            log_success "$service configuration is valid"
        else
            log_error "$service configuration is invalid"
            validation_failed=true
        fi
    done
    
    # Check user exists
    if id "$SERVICE_USER" &>/dev/null; then
        log_success "Service user $SERVICE_USER exists"
    else
        log_error "Service user $SERVICE_USER does not exist"
        validation_failed=true
    fi
    
    # Check directories
    if [[ -d "$INSTALL_PATH" ]]; then
        log_success "Installation directory exists: $INSTALL_PATH"
    else
        log_error "Installation directory missing: $INSTALL_PATH"
        validation_failed=true
    fi
    
    if [[ "$validation_failed" == true ]]; then
        error_exit "Installation validation failed"
    fi
    
    log_success "Installation validation completed successfully"
}

# Show post-installation instructions
show_post_install_instructions() {
    cat << EOF

${GREEN}Installation completed successfully!${NC}

${BLUE}Next Steps:${NC}
1. Configure environment variables in /etc/mcp-server/environment
2. Ensure PostgreSQL and Redis are running and accessible
3. Start the service:

   ${YELLOW}# Start service${NC}
EOF

    if [[ "$MODE" == "stdio" ]]; then
        echo "   sudo systemctl start mcp-server"
    elif [[ "$MODE" == "http" ]]; then
        echo "   sudo systemctl start mcp-server-http"
    else
        echo "   sudo systemctl start mcp-server        # For stdio mode"
        echo "   sudo systemctl start mcp-server-http   # For HTTP mode"
    fi
    
    cat << EOF
   
   ${YELLOW}# Check status${NC}
   sudo systemctl status mcp-server
   
   ${YELLOW}# View logs${NC}
   sudo journalctl -u mcp-server -f
   
   ${YELLOW}# Auto-start on boot${NC}
   sudo systemctl enable mcp-server

${BLUE}Configuration Files:${NC}
   Service: /etc/systemd/system/mcp-server.service
   Environment: /etc/mcp-server/environment
   Application: $INSTALL_PATH
   Logs: /var/log/mcp-server/

${BLUE}Management Commands:${NC}
   sudo systemctl restart mcp-server
   sudo systemctl stop mcp-server
   sudo systemctl reload mcp-server

For HTTP mode, replace 'mcp-server' with 'mcp-server-http' in commands above.
EOF
}

# Main execution
main() {
    log_info "MCP Server SystemD Installation Script"
    
    # Parse arguments
    parse_args "$@"
    
    # Check prerequisites
    check_root
    
    if [[ "$UNINSTALL" == true ]]; then
        uninstall_services
        exit 0
    fi
    
    # Installation steps
    create_service_user
    create_directories
    install_application
    install_service_files
    create_environment_config
    enable_services
    
    if [[ "$DRY_RUN" == false ]]; then
        validate_installation
        show_post_install_instructions
    else
        log_info "Dry run completed - no changes made"
    fi
}

# Execute main function
main "$@"