# GitHub Copilot Configuration Guide

This guide provides comprehensive setup instructions for optimizing GitHub Copilot's performance in the Prompt Improver project.

## Quick Setup Checklist

- [ ] Copy `.github/copilot-instructions.md` to your repository root
- [ ] Copy all `.github/instructions/*.instructions.md` files for targeted guidance
- [ ] Set up the GitHub Actions workflow (`.github/workflows/copilot-setup-steps.yml`)
- [ ] Configure MCP servers using `.github/copilot-mcp-config.json`
- [ ] Set up environment variables and project dependencies
- [ ] Test the configuration with a sample issue assignment

## File Structure Overview

```
.github/
├── copilot-instructions.md           # Main project instructions
├── copilot-mcp-config.json          # MCP server configuration
├── workflows/
│   └── copilot-setup-steps.yml      # Environment setup for Copilot
└── instructions/                     # Targeted file-specific instructions
    ├── api-routes.instructions.md
    ├── services.instructions.md
    ├── database-models.instructions.md
    ├── testing.instructions.md
    └── cache-services.instructions.md
```

## Configuration Files Explanation

### 1. Main Instructions (`copilot-instructions.md`)

This file provides GitHub Copilot with comprehensive context about your project:

- **Project overview and architecture**
- **Technology stack and dependencies**
- **Coding standards and best practices**
- **Development workflow patterns**
- **Performance and security guidelines**

**Key sections include:**
- Project structure and organization
- Technology stack overview
- Coding guidelines and standards
- Common patterns and best practices
- Performance considerations
- Security requirements

### 2. Targeted Instructions (`.github/instructions/`)

These files provide specific guidance for different types of files:

#### `api-routes.instructions.md`
- FastAPI route development patterns
- Authentication and authorization
- Error handling and validation
- Performance optimization

#### `services.instructions.md`
- Service layer architecture
- Dependency injection patterns
- Async/await best practices
- Error handling and logging

#### `database-models.instructions.md`
- SQLModel patterns and relationships
- Database indexing strategies
- Field validation and constraints
- Migration considerations

#### `testing.instructions.md`
- Test structure and organization
- Async testing patterns
- Mocking and fixture usage
- Coverage requirements

#### `cache-services.instructions.md`
- Redis caching patterns
- TTL management strategies
- Cache invalidation techniques
- Performance monitoring

### 3. Environment Setup (`copilot-setup-steps.yml`)

This GitHub Actions workflow automatically sets up the development environment when Copilot is assigned an issue:

- **Service Dependencies**: PostgreSQL, Redis
- **Python Environment**: Python 3.11 with uv package manager
- **Development Tools**: Ruff, pytest, pre-commit
- **Database Setup**: Schema creation and seed data
- **Validation**: Connection tests and code quality checks

### 4. MCP Server Configuration (`copilot-mcp-config.json`)

Extends Copilot's capabilities with specialized tools:

- **Prompt Evaluation**: Custom MCP server for prompt analysis
- **GitHub Integration**: Repository and issue management
- **Database Operations**: PostgreSQL query and schema tools
- **Observability**: Performance monitoring and metrics

## Environment Variables Setup

Create a `.env` file with the following variables:

```bash
# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/prompt_improver
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USERNAME=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DATABASE=prompt_improver

# Redis Configuration
REDIS_URL=redis://localhost:6379

# GitHub Integration (optional)
GITHUB_TOKEN=ghp_your_token_here
GITHUB_OWNER=your_username
GITHUB_REPO=prompt-improver

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
SECRET_KEY=your_secret_key_here

# Observability (optional)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=prompt-improver
METRICS_ENDPOINT=http://localhost:8080/metrics
```

## Project-Specific Customizations

### Technology Stack Alignment

The configuration is specifically tailored for:

- **Python 3.11+** with async/await patterns
- **FastAPI** for REST API development
- **PostgreSQL** with asyncpg and SQLModel
- **Redis** for caching with coredis
- **pytest** for testing with async support
- **Ruff** for code formatting and linting
- **OpenTelemetry** for observability

### Code Quality Standards

- **Type annotations** required for all functions
- **Async patterns** for all I/O operations
- **Error handling** with proper exception types
- **Logging** with correlation IDs
- **Testing** with >90% coverage requirement
- **Security** with input validation and JWT auth

### Performance Patterns

- **Connection pooling** for database and Redis
- **Caching strategies** with TTL management
- **Async operations** throughout the stack
- **Monitoring** with OpenTelemetry integration
- **Optimization** for database queries and API responses

## Testing the Configuration

### 1. Validate File Placement

Ensure all configuration files are in the correct locations:

```bash
# Check main instructions
ls -la .github/copilot-instructions.md

# Check targeted instructions
ls -la .github/instructions/*.instructions.md

# Check workflow
ls -la .github/workflows/copilot-setup-steps.yml

# Check MCP configuration
ls -la .github/copilot-mcp-config.json
```

### 2. Test GitHub Actions Workflow

You can manually trigger the setup workflow to ensure it works:

1. Go to your repository's Actions tab
2. Find "Copilot Setup Steps" workflow
3. Click "Run workflow"
4. Monitor the execution and verify all steps pass

### 3. Create a Test Issue

Create a simple issue to test Copilot's understanding:

```markdown
Title: Add health check endpoint

Description:
Create a new health check endpoint at `/health` that returns the status of the database and Redis connections.

Requirements:
- Endpoint should return HTTP 200 when all services are healthy
- Endpoint should return HTTP 503 when any service is unhealthy
- Include response time metrics for each service
- Add appropriate logging and error handling
- Include unit and integration tests

This will help test Copilot's understanding of our API patterns and testing requirements.
```

### 4. Assign to Copilot

1. Assign the issue to `@copilot`
2. Monitor the pull request creation
3. Review the generated code for adherence to your patterns
4. Check the "View session" button to see Copilot's thought process

## Troubleshooting

### Common Issues

1. **Environment Setup Failures**
   - Check GitHub Actions workflow logs
   - Verify service dependencies are available
   - Ensure environment variables are set correctly

2. **MCP Server Connection Issues**
   - Verify MCP server dependencies are installed
   - Check environment variable configuration
   - Test MCP servers independently

3. **Code Quality Issues**
   - Review targeted instruction files
   - Update patterns based on generated code feedback
   - Ensure pre-commit hooks are configured

### Optimization Tips

1. **Refine Instructions Based on Results**
   - Monitor generated code quality
   - Update instruction files with common corrections
   - Add specific patterns that Copilot should follow

2. **Environment Performance**
   - Optimize GitHub Actions workflow for faster setup
   - Cache dependencies where possible
   - Use appropriate resource allocation

3. **MCP Server Enhancement**
   - Add project-specific MCP tools
   - Customize server configurations for your needs
   - Monitor tool usage and effectiveness

## Maintenance

### Regular Updates

1. **Review Generated Code**: Regularly review Copilot-generated code to identify patterns that need refinement
2. **Update Instructions**: Update instruction files based on new patterns or requirements
3. **Environment Sync**: Keep the setup workflow in sync with project dependencies
4. **MCP Server Updates**: Update MCP server configurations as new capabilities are added

### Performance Monitoring

1. **Track Success Rate**: Monitor how often Copilot-generated PRs are accepted
2. **Code Quality Metrics**: Track adherence to coding standards and patterns
3. **Development Velocity**: Measure impact on development speed and productivity
4. **Error Patterns**: Identify common mistakes and update instructions accordingly

## Advanced Configuration

### Custom MCP Servers

You can create project-specific MCP servers for specialized functionality:

```json
{
  "mcpServers": {
    "custom-prompt-analyzer": {
      "type": "local",
      "command": "python",
      "args": ["-m", "custom_mcp_servers.prompt_analyzer"],
      "env": {
        "MODEL_PATH": "/path/to/your/model"
      }
    }
  }
}
```

### Repository-Specific Rules

Add repository-specific rules to the main instructions file:

```markdown
## Project-Specific Rules

### Prompt Analysis Domain
- All prompt analysis functions must include confidence scores
- Use standardized prompt quality metrics (clarity, specificity, completeness)
- Include improvement suggestions with actionable recommendations

### ML Model Integration
- Always version ML models using MLflow
- Include model performance metrics in responses
- Implement graceful fallbacks for model failures
```

### Integration Patterns

Define specific integration patterns for your project:

```markdown
## Integration Patterns

### Database + Cache Pattern
Always implement the database-cache pattern for expensive operations:
1. Check cache first
2. Fallback to database if cache miss
3. Update cache with result
4. Include proper error handling for both layers
```

This comprehensive configuration will help GitHub Copilot understand your project's specific requirements and generate higher-quality code that follows your established patterns and standards.
