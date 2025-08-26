# Comprehensive Architectural Validator

A custom architectural validation script that enforces key architectural boundaries, serving as a replacement for import-linter when it has internal parsing issues.

## Features

The validator checks for 5 key architectural compliance areas:

### 1. 🔄 Circular Import Detection
- Detects circular dependencies between modules
- Uses DFS algorithm to find import cycles
- Reports complete cycle paths for easy debugging

### 2. 🏗️ Clean Architecture Layer Validation
- Enforces proper layer dependencies (Presentation → Application → Domain → Repository → Infrastructure)
- Validates that layers only depend on allowed layers
- Prevents architectural drift over time

### 3. 🗃️ Repository Pattern Enforcement  
- Detects direct database imports in service/presentation layers
- Ensures proper use of repository interfaces and dependency injection
- Maintains separation of concerns

### 4. 🔧 Protocol Consolidation
- Checks for imports from deprecated protocol locations
- Ensures migration to consolidated protocol structure is complete
- Prevents usage of old `core.protocols` imports

### 5. 🎯 Domain Purity
- Prevents core/domain layers from importing heavy external libraries
- Calculates startup time penalties for violations
- Enforces lazy loading patterns for performance-critical imports

## Usage

### Basic Usage

```bash
# Run validation with text output
python3 scripts/comprehensive_architectural_validator.py

# Run with JSON output
python3 scripts/comprehensive_architectural_validator.py --format json

# Run with Markdown output  
python3 scripts/comprehensive_architectural_validator.py --format markdown
```

### Advanced Usage

```bash
# Strict mode - fail with exit code 1 if any violations found
python3 scripts/comprehensive_architectural_validator.py --strict

# Verbose logging
python3 scripts/comprehensive_architectural_validator.py --verbose

# Custom project root
python3 scripts/comprehensive_architectural_validator.py --project-root /path/to/project

# Combine options
python3 scripts/comprehensive_architectural_validator.py --format json --strict --verbose
```

### Integration with CI/CD

Add to your GitHub Actions workflow:

```yaml
- name: Architectural Validation
  run: |
    python3 scripts/comprehensive_architectural_validator.py --format json --strict
```

## Output Formats

### Text Format (Default)
Human-readable report with clear violation descriptions and suggestions.

### JSON Format
Machine-readable format for integration with tools and dashboards:

```json
{
  "summary": {
    "is_compliant": false,
    "total_violations": 317,
    "critical_violations": 16,
    "compliance_ratio": 0.79,
    "analysis_duration": 1.05,
    "files_analyzed": 639
  },
  "violations": [...],
  "violation_summary": {...}
}
```

### Markdown Format
Documentation-friendly format for reports and issue tracking.

## Violation Types and Severity

### Critical Violations (❌ Must Fix)
- **torch imports in domain**: Adds ~1007ms startup penalty
- **transformers imports in domain**: Adds ~892ms startup penalty  
- Other heavy ML libraries with >500ms impact

### High Violations (⚠️ Should Fix)
- Circular imports
- Layer violations
- Repository pattern violations
- Direct database imports in presentation/application layers

### Medium/Low Violations (💡 Nice to Fix)
- Protocol consolidation issues
- Minor architectural inconsistencies

## Performance Impact

The validator analyzes architectural violations and estimates their performance impact:

- **Startup Time Penalties**: Heavy imports in domain layers
- **Memory Usage**: Estimated penalty for unnecessary dependencies
- **Maintenance Cost**: Coupling and complexity issues

## Example Violations Found

### Critical Domain Purity Violation
```
📁 src/prompt_improver/rule_engine/prompt_analyzer.py:19
   Heavy import torch in domain layer
   💡 Use lazy loading or move import to infrastructure layer
   ⚠️  Adds ~1007ms to startup time
```

### Repository Pattern Violation
```
📁 src/prompt_improver/api/analytics_endpoints.py:25
   Direct database import in presentation layer: prompt_improver.database
   💡 Use repository interfaces and dependency injection instead
```

### Circular Import
```
📁 src/prompt_improver/performance/monitoring/performance_benchmark_factory.py:1
   Circular import: performance_benchmark_factory → performance_benchmark_enhanced → performance_benchmark_factory
   💡 Use dependency inversion or lazy imports to break the cycle
```

## Exit Codes

- `0`: Success - No violations found or non-strict mode with violations
- `1`: Failure - Critical violations found or strict mode with any violations

## Integration with Existing Tools

This validator complements existing architectural tools:
- Uses components from `src/prompt_improver/core/boundaries.py` 
- Integrates with performance monitoring from `src/prompt_improver/monitoring/regression/`
- Compatible with testcontainer-based real behavior testing

## Performance

- **Analysis Time**: ~1 second for 639 Python files
- **Memory Usage**: Minimal - AST parsing only
- **Accuracy**: High precision with specific file:line locations

## Configuration

The validator is configured with architectural rules for the prompt-improver project:

### Layer Hierarchy
- **Presentation**: api, cli, tui, mcp_server
- **Application**: application services  
- **Domain**: core/domain, rule_engine, ml
- **Infrastructure**: database, services, monitoring
- **Repository**: repositories
- **Shared**: shared, core/common

### Heavy Dependencies
- torch, transformers, pandas, numpy, sqlalchemy, asyncpg, redis, beartype

### Critical Import Penalties
- torch: 1007ms, transformers: 892ms, pandas: 245ms, etc.

This ensures the validator enforces the specific architectural requirements and performance constraints of the prompt-improver project.