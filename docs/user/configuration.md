# Configuration Guide

The Universal Prompt Testing Framework provides extensive configuration options to customize its behavior for different projects and requirements.

## Configuration Files

The framework supports both JSON and YAML configuration formats:

- `prompt-test-config.json` - JSON format configuration
- `prompt-test-config.yaml` - YAML format configuration

## Complete Configuration Reference

### Default Configuration

```yaml
# Complete configuration with all available options
analysis:
  includeTests: true              # Include test files in analysis
  includeDocs: true               # Include documentation files
  maxFileSize: "10MB"             # Maximum file size to analyze
  excludePatterns:                # Patterns to exclude from analysis
    - "node_modules/"
    - ".git/"
    - "dist/"
    - "build/"
    - "coverage/"
  contextInference:
    domainKeywords: true          # Use domain-specific keywords for context
    dependencyAnalysis: true      # Analyze package dependencies
    structureAnalysis: true       # Analyze project structure

testing:
  defaultTestCount: 100           # Default number of tests to generate
  complexityDistribution:         # Distribution of test complexity
    simple: 0.4                   # 40% simple tests
    moderate: 0.4                 # 40% moderate tests
    complex: 0.2                  # 20% complex tests
  categoryWeights:                # Weight of different test categories
    vague_instructions: 0.25      # 25% vague instruction tests
    missing_context: 0.25         # 25% missing context tests
    poor_structure: 0.20          # 20% poor structure tests
    missing_examples: 0.15        # 15% missing examples tests
    no_output_format: 0.15        # 15% no output format tests
  contextRelevanceWeight: 0.8     # How much to weight context relevance (0-1)

evaluation:
  methods:                        # Evaluation methods to use
    - "structural"
    - "llm-judge"
  llmJudge:
    model: "claude-3.5-sonnet"    # Model for LLM-as-a-judge evaluation
    temperature: 0.1              # Temperature for model responses
    maxTokens: 1000               # Maximum tokens per evaluation
    batchSize: 10                 # Number of evaluations per batch
    rateLimitDelay: 1000          # Delay between API calls (ms)
  structuralWeights:              # Weights for structural analysis
    clarity: 0.3                  # 30% clarity weight
    completeness: 0.25            # 25% completeness weight
    specificity: 0.25             # 25% specificity weight
    structure: 0.2                # 20% structure weight

statistics:
  significanceLevel: 0.05         # Statistical significance threshold
  confidenceLevel: 0.95           # Confidence level for intervals
  minimumSampleSize: 30           # Minimum sample size for statistics
  effectSizeThreshold: 0.2        # Minimum effect size to consider significant

optimization:
  autoUpdateRules: false          # Automatically apply rule updates
  minConfidenceForUpdate: 0.8     # Minimum confidence to apply updates
  maxRiskLevel: "medium"          # Maximum risk level for updates (low/medium/high)
  backupBeforeUpdate: true        # Create backup before applying updates
  maxUpdatesPerCycle: 5           # Maximum rule updates per optimization cycle
  validationTestCount: 50         # Number of tests for validation

output:
  defaultFormat: "markdown"       # Default output format (json/markdown/html/csv)
  includeExamples: true           # Include examples in reports
  includeStatistics: true         # Include detailed statistics
  includeRecommendations: true    # Include optimization recommendations
  maxExampleLength: 200           # Maximum length of example text
  reportSections:                 # Sections to include in reports
    - "executiveSummary"
    - "projectAnalysis"
    - "testResults"
    - "ruleEffectiveness"
    - "optimizationRecommendations"

paths:
  rulesFile: "./config/prompt-engineering-rules.json"  # Path to rules file
  backupDir: "./backups"          # Directory for backups
  outputDir: "./output"           # Directory for output files
  logsDir: "./logs"               # Directory for log files
```

## Configuration Examples

### Minimal Configuration

For basic usage with sensible defaults:

```yaml
# minimal-config.yaml
testing:
  defaultTestCount: 50

evaluation:
  methods: ["structural"]

output:
  defaultFormat: "markdown"
```

### High-Throughput Configuration

For processing large numbers of tests quickly:

```yaml
# high-throughput-config.yaml
testing:
  defaultTestCount: 500
  complexityDistribution:
    simple: 0.8      # Focus on simpler tests for speed
    moderate: 0.2
    complex: 0.0

evaluation:
  methods: ["structural"]  # Use only fast structural analysis

output:
  includeExamples: false   # Reduce output size
  includeStatistics: false
```

### Comprehensive Analysis Configuration

For thorough analysis with all evaluation methods:

```yaml
# comprehensive-config.yaml
testing:
  defaultTestCount: 200

evaluation:
  methods: ["structural", "llm-judge", "output-quality"]
  llmJudge:
    model: "claude-3.5-sonnet"
    batchSize: 5     # Smaller batches for reliability
    
optimization:
  autoUpdateRules: true
  minConfidenceForUpdate: 0.9  # Higher confidence threshold

output:
  includeExamples: true
  includeStatistics: true
  includeRecommendations: true
```

### CI/CD Configuration

Optimized for automated testing in pipelines:

```yaml
# ci-config.yaml
testing:
  defaultTestCount: 25     # Smaller number for faster execution

evaluation:
  methods: ["structural"]  # Fast evaluation only

optimization:
  autoUpdateRules: false   # No automatic updates in CI

output:
  defaultFormat: "json"    # Machine-readable format
  includeExamples: false   # Minimal output
```

### Development Configuration

For active development and experimentation:

```yaml
# dev-config.yaml
testing:
  defaultTestCount: 10     # Small number for quick iterations

evaluation:
  methods: ["structural", "llm-judge"]
  llmJudge:
    temperature: 0.3       # Higher temperature for variety

optimization:
  autoUpdateRules: false   # Manual control during development
  backupBeforeUpdate: true

output:
  includeExamples: true    # Helpful for understanding results
  maxExampleLength: 500    # Longer examples for context
```

## Configuration Validation

The framework automatically validates configuration values:

### Testing Validation
- `defaultTestCount`: Must be between 10 and 1000
- `complexityDistribution`: Values must sum to 1.0
- `contextRelevanceWeight`: Must be between 0 and 1

### Statistics Validation
- `significanceLevel`: Must be between 0 and 1
- `confidenceLevel`: Must be between 0 and 1
- `minimumSampleSize`: Must be at least 10

### Optimization Validation
- `minConfidenceForUpdate`: Must be between 0 and 1
- `maxRiskLevel`: Must be "low", "medium", or "high"
- `maxUpdatesPerCycle`: Must be positive integer

### Output Validation
- `defaultFormat`: Must be "json", "markdown", "html", or "csv"
- `maxExampleLength`: Must be positive integer

## Environment Variables

You can override configuration values using environment variables:

```bash
# Set default test count
export PROMPT_TEST_COUNT=150

# Set evaluation methods
export PROMPT_EVAL_METHODS="structural,llm-judge"

# Set output format
export PROMPT_OUTPUT_FORMAT="json"

# Set LLM model
export PROMPT_LLM_MODEL="gpt-4-turbo"
```

## Model-Specific Configurations

### Claude Configuration

```yaml
evaluation:
  llmJudge:
    model: "claude-3.5-sonnet"
    temperature: 0.1
    maxTokens: 2000
    rateLimitDelay: 1000
```

### GPT-4 Configuration

```yaml
evaluation:
  llmJudge:
    model: "gpt-4-turbo"
    temperature: 0.2
    maxTokens: 1500
    rateLimitDelay: 500
```

### Gemini Configuration

```yaml
evaluation:
  llmJudge:
    model: "gemini-pro"
    temperature: 0.15
    maxTokens: 1000
    rateLimitDelay: 800
```

## Project-Specific Configurations

### React Projects

```yaml
analysis:
  contextInference:
    domainKeywords: true
  
testing:
  categoryWeights:
    poor_structure: 0.35    # React components need good structure
    missing_context: 0.25   # Component context is crucial
    vague_instructions: 0.20
    missing_examples: 0.15
    no_output_format: 0.05
```

### API Projects

```yaml
testing:
  categoryWeights:
    missing_context: 0.35   # API context is crucial
    no_output_format: 0.30  # Output format very important
    vague_instructions: 0.20
    poor_structure: 0.10
    missing_examples: 0.05
```

### Data Science Projects

```yaml
testing:
  categoryWeights:
    missing_examples: 0.35  # Examples crucial for data science
    missing_context: 0.25   # Context about data is important
    vague_instructions: 0.20
    poor_structure: 0.15
    no_output_format: 0.05
```

## Configuration Loading Priority

The framework loads configuration in the following order (later values override earlier ones):

1. Default configuration (built-in)
2. Global configuration file (`~/.prompt-test-config.yaml`)
3. Project configuration file (`./prompt-test-config.yaml`)
4. Command-line specified configuration file (`--config path/to/config.yaml`)
5. Environment variables
6. Command-line options

## Configuration Management

### Creating Configuration Files

```bash
# Generate example configuration
prompt-test-framework config --generate --format yaml > prompt-test-config.yaml

# Validate existing configuration
prompt-test-framework config --validate

# Show current effective configuration
prompt-test-framework config --show
```

### Configuration Schema

The framework uses JSON Schema for configuration validation. You can get the complete schema:

```bash
# Get configuration schema
prompt-test-framework config --schema > config-schema.json
```

## Troubleshooting Configuration

### Common Configuration Errors

1. **Invalid complexity distribution**
   ```
   Error: testing.complexityDistribution values must sum to 1.0
   ```
   Solution: Ensure simple + moderate + complex = 1.0

2. **Invalid significance level**
   ```
   Error: statistics.significanceLevel must be between 0 and 1
   ```
   Solution: Use value like 0.05 (5%)

3. **Unknown evaluation method**
   ```
   Error: Invalid evaluation method: unknown-method
   ```
   Solution: Use "structural", "llm-judge", or "output-quality"

### Configuration Debugging

```bash
# Debug configuration loading
prompt-test-framework run-tests --verbose --config-debug

# Show configuration merge process
export DEBUG=prompt-test:config
prompt-test-framework analyze
```

## Best Practices

1. **Start Simple**: Begin with minimal configuration and add complexity as needed
2. **Environment-Specific**: Use different configurations for development, testing, and production
3. **Version Control**: Keep configuration files in version control
4. **Documentation**: Document custom configuration choices for your team
5. **Validation**: Always validate configuration before important runs
6. **Backup**: Keep backups of working configurations

## Advanced Configuration

### Custom Rule Files

```yaml
paths:
  rulesFile: "./my-custom-rules.json"
```

### Multiple Output Formats

```bash
# Generate multiple formats
prompt-test-framework report --format json --output results.json
prompt-test-framework report --format html --output results.html
prompt-test-framework report --format csv --output results.csv
```

### Conditional Configuration

Use environment-based configuration:

```yaml
# Use different settings based on environment
testing:
  defaultTestCount: !env_var PROMPT_TEST_COUNT 100

evaluation:
  methods: !env_var PROMPT_EVAL_METHODS ["structural"]
```

This comprehensive configuration system allows you to tailor the framework's behavior precisely to your project's needs and constraints.