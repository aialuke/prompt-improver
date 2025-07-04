# Getting Started Guide

Welcome to the Universal Prompt Testing Framework! This guide will help you get up and running in just a few minutes.

## Prerequisites

- Node.js 16 or higher
- npm or yarn package manager
- Git (for cloning the repository)

## Installation

### Option 1: Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd prompting

# Install dependencies
npm install

# Make CLI globally available (optional)
npm link
```

### Option 2: Direct Usage

If you don't want to install globally, you can run commands directly:

```bash
# Use npx to run commands
npx prompt-test-framework analyze

# Or use node directly
node prompt-test-framework.js analyze
```

## Your First Test Run

Let's run your first prompt improvement test in 3 simple steps:

### Step 1: Quick Analysis

First, let's see what the framework detects about your project:

```bash
prompt-test-framework analyze --verbose
```

This will:
- Scan your project directory
- Detect programming languages and frameworks
- Identify project type and domain
- Generate a context profile

Expected output:
```
🔍 Analyzing project...
✅ Context Analysis Complete:
  - Languages: JavaScript, TypeScript
  - Frameworks: React, Next.js
  - Domain: Web Application
  - Confidence: 94%
```

### Step 2: Generate and Run Tests

Now let's generate some tests and see how the framework improves prompts:

```bash
prompt-test-framework run-tests --test-count 25 --verbose
```

This will:
- Generate 25 test prompts relevant to your project
- Apply improvement techniques
- Evaluate the results
- Show you the outcomes

Expected output:
```
🧪 Generating test prompts...
✅ Generated 25 test prompts across 5 categories

🚀 Running evaluation...
✅ Structural analysis complete (23/25 passed)

📊 Results Summary:
  Tests executed: 25
  Average improvement: 78.4%
  Success rate: 92.0%
```

### Step 3: View Results

Generate a comprehensive report of your results:

```bash
prompt-test-framework report --format markdown --include-examples
```

This creates a detailed report showing:
- Executive summary of improvements
- Performance breakdown by category
- Top performing rules
- Specific examples of improvements
- Recommendations for optimization

## Understanding Your Results

### What the Numbers Mean

- **Average Improvement**: How much better the improved prompts are (0-100%)
- **Success Rate**: Percentage of tests that showed meaningful improvement
- **Category Breakdown**: Performance across different prompt issue types

### Key Metrics to Watch

1. **Overall Improvement > 70%**: Your prompts are getting significantly better
2. **Success Rate > 85%**: The framework is working well for your project type
3. **Structural Score > 80%**: Your improved prompts have good structure

### Common First Results

**Great Results (85%+ improvement)**
- Your project has clear patterns the framework can optimize
- Consider running more comprehensive tests
- Look into automatic rule optimization

**Good Results (65-85% improvement)**
- Solid baseline with room for optimization
- Review category breakdowns for specific areas to improve
- Consider customizing configuration for your domain

**Mixed Results (45-65% improvement)**
- May need configuration tuning
- Check if your project type is well-supported
- Consider providing more context in configuration

## Common First-Time Issues

### Issue: "No improvements detected"

**Cause**: Your prompts might already be well-structured
**Solution**: 
```bash
# Try with more complex test cases
prompt-test-framework run-tests --complexity complex --test-count 50
```

### Issue: "Context confidence is low"

**Cause**: Project structure might be unclear
**Solution**:
```bash
# Provide more explicit context
prompt-test-framework analyze --project-dir ./src --verbose
```

### Issue: "LLM evaluation failing"

**Cause**: API keys or model access issues
**Solution**:
```bash
# Use only structural evaluation for now
prompt-test-framework run-tests --evaluation-method structural
```

## Next Steps

### 1. Customize Configuration

Create a configuration file for your project:

```bash
# Generate example configuration
echo 'testing:
  defaultTestCount: 100
  complexityDistribution:
    simple: 0.3
    moderate: 0.5
    complex: 0.2

evaluation:
  methods: ["structural", "llm-judge"]

output:
  defaultFormat: "markdown"
  includeExamples: true' > prompt-test-config.yaml
```

### 2. Run Comprehensive Tests

Now run a more thorough analysis:

```bash
prompt-test-framework run-tests \
  --test-count 100 \
  --evaluation-method all \
  --output-dir ./detailed-results \
  --verbose
```

### 3. Set Up Optimization

Enable automatic rule optimization:

```bash
prompt-test-framework optimize \
  --auto-update-rules \
  --backup-rules \
  --min-confidence 0.8
```

### 4. Generate Multiple Report Formats

Create reports for different audiences:

```bash
# Technical report with statistics
prompt-test-framework report \
  --format json \
  --include-statistics \
  --output technical-report.json

# Executive summary
prompt-test-framework report \
  --format markdown \
  --include-recommendations \
  --output executive-summary.md

# Data export
prompt-test-framework report \
  --format csv \
  --output data-export.csv
```

## Project-Specific Quick Starts

### React Projects

```bash
# Optimal settings for React projects
prompt-test-framework run-tests \
  --test-count 50 \
  --categories "poor_structure,missing_context" \
  --evaluation-method structural
```

### API Projects

```bash
# Focus on API-relevant categories
prompt-test-framework run-tests \
  --test-count 75 \
  --categories "missing_context,no_output_format" \
  --evaluation-method "structural,output-quality"
```

### Data Science Projects

```bash
# Emphasize examples and context for data science
prompt-test-framework run-tests \
  --test-count 60 \
  --categories "missing_examples,missing_context" \
  --evaluation-method all
```

## Integration Examples

### VS Code Integration

Add to your VS Code tasks (`.vscode/tasks.json`):

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Analyze Prompts",
      "type": "shell",
      "command": "prompt-test-framework",
      "args": ["analyze", "--verbose"],
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    }
  ]
}
```

### npm Scripts Integration

Add to your `package.json`:

```json
{
  "scripts": {
    "prompt:test": "prompt-test-framework run-tests --test-count 50",
    "prompt:analyze": "prompt-test-framework analyze --verbose",
    "prompt:report": "prompt-test-framework report --format markdown",
    "prompt:optimize": "prompt-test-framework optimize --backup-rules"
  }
}
```

Then run:
```bash
npm run prompt:test
npm run prompt:analyze
npm run prompt:report
```

### Git Hooks Integration

Add to `.husky/pre-commit` or `.git/hooks/pre-commit`:

```bash
#!/bin/sh
# Run prompt analysis on commit
prompt-test-framework analyze --quiet
```

## Troubleshooting

### Installation Issues

**Error: "command not found"**
```bash
# Make sure you're in the right directory
cd /path/to/prompting

# Try running directly
node prompt-test-framework.js --help

# Or install globally
npm link
```

**Error: "missing dependencies"**
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Runtime Issues

**Error: "Cannot find module"**
```bash
# Check that you're running from the correct directory
ls -la src/

# Verify all files are present
find src/ -name "*.js" | wc -l
```

**Error: "Configuration validation failed"**
```bash
# Check your configuration syntax
prompt-test-framework config --validate

# Use minimal configuration
prompt-test-framework run-tests --evaluation-method structural
```

### Performance Issues

**Tests running slowly**
```bash
# Reduce test count
prompt-test-framework run-tests --test-count 10

# Use only fast evaluation
prompt-test-framework run-tests --evaluation-method structural
```

**Out of memory errors**
```bash
# Increase Node.js memory limit
node --max-old-space-size=4096 prompt-test-framework.js run-tests
```

## Getting Help

### Built-in Help

```bash
# General help
prompt-test-framework --help

# Command-specific help
prompt-test-framework run-tests --help

# Show current configuration
prompt-test-framework config --show
```

### Debug Mode

```bash
# Enable debug output
export DEBUG=prompt-test:*
prompt-test-framework run-tests --verbose
```

### Verbose Logging

```bash
# Maximum verbosity
prompt-test-framework run-tests --verbose --debug
```

## What's Next?

Now that you have the basics working:

1. **Read the Configuration Guide** - Learn how to customize the framework for your needs
2. **Explore Examples** - Check out project-specific examples in the `/examples` directory
3. **Set Up CI/CD** - Integrate prompt testing into your development workflow
4. **Contribute** - Help improve the framework by reporting issues or contributing code

Congratulations! You've successfully run your first prompt improvement tests. The framework is now analyzing and improving prompts based on your project's specific context and requirements.