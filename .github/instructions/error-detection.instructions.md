---
applyTo: "**/*.py"
---

## Error Detection & Code Quality Guidelines

### Primary Error Detection Strategy
Always use VS Code native tools for error detection to ensure consistency with the developer's IDE experience.

### Error Detection Workflow
```python
# 1. Primary error checking - matches VS Code IDE exactly
get_errors(filePaths=["/path/to/file.py"])

# 2. Use configured tasks for comprehensive analysis
run_task(id="shell: Type Check (pyright)", workspaceFolder=workspace_path)
run_task(id="shell: Lint Code", workspaceFolder=workspace_path)
run_task(id="shell: Run Tests with Coverage", workspaceFolder=workspace_path)
```

### Error Analysis Patterns
- **Type Errors**: Address type annotation mismatches and missing type hints
- **Unused Imports**: Remove or comment unused import statements
- **Unused Variables**: Remove or prefix with underscore if intentionally unused
- **Syntax Errors**: Fix parsing issues and malformed code
- **Import Errors**: Resolve missing dependencies and circular imports

### VS Code Integration
- **Respect Performance Settings**: Work with existing optimized VS Code settings
- **Use Native Tools**: Leverage VS Code's error detection over external tools
- **Task Integration**: Use configured tasks for comprehensive checking
- **Real-Time Feedback**: Get immediate feedback matching IDE experience

### Error Resolution Process
1. **Identify**: Use `get_errors` to find all issues in target files
2. **Analyze**: Understand the root cause of each error
3. **Fix**: Apply appropriate fixes maintaining code quality
4. **Validate**: Re-run error detection to confirm resolution
5. **Test**: Ensure fixes don't break existing functionality

### Quality Standards
- **Type Safety**: All functions and methods must have proper type annotations
- **Import Hygiene**: Only import what is actually used in the code
- **Variable Usage**: All declared variables should be used or explicitly marked as unused
- **Code Standards**: Follow project's ruff configuration for formatting and linting

### Error Prevention
- **Regular Checking**: Run error detection during development
- **Pre-commit Validation**: Ensure no errors before committing code
- **Incremental Fixes**: Address errors as they appear rather than accumulating
- **Documentation**: Keep error detection patterns documented and updated
