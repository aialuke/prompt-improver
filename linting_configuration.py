#!/usr/bin/env python3
"""
Enhanced Linting Configuration for Naming Convention Enforcement
Generates configuration files for various linting tools to enforce PEP 8 naming conventions.
"""

import json
from pathlib import Path


class LintingConfigGenerator:
    """Generates linting configuration files for naming convention enforcement."""
    
    def __init__(self):
        self.config_dir = Path("linting_configs")
        self.config_dir.mkdir(exist_ok=True)
    
    def generate_flake8_config(self) -> None:
        """Generate flake8 configuration with naming convention rules."""
        config = """[flake8]
# Enhanced naming convention enforcement
max-line-length = 88
extend-ignore = 
    E203,  # whitespace before ':'
    W503,  # line break before binary operator
    E501,  # line too long (handled by black)

# Naming convention checks
select = 
    E,     # pycodestyle errors
    W,     # pycodestyle warnings  
    F,     # pyflakes
    N,     # pep8-naming
    C90,   # mccabe complexity

# pep8-naming configuration
ignore-names = 
    setUp,
    tearDown,
    runTest,
    maxDiff,
    longMessage,
    DataFrame,
    Series,
    API,
    URL,
    HTTP,
    JSON,
    XML,
    CSV,
    SQL,
    UUID,
    JWT,
    T,
    K,
    V,
    X,
    y

# Exclude patterns
exclude = 
    .git,
    __pycache__,
    .venv,
    venv,
    .eggs,
    *.egg,
    build,
    dist

# Per-file ignores for legacy code during transition
per-file-ignores = 
    # Allow legacy naming during transition period
    src/prompt_improver/ml/types.py:N806,N815
    src/prompt_improver/ml/optimization/algorithms/dimensionality_reducer.py:N806,N815
"""
        
        with open(self.config_dir / "setup.cfg", "w") as f:
            f.write(config)
    
    def generate_pylint_config(self) -> None:
        """Generate pylint configuration with naming conventions."""
        config = """[MASTER]
# Use multiple processes to speed up Pylint
jobs=0

# Pickle collected data for later comparisons
persistent=yes

[MESSAGES CONTROL]
# Enable naming convention checks
enable=
    invalid-name,
    bad-classmethod-argument,
    bad-mcs-classmethod-argument,
    bad-mcs-method-argument

# Disable some checks during transition
disable=
    too-few-public-methods,
    too-many-arguments,
    too-many-locals,
    too-many-branches,
    too-many-statements

[BASIC]
# Naming style enforcement

# Variable names should be snake_case
variable-rgx=[a-z_][a-z0-9_]{2,30}$
variable-name-hint=snake_case

# Function names should be snake_case  
function-rgx=[a-z_][a-z0-9_]{2,30}$
function-name-hint=snake_case

# Method names should be snake_case
method-rgx=[a-z_][a-z0-9_]{2,30}$
method-name-hint=snake_case

# Class names should be PascalCase
class-rgx=[A-Z_][a-zA-Z0-9]+$
class-name-hint=PascalCase

# Module names should be snake_case
module-rgx=(([a-z_][a-z0-9_]*)|([A-Z][a-zA-Z0-9]+))$
module-name-hint=snake_case

# Constant names should be UPPER_CASE
const-rgx=(([A-Z_][A-Z0-9_]*)|(__.*__))$
const-name-hint=UPPER_CASE

# Attribute names should be snake_case
attr-rgx=[a-z_][a-z0-9_]{2,30}$
attr-name-hint=snake_case

# Argument names should be snake_case
argument-rgx=[a-z_][a-z0-9_]{2,30}$
argument-name-hint=snake_case

# Class attribute names should be snake_case
class-attribute-rgx=([A-Za-z_][A-Za-z0-9_]{2,30}|(__.*__))$

# Good variable names which should always be accepted
good-names=i,j,k,ex,Run,_,x,y,z,X,Y,Z,T,K,V,id

# Bad variable names which should always be refused
bad-names=foo,bar,baz,toto,tutu,tata

# Include a hint for the correct naming format with invalid-name
include-naming-hint=yes

[FORMAT]
# Maximum number of characters on a single line
max-line-length=88

# String used as indentation unit
indent-string='    '

[DESIGN]
# Maximum number of arguments for function / method
max-args=10

# Maximum number of locals for function / method body
max-locals=25

# Maximum number of return / yield for function / method body
max-returns=6

# Maximum number of branch for function / method body
max-branches=20

# Maximum number of statements in function / method body
max-statements=50

# Maximum number of parents for a class
max-parents=7

# Maximum number of attributes for a class
max-attributes=15

# Minimum number of public methods for a class
min-public-methods=1

# Maximum number of public methods for a class
max-public-methods=25
"""
        
        with open(self.config_dir / "pylintrc", "w") as f:
            f.write(config)
    
    def generate_mypy_config(self) -> None:
        """Generate mypy configuration for type checking."""
        config = """[mypy]
# Global options
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

# Per-module options for gradual typing adoption
[mypy-prompt_improver.ml.types]
# Allow flexible typing in types module during transition
disallow_untyped_defs = False
disallow_incomplete_defs = False

[mypy-prompt_improver.*.tests.*]
# Relax some rules for test files
disallow_untyped_defs = False
disallow_incomplete_defs = False

# Third-party library stubs
[mypy-numpy.*]
ignore_missing_imports = True

[mypy-sklearn.*]
ignore_missing_imports = True

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-umap.*]
ignore_missing_imports = True
"""
        
        with open(self.config_dir / "mypy.ini", "w") as f:
            f.write(config)
    
    def generate_pre_commit_config(self) -> None:
        """Generate pre-commit hooks configuration."""
        config = """# Pre-commit hooks for naming convention enforcement
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements
      - id: check-docstring-first

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [pep8-naming]
        args: ["--config=linting_configs/setup.cfg"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        args: ["--config-file=linting_configs/mypy.ini"]
        additional_dependencies: [types-all]

  # Custom naming convention checker
  - repo: local
    hooks:
      - id: naming-convention-check
        name: Naming Convention Check
        entry: python naming_convention_analyzer.py
        language: system
        files: \\.py$
        pass_filenames: false
"""
        
        with open(self.config_dir / ".pre-commit-config.yaml", "w") as f:
            f.write(config)
    
    def generate_github_actions_workflow(self) -> None:
        """Generate GitHub Actions workflow for CI/CD enforcement."""
        workflow = """name: Code Quality and Naming Convention Check

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 mypy black isort pep8-naming
        pip install -r requirements.txt
    
    - name: Run Black formatting check
      run: black --check --diff src/
    
    - name: Run isort import sorting check
      run: isort --check-only --diff src/
    
    - name: Run flake8 linting
      run: flake8 src/ --config=linting_configs/setup.cfg
    
    - name: Run mypy type checking
      run: mypy src/ --config-file=linting_configs/mypy.ini
    
    - name: Run naming convention analysis
      run: python naming_convention_analyzer.py
    
    - name: Check naming convention compliance
      run: |
        python -c "
        import json
        with open('naming_convention_report.json', 'r') as f:
            data = json.load(f)
        violations = data['summary']['total_violations']
        if violations > 50:  # Threshold for acceptable violations
            print(f'Too many naming violations: {violations}')
            exit(1)
        print(f'Naming violations within acceptable range: {violations}')
        "
    
    - name: Upload naming convention report
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: naming-convention-report
        path: naming_convention_report.json
"""
        
        workflows_dir = self.config_dir / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        with open(workflows_dir / "code-quality.yml", "w") as f:
            f.write(workflow)
    
    def generate_all_configs(self) -> None:
        """Generate all linting configuration files."""
        print("Generating linting configuration files...")
        
        self.generate_flake8_config()
        print("✓ Generated flake8 configuration (setup.cfg)")
        
        self.generate_pylint_config()
        print("✓ Generated pylint configuration (pylintrc)")
        
        self.generate_mypy_config()
        print("✓ Generated mypy configuration (mypy.ini)")
        
        self.generate_pre_commit_config()
        print("✓ Generated pre-commit hooks (.pre-commit-config.yaml)")
        
        self.generate_github_actions_workflow()
        print("✓ Generated GitHub Actions workflow (code-quality.yml)")
        
        # Generate installation script
        install_script = """#!/bin/bash
# Installation script for naming convention enforcement

echo "Installing naming convention enforcement tools..."

# Install pre-commit
pip install pre-commit

# Install linting tools
pip install flake8 pep8-naming pylint mypy black isort

# Install pre-commit hooks
pre-commit install

# Copy configuration files to project root
cp linting_configs/setup.cfg .
cp linting_configs/pylintrc .
cp linting_configs/mypy.ini .
cp linting_configs/.pre-commit-config.yaml .

# Create .github/workflows directory if it doesn't exist
mkdir -p .github/workflows
cp linting_configs/.github/workflows/code-quality.yml .github/workflows/

echo "✓ Naming convention enforcement installed successfully!"
echo ""
echo "Usage:"
echo "  pre-commit run --all-files    # Run all checks"
echo "  flake8 src/                   # Run flake8 linting"
echo "  mypy src/                     # Run type checking"
echo "  python naming_convention_analyzer.py  # Run naming analysis"
"""
        
        with open(self.config_dir / "install_enforcement.sh", "w") as f:
            f.write(install_script)
        
        # Make script executable
        import stat
        script_path = self.config_dir / "install_enforcement.sh"
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
        
        print("✓ Generated installation script (install_enforcement.sh)")
        print(f"\nAll configuration files generated in: {self.config_dir}")


if __name__ == "__main__":
    generator = LintingConfigGenerator()
    generator.generate_all_configs()
    
    print("\n=== Linting Configuration Summary ===")
    print("Generated files:")
    print("  - setup.cfg (flake8 configuration)")
    print("  - pylintrc (pylint configuration)")  
    print("  - mypy.ini (mypy type checking)")
    print("  - .pre-commit-config.yaml (pre-commit hooks)")
    print("  - .github/workflows/code-quality.yml (CI/CD)")
    print("  - install_enforcement.sh (installation script)")
    print()
    print("To install enforcement:")
    print("  cd linting_configs && ./install_enforcement.sh")
    print()
    print("To run checks:")
    print("  pre-commit run --all-files")
    print("  python naming_convention_analyzer.py")
