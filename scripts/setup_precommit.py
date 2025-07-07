#!/usr/bin/env python3
"""Pre-commit Setup Script for All Contributors"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd: list, cwd: str = None) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        print(f"   Error: {e.stderr}")
        return False


def check_prerequisites():
    """Check if required tools are available."""
    print("🔍 Checking prerequisites...")
    
    # Check Python
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ required")
        return False
    
    # Check git
    if not shutil.which('git'):
        print("❌ Git not found")
        return False
    
    # Check if in git repository
    if not Path('.git').exists():
        print("❌ Not in a git repository")
        return False
    
    print("✅ Prerequisites met")
    return True


def install_precommit():
    """Install pre-commit if not already installed."""
    print("📦 Installing pre-commit...")
    
    # Check if pre-commit is already installed
    if shutil.which('pre-commit'):
        print("✅ Pre-commit already installed")
        return True
    
    # Try to install via pip
    success = run_command([sys.executable, '-m', 'pip', 'install', 'pre-commit>=3.6.0'])
    if success:
        print("✅ Pre-commit installed")
        return True
    
    print("❌ Failed to install pre-commit")
    return False


def install_hooks():
    """Install pre-commit hooks."""
    print("🪝 Installing pre-commit hooks...")
    
    success = run_command(['pre-commit', 'install'])
    if not success:
        return False
    
    # Install commit-msg hook for commitizen
    success = run_command(['pre-commit', 'install', '--hook-type', 'commit-msg'])
    if not success:
        print("⚠️  Failed to install commit-msg hook (optional)")
    
    # Install pre-push hook
    success = run_command(['pre-commit', 'install', '--hook-type', 'pre-push'])
    if not success:
        print("⚠️  Failed to install pre-push hook (optional)")
    
    print("✅ Pre-commit hooks installed")
    return True


def setup_git_config():
    """Setup git configuration for better commit workflow."""
    print("⚙️  Configuring git settings...")
    
    # Set up git hooks path (if needed)
    hooks_path = Path('.git/hooks')
    if hooks_path.exists():
        print("✅ Git hooks directory exists")
    
    # Configure commit template (optional)
    template_path = Path('.gitmessage')
    if not template_path.exists():
        with open(template_path, 'w') as f:
            f.write("""# feat: add new feature
# fix: fix a bug
# docs: update documentation
# style: format code (no functional changes)
# refactor: refactor code
# test: add or update tests
# chore: update build process or auxiliary tools

# What does this commit do?
# 

# Why is this change needed?
# 

# How does it address the issue?
# 

# Are there any side effects?
# 
""")
        
        # Set git commit template
        run_command(['git', 'config', 'commit.template', '.gitmessage'])
        print("✅ Git commit template configured")
    
    return True


def validate_setup():
    """Validate that pre-commit is working correctly."""
    print("🧪 Validating pre-commit setup...")
    
    # Run pre-commit on all files (dry run)
    print("   Running pre-commit validation...")
    success = run_command(['pre-commit', 'run', '--all-files', '--show-diff-on-failure'])
    
    if success:
        print("✅ Pre-commit validation passed")
        return True
    else:
        print("⚠️  Pre-commit validation found issues")
        print("   This is normal for first-time setup")
        print("   Issues will be auto-fixed in future commits")
        return True  # Don't fail setup for validation issues


def create_contributor_guide():
    """Create a guide for contributors."""
    print("📖 Creating contributor guide...")
    
    guide_content = """# Pre-commit Hooks Guide

## What are pre-commit hooks?

Pre-commit hooks automatically run code quality checks before each commit, ensuring:
- Code formatting consistency (Ruff)
- No linting errors (Ruff with --preview)
- Type checking (MyPy)
- Security scanning (Bandit)
- ML contract validation
- Performance regression detection

## How to use

### First-time setup (already done for you):
```bash
python scripts/setup_precommit.py
```

### Daily workflow:
1. Make your changes
2. Stage files: `git add .`
3. Commit: `git commit -m "your message"`
   - Hooks run automatically
   - Issues are auto-fixed when possible
   - Commit fails if critical issues found
4. Push: `git push`

### Common scenarios:

#### Hook finds and fixes issues:
```bash
$ git commit -m "fix: update algorithm"
ruff format.............................(no files to check)Skipped
ruff check..............................Failed
- hook id: ruff
- files were modified by this hook

# Files were auto-formatted, re-stage and commit:
$ git add .
$ git commit -m "fix: update algorithm"
```

#### Hook finds critical issues:
```bash
$ git commit -m "feat: add feature"
ML Contract Validation..................Failed
- hook id: ml-contract-validation
❌ src/new_feature.py: ML contract validation failed
   ERROR: Performance function 'evaluate' should include timing/metrics

# Fix the issues and try again
```

#### Bypass hooks (not recommended):
```bash
# Skip all hooks (emergency only)
$ git commit --no-verify -m "emergency fix"

# Skip specific performance check
$ SKIP_PERFORMANCE_CHECK=1 git commit -m "docs: update readme"
```

## Troubleshooting

### "pre-commit command not found"
```bash
pip install pre-commit>=3.6.0
pre-commit install
```

### "Hook failed to run"
```bash
# Update hooks
pre-commit autoupdate

# Clear cache and reinstall
pre-commit clean
pre-commit install --install-hooks
```

### "Performance check fails"
```bash
# Check what changed
python scripts/check_performance_regression.py

# Skip if necessary (not recommended)
SKIP_PERFORMANCE_CHECK=1 git commit -m "your message"
```

## Available hooks

1. **Ruff Check**: Code linting with preview features
2. **Ruff Format**: Code formatting
3. **MyPy**: Type checking
4. **Bandit**: Security scanning
5. **Basic Checks**: Trailing whitespace, YAML/JSON validation
6. **MCP Protocol Validation**: Custom MCP contract checks
7. **ML Contract Validation**: ML component contract verification
8. **Performance Regression**: Pre-push performance validation

## Getting help

- Check hook output for specific guidance
- Review this guide: `PRECOMMIT_GUIDE.md`
- Run setup script again: `python scripts/setup_precommit.py`
- Ask team members for assistance
"""
    
    with open('PRECOMMIT_GUIDE.md', 'w') as f:
        f.write(guide_content)
    
    print("✅ Contributor guide created: PRECOMMIT_GUIDE.md")
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up pre-commit hooks for contributors...")
    print("=" * 50)
    
    if not check_prerequisites():
        print("\n❌ Setup failed: Prerequisites not met")
        sys.exit(1)
    
    if not install_precommit():
        print("\n❌ Setup failed: Could not install pre-commit")
        sys.exit(1)
    
    if not install_hooks():
        print("\n❌ Setup failed: Could not install hooks")
        sys.exit(1)
    
    if not setup_git_config():
        print("\n⚠️  Git configuration had issues (continuing)")
    
    if not validate_setup():
        print("\n⚠️  Validation had issues (continuing)")
    
    if not create_contributor_guide():
        print("\n⚠️  Could not create guide (continuing)")
    
    print("\n" + "=" * 50)
    print("✅ Pre-commit setup complete!")
    print("\n📖 Next steps:")
    print("   1. Read the contributor guide: PRECOMMIT_GUIDE.md")
    print("   2. Make a test commit to verify hooks are working")
    print("   3. Share this setup with other contributors")
    print("\n💡 To run hooks manually: pre-commit run --all-files")


if __name__ == '__main__':
    main()
