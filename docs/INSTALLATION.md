# Prompt Improvement Toolkit Installation

## Quick Setup

### 1. Install Dependencies
```bash
cd Prompting
npm install
```

### 2. Make Executable (Optional)
```bash
chmod +x improve-prompt.js
```

### 3. Global Installation (Recommended)
```bash
npm install -g .
```

After global installation, you can use from anywhere:
```bash
improve-prompt "Write a function to sort an array"
```

## Usage

### Local Usage (from Prompting directory)
```bash
# Basic usage
node improve-prompt.js "Create a React component"

# Interactive mode
node improve-prompt.js --interactive

# With model specification
node improve-prompt.js "Debug this Python code" --model gpt4

# Verbose output
node improve-prompt.js "Analyze user behavior" --verbose
```

### Global Usage (after npm install -g)
```bash
# From any directory
improve-prompt "Write API documentation"

# Interactive mode
improve-prompt --interactive

# Help
improve-prompt --help
```

## Cross-Project Usage

Since this toolkit is git-ignored, you can:

1. **Copy to new projects**: Copy the entire `Prompting/` folder
2. **Symlink**: Create symlinks to use the same installation across projects
3. **Global install**: Use `npm install -g .` for system-wide access

### Symlink Setup (Advanced)
```bash
# From your new project directory
ln -s /path/to/original/Prompting ./Prompting

# Add to new project's .gitignore
echo "Prompting/" >> .gitignore
```

## Verification

Test the installation:
```bash
improve-prompt "Write a hello world function"
```

You should see prompt analysis and improvement suggestions.

## Troubleshooting

### Permission Issues
```bash
chmod +x improve-prompt.js
```

### Missing Dependencies
```bash
npm install chalk commander
```

### Node.js Version
Requires Node.js 14+ for compatibility.