#!/bin/bash
set -e

echo "🚀 Setting up Prompt Improver development environment (2025)..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the correct directory
if [ ! -f "pyproject.toml" ] && [ ! -f "requirements.txt" ]; then
    print_error "Not in project root directory. Please run from /workspaces/prompt-improver"
    exit 1
fi

print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Python dependencies installed"
else
    print_warning "No requirements.txt found, skipping Python dependencies"
fi

# Install development dependencies if they exist
if [ -f "requirements-dev.txt" ]; then
    print_status "Installing Python development dependencies..."
    pip install -r requirements-dev.txt
    print_success "Development dependencies installed"
fi

# Create package.json if it doesn't exist
if [ ! -f "package.json" ]; then
    print_status "Creating initial package.json..."
    cat > package.json << 'EOF'
{
  "name": "prompt-improver-frontend",
  "version": "1.0.0",
  "type": "module",
  "description": "Frontend development for Prompt Improver with Vite and TypeScript",
  "scripts": {
    "dev": "vite --host 0.0.0.0 --port 5173",
    "build": "tsc && vite build",
    "preview": "vite preview --host 0.0.0.0 --port 4173",
    "lint": "eslint . --ext ts,tsx,js,jsx --report-unused-disable-directives --max-warnings 0",
    "lint:fix": "eslint . --ext ts,tsx,js,jsx --fix",
    "type-check": "tsc --noEmit",
    "format": "prettier --write \"src/**/*.{ts,tsx,js,jsx,css,md,json}\"",
    "format:check": "prettier --check \"src/**/*.{ts,tsx,js,jsx,css,md,json}\""
  },
  "dependencies": {
    "chart.js": "^4.4.1",
    "ws": "^8.16.0"
  },
  "devDependencies": {
    "@types/node": "^20.11.17",
    "@types/ws": "^8.5.10",
    "@typescript-eslint/eslint-plugin": "^6.21.0",
    "@typescript-eslint/parser": "^6.21.0",
    "@vitejs/plugin-legacy": "^5.3.1",
    "autoprefixer": "^10.4.17",
    "eslint": "^8.56.0",
    "postcss": "^8.4.35",
    "prettier": "^3.2.5",
    "tailwindcss": "^3.4.1",
    "typescript": "^5.3.3",
    "vite": "^5.1.1"
  },
  "engines": {
    "node": ">=20.0.0",
    "npm": ">=10.0.0"
  }
}
EOF
    print_success "package.json created"
fi

# Install Node.js dependencies
if [ -f "package.json" ]; then
    print_status "Installing Node.js dependencies..."
    npm install
    print_success "Node.js dependencies installed"
fi

# Create TypeScript config if it doesn't exist
if [ ! -f "tsconfig.json" ]; then
    print_status "Creating TypeScript configuration..."
    cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "preserve",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"],
      "@/components/*": ["./src/components/*"],
      "@/utils/*": ["./src/utils/*"]
    }
  },
  "include": [
    "src/**/*.ts",
    "src/**/*.tsx",
    "src/**/*.js",
    "src/**/*.jsx",
    "vite.config.ts"
  ],
  "references": [{ "path": "./tsconfig.node.json" }]
}
EOF

    cat > tsconfig.node.json << 'EOF'
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
EOF
    print_success "TypeScript configuration created"
fi

# Create Vite config if it doesn't exist
if [ ! -f "vite.config.ts" ]; then
    print_status "Creating Vite configuration..."
    cat > vite.config.ts << 'EOF'
import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  root: '.',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    sourcemap: true,
    target: 'es2022',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'src/prompt_improver/dashboard/index.html'),
      },
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    hmr: {
      port: 5173,
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
  esbuild: {
    target: 'es2022',
  },
  optimizeDeps: {
    include: ['chart.js', 'ws'],
  },
})
EOF
    print_success "Vite configuration created"
fi

# Set up database if needed
print_status "Setting up database..."
if command -v python >/dev/null 2>&1; then
    if python -c "import alembic" 2>/dev/null; then
        if [ -f "alembic.ini" ]; then
            print_status "Running database migrations..."
            python -m alembic upgrade head
            print_success "Database migrations completed"
        else
            print_warning "No alembic.ini found, skipping migrations"
        fi
    else
        print_warning "Alembic not installed, skipping migrations"
    fi
else
    print_warning "Python not found, skipping database setup"
fi

# Create frontend directory structure
print_status "Setting up frontend directory structure..."
mkdir -p src/frontend/{components,utils,types,styles}
mkdir -p public
mkdir -p dist

# Create a basic index.html for Vite if it doesn't exist
if [ ! -f "index.html" ]; then
    cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Prompt Improver Dashboard</title>
  </head>
  <body>
    <div id="app">
      <h1>Prompt Improver Development Environment</h1>
      <p>Vite + TypeScript development server is running!</p>
      <p>Visit <a href="/src/prompt_improver/dashboard/">/src/prompt_improver/dashboard/</a> for the main dashboard.</p>
    </div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
EOF
fi

# Create a basic TypeScript entry point
if [ ! -f "src/main.ts" ]; then
    mkdir -p src
    cat > src/main.ts << 'EOF'
// Main TypeScript entry point for Vite development
console.log('🚀 Prompt Improver development environment loaded!');
console.log('📊 Dashboard available at /src/prompt_improver/dashboard/');

// Hot Module Replacement (HMR) setup
if (import.meta.hot) {
  import.meta.hot.accept();
  console.log('🔥 Hot Module Replacement enabled');
}
EOF
fi

print_success "Frontend directory structure created"

# Set up pre-commit hooks if .pre-commit-config.yaml exists
if [ -f ".pre-commit-config.yaml" ]; then
    print_status "Installing pre-commit hooks..."
    pre-commit install
    print_success "Pre-commit hooks installed"
fi

# Final status
print_success "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick Start Commands:"
echo "  • npm run dev       - Start Vite development server with HMR"
echo "  • npm run build     - Build for production"
echo "  • npm run lint      - Run ESLint"
echo "  • python -m pytest  - Run Python tests"
echo "  • alembic upgrade head - Run database migrations"
echo ""
echo "🌐 Development URLs:"
echo "  • Vite Dev Server:   http://localhost:5173"
echo "  • Python API:       http://localhost:8000"
echo "  • TUI Dashboard:     python -m prompt_improver.tui.dashboard"
echo ""
print_success "Happy coding! 🚀"