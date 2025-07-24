# Multi-stage development container optimized for 2025 with multi-architecture support
ARG PYTHON_VERSION=3.12
ARG TARGETPLATFORM
ARG BUILDPLATFORM
FROM --platform=$BUILDPLATFORM python:${PYTHON_VERSION}-slim-bookworm AS base

# Multi-architecture build information
RUN echo "Building on $BUILDPLATFORM, targeting $TARGETPLATFORM" && \
    uname -a

# Set build arguments
ARG NODE_VERSION=20
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG ESBUILD_VERSION=latest

# Configure environment for optimal development
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspaces/prompt-improver/src \
    NODE_ENV=development \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies with performance optimizations
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials for Python packages
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    # PostgreSQL development files
    libpq-dev \
    postgresql-client \
    # Redis tools
    redis-tools \
    # Git and development tools
    git \
    curl \
    wget \
    unzip \
    vim \
    nano \
    htop \
    jq \
    tree \
    # Python development
    python3-dev \
    python3-venv \
    # ML dependencies
    libgomp1 \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    # System monitoring
    procps \
    lsof \
    net-tools \
    iputils-ping \
    dnsutils \
    iotop \
    nethogs \
    # Network utilities
    telnet \
    netcat-openbsd \
    # Node.js prerequisites  
    ca-certificates \
    gnupg \
    lsb-release \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 with performance optimizations
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest \
    && npm install -g pnpm@latest \
    && npm cache clean --force

# Create development user with proper permissions
RUN if [ "$USER_GID" != "1000" ] || [ "$USER_UID" != "1000" ]; then \
        groupadd --gid $USER_GID $USERNAME; \
    else \
        groupadd --gid $USER_GID $USERNAME; \
    fi \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python development tools globally
RUN pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    # Code formatting and linting
    black \
    isort \
    flake8 \
    mypy \
    # Testing
    pytest \
    pytest-cov \
    pytest-asyncio \
    # Development utilities
    ipython \
    jupyter \
    # Performance profiling
    line_profiler \
    memory_profiler \
    # Database tools
    psycopg[binary] \
    redis \
    # Modern development tools
    ruff \
    pre-commit

# Install Vite, esbuild, and TypeScript globally for fast access
RUN npm install -g \
    vite@latest \
    esbuild@${ESBUILD_VERSION} \
    typescript@latest \
    @types/node@latest \
    eslint@latest \
    prettier@latest \
    concurrently@latest \
    tsx@latest \
    nodemon@latest \
    # Performance monitoring tools
    clinic@latest \
    autocannon@latest \
    # Multi-architecture binary optimizer
    @esbuild/linux-arm64@latest \
    @esbuild/linux-x64@latest

# Configure git for development
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false \
    && git config --global credential.helper store

# Set up workspace directory with proper permissions
RUN mkdir -p /workspaces/prompt-improver \
    && chown -R $USERNAME:$USERNAME /workspaces/prompt-improver

# Switch to development user
USER $USERNAME
WORKDIR /workspaces/prompt-improver

# Configure shell environment for development
RUN echo 'export PATH="$PATH:/workspaces/prompt-improver/scripts"' >> ~/.bashrc \
    && echo 'export PYTHONPATH="/workspaces/prompt-improver/src:$PYTHONPATH"' >> ~/.bashrc \
    && echo 'alias ll="ls -alF"' >> ~/.bashrc \
    && echo 'alias la="ls -A"' >> ~/.bashrc \
    && echo 'alias l="ls -CF"' >> ~/.bashrc \
    && echo 'alias ..="cd .."' >> ~/.bashrc \
    && echo 'alias ...="cd ../.."' >> ~/.bashrc \
    && echo 'alias dev="npm run dev"' >> ~/.bashrc \
    && echo 'alias build="npm run build"' >> ~/.bashrc \
    && echo 'alias test="pytest tests/ -v"' >> ~/.bashrc \
    && echo 'alias lint="ruff check . && mypy src/"' >> ~/.bashrc

# Install Oh My Zsh for enhanced terminal experience
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && echo 'export PATH="$PATH:/workspaces/prompt-improver/scripts"' >> ~/.zshrc \
    && echo 'export PYTHONPATH="/workspaces/prompt-improver/src:$PYTHONPATH"' >> ~/.zshrc \
    && echo 'alias dev="npm run dev"' >> ~/.zshrc \
    && echo 'alias build="npm run build"' >> ~/.zshrc \
    && echo 'alias test="pytest tests/ -v"' >> ~/.zshrc \
    && echo 'alias lint="ruff check . && mypy src/"' >> ~/.zshrc

# Create directories for development caches
RUN mkdir -p ~/.cache/pip ~/.npm ~/.local/bin

# Health check for development environment
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 --version && node --version && npm --version || exit 1

# Default command
CMD ["bash"]

# Performance tuning for development
RUN echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf && \
    echo 'fs.inotify.max_user_watches=524288' | sudo tee -a /etc/sysctl.conf

# Labels for container metadata
LABEL maintainer="Prompt Improver Development Team" \
      version="2025.1" \
      description="Multi-architecture optimized development container for Prompt Improver with Python 3.12, Node.js 20, Vite, and esbuild support" \
      python.version="3.12" \
      node.version="20" \
      vite.enabled="true" \
      esbuild.enabled="true" \
      typescript.enabled="true" \
      architecture="multi" \
      hmr.optimized="true" \
      performance.target="sub-50ms"