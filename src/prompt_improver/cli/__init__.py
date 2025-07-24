"""
APES CLI - Ultra-Minimal 3-Command Interface
Complete replacement for legacy 36-command CLI with training-focused design.
"""

# Import the clean 3-command CLI
from .clean_cli import app

# Export the main CLI app
__all__ = ["app"]

# CLI transformation complete:
# - 36 commands → 3 commands (92% simplification)
# - MCP server management removed (architectural separation)
# - Training-focused design with continuous adaptive learning
# - Zero-configuration ML training with intelligent orchestrator-driven workflows
# - Comprehensive session reporting and progress preservation
