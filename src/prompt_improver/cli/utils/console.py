"""Rich console utilities for APES CLI - Functional Implementation.

This module provides console utility functions following modern functional patterns.
All functions delegate to the console_utils module for consistency.
"""

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from typing import Any, Dict, List

# Import all utility functions from console_utils
from .console_utils import (
    print_success,
    print_error, 
    print_warning,
    print_info,
    print_dim,
    print_json,
    create_table,
    create_panel,
    print_status_table,
    get_console,
    print_with_style,
    print_rule,
    clear_console,
    print_tree_structure,
    print_columns
)

def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a standardized progress bar for CLI operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=get_console(),
        expand=True,
    )

# Global console instance for backward compatibility
console = get_console()