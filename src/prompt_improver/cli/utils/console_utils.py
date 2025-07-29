"""Console utilities for APES CLI - Converted from ConsoleManager class.

This module replaces the ConsoleManager class with standalone functions,
following the principle that simple wrapper classes should be converted
to utility functions for better simplicity and reduced overhead.
"""

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from typing import Any, Dict, List

# Global console instance for shared state
_console = Console()


def print_success(message: str) -> None:
    """Print a success message with green styling."""
    _console.print(f"✅ {message}", style="green")


def print_error(message: str) -> None:
    """Print an error message with red styling."""
    _console.print(f"❌ {message}", style="red")


def print_warning(message: str) -> None:
    """Print a warning message with yellow styling."""
    _console.print(f"⚠️  {message}", style="yellow")


def print_info(message: str) -> None:
    """Print an info message with blue styling."""
    _console.print(f"ℹ️  {message}", style="blue")


def print_dim(message: str) -> None:
    """Print a dimmed message."""
    _console.print(f"📍 {message}", style="dim")


def print_json(data: Dict[str, Any]) -> None:
    """Print formatted JSON data."""
    _console.print_json(data=data)


def create_table(title: str, columns: List[str]) -> Table:
    """Create a formatted table with given title and columns."""
    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    return table


def create_panel(content: str, title: str = "", style: str = "blue") -> Panel:
    """Create a formatted panel with content."""
    return Panel(content, title=title, border_style=style)


def print_status_table(status_data: Dict[str, Any]) -> None:
    """Print a formatted status table."""
    table = create_table("APES System Status", ["Component", "Status", "Details"])

    for component, details in status_data.items():
        if isinstance(details, dict):
            status = "✅ Running" if details.get("status") == "healthy" else "❌ Error"
            info = details.get("message", "N/A")
        else:
            status = str(details)
            info = ""

        table.add_row(component, status, info)

    _console.print(table)


def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a standardized progress bar for CLI operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=_console,
        expand=True,
    )


def get_console() -> Console:
    """Get the global console instance.
    
    Returns:
        Shared Console instance for advanced operations
    """
    return _console


def print_with_style(message: str, style: str) -> None:
    """Print message with custom style.
    
    Args:
        message: Message to print
        style: Rich style string (e.g., "bold red", "italic blue")
    """
    _console.print(message, style=style)


def print_rule(title: str = "", style: str = "blue") -> None:
    """Print a horizontal rule with optional title.
    
    Args:
        title: Optional title for the rule
        style: Style for the rule
    """
    _console.rule(title, style=style)


def clear_console() -> None:
    """Clear the console screen."""
    _console.clear()


def print_tree_structure(data: Dict[str, Any], title: str = "Structure") -> None:
    """Print hierarchical data as a tree structure.
    
    Args:
        data: Hierarchical data dictionary
        title: Title for the tree
    """
    from rich.tree import Tree
    
    tree = Tree(title)
    
    def add_branch(node: Tree, data_dict: Dict[str, Any]):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                branch = node.add(f"[bold]{key}[/bold]")
                add_branch(branch, value)
            elif isinstance(value, list):
                branch = node.add(f"[bold]{key}[/bold] ({len(value)} items)")
                for i, item in enumerate(value[:5]):  # Show first 5 items
                    branch.add(f"[dim]{i}: {str(item)[:50]}...[/dim]" if len(str(item)) > 50 else f"{i}: {item}")
                if len(value) > 5:
                    branch.add(f"[dim]... and {len(value) - 5} more items[/dim]")
            else:
                node.add(f"{key}: [cyan]{value}[/cyan]")
    
    add_branch(tree, data)
    _console.print(tree)


def print_columns(data: List[str], title: str = "", columns: int = 2) -> None:
    """Print data in columns layout.
    
    Args:
        data: List of strings to print
        title: Optional title
        columns: Number of columns
    """
    from rich.columns import Columns
    
    if title:
        print_rule(title)
    
    _console.print(Columns(data, equal=True, expand=True))


# Global console instance for backward compatibility
console = get_console()