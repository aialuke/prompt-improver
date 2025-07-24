"""Rich console utilities for APES CLI."""

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from typing import Any, Dict, List, Optional
import json

class ConsoleManager:
    """Centralized console management for APES CLI."""

    def __init__(self):
        self.console = Console()

    def print_success(self, message: str) -> None:
        """Print a success message with green styling."""
        self.console.print(f"✅ {message}", style="green")

    def print_error(self, message: str) -> None:
        """Print an error message with red styling."""
        self.console.print(f"❌ {message}", style="red")

    def print_warning(self, message: str) -> None:
        """Print a warning message with yellow styling."""
        self.console.print(f"⚠️  {message}", style="yellow")

    def print_info(self, message: str) -> None:
        """Print an info message with blue styling."""
        self.console.print(f"ℹ️  {message}", style="blue")

    def print_dim(self, message: str) -> None:
        """Print a dimmed message."""
        self.console.print(f"📍 {message}", style="dim")

    def print_json(self, data: Dict[str, Any]) -> None:
        """Print formatted JSON data."""
        self.console.print_json(data=data)

    def create_table(self, title: str, columns: List[str]) -> Table:
        """Create a formatted table with given title and columns."""
        table = Table(title=title)
        for column in columns:
            table.add_column(column)
        return table

    def create_panel(self, content: str, title: str = "", style: str = "blue") -> Panel:
        """Create a formatted panel with content."""
        return Panel(content, title=title, border_style=style)

    def print_status_table(self, status_data: Dict[str, Any]) -> None:
        """Print a formatted status table."""
        table = self.create_table("APES System Status", ["Component", "Status", "Details"])

        for component, details in status_data.items():
            if isinstance(details, dict):
                status = "✅ Running" if details.get("status") == "healthy" else "❌ Error"
                info = details.get("message", "N/A")
            else:
                status = str(details)
                info = ""

            table.add_row(component, status, info)

        self.console.print(table)

def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a standardized progress bar for CLI operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=Console(),
        expand=True,
    )

# Global console instance for backward compatibility
console = ConsoleManager().console