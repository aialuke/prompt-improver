#!/usr/bin/env python3
"""Demo script to show APES TUI dashboard functionality."""

import sys
import asyncio
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text

def create_demo_dashboard():
    """Create a demo dashboard showing what the TUI looks like."""
    console = Console()
    
    # Create layout
    layout = Layout()
    
    # Split into header and body
    layout.split_row(
        Layout(name="left", size=50),
        Layout(name="right", size=50)
    )
    
    # System Overview Panel
    system_table = Table(title="System Overview", show_header=False)
    system_table.add_column("Metric", style="cyan")
    system_table.add_column("Value", style="white")
    system_table.add_row("Status", "[green]ONLINE[/green]")
    system_table.add_row("Version", "3.0.0")
    system_table.add_row("Uptime", "7d 5h")
    system_table.add_row("Services", "[green]3/3[/green]")
    system_table.add_row("Memory", "[yellow]65.2%[/yellow]")
    system_table.add_row("CPU", "[green]23.1%[/green]")
    
    # AutoML Status Panel
    automl_table = Table(title="AutoML Status", show_header=False)
    automl_table.add_column("Metric", style="cyan")
    automl_table.add_column("Value", style="white")
    automl_table.add_row("Status", "[green]OPTIMIZING[/green]")
    automl_table.add_row("Progress", "65/100 (65%)")
    automl_table.add_row("Best Score", "0.8456")
    automl_table.add_row("Runtime", "2h 15m")
    automl_table.add_row("ETA", "45m")
    
    # Performance Metrics Panel
    perf_table = Table(title="Performance Metrics", show_header=False)
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="white")
    perf_table.add_row("Response Time", "[green]125.5ms[/green]")
    perf_table.add_row("Throughput", "45.2 req/s")
    perf_table.add_row("Error Rate", "[green]0.02%[/green]")
    perf_table.add_row("Cache Hit Rate", "[green]94.1%[/green]")
    perf_table.add_row("Queue Length", "[green]2[/green]")
    
    # A/B Testing Panel
    ab_table = Table(title="A/B Testing", show_header=False)
    ab_table.add_column("Metric", style="cyan")
    ab_table.add_column("Value", style="white")
    ab_table.add_row("Active Experiments", "3")
    ab_table.add_row("Success Rate", "75.0%")
    ab_table.add_row("Avg Improvement", "12.3%")
    ab_table.add_row("Significant Results", "2")
    
    # Arrange panels
    layout["left"].split_column(
        Layout(Panel(system_table, border_style="blue"), name="system"),
        Layout(Panel(perf_table, border_style="green"), name="performance")
    )
    
    layout["right"].split_column(
        Layout(Panel(automl_table, border_style="yellow"), name="automl"),
        Layout(Panel(ab_table, border_style="red"), name="abtesting")
    )
    
    return layout

def main():
    """Main demo function."""
    console = Console()
    
    console.print("🎛️ APES Interactive Dashboard Demo", style="bold blue")
    console.print("=" * 60)
    console.print()
    
    # Show what the TUI dashboard looks like
    console.print("📊 This is what the Rich TUI Dashboard looks like:", style="cyan")
    console.print()
    
    # Create and display demo dashboard
    dashboard_layout = create_demo_dashboard()
    console.print(dashboard_layout)
    
    console.print()
    console.print("🎯 Key Features:", style="bold green")
    console.print("• Real-time system monitoring")
    console.print("• AutoML optimization tracking")
    console.print("• A/B testing experiment results")
    console.print("• Performance metrics visualization")
    console.print("• Interactive service management")
    console.print("• Live data updates every 2 seconds")
    
    console.print()
    console.print("🚀 To launch the full interactive dashboard:", style="bold yellow")
    console.print("   python3 -m prompt_improver.cli interactive")
    
    console.print()
    console.print("💡 Dashboard Controls:", style="bold cyan")
    console.print("• Tab: Switch between panels")
    console.print("• Ctrl+C: Exit dashboard")
    console.print("• Ctrl+R: Refresh data")
    console.print("• Ctrl+T: Toggle dark/light mode")
    
    console.print()
    console.print("✅ TUI Dashboard Implementation Complete!", style="bold green")

if __name__ == "__main__":
    main()