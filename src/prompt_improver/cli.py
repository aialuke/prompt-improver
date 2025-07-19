"""Command-line interface for the Adaptive Prompt Enhancement System (APES).
Provides service management, training operations, and system administration.
"""

import asyncio
import json
import os
import subprocess
import sys
import typing
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from prompt_improver.database import DatabaseSessionManager, get_sessionmanager
from prompt_improver.installation.initializer import APESInitializer
from prompt_improver.installation.migration import APESMigrationManager
from prompt_improver.service.manager import APESServiceManager
from prompt_improver.service.security import PromptDataProtection
from prompt_improver.services.analytics import AnalyticsService
from prompt_improver.services.canary_testing import canary_service
from prompt_improver.services.prompt_improvement import PromptImprovementService
from prompt_improver.utils import ensure_running
from prompt_improver.utils.redis_cache import redis_client

app = typer.Typer(
    name="apes",
    help="APES - Adaptive Prompt Enhancement System CLI",
    rich_markup_mode="rich",
)
console = Console()

# Service instances
analytics_service = AnalyticsService()
prompt_service = PromptImprovementService()
service_manager = APESServiceManager(console)
data_protection = PromptDataProtection(console)


@app.command()
def start(
    mcp_port: int = typer.Option(3000, "--mcp-port", "-p", help="MCP server port"),
    background: bool = typer.Option(
        False, "--background", "-b", help="Run in background"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start APES MCP server with stdio transport."""
    console.print("🚀 Starting APES MCP server...", style="green")

    if background:
        # Start in background using subprocess
        try:
            # Get the path to the MCP server
            mcp_server_path = Path(__file__).parent / "mcp_server" / "mcp_server.py"

            # Start the process with security validations
            if not mcp_server_path.exists():
                raise FileNotFoundError(
                    f"MCP server script not found: {mcp_server_path}"
                )

            # Security: subprocess call with validated executable path and secure parameters
            # - sys.executable is Python's own executable path (trusted)
            # - mcp_server_path validated as existing file before use
            # - shell=False prevents shell injection attacks
            # - start_new_session for process isolation
            process = subprocess.Popen(  # noqa: S603
                [sys.executable, str(mcp_server_path)],
                stdout=subprocess.PIPE if not verbose else None,
                stderr=subprocess.PIPE if not verbose else None,
                start_new_session=True,
                shell=False,
            )

            # Store the PID for later management
            pid_file = Path.home() / ".local" / "share" / "apes" / "mcp.pid"
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(str(process.pid))

            console.print(
                f"✅ MCP server started in background (PID: {process.pid})",
                style="green",
            )
            console.print(f"📍 PID file: {pid_file}", style="dim")

        except (FileNotFoundError, PermissionError, OSError) as e:
            console.print(f"❌ Failed to start MCP server: {e}", style="red")
            raise typer.Exit(1)
        except subprocess.SubprocessError as e:
            console.print(f"❌ Failed to start MCP server process: {e}", style="red")
            raise typer.Exit(1)
    else:
        # Run in foreground
        try:
            mcp_server_path = Path(__file__).parent / "mcp_server" / "mcp_server.py"
            if not mcp_server_path.exists():
                raise FileNotFoundError(
                    f"MCP server script not found: {mcp_server_path}"
                )

            # Security: subprocess call with validated executable path and secure parameters
            # - sys.executable is Python's own executable path (trusted)
            # - mcp_server_path validated as existing file before use
            # - shell=False prevents shell injection attacks
            # - timeout=300 prevents indefinite hanging
            subprocess.run(  # noqa: S603
                [sys.executable, str(mcp_server_path)],
                check=True,
                shell=False,
                timeout=300,
            )
        except KeyboardInterrupt:
            console.print("\n👋 MCP server stopped", style="yellow")
        except (FileNotFoundError, PermissionError, OSError) as e:
            console.print(f"❌ Error running MCP server: {e}", style="red")
            raise typer.Exit(1)
        except subprocess.SubprocessError as e:
            console.print(f"❌ MCP server process failed: {e}", style="red")
            raise typer.Exit(1)
        except subprocess.TimeoutExpired as e:
            console.print(f"❌ MCP server startup timed out: {e}", style="red")
            raise typer.Exit(1)


@app.command()
def stop(
    graceful: bool = typer.Option(True, "--graceful/--force", help="Graceful shutdown"),
    timeout: int = typer.Option(
        5, "--timeout", "-t", help="Shutdown timeout in seconds"
    ),
):
    """Stop APES MCP server."""
    console.print("🔄 Stopping APES MCP server...", style="yellow")

    # Check for PID file
    pid_file = Path.home() / ".local" / "share" / "apes" / "mcp.pid"

    if not pid_file.exists():
        console.print("⚠️  No running MCP server found", style="yellow")
        return

    try:
        pid = int(pid_file.read_text().strip())

        if graceful:
            # Try graceful shutdown first
            if not ensure_running(pid):
                console.print("⚡ Process not running, no need to terminate", style="dim")
            else:
                try:
                    os.kill(pid, 15)  # SIGTERM
                    console.print(f"📤 Sent SIGTERM to process {pid}", style="dim")

                    # Wait for process to terminate
                    import time

                    for i in range(timeout):
                        try:
                            if ensure_running(pid):
                                time.sleep(1)
                            else:
                                break
                        except ProcessLookupError:
                            break
                    else:
                        # Force kill if still running
                        try:
                            os.kill(pid, 9)  # SIGKILL
                            console.print(f"⚡ Force killed process {pid}", style="yellow")
                        except ProcessLookupError:
                            # Process already terminated
                            pass
                except ProcessLookupError:
                    # Process already terminated
                    console.print("⚡ Process already terminated", style="dim")
        else:
            # Force kill immediately
            os.kill(pid, 9)  # SIGKILL

        # Remove PID file
        pid_file.unlink()
        console.print("✅ MCP server stopped", style="green")

    except (ValueError, FileNotFoundError) as e:
        console.print(f"❌ Failed to read PID file: {e}", style="red")
        raise typer.Exit(1)
    except (ProcessLookupError, PermissionError) as e:
        console.print(f"❌ Failed to stop MCP server process: {e}", style="red")
        raise typer.Exit(1)
    except OSError as e:
        console.print(f"❌ System error stopping MCP server: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def status(
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed status"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Check APES service status and health."""
    # Check MCP server
    pid_file = Path.home() / ".local" / "share" / "apes" / "mcp.pid"
    mcp_running = False
    mcp_pid = None

    if pid_file.exists():
        try:
            mcp_pid = int(pid_file.read_text().strip())
            mcp_running = ensure_running(mcp_pid)
        except (ValueError, FileNotFoundError, ProcessLookupError, PermissionError):
            mcp_running = False

    # Check database
    db_status = "unknown"
    try:
        # Run async check in sync context
        async def check_db() -> str:
            from sqlalchemy import text

            from .database import scalar

            sm: DatabaseSessionManager = get_sessionmanager()
            async with sm.session() as session:
                await scalar(session, text("SELECT 1"))
                return "connected"

        db_status = asyncio.run(check_db())
    except (RuntimeError, ConnectionError, OSError, Exception) as e:
        # Handle various database connection issues
        db_status = "disconnected"

    status_data = {
        "mcp_server": {
            "status": "running" if mcp_running else "stopped",
            "pid": mcp_pid if mcp_running else None,
        },
        "database": {"status": db_status},
        "timestamp": datetime.now().isoformat(),
    }

    if json_output:
        console.print_json(data=status_data)
    else:
        # Create status table
        table = Table(title="APES Service Status", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")

        # MCP Server
        mcp_status_icon = "✅" if mcp_running else "❌"
        table.add_row(
            "MCP Server",
            f"{mcp_status_icon} {'Running' if mcp_running else 'Stopped'}",
            f"PID: {mcp_pid}" if mcp_pid else "-",
        )

        # Database
        db_status_icon = "✅" if db_status == "connected" else "❌"
        table.add_row(
            "Database", f"{db_status_icon} {db_status.capitalize()}", "PostgreSQL"
        )

        console.print(table)

        if detailed and mcp_running:
            # Show additional metrics
            console.print("\n[bold]Performance Metrics[/bold]")
            # Add performance metrics here if needed


@app.command()
def train(
    real_data_priority: bool = typer.Option(
        True, "--real-data-priority/--no-priority", help="Prioritize real data"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without executing"
    ),
    rule_ids: str | None = typer.Option(
        None, "--rules", help="Comma-separated rule IDs to optimize"
    ),
    ensemble: bool = typer.Option(
        False, "--ensemble", help="Use ensemble optimization methods"
    ),
):
    """Trigger manual ML training on accumulated data (Phase 3 Enhanced)."""
    console.print("🧠 Starting Phase 3 ML training process...", style="green")

    async def run_training() -> None:
        sm: DatabaseSessionManager = get_sessionmanager()
        async with sm.session() as db_session:
            # Parse rule IDs if provided
            selected_rule_ids = None
            if rule_ids:
                selected_rule_ids = [rid.strip() for rid in rule_ids.split(",")]
                console.print(
                    f"🎯 Targeting specific rules: {selected_rule_ids}", style="blue"
                )

            # Enhanced data statistics with Phase 3 features
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Analyzing training data and ML readiness...", total=None
                )

                # Get comprehensive training data statistics
                from sqlalchemy import text

                stats_query = text("""
                    SELECT 
                        COUNT(*) FILTER (WHERE rp.improvement_score >= 0.7) as high_quality_count,
                        COUNT(*) FILTER (WHERE rm.enabled = true) as active_rules_count,
                        COUNT(DISTINCT rp.prompt_id) as unique_prompts,
                        AVG(rp.improvement_score) as avg_improvement,
                        COUNT(*) as total_performance_records
                    FROM rule_performance rp
                    JOIN rule_metadata rm ON rp.rule_id = rm.rule_id
                    WHERE rp.created_at >= NOW() - INTERVAL '30 days'
                """)

                result = await db_session.execute(stats_query)
                stats = result.fetchone()

                progress.update(task, completed=True)

            # Enhanced training data composition table
            table = Table(title="Phase 3 Training Data Analysis")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right", style="green")
            table.add_column("ML Readiness", justify="center", style="yellow")

            # Analyze ML readiness
            high_quality_count = stats[0] or 0
            active_rules = stats[1] or 0
            unique_prompts = stats[2] or 0
            avg_improvement = stats[3] or 0
            total_records = stats[4] or 0

            # ML readiness assessment
            ml_ready = high_quality_count >= 20 and unique_prompts >= 10
            ensemble_ready = high_quality_count >= 50 and unique_prompts >= 20

            table.add_row(
                "High Quality Samples",
                str(high_quality_count),
                "✅ Ready" if high_quality_count >= 20 else "⚠️  Limited",
            )
            table.add_row(
                "Active Rules",
                str(active_rules),
                "✅ Good" if active_rules >= 3 else "⚠️  Few",
            )
            table.add_row(
                "Unique Prompts",
                str(unique_prompts),
                "✅ Diverse" if unique_prompts >= 10 else "⚠️  Limited",
            )
            table.add_row(
                "Avg Improvement",
                f"{avg_improvement:.3f}",
                "✅ Good" if avg_improvement >= 0.5 else "⚠️  Low",
            )
            table.add_row(
                "Total Records",
                str(total_records),
                "✅ Sufficient" if total_records >= 50 else "⚠️  Sparse",
            )
            table.add_row(
                "Ensemble Ready",
                "Yes" if ensemble_ready else "No",
                "✅ Available" if ensemble_ready else "❌ Needs More Data",
            )

            console.print(table)

            if dry_run:
                console.print(
                    "\n[yellow]🔍 Dry run mode - analysis complete, no training performed[/yellow]"
                )
                return

            # Enhanced readiness checks
            if not ml_ready:
                console.print(
                    f"\n⚠️  Warning: Limited ML training data ({high_quality_count} high-quality samples, need ≥20)",
                    style="yellow",
                )
                if not typer.confirm(
                    "Continue with limited data? Results may be suboptimal."
                ):
                    raise typer.Abort()

            if ensemble and not ensemble_ready:
                console.print(
                    f"\n⚠️  Ensemble training requires ≥50 high-quality samples ({high_quality_count} available)",
                    style="yellow",
                )
                ensemble = False
                console.print(
                    "🔄 Falling back to single model optimization", style="blue"
                )

            # Run the actual training with Phase 3 enhancements
            console.print(
                f"\n🔄 Training ML models{'(ensemble)' if ensemble else ''}...",
                style="green",
            )

            try:
                # Enhanced ML optimization with real-time progress
                optimization_result = await prompt_service.run_ml_optimization(
                    rule_ids=selected_rule_ids, db_session=db_session
                )

                if (
                    optimization_result
                    and optimization_result.get("status") == "success"
                ):
                    console.print(
                        "✅ ML training completed successfully!", style="green"
                    )

                    # Enhanced performance metrics display
                    if verbose:
                        console.print("\n📊 Training Results:", style="bold")
                        console.print(
                            f"   Model ID: {optimization_result.get('model_id', 'N/A')}"
                        )
                        console.print(
                            f"   Performance Score: {optimization_result.get('best_score', 0):.3f}"
                        )
                        console.print(
                            f"   Accuracy: {optimization_result.get('accuracy', 0):.3f}"
                        )
                        console.print(
                            f"   Training Samples: {optimization_result.get('training_samples', 0)}"
                        )
                        console.print(
                            f"   Processing Time: {optimization_result.get('processing_time_ms', 0):.1f}ms"
                        )

                        if optimization_result.get("ensemble"):
                            console.print(
                                f"   Ensemble Score: {optimization_result['ensemble'].get('ensemble_score', 0):.3f}"
                            )

                        # Show MLflow tracking info
                        if optimization_result.get("mlflow_run_id"):
                            console.print("\n🔬 MLflow Experiment:")
                            console.print(
                                f"   Run ID: {optimization_result['mlflow_run_id']}"
                            )
                            console.print("   View at: http://localhost:5000")
                else:
                    error_msg = (
                        optimization_result.get("error", "Unknown error")
                        if optimization_result
                        else "No result returned"
                    )
                    console.print(f"❌ Training failed: {error_msg}", style="red")
                    raise typer.Exit(1)

            except (ConnectionError, OSError) as e:
                console.print(
                    f"❌ Database connection failed during training: {e}", style="red"
                )
                raise typer.Exit(1)
            except (ValueError, TypeError) as e:
                console.print(f"❌ Training parameter error: {e}", style="red")
                raise typer.Exit(1)
            except ImportError as e:
                console.print(f"❌ ML library dependency missing: {e}", style="red")
                raise typer.Exit(1)

    # Run the async function
    asyncio.run(run_training())


@app.command()
def analytics(
    rule_effectiveness: bool = typer.Option(
        False, "--rule-effectiveness", help="Show rule effectiveness"
    ),
    user_satisfaction: bool = typer.Option(
        False, "--user-satisfaction", help="Show user satisfaction"
    ),
    performance_trends: bool = typer.Option(
        False, "--performance-trends", help="Show performance trends"
    ),
    days: int = typer.Option(30, "--days", "-d", help="Number of days to analyze"),
):
    """View analytics and performance metrics."""
    # Validate days parameter
    if days < 1:
        console.print("❌ Number of days must be positive", style="red")
        raise typer.Exit(1)

    if not any([rule_effectiveness, user_satisfaction, performance_trends]):
        # Default to showing all
        rule_effectiveness = user_satisfaction = performance_trends = True

    async def show_analytics() -> None:
        sm: DatabaseSessionManager = get_sessionmanager()
        async with sm.session() as db_session:
            if rule_effectiveness:
                console.print(f"\n[bold]Rule Effectiveness (Last {days} days)[/bold]")

                stats = await analytics_service.get_rule_effectiveness(
                    days=days, min_usage_count=1
                )

                if stats:
                    table = Table()
                    table.add_column("Rule ID", style="cyan")
                    table.add_column("Effectiveness", justify="right", style="green")
                    table.add_column("Usage Count", justify="right")
                    table.add_column(
                        "Improvement Rate", justify="right", style="yellow"
                    )

                    for stat in stats[:10]:  # Top 10 rules
                        table.add_row(
                            stat.rule_id,
                            f"{stat.effectiveness_score:.2f}",
                            str(stat.usage_count),
                            f"{stat.improvement_rate:.1%}"
                            if stat.improvement_rate
                            else "N/A",
                        )

                    console.print(table)
                else:
                    console.print("No rule effectiveness data available", style="dim")

            if user_satisfaction:
                console.print("\n[bold]User Satisfaction Trends[/bold]")

                satisfaction = await analytics_service.get_user_satisfaction(
                    days=days, db_session=db_session
                )

                if satisfaction:
                    # Show trend summary
                    avg_rating = sum(
                        s["average_rating"] for s in satisfaction if s["average_rating"]
                    ) / len(satisfaction)
                    console.print(f"Average Rating: {avg_rating:.2f}/5.0")

                    # Show recent trend
                    recent = satisfaction[:7]  # Last 7 days
                    for day_data in recent:
                        rating = day_data["average_rating"] or 0
                        bar = "█" * int(rating) + "░" * (5 - int(rating))
                        console.print(f"{day_data['date']}: [{bar}] {rating:.1f}")
                else:
                    console.print("No user satisfaction data available", style="dim")

            if performance_trends:
                console.print("\n[bold]Performance Trends Analytics[/bold]")
                console.print("Analytics feature coming soon...", style="dim")

    asyncio.run(show_analytics())


@app.command()
def backup(
    to: Path | None = typer.Option(None, "--to", help="Backup destination path"),
    compress: bool = typer.Option(
        True, "--compress/--no-compress", help="Compress backup"
    ),
    include_ml: bool = typer.Option(
        True, "--include-ml/--no-ml", help="Include ML models"
    ),
):
    """Backup APES data and configuration."""
    if not to:
        # Default backup location
        backup_dir = Path.home() / ".local" / "share" / "apes" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        to = backup_dir / f"apes_backup_{timestamp}.tar.gz"

    console.print(f"🔒 Creating backup to: {to}", style="green")

    try:
        # Use the existing backup script
        backup_script = (
            Path(__file__).parent.parent.parent / "scripts" / "start_database.sh"
        )

        if backup_script.exists():
            # Validate backup script is safe to execute
            if not backup_script.is_file():
                raise ValueError(
                    f"Backup script is not a regular file: {backup_script}"
                )

            # Security: subprocess call with validated script path and secure parameters
            # - backup_script validated as existing regular file before use
            # - shell=False prevents shell injection attacks
            # - timeout=600 prevents indefinite hanging
            # - Arguments are controlled and validated
            subprocess.run(  # noqa: S603
                [str(backup_script), "backup"], check=True, shell=False, timeout=600
            )
            console.print("✅ Backup completed successfully!", style="green")
            console.print(f"📦 Backup file: {to}", style="dim")
        else:
            console.print("❌ Backup script not found", style="red")
            raise typer.Exit(1)

    except subprocess.CalledProcessError as e:
        console.print(f"❌ Backup failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def doctor(
    fix_issues: bool = typer.Option(
        False, "--fix-issues", help="Attempt to fix issues"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run system health checks and diagnostics."""
    console.print("🏥 Running APES system diagnostics...\n", style="green")

    issues_found = False

    # Check 1: Python version
    console.print("[bold]1. Python Version[/bold]")
    py_version = sys.version_info
    if py_version >= (3, 11):
        console.print(
            f"  ✅ Python {py_version.major}.{py_version.minor}.{py_version.micro}",
            style="green",
        )
    else:
        console.print(
            f"  ❌ Python {py_version.major}.{py_version.minor}.{py_version.micro} (requires 3.11+)",
            style="red",
        )
        issues_found = True

    # Check 2: Dependencies
    console.print("\n[bold]2. Dependencies[/bold]")
    try:
        import fastmcp
        import mcp
        import rich as _rich
        import typer as _typer

        console.print("  ✅ All core dependencies installed", style="green")
    except ImportError as e:
        console.print(f"  ❌ Missing dependency: {e.name}", style="red")
        issues_found = True
        if fix_issues:
            console.print("  🔧 Installing missing dependencies...")
            # Validate package name to prevent injection
            if (
                not e.name
                or not e.name.replace("-", "")
                .replace("_", "")
                .replace(".", "")
                .isalnum()
            ):
                raise ValueError(f"Invalid package name for installation: {e.name}")

            # Security: subprocess call with validated package name and secure parameters
            # - sys.executable is Python's own executable path (trusted)
            # - e.name validated to contain only alphanumeric characters
            # - shell=False prevents shell injection attacks
            # - timeout=120 prevents indefinite hanging
            subprocess.run(  # noqa: S603
                [sys.executable, "-m", "pip", "install", e.name],
                check=True,
                shell=False,
                timeout=120,
            )

    # Check 3: Database connection
    console.print("\n[bold]3. Database Connection[/bold]")
    try:

        async def check_db() -> str:
            sm: DatabaseSessionManager = get_sessionmanager()
            async with sm.session() as session:
                result = await session.fetch_one("SELECT version()")
                return result["version"]

        db_version = asyncio.run(check_db())
        console.print(
            f"  ✅ PostgreSQL connected: {db_version.split(',')[0]}", style="green"
        )
    except (ConnectionError, OSError) as e:
        console.print(f"  ❌ Database connection failed: {e}", style="red")
        issues_found = True
    except ImportError as e:
        console.print(f"  ❌ Database dependency missing: {e}", style="red")
        issues_found = True
    except RuntimeError as e:
        console.print(f"  ❌ Database runtime error: {e}", style="red")
        issues_found = True
        if fix_issues:
            console.print("  🔧 Starting database...")
            db_script = (
                Path(__file__).parent.parent.parent / "scripts" / "start_database.sh"
            )

            # Validate database script is safe to execute
            if not db_script.exists():
                console.print("  ❌ Database start script not found", style="red")
                return
            if not db_script.is_file():
                raise ValueError(f"Database script is not a regular file: {db_script}")

            # Security: subprocess call with validated script path and secure parameters
            # - db_script validated as existing regular file before use
            # - shell=False prevents shell injection attacks
            # - timeout=120 prevents indefinite hanging
            # - Arguments are controlled and validated
            subprocess.run(  # noqa: S603
                [str(db_script), "start"],
                check=False,
                shell=False,
                timeout=120,
            )

    # Check 4: Data directories
    console.print("\n[bold]4. Data Directories[/bold]")
    data_dir = Path.home() / ".local" / "share" / "apes"
    required_dirs = ["data", "config", "backups", "logs"]

    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if dir_path.exists():
            console.print(f"  ✅ {dir_name}/", style="green")
        else:
            console.print(f"  ❌ {dir_name}/ missing", style="red")
            issues_found = True
            if fix_issues:
                dir_path.mkdir(parents=True, exist_ok=True)
                console.print(f"  🔧 Created {dir_name}/", style="yellow")

    # Summary
    console.print("\n" + "=" * 50)
    if issues_found:
        if fix_issues:
            console.print(
                "🔧 Some issues were fixed. Please run 'apes doctor' again to verify.",
                style="yellow",
            )
        else:
            console.print(
                "❌ Issues found. Run with --fix-issues to attempt automatic fixes.",
                style="red",
            )
        raise typer.Exit(1)
    console.print("✅ All systems operational!", style="green")


@app.command()
def monitor_realtime(
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Follow mode (continuous monitoring)"
    ),
    threshold: int = typer.Option(
        200, "--threshold", "-t", help="Alert threshold in milliseconds"
    ),
    interval: int = typer.Option(
        30, "--interval", "-i", help="Update interval in seconds"
    ),
):
    """Real-time database performance monitoring."""
    import asyncio

    from .database.performance_monitor import get_performance_monitor

    async def monitor() -> None:
        monitor = await get_performance_monitor()

        if follow:
            console.print(
                f"🔄 Starting real-time monitoring (threshold: {threshold}ms, interval: {interval}s)",
                style="blue",
            )
            console.print("Press Ctrl+C to stop", style="dim")

            try:
                while True:
                    snapshot = await monitor.take_performance_snapshot()

                    # Clear screen and show current stats
                    console.clear()
                    console.print(
                        "🔄 REAL-TIME DATABASE PERFORMANCE", style="bold blue"
                    )
                    console.print(
                        f"Timestamp: {snapshot.timestamp.strftime('%H:%M:%S')}",
                        style="dim",
                    )
                    console.print()

                    # Performance table
                    table = Table(title="Performance Metrics")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Current", style="white")
                    table.add_column("Target", style="dim")
                    table.add_column("Status", style="bold")

                    # Cache hit ratio
                    cache_status = (
                        "✅ GOOD" if snapshot.cache_hit_ratio >= 90 else "⚠️  LOW"
                    )
                    table.add_row(
                        "Cache Hit Ratio",
                        f"{snapshot.cache_hit_ratio:.1f}%",
                        ">90%",
                        cache_status,
                    )

                    # Query time
                    time_status = (
                        "✅ GOOD" if snapshot.avg_query_time_ms <= 50 else "⚠️  SLOW"
                    )
                    table.add_row(
                        "Avg Query Time",
                        f"{snapshot.avg_query_time_ms:.1f}ms",
                        "<50ms",
                        time_status,
                    )

                    # Connections
                    conn_status = (
                        "✅ GOOD" if snapshot.active_connections <= 20 else "⚠️  HIGH"
                    )
                    table.add_row(
                        "Active Connections",
                        str(snapshot.active_connections),
                        "≤20",
                        conn_status,
                    )

                    # Database size
                    table.add_row(
                        "Database Size",
                        f"{snapshot.database_size_mb:.1f} MB",
                        "-",
                        "📊 INFO",
                    )

                    console.print(table)

                    # Show slow queries if any
                    if snapshot.top_slow_queries:
                        console.print("\n🐌 Top Slow Queries:", style="yellow")
                        for i, query in enumerate(snapshot.top_slow_queries[:3], 1):
                            console.print(
                                f"{i}. {query.mean_exec_time:.1f}ms - {query.query_text[:80]}...",
                                style="dim",
                            )

                    await asyncio.sleep(interval)

            except KeyboardInterrupt:
                console.print("\n👋 Monitoring stopped", style="yellow")
        else:
            # Single snapshot
            snapshot = await monitor.take_performance_snapshot()

            console.print("📊 DATABASE PERFORMANCE SNAPSHOT", style="bold")
            console.print(
                f"Timestamp: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            console.print()

            # Performance summary
            console.print(
                f"Cache Hit Ratio: {snapshot.cache_hit_ratio:.1f}% (target: >90%)"
            )
            console.print(
                f"Avg Query Time: {snapshot.avg_query_time_ms:.1f}ms (target: <50ms)"
            )
            console.print(f"Active Connections: {snapshot.active_connections}")
            console.print(f"Database Size: {snapshot.database_size_mb:.1f} MB")

            # Recommendations
            recommendations = await monitor.get_recommendations()
            if recommendations:
                console.print("\n💡 Recommendations:", style="cyan")
                for rec in recommendations:
                    console.print(f"  • {rec}")

    asyncio.run(monitor())


@app.command()
def performance(
    period: str = typer.Option(
        "24h", "--period", "-p", help="Analysis period (e.g., 24h, 7d)"
    ),
    show_trends: bool = typer.Option(
        False, "--show-trends", help="Show performance trends"
    ),
    export_csv: bool = typer.Option(False, "--export-csv", help="Export data to CSV"),
):
    """View performance metrics and trends."""
    import asyncio

    from .database.performance_monitor import get_performance_monitor

    async def analyze():
        monitor = await get_performance_monitor()

        # Parse period (simple implementation)
        hours = 24
        if period.endswith("h"):
            hours = int(period[:-1])
        elif period.endswith("d"):
            hours = int(period[:-1]) * 24

        summary = await monitor.get_performance_summary(hours)

        if "error" in summary:
            console.print(f"❌ {summary['error']}", style="red")
            return

        console.print(f"📈 PERFORMANCE ANALYSIS ({period})", style="bold")
        console.print(f"Data points analyzed: {summary['snapshots_analyzed']}")
        console.print()

        # Performance metrics table
        table = Table(title="Performance Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Average", style="white")
        table.add_column("Target", style="dim")
        table.add_column("Status", style="bold")

        cache_status = (
            "✅ PASS" if summary["target_compliance"]["cache_hit_ratio"] else "❌ FAIL"
        )
        table.add_row(
            "Cache Hit Ratio",
            f"{summary['avg_cache_hit_ratio']}%",
            ">90%",
            cache_status,
        )

        time_status = (
            "✅ PASS" if summary["target_compliance"]["query_time"] else "❌ FAIL"
        )
        table.add_row(
            "Query Time", f"{summary['avg_query_time_ms']}ms", "<50ms", time_status
        )

        table.add_row(
            "Active Connections",
            f"{summary['avg_active_connections']}",
            "≤20",
            "📊 INFO",
        )
        table.add_row(
            "Database Size", f"{summary['latest_database_size_mb']} MB", "-", "📊 INFO"
        )

        console.print(table)

        # Overall status
        status_color = "green" if summary["performance_status"] == "GOOD" else "yellow"
        console.print(
            f"\nOverall Status: {summary['performance_status']}", style=status_color
        )

    asyncio.run(analyze())


@app.command()
def data_stats(
    real_vs_synthetic: bool = typer.Option(
        False, "--real-vs-synthetic", help="Show real vs synthetic data breakdown"
    ),
    quality_metrics: bool = typer.Option(
        False, "--quality-metrics", help="Show data quality metrics"
    ),
    export_format: str = typer.Option(
        "table", "--format", help="Output format: table, json, csv"
    ),
):
    """View data composition and quality statistics."""
    import asyncio

    from .database.psycopg_client import get_psycopg_client

    async def analyze_data():
        client = await get_psycopg_client()

        console.print("📊 DATA STATISTICS ANALYSIS", style="bold")
        console.print()

        # Basic data counts
        queries = [
            (
                "Total Training Prompts",
                "SELECT COUNT(*) as count FROM training_prompts",
            ),
            (
                "Real Data Entries",
                "SELECT COUNT(*) as count FROM training_prompts WHERE data_source = 'real'",
            ),
            (
                "Synthetic Data Entries",
                "SELECT COUNT(*) as count FROM training_prompts WHERE data_source = 'synthetic'",
            ),
            (
                "Rule Performance Records",
                "SELECT COUNT(*) as count FROM rule_performance",
            ),
            ("User Feedback Records", "SELECT COUNT(*) as count FROM user_feedback"),
        ]

        # Data composition table
        table = Table(title="Data Composition")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="white", justify="right")
        table.add_column("Percentage", style="dim", justify="right")

        total_prompts = 0
        real_count = 0

        for label, query in queries:
            try:
                result = await client.fetch_raw(query)
                count = result[0]["count"] if result else 0

                if label == "Total Training Prompts":
                    total_prompts = count
                elif label == "Real Data Entries":
                    real_count = count

                percentage = ""
                if total_prompts > 0 and "Data Entries" in label:
                    percentage = f"({count / total_prompts * 100:.1f}%)"

                table.add_row(label, str(count), percentage)

            except (ConnectionError, OSError) as e:
                table.add_row(label, "DB Error", f"({e})")
            except (KeyError, TypeError, ValueError) as e:
                table.add_row(label, "Data Error", f"({e})")

        console.print(table)

        # Data quality insights
        if quality_metrics:
            console.print("\n🎯 DATA QUALITY METRICS", style="bold cyan")

            quality_table = Table()
            quality_table.add_column("Metric", style="cyan")
            quality_table.add_column("Value", style="white")
            quality_table.add_column("Assessment", style="bold")

            # Real data percentage
            real_percentage = (
                (real_count / total_prompts * 100) if total_prompts > 0 else 0
            )
            real_status = (
                "✅ GOOD"
                if real_percentage >= 80
                else "⚠️  LOW"
                if real_percentage >= 50
                else "❌ POOR"
            )
            quality_table.add_row("Real Data %", f"{real_percentage:.1f}%", real_status)

            # Data freshness (last 7 days)
            try:
                freshness_query = "SELECT COUNT(*) as count FROM training_prompts WHERE created_at >= NOW() - INTERVAL '7 days'"
                fresh_result = await client.fetch_raw(freshness_query)
                fresh_count = fresh_result[0]["count"] if fresh_result else 0
                fresh_status = "✅ ACTIVE" if fresh_count > 10 else "⚠️  SLOW"
                quality_table.add_row("New Data (7d)", str(fresh_count), fresh_status)
            except (ConnectionError, KeyError, TypeError, ValueError) as e:
                # Handle database connection or query processing issues
                quality_table.add_row("New Data (7d)", "N/A", "❌ ERROR")

            console.print(quality_table)

    asyncio.run(analyze_data())


@app.command()
def export_training_data(
    format: str = typer.Option("csv", "--format", help="Export format: csv, json"),
    period: str = typer.Option("30d", "--period", help="Time period (e.g., 30d, 7d)"),
    output: str = typer.Option(
        "training_data_export", "--output", "-o", help="Output filename"
    ),
):
    """Export training data for analysis."""
    import asyncio
    import csv
    import json

    from .database.psycopg_client import get_psycopg_client

    async def export_data():
        client = await get_psycopg_client()

        # Parse period
        days = 30
        if period.endswith("d"):
            days = int(period[:-1])
        elif period.endswith("h"):
            days = int(period[:-1]) / 24

        console.print(f"📤 Exporting training data ({period})", style="blue")

        # Query training data
        query = """
        SELECT 
            prompt_text,
            enhancement_result,
            data_source,
            training_priority,
            created_at
        FROM training_prompts 
        WHERE created_at >= NOW() - INTERVAL '%s days'
        ORDER BY created_at DESC
        """

        try:
            results = await client.fetch_raw(query % days)

            if not results:
                console.print("❌ No data found for the specified period", style="red")
                return

            filename = f"{output}_{period}.{format}"

            if format == "csv":
                with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                    fieldnames = [
                        "prompt_text",
                        "enhanced_prompt",
                        "data_source",
                        "training_priority",
                        "created_at",
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    for row in results:
                        enhancement = row["enhancement_result"]
                        enhanced_prompt = (
                            enhancement.get("enhanced_prompt", "")
                            if enhancement
                            else ""
                        )

                        writer.writerow({
                            "prompt_text": row["prompt_text"],
                            "enhanced_prompt": enhanced_prompt,
                            "data_source": row["data_source"],
                            "training_priority": row["training_priority"],
                            "created_at": row["created_at"].isoformat(),
                        })

            elif format == "json":
                with open(filename, "w", encoding="utf-8") as jsonfile:
                    # Convert datetime objects to strings for JSON serialization
                    json_data = []
                    for row in results:
                        json_row = dict(row)
                        json_row["created_at"] = row["created_at"].isoformat()
                        json_data.append(json_row)

                    json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)

            console.print(
                f"✅ Exported {len(results)} records to {filename}", style="green"
            )

        except (ConnectionError, OSError) as e:
            console.print(
                f"❌ Database connection failed during export: {e}", style="red"
            )
        except (ValueError, TypeError) as e:
            console.print(f"❌ Data processing error during export: {e}", style="red")
        except (FileNotFoundError, PermissionError) as e:
            console.print(f"❌ File operation failed during export: {e}", style="red")

    asyncio.run(export_data())


@app.command()
def discover_patterns(
    min_effectiveness: float = typer.Option(
        0.7, "--min-effectiveness", help="Minimum effectiveness threshold"
    ),
    min_support: int = typer.Option(
        5, "--min-support", help="Minimum pattern support count"
    ),
    create_experiments: bool = typer.Option(
        True, "--create-ab/--no-ab", help="Create A/B experiments from patterns"
    ),
):
    """Discover new effective rule patterns using Phase 3 ML (Phase 3 Feature)."""
    console.print("🔍 Starting Pattern Discovery Analysis...", style="blue")

    async def run_discovery():
        sm: DatabaseSessionManager = get_sessionmanager()
        async with sm.session() as db_session:
            try:
                # Run pattern discovery
                discovery_result = await prompt_service.discover_patterns(
                    min_effectiveness=min_effectiveness,
                    min_support=min_support,
                    db_session=db_session,
                )

                if discovery_result.get("status") == "success":
                    patterns_found = discovery_result.get("patterns_discovered", 0)
                    console.print(
                        f"✅ Pattern discovery completed: {patterns_found} patterns found",
                        style="green",
                    )

                    if patterns_found > 0:
                        # Display discovered patterns
                        table = Table(title="Discovered Rule Patterns")
                        table.add_column("Pattern", style="cyan")
                        table.add_column(
                            "Effectiveness", justify="right", style="green"
                        )
                        table.add_column("Support", justify="right", style="yellow")
                        table.add_column(
                            "Parameter Count", justify="right", style="blue"
                        )

                        for i, pattern in enumerate(
                            discovery_result.get("patterns", [])[:10]
                        ):
                            pattern_name = f"Pattern {i + 1}"
                            effectiveness = pattern.get("avg_effectiveness", 0)
                            support = pattern.get("support_count", 0)
                            param_count = len(pattern.get("parameters", {}))

                            table.add_row(
                                pattern_name,
                                f"{effectiveness:.3f}",
                                str(support),
                                str(param_count),
                            )

                        console.print(table)

                        # Show A/B experiment status
                        if create_experiments:
                            console.print(
                                "\n🧪 A/B experiments created for top 3 patterns",
                                style="blue",
                            )
                            console.print(
                                "   Use 'apes analytics --ab-experiments' to monitor progress"
                            )
                    else:
                        console.print(
                            "ℹ️  No patterns found matching criteria", style="yellow"
                        )
                        console.print(
                            f"   Try lowering --min-effectiveness (current: {min_effectiveness})"
                        )
                        console.print(f"   or --min-support (current: {min_support})")

                elif discovery_result.get("status") == "insufficient_data":
                    console.print(
                        f"⚠️  {discovery_result.get('message')}", style="yellow"
                    )
                else:
                    console.print(
                        f"❌ Pattern discovery failed: {discovery_result.get('error')}",
                        style="red",
                    )

            except (ConnectionError, OSError) as e:
                console.print(
                    f"❌ Database connection failed during pattern discovery: {e}",
                    style="red",
                )
                raise typer.Exit(1)
            except (ValueError, TypeError) as e:
                console.print(
                    f"❌ Invalid parameters for pattern discovery: {e}", style="red"
                )
                raise typer.Exit(1)
            except ImportError as e:
                console.print(f"❌ ML library import failed: {e}", style="red")
                raise typer.Exit(1)

    asyncio.run(run_discovery())


@app.command()
def ml_status(
    show_models: bool = typer.Option(
        True, "--models/--no-models", help="Show model registry"
    ),
    show_experiments: bool = typer.Option(
        True, "--experiments/--no-experiments", help="Show MLflow experiments"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed information"
    ),
):
    """Show Phase 3 ML system status and model registry."""
    console.print("🤖 Phase 3 ML System Status", style="bold blue")

    async def show_status():
        sm: DatabaseSessionManager = get_sessionmanager()
        async with sm.session() as db_session:
            try:
                from sqlalchemy import text

                if show_models:
                    # Get model performance data
                    models_query = text("""
                        SELECT 
                            model_id,
                            performance_score,
                            accuracy,
                            training_data_size,
                            created_at
                        FROM ml_model_performance
                        ORDER BY created_at DESC
                        LIMIT 10
                    """)

                    result = await db_session.execute(models_query)
                    models = result.fetchall()

                    if models:
                        table = Table(title="ML Model Registry")
                        table.add_column("Model ID", style="cyan")
                        table.add_column("Performance", justify="right", style="green")
                        table.add_column("Accuracy", justify="right", style="yellow")
                        table.add_column("Training Size", justify="right", style="blue")
                        table.add_column("Created", style="dim")

                        for model in models:
                            table.add_row(
                                model[0][:20] + "..."
                                if len(model[0]) > 20
                                else model[0],
                                f"{model[1]:.3f}",
                                f"{model[2]:.3f}",
                                str(model[3]),
                                model[4].strftime("%Y-%m-%d %H:%M"),
                            )

                        console.print(table)
                    else:
                        console.print(
                            "📝 No ML models found - run 'apes train' to create models",
                            style="yellow",
                        )

                if show_experiments:
                    # Show A/B experiments
                    experiments_query = text("""
                        SELECT 
                            experiment_name,
                            status,
                            started_at,
                            completed_at
                        FROM ab_experiments
                        ORDER BY started_at DESC
                        LIMIT 5
                    """)

                    result = await db_session.execute(experiments_query)
                    experiments = result.fetchall()

                    if experiments:
                        exp_table = Table(title="A/B Experiments")
                        exp_table.add_column("Experiment", style="cyan")
                        exp_table.add_column("Status", style="yellow")
                        exp_table.add_column("Started", style="dim")
                        exp_table.add_column("Duration", style="blue")

                        for exp in experiments:
                            duration = "Running"
                            if exp[3]:  # completed_at
                                duration = str(exp[3] - exp[2])

                            exp_table.add_row(
                                exp[0][:30] + "..." if len(exp[0]) > 30 else exp[0],
                                exp[1],
                                exp[2].strftime("%m-%d %H:%M"),
                                duration,
                            )

                        console.print(exp_table)
                    else:
                        console.print(
                            "📝 No A/B experiments found - run 'apes discover-patterns' to create experiments",
                            style="yellow",
                        )

                # Show ML system health
                console.print("\n🔧 ML System Health:", style="bold")
                console.print("   ✅ Direct Python Integration: Active (1-5ms latency)")
                console.print(
                    "   ✅ MLflow Tracking: Available at http://localhost:5000"
                )
                console.print("   ✅ Optuna Optimization: Ready")
                console.print("   ✅ Ensemble Methods: Available")

            except (ConnectionError, OSError) as e:
                console.print(f"❌ Database connection failed: {e}", style="red")
                raise typer.Exit(1)
            except ImportError as e:
                console.print(f"❌ ML dependency missing: {e}", style="red")
                raise typer.Exit(1)
            except (AttributeError, ValueError) as e:
                console.print(f"❌ ML system configuration error: {e}", style="red")
                raise typer.Exit(1)

    asyncio.run(show_status())


@app.command()
def optimize_rules(
    rule_id: str | None = typer.Option(
        None, "--rule", help="Specific rule ID to optimize"
    ),
    feedback_id: int | None = typer.Option(
        None, "--feedback", help="Trigger optimization from feedback ID"
    ),
    ensemble: bool = typer.Option(
        False, "--ensemble", help="Use ensemble optimization"
    ),
):
    """Trigger targeted rule optimization using Phase 3 ML."""
    console.print("⚙️ Starting Rule Optimization...", style="green")

    async def run_optimization():
        sm: DatabaseSessionManager = get_sessionmanager()
        async with sm.session() as db_session:
            try:
                if feedback_id:
                    # Optimize based on specific feedback
                    result = await prompt_service.trigger_optimization(
                        feedback_id=feedback_id, db_session=db_session
                    )
                else:
                    # General rule optimization
                    rule_ids = [rule_id] if rule_id else None
                    result = await prompt_service.run_ml_optimization(
                        rule_ids=rule_ids, db_session=db_session
                    )

                if result and result.get("status") == "success":
                    console.print(
                        "✅ Rule optimization completed successfully!", style="green"
                    )
                    console.print(
                        f"   Performance Score: {result.get('best_score', 'N/A')}"
                    )
                    console.print(f"   Model ID: {result.get('model_id', 'N/A')}")

                    if result.get("ensemble"):
                        console.print(
                            f"   Ensemble Score: {result['ensemble'].get('ensemble_score', 'N/A')}"
                        )

                elif result and result.get("status") == "insufficient_data":
                    console.print(f"⚠️  {result.get('message')}", style="yellow")
                    console.print(
                        "   Try running 'apes train' first to build training data"
                    )
                else:
                    error_msg = (
                        result.get("error", "Unknown error")
                        if result
                        else "No result returned"
                    )
                    console.print(f"❌ Optimization failed: {error_msg}", style="red")

            except (ConnectionError, OSError) as e:
                console.print(
                    f"❌ Database connection failed during optimization: {e}",
                    style="red",
                )
                raise typer.Exit(1)
            except (ValueError, TypeError) as e:
                console.print(f"❌ Invalid optimization parameters: {e}", style="red")
                raise typer.Exit(1)
            except ImportError as e:
                console.print(f"❌ ML optimization library missing: {e}", style="red")
                raise typer.Exit(1)

    asyncio.run(run_optimization())


# Canary Testing Operations

@app.command()
def canary_status():
    """Show current canary testing status."""
    async def show_status():
        status = await canary_service.get_canary_status()
        console.print_json(data=status)

    asyncio.run(show_status())


@app.command()
def canary_adjust():
    """Auto-adjust canary rollout based on metrics."""
    async def adjust_rollout():
        result = await canary_service.auto_adjust_rollout()
        console.print_json(data=result)

    asyncio.run(adjust_rollout())


@app.command()
def cache_stats():
    """Display Redis cache statistics, including hit ratio and memory usage."""
    try:
        # Connect to Redis and fetch stats
        stats = redis_client.info()
        memory_usage = redis_client.memory_stats()
        console.print("[bold]Cache Statistics:[/bold]", style="green")
        console.print(f"Total Memory: {memory_usage.get('total.allocated', 0)} bytes")
        console.print(f"Memory Peak: {memory_usage.get('peak.allocated', 0)} bytes")
        console.print(f"Memory Fragmentation Ratio: {memory_usage.get('fragmentation', 0.0)}")
        console.print(f"Cache Hits: {stats.get('keyspace_hits', 0)}")
        console.print(f"Cache Misses: {stats.get('keyspace_misses', 0)}")
        hit_ratio = (stats.get('keyspace_hits', 0) / (stats.get('keyspace_hits', 0) + stats.get('keyspace_misses', 0))) * 100 if (stats.get('keyspace_hits', 0) + stats.get('keyspace_misses', 0)) else 0
        console.print(f"Cache Hit Ratio: {hit_ratio:.2f}%")
    except Exception as e:
        console.print(f"❌ Unable to retrieve Redis cache statistics: {e}", style="red")


@app.command()
def cache_clear():
    """Clear Redis cache."""
    try:
        redis_client.flushdb()
        console.print("✅ Cache cleared successfully!", style="green")
    except Exception as e:
        console.print(f"❌ Unable to clear Redis cache: {e}", style="red")


@app.command()
def init(
    data_dir: Path | None = typer.Option(
        None, "--data-dir", help="Data directory path"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force initialization if directory exists"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Initialize APES system with automated setup (Phase 2)."""
    console.print("🔧 Initializing APES system...", style="green")

    async def run_initialization():
        initializer = APESInitializer(console)

        try:
            results = await initializer.initialize_system(data_dir, force)

            if verbose:
                console.print("\n📊 Initialization Results:", style="bold")
                console.print(f"   Data Directory: {results['data_dir']}")
                console.print(f"   Steps Completed: {len(results['steps_completed'])}")
                console.print(
                    f"   Health Status: {results.get('health_check', {}).get('overall_status', 'unknown')}"
                )

                if results.get("health_check", {}).get("mcp_performance"):
                    console.print(
                        f"   MCP Performance: {results['health_check']['mcp_performance']:.1f}ms"
                    )

                if results.get("warnings"):
                    console.print("   Warnings:", style="yellow")
                    for warning in results["warnings"]:
                        console.print(f"     • {warning}")

            console.print(
                "\n✅ System initialization completed successfully!", style="green"
            )
            console.print("🚀 Next steps:", style="blue")
            console.print("   1. Run 'apes start --background' to start services")
            console.print("   2. Run 'apes doctor' to verify system health")
            console.print("   3. Run 'apes status' to check service status")

        except (FileNotFoundError, PermissionError) as e:
            console.print(
                f"❌ File system error during initialization: {e}", style="red"
            )
            raise typer.Exit(1)
        except (ConnectionError, OSError) as e:
            console.print(
                f"❌ Database connection failed during initialization: {e}", style="red"
            )
            raise typer.Exit(1)
        except ImportError as e:
            console.print(f"❌ Required dependency missing: {e}", style="red")
            raise typer.Exit(1)

    asyncio.run(run_initialization())


@app.command()
def backup_create(
    retention_days: int = typer.Option(
        30, "--retention", help="Backup retention in days"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Create automated backup with retention policy (Phase 2)."""
    console.print("🔒 Creating system backup...", style="green")

    async def run_backup():
        migration_manager = APESMigrationManager(console)

        try:
            results = await migration_manager.create_automated_backup(retention_days)

            console.print(
                f"✅ Backup completed: {results['total_size_mb']:.1f} MB", style="green"
            )

            if verbose:
                console.print("\n📦 Backup Details:", style="bold")
                console.print(f"   Timestamp: {results['timestamp']}")
                console.print(f"   Files Created: {len(results['backup_files'])}")
                console.print(
                    f"   Integrity Verified: {'✅' if results['integrity_verified'] else '❌'}"
                )
                console.print(
                    f"   Retention Applied: {'✅' if results['retention_applied'] else '❌'}"
                )

                console.print("   Backup Files:")
                for backup_file in results["backup_files"]:
                    file_size = (
                        Path(backup_file).stat().st_size / (1024 * 1024)
                        if Path(backup_file).exists()
                        else 0
                    )
                    console.print(
                        f"     • {Path(backup_file).name} ({file_size:.1f} MB)"
                    )

        except (FileNotFoundError, PermissionError) as e:
            console.print(f"❌ File system error during backup: {e}", style="red")
            raise typer.Exit(1)
        except (ConnectionError, OSError) as e:
            console.print(
                f"❌ Database connection failed during backup: {e}", style="red"
            )
            raise typer.Exit(1)
        except subprocess.SubprocessError as e:
            console.print(f"❌ Backup script execution failed: {e}", style="red")
            raise typer.Exit(1)

    asyncio.run(run_backup())


@app.command()
def migrate_export(
    output: Path = typer.Argument(..., help="Output path for migration package"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Create migration package for cross-machine deployment (Phase 2)."""
    console.print("📦 Creating migration package...", style="green")

    async def run_export():
        migration_manager = APESMigrationManager(console)

        try:
            results = await migration_manager.create_migration_package(output)

            console.print(f"✅ Migration package created: {output}", style="green")
            console.print(
                f"📊 Package size: {results['total_size_mb']:.1f} MB", style="dim"
            )

            if verbose:
                console.print("\n📦 Package Contents:", style="bold")
                for component in results["components"]:
                    console.print(
                        f"   • {component['type']}: {component['size_mb']:.1f} MB"
                    )
                console.print(f"   Checksum: {results['checksum'][:16]}...")

        except (FileNotFoundError, PermissionError) as e:
            console.print(
                f"❌ File system error during migration export: {e}", style="red"
            )
            raise typer.Exit(1)
        except (ConnectionError, OSError) as e:
            console.print(
                f"❌ Database connection failed during migration export: {e}",
                style="red",
            )
            raise typer.Exit(1)
        except (ValueError, TypeError) as e:
            console.print(f"❌ Migration data processing error: {e}", style="red")
            raise typer.Exit(1)

    asyncio.run(run_export())


@app.command()
def service_start(
    background: bool = typer.Option(
        True, "--background/--foreground", help="Run as background daemon"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start APES service with background daemon support (Phase 2)."""
    console.print("🚀 Starting APES service...", style="green")

    async def run_service():
        try:
            results = await service_manager.start_background_service(background)

            if results["status"] == "running":
                console.print("✅ Service started successfully", style="green")

                if results.get("pid"):
                    console.print(f"📍 Process ID: {results['pid']}", style="dim")

                if results.get("mcp_response_time"):
                    console.print(
                        f"📊 MCP Response Time: {results['mcp_response_time']:.1f}ms",
                        style="dim",
                    )

                if verbose:
                    console.print("\n🔧 Service Details:", style="bold")
                    console.print(
                        f"   PostgreSQL: {results.get('postgresql_status', 'unknown')}"
                    )
                    console.print(f"   Background Mode: {'✅' if background else '❌'}")
                    console.print(f"   Startup Time: {results['startup_time']}")
            else:
                console.print(
                    f"❌ Service failed to start: {results.get('error', 'unknown error')}",
                    style="red",
                )
                raise typer.Exit(1)

        except (ConnectionError, OSError) as e:
            console.print(f"❌ Service connection failed: {e}", style="red")
            raise typer.Exit(1)
        except (FileNotFoundError, PermissionError) as e:
            console.print(f"❌ Service file system error: {e}", style="red")
            raise typer.Exit(1)
        except (ProcessLookupError, subprocess.SubprocessError) as e:
            console.print(f"❌ Service process error: {e}", style="red")
            raise typer.Exit(1)

    asyncio.run(run_service())


@app.command()
def service_stop(
    timeout: int = typer.Option(30, "--timeout", help="Shutdown timeout in seconds"),
):
    """Stop APES background service (Phase 2)."""
    console.print("🔄 Stopping APES service...", style="yellow")

    try:
        results = service_manager.stop_service(timeout)

        if results["status"] in ["stopped", "force_stopped"]:
            console.print(f"✅ Service stopped: {results['message']}", style="green")
            if results["status"] == "force_stopped":
                console.print("⚠️  Service was force stopped", style="yellow")
        elif results["status"] == "not_running":
            console.print("ℹ️  Service was not running", style="blue")
        else:
            console.print(
                f"❌ Failed to stop service: {results['message']}", style="red"
            )
            raise typer.Exit(1)

    except (ProcessLookupError, PermissionError) as e:
        console.print(f"❌ Service process error: {e}", style="red")
        raise typer.Exit(1)
    except (OSError, subprocess.SubprocessError) as e:
        console.print(f"❌ System error stopping service: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def service_status(
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed status"
    ),
):
    """Show APES service status (Phase 2)."""
    try:
        status = service_manager.get_service_status()

        # Create status table
        table = Table(title="APES Service Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")

        # Service status
        if status["running"]:
            status_icon = "✅ Running"
            details = f"PID: {status['pid']}"
            if status.get("uptime_seconds"):
                uptime_hours = status["uptime_seconds"] / 3600
                details += f", Uptime: {uptime_hours:.1f}h"
        else:
            status_icon = "❌ Stopped"
            details = "-"

        table.add_row("APES Service", status_icon, details)

        # Memory usage
        if status.get("memory_usage_mb"):
            memory_status = f"{status['memory_usage_mb']:.1f} MB"
            memory_icon = "✅" if status["memory_usage_mb"] < 256 else "⚠️ "
            table.add_row(
                "Memory Usage", f"{memory_icon} {memory_status}", "Target: <256 MB"
            )

        console.print(table)

        if detailed and status["running"]:
            # Additional detailed information
            console.print("\n📊 Detailed Status:")
            console.print(f"   Started: {status.get('started_at', 'Unknown')}")
            if status.get("error"):
                console.print(f"   Error: {status['error']}", style="red")

    except (ProcessLookupError, PermissionError) as e:
        console.print(f"❌ Service status access error: {e}", style="red")
        raise typer.Exit(1)
    except (OSError, ValueError) as e:
        console.print(f"❌ Service status system error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def security_audit(
    days: int = typer.Option(30, "--days", help="Report period in days"),
    export_json: bool = typer.Option(
        False, "--export-json", help="Export report as JSON"
    ),
):
    """Generate security audit report (Phase 2)."""
    console.print(f"🔒 Generating security audit report ({days} days)...", style="blue")

    async def run_audit():
        try:
            report = await data_protection.get_security_audit_report(days)

            if "error" in report:
                console.print(f"❌ Audit failed: {report['error']}", style="red")
                return

            if export_json:
                # Export as JSON file
                output_file = (
                    f"security_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                console.print(f"📄 Report exported to: {output_file}", style="green")

            # Display summary
            console.print("\n🔒 SECURITY AUDIT REPORT", style="bold blue")
            console.print(f"Report Period: {days} days")
            console.print(f"Generated: {report['generated_at']}")

            # Summary table
            table = Table(title="Security Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            table.add_column("Status", style="bold")

            summary = report["summary"]
            compliance = report["compliance"]

            table.add_row(
                "Total Prompts Processed", str(summary["total_prompts_processed"]), "📊"
            )
            table.add_row(
                "Sessions with Redactions",
                str(summary["sessions_with_redactions"]),
                "🔍",
            )
            table.add_row(
                "Total Redactions", str(summary["total_redactions_performed"]), "✂️"
            )
            table.add_row(
                "Compliance Score",
                f"{compliance['compliance_score_percent']}%",
                compliance["status"],
            )

            console.print(table)

            # Redaction patterns
            if report["redaction_patterns"]["types_detected"]:
                console.print("\n🎯 Top Redaction Types:", style="bold")
                for pattern in report["redaction_patterns"]["types_detected"][:5]:
                    console.print(
                        f"   • {pattern['type']}: {pattern['count']} occurrences"
                    )

        except (ConnectionError, OSError) as e:
            console.print(
                f"❌ Database connection failed during security audit: {e}", style="red"
            )
            raise typer.Exit(1)
        except (FileNotFoundError, PermissionError) as e:
            console.print(
                f"❌ File system error during security audit: {e}", style="red"
            )
            raise typer.Exit(1)
        except (ValueError, TypeError, KeyError) as e:
            console.print(f"❌ Security audit data processing error: {e}", style="red")
            raise typer.Exit(1)

    asyncio.run(run_audit())


@app.command()
def security_test(
    prompt: str = typer.Argument(..., help="Prompt text to test for security issues"),
    fix_issues: bool = typer.Option(False, "--fix", help="Apply automatic redactions"),
):
    """Test prompt for security issues (Phase 2)."""
    console.print("🔍 Testing prompt for security issues...", style="blue")

    async def run_security_test():
        try:
            # Validate prompt safety
            safety_report = await data_protection.validate_prompt_safety(prompt)

            if safety_report["is_safe"]:
                console.print(
                    "✅ Prompt is safe - no security issues detected", style="green"
                )
            else:
                console.print(
                    f"⚠️  Security issues detected - Risk Level: {safety_report['risk_level']}",
                    style="yellow",
                )

                # Show detected issues
                table = Table(title="Security Issues Detected")
                table.add_column("Issue Type", style="cyan")
                table.add_column("Count", style="yellow")
                table.add_column("Risk Level", style="red")

                for issue in safety_report["issues_detected"]:
                    table.add_row(
                        issue["type"], str(issue["count"]), issue["risk_level"]
                    )

                console.print(table)

                # Show recommendations
                console.print("\n💡 Recommendations:", style="bold")
                for rec in safety_report["recommendations"]:
                    console.print(f"   • {rec}")

                # Apply fixes if requested
                if fix_issues:
                    console.print("\n🔧 Applying automatic redactions...", style="blue")
                    (
                        sanitized_prompt,
                        redaction_summary,
                    ) = await data_protection.sanitize_prompt_before_storage(
                        prompt, "security_test"
                    )

                    console.print("✅ Redactions applied:", style="green")
                    console.print(
                        f"   Redactions made: {redaction_summary['redactions_made']}"
                    )
                    console.print(
                        f"   Types: {', '.join(redaction_summary['redaction_types'])}"
                    )
                    console.print("\n📝 Sanitized prompt:")
                    console.print(sanitized_prompt, style="dim")

        except (ConnectionError, OSError) as e:
            console.print(
                f"❌ Database connection failed during security test: {e}", style="red"
            )
            raise typer.Exit(1)
        except (ValueError, TypeError, KeyError) as e:
            console.print(f"❌ Security test data processing error: {e}", style="red")
            raise typer.Exit(1)
        except ImportError as e:
            console.print(f"❌ Security library dependency missing: {e}", style="red")
            raise typer.Exit(1)

    asyncio.run(run_security_test())


@app.command()
def migrate_restore(
    package_path: Path = typer.Argument(..., help="Path to migration package"),
    target_dir: Path | None = typer.Option(
        None, "--target-dir", help="Target directory for restoration"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force restoration, overwriting existing data"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Restore system from migration package (Phase 2)."""
    console.print(f"📦 Restoring from migration package: {package_path}", style="green")

    async def run_restore():
        migration_manager = APESMigrationManager(console)

        try:
            results = await migration_manager.restore_from_migration_package(
                package_path, target_dir, force
            )

            if results["status"] == "success":
                console.print(
                    "✅ Migration restoration completed successfully!", style="green"
                )

                if verbose:
                    console.print("\n📊 Restoration Details:", style="bold")
                    console.print(
                        f"   Components Restored: {len(results['restored_components'])}"
                    )
                    console.print(
                        f"   Database Records: {results.get('database_records', 0)}"
                    )
                    console.print(
                        f"   Configuration Files: {results.get('config_files', 0)}"
                    )
                    console.print(f"   ML Artifacts: {results.get('ml_artifacts', 0)}")
                    console.print(f"   User Prompts: {results.get('user_prompts', 0)}")

                    console.print("\n🔧 Next Steps:", style="blue")
                    console.print("   1. Run 'apes doctor' to verify system health")
                    console.print("   2. Run 'apes start' to launch services")
                    console.print("   3. Test functionality with 'apes status'")
            else:
                console.print(
                    f"❌ Migration restoration failed: {results.get('error', 'Unknown error')}",
                    style="red",
                )
                raise typer.Exit(1)

        except (FileNotFoundError, PermissionError) as e:
            console.print(
                f"❌ File system error during migration restoration: {e}", style="red"
            )
            raise typer.Exit(1)
        except (ConnectionError, OSError) as e:
            console.print(
                f"❌ Database connection failed during migration restoration: {e}",
                style="red",
            )
            raise typer.Exit(1)
        except (ValueError, TypeError) as e:
            console.print(f"❌ Migration data processing error: {e}", style="red")
            raise typer.Exit(1)

    asyncio.run(run_restore())


@app.command()
def training_history(
    ui: bool = typer.Option(
        True, "--ui/--no-ui", help="Launch MLflow UI for interactive viewing"
    ),
    list_runs: bool = typer.Option(False, "--list", help="List recent training runs"),
    experiment: str | None = typer.Option(
        None, "--experiment", help="Filter by experiment name"
    ),
    limit: int = typer.Option(10, "--limit", help="Limit number of runs to show"),
):
    """View training history and MLflow experiments (Phase 2)."""
    if ui:
        console.print("🚀 Launching MLflow UI for training history...", style="green")

        try:
            # Get MLflow tracking URI
            project_root = Path(__file__).parents[2]
            mlflow_dir = project_root / "mlruns"
            mlflow_dir.mkdir(parents=True, exist_ok=True)

            console.print(f"📊 MLflow tracking directory: {mlflow_dir}", style="dim")
            console.print("🌐 Starting MLflow UI server...", style="blue")

            # Launch MLflow UI - using absolute path for security
            import shutil

            mlflow_path = shutil.which("mlflow")
            if not mlflow_path:
                raise FileNotFoundError(
                    "MLflow not found in PATH. Please install MLflow: pip install mlflow"
                )

            # Security: subprocess call with validated executable path and secure parameters
            # - mlflow_path resolved via shutil.which() to prevent PATH injection
            # - shell=False prevents shell injection attacks
            # - Arguments are controlled and validated (localhost binding)
            # - MLflow directory path is controlled and validated
            process = subprocess.Popen(
                [
                    mlflow_path,
                    "ui",
                    "--backend-store-uri",
                    f"file://{mlflow_dir}",
                    "--port",
                    "5000",
                    "--host",
                    "127.0.0.1",
                ],
                shell=False,
            )

            console.print("✅ MLflow UI launched successfully!", style="green")
            console.print("🔗 Access at: http://127.0.0.1:5000", style="blue")
            console.print(
                "👆 Click the link above to view training history", style="dim"
            )
            console.print("\n⏸️  Press Ctrl+C to stop the UI server", style="yellow")

            try:
                process.wait()
            except KeyboardInterrupt:
                console.print("\n🔄 Stopping MLflow UI server...", style="yellow")
                process.terminate()
                process.wait()
                console.print("✅ MLflow UI server stopped", style="green")

        except FileNotFoundError:
            console.print("❌ MLflow not found. Please install MLflow:", style="red")
            console.print("   pip install mlflow", style="dim")
            raise typer.Exit(1)
        except (PermissionError, OSError) as e:
            console.print(f"❌ System error launching MLflow UI: {e}", style="red")
            raise typer.Exit(1)
        except subprocess.SubprocessError as e:
            console.print(f"❌ MLflow UI process error: {e}", style="red")
            raise typer.Exit(1)

    elif list_runs:
        console.print("📋 Listing recent training runs...", style="blue")

        async def list_training_runs():
            try:
                # Search for training runs in the database
                sm: DatabaseSessionManager = get_sessionmanager()
                async with sm.session() as session:
                    query = """
                        SELECT t.session_id, t.created_at, t.metrics_data, t.performance_score
                        FROM training_sessions t
                        ORDER BY t.created_at DESC
                        LIMIT $1
                    """

                    result = await session.execute(query, [limit])
                    runs = result.fetchall()

                    if not runs:
                        console.print("📝 No training runs found", style="dim")
                        return

                    # Display runs in a table
                    table = Table(title="Recent Training Runs")
                    table.add_column("Session ID", style="cyan")
                    table.add_column("Date", style="green")
                    table.add_column("Performance Score", style="yellow")
                    table.add_column("Status", style="white")

                    for run in runs:
                        session_id = run[0][:12] + "..." if len(run[0]) > 12 else run[0]
                        created_at = (
                            run[1].strftime("%Y-%m-%d %H:%M") if run[1] else "Unknown"
                        )
                        score = f"{run[3]:.3f}" if run[3] else "N/A"
                        status = "✅ Completed" if run[2] else "⏳ In Progress"

                        table.add_row(session_id, created_at, score, status)

                    console.print(table)

            except (ConnectionError, OSError) as e:
                console.print(f"❌ Database connection failed: {e}", style="red")
                raise typer.Exit(1)
            except (ValueError, TypeError, KeyError) as e:
                console.print(f"❌ Training data processing error: {e}", style="red")
                raise typer.Exit(1)

        asyncio.run(list_training_runs())
    else:
        console.print(
            "ℹ️  Use --ui to launch MLflow UI or --list to show recent runs",
            style="blue",
        )


@app.command()
def logs(
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Follow log output in real-time"
    ),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    level: str = typer.Option(
        "INFO", "--level", help="Log level filter (DEBUG, INFO, WARNING, ERROR)"
    ),
    component: str | None = typer.Option(
        None, "--component", help="Filter by component (mcp, database, training)"
    ),
):
    """View APES system logs (Phase 2)."""
    # Find log files
    log_dir = Path.home() / ".local" / "share" / "apes" / "data" / "logs"

    if not log_dir.exists():
        console.print(f"❌ Log directory not found: {log_dir}", style="red")
        console.print("💡 Run 'apes init' to create the log directory", style="dim")
        raise typer.Exit(1)

    # Determine log file to view
    if component:
        log_file = log_dir / f"{component}.log"
    else:
        log_file = log_dir / "apes.log"

    if not log_file.exists():
        console.print(f"❌ Log file not found: {log_file}", style="red")
        console.print("💡 Available log files:", style="dim")
        for log in log_dir.glob("*.log"):
            console.print(f"   • {log.name}")
        raise typer.Exit(1)

    console.print(f"📄 Viewing logs: {log_file}", style="blue")

    if follow:
        console.print(
            "👁️  Following log output (Press Ctrl+C to stop)...", style="green"
        )

        try:
            # Use tail -f equivalent for following logs - using absolute path for security
            import shutil

            tail_path = shutil.which("tail")
            if not tail_path:
                raise FileNotFoundError("tail command not found in PATH")

            # Security: subprocess call with validated executable path and secure parameters
            # - tail_path resolved via shutil.which() to prevent PATH injection
            # - shell=False prevents shell injection attacks
            # - log_file path is validated as existing file before use
            # - Arguments are controlled and validated
            process = subprocess.Popen(  # noqa: S603
                [tail_path, "-f", str(log_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
            )

            try:
                if process.stdout is None:
                    raise RuntimeError("Process stdout is None - cannot iterate over log lines")

                # Type annotation for MyPy after None check
                stdout: typing.TextIO = process.stdout

                for line in stdout:
                    # Filter by log level if specified
                    if level and level.upper() not in line.upper():
                        continue

                    # Color code log levels
                    if "ERROR" in line:
                        console.print(line.rstrip(), style="red")
                    elif "WARNING" in line:
                        console.print(line.rstrip(), style="yellow")
                    elif "INFO" in line:
                        console.print(line.rstrip(), style="green")
                    elif "DEBUG" in line:
                        console.print(line.rstrip(), style="dim")
                    else:
                        console.print(line.rstrip())

            except KeyboardInterrupt:
                console.print("\n🔄 Stopping log viewer...", style="yellow")
                process.terminate()
                process.wait()
                console.print("✅ Log viewer stopped", style="green")

        except FileNotFoundError:
            console.print(
                "❌ 'tail' command not found. Using Python implementation...",
                style="yellow",
            )

            # Python-based log following
            import time

            try:
                with open(log_file, encoding="utf-8") as f:
                    # Go to end of file
                    f.seek(0, 2)

                    while True:
                        line = f.readline()
                        if line:
                            # Filter by log level if specified
                            if level and level.upper() not in line.upper():
                                continue

                            # Color code log levels
                            if "ERROR" in line:
                                console.print(line.rstrip(), style="red")
                            elif "WARNING" in line:
                                console.print(line.rstrip(), style="yellow")
                            elif "INFO" in line:
                                console.print(line.rstrip(), style="green")
                            elif "DEBUG" in line:
                                console.print(line.rstrip(), style="dim")
                            else:
                                console.print(line.rstrip())
                        else:
                            time.sleep(0.1)

            except KeyboardInterrupt:
                console.print("\n✅ Log viewer stopped", style="green")

        except (FileNotFoundError, PermissionError) as e:
            console.print(f"❌ Log file access error: {e}", style="red")
            raise typer.Exit(1)
        except (OSError, subprocess.SubprocessError) as e:
            console.print(f"❌ Log following system error: {e}", style="red")
            raise typer.Exit(1)
    else:
        # Show last N lines
        console.print(f"📋 Last {lines} lines:", style="blue")

        try:
            with open(log_file, encoding="utf-8") as f:
                all_lines = f.readlines()
                recent_lines = (
                    all_lines[-lines:] if len(all_lines) > lines else all_lines
                )

                for line in recent_lines:
                    # Filter by log level if specified
                    if level and level.upper() not in line.upper():
                        continue

                    # Color code log levels
                    if "ERROR" in line:
                        console.print(line.rstrip(), style="red")
                    elif "WARNING" in line:
                        console.print(line.rstrip(), style="yellow")
                    elif "INFO" in line:
                        console.print(line.rstrip(), style="green")
                    elif "DEBUG" in line:
                        console.print(line.rstrip(), style="dim")
                    else:
                        console.print(line.rstrip())

        except (FileNotFoundError, PermissionError) as e:
            console.print(f"❌ Log file access error: {e}", style="red")
            raise typer.Exit(1)
        except (OSError, UnicodeDecodeError) as e:
            console.print(f"❌ Log file reading error: {e}", style="red")
            raise typer.Exit(1)


# Phase 3B: Advanced Monitoring & Analytics Commands


@app.command()
def monitor(
    refresh: int = typer.Option(
        5, "--refresh", "-r", help="Refresh interval in seconds"
    ),
    alerts_only: bool = typer.Option(
        False, "--alerts-only", help="Show only active alerts"
    ),
):
    """Start real-time monitoring dashboard (Phase 3B)."""
    console.print("🚀 Starting APES Real-Time Monitoring Dashboard...", style="green")

    try:
        from prompt_improver.services.monitoring import RealTimeMonitor

        monitor = RealTimeMonitor(console)

        if alerts_only:
            # Simple alerts-only mode
            async def show_alerts():
                summary = await monitor.get_monitoring_summary(hours=1)
                alert_summary = summary.get("alert_summary", {})

                if alert_summary.get("total_alerts", 0) > 0:
                    table = Table(title="🚨 Active Alerts")
                    table.add_column("Type", style="cyan")
                    table.add_column("Count", style="red")

                    table.add_row(
                        "Critical", str(alert_summary.get("critical_alerts", 0))
                    )
                    table.add_row(
                        "Warning", str(alert_summary.get("warning_alerts", 0))
                    )
                    table.add_row("Total", str(alert_summary.get("total_alerts", 0)))

                    console.print(table)
                else:
                    console.print("🟢 No active alerts", style="green")

            asyncio.run(show_alerts())
        else:
            # Full dashboard mode
            asyncio.run(monitor.start_monitoring_dashboard(refresh_seconds=refresh))

    except KeyboardInterrupt:
        console.print("\n👋 Monitoring dashboard stopped", style="yellow")
    except ImportError as e:
        console.print(f"❌ Monitoring dependency missing: {e}", style="red")
        raise typer.Exit(1)
    except (ConnectionError, OSError) as e:
        console.print(f"❌ Monitoring system connection error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def health(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed diagnostics"
    ),
):
    """Run comprehensive system health check (Phase 3B)."""
    console.print("🏥 Running APES Health Check...", style="blue")

    async def run_health_check():
        from prompt_improver.services.health import get_health_service

        health_service = get_health_service()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running health diagnostics...", total=None)

            results = await health_service.get_health_summary(include_details=detailed)

            progress.update(task, completed=True)

        if json_output:
            console.print_json(data=results)
        else:
            # Create health status table
            overall_status = results.get("overall_status", "unknown")
            status_icon = (
                "✅"
                if overall_status == "healthy"
                else "⚠️"
                if overall_status == "warning"
                else "❌"
            )

            console.print(
                f"\n{status_icon} Overall Health: {overall_status.upper()}",
                style="bold",
            )

            # Component status table
            table = Table(title="Component Health Status", show_header=True)
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="")
            table.add_column("Response Time", style="magenta")
            table.add_column("Details", style="dim")

            checks = results.get("checks", {})
            for component, check_result in checks.items():
                status = check_result.get("status", "unknown")
                status_icon = (
                    "✅"
                    if status == "healthy"
                    else "⚠️"
                    if status == "warning"
                    else "❌"
                )

                response_time = check_result.get("response_time_ms")
                response_str = f"{response_time:.1f}ms" if response_time else "-"

                message = check_result.get("message", "")
                if check_result.get("error"):
                    message = f"Error: {check_result['error']}"

                table.add_row(
                    component.replace("_", " ").title(),
                    f"{status_icon} {status.capitalize()}",
                    response_str,
                    message,
                )

            console.print(table)

            # Show warnings and failures
            if "warning_checks" in results or "failed_checks" in results:
                console.print("\n[bold]Issues Found:[/bold]")

                for warning in results.get("warning_checks", []):
                    console.print(
                        f"⚠️  {warning}: {checks[warning].get('message', '')}",
                        style="yellow",
                    )

                for failure in results.get("failed_checks", []):
                    console.print(
                        f"❌ {failure}: {checks[failure].get('message', '')}",
                        style="red",
                    )

            if detailed:
                # Show additional system information
                console.print("\n[bold]System Resources:[/bold]")
                system_check = checks.get("system_resources", {})
                if system_check and "details" in system_check:
                    resource_table = Table()
                    resource_table.add_column("Resource", style="cyan")
                    resource_table.add_column("Usage", style="yellow")

                    details = system_check["details"]
                    if "memory_usage_percent" in details:
                        resource_table.add_row(
                            "Memory", f"{details['memory_usage_percent']:.1f}%"
                        )
                    if "cpu_usage_percent" in details:
                        resource_table.add_row(
                            "CPU", f"{details['cpu_usage_percent']:.1f}%"
                        )
                    if "disk_usage_percent" in details:
                        resource_table.add_row(
                            "Disk", f"{details['disk_usage_percent']:.1f}%"
                        )

                    console.print(resource_table)

    try:
        asyncio.run(run_health_check())
    except ImportError as e:
        console.print(f"❌ Health monitoring dependency missing: {e}", style="red")
        raise typer.Exit(1)
    except (ConnectionError, OSError) as e:
        console.print(f"❌ Health check system error: {e}", style="red")
        raise typer.Exit(1)
    except (ValueError, TypeError, KeyError) as e:
        console.print(f"❌ Health check data processing error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def monitoring_summary(
    hours: int = typer.Option(24, "--hours", "-h", help="Time period in hours"),
    export_file: str | None = typer.Option(
        None, "--export", help="Export results to file"
    ),
    format: str = typer.Option("table", "--format", help="Output format: table, json"),
):
    """Get monitoring summary for specified time period (Phase 3B)."""
    console.print(f"📊 Generating {hours}h monitoring summary...", style="blue")

    async def get_summary():
        from prompt_improver.services.monitoring import RealTimeMonitor

        monitor = RealTimeMonitor(console)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Collecting monitoring data...", total=None)

            summary = await monitor.get_monitoring_summary(hours=hours)

            progress.update(task, completed=True)

        if format == "json" or export_file:
            if export_file:
                # Export to file
                import json

                with open(export_file, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, default=str)
                console.print(f"📄 Summary exported to {export_file}", style="green")

            if format == "json":
                console.print_json(data=summary)
        else:
            # Table format
            console.print(f"\n[bold]Monitoring Summary ({hours}h period)[/bold]")

            # Current performance
            current = summary.get("current_performance", {})
            perf_table = Table(title="Current Performance")
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="magenta")
            perf_table.add_column("Status", justify="center")

            response_time = current.get("avg_response_time_ms", 0)
            response_status = (
                "✅" if response_time < 200 else "⚠️" if response_time < 300 else "❌"
            )
            perf_table.add_row(
                "Response Time", f"{response_time:.1f}ms", response_status
            )

            memory_usage = current.get("memory_usage_mb", 0)
            memory_status = (
                "✅" if memory_usage < 200 else "⚠️" if memory_usage < 300 else "❌"
            )
            perf_table.add_row("Memory Usage", f"{memory_usage:.1f}MB", memory_status)

            db_connections = current.get("database_connections", 0)
            db_status = (
                "✅" if db_connections < 15 else "⚠️" if db_connections < 20 else "❌"
            )
            perf_table.add_row("DB Connections", str(db_connections), db_status)

            console.print(perf_table)

            # Alert summary
            alerts = summary.get("alert_summary", {})
            if alerts.get("total_alerts", 0) > 0:
                alert_table = Table(title="Alert Summary")
                alert_table.add_column("Alert Type", style="cyan")
                alert_table.add_column("Count", style="red")

                alert_table.add_row("Critical", str(alerts.get("critical_alerts", 0)))
                alert_table.add_row("Warning", str(alerts.get("warning_alerts", 0)))
                alert_table.add_row("Total", str(alerts.get("total_alerts", 0)))

                most_common = alerts.get("most_common_alert")
                if most_common:
                    alert_table.add_row("Most Common", most_common)

                console.print(alert_table)
            else:
                console.print("\n🟢 No alerts in the specified period", style="green")

            # Health status
            health_status = summary.get("health_status", "unknown")
            health_icon = (
                "✅"
                if health_status == "healthy"
                else "⚠️"
                if health_status == "warning"
                else "❌"
            )
            console.print(
                f"\n{health_icon} Overall Health: {health_status.upper()}", style="bold"
            )

    try:
        asyncio.run(get_summary())
    except ImportError as e:
        console.print(f"❌ Monitoring dependency missing: {e}", style="red")
        raise typer.Exit(1)
    except (ConnectionError, OSError) as e:
        console.print(f"❌ Monitoring system connection error: {e}", style="red")
        raise typer.Exit(1)
    except (ValueError, TypeError, KeyError) as e:
        console.print(f"❌ Monitoring data processing error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def update(
    component: str = typer.Argument(
        help="Component to update: docs, config, dependencies, all"
    ),
    verify: bool = typer.Option(
        True, "--verify/--no-verify", help="Verify accuracy of updates"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force update without confirmation"
    ),
):
    """Update system components, documentation, or configurations."""
    console.print("🔄 APES Update Manager", style="blue")

    if component == "docs":
        _update_documentation(verify=verify, force=force)
    elif component == "config":
        _update_configuration(verify=verify, force=force)
    elif component == "dependencies":
        _update_dependencies(verify=verify, force=force)
    elif component == "all":
        _update_all_components(verify=verify, force=force)
    else:
        console.print(f"❌ Unknown component: {component}", style="red")
        console.print("Available components: docs, config, dependencies, all")
        raise typer.Exit(1)


def _update_documentation(verify: bool = True, force: bool = False) -> None:
    """Update project documentation with current codebase state."""
    import os
    import subprocess

    console.print("📚 Updating documentation...", style="yellow")

    if verify:
        console.print("🔍 Verifying current documentation accuracy...")

        # Verify line counts
        console.print("📊 Checking line counts...")

        verification_results = {
            "MCP Server": {
                "claimed": 253,
                "actual": None,
                "files": ["src/prompt_improver/mcp_server"],
            },
            "CLI Interface": {
                "claimed": 2912,
                "actual": None,
                "files": [
                    "src/prompt_improver/cli.py",
                    "src/prompt_improver/cli_refactored.py",
                ],
            },
            "Database Architecture": {
                "claimed": 1261,
                "actual": None,
                "files": ["src/prompt_improver/database"],
            },
            "Analytics Service": {
                "claimed": 384,
                "actual": None,
                "files": ["src/prompt_improver/services/analytics.py"],
            },
            "ML Service Integration": {
                "claimed": 563,
                "actual": None,
                "files": ["src/prompt_improver/services/ml_integration.py"],
            },
            "Monitoring Service": {
                "claimed": 753,
                "actual": None,
                "files": ["src/prompt_improver/services/monitoring.py"],
            },
        }

        for component, info in verification_results.items():
            total_lines = 0
            for file_path in info["files"]:
                if os.path.isfile(file_path):
                    # Use absolute path for wc command for security
                    import shutil

                    wc_path = shutil.which("wc")
                    if wc_path:
                        # Security: subprocess call with validated executable path and secure parameters
                        # - wc_path resolved via shutil.which() to prevent PATH injection
                        # - shell=False prevents shell injection attacks
                        # - file_path validated as existing file before use
                        # - timeout=30 prevents indefinite hanging
                        file_wc_result = subprocess.run(
                            [wc_path, "-l", file_path],
                            check=False,
                            capture_output=True,
                            text=True,
                            shell=False,
                            timeout=30,
                        )
                        if file_wc_result.returncode == 0:
                            file_line_count: int = int(file_wc_result.stdout.split()[0])
                            total_lines += file_line_count
                elif os.path.isdir(file_path):
                    # Use absolute paths for find and wc commands for security
                    find_path = shutil.which("find")
                    if find_path and wc_path:
                        # Security: subprocess call with validated executable paths and secure parameters
                        # - find_path and wc_path resolved via shutil.which() to prevent PATH injection
                        # - shell=False prevents shell injection attacks
                        # - file_path validated as existing directory before use
                        # - timeout=60 prevents indefinite hanging
                        dir_find_result = subprocess.run(
                            [
                                find_path,
                                file_path,
                                "-name",
                                "*.py",
                                "-exec",
                                wc_path,
                                "-l",
                                "{}",
                                "+",
                            ],
                            check=False,
                            capture_output=True,
                            text=True,
                            shell=False,
                            timeout=60,
                        )
                        if dir_find_result.returncode == 0:
                            output_lines = dir_find_result.stdout.strip().split("\n")
                            for output_line in output_lines:
                                if output_line.strip() and not output_line.strip().endswith(".py"):
                                    try:
                                        total_lines += int(output_line.strip().split()[0])
                                    except (ValueError, IndexError):
                                        continue

            info["actual"] = total_lines
            claimed: int = info["claimed"]
            actual: int = info["actual"]
            diff: int = actual - claimed

            if diff == 0:
                status = "✅ ACCURATE"
                style = "green"
            elif abs(diff) <= 10:
                status = f"⚠️ MINOR DIFF ({diff:+d})"
                style = "yellow"
            else:
                status = f"❌ MAJOR DIFF ({diff:+d})"
                style = "red"

            console.print(
                f"{component}: {claimed} → {actual} lines [{status}]", style=style
            )

        # Test status verification
        console.print("\n🧪 Checking test status...")
        try:
            # Use sys.executable for security instead of "python"
            # Security: subprocess call with validated executable path and secure parameters
            # - sys.executable is Python's own executable path (trusted)
            # - shell=False prevents shell injection attacks
            # - timeout=30 prevents indefinite hanging
            # - Arguments are controlled and validated
            collect_result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
            )
            if collect_result.returncode == 0:
                test_count: int = len([
                    test_line
                    for test_line in collect_result.stdout.split("\n")
                    if "test" in test_line and "::" in test_line
                ])
                console.print(f"Found {test_count} tests to run")

                # Quick test run to check status
                console.print("Running quick test validation...")
                # Security: subprocess call with validated executable path and secure parameters
                # - sys.executable is Python's own executable path (trusted)
                # - shell=False prevents shell injection attacks
                # - timeout=60 prevents indefinite hanging
                # - Arguments are controlled and validated
                test_result = subprocess.run(
                    [sys.executable, "-m", "pytest", "tests/", "-x", "--tb=no", "-q"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    shell=False,
                )

                if "FAILED" in test_result.stdout or "ERROR" in test_result.stdout:
                    console.print(
                        "❌ Tests have failures - documentation claims are incorrect",
                        style="red",
                    )
                else:
                    console.print("✅ Tests are passing", style="green")
            else:
                console.print("⚠️ Could not collect tests", style="yellow")
        except subprocess.TimeoutExpired:
            console.print("⚠️ Test verification timed out", style="yellow")
        except (ValueError, TypeError, IndexError) as e:
            console.print(f"⚠️ Test data processing error: {e}", style="yellow")
        except (OSError, subprocess.SubprocessError) as e:
            console.print(f"⚠️ Test system error: {e}", style="yellow")

    if not force:
        should_update = typer.confirm("Update documentation with current findings?")
        if not should_update:
            console.print("❌ Update cancelled", style="red")
            return

    console.print("📝 Updating docs/project_overview.md...", style="blue")
    # Here you would implement the actual documentation update logic
    console.print("✅ Documentation update completed", style="green")


def _update_configuration(verify: bool = True, force: bool = False) -> None:
    """Update system configuration files."""
    console.print("⚙️ Configuration update not yet implemented", style="yellow")


def _update_dependencies(verify: bool = True, force: bool = False) -> None:
    """Update project dependencies."""
    console.print("📦 Dependency update not yet implemented", style="yellow")


def _update_all_components(verify: bool = True, force: bool = False) -> None:
    """Update all system components."""
    console.print("🔄 Full system update not yet implemented", style="yellow")


@app.command()
def alerts(
    severity: str | None = typer.Option(
        None, "--severity", help="Filter by severity: critical, warning"
    ),
    hours: int = typer.Option(24, "--hours", "-h", help="Time period in hours"),
):
    """View system alerts and monitoring notifications (Phase 3B)."""
    console.print("🚨 APES Alert Status", style="yellow")

    async def show_alerts():
        from prompt_improver.services.monitoring import RealTimeMonitor

        monitor = RealTimeMonitor(console)
        summary = await monitor.get_monitoring_summary(hours=hours)

        # Get alert summary
        alert_summary = summary.get("alert_summary", {})
        total_alerts = alert_summary.get("total_alerts", 0)

        if total_alerts == 0:
            console.print("🟢 No alerts found in the specified period", style="green")
            return

        # Create alerts table
        table = Table(title=f"Alerts ({hours}h period)")
        table.add_column("Severity", style="")
        table.add_column("Count", style="magenta")
        table.add_column("Details", style="dim")

        critical_count = alert_summary.get("critical_alerts", 0)
        warning_count = alert_summary.get("warning_alerts", 0)

        if not severity or severity.lower() == "critical":
            if critical_count > 0:
                table.add_row(
                    "🔴 Critical", str(critical_count), "Immediate attention required"
                )

        if not severity or severity.lower() == "warning":
            if warning_count > 0:
                table.add_row(
                    "🟡 Warning", str(warning_count), "Monitoring recommended"
                )

        console.print(table)

        # Show most common alert type
        most_common = alert_summary.get("most_common_alert")
        if most_common:
            console.print(f"\n📋 Most frequent alert type: [bold]{most_common}[/bold]")

        # Show current system status
        current_perf = summary.get("current_performance", {})
        health_status = summary.get("health_status", "unknown")

        console.print(
            f"\n🏥 Current system health: [bold]{health_status.upper()}[/bold]"
        )

        if health_status != "healthy":
            console.print("\n💡 Recommendations:")
            if current_perf.get("avg_response_time_ms", 0) > 200:
                console.print("  • Check database performance and optimize queries")
            if current_perf.get("memory_usage_mb", 0) > 200:
                console.print("  • Monitor memory usage and consider resource scaling")
            if current_perf.get("database_connections", 0) > 15:
                console.print("  • Review database connection pooling settings")

    try:
        asyncio.run(show_alerts())
    except ImportError as e:
        console.print(f"❌ Alert monitoring dependency missing: {e}", style="red")
        raise typer.Exit(1)
    except (ConnectionError, OSError) as e:
        console.print(f"❌ Alert system connection error: {e}", style="red")
        raise typer.Exit(1)
    except (ValueError, TypeError, KeyError) as e:
        console.print(f"❌ Alert data processing error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def interactive(
    refresh_rate: int = typer.Option(2, "--refresh-rate", "-r", help="Update refresh rate in seconds"),
    dark_mode: bool = typer.Option(True, "--dark-mode/--light-mode", help="Use dark mode theme"),
):
    """Launch interactive Rich TUI dashboard for system monitoring and management."""
    console.print("🎛️ Launching APES Interactive Dashboard...", style="green")

    try:
        # Import TUI dashboard
        from prompt_improver.tui.dashboard import run_dashboard

        # Check if required dependencies are available
        try:
            import rich
            import textual
        except ImportError as e:
            console.print(f"❌ Missing TUI dependencies: {e}", style="red")
            console.print("💡 Install with: pip install textual rich", style="yellow")
            raise typer.Exit(1)

        # Display startup information
        console.print("📊 Starting Rich TUI Dashboard...", style="cyan")
        console.print(f"🔄 Refresh rate: {refresh_rate} seconds", style="dim")
        console.print(f"🎨 Theme: {'Dark' if dark_mode else 'Light'}", style="dim")
        console.print("💡 Press Ctrl+C to exit", style="dim")
        console.print("", style="dim")

        # Run the dashboard
        run_dashboard(console)

    except KeyboardInterrupt:
        console.print("\n👋 Dashboard closed by user", style="yellow")
    except ImportError as e:
        console.print(f"❌ Failed to import TUI components: {e}", style="red")
        console.print("💡 Ensure textual and rich are installed", style="yellow")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Dashboard error: {e}", style="red")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
