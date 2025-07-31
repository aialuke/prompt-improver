"""
Clean 3-Command CLI - Ultra-Minimal Training-Focused Interface
Implements complete replacement for 36-command legacy CLI.
"""

import asyncio
import signal
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from typing import Optional

from .core.training_system_manager import TrainingSystemManager
from .core.cli_orchestrator import CLIOrchestrator
from .core.progress_preservation import ProgressPreservationManager

# Create clean CLI app with no legacy dependencies
app = typer.Typer(
    name="apes",
    help="APES - Ultra-Minimal ML Training System",
    rich_markup_mode="rich",
)
console = Console()

# Clean training system components (no MCP dependencies)
training_manager = TrainingSystemManager(console)
cli_orchestrator = CLIOrchestrator(console)

# Global state for graceful shutdown
current_training_session = None
shutdown_requested = False

def setup_signal_handlers():
    """
    Setup signal handlers for graceful shutdown.

    Implements 2025 best practices for async signal handling:
    - SIGINT (Ctrl+C) for user interruption
    - SIGTERM for system shutdown
    - Graceful workflow termination with progress preservation
    """
    def signal_handler(signum, _frame):
        global shutdown_requested
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        console.print(f"\n⚠️  Received {signal_name} - Initiating graceful shutdown...", style="yellow")
        shutdown_requested = True

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def graceful_shutdown():
    """
    Perform graceful shutdown of active training sessions.

    Features:
    - Progress preservation to database
    - Workflow completion waiting with timeout
    - Resource cleanup and session finalization
    """
    global current_training_session

    if current_training_session:
        console.print("🛑 Stopping active training session gracefully...", style="yellow")

        try:
            # Stop training gracefully with progress preservation
            result = await cli_orchestrator.stop_training_gracefully(
                session_id=current_training_session,
                _timeout=30,
                save_progress=True
            )

            if result.get("success"):
                console.print("✅ Training session stopped gracefully", style="green")
                if result.get("progress_saved"):
                    console.print("💾 Training progress preserved", style="cyan")
            else:
                console.print(f"⚠️  Graceful stop completed with issues: {result.get('error', 'Unknown')}", style="yellow")

        except Exception as e:
            console.print(f"❌ Error during graceful shutdown: {e}", style="red")
            # Force stop as fallback
            try:
                await cli_orchestrator.force_stop_training(current_training_session)
                console.print("⚡ Training force stopped", style="yellow")
            except Exception as force_error:
                console.print(f"💥 Force stop failed: {force_error}", style="red")

        current_training_session = None

    console.print("👋 Shutdown complete", style="green")

@app.command()
def train(
    continuous: bool = typer.Option(
        True, "--continuous/--single", help="Continuous learning mode (default) or single run"
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", "-i", help="Maximum training iterations (unlimited by default)"
    ),
    improvement_threshold: float = typer.Option(
        0.02, "--threshold", "-t", help="Minimum improvement threshold (2% default)"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Training timeout in seconds (1 hour default)"
    ),
    auto_init: bool = typer.Option(
        True, "--auto-init/--no-init", help="Auto-initialize missing components"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    🚀 Start ML training with continuous adaptive learning.

    The train command runs the complete ML pipeline with:
    - Automatic system initialization and synthetic data generation
    - Continuous loop: Trains → Analyzes gaps → Generates data → Repeats
    - Intelligent stopping when improvement plateaus or user interrupts
    - Comprehensive session tracking and progress preservation
    """
    console.print("🚀 Starting APES ML Training System...", style="bold green")

    if continuous:
        console.print("🔄 Continuous adaptive learning mode enabled", style="blue")
    else:
        console.print("⚡ Single training run mode", style="blue")

    async def run_training():
        global current_training_session, shutdown_requested

        # Setup signal handlers for graceful interruption
        setup_signal_handlers()

        try:
            # Auto-initialize system if needed
            if auto_init:
                with console.status("🔧 Checking system initialization..."):
                    init_result = await training_manager.smart_initialize()
                    if not init_result.get("success", False):
                        console.print(f"❌ Initialization failed: {init_result.get('error', 'Unknown error')}", style="red")
                        raise typer.Exit(1)

                    components_initialized = init_result.get("components_initialized", [])
                    if components_initialized:
                        console.print(f"✅ Initialized: {', '.join(components_initialized)}", style="green")

            # Validate system readiness
            if not await training_manager.validate_ready_for_training():
                console.print("❌ System not ready for training. Run with --auto-init to fix.", style="red")
                raise typer.Exit(1)

            # Create training session
            session_config = {
                "continuous_mode": continuous,
                "max_iterations": max_iterations,
                "improvement_threshold": improvement_threshold,
                "timeout": timeout,
                "verbose": verbose
            }

            session = await training_manager.create_training_session(session_config)
            current_training_session = session.session_id  # Track for graceful shutdown
            console.print(f"📊 Training session created: {session.session_id}", style="cyan")

            # Start continuous training workflow with signal monitoring
            if continuous:
                console.print("🎯 Starting continuous adaptive training loop...", style="bold blue")
                console.print("   Press Ctrl+C for graceful shutdown with progress preservation", style="dim")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                    expand=True,
                ) as progress:
                    task = progress.add_task("Training in progress...", total=None)

                    # Run continuous training with signal monitoring
                    training_task = asyncio.create_task(
                        cli_orchestrator.start_continuous_training(
                            session_id=session.session_id,
                            config=session_config
                        )
                    )

                    # Monitor for shutdown signals
                    while not training_task.done():
                        if shutdown_requested:
                            console.print("\n🛑 Graceful shutdown requested...", style="yellow")
                            training_task.cancel()
                            await graceful_shutdown()
                            return

                        await asyncio.sleep(0.1)  # Check every 100ms

                    results = await training_task
                    progress.update(task, completed=100, description="Training completed")
            else:
                # Single training run
                console.print("⚡ Running single training iteration...", style="bold blue")
                results = await cli_orchestrator.start_single_training(
                    session_id=session.session_id,
                    config=session_config
                )

            # Display results
            console.print("\n🎉 Training completed successfully!", style="bold green")
            console.print(f"📈 Final performance: {results.get('final_performance', 'N/A')}")
            console.print(f"⏱️  Total time: {results.get('duration', 'N/A')}")
            console.print(f"🔄 Iterations: {results.get('iterations', 'N/A')}")

            if results.get('improvements'):
                console.print(f"📊 Improvements: {results['improvements']}")

            # Clear current session on successful completion
            current_training_session = None

        except KeyboardInterrupt:
            console.print("\n⚠️  Training interrupted by user", style="yellow")
            # Graceful shutdown handled by signal handlers
            await graceful_shutdown()
        except asyncio.CancelledError:
            console.print("\n🛑 Training cancelled gracefully", style="yellow")
            # Session cleanup handled by graceful_shutdown()
        except Exception as e:
            console.print(f"\n❌ Training failed: {e}", style="red")
            # Cleanup on error
            if current_training_session:
                try:
                    await cli_orchestrator.force_stop_training(current_training_session)
                    current_training_session = None
                except Exception:
                    pass  # Best effort cleanup
            raise typer.Exit(1)

    # Run the async training function
    asyncio.run(run_training())

@app.command()
def status(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed status"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    refresh: int = typer.Option(0, "--refresh", "-r", help="Auto-refresh interval in seconds"),
) -> None:
    """Show training system status and active workflows."""

    async def show_status():
        try:
            # Get system status
            system_status = await training_manager.get_system_status()

            if json_output:
                import json
                console.print(json.dumps(system_status, indent=2))
                return

            # Display system health
            console.print("📊 APES Training System Status", style="bold blue")
            health_color = "green" if system_status["healthy"] else "red"
            console.print(f"🏥 System Health: {system_status['status']}", style=health_color)

            # Display active sessions with enhanced metrics
            if system_status.get("active_sessions"):
                console.print("\n🔄 Active Training Sessions:", style="bold")
                for session in system_status["active_sessions"]:
                    session_id = session['session_id']
                    status = session['status']

                    # Color-code session status
                    status_color = "green" if status == "running" else "yellow" if status == "paused" else "red"
                    console.print(f"  📋 {session_id}: {status}", style=status_color)

                    if detailed:
                        # Basic session info
                        console.print(f"    ⏱️  Started: {session['started_at']}")
                        console.print(f"    🔄 Iterations: {session['iterations']}")

                        # Performance metrics with trend indicators
                        current_perf = session.get('current_performance', 0.0)
                        if isinstance(current_perf, (int, float)):
                            console.print(f"    📈 Performance: {current_perf:.4f}")
                        else:
                            console.print(f"    📈 Performance: {current_perf}")

                        # Training progress and trends
                        if session.get('performance_history'):
                            history = session['performance_history']
                            if len(history) >= 2:
                                recent_trend = history[-1].get('performance', 0) - history[-2].get('performance', 0)
                                trend_icon = "📈" if recent_trend > 0 else "📉" if recent_trend < 0 else "➡️"
                                console.print(f"    {trend_icon} Recent Trend: {recent_trend:+.4f}")

                        # Time since last improvement
                        if session.get('last_improvement_time'):
                            import time
                            time_since = time.time() - session['last_improvement_time']
                            console.print(f"    ⏰ Last Improvement: {time_since:.0f}s ago")

                        # Training mode and configuration
                        mode = "Continuous" if session.get('continuous_mode', True) else "Single Run"
                        console.print(f"    🎯 Mode: {mode}")

                        if session.get('improvement_threshold'):
                            console.print(f"    🎚️  Threshold: {session['improvement_threshold']:.4f}")
            else:
                console.print("\n💤 No active training sessions", style="dim")

            # Display component status with enhanced details
            if detailed:
                console.print("\n🔧 Component Status:", style="bold")
                components = system_status.get("components", {})
                if components:
                    for component, status in components.items():
                        status_color = "green" if status == "healthy" else "red"
                        console.print(f"  • {component}: {status}", style=status_color)
                else:
                    console.print("  No component status available", style="dim")

                # Display recent performance with enhanced metrics
                if system_status.get("recent_performance"):
                    console.print("\n📈 Performance Metrics:", style="bold")
                    perf = system_status["recent_performance"]

                    # Core performance metrics
                    model_acc = perf.get('model_accuracy', 'N/A')
                    rule_eff = perf.get('rule_effectiveness', 'N/A')
                    pattern_cov = perf.get('pattern_coverage', 'N/A')

                    if isinstance(model_acc, (int, float)):
                        acc_color = "green" if model_acc > 0.8 else "yellow" if model_acc > 0.6 else "red"
                        console.print(f"  • Model Accuracy: {model_acc:.3f}", style=acc_color)
                    else:
                        console.print(f"  • Model Accuracy: {model_acc}")

                    if isinstance(rule_eff, (int, float)):
                        eff_color = "green" if rule_eff > 0.8 else "yellow" if rule_eff > 0.6 else "red"
                        console.print(f"  • Rule Effectiveness: {rule_eff:.3f}", style=eff_color)
                    else:
                        console.print(f"  • Rule Effectiveness: {rule_eff}")

                    if isinstance(pattern_cov, (int, float)):
                        cov_color = "green" if pattern_cov > 0.7 else "yellow" if pattern_cov > 0.5 else "red"
                        console.print(f"  • Pattern Coverage: {pattern_cov:.3f}", style=cov_color)
                    else:
                        console.print(f"  • Pattern Coverage: {pattern_cov}")

                    # Additional performance indicators
                    if perf.get('training_loss'):
                        loss = perf['training_loss']
                        loss_color = "green" if loss < 0.1 else "yellow" if loss < 0.3 else "red"
                        console.print(f"  • Training Loss: {loss:.4f}", style=loss_color)

                    if perf.get('improvement_rate'):
                        rate = perf['improvement_rate']
                        rate_color = "green" if rate > 0 else "red"
                        console.print(f"  • Improvement Rate: {rate:+.4f}/iter", style=rate_color)

                # Display resource usage with health indicators
                if system_status.get("resource_usage"):
                    console.print("\n💻 Resource Usage:", style="bold")
                    res = system_status["resource_usage"]

                    # Memory usage with color coding
                    memory_mb = res.get('memory_mb', 0)
                    if isinstance(memory_mb, (int, float)):
                        memory_color = "red" if memory_mb > 8000 else "yellow" if memory_mb > 4000 else "green"
                        console.print(f"  • Memory: {memory_mb:.1f} MB", style=memory_color)
                    else:
                        console.print(f"  • Memory: {memory_mb}")

                    # CPU usage with color coding
                    cpu_percent = res.get('cpu_percent', 0)
                    if isinstance(cpu_percent, (int, float)):
                        cpu_color = "red" if cpu_percent > 90 else "yellow" if cpu_percent > 70 else "green"
                        console.print(f"  • CPU: {cpu_percent:.1f}%", style=cpu_color)
                    else:
                        console.print(f"  • CPU: {cpu_percent}")

                    # Additional resource metrics
                    if res.get('disk_usage_mb'):
                        disk = res['disk_usage_mb']
                        console.print(f"  • Disk: {disk:.1f} MB")

                    if res.get('active_workflows'):
                        workflows = res['active_workflows']
                        console.print(f"  • Active Workflows: {workflows}")

                # Display database status
                if system_status.get("database_status"):
                    console.print("\n🗄️  Database Status:", style="bold")
                    db = system_status["database_status"]
                    db_color = "green" if db.get("connected", False) else "red"
                    console.print(f"  • Connection: {'Connected' if db.get('connected') else 'Disconnected'}", style=db_color)

                    if db.get("training_sessions_count"):
                        console.print(f"  • Training Sessions: {db['training_sessions_count']}")

                    if db.get("rules_count"):
                        console.print(f"  • Rules in Database: {db['rules_count']}")

                    if db.get("patterns_count"):
                        console.print(f"  • Discovered Patterns: {db['patterns_count']}")

        except Exception as e:
            console.print(f"❌ Failed to get status: {e}", style="red")
            raise typer.Exit(1)

    if refresh > 0:
        console.print(f"🔄 Auto-refreshing every {refresh}s (Ctrl+C to stop)...", style="yellow")
        try:
            while True:
                console.clear()
                asyncio.run(show_status())
                import time
                time.sleep(refresh)
        except KeyboardInterrupt:
            console.print("\n👋 Status monitoring stopped", style="yellow")
    else:
        asyncio.run(show_status())

@app.command()
def stop(
    graceful: bool = typer.Option(True, "--graceful/--force", help="Graceful shutdown"),
    timeout: int = typer.Option(
        30, "--timeout", "-t", help="Shutdown timeout in seconds"
    ),
    save_progress: bool = typer.Option(
        True, "--save-progress/--no-save", help="Save training progress"
    ),
    session_id: Optional[str] = typer.Option(
        None, "--session", "-s", help="Specific session ID to stop"
    ),
    export_results: bool = typer.Option(
        True, "--export-results/--no-export", help="Export training results during shutdown"
    ),
    export_format: str = typer.Option(
        "json", "--export-format", help="Export format (json or csv)"
    ),
) -> None:
    """
    🛑 Stop training with comprehensive progress preservation.

    Enhanced Features:
    - Graceful shutdown with workflow completion waiting
    - Training progress preservation to PostgreSQL
    - Current results export during shutdown (JSON/CSV)
    - Force shutdown for emergency cases
    - Session-specific shutdown support
    - Signal-aware shutdown coordination
    """
    console.print("🛑 Stopping APES Training System...", style="yellow")

    async def enhanced_stop_training():
        try:
            # Initialize enhanced components
            training_manager = TrainingSystemManager()
            cli_orchestrator = CLIOrchestrator()
            progress_manager = ProgressPreservationManager()

            # Get active sessions
            active_sessions = await training_manager.get_active_sessions()

            if not active_sessions:
                console.print("💤 No active training sessions to stop", style="dim")
                return

            # Determine which sessions to stop
            sessions_to_stop = []
            if session_id:
                session = next((s for s in active_sessions if s.session_id == session_id), None)
                if session:
                    sessions_to_stop = [session]
                else:
                    console.print(f"❌ Session {session_id} not found", style="red")
                    raise typer.Exit(1)
            else:
                sessions_to_stop = active_sessions

            # Stop each session with enhanced progress preservation
            for session in sessions_to_stop:
                console.print(f"🔄 Stopping session {session.session_id}...", style="blue")

                # Create checkpoint before shutdown if saving progress
                if save_progress:
                    console.print("📸 Creating checkpoint before shutdown...", style="dim")
                    checkpoint_id = await progress_manager.create_checkpoint(session.session_id)
                    if checkpoint_id:
                        console.print(f"✅ Checkpoint created: {checkpoint_id}", style="dim green")

                if graceful:
                    # Graceful shutdown with progress saving
                    result = await cli_orchestrator.stop_training_gracefully(
                        session_id=session.session_id,
                        _timeout=timeout,
                        save_progress=save_progress
                    )

                    if result.get("success"):
                        console.print(f"✅ Session {session.session_id} stopped gracefully", style="green")

                        # Export results if requested
                        if export_results:
                            console.print("📊 Exporting training results...", style="dim")
                            export_path = await progress_manager.export_session_results(
                                session_id=session.session_id,
                                export_format=export_format,
                                include_iterations=True
                            )
                            if export_path:
                                console.print(f"📁 Results exported to: {export_path}", style="cyan")

                        if result.get("progress_saved"):
                            console.print(f"💾 Progress preserved", style="cyan")
                    else:
                        console.print(f"⚠️  Session {session.session_id} stopped with issues: {result.get('message', 'Unknown error')}", style="yellow")

                        # Still try to export results on partial failure
                        if export_results:
                            console.print("📊 Attempting results export despite issues...", style="yellow")
                            export_path = await progress_manager.export_session_results(
                                session_id=session.session_id,
                                export_format=export_format,
                                include_iterations=True
                            )
                            if export_path:
                                console.print(f"📁 Results exported to: {export_path}", style="cyan")
                else:
                    # Force shutdown with minimal progress preservation
                    console.print("⚡ Initiating force shutdown...", style="red")

                    # Save critical progress data quickly
                    if save_progress:
                        console.print("💾 Saving critical progress data...", style="yellow")
                        try:
                            await progress_manager.save_training_progress(
                                session_id=session.session_id,
                                iteration=session.total_iterations or 0,
                                performance_metrics=session.best_performance or {},
                                rule_optimizations={},
                                workflow_state={"force_shutdown": True},
                                improvement_score=0.0
                            )
                            console.print("✅ Critical progress saved", style="green")
                        except Exception as e:
                            console.print(f"⚠️  Could not save progress: {e}", style="yellow")

                    # Force stop the session
                    await cli_orchestrator.force_stop_training(session.session_id)
                    console.print(f"⚡ Session {session.session_id} force stopped", style="yellow")

            # Final cleanup
            console.print("🧹 Performing final cleanup...", style="blue")

            # Cleanup old backups
            cleaned_files = await progress_manager.cleanup_old_backups(days_to_keep=30)
            if cleaned_files > 0:
                console.print(f"   🗑️  Cleaned up {cleaned_files} old backup files", style="dim")

            # Stop training system
            success = await training_manager.stop_training_system(graceful=graceful)
            if success:
                console.print("✅ Training system stopped successfully", style="green")
            else:
                console.print("⚠️  Training system stopped with issues", style="yellow")

            console.print("🎯 Shutdown completed", style="bold green")

        except KeyboardInterrupt:
            console.print("\n⚠️  Stop operation interrupted", style="yellow")
        except Exception as e:
            console.print(f"❌ Failed to stop training: {e}", style="red")
            console.print("💡 Try using --force for emergency shutdown", style="yellow")
            raise typer.Exit(1)

    asyncio.run(enhanced_stop_training())

def main():
    """Main entry point for clean CLI."""
    app()

if __name__ == "__main__":
    main()
