"""Clean 3-Command CLI - Ultra-Minimal Training-Focused Interface
Implements complete replacement for 36-command legacy CLI.
"""

import asyncio
import signal
from datetime import UTC, datetime, timezone
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from prompt_improver.cli.core import get_background_manager, get_shared_signal_handler
from prompt_improver.cli.core.cli_orchestrator import CLIOrchestrator
from prompt_improver.cli.core.progress_preservation import ProgressService
from prompt_improver.cli.services.training_orchestrator import TrainingOrchestrator

app = typer.Typer(
    name="apes", help="APES - Ultra-Minimal ML Training System", rich_markup_mode="rich"
)
console = Console()
signal_handler = get_shared_signal_handler()
background_manager = get_background_manager()
training_manager = TrainingOrchestrator(console)
cli_orchestrator = CLIOrchestrator(console)
current_training_session = None
shutdown_requested = False


async def setup_enhanced_signal_handling():
    """Setup enhanced signal handlers using AsyncSignalHandler.

    Implements 2025 best practices for coordinated signal handling:
    - SIGINT (Ctrl+C) for user interruption with progress preservation
    - SIGTERM for system shutdown with coordinated component cleanup
    - SIGUSR1 for emergency checkpoints across all components
    - SIGUSR2 for comprehensive status reporting
    - Signal chaining for coordinated shutdown sequencing
    """
    loop = asyncio.get_running_loop()
    signal_handler.setup_signal_handlers(loop)
    console.print("🔧 Enhanced signal handling initialized", style="dim green")
    console.print("   • SIGUSR1: Emergency checkpoint creation", style="dim")
    console.print("   • SIGUSR2: Comprehensive status reporting", style="dim")
    console.print(
        "   • Ctrl+C: Graceful shutdown with progress preservation", style="dim"
    )


async def enhanced_graceful_shutdown():
    """Enhanced graceful shutdown using AsyncSignalHandler coordination.

    Features:
    - Coordinated shutdown across all CLI components with signal chaining
    - Progress preservation with emergency checkpoints
    - Background task coordination and cleanup
    - Resource cleanup with priority-based sequencing
    """
    global current_training_session
    console.print("🛑 Enhanced graceful shutdown initiated...", style="blue")
    try:
        shutdown_context = await signal_handler.wait_for_shutdown()
        if shutdown_context:
            console.print(
                f"🔧 Shutdown reason: {shutdown_context.reason.value}", style="dim"
            )
            shutdown_results = await signal_handler.execute_graceful_shutdown()
            if shutdown_results["status"] == "success":
                console.print("✅ All components shut down gracefully", style="green")
                console.print(
                    f"⏱️  Shutdown duration: {shutdown_results['duration_seconds']:.2f}s",
                    style="dim",
                )
                if shutdown_results.get("shutdown_results"):
                    for component, result in shutdown_results[
                        "shutdown_results"
                    ].items():
                        if result["status"] == "success":
                            console.print(f"   ✓ {component}", style="dim green")
                        else:
                            console.print(
                                f"   ⚠ {component} ({result.get('status', 'unknown')})",
                                style="dim yellow",
                            )
                if shutdown_results.get("progress_saved"):
                    console.print("💾 Training progress preserved", style="cyan")
            else:
                console.print(
                    f"⚠️  Shutdown completed with status: {shutdown_results['status']}",
                    style="yellow",
                )
        current_training_session = None
    except Exception as e:
        console.print(f"❌ Error during enhanced shutdown: {e}", style="red")
        await basic_cleanup_fallback()
    console.print("👋 Enhanced shutdown complete", style="green")


async def basic_cleanup_fallback():
    """Fallback cleanup when enhanced shutdown fails."""
    try:
        if current_training_session:
            await cli_orchestrator.force_stop_training(current_training_session)
        await background_manager.stop(timeout=10.0)
    except Exception as e:
        console.print(f"⚠️  Fallback cleanup issue: {e}", style="yellow")


@app.command()
def train(
    continuous: bool = typer.Option(
        True,
        "--continuous/--single",
        help="Continuous learning mode (default) or single run",
    ),
    max_iterations: int | None = typer.Option(
        None,
        "--max-iterations",
        "-i",
        help="Maximum training iterations (unlimited by default)",
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
    """🚀 Start ML training with continuous adaptive learning.

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
        await setup_enhanced_signal_handling()
        try:
            if auto_init:
                with console.status("🔧 Checking system initialization..."):
                    init_result = await training_manager.smart_initialize()
                    if not init_result.get("success", False):
                        console.print(
                            f"❌ Initialization failed: {init_result.get('error', 'Unknown error')}",
                            style="red",
                        )
                        raise typer.Exit(1)
                    components_initialized = init_result.get(
                        "components_initialized", []
                    )
                    if components_initialized:
                        console.print(
                            f"✅ Initialized: {', '.join(components_initialized)}",
                            style="green",
                        )
            if not await training_manager.validate_ready_for_training():
                console.print(
                    "❌ System not ready for training. Run with --auto-init to fix.",
                    style="red",
                )
                raise typer.Exit(1)
            session_config = {
                "continuous_mode": continuous,
                "max_iterations": max_iterations,
                "improvement_threshold": improvement_threshold,
                "timeout": timeout,
                "verbose": verbose,
            }
            session = await training_manager.create_training_session(session_config)
            current_training_session = session.session_id
            console.print(
                f"📊 Training session created: {session.session_id}", style="cyan"
            )
            if continuous:
                console.print(
                    "🎯 Starting continuous adaptive training loop...",
                    style="bold blue",
                )
                console.print(
                    "   Press Ctrl+C for graceful shutdown with progress preservation",
                    style="dim",
                )
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                    expand=True,
                ) as progress:
                    task = progress.add_task("Training in progress...", total=None)
                    import uuid

                    from ...performance.monitoring.health.background_manager import (
                        TaskPriority,
                        get_background_task_manager,
                    )

                    task_manager = get_background_task_manager()
                    training_task_id = await task_manager.submit_enhanced_task(
                        task_id=f"cli_training_{session.session_id}_{str(uuid.uuid4())[:8]}",
                        coroutine=cli_orchestrator.start_continuous_training(
                            session_id=session.session_id, config=session_config
                        ),
                        priority=TaskPriority.CRITICAL,
                        tags={
                            "service": "cli",
                            "type": "training",
                            "component": "clean_cli",
                            "session_id": session.session_id,
                        },
                    )
                    shutdown_task_id = await task_manager.submit_enhanced_task(
                        task_id=f"cli_shutdown_{str(uuid.uuid4())[:8]}",
                        coroutine=enhanced_graceful_shutdown(),
                        priority=TaskPriority.HIGH,
                        tags={
                            "service": "cli",
                            "type": "shutdown",
                            "component": "clean_cli",
                        },
                    )
                    training_task = task_manager.get_task(training_task_id)
                    shutdown_task = task_manager.get_task(shutdown_task_id)
                    done, pending = await asyncio.wait(
                        [training_task, shutdown_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    if shutdown_task in done:
                        console.print(
                            "\n🛑 Enhanced graceful shutdown completed", style="green"
                        )
                        return
                    results = await training_task
                    progress.update(
                        task, completed=100, description="Training completed"
                    )
            else:
                console.print(
                    "⚡ Running single training iteration...", style="bold blue"
                )
                results = await cli_orchestrator.start_single_training(
                    session_id=session.session_id, config=session_config
                )
            console.print("\n🎉 Training completed successfully!", style="bold green")
            console.print(
                f"📈 Final performance: {results.get('final_performance', 'N/A')}"
            )
            console.print(f"⏱️  Total time: {results.get('duration', 'N/A')}")
            console.print(f"🔄 Iterations: {results.get('iterations', 'N/A')}")
            if results.get("improvements"):
                console.print(f"📊 Improvements: {results['improvements']}")
            current_training_session = None
        except KeyboardInterrupt:
            console.print("\n⚠️  Training interrupted by user", style="yellow")
            await enhanced_graceful_shutdown()
        except asyncio.CancelledError:
            console.print("\n🛑 Training cancelled gracefully", style="yellow")
        except Exception as e:
            console.print(f"\n❌ Training failed: {e}", style="red")
            await basic_cleanup_fallback()
            raise typer.Exit(1)

    asyncio.run(run_training())


@app.command()
def status(
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed status"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    refresh: int = typer.Option(
        0, "--refresh", "-r", help="Auto-refresh interval in seconds"
    ),
) -> None:
    """Show training system status and active workflows."""

    async def show_status():
        try:
            system_status = await training_manager.get_system_status()
            if json_output:
                import json

                console.print(json.dumps(system_status, indent=2))
                return
            console.print("📊 APES Training System Status", style="bold blue")
            health_color = "green" if system_status["healthy"] else "red"
            console.print(
                f"🏥 System Health: {system_status['status']}", style=health_color
            )
            if system_status.get("active_sessions"):
                console.print("\n🔄 Active Training Sessions:", style="bold")
                for session in system_status["active_sessions"]:
                    session_id = session["session_id"]
                    status = session["status"]
                    status_color = (
                        "green"
                        if status == "running"
                        else "yellow"
                        if status == "paused"
                        else "red"
                    )
                    console.print(f"  📋 {session_id}: {status}", style=status_color)
                    if detailed:
                        console.print(f"    ⏱️  Started: {session['started_at']}")
                        console.print(f"    🔄 Iterations: {session['iterations']}")
                        current_perf = session.get("current_performance", 0.0)
                        if isinstance(current_perf, (int, float)):
                            console.print(f"    📈 Performance: {current_perf:.4f}")
                        else:
                            console.print(f"    📈 Performance: {current_perf}")
                        if session.get("performance_history"):
                            history = session["performance_history"]
                            if len(history) >= 2:
                                recent_trend = history[-1].get(
                                    "performance", 0
                                ) - history[-2].get("performance", 0)
                                trend_icon = (
                                    "📈"
                                    if recent_trend > 0
                                    else "📉"
                                    if recent_trend < 0
                                    else "➡️"
                                )
                                console.print(
                                    f"    {trend_icon} Recent Trend: {recent_trend:+.4f}"
                                )
                        if session.get("last_improvement_time"):
                            import time

                            time_since = time.time() - session["last_improvement_time"]
                            console.print(
                                f"    ⏰ Last Improvement: {time_since:.0f}s ago"
                            )
                        mode = (
                            "Continuous"
                            if session.get("continuous_mode", True)
                            else "Single Run"
                        )
                        console.print(f"    🎯 Mode: {mode}")
                        if session.get("improvement_threshold"):
                            console.print(
                                f"    🎚️  Threshold: {session['improvement_threshold']:.4f}"
                            )
            else:
                console.print("\n💤 No active training sessions", style="dim")
            if detailed:
                console.print("\n🔧 Component Status:", style="bold")
                components = system_status.get("components", {})
                if components:
                    for component, status in components.items():
                        status_color = "green" if status == "healthy" else "red"
                        console.print(f"  • {component}: {status}", style=status_color)
                else:
                    console.print("  No component status available", style="dim")
                if system_status.get("recent_performance"):
                    console.print("\n📈 Performance Metrics:", style="bold")
                    perf = system_status["recent_performance"]
                    model_acc = perf.get("model_accuracy", "N/A")
                    rule_eff = perf.get("rule_effectiveness", "N/A")
                    pattern_cov = perf.get("pattern_coverage", "N/A")
                    if isinstance(model_acc, (int, float)):
                        acc_color = (
                            "green"
                            if model_acc > 0.8
                            else "yellow"
                            if model_acc > 0.6
                            else "red"
                        )
                        console.print(
                            f"  • Model Accuracy: {model_acc:.3f}", style=acc_color
                        )
                    else:
                        console.print(f"  • Model Accuracy: {model_acc}")
                    if isinstance(rule_eff, (int, float)):
                        eff_color = (
                            "green"
                            if rule_eff > 0.8
                            else "yellow"
                            if rule_eff > 0.6
                            else "red"
                        )
                        console.print(
                            f"  • Rule Effectiveness: {rule_eff:.3f}", style=eff_color
                        )
                    else:
                        console.print(f"  • Rule Effectiveness: {rule_eff}")
                    if isinstance(pattern_cov, (int, float)):
                        cov_color = (
                            "green"
                            if pattern_cov > 0.7
                            else "yellow"
                            if pattern_cov > 0.5
                            else "red"
                        )
                        console.print(
                            f"  • Pattern Coverage: {pattern_cov:.3f}", style=cov_color
                        )
                    else:
                        console.print(f"  • Pattern Coverage: {pattern_cov}")
                    if perf.get("training_loss"):
                        loss = perf["training_loss"]
                        loss_color = (
                            "green" if loss < 0.1 else "yellow" if loss < 0.3 else "red"
                        )
                        console.print(
                            f"  • Training Loss: {loss:.4f}", style=loss_color
                        )
                    if perf.get("improvement_rate"):
                        rate = perf["improvement_rate"]
                        rate_color = "green" if rate > 0 else "red"
                        console.print(
                            f"  • Improvement Rate: {rate:+.4f}/iter", style=rate_color
                        )
                if system_status.get("resource_usage"):
                    console.print("\n💻 Resource Usage:", style="bold")
                    res = system_status["resource_usage"]
                    memory_mb = res.get("memory_mb", 0)
                    if isinstance(memory_mb, (int, float)):
                        memory_color = (
                            "red"
                            if memory_mb > 8000
                            else "yellow"
                            if memory_mb > 4000
                            else "green"
                        )
                        console.print(
                            f"  • Memory: {memory_mb:.1f} MB", style=memory_color
                        )
                    else:
                        console.print(f"  • Memory: {memory_mb}")
                    cpu_percent = res.get("cpu_percent", 0)
                    if isinstance(cpu_percent, (int, float)):
                        cpu_color = (
                            "red"
                            if cpu_percent > 90
                            else "yellow"
                            if cpu_percent > 70
                            else "green"
                        )
                        console.print(f"  • CPU: {cpu_percent:.1f}%", style=cpu_color)
                    else:
                        console.print(f"  • CPU: {cpu_percent}")
                    if res.get("disk_usage_mb"):
                        disk = res["disk_usage_mb"]
                        console.print(f"  • Disk: {disk:.1f} MB")
                    if res.get("active_workflows"):
                        workflows = res["active_workflows"]
                        console.print(f"  • Active Workflows: {workflows}")
                if system_status.get("database_status"):
                    console.print("\n🗄️  Database Status:", style="bold")
                    db = system_status["database_status"]
                    db_color = "green" if db.get("connected", False) else "red"
                    console.print(
                        f"  • Connection: {('Connected' if db.get('connected') else 'Disconnected')}",
                        style=db_color,
                    )
                    if db.get("training_sessions_count"):
                        console.print(
                            f"  • Training Sessions: {db['training_sessions_count']}"
                        )
                    if db.get("rules_count"):
                        console.print(f"  • Rules in Database: {db['rules_count']}")
                    if db.get("patterns_count"):
                        console.print(
                            f"  • Discovered Patterns: {db['patterns_count']}"
                        )
        except Exception as e:
            console.print(f"❌ Failed to get status: {e}", style="red")
            raise typer.Exit(1)

    if refresh > 0:
        console.print(
            f"🔄 Auto-refreshing every {refresh}s (Ctrl+C to stop)...", style="yellow"
        )

        async def monitoring_loop():
            try:
                while True:
                    console.clear()
                    await show_status()
                    await asyncio.sleep(refresh)
            except KeyboardInterrupt:
                console.print("\n👋 Status monitoring stopped", style="yellow")

        asyncio.run(monitoring_loop())
    else:
        asyncio.run(show_status())


@app.command()
def why5(
    issue: str | None = typer.Argument(None, help="The problem or issue to analyze"),
    depth: int = typer.Option(
        5, "--depth", "-d", help="Number of 'why' iterations (default: 5)"
    ),
    export: bool = typer.Option(
        False, "--export", "-e", help="Export analysis to file"
    ),
    format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text, json, or markdown"
    ),
) -> None:
    """🔍 Five Whys root cause analysis - drill down from symptoms to root causes.

    Apply the Five Whys methodology to systematically investigate issues by
    iteratively asking "why" to move beyond surface symptoms to fundamental causes.

    Examples:
      apes why5 "Application crashes on startup"
      apes why5 --depth 7 "Performance is slow"
      apes why5 "Database connection fails" --export --format json
    """

    async def run_five_whys():
        """Execute the Five Whys analysis workflow."""
        try:
            if not issue:
                issue_description = typer.prompt(
                    "🤔 What issue would you like to analyze?"
                )
            else:
                issue_description = issue
            console.print(
                f"\n🔍 Five Whys Analysis: {issue_description}", style="bold blue"
            )
            console.print("=" * (len(issue_description) + 24), style="dim")
            analysis_results = {
                "problem_statement": issue_description,
                "timestamp": datetime.now(UTC).isoformat(),
                "analysis_chain": [],
                "root_cause": None,
                "proposed_solutions": [],
            }
            current_question = issue_description
            for i in range(depth):
                why_number = i + 1
                console.print(
                    f"\n❓ Why #{why_number}: {current_question}", style="yellow"
                )
                answer = typer.prompt("   Answer")
                analysis_step = {
                    "why_number": why_number,
                    "question": current_question,
                    "answer": answer,
                }
                analysis_results["analysis_chain"].append(analysis_step)
                console.print(f"   💡 {answer}", style="green")
                if why_number < depth:
                    continue_analysis = typer.confirm(
                        f"Continue to Why #{why_number + 1}?", default=True
                    )
                    if not continue_analysis:
                        break
                current_question = answer
            if analysis_results["analysis_chain"]:
                root_cause = analysis_results["analysis_chain"][-1]["answer"]
                analysis_results["root_cause"] = root_cause
                console.print("\n🎯 Root Cause Identified:", style="bold red")
                console.print(f"   {root_cause}", style="red")
            console.print(
                "\n🔄 Validation: Working backwards through the causal chain...",
                style="blue",
            )
            for step in reversed(analysis_results["analysis_chain"]):
                console.print(
                    f"   • {step['answer']} → {step['question']}", style="dim"
                )
            console.print("\n💡 Proposed Solutions:", style="bold green")
            console.print(
                "   Based on the root cause, what solutions would address this?"
            )
            solutions = []
            solution_count = 1
            while True:
                solution = typer.prompt(
                    f"   Solution #{solution_count} (or press Enter to finish)",
                    default="",
                    show_default=False,
                )
                if not solution.strip():
                    break
                solutions.append(solution)
                console.print(f"   ✅ {solution}", style="green")
                solution_count += 1
            analysis_results["proposed_solutions"] = solutions
            console.print("\n📋 Analysis Summary:", style="bold cyan")
            console.print(f"   • Problem: {issue_description}")
            console.print(
                f"   • Analysis Depth: {len(analysis_results['analysis_chain'])} whys"
            )
            console.print(f"   • Root Cause: {root_cause}")
            console.print(f"   • Solutions Identified: {len(solutions)}")
            if export:
                await export_analysis(analysis_results, format)
        except KeyboardInterrupt:
            console.print("\n⚠️  Analysis interrupted by user", style="yellow")
        except Exception as e:
            console.print(f"\n❌ Analysis failed: {e}", style="red")
            raise typer.Exit(1)

    async def export_analysis(results: dict[str, Any], export_format: str):
        """Export analysis results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if export_format.lower() == "json":
                import json

                filename = f"five_whys_analysis_{timestamp}.json"
                with open(filename, "w") as f:
                    json.dump(results, f, indent=2)
                console.print(f"📁 Analysis exported to: {filename}", style="cyan")
            elif export_format.lower() == "markdown":
                filename = f"five_whys_analysis_{timestamp}.md"
                with open(filename, "w") as f:
                    f.write("# Five Whys Analysis\n\n")
                    f.write(
                        f"**Problem Statement:** {results['problem_statement']}\n\n"
                    )
                    f.write(f"**Analysis Date:** {results['timestamp']}\n\n")
                    f.write("## Analysis Chain\n\n")
                    for step in results["analysis_chain"]:
                        f.write(f"**Why #{step['why_number']}:** {step['question']}\n")
                        f.write(f"**Answer:** {step['answer']}\n\n")
                    f.write(f"## Root Cause\n\n{results['root_cause']}\n\n")
                    if results["proposed_solutions"]:
                        f.write("## Proposed Solutions\n\n")
                        for i, solution in enumerate(results["proposed_solutions"], 1):
                            f.write(f"{i}. {solution}\n")
                console.print(f"📁 Analysis exported to: {filename}", style="cyan")
            else:
                filename = f"five_whys_analysis_{timestamp}.txt"
                with open(filename, "w") as f:
                    f.write("Five Whys Analysis\n")
                    f.write("==================\n\n")
                    f.write(f"Problem Statement: {results['problem_statement']}\n")
                    f.write(f"Analysis Date: {results['timestamp']}\n\n")
                    f.write("Analysis Chain:\n")
                    for step in results["analysis_chain"]:
                        f.write(f"Why #{step['why_number']}: {step['question']}\n")
                        f.write(f"Answer: {step['answer']}\n\n")
                    f.write(f"Root Cause: {results['root_cause']}\n\n")
                    if results["proposed_solutions"]:
                        f.write("Proposed Solutions:\n")
                        for i, solution in enumerate(results["proposed_solutions"], 1):
                            f.write(f"{i}. {solution}\n")
                console.print(f"📁 Analysis exported to: {filename}", style="cyan")
        except Exception as e:
            console.print(f"⚠️  Export failed: {e}", style="yellow")

    asyncio.run(run_five_whys())


@app.command()
def stop(
    graceful: bool = typer.Option(True, "--graceful/--force", help="Graceful shutdown"),
    timeout: int = typer.Option(
        30, "--timeout", "-t", help="Shutdown timeout in seconds"
    ),
    save_progress: bool = typer.Option(
        True, "--save-progress/--no-save", help="Save training progress"
    ),
    session_id: str | None = typer.Option(
        None, "--session", "-s", help="Specific session ID to stop"
    ),
    export_results: bool = typer.Option(
        True,
        "--export-results/--no-export",
        help="Export training results during shutdown",
    ),
    export_format: str = typer.Option(
        "json", "--export-format", help="Export format (json or csv)"
    ),
) -> None:
    """🛑 Stop training with comprehensive progress preservation.

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
            training_manager = TrainingOrchestrator(console)
            cli_orchestrator = CLIOrchestrator()
            progress_manager = ProgressService()
            active_sessions = await training_manager.get_active_sessions()
            if not active_sessions:
                console.print("💤 No active training sessions to stop", style="dim")
                return
            sessions_to_stop = []
            if session_id:
                session = next(
                    (s for s in active_sessions if s.session_id == session_id), None
                )
                if session:
                    sessions_to_stop = [session]
                else:
                    console.print(f"❌ Session {session_id} not found", style="red")
                    raise typer.Exit(1)
            else:
                sessions_to_stop = active_sessions
            for session in sessions_to_stop:
                console.print(
                    f"🔄 Stopping session {session.session_id}...", style="blue"
                )
                if save_progress:
                    console.print(
                        "📸 Creating checkpoint before shutdown...", style="dim"
                    )
                    checkpoint_id = await progress_manager.create_checkpoint(
                        session.session_id
                    )
                    if checkpoint_id:
                        console.print(
                            f"✅ Checkpoint created: {checkpoint_id}", style="dim green"
                        )
                if graceful:
                    result = await cli_orchestrator.stop_training_gracefully(
                        session_id=session.session_id,
                        _timeout=timeout,
                        save_progress=save_progress,
                    )
                    if result.get("success"):
                        console.print(
                            f"✅ Session {session.session_id} stopped gracefully",
                            style="green",
                        )
                        if export_results:
                            console.print(
                                "📊 Exporting training results...", style="dim"
                            )
                            export_path = await progress_manager.export_session_results(
                                session_id=session.session_id,
                                export_format=export_format,
                                include_iterations=True,
                            )
                            if export_path:
                                console.print(
                                    f"📁 Results exported to: {export_path}",
                                    style="cyan",
                                )
                        if result.get("progress_saved"):
                            console.print("💾 Progress preserved", style="cyan")
                    else:
                        console.print(
                            f"⚠️  Session {session.session_id} stopped with issues: {result.get('message', 'Unknown error')}",
                            style="yellow",
                        )
                        if export_results:
                            console.print(
                                "📊 Attempting results export despite issues...",
                                style="yellow",
                            )
                            export_path = await progress_manager.export_session_results(
                                session_id=session.session_id,
                                export_format=export_format,
                                include_iterations=True,
                            )
                            if export_path:
                                console.print(
                                    f"📁 Results exported to: {export_path}",
                                    style="cyan",
                                )
                else:
                    console.print("⚡ Initiating force shutdown...", style="red")
                    if save_progress:
                        console.print(
                            "💾 Saving critical progress data...", style="yellow"
                        )
                        try:
                            await progress_manager.save_training_progress(
                                session_id=session.session_id,
                                iteration=session.total_iterations or 0,
                                performance_metrics=session.best_performance or {},
                                rule_optimizations={},
                                workflow_state={"force_shutdown": True},
                                improvement_score=0.0,
                            )
                            console.print("✅ Critical progress saved", style="green")
                        except Exception as e:
                            console.print(
                                f"⚠️  Could not save progress: {e}", style="yellow"
                            )
                    await cli_orchestrator.force_stop_training(session.session_id)
                    console.print(
                        f"⚡ Session {session.session_id} force stopped", style="yellow"
                    )
            console.print("🧹 Performing final cleanup...", style="blue")
            cleaned_files = await progress_manager.cleanup_old_backups(days_to_keep=30)
            if cleaned_files > 0:
                console.print(
                    f"   🗑️  Cleaned up {cleaned_files} old backup files", style="dim"
                )
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
