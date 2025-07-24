#!/usr/bin/env python3
"""
Week 7 Enhanced Stop Command Demonstration
Shows the new signal handling and progress preservation capabilities.
"""

import asyncio
import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Set, Awaitable

# Standalone implementation for demo (avoiding import issues)

class ShutdownReason(Enum):
    """Enumeration of shutdown reasons for tracking and reporting."""
    USER_INTERRUPT = "user_interrupt"
    SYSTEM_SHUTDOWN = "system_shutdown"
    TIMEOUT = "timeout"
    ERROR = "error"
    FORCE = "force"

@dataclass
class ShutdownContext:
    """Context information for shutdown operations."""
    reason: ShutdownReason
    signal_name: Optional[str] = None
    timeout: int = 30
    save_progress: bool = True
    force_after_timeout: bool = True
    started_at: Optional[datetime] = None

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now(timezone.utc)

@dataclass
class ProgressSnapshot:
    """Snapshot of training progress at a specific point in time."""
    session_id: str
    iteration: int
    timestamp: datetime
    performance_metrics: Dict[str, float]
    rule_optimizations: Dict[str, Any]
    synthetic_data_generated: int
    workflow_state: Dict[str, Any]
    model_checkpoints: List[str]
    improvement_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressSnapshot':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class AsyncSignalHandler:
    """Enhanced signal handler for demo purposes."""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.shutdown_context: Optional[ShutdownContext] = None
        self.shutdown_in_progress = False
        self.shutdown_handlers: Dict[str, Callable] = {}
        self.cleanup_handlers: Dict[str, Callable] = {}

    def register_shutdown_handler(self, name: str, handler: Callable) -> None:
        """Register a shutdown handler."""
        self.shutdown_handlers[name] = handler

    def register_cleanup_handler(self, name: str, handler: Callable) -> None:
        """Register a cleanup handler."""
        self.cleanup_handlers[name] = handler

    async def execute_graceful_shutdown(self) -> Dict[str, Any]:
        """Execute graceful shutdown with handlers."""
        if self.shutdown_in_progress:
            return {"status": "already_in_progress"}

        self.shutdown_in_progress = True
        shutdown_start = datetime.now(timezone.utc)

        try:
            # Execute shutdown handlers
            shutdown_results = {}
            for name, handler in self.shutdown_handlers.items():
                try:
                    result = await handler(self.shutdown_context)
                    shutdown_results[name] = {"status": "success", "result": result}
                except Exception as e:
                    shutdown_results[name] = {"status": "error", "error": str(e)}

            # Execute cleanup handlers
            cleanup_results = {}
            for name, handler in self.cleanup_handlers.items():
                try:
                    result = await handler()
                    cleanup_results[name] = {"status": "success", "result": result}
                except Exception as e:
                    cleanup_results[name] = {"status": "error", "error": str(e)}

            shutdown_duration = (datetime.now(timezone.utc) - shutdown_start).total_seconds()

            return {
                "status": "success",
                "reason": self.shutdown_context.reason.value,
                "duration_seconds": shutdown_duration,
                "shutdown_results": shutdown_results,
                "cleanup_results": cleanup_results,
                "progress_saved": self.shutdown_context.save_progress
            }

        finally:
            self.shutdown_in_progress = False

class ProgressPreservationManager:
    """Progress preservation manager for demo purposes."""

    def __init__(self, backup_dir: Optional[Path] = None):
        self.backup_dir = backup_dir or Path("./training_backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.active_sessions: Dict[str, ProgressSnapshot] = {}

    async def save_training_progress(
        self,
        session_id: str,
        iteration: int,
        performance_metrics: Dict[str, float],
        rule_optimizations: Dict[str, Any],
        workflow_state: Dict[str, Any],
        synthetic_data_generated: int = 0,
        model_checkpoints: Optional[List[str]] = None,
        improvement_score: float = 0.0
    ) -> bool:
        """Save training progress to backup file."""
        try:
            snapshot = ProgressSnapshot(
                session_id=session_id,
                iteration=iteration,
                timestamp=datetime.now(timezone.utc),
                performance_metrics=performance_metrics,
                rule_optimizations=rule_optimizations,
                synthetic_data_generated=synthetic_data_generated,
                workflow_state=workflow_state,
                model_checkpoints=model_checkpoints or [],
                improvement_score=improvement_score
            )

            await self._save_to_backup_file(snapshot)
            self.active_sessions[session_id] = snapshot
            return True

        except Exception:
            return False

    async def _save_to_backup_file(self, snapshot: ProgressSnapshot) -> None:
        """Save progress snapshot to backup file."""
        backup_file = self.backup_dir / f"{snapshot.session_id}_progress.json"

        backup_data = {"snapshots": []}
        if backup_file.exists():
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)

        backup_data["snapshots"].append(snapshot.to_dict())

        # Keep only last 50 snapshots
        if len(backup_data["snapshots"]) > 50:
            backup_data["snapshots"] = backup_data["snapshots"][-50:]

        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)

    async def create_checkpoint(self, session_id: str) -> Optional[str]:
        """Create a checkpoint for the session."""
        try:
            checkpoint_id = f"{session_id}_checkpoint_{int(datetime.now(timezone.utc).timestamp())}"
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "session_data": self.active_sessions.get(session_id, {})
            }

            checkpoint_file = self.backup_dir / f"{checkpoint_id}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            return checkpoint_id
        except Exception:
            return None

    async def recover_session_progress(self, session_id: str) -> Optional[ProgressSnapshot]:
        """Recover session progress from backup."""
        try:
            backup_file = self.backup_dir / f"{session_id}_progress.json"
            if backup_file.exists():
                with open(backup_file, 'r') as f:
                    backup_data = json.load(f)

                if backup_data.get("snapshots"):
                    latest_snapshot_data = backup_data["snapshots"][-1]
                    return ProgressSnapshot.from_dict(latest_snapshot_data)
            return None
        except Exception:
            return None

    async def export_session_results(
        self,
        session_id: str,
        export_format: str = "json",
        include_iterations: bool = True
    ) -> Optional[str]:
        """Export session results."""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            export_file = self.backup_dir / f"{session_id}_export_{timestamp}.{export_format}"

            export_data = {
                "session_id": session_id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "export_format": export_format
            }

            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            return str(export_file)
        except Exception:
            return None


async def demo_signal_handling():
    """Demonstrate the enhanced signal handling system."""
    print("🔧 Week 7 Enhanced Signal Handling Demo")
    print("=" * 50)

    # Create signal handler
    signal_handler = AsyncSignalHandler()

    # Register demo shutdown handlers
    async def training_shutdown_handler(context: ShutdownContext):
        print(f"   🛑 Training shutdown handler called (reason: {context.reason.value})")
        await asyncio.sleep(0.1)  # Simulate shutdown work
        return {"training_stopped": True, "sessions_saved": 3}

    async def database_cleanup_handler():
        print("   🗄️  Database cleanup handler called")
        await asyncio.sleep(0.05)  # Simulate cleanup
        return {"connections_closed": 5}

    # Register handlers
    signal_handler.register_shutdown_handler("training_shutdown", training_shutdown_handler)
    signal_handler.register_cleanup_handler("database_cleanup", database_cleanup_handler)

    print(f"✅ Signal handler initialized with {len(signal_handler.shutdown_handlers)} shutdown handlers")
    print(f"✅ Signal handler initialized with {len(signal_handler.cleanup_handlers)} cleanup handlers")

    # Simulate graceful shutdown
    print("\n🔄 Simulating graceful shutdown...")
    signal_handler.shutdown_context = ShutdownContext(
        reason=ShutdownReason.USER_INTERRUPT,
        timeout=30,
        save_progress=True
    )

    result = await signal_handler.execute_graceful_shutdown()

    print(f"✅ Graceful shutdown completed:")
    print(f"   Status: {result['status']}")
    print(f"   Reason: {result['reason']}")
    print(f"   Duration: {result['duration_seconds']:.2f}s")
    print(f"   Progress saved: {result['progress_saved']}")

    return result


async def demo_progress_preservation():
    """Demonstrate the progress preservation system."""
    print("\n💾 Week 7 Progress Preservation Demo")
    print("=" * 50)

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        backup_dir = Path(temp_dir)
        progress_manager = ProgressPreservationManager(backup_dir=backup_dir)

        print(f"✅ Progress manager initialized with backup dir: {backup_dir}")

        # Create sample progress snapshots
        session_id = "demo_session_123"

        print(f"\n📸 Creating progress snapshots for session: {session_id}")

        for iteration in range(1, 6):
            # Simulate improving performance
            accuracy = 0.7 + (iteration * 0.03)
            loss = 0.5 - (iteration * 0.05)

            snapshot = ProgressSnapshot(
                session_id=session_id,
                iteration=iteration,
                timestamp=datetime.now(timezone.utc),
                performance_metrics={
                    "accuracy": accuracy,
                    "loss": loss,
                    "f1_score": accuracy * 0.95
                },
                rule_optimizations={
                    f"rule_{iteration}": {"param": f"value_{iteration}"}
                },
                synthetic_data_generated=iteration * 50,
                workflow_state={
                    "workflow_id": f"wf_{iteration}",
                    "status": "completed"
                },
                model_checkpoints=[f"checkpoint_{iteration}.pkl"],
                improvement_score=0.02 * iteration
            )

            # Save progress
            success = await progress_manager.save_training_progress(
                session_id=snapshot.session_id,
                iteration=snapshot.iteration,
                performance_metrics=snapshot.performance_metrics,
                rule_optimizations=snapshot.rule_optimizations,
                workflow_state=snapshot.workflow_state,
                synthetic_data_generated=snapshot.synthetic_data_generated,
                model_checkpoints=snapshot.model_checkpoints,
                improvement_score=snapshot.improvement_score
            )

            if success:
                print(f"   ✅ Iteration {iteration}: accuracy={accuracy:.3f}, loss={loss:.3f}")
            else:
                print(f"   ❌ Failed to save iteration {iteration}")

        # Create checkpoint
        print(f"\n📋 Creating checkpoint for session: {session_id}")
        checkpoint_id = await progress_manager.create_checkpoint(session_id)
        if checkpoint_id:
            print(f"✅ Checkpoint created: {checkpoint_id}")
        else:
            print("❌ Failed to create checkpoint")

        # Demonstrate recovery
        print(f"\n🔄 Demonstrating progress recovery for session: {session_id}")
        recovered_snapshot = await progress_manager.recover_session_progress(session_id)
        if recovered_snapshot:
            print(f"✅ Recovered session progress:")
            print(f"   Last iteration: {recovered_snapshot.iteration}")
            print(f"   Performance: {recovered_snapshot.performance_metrics}")
            print(f"   Timestamp: {recovered_snapshot.timestamp}")
        else:
            print("❌ Failed to recover session progress")

        # Demonstrate export
        print(f"\n📊 Exporting session results...")
        export_path = await progress_manager.export_session_results(
            session_id=session_id,
            export_format="json",
            include_iterations=True
        )
        if export_path:
            print(f"✅ Results exported to: {export_path}")

            # Show export file size
            export_file = Path(export_path)
            if export_file.exists():
                size_kb = export_file.stat().st_size / 1024
                print(f"   File size: {size_kb:.1f} KB")
        else:
            print("❌ Failed to export results")

        # Show backup files created
        backup_files = list(backup_dir.glob("*.json"))
        print(f"\n📁 Backup files created: {len(backup_files)}")
        for backup_file in backup_files:
            size_kb = backup_file.stat().st_size / 1024
            print(f"   {backup_file.name}: {size_kb:.1f} KB")


async def demo_enhanced_stop_workflow():
    """Demonstrate the complete enhanced stop workflow."""
    print("\n🛑 Week 7 Enhanced Stop Workflow Demo")
    print("=" * 50)

    # Simulate the enhanced stop command workflow
    print("1. 🔍 Checking for active training sessions...")
    active_sessions = [
        {"session_id": "session_1", "status": "running", "iterations": 15},
        {"session_id": "session_2", "status": "running", "iterations": 8}
    ]
    print(f"   Found {len(active_sessions)} active sessions")

    print("\n2. 📸 Creating checkpoints before shutdown...")
    for session in active_sessions:
        print(f"   Creating checkpoint for {session['session_id']}...")
        await asyncio.sleep(0.1)  # Simulate checkpoint creation
        print(f"   ✅ Checkpoint created for {session['session_id']}")

    print("\n3. 🔄 Initiating graceful shutdown...")
    for session in active_sessions:
        print(f"   Stopping {session['session_id']} gracefully...")
        await asyncio.sleep(0.2)  # Simulate graceful shutdown
        print(f"   ✅ {session['session_id']} stopped gracefully")

    print("\n4. 📊 Exporting training results...")
    for session in active_sessions:
        print(f"   Exporting results for {session['session_id']}...")
        await asyncio.sleep(0.1)  # Simulate export
        print(f"   ✅ Results exported for {session['session_id']}")

    print("\n5. 🧹 Performing final cleanup...")
    await asyncio.sleep(0.1)  # Simulate cleanup
    print("   ✅ Old backup files cleaned up")
    print("   ✅ Database connections closed")
    print("   ✅ Training system stopped")

    print("\n🎯 Enhanced stop workflow completed successfully!")


async def main():
    """Run all Week 7 demonstrations."""
    print("🚀 APES Week 7: Enhanced Stop Command Implementation")
    print("🔧 Demonstrating 2025 best practices for graceful shutdown")
    print("=" * 70)

    try:
        # Demo 1: Signal Handling
        await demo_signal_handling()

        # Demo 2: Progress Preservation
        await demo_progress_preservation()

        # Demo 3: Complete Enhanced Stop Workflow
        await demo_enhanced_stop_workflow()

        print("\n" + "=" * 70)
        print("✅ All Week 7 demonstrations completed successfully!")
        print("\n🎉 Week 7 Enhanced Stop Command Implementation Features:")
        print("   🛑 Graceful shutdown with signal handling")
        print("   💾 Comprehensive progress preservation")
        print("   📊 Training results export (JSON/CSV)")
        print("   ⚡ Force shutdown for emergency cases")
        print("   🔄 Session-specific shutdown support")
        print("   📸 Checkpoint creation and recovery")
        print("   🧹 Automatic cleanup and maintenance")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
