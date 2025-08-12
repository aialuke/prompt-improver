"""Baseline profiling script for Phase 4 refactoring.
This script profiles the complex CLI functions before refactoring.
"""

import cProfile
import pstats
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def profile_logs_function():
    """Profile the logs function indirectly by testing its core logic."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        log_entries = [
            "2024-01-15 10:00:00 INFO Starting application...",
            "2024-01-15 10:00:01 DEBUG Database connection established",
            "2024-01-15 10:00:02 WARNING High memory usage detected: 85%",
            "2024-01-15 10:00:03 ERROR Failed to process request: timeout",
            "2024-01-15 10:00:04 INFO Request processed successfully",
        ] * 100
        for entry in log_entries:
            f.write(entry + "\n")
        temp_log_file = f.name

    def simulate_log_processing() -> None:
        """Simulate the core logic of the logs function."""
        level_filter = "INFO"
        lines_limit = 50
        log_path = Path(temp_log_file)
        with log_path.open(encoding="utf-8") as f:
            all_lines = f.readlines()
            recent_lines = (
                all_lines[-lines_limit:] if len(all_lines) > lines_limit else all_lines
            )
            for line in recent_lines:
                if level_filter and level_filter.upper() not in line.upper():
                    continue
                if "ERROR" in line:
                    color = "red"
                elif "WARNING" in line:
                    color = "yellow"
                elif "INFO" in line:
                    color = "green"
                elif "DEBUG" in line:
                    color = "dim"
                else:
                    color = "default"
                _ = f"[{color}]{line.rstrip()}[/{color}]"

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(10):
        simulate_log_processing()
    pr.disable()
    profile_path = Path(__file__).parent / "cprofile_logs_baseline.prof"
    pr.dump_stats(str(profile_path))
    stats = pstats.Stats(pr)
    stats.sort_stats("cumulative")
    print("=== LOGS FUNCTION BASELINE PROFILE ===")
    stats.print_stats(20)
    Path(temp_log_file).unlink()
    return profile_path


if __name__ == "__main__":
    print("Creating baseline performance profiles...")
    logs_profile = profile_logs_function()
    print("\nProfiles saved:")
    print(f"  Logs function: {logs_profile}")
    print("\nBaseline profiling complete!")
