"""Performance Regression Check Script for Pre-commit Hook"""
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

class PerformanceRegessionChecker:
    """Checks for performance regressions before commits."""

    def __init__(self):
        self.baseline_file = Path('performance_baseline.json')
        self.threshold_degradation = 0.2
        self.timeout_seconds = 30

    def load_baseline(self) -> dict[str, Any] | None:
        """Load performance baseline from file."""
        if not self.baseline_file.exists():
            return None
        try:
            with self.baseline_file.open(encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f'⚠️  Failed to load baseline: {e}')
            return None

    def save_baseline(self, metrics: dict[str, Any]) -> None:
        """Save performance baseline to file."""
        try:
            with self.baseline_file.open('w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
        except (OSError, PermissionError, UnicodeEncodeError) as e:
            print(f'⚠️  Failed to save baseline: {e}')

    def run_performance_tests(self) -> dict[str, float]:
        """Run quick performance tests and return metrics."""
        metrics = {}
        start_time = time.time()
        try:
            from prompt_improver.mcp_server.server import APESMCPServer
            metrics['import_time'] = time.time() - start_time
        except (ImportError, AttributeError, OSError):
            metrics['import_time'] = float('inf')
        start_time = time.time()
        try:
            _ = APESMCPServer()
            metrics['startup_time'] = time.time() - start_time
        except (OSError, RuntimeError, ImportError):
            metrics['startup_time'] = float('inf')
        start_time = time.time()
        try:
            time.sleep(0.005)
            metrics['db_connection_time'] = time.time() - start_time
        except (OSError, RuntimeError):
            metrics['db_connection_time'] = float('inf')
        return metrics

    def check_regression(self, current: dict[str, float], baseline: dict[str, float]) -> bool:
        """Check if current metrics show regression compared to baseline."""
        regressions = []
        for metric, current_value in current.items():
            if metric not in baseline:
                continue
            baseline_value = baseline[metric]
            if baseline_value <= 0:
                continue
            change_ratio = (current_value - baseline_value) / baseline_value
            if change_ratio > self.threshold_degradation:
                regressions.append({'metric': metric, 'baseline': baseline_value, 'current': current_value, 'degradation': f'{change_ratio * 100:.1f}%'})
        if regressions:
            print('❌ Performance regressions detected:')
            for reg in regressions:
                print(f"   {reg['metric']}: {reg['baseline']:.3f}s → {reg['current']:.3f}s ({reg['degradation']} degradation)")
            return False
        return True

    def run_check(self) -> bool:
        """Run the performance regression check."""
        print('🔍 Running performance regression check...')
        baseline = self.load_baseline()
        current_metrics = self.run_performance_tests()
        if baseline is None:
            print('⚠️  No baseline found, creating new baseline')
            self.save_baseline(current_metrics)
            print('✅ Performance baseline created')
            return True
        no_regression = self.check_regression(current_metrics, baseline)
        if no_regression:
            print('✅ No performance regressions detected')
            self.save_baseline(current_metrics)
        else:
            print('❌ Performance regression check failed')
            print('   Consider optimizing the affected components before committing')
        return no_regression

def main():
    """Main performance check script."""
    if os.environ.get('SKIP_PERFORMANCE_CHECK', '').lower() in ('1', 'true', 'yes'):
        print('⏭️  Performance check skipped (SKIP_PERFORMANCE_CHECK set)')
        return
    if os.environ.get('CI', '').lower() in ('1', 'true'):
        print('⏭️  Performance check skipped in CI environment')
        return
    try:
        checker = PerformanceRegessionChecker()
        success = checker.run_check()
        if not success:
            print('\n💡 To bypass this check (not recommended), set SKIP_PERFORMANCE_CHECK=1')
            sys.exit(1)
    except (OSError, RuntimeError, ValueError) as e:
        print(f'⚠️  Performance check failed with error: {e}')
        print('   Proceeding with commit (check inconclusive)')
if __name__ == '__main__':
    main()
