#!/usr/bin/env python3
"""Test script for APES TUI dashboard."""

import asyncio
from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prompt_improver.tui.data_provider import APESDataProvider
from prompt_improver.tui.dashboard import APESDashboard


async def test_data_provider_ab_testing_real_behavior():
    """Test the data provider's A/B testing integration with real behavior."""
    print("🧪 Testing Data Provider A/B Testing Real Behavior...")

    # Mock ExperimentOrchestrator
    mock_orchestrator = MagicMock()
    mock_orchestrator.get_active_experiments = AsyncMock(
        return_value=[
            {
                "id": "exp1",
                "name": "Test Experiment 1",
                "status": "running",
                "results": {"statistical_significance": True, "effect_size": 0.15},
            }
        ]
    )
    mock_orchestrator.get_all_experiments = AsyncMock(
        return_value=[
            {"id": "exp1"},
            {"id": "exp2"},
        ]
    )

    provider = APESDataProvider()
    provider.experiment_orchestrator = mock_orchestrator

    # Get A/B testing results
    ab_data = await provider.get_ab_testing_results()

    # Assertions
    assert ab_data["active_experiments"] == 1
    assert ab_data["total_experiments"] == 2
    assert ab_data["significant_results"] == 1
    assert "Test Experiment 1" in [exp["name"] for exp in ab_data["experiments"]]
    print("✅ A/B Testing Real Behavior Test Completed!")


async def test_data_provider():
    """Test the data provider functionality."""
    print("🧪 Testing Data Provider...")

    provider = APESDataProvider()
    await provider.initialize()

    # Test system overview
    system_data = await provider.get_system_overview()
    assert "status" in system_data
    print(f"✅ System Overview: {system_data.get('status', 'unknown')}")

    # Test AutoML status
    automl_data = await provider.get_automl_status()
    assert "status" in automl_data
    print(f"✅ AutoML Status: {automl_data.get('status', 'unknown')}")

    # Test A/B testing (basic)
    ab_data = await provider.get_ab_testing_results()
    assert "active_experiments" in ab_data
    print(f"✅ A/B Testing: {ab_data.get('active_experiments', 0)} experiments")

    # Test performance metrics
    perf_data = await provider.get_performance_metrics()
    assert "response_time" in perf_data
    print(f"✅ Performance: {perf_data.get('response_time', 0):.1f}ms response time")

    # Test service status
    service_data = await provider.get_service_status()
    assert "total_services" in service_data
    print(
        f"✅ Services: {service_data.get('running_services', 0)}/{service_data.get('total_services', 0)} running"
    )

    print("✅ Data Provider Test Completed!")


def test_dashboard_dry_run():
    """Test dashboard creation without running."""
    print("🧪 Testing Dashboard Creation...")

    try:
        dashboard = APESDashboard()
        print("✅ Dashboard instance created successfully")
        print(f"✅ Dashboard title: {dashboard.title}")
        print(f"✅ CSS path: {dashboard.CSS_PATH}")
        print("✅ Dashboard Creation Test Completed!")
    except Exception as e:
        print(f"❌ Dashboard creation failed: {e}")
        return False

    return True


def run_dashboard_test():
    """Run the dashboard for a short test."""
    print("🧪 Running Dashboard Test...")

    from rich.console import Console

    console = Console()

    try:
        dashboard = APESDashboard(console)
        print("✅ Dashboard ready to run")
        print(
            "💡 To run full dashboard, use: python3 -m prompt_improver.cli interactive"
        )
        return True
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 APES TUI Dashboard Test Suite")
    print("=" * 50)

    # Test data provider
    await test_data_provider()
    print()

    # Test dashboard creation
    if not test_dashboard_dry_run():
        return False
    print()

    # Test dashboard setup
    if not run_dashboard_test():
        return False
    print()

    print("🎉 All tests passed! TUI dashboard is ready.")
    print("💡 Run with: python3 -m prompt_improver.cli interactive")

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
