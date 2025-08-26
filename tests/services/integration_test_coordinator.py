"""
Integration test coordinator for ML testing.

This module contains integration test coordination utilities extracted from conftest.py
to maintain clean architecture and separation of concerns.
"""
import asyncio
import logging
import time
from typing import Any, Callable
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class IntegrationTestCoordinator:
    """Integration test coordinator for real behavior testing.

    Coordinates complex integration scenarios across multiple services,
    manages test data consistency, and validates end-to-end workflows.
    """
    
    def __init__(self):
        self.test_scenarios = {}
        self.scenario_results = {}
        self.cleanup_tasks = []
        self.service_states = {}

    async def register_test_scenario(
        self,
        scenario_name: str,
        services: list[str],
        setup_data: dict[str, Any],
        cleanup_data: dict[str, Any] | None = None,
    ) -> None:
        """Register a complex integration test scenario."""
        self.test_scenarios[scenario_name] = {
            "services": services,
            "setup_data": setup_data,
            "cleanup_data": cleanup_data or {},
            "registered_at": aware_utc_now().isoformat(),
        }

    async def execute_scenario(
        self,
        scenario_name: str,
        service_container,
        validation_steps: list[Callable] = None,
    ) -> dict[str, Any]:
        """Execute integration test scenario with validation."""
        if scenario_name not in self.test_scenarios:
            raise ValueError(f"Scenario not registered: {scenario_name}")
        scenario = self.test_scenarios[scenario_name]
        start_time = time.perf_counter()
        execution_result = {
            "scenario_name": scenario_name,
            "status": "running",
            "steps_completed": [],
            "validation_results": [],
            "errors": [],
        }
        try:
            for service_name in scenario["services"]:
                service = await service_container.get_service(service_name)
                if hasattr(service, "get_all_data"):
                    self.service_states[
                        f"{scenario_name}_{service_name}_initial"
                    ] = await service.get_all_data()
                setup_data = scenario["setup_data"].get(service_name, {})
                if setup_data and hasattr(service, "apply_test_data"):
                    await service.apply_test_data(setup_data)
            execution_result["steps_completed"].append("setup")
            if validation_steps:
                for i, validation_step in enumerate(validation_steps):
                    try:
                        if asyncio.iscoroutinefunction(validation_step):
                            validation_result = await validation_step(
                                service_container
                            )
                        else:
                            validation_result = validation_step(service_container)
                        execution_result["validation_results"].append({
                            "step": i,
                            "result": validation_result,
                            "status": "passed",
                        })
                    except Exception as e:
                        execution_result["validation_results"].append({
                            "step": i,
                            "error": str(e),
                            "status": "failed",
                        })
                        execution_result["errors"].append(
                            f"Validation step {i}: {e}"
                        )
            execution_result["steps_completed"].append("validation")
            execution_result["status"] = (
                "completed" if not execution_result["errors"] else "failed"
            )
        except Exception as e:
            execution_result["status"] = "error"
            execution_result["errors"].append(f"Scenario execution error: {e}")
        finally:
            cleanup_tasks = []
            for service_name in scenario["services"]:
                cleanup_data = scenario["cleanup_data"].get(service_name, {})
                if cleanup_data:
                    task = self._cleanup_service_data(
                        service_container, service_name, cleanup_data
                    )
                    cleanup_tasks.append(task)
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                execution_result["steps_completed"].append("cleanup")
        execution_result["duration_ms"] = (time.perf_counter() - start_time) * 1000
        self.scenario_results[scenario_name] = execution_result
        return execution_result

    async def _cleanup_service_data(
        self, service_container, service_name: str, cleanup_data: dict[str, Any]
    ) -> None:
        """Clean up service data after test scenario."""
        try:
            service = await service_container.get_service(service_name)
            if hasattr(service, "clear"):
                await service.clear()
            elif hasattr(service, "cleanup_test_data"):
                await service.cleanup_test_data(cleanup_data)
        except Exception as e:
            logger.warning(f"Cleanup failed for {service_name}: {e}")

    async def validate_service_consistency(
        self, service_container, consistency_checks: dict[str, Callable]
    ) -> dict[str, Any]:
        """Validate data consistency across services."""
        consistency_results = {"overall_consistent": True, "check_results": {}}
        for check_name, check_function in consistency_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_function):
                    check_result = await check_function(service_container)
                else:
                    check_result = check_function(service_container)
                consistency_results["check_results"][check_name] = {
                    "status": "passed",
                    "result": check_result,
                }
            except Exception as e:
                consistency_results["overall_consistent"] = False
                consistency_results["check_results"][check_name] = {
                    "status": "failed",
                    "error": str(e),
                }
        return consistency_results

    async def simulate_service_failure(
        self,
        service_container,
        service_name: str,
        failure_duration_seconds: int = 5,
    ) -> dict[str, Any]:
        """Simulate service failure for resilience testing."""
        service = await service_container.get_service(service_name)
        original_health = True
        if hasattr(service, "set_health_status"):
            original_health = service._is_healthy
        failure_result = {
            "service_name": service_name,
            "failure_duration_seconds": failure_duration_seconds,
            "started_at": aware_utc_now().isoformat(),
        }
        try:
            if hasattr(service, "set_health_status"):
                service.set_health_status(False)
            await asyncio.sleep(failure_duration_seconds)
            if hasattr(service, "set_health_status"):
                service.set_health_status(original_health)
            failure_result["status"] = "recovered"
            failure_result["ended_at"] = aware_utc_now().isoformat()
        except Exception as e:
            failure_result["status"] = "error"
            failure_result["error"] = str(e)
        return failure_result

    def get_scenario_summary(self) -> dict[str, Any]:
        """Get summary of all executed scenarios."""
        if not self.scenario_results:
            return {"status": "no_scenarios_executed"}
        successful_scenarios = [
            r for r in self.scenario_results.values() if r["status"] == "completed"
        ]
        failed_scenarios = [
            r
            for r in self.scenario_results.values()
            if r["status"] in ["failed", "error"]
        ]
        return {
            "total_scenarios": len(self.scenario_results),
            "successful_scenarios": len(successful_scenarios),
            "failed_scenarios": len(failed_scenarios),
            "success_rate": len(successful_scenarios)
            / len(self.scenario_results)
            * 100,
            "avg_duration_ms": sum(
                r["duration_ms"] for r in self.scenario_results.values()
            )
            / len(self.scenario_results),
            "scenario_details": self.scenario_results,
        }