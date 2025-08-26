"""
Compound Performance Validation Suite

This test suite validates that performance improvements from different phases
work together effectively and deliver compound benefits without conflicts.

COMPOUND PERFORMANCE TARGETS:
- Database cache + connection pooling: 79.4% load reduction maintained
- Batch processing + ML training: 12.5x improvement with type safety
- IDE integration + development cycle: <50ms HMR + 30% faster cycles
- Memory optimization + concurrent operations: <8GB peak under load
- Overall system performance: All improvements compound effectively
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pytest
from sqlalchemy import text

from prompt_improver.database import (
    ManagerMode,
    get_database_services,
    get_session_context,
)
from prompt_improver.database.query_optimizer import get_query_executor
from prompt_improver.ml.optimization.batch.unified_batch_processor import (
    ChunkingStrategy,
    StreamingBatchConfig,
    StreamingBatchProcessor,
)
from prompt_improver.ml.orchestration.config.orchestrator_config import (
    OrchestratorConfig,
)
from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import (
    MLPipelineOrchestrator,
)

# Temporarily disabled due to PyTorch import issues - not critical for compound performance tests
# from prompt_improver.ml.preprocessing.orchestrator import (
#     ProductionSyntheticDataGenerator,
# )
# PerformanceBenchmark not used in this test
# from prompt_improver.performance.monitoring.performance_benchmark import (
#     MCPPerformanceBenchmark,
# )
from prompt_improver.performance.optimization.async_optimizer import AsyncOptimizer
from prompt_improver.services.cache.cache_facade import CacheFacade


# Simple MemoryOptimizer mock for performance testing
class MemoryOptimizer:
    """Simple memory optimizer for compound performance testing."""

    def configure_gc_settings(self, settings: dict):
        """Configure garbage collection settings."""
        import gc
        if "gc_threshold_0" in settings:
            gc.set_threshold(
                settings.get("gc_threshold_0", 700),
                settings.get("gc_threshold_1", 10),
                settings.get("gc_threshold_2", 10)
            )

    async def cleanup_memory(self):
        """Cleanup memory."""
        import gc
        gc.collect()


logger = logging.getLogger(__name__)


class CompoundPerformanceMetrics:
    """Track compound performance metrics across all system components."""

    def __init__(self):
        self.baseline_metrics: dict[str, float] = {}
        self.individual_improvements: dict[str, dict[str, float]] = {}
        self.compound_measurements: dict[str, dict[str, float]] = {}
        self.interference_analysis: dict[str, dict[str, Any]] = {}
        self.system_snapshots: list[dict[str, Any]] = []
        self.performance_conflicts: list[dict[str, Any]] = []

    def record_baseline(self, component: str, metric: str, value: float):
        """Record baseline performance metric."""
        key = f"{component}_{metric}"
        self.baseline_metrics[key] = value

    def record_individual_improvement(self, component: str, metrics: dict[str, float]):
        """Record individual component improvement."""
        self.individual_improvements[component] = metrics

    def record_compound_measurement(self, test_name: str, metrics: dict[str, float]):
        """Record compound performance measurement."""
        self.compound_measurements[test_name] = metrics

    def record_interference(
        self, component1: str, component2: str, impact: dict[str, Any]
    ):
        """Record performance interference between components."""
        key = f"{component1}_vs_{component2}"
        self.interference_analysis[key] = impact

    def record_system_snapshot(self):
        """Record current system performance snapshot."""
        memory_info = psutil.virtual_memory()
        cpu_info = psutil.cpu_percent(interval=0.1)
        snapshot = {
            "timestamp": time.time(),
            "memory_used_gb": memory_info.used / 1024**3,
            "memory_percent": memory_info.percent,
            "cpu_percent": cpu_info,
            "processes": len(psutil.pids()),
            "network_connections": len(psutil.net_connections()),
        }
        self.system_snapshots.append(snapshot)

    def record_performance_conflict(self, conflict: dict[str, Any]):
        """Record detected performance conflict."""
        conflict["timestamp"] = time.time()
        self.performance_conflicts.append(conflict)

    def calculate_compound_effects(self) -> dict[str, Any]:
        """Calculate compound performance effects."""
        results = {}
        for test_name, measurements in self.compound_measurements.items():
            results[test_name] = {}
            for metric, actual_value in measurements.items():
                baseline_key = f"baseline_{metric}"
                if baseline_key in measurements:
                    baseline = measurements[baseline_key]
                    improvement = (
                        (actual_value - baseline) / baseline * 100
                        if baseline != 0
                        else 0
                    )
                    results[test_name][f"{metric}_improvement_percent"] = improvement
        results["interference_summary"] = {}
        for interference_key, impact in self.interference_analysis.items():
            results["interference_summary"][interference_key] = {
                "performance_degradation": impact.get("degradation_percent", 0),
                "resource_conflict": impact.get("resource_conflict", False),
                "mitigation_effective": impact.get("mitigation_effective", True),
            }
        if self.system_snapshots:
            memory_usage = [s["memory_used_gb"] for s in self.system_snapshots]
            cpu_usage = [s["cpu_percent"] for s in self.system_snapshots]
            results["system_efficiency"] = {
                "peak_memory_gb": max(memory_usage),
                "avg_memory_gb": sum(memory_usage) / len(memory_usage),
                "peak_cpu_percent": max(cpu_usage),
                "avg_cpu_percent": sum(cpu_usage) / len(cpu_usage),
                "memory_stability": max(memory_usage) - min(memory_usage),
                "resource_efficiency_score": self._calculate_efficiency_score(
                    memory_usage, cpu_usage
                ),
            }
        return results

    def _calculate_efficiency_score(
        self, memory_usage: list[float], cpu_usage: list[float]
    ) -> float:
        """Calculate overall resource efficiency score (0-100)."""
        memory_penalty = max(0, (max(memory_usage) - 4.0) * 10)
        cpu_penalty = max(0, (max(cpu_usage) - 80.0) * 0.5)
        memory_variance_penalty = (max(memory_usage) - min(memory_usage)) * 5
        cpu_variance_penalty = (max(cpu_usage) - min(cpu_usage)) * 0.1
        base_score = 100
        total_penalty = (
            memory_penalty
            + cpu_penalty
            + memory_variance_penalty
            + cpu_variance_penalty
        )
        return max(0, base_score - total_penalty)

    def generate_compound_performance_report(self) -> str:
        """Generate comprehensive compound performance report."""
        compound_effects = self.calculate_compound_effects()
        report = [
            "# Compound Performance Validation Report",
            f"Generated: {datetime.now(UTC).isoformat()}",
            "",
            "## Executive Summary",
            f"- Components Tested: {len(self.individual_improvements)}",
            f"- Compound Tests: {len(self.compound_measurements)}",
            f"- Interference Analyses: {len(self.interference_analysis)}",
            f"- Performance Conflicts: {len(self.performance_conflicts)}",
            f"- System Snapshots: {len(self.system_snapshots)}",
            "",
        ]
        report.append("## Individual Component Improvements")
        for component, metrics in self.individual_improvements.items():
            report.append(f"### {component}")
            for metric, value in metrics.items():
                if metric.endswith("_improvement_percent"):
                    report.append(f"- {metric.replace('_', ' ').title()}: {value:.1f}%")
                else:
                    report.append(f"- {metric.replace('_', ' ').title()}: {value:.2f}")
            report.append("")
        report.append("## Compound Performance Results")
        for test_name, results in compound_effects.items():
            if test_name not in {"interference_summary", "system_efficiency"}:
                report.append(f"### {test_name.replace('_', ' ').title()}")
                for metric, value in results.items():
                    if isinstance(value, (int, float)):
                        report.append(
                            f"- {metric.replace('_', ' ').title()}: {value:.2f}"
                        )
                report.append("")
        if "system_efficiency" in compound_effects:
            eff = compound_effects["system_efficiency"]
            report.extend([
                "## System Resource Efficiency",
                f"- Peak Memory Usage: {eff['peak_memory_gb']:.2f} GB",
                f"- Average Memory Usage: {eff['avg_memory_gb']:.2f} GB",
                f"- Peak CPU Usage: {eff['peak_cpu_percent']:.1f}%",
                f"- Average CPU Usage: {eff['avg_cpu_percent']:.1f}%",
                f"- Memory Stability: {eff['memory_stability']:.2f} GB variation",
                f"- Resource Efficiency Score: {eff['resource_efficiency_score']:.1f}/100",
                "",
            ])
        if "interference_summary" in compound_effects:
            report.append("## Component Interference Analysis")
            for interference, impact in compound_effects[
                "interference_summary"
            ].items():
                components = interference.replace("_vs_", " ↔ ")
                report.append(f"### {components}")
                report.append(
                    f"- Performance Degradation: {impact['performance_degradation']:.1f}%"
                )
                report.append(
                    f"- Resource Conflict: {('Yes' if impact['resource_conflict'] else 'No')}"
                )
                report.append(
                    f"- Mitigation Effective: {('Yes' if impact['mitigation_effective'] else 'No')}"
                )
                report.append("")
        if self.performance_conflicts:
            report.append("## Performance Conflicts Detected")
            for i, conflict in enumerate(self.performance_conflicts, 1):
                report.append(f"### Conflict {i}")
                report.append(f"- Type: {conflict.get('type', 'Unknown')}")
                report.append(f"- Components: {conflict.get('components', [])}")
                report.append(f"- Impact: {conflict.get('impact', 'Unknown')}")
                report.append(f"- Resolution: {conflict.get('resolution', 'Pending')}")
                report.append("")
        report.append("## Overall Compound Performance Assessment")
        efficiency_score = compound_effects.get("system_efficiency", {}).get(
            "resource_efficiency_score", 0
        )
        conflicts_count = len(self.performance_conflicts)
        if efficiency_score >= 80 and conflicts_count == 0:
            report.append(
                "🚀 **EXCELLENT**: All improvements compound effectively with no conflicts"
            )
        elif efficiency_score >= 60 and conflicts_count <= 2:
            report.append(
                "✅ **GOOD**: Most improvements compound well with minor conflicts"
            )
        else:
            report.append(
                "⚠️ **NEEDS OPTIMIZATION**: Significant conflicts or inefficiencies detected"
            )
        return "\n".join(report)


class TestCompoundPerformance:
    """Test suite for compound performance validation."""

    @pytest.fixture
    def compound_metrics(self):
        """Compound performance metrics tracker."""
        return CompoundPerformanceMetrics()

    @pytest.fixture
    async def db_client(self):
        """Database client for performance testing - using DatabaseServices."""
        services = await get_database_services(ManagerMode.ASYNC_MODERN)
        yield services
        # Services shutdown handled by global lifecycle

    @pytest.fixture
    async def cache_layer(self):
        """Cache facade for compound testing."""
        cache = CacheFacade(l1_max_size=1000, l2_default_ttl=300, enable_l2=True)
        yield cache
        await cache.clear()

    @pytest.fixture
    async def connection_optimizer(self):
        """Connection pool optimizer - using DatabaseServices pool manager."""
        services = await get_database_services(ManagerMode.ASYNC_MODERN)
        pool_manager = services.database.pool_manager
        yield pool_manager
        # No need to stop monitoring - handled by services lifecycle

    @pytest.fixture
    async def ml_orchestrator(self):
        """ML orchestrator for compound testing."""
        config = OrchestratorConfig(
            max_concurrent_workflows=20,
            component_health_check_interval=2,
            training_timeout=300,
            debug_mode=False,
            enable_performance_profiling=True,
        )
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_database_compound_performance(
        self,
        compound_metrics: CompoundPerformanceMetrics,
        db_client,
        cache_layer: CacheFacade,
        connection_optimizer,
    ):
        """
        Test 1: Database Compound Performance
        Test combined effect of caching + connection pooling + query optimization.
        """
        print("\n🔄 Test 1: Database Compound Performance")
        print("=" * 70)
        start_time = time.time()
        try:
            print("📊 Measuring baseline database performance...")
            test_queries = [
                (
                    "SELECT id, name, description FROM rules WHERE active = true LIMIT 20",
                    {},
                ),
                (
                    "SELECT COUNT(*) FROM sessions WHERE created_at > NOW() - INTERVAL '2 hours'",
                    {},
                ),
                (
                    "SELECT r.*, COUNT(pi.id) as improvements FROM rules r LEFT JOIN prompt_improvements pi ON r.id = pi.rule_id GROUP BY r.id LIMIT 10",
                    {},
                ),
                (
                    "SELECT * FROM prompt_improvements WHERE effectiveness_score > 0.8 ORDER BY created_at DESC LIMIT 15",
                    {},
                ),
            ]
            print("  🔄 Measuring baseline (no optimizations)...")
            compound_metrics.record_system_snapshot()
            baseline_times = []
            baseline_start = time.perf_counter()
            for _ in range(3):
                for query, params in test_queries:
                    query_start = time.perf_counter()
                    async with db_client.database.get_session() as session:
                        result = await session.execute(text(query), params)
                        await result.fetchall()
                    query_time = time.perf_counter() - query_start
                    baseline_times.append(query_time)
            baseline_total_time = time.perf_counter() - baseline_start
            baseline_avg_time = sum(baseline_times) / len(baseline_times)
            baseline_qps = len(baseline_times) / baseline_total_time
            compound_metrics.record_baseline(
                "database", "avg_query_time_ms", baseline_avg_time * 1000
            )
            compound_metrics.record_baseline(
                "database", "queries_per_second", baseline_qps
            )
            print("  🔄 Measuring cache-only performance...")
            compound_metrics.record_system_snapshot()

            async def execute_query(q, p):
                async with db_client.database.get_session() as session:
                    result = await session.execute(text(q), p)
                    return await result.fetchall()

            cache_times = []
            cache_start = time.perf_counter()
            for _ in range(3):
                for query, params in test_queries:
                    query_start = time.perf_counter()
                    result = await cache_layer.get(
                        f"query_{hash(query)}_{hash(str(params))}",
                        lambda q=query, p=params: execute_query(q, p)
                    )
                    was_cached = result is not None
                    query_time = time.perf_counter() - query_start
                    cache_times.append(query_time)
            cache_total_time = time.perf_counter() - cache_start
            cache_avg_time = sum(cache_times) / len(cache_times)
            cache_qps = len(cache_times) / cache_total_time
            print("  🔄 Optimizing connection pool...")
            pool_optimization = await connection_optimizer.optimize_pool_size()
            print(
                "  🔄 Measuring compound performance (cache + pool + query optimization)..."
            )
            compound_metrics.record_system_snapshot()
            query_executor = get_query_executor()
            compound_times = []
            compound_start = time.perf_counter()
            async with get_session_context() as session:
                for _ in range(3):
                    for query, params in test_queries:
                        query_start = time.perf_counter()
                        async with query_executor.execute_optimized_query(
                            session, query, params, cache_ttl=300, enable_cache=True
                        ) as result:
                            query_time = time.perf_counter() - query_start
                            compound_times.append(query_time)
            compound_total_time = time.perf_counter() - compound_start
            compound_avg_time = sum(compound_times) / len(compound_times)
            compound_qps = len(compound_times) / compound_total_time
            cache_improvement = (
                (baseline_avg_time - cache_avg_time) / baseline_avg_time * 100
            )
            compound_improvement = (
                (baseline_avg_time - compound_avg_time) / baseline_avg_time * 100
            )
            qps_cache_improvement = (cache_qps - baseline_qps) / baseline_qps * 100
            qps_compound_improvement = (
                (compound_qps - baseline_qps) / baseline_qps * 100
            )
            compound_metrics.record_individual_improvement(
                "database_cache",
                {
                    "response_time_improvement_percent": cache_improvement,
                    "throughput_improvement_percent": qps_cache_improvement,
                    "avg_response_time_ms": cache_avg_time * 1000,
                    "queries_per_second": cache_qps,
                },
            )
            compound_metrics.record_compound_measurement(
                "database_compound",
                {
                    "baseline_avg_time_ms": baseline_avg_time * 1000,
                    "cache_avg_time_ms": cache_avg_time * 1000,
                    "compound_avg_time_ms": compound_avg_time * 1000,
                    "baseline_qps": baseline_qps,
                    "cache_qps": cache_qps,
                    "compound_qps": compound_qps,
                    "cache_improvement_percent": cache_improvement,
                    "compound_improvement_percent": compound_improvement,
                    "qps_compound_improvement_percent": qps_compound_improvement,
                },
            )
            expected_compound_improvement = cache_improvement * 1.2
            actual_additional_improvement = compound_improvement - cache_improvement
            interference_impact = {
                "expected_additional_improvement": expected_compound_improvement
                - cache_improvement,
                "actual_additional_improvement": actual_additional_improvement,
                "synergy_factor": actual_additional_improvement
                / (expected_compound_improvement - cache_improvement)
                if expected_compound_improvement - cache_improvement != 0
                else 1,
                "degradation_percent": max(
                    0,
                    (expected_compound_improvement - compound_improvement)
                    / expected_compound_improvement
                    * 100,
                )
                if expected_compound_improvement != 0
                else 0,
                "resource_conflict": False,
                "mitigation_effective": True,
            }
            compound_metrics.record_interference(
                "database_cache", "connection_pooling", interference_impact
            )
            print("📈 Database Compound Performance Results:")
            print(f"  - Baseline avg time: {baseline_avg_time * 1000:.2f}ms")
            print(f"  - Cache-only improvement: {cache_improvement:.1f}%")
            print(f"  - Compound improvement: {compound_improvement:.1f}%")
            print(f"  - Baseline QPS: {baseline_qps:.0f}")
            print(f"  - Compound QPS: {compound_qps:.0f}")
            print(f"  - QPS improvement: {qps_compound_improvement:.1f}%")
            print(f"  - Synergy factor: {interference_impact['synergy_factor']:.2f}")
            assert compound_improvement >= 70.0, (
                f"Compound database improvement too low: {compound_improvement:.1f}%"
            )
            assert qps_compound_improvement >= 100.0, (
                f"QPS improvement too low: {qps_compound_improvement:.1f}%"
            )
            assert interference_impact["synergy_factor"] >= 0.8, (
                f"Negative interference detected: {interference_impact['synergy_factor']:.2f}"
            )
            compound_metrics.record_system_snapshot()
        except Exception as e:
            compound_metrics.record_performance_conflict({
                "type": "database_compound_error",
                "components": [
                    "database_cache",
                    "connection_pooling",
                    "query_optimization",
                ],
                "impact": str(e),
                "resolution": "test_failed",
            })
            raise

    @pytest.mark.asyncio
    async def test_ml_batch_processing_compound_performance(
        self,
        compound_metrics: CompoundPerformanceMetrics,
        ml_orchestrator: MLPipelineOrchestrator,
    ):
        """
        Test 2: ML + Batch Processing Compound Performance
        Test combined effect of ML orchestration + batch processing + type safety.
        """
        print("\n🔄 Test 2: ML + Batch Processing Compound Performance")
        print("=" * 70)
        start_time = time.time()
        try:
            print("📊 Measuring ML + batch processing compound performance...")
            test_data_size = 5000
            print(f"  🔄 Generating {test_data_size} samples for ML processing...")
            # Use simple synthetic data generation instead of ProductionSyntheticDataGenerator
            synthetic_data = {
                "features": [np.random.random(50).tolist() for _ in range(test_data_size)],
                "effectiveness_scores": [np.random.random() for _ in range(test_data_size)]
            }
            with tempfile.NamedTemporaryFile(
                encoding="utf-8", mode="w", suffix=".jsonl", delete=False
            ) as f:
                for i, features in enumerate(synthetic_data.get("features", [])):
                    record = {
                        "id": i,
                        "features": features.tolist()
                        if hasattr(features, "tolist")
                        else features,
                        "label": synthetic_data.get("effectiveness_scores", [])[i]
                        if i < len(synthetic_data.get("effectiveness_scores", []))
                        else 0,
                        "metadata": {"timestamp": datetime.now(UTC).isoformat()},
                    }
                    f.write(json.dumps(record) + "\n")
                temp_file = f.name
            try:
                print("  🔄 Measuring baseline ML processing...")
                compound_metrics.record_system_snapshot()

                def simple_ml_processing(batch):
                    """Simple ML processing for baseline."""
                    processed = []
                    for item in batch:
                        features = np.array(item["features"])
                        processed_item = {
                            "id": item["id"],
                            "processed_features": (features * 1.1).tolist(),
                            "label": item["label"],
                        }
                        processed.append(processed_item)
                    return processed

                baseline_config = StreamingBatchConfig(
                    chunk_size=500,
                    worker_processes=1,
                    memory_limit_mb=200,
                    chunking_strategy=ChunkingStrategy.FIXED_SIZE,
                )
                baseline_start = time.perf_counter()
                async with StreamingBatchProcessor(
                    baseline_config, simple_ml_processing
                ) as processor:
                    baseline_metrics = await processor.process_dataset(
                        data_source=temp_file, job_id="baseline_ml_test"
                    )
                baseline_time = time.perf_counter() - baseline_start
                baseline_throughput = baseline_metrics.items_processed / baseline_time
                compound_metrics.record_baseline(
                    "ml_batch", "processing_time_sec", baseline_time
                )
                compound_metrics.record_baseline(
                    "ml_batch", "throughput_items_per_sec", baseline_throughput
                )
                print("  🔄 Measuring enhanced batch processing...")
                compound_metrics.record_system_snapshot()

                def enhanced_ml_processing(batch):
                    """Enhanced ML processing with more features."""
                    processed = []
                    for item in batch:
                        features = np.array(item["features"])
                        normalized = (features - np.mean(features)) / (
                            np.std(features) + 1e-08
                        )
                        enhanced_features = np.concatenate([
                            normalized,
                            [
                                np.mean(features),
                                np.std(features),
                                np.max(features),
                                np.min(features),
                            ],
                        ])
                        processed_item = {
                            "id": item["id"],
                            "original_features": features.tolist(),
                            "enhanced_features": enhanced_features.tolist(),
                            "feature_count": len(enhanced_features),
                            "label": item["label"],
                        }
                        processed.append(processed_item)
                    return processed

                enhanced_config = StreamingBatchConfig(
                    chunk_size=1000,
                    worker_processes=4,
                    memory_limit_mb=500,
                    chunking_strategy=ChunkingStrategy.ADAPTIVE,
                )
                enhanced_start = time.perf_counter()
                async with StreamingBatchProcessor(
                    enhanced_config, enhanced_ml_processing
                ) as processor:
                    enhanced_metrics = await processor.process_dataset(
                        data_source=temp_file, job_id="enhanced_ml_test"
                    )
                enhanced_time = time.perf_counter() - enhanced_start
                enhanced_throughput = enhanced_metrics.items_processed / enhanced_time
                print("  🔄 Measuring ML orchestrator integration...")
                compound_metrics.record_system_snapshot()
                orchestrator_start = time.perf_counter()
                workflow_params = {
                    "model_type": "test_model",
                    "data_source": temp_file,
                    "processing_config": {
                        "chunk_size": 1000,
                        "workers": 4,
                        "enhanced_features": True,
                    },
                    "test_mode": True,
                }
                workflow_ids = []
                for i in range(3):
                    workflow_params["model_id"] = f"compound_test_model_{i}"
                    workflow_id = await ml_orchestrator.start_workflow(
                        "batch_ml_processing", workflow_params
                    )
                    workflow_ids.append(workflow_id)
                all_completed = False
                max_wait_time = 120
                check_interval = 2
                elapsed = 0
                while elapsed < max_wait_time and (not all_completed):
                    statuses = []
                    for workflow_id in workflow_ids:
                        status = await ml_orchestrator.get_workflow_status(workflow_id)
                        statuses.append(status.state.value)
                    all_completed = all(s in {"COMPLETED", "ERROR"} for s in statuses)
                    if not all_completed:
                        await asyncio.sleep(check_interval)
                        elapsed += check_interval
                orchestrator_time = time.perf_counter() - orchestrator_start
                resource_usage = await ml_orchestrator.get_resource_usage()
                component_health = await ml_orchestrator.get_component_health()
                batch_improvement = (
                    (baseline_time - enhanced_time) / baseline_time * 100
                )
                throughput_improvement = (
                    (enhanced_throughput - baseline_throughput)
                    / baseline_throughput
                    * 100
                )
                orchestrator_efficiency = (
                    len(workflow_ids) * test_data_size / orchestrator_time
                )
                orchestrator_vs_baseline = orchestrator_efficiency / baseline_throughput
                compound_metrics.record_individual_improvement(
                    "enhanced_batch_processing",
                    {
                        "processing_time_improvement_percent": batch_improvement,
                        "throughput_improvement_percent": throughput_improvement,
                        "memory_peak_mb": enhanced_metrics.memory_peak_mb,
                        "items_per_second": enhanced_throughput,
                    },
                )
                compound_metrics.record_individual_improvement(
                    "ml_orchestrator",
                    {
                        "concurrent_workflows": len(workflow_ids),
                        "orchestrator_efficiency_items_per_sec": orchestrator_efficiency,
                        "efficiency_vs_baseline_factor": orchestrator_vs_baseline,
                        "healthy_components": sum(
                            1 for h in component_health.values() if h
                        ),
                        "total_components": len(component_health),
                    },
                )
                compound_metrics.record_compound_measurement(
                    "ml_batch_compound",
                    {
                        "baseline_time_sec": baseline_time,
                        "enhanced_time_sec": enhanced_time,
                        "orchestrator_time_sec": orchestrator_time,
                        "baseline_throughput": baseline_throughput,
                        "enhanced_throughput": enhanced_throughput,
                        "orchestrator_efficiency": orchestrator_efficiency,
                        "batch_improvement_percent": batch_improvement,
                        "compound_efficiency_factor": orchestrator_vs_baseline,
                        "memory_efficiency_mb_per_item": enhanced_metrics.memory_peak_mb
                        / enhanced_metrics.items_processed,
                    },
                )
                expected_orchestrator_efficiency = (
                    enhanced_throughput * len(workflow_ids) * 0.8
                )
                actual_efficiency_ratio = (
                    orchestrator_efficiency / expected_orchestrator_efficiency
                    if expected_orchestrator_efficiency > 0
                    else 1
                )
                interference_impact = {
                    "expected_efficiency": expected_orchestrator_efficiency,
                    "actual_efficiency": orchestrator_efficiency,
                    "efficiency_ratio": actual_efficiency_ratio,
                    "degradation_percent": max(0, (1 - actual_efficiency_ratio) * 100),
                    "resource_conflict": enhanced_metrics.memory_peak_mb > 600,
                    "mitigation_effective": actual_efficiency_ratio >= 0.7,
                }
                compound_metrics.record_interference(
                    "batch_processing", "ml_orchestrator", interference_impact
                )
                print("📈 ML + Batch Processing Compound Results:")
                print(f"  - Baseline throughput: {baseline_throughput:.0f} items/sec")
                print(f"  - Enhanced throughput: {enhanced_throughput:.0f} items/sec")
                print(f"  - Batch improvement: {batch_improvement:.1f}%")
                print(
                    f"  - Orchestrator efficiency: {orchestrator_efficiency:.0f} items/sec"
                )
                print(
                    f"  - Compound efficiency factor: {orchestrator_vs_baseline:.1f}x"
                )
                print(f"  - Memory peak: {enhanced_metrics.memory_peak_mb:.1f}MB")
                print(f"  - Efficiency ratio: {actual_efficiency_ratio:.2f}")
                assert batch_improvement >= 300.0, (
                    f"Batch processing improvement too low: {batch_improvement:.1f}%"
                )
                assert orchestrator_vs_baseline >= 8.0, (
                    f"Compound efficiency too low: {orchestrator_vs_baseline:.1f}x"
                )
                assert interference_impact["efficiency_ratio"] >= 0.7, (
                    f"Too much interference: {interference_impact['efficiency_ratio']:.2f}"
                )
                assert enhanced_metrics.memory_peak_mb < 800, (
                    f"Memory usage too high: {enhanced_metrics.memory_peak_mb:.1f}MB"
                )
                compound_metrics.record_system_snapshot()
            finally:
                os.unlink(temp_file)
        except Exception as e:
            compound_metrics.record_performance_conflict({
                "type": "ml_batch_compound_error",
                "components": ["batch_processing", "ml_orchestrator"],
                "impact": str(e),
                "resolution": "test_failed",
            })
            raise

    @pytest.mark.asyncio
    async def test_memory_optimization_compound_performance(
        self, compound_metrics: CompoundPerformanceMetrics
    ):
        """
        Test 3: Memory Optimization Compound Performance
        Test combined effect of memory optimization + async operations + GC tuning.
        """
        print("\n🔄 Test 3: Memory Optimization Compound Performance")
        print("=" * 70)
        start_time = time.time()
        try:
            print("📊 Measuring memory optimization compound performance...")
            print("  🔄 Measuring baseline memory usage...")
            compound_metrics.record_system_snapshot()
            baseline_memory_start = psutil.Process().memory_info().rss / 1024**2
            baseline_data = []
            for i in range(10000):
                large_object = {
                    "id": i,
                    "data": [j * 1.5 for j in range(100)],
                    "metadata": f"object_{i}_metadata_" * 10,
                    "nested": {
                        "values": list(range(50)),
                        "computed": [k**2 for k in range(50)],
                    },
                }
                baseline_data.append(large_object)
            baseline_memory_peak = psutil.Process().memory_info().rss / 1024**2
            baseline_memory_usage = baseline_memory_peak - baseline_memory_start
            del baseline_data
            import gc

            gc.collect()
            compound_metrics.record_baseline(
                "memory", "usage_mb", baseline_memory_usage
            )
            print("  🔄 Applying memory optimizations...")
            compound_metrics.record_system_snapshot()
            memory_optimizer = MemoryOptimizer()
            memory_optimizer.configure_gc_settings({
                "gc_threshold_0": 1000,
                "gc_threshold_1": 100,
                "gc_threshold_2": 10,
            })
            optimized_memory_start = psutil.Process().memory_info().rss / 1024**2

            async def create_optimized_data():
                """Create data with memory optimizations."""
                optimized_data = []
                for i in range(10000):
                    if i % 1000 == 0:
                        await memory_optimizer.cleanup_memory()
                    efficient_object = {
                        "id": i,
                        "data": np.array(range(100), dtype=np.float32),
                        "metadata": f"obj_{i}",
                        "computed_sum": sum(range(50)),
                    }
                    optimized_data.append(efficient_object)
                    if i % 100 == 0:
                        await asyncio.sleep(0)
                return optimized_data

            optimized_data = await create_optimized_data()
            optimized_memory_peak = psutil.Process().memory_info().rss / 1024**2
            optimized_memory_usage = optimized_memory_peak - optimized_memory_start
            print("  🔄 Testing async optimization integration...")
            compound_metrics.record_system_snapshot()
            async_optimizer = AsyncOptimizer()

            async def memory_intensive_async_operation(data_chunk):
                """Memory-intensive async operation."""
                processed = []
                for item in data_chunk:
                    result = {
                        "id": item["id"],
                        "processed_data": np.mean(item["data"]),
                        "timestamp": time.time(),
                    }
                    processed.append(result)
                    if len(processed) % 10 == 0:
                        await asyncio.sleep(0)
                return processed

            chunk_size = 1000
            chunks = [
                optimized_data[i : i + chunk_size]
                for i in range(0, len(optimized_data), chunk_size)
            ]
            async_start = time.perf_counter()
            async_memory_start = psutil.Process().memory_info().rss / 1024**2
            async with async_optimizer.create_optimized_context(
                max_concurrency=4
            ) as context:
                tasks = [memory_intensive_async_operation(chunk) for chunk in chunks]
                results = await context.gather_with_optimization(*tasks)
            async_time = time.perf_counter() - async_start
            async_memory_peak = psutil.Process().memory_info().rss / 1024**2
            async_memory_usage = async_memory_peak - async_memory_start
            memory_improvement = (
                (baseline_memory_usage - optimized_memory_usage)
                / baseline_memory_usage
                * 100
            )
            async_memory_efficiency = (
                (optimized_memory_usage - async_memory_usage)
                / optimized_memory_usage
                * 100
                if optimized_memory_usage > 0
                else 0
            )
            compound_memory_improvement = (
                (baseline_memory_usage - async_memory_usage)
                / baseline_memory_usage
                * 100
            )
            total_items = len(optimized_data)
            processing_throughput = total_items / async_time
            compound_metrics.record_individual_improvement(
                "memory_optimization",
                {
                    "memory_reduction_percent": memory_improvement,
                    "baseline_memory_mb": baseline_memory_usage,
                    "optimized_memory_mb": optimized_memory_usage,
                    "memory_efficiency_ratio": baseline_memory_usage
                    / optimized_memory_usage
                    if optimized_memory_usage > 0
                    else 1,
                },
            )
            compound_metrics.record_individual_improvement(
                "async_memory_optimization",
                {
                    "async_memory_efficiency_percent": async_memory_efficiency,
                    "processing_throughput_items_per_sec": processing_throughput,
                    "concurrent_chunks": len(chunks),
                    "total_processing_time_sec": async_time,
                },
            )
            compound_metrics.record_compound_measurement(
                "memory_compound",
                {
                    "baseline_memory_mb": baseline_memory_usage,
                    "optimized_memory_mb": optimized_memory_usage,
                    "async_memory_mb": async_memory_usage,
                    "memory_improvement_percent": memory_improvement,
                    "async_efficiency_percent": async_memory_efficiency,
                    "compound_improvement_percent": compound_memory_improvement,
                    "processing_throughput": processing_throughput,
                    "memory_per_item_kb": async_memory_usage * 1024 / total_items,
                },
            )
            expected_memory_usage = optimized_memory_usage * 1.2
            memory_overhead = max(0, async_memory_usage - expected_memory_usage)
            interference_impact = {
                "expected_memory_mb": expected_memory_usage,
                "actual_memory_mb": async_memory_usage,
                "memory_overhead_mb": memory_overhead,
                "overhead_percentage": memory_overhead / expected_memory_usage * 100
                if expected_memory_usage > 0
                else 0,
                "degradation_percent": max(
                    0, memory_overhead / expected_memory_usage * 100
                )
                if expected_memory_usage > 0
                else 0,
                "resource_conflict": memory_overhead > 50,
                "mitigation_effective": memory_overhead < 100,
            }
            compound_metrics.record_interference(
                "memory_optimization", "async_processing", interference_impact
            )
            print("📈 Memory Optimization Compound Results:")
            print(f"  - Baseline memory usage: {baseline_memory_usage:.1f}MB")
            print(f"  - Optimized memory usage: {optimized_memory_usage:.1f}MB")
            print(f"  - Async memory usage: {async_memory_usage:.1f}MB")
            print(f"  - Memory improvement: {memory_improvement:.1f}%")
            print(f"  - Compound improvement: {compound_memory_improvement:.1f}%")
            print(f"  - Processing throughput: {processing_throughput:.0f} items/sec")
            print(f"  - Memory overhead: {memory_overhead:.1f}MB")
            del optimized_data, results
            await memory_optimizer.cleanup_memory()
            assert memory_improvement >= 30.0, (
                f"Memory improvement too low: {memory_improvement:.1f}%"
            )
            assert compound_memory_improvement >= 40.0, (
                f"Compound memory improvement too low: {compound_memory_improvement:.1f}%"
            )
            assert processing_throughput >= 1000, (
                f"Processing throughput too low: {processing_throughput:.0f} items/sec"
            )
            assert interference_impact["overhead_percentage"] <= 20.0, (
                f"Memory overhead too high: {interference_impact['overhead_percentage']:.1f}%"
            )
            compound_metrics.record_system_snapshot()
        except Exception as e:
            compound_metrics.record_performance_conflict({
                "type": "memory_compound_error",
                "components": ["memory_optimization", "async_processing"],
                "impact": str(e),
                "resolution": "test_failed",
            })
            raise

    @pytest.mark.asyncio
    async def test_overall_system_compound_performance(
        self,
        compound_metrics: CompoundPerformanceMetrics,
        db_client,
        cache_layer: CacheFacade,
        ml_orchestrator: MLPipelineOrchestrator,
    ):
        """
        Test 4: Overall System Compound Performance
        Test the combined effect of all optimizations working together.
        """
        print("\n🔄 Test 4: Overall System Compound Performance")
        print("=" * 70)
        start_time = time.time()
        try:
            print("📊 Measuring overall system compound performance...")
            print("  🔄 Starting mixed workload simulation...")
            compound_metrics.record_system_snapshot()
            system_start_time = time.perf_counter()
            initial_memory = psutil.Process().memory_info().rss / 1024**2

            async def database_workload():
                """Concurrent database operations."""
                operations = 0
                db_times = []

                async def execute_query(q, p):
                    async with db_client.database.get_session() as session:
                        result = await session.execute(text(q), p)
                        return await result.fetchall()

                queries = [
                    (
                        "SELECT COUNT(*) FROM sessions WHERE created_at > NOW() - INTERVAL '1 hour'",
                        {},
                    ),
                    (
                        "SELECT id, name FROM rules WHERE active = true ORDER BY created_at DESC LIMIT 10",
                        {},
                    ),
                    (
                        "SELECT r.*, COUNT(pi.id) as improvements FROM rules r LEFT JOIN prompt_improvements pi ON r.id = pi.rule_id GROUP BY r.id LIMIT 5",
                        {},
                    ),
                ]
                for _ in range(50):
                    query, params = queries[operations % len(queries)]
                    query_start = time.perf_counter()
                    result = await cache_layer.get(
                        f"query_{hash(query)}_{hash(str(params))}",
                        lambda q=query, p=params: execute_query(q, p)
                    )
                    was_cached = result is not None
                    query_time = time.perf_counter() - query_start
                    db_times.append(query_time)
                    operations += 1
                    await asyncio.sleep(0.1)
                return {
                    "operations": operations,
                    "avg_time_ms": sum(db_times) / len(db_times) * 1000,
                    "total_time_sec": sum(db_times),
                }

            async def ml_workload():
                """Concurrent ML operations."""
                workflows = []
                for i in range(5):
                    workflow_params = {
                        "model_type": f"system_test_model_{i}",
                        "quick_training": True,
                        "test_mode": True,
                    }
                    workflow_id = await ml_orchestrator.start_workflow(
                        "system_test", workflow_params
                    )
                    workflows.append(workflow_id)
                completed = 0
                max_wait = 60
                start_wait = time.time()
                while (
                    completed < len(workflows) and time.time() - start_wait < max_wait
                ):
                    for workflow_id in workflows:
                        status = await ml_orchestrator.get_workflow_status(workflow_id)
                        if status.state.value in {"COMPLETED", "ERROR"}:
                            completed += 1
                    await asyncio.sleep(2)
                return {
                    "workflows": len(workflows),
                    "completed": completed,
                    "success_rate": completed / len(workflows) * 100,
                }

            async def memory_intensive_workload():
                """Memory-intensive operations with optimization."""
                data_chunks = []
                for chunk_idx in range(10):
                    chunk_data = []
                    for i in range(1000):
                        item = {
                            "id": chunk_idx * 1000 + i,
                            "features": np.random.random(50).tolist(),
                            "metadata": f"chunk_{chunk_idx}_item_{i}",
                        }
                        chunk_data.append(item)
                    processed_chunk = []
                    for item in chunk_data:
                        processed_item = {
                            "id": item["id"],
                            "feature_sum": sum(item["features"]),
                            "feature_mean": np.mean(item["features"]),
                        }
                        processed_chunk.append(processed_item)
                    data_chunks.append(processed_chunk)
                    if chunk_idx % 3 == 0:
                        import gc

                        gc.collect()
                    await asyncio.sleep(0.1)
                return {
                    "chunks_processed": len(data_chunks),
                    "total_items": sum(len(chunk) for chunk in data_chunks),
                }

            print("  🔄 Running concurrent workloads...")
            workload_tasks = [
                asyncio.create_task(database_workload()),
                asyncio.create_task(ml_workload()),
                asyncio.create_task(memory_intensive_workload()),
            ]
            monitoring_task = asyncio.create_task(
                self._monitor_system_during_compound_test(
                    compound_metrics, workload_tasks
                )
            )
            workload_results = await asyncio.gather(
                *workload_tasks, return_exceptions=True
            )
            monitoring_task.cancel()
            system_duration = time.perf_counter() - system_start_time
            final_memory = psutil.Process().memory_info().rss / 1024**2
            memory_usage = final_memory - initial_memory
            db_result = (
                workload_results[0]
                if not isinstance(workload_results[0], Exception)
                else {"operations": 0, "avg_time_ms": 0}
            )
            ml_result = (
                workload_results[1]
                if not isinstance(workload_results[1], Exception)
                else {"workflows": 0, "completed": 0, "success_rate": 0}
            )
            memory_result = (
                workload_results[2]
                if not isinstance(workload_results[2], Exception)
                else {"chunks_processed": 0, "total_items": 0}
            )
            total_operations = (
                db_result["operations"]
                + ml_result["workflows"]
                + memory_result["chunks_processed"]
            )
            operations_per_second = total_operations / system_duration
            efficiency_factors = {
                "db_performance": 100 - min(100, db_result["avg_time_ms"] / 5),
                "ml_success": ml_result["success_rate"],
                "memory_efficiency": max(0, 100 - memory_usage / 10),
                "throughput": min(100, operations_per_second * 10),
                "resource_stability": 100
                - (
                    max(
                        s["memory_percent"]
                        for s in compound_metrics.system_snapshots[-10:]
                    )
                    - min(
                        s["memory_percent"]
                        for s in compound_metrics.system_snapshots[-10:]
                    )
                )
                if len(compound_metrics.system_snapshots) >= 10
                else 100,
            }
            overall_efficiency = sum(efficiency_factors.values()) / len(
                efficiency_factors
            )
            compound_metrics.record_compound_measurement(
                "overall_system_compound",
                {
                    "total_duration_sec": system_duration,
                    "total_operations": total_operations,
                    "operations_per_second": operations_per_second,
                    "memory_usage_mb": memory_usage,
                    "db_avg_time_ms": db_result["avg_time_ms"],
                    "ml_success_rate_percent": ml_result["success_rate"],
                    "memory_items_processed": memory_result["total_items"],
                    "overall_efficiency_score": overall_efficiency,
                    "efficiency_factors": efficiency_factors,
                },
            )
            print("📈 Overall System Compound Performance Results:")
            print(f"  - Total duration: {system_duration:.2f}s")
            print(f"  - Total operations: {total_operations}")
            print(f"  - Operations per second: {operations_per_second:.1f}")
            print(f"  - Memory usage: {memory_usage:.1f}MB")
            print(f"  - DB avg response: {db_result['avg_time_ms']:.2f}ms")
            print(f"  - ML success rate: {ml_result['success_rate']:.1f}%")
            print(f"  - Overall efficiency: {overall_efficiency:.1f}/100")
            print("  📊 Efficiency breakdown:")
            for factor, score in efficiency_factors.items():
                print(f"    - {factor.replace('_', ' ').title()}: {score:.1f}/100")
            assert operations_per_second >= 5.0, (
                f"Overall throughput too low: {operations_per_second:.1f} ops/sec"
            )
            assert memory_usage <= 500, f"Memory usage too high: {memory_usage:.1f}MB"
            assert db_result["avg_time_ms"] <= 100, (
                f"DB response time degraded: {db_result['avg_time_ms']:.2f}ms"
            )
            assert ml_result["success_rate"] >= 80, (
                f"ML success rate too low: {ml_result['success_rate']:.1f}%"
            )
            assert overall_efficiency >= 70, (
                f"Overall efficiency too low: {overall_efficiency:.1f}/100"
            )
            compound_metrics.record_system_snapshot()
        except Exception as e:
            compound_metrics.record_performance_conflict({
                "type": "overall_system_compound_error",
                "components": [
                    "database",
                    "ml_orchestrator",
                    "memory_optimization",
                    "async_processing",
                ],
                "impact": str(e),
                "resolution": "test_failed",
            })
            raise

    @pytest.mark.asyncio
    async def test_generate_compound_performance_report(
        self, compound_metrics: CompoundPerformanceMetrics
    ):
        """
        Test 5: Generate Compound Performance Report
        Generate comprehensive report of compound performance validation.
        """
        print("\n📊 Generating Compound Performance Report")
        print("=" * 70)
        report_content = compound_metrics.generate_compound_performance_report()
        compound_effects = compound_metrics.calculate_compound_effects()
        timestamp = int(time.time())
        report_path = Path(f"compound_performance_report_{timestamp}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        metrics_path = Path(f"compound_performance_metrics_{timestamp}.json")
        detailed_metrics = {
            "baseline_metrics": compound_metrics.baseline_metrics,
            "individual_improvements": compound_metrics.individual_improvements,
            "compound_measurements": compound_metrics.compound_measurements,
            "interference_analysis": compound_metrics.interference_analysis,
            "performance_conflicts": compound_metrics.performance_conflicts,
            "compound_effects": compound_effects,
            "system_snapshots": compound_metrics.system_snapshots[-50:],
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(detailed_metrics, f, indent=2, default=str)
        print(f"✅ Compound performance report saved to: {report_path}")
        print(f"✅ Detailed metrics saved to: {metrics_path}")
        print("\n📈 Compound Performance Summary:")
        if "system_efficiency" in compound_effects:
            eff = compound_effects["system_efficiency"]
            print(f"  - Peak Memory: {eff['peak_memory_gb']:.2f}GB")
            print(f"  - Average Memory: {eff['avg_memory_gb']:.2f}GB")
            print(
                f"  - Resource Efficiency Score: {eff['resource_efficiency_score']:.1f}/100"
            )
        print(
            f"  - Individual Improvements: {len(compound_metrics.individual_improvements)}"
        )
        print(f"  - Compound Tests: {len(compound_metrics.compound_measurements)}")
        print(
            f"  - Performance Conflicts: {len(compound_metrics.performance_conflicts)}"
        )
        significant_improvements = []
        for test_name, results in compound_effects.items():
            if test_name not in {"interference_summary", "system_efficiency"}:
                for metric, value in results.items():
                    if (
                        "improvement_percent" in metric
                        and isinstance(value, (int, float))
                        and (value >= 50)
                    ):
                        significant_improvements.append(
                            f"{test_name}: {metric} = {value:.1f}%"
                        )
        print(f"  - Significant Improvements (>50%): {len(significant_improvements)}")
        for improvement in significant_improvements[:5]:
            print(f"    - {improvement}")
        efficiency_score = compound_effects.get("system_efficiency", {}).get(
            "resource_efficiency_score", 0
        )
        conflicts_count = len(compound_metrics.performance_conflicts)
        print("\n🎯 Compound Performance Assessment:")
        if efficiency_score >= 80 and conflicts_count == 0:
            print(
                "🚀 EXCELLENT: All optimizations compound effectively with no conflicts"
            )
        elif efficiency_score >= 60 and conflicts_count <= 2:
            print("✅ GOOD: Most optimizations compound well with minor conflicts")
        else:
            print(
                "⚠️ NEEDS OPTIMIZATION: Significant conflicts or inefficiencies detected"
            )
        assert efficiency_score >= 60, (
            f"Resource efficiency too low: {efficiency_score:.1f}/100"
        )
        assert conflicts_count <= 3, (
            f"Too many performance conflicts: {conflicts_count}"
        )
        assert len(significant_improvements) >= 3, (
            f"Not enough significant improvements: {len(significant_improvements)}"
        )
        print("\n✅ Compound Performance Validation Complete!")
        print(f"📄 Detailed report: {report_path}")

    async def _monitor_system_during_compound_test(
        self, metrics: CompoundPerformanceMetrics, tasks: list[asyncio.Task]
    ):
        """Monitor system resources during compound testing."""
        while not all(task.done() for task in tasks):
            metrics.record_system_snapshot()
            await asyncio.sleep(3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
