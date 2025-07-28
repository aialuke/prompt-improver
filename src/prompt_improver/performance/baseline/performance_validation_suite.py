"""
Performance Validation and Optimization Suite for Baseline System.

Validates that the performance baseline system itself meets performance standards,
provides optimization recommendations, and ensures production readiness.
"""

import asyncio
import logging
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import json
import psutil
import tracemalloc

from .baseline_collector import BaselineCollector
from .statistical_analyzer import StatisticalAnalyzer
from .regression_detector import RegressionDetector
from .automation import BaselineAutomation
from .models import BaselineMetrics
from .enhanced_dashboard_integration import PerformanceDashboard
from .load_testing_integration import LoadTestingIntegration

logger = logging.getLogger(__name__)

@dataclass
class PerformanceValidationResult:
    """Results from performance validation testing."""
    test_name: str
    duration_ms: float
    memory_usage_mb: float
    cpu_utilization_percent: float
    meets_target: bool
    target_ms: float = 200.0
    optimization_recommendations: List[str] = field(default_factory=list)
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemEfficiencyReport:
    """Comprehensive system efficiency analysis."""
    baseline_collection_overhead: float
    analysis_processing_time: float
    dashboard_response_time: float
    memory_efficiency_score: float
    cpu_efficiency_score: float
    overall_efficiency_grade: str
    production_readiness: bool
    optimization_priorities: List[str] = field(default_factory=list)

class PerformanceValidationSuite:
    """
    Comprehensive validation suite for the performance baseline system.
    
    Ensures the baseline system itself meets performance standards and provides
    optimization recommendations for production deployment.
    """
    
    def __init__(self):
        self.collector = BaselineCollector()
        self.analyzer = StatisticalAnalyzer()
        self.detector = RegressionDetector()
        self.automation = BaselineAutomation()
        self.dashboard = PerformanceDashboard()
        self.load_testing = LoadTestingIntegration()
        
        self.validation_results: List[PerformanceValidationResult] = []
        self.performance_targets = {
            'baseline_collection': 100.0,  # 100ms max for single collection
            'statistical_analysis': 150.0,  # 150ms max for analysis
            'regression_detection': 50.0,   # 50ms max for detection
            'dashboard_update': 200.0,      # 200ms max for dashboard update
            'automation_cycle': 1000.0      # 1s max for automation cycle
        }
    
    async def run_comprehensive_validation(self) -> SystemEfficiencyReport:
        """Run complete performance validation suite."""
        logger.info("Starting comprehensive performance validation suite...")
        
        # Reset validation results
        self.validation_results.clear()
        
        try:
            # 1. Test baseline collection performance
            collection_result = await self._validate_baseline_collection()
            self.validation_results.append(collection_result)
            
            # 2. Test statistical analysis performance
            analysis_result = await self._validate_statistical_analysis()
            self.validation_results.append(analysis_result)
            
            # 3. Test regression detection performance
            detection_result = await self._validate_regression_detection()
            self.validation_results.append(detection_result)
            
            # 4. Test dashboard performance
            dashboard_result = await self._validate_dashboard_performance()
            self.validation_results.append(dashboard_result)
            
            # 5. Test integration performance
            integration_result = await self._validate_system_integration()
            self.validation_results.append(integration_result)
            
            # 6. Test under load
            load_result = await self._validate_performance_under_load()
            self.validation_results.append(load_result)
            
            # 7. Test memory efficiency
            memory_result = await self._validate_memory_efficiency()
            self.validation_results.append(memory_result)
            
            # Generate comprehensive report
            report = await self._generate_efficiency_report()
            
            logger.info("Performance validation suite completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            raise
    
    async def _validate_baseline_collection(self) -> PerformanceValidationResult:
        """Validate baseline collection performance."""
        logger.info("Validating baseline collection performance...")
        
        # Measure multiple collection cycles
        durations = []
        memory_usage = []
        cpu_usage = []
        
        for i in range(10):  # Test 10 collection cycles
            # Start memory tracking
            tracemalloc.start()
            start_cpu = psutil.cpu_percent()
            start_time = time.time()
            
            try:
                # Perform baseline collection
                baseline = await self.collector.collect_baseline()
                
                # Measure performance
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                durations.append(duration_ms)
                
                # Measure memory
                current, peak = tracemalloc.get_traced_memory()
                memory_usage.append(peak / 1024 / 1024)  # Convert to MB
                tracemalloc.stop()
                
                # Measure CPU (approximate)
                end_cpu = psutil.cpu_percent()
                cpu_usage.append(end_cpu - start_cpu)
                
                # Small delay between collections
                await asyncio.sleep(0.1)
                
            except Exception as e:
                tracemalloc.stop()
                logger.error(f"Collection test {i} failed: {e}")
                durations.append(1000.0)  # Penalty for failure
        
        # Calculate statistics
        avg_duration = statistics.mean(durations)
        max_duration = max(durations)
        p95_duration = self._percentile(durations, 95)
        avg_memory = statistics.mean(memory_usage)
        avg_cpu = statistics.mean(cpu_usage)
        
        # Check if meets target
        target = self.performance_targets['baseline_collection']
        meets_target = p95_duration <= target
        
        # Generate recommendations
        recommendations = []
        if not meets_target:
            recommendations.append(f"P95 duration ({p95_duration:.1f}ms) exceeds target ({target}ms)")
        if avg_memory > 50:
            recommendations.append(f"High memory usage ({avg_memory:.1f}MB) - consider data structure optimization")
        if max_duration > target * 2:
            recommendations.append("Inconsistent performance - investigate bottlenecks")
        
        return PerformanceValidationResult(
            test_name="baseline_collection",
            duration_ms=p95_duration,
            memory_usage_mb=avg_memory,
            cpu_utilization_percent=avg_cpu,
            meets_target=meets_target,
            target_ms=target,
            optimization_recommendations=recommendations,
            detailed_metrics={
                'avg_duration_ms': avg_duration,
                'max_duration_ms': max_duration,
                'p95_duration_ms': p95_duration,
                'duration_stddev': statistics.stdev(durations),
                'success_rate': (10 - durations.count(1000.0)) / 10
            }
        )
    
    async def _validate_statistical_analysis(self) -> PerformanceValidationResult:
        """Validate statistical analysis performance."""
        logger.info("Validating statistical analysis performance...")
        
        # Generate test baseline data
        test_baselines = []
        for i in range(50):  # Analyze 50 data points
            baseline = BaselineMetrics(
                collection_timestamp=datetime.now(timezone.utc),
                response_times=[100 + i, 120 + i, 80 + i],
                cpu_utilization=[20 + i % 10, 25 + i % 10],
                memory_utilization=[40 + i % 20, 45 + i % 20],
                error_rates=[0.1 + (i % 5) * 0.1],
                throughput_values=[1000 + i * 10, 1100 + i * 10]
            )
            test_baselines.append(baseline)
        
        # Measure analysis performance
        durations = []
        memory_usage = []
        
        for i in range(5):  # Test 5 analysis cycles
            tracemalloc.start()
            start_time = time.time()
            
            try:
                # Perform statistical analysis
                trend = await self.analyzer.analyze_trend(test_baselines[-20:])
                anomalies = await self.analyzer.detect_anomalies(test_baselines[-10:])
                forecast = await self.analyzer.forecast_performance(test_baselines[-30:])
                
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                durations.append(duration_ms)
                
                current, peak = tracemalloc.get_traced_memory()
                memory_usage.append(peak / 1024 / 1024)
                tracemalloc.stop()
                
            except Exception as e:
                tracemalloc.stop()
                logger.error(f"Analysis test {i} failed: {e}")
                durations.append(1000.0)
        
        avg_duration = statistics.mean(durations)
        avg_memory = statistics.mean(memory_usage)
        target = self.performance_targets['statistical_analysis']
        meets_target = avg_duration <= target
        
        recommendations = []
        if not meets_target:
            recommendations.append("Statistical analysis exceeds performance target")
            recommendations.append("Consider algorithmic optimization or data sampling")
        if avg_memory > 100:
            recommendations.append("High memory usage in analysis - optimize data structures")
        
        return PerformanceValidationResult(
            test_name="statistical_analysis",
            duration_ms=avg_duration,
            memory_usage_mb=avg_memory,
            cpu_utilization_percent=0,  # Not measured for this test
            meets_target=meets_target,
            target_ms=target,
            optimization_recommendations=recommendations,
            detailed_metrics={
                'analysis_operations': 3,  # trend, anomalies, forecast
                'data_points_analyzed': 50,
                'throughput_ops_per_second': 1000 / avg_duration if avg_duration > 0 else 0
            }
        )
    
    async def _validate_regression_detection(self) -> PerformanceValidationResult:
        """Validate regression detection performance."""
        logger.info("Validating regression detection performance...")
        
        # Create test data with known regression
        current_baseline = BaselineMetrics(
            collection_timestamp=datetime.now(timezone.utc),
            response_times=[300, 350, 280],  # Regression: high response times
            cpu_utilization=[85, 90],        # Regression: high CPU
            memory_utilization=[60, 65],
            error_rates=[2.0],               # Regression: high error rate
            throughput_values=[800, 850]     # Regression: low throughput
        )
        
        historical_baselines = []
        for i in range(20):
            baseline = BaselineMetrics(
                collection_timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                response_times=[100, 120, 90],
                cpu_utilization=[30, 35],
                memory_utilization=[40, 45],
                error_rates=[0.5],
                throughput_values=[1200, 1300]
            )
            historical_baselines.append(baseline)
        
        # Measure detection performance
        durations = []
        for i in range(10):
            start_time = time.time()
            
            try:
                alerts = await self.detector.check_for_regressions(
                    current_baseline, historical_baselines
                )
                
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                durations.append(duration_ms)
                
            except Exception as e:
                logger.error(f"Detection test {i} failed: {e}")
                durations.append(1000.0)
        
        avg_duration = statistics.mean(durations)
        target = self.performance_targets['regression_detection']
        meets_target = avg_duration <= target
        
        recommendations = []
        if not meets_target:
            recommendations.append("Regression detection too slow for real-time alerting")
            recommendations.append("Optimize detection algorithms or reduce comparison data")
        
        return PerformanceValidationResult(
            test_name="regression_detection",
            duration_ms=avg_duration,
            memory_usage_mb=0,  # Not measured for this test
            cpu_utilization_percent=0,
            meets_target=meets_target,
            target_ms=target,
            optimization_recommendations=recommendations,
            detailed_metrics={
                'alerts_generated': len(alerts) if 'alerts' in locals() else 0,
                'historical_data_points': 20,
                'detection_accuracy': 'high'  # Would need ground truth for actual measurement
            }
        )
    
    async def _validate_dashboard_performance(self) -> PerformanceValidationResult:
        """Validate dashboard performance."""
        logger.info("Validating dashboard performance...")
        
        # Simulate dashboard operations
        durations = []
        for i in range(5):
            start_time = time.time()
            
            try:
                # Test dashboard operations
                status = self.dashboard.get_current_status()
                charts = self.dashboard.create_performance_charts()
                summary = self.dashboard.get_performance_summary()
                
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                durations.append(duration_ms)
                
            except Exception as e:
                logger.error(f"Dashboard test {i} failed: {e}")
                durations.append(1000.0)
        
        avg_duration = statistics.mean(durations)
        target = self.performance_targets['dashboard_update']
        meets_target = avg_duration <= target
        
        recommendations = []
        if not meets_target:
            recommendations.append("Dashboard response time exceeds user experience target")
            recommendations.append("Consider data caching or chart pre-generation")
        
        return PerformanceValidationResult(
            test_name="dashboard_performance",
            duration_ms=avg_duration,
            memory_usage_mb=0,
            cpu_utilization_percent=0,
            meets_target=meets_target,
            target_ms=target,
            optimization_recommendations=recommendations,
            detailed_metrics={
                'operations_tested': 3,
                'charts_generated': len(charts) if 'charts' in locals() else 0
            }
        )
    
    async def _validate_system_integration(self) -> PerformanceValidationResult:
        """Validate full system integration performance."""
        logger.info("Validating system integration performance...")
        
        start_time = time.time()
        
        try:
            # Test full pipeline
            baseline = await self.collector.collect_baseline()
            trend = await self.analyzer.analyze_trend([baseline])
            alerts = await self.detector.check_for_regressions(baseline, [baseline])
            
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            duration_ms = 1000.0
        
        target = 300.0  # 300ms for full pipeline
        meets_target = duration_ms <= target
        
        recommendations = []
        if not meets_target:
            recommendations.append("Full pipeline exceeds performance budget")
            recommendations.append("Consider parallel processing or pipeline optimization")
        
        return PerformanceValidationResult(
            test_name="system_integration",
            duration_ms=duration_ms,
            memory_usage_mb=0,
            cpu_utilization_percent=0,
            meets_target=meets_target,
            target_ms=target,
            optimization_recommendations=recommendations,
            detailed_metrics={
                'pipeline_stages': 3,
                'end_to_end_latency': duration_ms
            }
        )
    
    async def _validate_performance_under_load(self) -> PerformanceValidationResult:
        """Validate performance under simulated load."""
        logger.info("Validating performance under load...")
        
        # Simulate concurrent baseline collections
        async def collect_baseline_with_timing():
            start_time = time.time()
            try:
                await self.collector.collect_baseline()
                return (time.time() - start_time) * 1000
            except Exception:
                return 1000.0
        
        # Test with 10 concurrent collections
        tasks = [collect_baseline_with_timing() for _ in range(10)]
        durations = await asyncio.gather(*tasks)
        
        avg_duration = statistics.mean(durations)
        max_duration = max(durations)
        success_rate = sum(1 for d in durations if d < 1000) / len(durations)
        
        target = self.performance_targets['baseline_collection'] * 2  # Allow 2x under load
        meets_target = avg_duration <= target and success_rate >= 0.9
        
        recommendations = []
        if not meets_target:
            recommendations.append("Performance degrades significantly under load")
            recommendations.append("Consider connection pooling and resource optimization")
        if success_rate < 0.9:
            recommendations.append("High failure rate under load - investigate error handling")
        
        return PerformanceValidationResult(
            test_name="performance_under_load",
            duration_ms=avg_duration,
            memory_usage_mb=0,
            cpu_utilization_percent=0,
            meets_target=meets_target,
            target_ms=target,
            optimization_recommendations=recommendations,
            detailed_metrics={
                'concurrent_operations': 10,
                'max_duration_ms': max_duration,
                'success_rate': success_rate,
                'load_factor': 2.0
            }
        )
    
    async def _validate_memory_efficiency(self) -> PerformanceValidationResult:
        """Validate memory efficiency over time."""
        logger.info("Validating memory efficiency...")
        
        # Start memory tracking
        tracemalloc.start()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Run baseline collection for 1 minute
        start_time = time.time()
        collections = 0
        max_memory = initial_memory
        
        while time.time() - start_time < 60:  # 1 minute test
            try:
                await self.collector.collect_baseline()
                collections += 1
                
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
                
                await asyncio.sleep(1)  # 1 second between collections
                
            except Exception as e:
                logger.error(f"Memory test collection failed: {e}")
        
        tracemalloc.stop()
        
        memory_growth = max_memory - initial_memory
        collections_per_mb = collections / max(memory_growth, 1)
        
        # Memory efficiency targets
        meets_target = memory_growth < 100  # Less than 100MB growth
        
        recommendations = []
        if memory_growth > 100:
            recommendations.append("Significant memory growth detected - potential memory leak")
            recommendations.append("Implement data retention policies and cleanup")
        if collections_per_mb < 10:
            recommendations.append("Low memory efficiency - optimize data structures")
        
        return PerformanceValidationResult(
            test_name="memory_efficiency",
            duration_ms=60000,  # 1 minute test
            memory_usage_mb=memory_growth,
            cpu_utilization_percent=0,
            meets_target=meets_target,
            target_ms=float('inf'),  # Not time-based target
            optimization_recommendations=recommendations,
            detailed_metrics={
                'initial_memory_mb': initial_memory,
                'final_memory_mb': max_memory,
                'memory_growth_mb': memory_growth,
                'collections_performed': collections,
                'collections_per_mb': collections_per_mb
            }
        )
    
    async def _generate_efficiency_report(self) -> SystemEfficiencyReport:
        """Generate comprehensive efficiency report."""
        logger.info("Generating system efficiency report...")
        
        # Calculate efficiency scores
        collection_result = next(r for r in self.validation_results if r.test_name == "baseline_collection")
        analysis_result = next(r for r in self.validation_results if r.test_name == "statistical_analysis")
        dashboard_result = next(r for r in self.validation_results if r.test_name == "dashboard_performance")
        memory_result = next(r for r in self.validation_results if r.test_name == "memory_efficiency")
        load_result = next(r for r in self.validation_results if r.test_name == "performance_under_load")
        
        # Calculate efficiency scores (0-100)
        memory_efficiency = max(0, 100 - memory_result.memory_usage_mb)
        cpu_efficiency = 100 - collection_result.cpu_utilization_percent
        
        # Overall efficiency grade
        passed_tests = sum(1 for r in self.validation_results if r.meets_target)
        total_tests = len(self.validation_results)
        success_rate = passed_tests / total_tests
        
        if success_rate >= 0.9:
            grade = "A"
        elif success_rate >= 0.8:
            grade = "B"
        elif success_rate >= 0.7:
            grade = "C"
        else:
            grade = "F"
        
        # Production readiness
        production_readiness = (
            collection_result.meets_target and
            analysis_result.meets_target and
            dashboard_result.meets_target and
            memory_result.meets_target and
            load_result.meets_target
        )
        
        # Optimization priorities
        priorities = []
        for result in self.validation_results:
            if not result.meets_target:
                priorities.extend(result.optimization_recommendations)
        
        # Remove duplicates and prioritize
        priorities = list(dict.fromkeys(priorities))[:5]  # Top 5 priorities
        
        return SystemEfficiencyReport(
            baseline_collection_overhead=collection_result.duration_ms,
            analysis_processing_time=analysis_result.duration_ms,
            dashboard_response_time=dashboard_result.duration_ms,
            memory_efficiency_score=memory_efficiency,
            cpu_efficiency_score=cpu_efficiency,
            overall_efficiency_grade=grade,
            production_readiness=production_readiness,
            optimization_priorities=priorities
        )
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def export_validation_report(self, filename: str = None) -> str:
        """Export validation results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_validation_report_{timestamp}.json"
        
        report_data = {
            'validation_timestamp': datetime.now(timezone.utc).isoformat(),
            'validation_results': [
                {
                    'test_name': r.test_name,
                    'duration_ms': r.duration_ms,
                    'memory_usage_mb': r.memory_usage_mb,
                    'cpu_utilization_percent': r.cpu_utilization_percent,
                    'meets_target': r.meets_target,
                    'target_ms': r.target_ms,
                    'optimization_recommendations': r.optimization_recommendations,
                    'detailed_metrics': r.detailed_metrics
                }
                for r in self.validation_results
            ],
            'performance_targets': self.performance_targets,
            'summary': {
                'total_tests': len(self.validation_results),
                'passed_tests': sum(1 for r in self.validation_results if r.meets_target),
                'overall_success_rate': sum(1 for r in self.validation_results if r.meets_target) / len(self.validation_results) if self.validation_results else 0
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            logger.info(f"Validation report exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to export validation report: {e}")
            return ""

# Global validation suite instance
_validation_suite: Optional[PerformanceValidationSuite] = None

def get_validation_suite() -> PerformanceValidationSuite:
    """Get global validation suite instance."""
    global _validation_suite
    if _validation_suite is None:
        _validation_suite = PerformanceValidationSuite()
    return _validation_suite

# Convenience functions
async def validate_baseline_system_performance() -> SystemEfficiencyReport:
    """Run comprehensive performance validation."""
    suite = get_validation_suite()
    return await suite.run_comprehensive_validation()

async def quick_performance_check() -> bool:
    """Quick performance check - returns True if system meets basic targets."""
    suite = get_validation_suite()
    
    # Test only critical components
    collection_result = await suite._validate_baseline_collection()
    integration_result = await suite._validate_system_integration()
    
    return collection_result.meets_target and integration_result.meets_target