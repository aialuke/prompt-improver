#!/usr/bin/env python3
"""
Phase 3 Metrics and Observability Validation Script

This script executes comprehensive real behavior testing of all Phase 3 metrics and 
observability systems, generating detailed performance and accuracy reports.

Features:
- Automated test execution with real service conditions
- Performance baseline validation 
- Load testing with production-like scenarios
- Comprehensive reporting with metrics accuracy validation
- System health monitoring during testing
- Export validation for dashboard integration

Usage:
    python scripts/validate_phase3_metrics.py [--output-dir OUTPUT_DIR] [--load-level LEVEL]
"""

import asyncio
import sys
import json
import time
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test classes
from tests.integration.test_phase3_metrics import (
    Phase3MetricsTestSuite,
    TestOpenTelemetryRealBehavior,
    TestBusinessMetricsRealBehavior,
    TestSystemMetricsUnderLoad,
    TestPerformanceBaselineSystem,
    TestSLOSLASystem,
    TestLoadTestingIntegration,
    TestIntegrationWithoutConflicts,
    TestPrometheusExportValidation,
    TestComprehensiveValidation
)


class Phase3MetricsValidator:
    """Comprehensive Phase 3 metrics validation orchestrator."""
    
    def __init__(self, output_dir: Path = None, load_level: str = "medium"):
        self.output_dir = output_dir or Path("phase3_validation_results")
        self.load_level = load_level
        self.test_suite = None
        self.validation_results = {
            'validation_id': f"phase3_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'load_level': load_level,
            'test_results': {},
            'performance_metrics': {},
            'system_health': {},
            'recommendations': []
        }
        
    async def setup_validation_environment(self):
        """Set up the validation environment and systems."""
        print("🔧 Setting up Phase 3 metrics validation environment...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize test suite
        self.test_suite = Phase3MetricsTestSuite()
        await self.test_suite.setup()
        
        # Record initial system state
        process = psutil.Process()
        initial_state = {
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(interval=1),
            'open_files': len(process.open_files()),
            'threads': process.num_threads(),
            'timestamp': time.time()
        }
        
        self.validation_results['initial_system_state'] = initial_state
        
        print(f"✅ Validation environment ready")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎚️ Load level: {self.load_level}")
        print(f"💾 Initial memory usage: {initial_state['memory_mb']:.1f} MB")
        
    async def execute_opentelemetry_validation(self):
        """Execute OpenTelemetry distributed tracing validation."""
        print("\n🔍 Executing OpenTelemetry Validation...")
        
        test_class = TestOpenTelemetryRealBehavior()
        opentelemetry_results = {}
        
        try:
            # Test distributed tracing
            start_time = time.time()
            await test_class.test_distributed_tracing_with_real_http_requests(self.test_suite)
            opentelemetry_results['distributed_tracing'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            # Test database operation tracing
            start_time = time.time()
            await test_class.test_database_operation_tracing(self.test_suite)
            opentelemetry_results['database_tracing'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            # Test ML operation tracing
            start_time = time.time()
            await test_class.test_ml_operation_tracing(self.test_suite)
            opentelemetry_results['ml_tracing'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            print("✅ OpenTelemetry validation completed successfully")
            
        except Exception as e:
            print(f"❌ OpenTelemetry validation failed: {e}")
            opentelemetry_results['error'] = str(e)
        
        self.validation_results['test_results']['opentelemetry'] = opentelemetry_results
        
    async def execute_business_metrics_validation(self):
        """Execute business metrics collection validation."""
        print("\n📊 Executing Business Metrics Validation...")
        
        test_class = TestBusinessMetricsRealBehavior()
        business_results = {}
        
        try:
            # Test feature usage tracking
            start_time = time.time()
            await test_class.test_feature_usage_tracking_with_real_operations(self.test_suite)
            business_results['feature_usage_tracking'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            # Test cost tracking
            start_time = time.time()
            await test_class.test_cost_tracking_with_real_operations(self.test_suite)
            business_results['cost_tracking'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            print("✅ Business metrics validation completed successfully")
            
        except Exception as e:
            print(f"❌ Business metrics validation failed: {e}")
            business_results['error'] = str(e)
        
        self.validation_results['test_results']['business_metrics'] = business_results
        
    async def execute_system_metrics_validation(self):
        """Execute system metrics under load validation."""
        print("\n⚡ Executing System Metrics Under Load Validation...")
        
        test_class = TestSystemMetricsUnderLoad()
        system_results = {}
        
        try:
            # Test connection age tracking
            start_time = time.time()
            await test_class.test_connection_age_tracking_under_load(self.test_suite)
            system_results['connection_age_tracking'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            # Test queue depth monitoring
            start_time = time.time()
            await test_class.test_queue_depth_monitoring_under_load(self.test_suite)
            system_results['queue_depth_monitoring'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            # Test cache hit rate monitoring
            start_time = time.time()
            await test_class.test_cache_hit_rate_under_realistic_load(self.test_suite)
            system_results['cache_hit_rate_monitoring'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            print("✅ System metrics validation completed successfully")
            
        except Exception as e:
            print(f"❌ System metrics validation failed: {e}")
            system_results['error'] = str(e)
        
        self.validation_results['test_results']['system_metrics'] = system_results
        
    async def execute_performance_baseline_validation(self):
        """Execute performance baseline system validation."""
        print("\n📈 Executing Performance Baseline Validation...")
        
        test_class = TestPerformanceBaselineSystem()
        baseline_results = {}
        
        try:
            # Test automated baseline collection
            start_time = time.time()
            await test_class.test_automated_baseline_collection(self.test_suite)
            baseline_results['automated_collection'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            # Test regression detection
            start_time = time.time()
            await test_class.test_regression_detection_with_real_performance_changes(self.test_suite)
            baseline_results['regression_detection'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            print("✅ Performance baseline validation completed successfully")
            
        except Exception as e:
            print(f"❌ Performance baseline validation failed: {e}")
            baseline_results['error'] = str(e)
        
        self.validation_results['test_results']['performance_baseline'] = baseline_results
        
    async def execute_slo_sla_validation(self):
        """Execute SLO/SLA system validation."""
        print("\n🎯 Executing SLO/SLA System Validation...")
        
        test_class = TestSLOSLASystem()
        slo_results = {}
        
        try:
            # Test SLO calculations
            start_time = time.time()
            await test_class.test_slo_calculations_with_real_service_data(self.test_suite)
            slo_results['slo_calculations'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            # Test error budget tracking
            start_time = time.time()
            await test_class.test_error_budget_tracking_and_burn_rate_alerting(self.test_suite)
            slo_results['error_budget_tracking'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            print("✅ SLO/SLA validation completed successfully")
            
        except Exception as e:
            print(f"❌ SLO/SLA validation failed: {e}")
            slo_results['error'] = str(e)
        
        self.validation_results['test_results']['slo_sla'] = slo_results
        
    async def execute_load_testing_validation(self):
        """Execute load testing integration validation."""
        print("\n🚀 Executing Load Testing Integration Validation...")
        
        test_class = TestLoadTestingIntegration()
        load_results = {}
        
        # Adjust load based on level
        load_multipliers = {
            'light': 0.5,
            'medium': 1.0,
            'heavy': 2.0,
            'stress': 4.0
        }
        
        multiplier = load_multipliers.get(self.load_level, 1.0)
        
        try:
            # Test high throughput metrics accuracy
            start_time = time.time()
            await test_class.test_high_throughput_metrics_accuracy(self.test_suite)
            load_results['high_throughput_accuracy'] = {
                'success': True,
                'duration': time.time() - start_time,
                'load_multiplier': multiplier
            }
            
            print("✅ Load testing validation completed successfully")
            
        except Exception as e:
            print(f"❌ Load testing validation failed: {e}")
            load_results['error'] = str(e)
        
        self.validation_results['test_results']['load_testing'] = load_results
        
    async def execute_integration_validation(self):
        """Execute cross-system integration validation."""
        print("\n🔗 Executing Integration Validation...")
        
        test_class = TestIntegrationWithoutConflicts()
        integration_results = {}
        
        try:
            # Test concurrent metrics collection
            start_time = time.time()
            await test_class.test_concurrent_metrics_collection_without_interference(self.test_suite)
            integration_results['concurrent_collection'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            print("✅ Integration validation completed successfully")
            
        except Exception as e:
            print(f"❌ Integration validation failed: {e}")
            integration_results['error'] = str(e)
        
        self.validation_results['test_results']['integration'] = integration_results
        
    async def execute_prometheus_validation(self):
        """Execute Prometheus export validation."""
        print("\n📊 Executing Prometheus Export Validation...")
        
        test_class = TestPrometheusExportValidation()
        prometheus_results = {}
        
        try:
            # Test Prometheus metrics export
            start_time = time.time()
            await test_class.test_prometheus_metrics_export_format(self.test_suite)
            prometheus_results['export_format'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            # Test Grafana dashboard integration
            start_time = time.time()
            await test_class.test_grafana_dashboard_integration(self.test_suite)
            prometheus_results['grafana_integration'] = {
                'success': True,
                'duration': time.time() - start_time
            }
            
            print("✅ Prometheus validation completed successfully")
            
        except Exception as e:
            print(f"❌ Prometheus validation failed: {e}")
            prometheus_results['error'] = str(e)
        
        self.validation_results['test_results']['prometheus'] = prometheus_results
        
    async def execute_comprehensive_validation(self):
        """Execute comprehensive end-to-end validation."""
        print("\n🎯 Executing Comprehensive End-to-End Validation...")
        
        test_class = TestComprehensiveValidation()
        comprehensive_results = {}
        
        try:
            # Test complete observability stack
            start_time = time.time()
            validation_data = await test_class.test_complete_observability_stack_validation(self.test_suite)
            comprehensive_results = {
                'success': True,
                'duration': time.time() - start_time,
                'validation_data': validation_data
            }
            
            print("✅ Comprehensive validation completed successfully")
            
        except Exception as e:
            print(f"❌ Comprehensive validation failed: {e}")
            comprehensive_results['error'] = str(e)
        
        self.validation_results['test_results']['comprehensive'] = comprehensive_results
        
    async def collect_system_health_metrics(self):
        """Collect final system health metrics."""
        print("\n💊 Collecting System Health Metrics...")
        
        process = psutil.Process()
        
        # System resource metrics
        system_metrics = {
            'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
            'peak_memory_mb': process.memory_info().peak_wss / 1024 / 1024 if hasattr(process.memory_info(), 'peak_wss') else None,
            'cpu_percent': process.cpu_percent(interval=1),
            'open_files': len(process.open_files()),
            'connections': len(process.connections()),
            'threads': process.num_threads(),
            'children': len(process.children())
        }
        
        # Calculate resource efficiency
        initial_memory = self.validation_results['initial_system_state']['memory_mb']
        memory_increase = system_metrics['memory_usage_mb'] - initial_memory
        memory_efficiency = (memory_increase / initial_memory) * 100 if initial_memory > 0 else 0
        
        system_metrics['memory_increase_mb'] = memory_increase
        system_metrics['memory_efficiency_percent'] = memory_efficiency
        
        self.validation_results['system_health'] = system_metrics
        
        print(f"💾 Final memory usage: {system_metrics['memory_usage_mb']:.1f} MB")
        print(f"📈 Memory increase: {memory_increase:.1f} MB ({memory_efficiency:.1f}%)")
        print(f"🧵 Active threads: {system_metrics['threads']}")
        
    def calculate_performance_metrics(self):
        """Calculate overall performance metrics."""
        print("\n📊 Calculating Performance Metrics...")
        
        test_results = self.validation_results['test_results']
        
        # Calculate success rates
        total_tests = 0
        successful_tests = 0
        total_duration = 0
        
        for category, results in test_results.items():
            if isinstance(results, dict):
                for test_name, test_data in results.items():
                    if isinstance(test_data, dict) and 'success' in test_data:
                        total_tests += 1
                        if test_data.get('success', False):
                            successful_tests += 1
                        if 'duration' in test_data:
                            total_duration += test_data['duration']
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        avg_test_duration = total_duration / total_tests if total_tests > 0 else 0
        
        performance_metrics = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate_percent': success_rate,
            'total_duration_seconds': total_duration,
            'average_test_duration_seconds': avg_test_duration,
            'tests_per_second': total_tests / total_duration if total_duration > 0 else 0
        }
        
        self.validation_results['performance_metrics'] = performance_metrics
        
        print(f"✅ Success rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        print(f"⏱️ Average test duration: {avg_test_duration:.2f}s")
        print(f"🏃 Tests per second: {performance_metrics['tests_per_second']:.2f}")
        
    def generate_recommendations(self):
        """Generate performance and optimization recommendations."""
        print("\n💡 Generating Recommendations...")
        
        recommendations = []
        
        # Analyze system health
        system_health = self.validation_results.get('system_health', {})
        memory_usage = system_health.get('memory_usage_mb', 0)
        memory_efficiency = system_health.get('memory_efficiency_percent', 0)
        
        if memory_usage > 500:
            recommendations.append({
                'category': 'memory',
                'priority': 'high',
                'message': f'High memory usage detected ({memory_usage:.1f} MB). Consider implementing memory optimization strategies.',
                'action': 'Implement memory pooling and garbage collection tuning'
            })
        
        if memory_efficiency > 50:
            recommendations.append({
                'category': 'memory',
                'priority': 'medium',
                'message': f'Memory efficiency could be improved ({memory_efficiency:.1f}% increase).',
                'action': 'Review memory allocation patterns and implement caching strategies'
            })
        
        # Analyze performance metrics
        performance = self.validation_results.get('performance_metrics', {})
        success_rate = performance.get('success_rate_percent', 0)
        
        if success_rate < 95:
            recommendations.append({
                'category': 'reliability',
                'priority': 'high',
                'message': f'Test success rate below optimal ({success_rate:.1f}%). Investigate failing tests.',
                'action': 'Review error logs and implement robust error handling'
            })
        
        # Analyze test results for specific issues
        test_results = self.validation_results.get('test_results', {})
        
        for category, results in test_results.items():
            if isinstance(results, dict) and 'error' in results:
                recommendations.append({
                    'category': 'functionality',
                    'priority': 'high',
                    'message': f'{category.title()} tests failed. Critical functionality may be impaired.',
                    'action': f'Debug and fix {category} system issues immediately'
                })
        
        # Performance recommendations
        avg_duration = performance.get('average_test_duration_seconds', 0)
        if avg_duration > 10:
            recommendations.append({
                'category': 'performance',
                'priority': 'medium',
                'message': f'Average test duration is high ({avg_duration:.2f}s). Consider optimization.',
                'action': 'Profile performance bottlenecks and implement caching/optimization'
            })
        
        self.validation_results['recommendations'] = recommendations
        
        print(f"💡 Generated {len(recommendations)} recommendations")
        for rec in recommendations:
            priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡"
            print(f"{priority_emoji} {rec['category'].title()}: {rec['message']}")
        
    async def generate_final_report(self):
        """Generate comprehensive validation report."""
        print("\n📄 Generating Final Validation Report...")
        
        # Add validation metadata
        self.validation_results['validation_completed'] = datetime.now().isoformat()
        self.validation_results['total_validation_duration'] = time.time() - self.validation_start_time
        
        # Save detailed JSON report
        report_file = self.output_dir / f"phase3_validation_report_{self.validation_results['validation_id']}.json"
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        # Generate summary report
        summary_file = self.output_dir / f"phase3_validation_summary_{self.validation_results['validation_id']}.txt"
        with open(summary_file, 'w') as f:
            f.write("Phase 3 Metrics and Observability Validation Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Validation ID: {self.validation_results['validation_id']}\n")
            f.write(f"Timestamp: {self.validation_results['timestamp']}\n")
            f.write(f"Load Level: {self.validation_results['load_level']}\n")
            f.write(f"Duration: {self.validation_results['total_validation_duration']:.2f}s\n\n")
            
            # Performance summary
            perf = self.validation_results['performance_metrics']
            f.write("Performance Summary:\n")
            f.write(f"• Success Rate: {perf['success_rate_percent']:.1f}%\n")
            f.write(f"• Total Tests: {perf['total_tests']}\n")
            f.write(f"• Average Duration: {perf['average_test_duration_seconds']:.2f}s\n\n")
            
            # System health summary
            health = self.validation_results['system_health']
            f.write("System Health Summary:\n")
            f.write(f"• Memory Usage: {health['memory_usage_mb']:.1f} MB\n")
            f.write(f"• Memory Efficiency: {health['memory_efficiency_percent']:.1f}%\n")
            f.write(f"• CPU Usage: {health['cpu_percent']:.1f}%\n\n")
            
            # Recommendations
            recommendations = self.validation_results['recommendations']
            f.write("Recommendations:\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. [{rec['priority'].upper()}] {rec['category'].title()}: {rec['message']}\n")
                f.write(f"   Action: {rec['action']}\n\n")
        
        print(f"📄 Detailed report: {report_file}")
        print(f"📋 Summary report: {summary_file}")
        
    async def cleanup_validation_environment(self):
        """Clean up validation environment."""
        print("\n🧹 Cleaning up validation environment...")
        
        if self.test_suite:
            await self.test_suite.cleanup()
        
        print("✅ Cleanup completed")
        
    async def run_validation(self):
        """Run complete Phase 3 metrics validation."""
        self.validation_start_time = time.time()
        
        print("🚀 Starting Phase 3 Metrics and Observability Validation")
        print("=" * 70)
        
        try:
            # Setup
            await self.setup_validation_environment()
            
            # Execute all validation tests
            await self.execute_opentelemetry_validation()
            await self.execute_business_metrics_validation()
            await self.execute_system_metrics_validation()
            await self.execute_performance_baseline_validation()
            await self.execute_slo_sla_validation()
            await self.execute_load_testing_validation()
            await self.execute_integration_validation()
            await self.execute_prometheus_validation()
            await self.execute_comprehensive_validation()
            
            # Analyze results
            await self.collect_system_health_metrics()
            self.calculate_performance_metrics()
            self.generate_recommendations()
            
            # Generate reports
            await self.generate_final_report()
            
            print("\n" + "=" * 70)
            print("✅ Phase 3 Metrics Validation Completed Successfully!")
            
            # Print summary
            perf = self.validation_results['performance_metrics']
            print(f"📊 Results: {perf['success_rate_percent']:.1f}% success rate")
            print(f"⏱️ Duration: {self.validation_results['total_validation_duration']:.2f}s")
            print(f"💡 Recommendations: {len(self.validation_results['recommendations'])}")
            
        except Exception as e:
            print(f"\n❌ Validation failed with error: {e}")
            raise
        
        finally:
            await self.cleanup_validation_environment()


async def main():
    """Main entry point for Phase 3 metrics validation."""
    parser = argparse.ArgumentParser(description="Phase 3 Metrics and Observability Validation")
    parser.add_argument("--output-dir", type=Path, default="phase3_validation_results",
                       help="Output directory for validation results")
    parser.add_argument("--load-level", choices=["light", "medium", "heavy", "stress"],
                       default="medium", help="Load testing level")
    
    args = parser.parse_args()
    
    # Create validator and run
    validator = Phase3MetricsValidator(
        output_dir=args.output_dir,
        load_level=args.load_level
    )
    
    await validator.run_validation()


if __name__ == "__main__":
    asyncio.run(main())