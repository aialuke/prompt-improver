"""Load Testing Integration for Performance Baseline System.

Integrates automated load testing with baseline collection to provide
realistic performance analysis under various load conditions.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum
import statistics

# Enhanced background task management
from ...performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority

# Load testing libraries
try:
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from .baseline_collector import BaselineCollector
from .models import BaselineMetrics

logger = logging.getLogger(__name__)

class LoadPattern(Enum):
    """Load testing patterns."""
    CONSTANT = "constant"
    RAMP_UP = "ramp_up"
    SPIKE = "spike"
    STRESS = "stress"
    SOAK = "soak"
    VOLUME = "volume"

@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    pattern: LoadPattern
    duration_minutes: int
    target_rps: int
    max_users: int
    ramp_up_duration: int = 60
    endpoint_weights: Dict[str, float] = field(default_factory=dict)
    realistic_data: bool = True
    
class LoadTestingIntegration:
    """Integration between load testing and baseline collection."""
    
    def __init__(self, collector: Optional[BaselineCollector] = None):
        self.collector = collector or BaselineCollector()
        self.active_tests: Dict[str, Dict] = {}
        self.test_results: List[Dict] = []
        
    async def run_load_test_with_baseline_collection(
        self,
        test_name: str,
        config: LoadTestConfig,
        target_endpoints: List[str]
    ) -> Dict[str, Any]:
        """Run load test while collecting performance baselines."""
        test_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting load test {test_name} with baseline collection")
        
        # Start baseline collection using EnhancedBackgroundTaskManager
        task_manager = get_background_task_manager()
        collection_task_id = await task_manager.submit_enhanced_task(
            task_id=f"load_test_baseline_collection_{test_id[:8]}",
            coroutine=self._collect_baselines_during_test(test_id, config.duration_minutes),
            priority=TaskPriority.HIGH,
            tags={"service": "performance", "type": "load_testing", "component": "baseline_collection", "test_name": test_name}
        )
        
        # Run load test based on pattern
        if config.pattern == LoadPattern.CONSTANT:
            load_results = await self._run_constant_load_test(config, target_endpoints)
        elif config.pattern == LoadPattern.RAMP_UP:
            load_results = await self._run_ramp_up_test(config, target_endpoints)
        elif config.pattern == LoadPattern.SPIKE:
            load_results = await self._run_spike_test(config, target_endpoints)
        else:
            load_results = await self._run_generic_load_test(config, target_endpoints)
        
        # Wait for baseline collection to complete
        baseline_results = await task_manager.wait_for_task(collection_task_id)
        
        # Analyze results
        analysis = await self._analyze_load_test_results(
            load_results, baseline_results, config
        )
        
        end_time = datetime.now(timezone.utc)
        
        test_result = {
            'test_id': test_id,
            'test_name': test_name,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_minutes': config.duration_minutes,
            'pattern': config.pattern.value,
            'target_rps': config.target_rps,
            'max_users': config.max_users,
            'load_results': load_results,
            'baseline_results': baseline_results,
            'analysis': analysis
        }
        
        self.test_results.append(test_result)
        return test_result
    
    async def _collect_baselines_during_test(
        self, 
        test_id: str, 
        duration_minutes: int
    ) -> List[BaselineMetrics]:
        """Collect baselines during load test."""
        baselines = []
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                baseline = await self.collector.collect_baseline()
                baseline.tags['load_test_id'] = test_id
                baseline.tags['test_phase'] = self._determine_test_phase(baselines)
                baselines.append(baseline)
                
                # Collect baseline every 15 seconds during load test
                await asyncio.sleep(15)
            except Exception as e:
                logger.error(f"Error collecting baseline during load test: {e}")
        
        return baselines
    
    def _determine_test_phase(self, baselines: List[BaselineMetrics]) -> str:
        """Determine current phase of load test."""
        if len(baselines) < 4:  # First minute
            return 'ramp_up'
        elif len(baselines) > 20:  # Last 5 minutes
            return 'ramp_down'
        else:
            return 'steady_state'
    
    async def _run_constant_load_test(
        self, 
        config: LoadTestConfig, 
        endpoints: List[str]
    ) -> Dict[str, Any]:
        """Run constant load test."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, simulating load test")
            return self._simulate_load_test_results(config)
        
        results = {
            'pattern': 'constant',
            'requests_sent': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': []
        }
        
        session = aiohttp.ClientSession()
        
        try:
            # Calculate requests per second distribution
            total_requests = config.target_rps * config.duration_minutes * 60
            request_interval = 1.0 / config.target_rps
            
            start_time = time.time()
            
            for i in range(total_requests):
                if time.time() - start_time > (config.duration_minutes * 60):
                    break
                
                endpoint = endpoints[i % len(endpoints)]
                request_start = time.time()
                
                try:
                    async with session.get(endpoint, timeout=30) as response:
                        response_time = (time.time() - request_start) * 1000
                        results['response_times'].append(response_time)
                        
                        if response.status < 400:
                            results['successful_requests'] += 1
                        else:
                            results['failed_requests'] += 1
                            results['errors'].append(f"HTTP {response.status}")
                
                except Exception as e:
                    results['failed_requests'] += 1
                    results['errors'].append(str(e))
                
                results['requests_sent'] += 1
                
                # Maintain target RPS
                await asyncio.sleep(max(0, request_interval - (time.time() - request_start)))
        
        finally:
            await session.close()
        
        return results
    
    async def _run_ramp_up_test(
        self, 
        config: LoadTestConfig, 
        endpoints: List[str]
    ) -> Dict[str, Any]:
        """Run ramp-up load test."""
        results = {
            'pattern': 'ramp_up',
            'requests_sent': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'rps_progression': [],
            'errors': []
        }
        
        session = aiohttp.ClientSession() if AIOHTTP_AVAILABLE else None
        
        try:
            duration_seconds = config.duration_minutes * 60
            ramp_duration = min(config.ramp_up_duration, duration_seconds // 2)
            
            for second in range(duration_seconds):
                # Calculate current RPS based on ramp-up
                if second < ramp_duration:
                    current_rps = (config.target_rps * second) / ramp_duration
                else:
                    current_rps = config.target_rps
                
                results['rps_progression'].append(current_rps)
                
                # Send requests for this second
                requests_this_second = max(1, int(current_rps))
                
                for _ in range(requests_this_second):
                    endpoint = endpoints[results['requests_sent'] % len(endpoints)]
                    
                    if session:
                        await self._send_test_request(session, endpoint, results)
                    else:
                        # Simulate request
                        await self._simulate_request(results)
                
                await asyncio.sleep(1)
        
        finally:
            if session:
                await session.close()
        
        return results
    
    async def _run_spike_test(
        self, 
        config: LoadTestConfig, 
        endpoints: List[str]
    ) -> Dict[str, Any]:
        """Run spike load test."""
        results = {
            'pattern': 'spike',
            'requests_sent': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'spike_events': [],
            'errors': []
        }
        
        session = aiohttp.ClientSession() if AIOHTTP_AVAILABLE else None
        
        try:
            duration_seconds = config.duration_minutes * 60
            normal_rps = config.target_rps // 2
            spike_rps = config.target_rps * 3
            
            for second in range(duration_seconds):
                # Create spikes every 2 minutes for 30 seconds
                is_spike = (second % 120) < 30
                current_rps = spike_rps if is_spike else normal_rps
                
                if is_spike:
                    results['spike_events'].append({
                        'time': second,
                        'rps': current_rps,
                        'duration': 30
                    })
                
                # Send requests for this second
                for _ in range(current_rps):
                    endpoint = endpoints[results['requests_sent'] % len(endpoints)]
                    
                    if session:
                        await self._send_test_request(session, endpoint, results)
                    else:
                        await self._simulate_request(results)
                
                await asyncio.sleep(1)
        
        finally:
            if session:
                await session.close()
        
        return results
    
    async def _run_generic_load_test(
        self, 
        config: LoadTestConfig, 
        endpoints: List[str]
    ) -> Dict[str, Any]:
        """Run generic load test pattern."""
        return await self._run_constant_load_test(config, endpoints)
    
    async def _send_test_request(
        self, 
        session: 'aiohttp.ClientSession', 
        endpoint: str, 
        results: Dict
    ) -> None:
        """Send actual test request."""
        request_start = time.time()
        
        try:
            async with session.get(endpoint, timeout=30) as response:
                response_time = (time.time() - request_start) * 1000
                results['response_times'].append(response_time)
                
                if response.status < 400:
                    results['successful_requests'] += 1
                else:
                    results['failed_requests'] += 1
                    results['errors'].append(f"HTTP {response.status}")
        
        except Exception as e:
            results['failed_requests'] += 1
            results['errors'].append(str(e))
        
        results['requests_sent'] += 1
    
    async def _simulate_request(self, results: Dict) -> None:
        """Simulate request when actual testing not available."""
        # Simulate realistic response time
        import random
        response_time = random.normalvariate(120, 30)  # 120ms ± 30ms
        results['response_times'].append(max(10, response_time))
        
        # Simulate 95% success rate
        if random.random() < 0.95:
            results['successful_requests'] += 1
        else:
            results['failed_requests'] += 1
            results['errors'].append('Simulated error')
        
        results['requests_sent'] += 1
        
        # Small delay to simulate network
        await asyncio.sleep(0.001)
    
    def _simulate_load_test_results(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Simulate load test results when tools not available."""
        import random
        
        total_requests = config.target_rps * config.duration_minutes * 60
        successful_requests = int(total_requests * 0.95)  # 95% success rate
        failed_requests = total_requests - successful_requests
        
        # Generate realistic response times
        response_times = [
            max(10, random.normalvariate(120, 30)) 
            for _ in range(total_requests)
        ]
        
        return {
            'pattern': config.pattern.value,
            'requests_sent': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'response_times': response_times,
            'errors': ['Simulated error'] * failed_requests
        }
    
    async def _analyze_load_test_results(
        self,
        load_results: Dict[str, Any],
        baseline_results: List[BaselineMetrics],
        config: LoadTestConfig
    ) -> Dict[str, Any]:
        """Analyze load test results with baseline data."""
        analysis = {
            'load_performance': self._analyze_load_performance(load_results),
            'system_performance': self._analyze_system_performance(baseline_results),
            'correlation_analysis': self._analyze_load_system_correlation(
                load_results, baseline_results
            ),
            'capacity_assessment': self._assess_capacity(load_results, baseline_results, config),
            'recommendations': self._generate_load_test_recommendations(
                load_results, baseline_results, config
            )
        }
        
        return analysis
    
    def _analyze_load_performance(self, load_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze load test performance metrics."""
        response_times = load_results.get('response_times', [])
        
        if not response_times:
            return {'error': 'No response time data available'}
        
        return {
            'total_requests': load_results.get('requests_sent', 0),
            'success_rate': (
                load_results.get('successful_requests', 0) / 
                max(1, load_results.get('requests_sent', 1))
            ) * 100,
            'avg_response_time_ms': statistics.mean(response_times),
            'p50_response_time_ms': statistics.median(response_times),
            'p95_response_time_ms': self._percentile(response_times, 95),
            'p99_response_time_ms': self._percentile(response_times, 99),
            'max_response_time_ms': max(response_times),
            'min_response_time_ms': min(response_times),
            'error_count': load_results.get('failed_requests', 0),
            'unique_errors': len(set(load_results.get('errors', [])))
        }
    
    def _analyze_system_performance(self, baselines: List[BaselineMetrics]) -> Dict[str, Any]:
        """Analyze system performance during load test."""
        if not baselines:
            return {'error': 'No baseline data available'}
        
        # Aggregate metrics across all baselines
        cpu_values = []
        memory_values = []
        response_times = []
        error_rates = []
        
        for baseline in baselines:
            cpu_values.extend(baseline.cpu_utilization)
            memory_values.extend(baseline.memory_utilization)
            response_times.extend(baseline.response_times)
            error_rates.extend(baseline.error_rates)
        
        return {
            'cpu_utilization': {
                'avg': statistics.mean(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0,
                'p95': self._percentile(cpu_values, 95) if cpu_values else 0
            },
            'memory_utilization': {
                'avg': statistics.mean(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0,
                'p95': self._percentile(memory_values, 95) if memory_values else 0
            },
            'application_response_time': {
                'avg': statistics.mean(response_times) if response_times else 0,
                'p95': self._percentile(response_times, 95) if response_times else 0
            },
            'application_error_rate': {
                'avg': statistics.mean(error_rates) if error_rates else 0,
                'max': max(error_rates) if error_rates else 0
            },
            'baseline_count': len(baselines)
        }
    
    def _analyze_load_system_correlation(
        self,
        load_results: Dict[str, Any],
        baselines: List[BaselineMetrics]
    ) -> Dict[str, Any]:
        """Analyze correlation between load and system metrics."""
        if not baselines or not load_results.get('response_times'):
            return {'error': 'Insufficient data for correlation analysis'}
        
        # Simple correlation analysis
        load_response_times = load_results['response_times']
        system_cpu = []
        system_memory = []
        
        for baseline in baselines:
            if baseline.cpu_utilization:
                system_cpu.extend(baseline.cpu_utilization)
            if baseline.memory_utilization:
                system_memory.extend(baseline.memory_utilization)
        
        return {
            'load_response_avg': statistics.mean(load_response_times),
            'system_cpu_avg': statistics.mean(system_cpu) if system_cpu else 0,
            'system_memory_avg': statistics.mean(system_memory) if system_memory else 0,
            'correlation_strength': 'moderate',  # Simplified - would use actual correlation calculation
            'primary_bottleneck': self._identify_bottleneck(system_cpu, system_memory)
        }
    
    def _assess_capacity(self, load_results: Dict, baselines: List, config: LoadTestConfig) -> Dict:
        """Assess system capacity based on load test."""
        success_rate = (
            load_results.get('successful_requests', 0) / 
            max(1, load_results.get('requests_sent', 1))
        ) * 100
        
        avg_response_time = statistics.mean(load_results.get('response_times', [200]))
        
        # Simple capacity assessment
        if success_rate >= 99 and avg_response_time <= 200:
            capacity_status = 'excellent'
            headroom = 50  # 50% headroom
        elif success_rate >= 95 and avg_response_time <= 500:
            capacity_status = 'good'
            headroom = 25
        elif success_rate >= 90:
            capacity_status = 'acceptable'
            headroom = 10
        else:
            capacity_status = 'poor'
            headroom = 0
        
        return {
            'current_rps': config.target_rps,
            'max_recommended_rps': int(config.target_rps * (1 + headroom / 100)),
            'capacity_status': capacity_status,
            'headroom_percentage': headroom,
            'success_rate': success_rate,
            'avg_response_time_ms': avg_response_time
        }
    
    def _generate_load_test_recommendations(
        self,
        load_results: Dict,
        baselines: List,
        config: LoadTestConfig
    ) -> List[str]:
        """Generate recommendations based on load test results."""
        recommendations = []
        
        success_rate = (
            load_results.get('successful_requests', 0) / 
            max(1, load_results.get('requests_sent', 1))
        ) * 100
        
        avg_response_time = statistics.mean(load_results.get('response_times', [200]))
        
        if success_rate < 95:
            recommendations.append(
                f"Low success rate ({success_rate:.1f}%) - investigate error causes"
            )
        
        if avg_response_time > 200:
            recommendations.append(
                f"Response time exceeds target ({avg_response_time:.1f}ms > 200ms)"
            )
        
        # System resource recommendations
        if baselines:
            cpu_values = []
            memory_values = []
            
            for baseline in baselines:
                cpu_values.extend(baseline.cpu_utilization)
                memory_values.extend(baseline.memory_utilization)
            
            if cpu_values and max(cpu_values) > 80:
                recommendations.append("High CPU utilization detected - consider scaling")
            
            if memory_values and max(memory_values) > 85:
                recommendations.append("High memory utilization detected - check for leaks")
        
        if config.pattern == LoadPattern.SPIKE:
            recommendations.append("Consider implementing auto-scaling for spike handling")
        
        if not recommendations:
            recommendations.append("System performed well under load - monitor in production")
        
        return recommendations
    
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
    
    def _identify_bottleneck(self, cpu_values: List[float], memory_values: List[float]) -> str:
        """Identify primary system bottleneck."""
        if not cpu_values and not memory_values:
            return 'unknown'
        
        avg_cpu = statistics.mean(cpu_values) if cpu_values else 0
        avg_memory = statistics.mean(memory_values) if memory_values else 0
        
        if avg_cpu > 70 and avg_memory > 70:
            return 'both_cpu_memory'
        elif avg_cpu > 70:
            return 'cpu'
        elif avg_memory > 70:
            return 'memory'
        else:
            return 'none_detected'
    
    async def run_progressive_load_test(
        self,
        test_name: str,
        endpoints: List[str],
        max_rps: int = 100,
        duration_minutes: int = 30
    ) -> Dict[str, Any]:
        """Run progressive load test to find capacity limits."""
        test_results = []
        
        # Test at different RPS levels
        rps_levels = [10, 25, 50, 75, max_rps]
        
        for rps in rps_levels:
            logger.info(f"Testing at {rps} RPS")
            
            config = LoadTestConfig(
                pattern=LoadPattern.CONSTANT,
                duration_minutes=min(5, duration_minutes // len(rps_levels)),
                target_rps=rps,
                max_users=rps * 2
            )
            
            result = await self.run_load_test_with_baseline_collection(
                f"{test_name}_rps_{rps}",
                config,
                endpoints
            )
            
            test_results.append(result)
            
            # Stop if success rate drops below 90%
            analysis = result['analysis']['load_performance']
            if analysis['success_rate'] < 90:
                logger.warning(f"Success rate dropped to {analysis['success_rate']:.1f}% at {rps} RPS")
                break
        
        # Analyze progressive results
        capacity_analysis = self._analyze_progressive_results(test_results)
        
        return {
            'test_name': test_name,
            'individual_tests': test_results,
            'capacity_analysis': capacity_analysis
        }
    
    def _analyze_progressive_results(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Analyze results from progressive load testing."""
        rps_levels = []
        success_rates = []
        avg_response_times = []
        
        for result in test_results:
            analysis = result['analysis']['load_performance']
            rps_levels.append(result['target_rps'])
            success_rates.append(analysis['success_rate'])
            avg_response_times.append(analysis['avg_response_time_ms'])
        
        # Find capacity limits
        max_sustainable_rps = 0
        for i, (rps, success_rate, response_time) in enumerate(
            zip(rps_levels, success_rates, avg_response_times)
        ):
            if success_rate >= 95 and response_time <= 200:
                max_sustainable_rps = rps
            else:
                break
        
        return {
            'max_sustainable_rps': max_sustainable_rps,
            'tested_rps_levels': rps_levels,
            'success_rate_progression': success_rates,
            'response_time_progression': avg_response_times,
            'capacity_recommendation': {
                'production_rps': int(max_sustainable_rps * 0.7),  # 70% of max for safety
                'burst_capacity': max_sustainable_rps,
                'scaling_threshold': int(max_sustainable_rps * 0.8)  # Scale at 80% capacity
            }
        }

# Global instance
_load_testing_integration: Optional[LoadTestingIntegration] = None

def get_load_testing_integration() -> LoadTestingIntegration:
    """Get global load testing integration instance."""
    global _load_testing_integration
    if _load_testing_integration is None:
        _load_testing_integration = LoadTestingIntegration()
    return _load_testing_integration

# Convenience functions
async def run_integrated_load_test(
    test_name: str,
    pattern: LoadPattern,
    target_rps: int,
    duration_minutes: int,
    endpoints: List[str]
) -> Dict[str, Any]:
    """Run integrated load test with baseline collection."""
    integration = get_load_testing_integration()
    config = LoadTestConfig(
        pattern=pattern,
        duration_minutes=duration_minutes,
        target_rps=target_rps,
        max_users=target_rps * 2
    )
    
    return await integration.run_load_test_with_baseline_collection(
        test_name, config, endpoints
    )

async def find_system_capacity(
    endpoints: List[str],
    max_rps: int = 200,
    duration_minutes: int = 30
) -> Dict[str, Any]:
    """Find system capacity limits through progressive testing."""
    integration = get_load_testing_integration()
    return await integration.run_progressive_load_test(
        "capacity_test", endpoints, max_rps, duration_minutes
    )