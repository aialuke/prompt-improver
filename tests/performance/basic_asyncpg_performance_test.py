"""
Basic AsyncPG Performance Test

A simplified performance test to validate basic database operations
and establish baseline metrics for the asyncpg migration.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
import numpy as np
from sqlalchemy import text

# Import actual database components (no mocks)
from prompt_improver.database import get_unified_manager, ManagerMode

logger = logging.getLogger(__name__)

@dataclass
class BasicPerformanceResult:
    """Basic performance test result."""
    operation: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    iterations: int
    success_rate: float
    timestamp: datetime

class BasicAsyncPGPerformanceTest:
    """Basic performance test for AsyncPG migration validation."""
    
    def __init__(self):
        self.manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        self.results: List[BasicPerformanceResult] = []
    
    async def run_basic_performance_test(self) -> Dict[str, BasicPerformanceResult]:
        """Run basic performance tests."""
        logger.info("🚀 Starting basic AsyncPG performance test")
        
        # Initialize the manager
        await self.manager.initialize()
        
        results = {}
        
        # Test basic operations
        operations = [
            ("SELECT_SIMPLE", "SELECT 1", 100),
            ("SELECT_NOW", "SELECT NOW()", 100),
            ("SELECT_VERSION", "SELECT version()", 50),
        ]
        
        for op_name, query, iterations in operations:
            logger.info(f"Testing {op_name} with {iterations} iterations")
            result = await self._test_operation(op_name, query, iterations)
            results[op_name] = result
            
            # Log result
            logger.info(f"{op_name}: avg={result.avg_time_ms:.2f}ms, "
                       f"min={result.min_time_ms:.2f}ms, max={result.max_time_ms:.2f}ms, "
                       f"success_rate={result.success_rate:.1f}%")
        
        # Test connection establishment
        logger.info("Testing connection establishment")
        conn_result = await self._test_connection_establishment(50)
        results["CONNECTION_ESTABLISHMENT"] = conn_result
        
        logger.info(f"Connection establishment: avg={conn_result.avg_time_ms:.2f}ms, "
                   f"success_rate={conn_result.success_rate:.1f}%")
        
        # Save results
        await self._save_results(results)
        
        return results
    
    async def _test_operation(self, operation_name: str, query: str, iterations: int) -> BasicPerformanceResult:
        """Test a specific database operation."""
        times = []
        errors = 0
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            try:
                async with self.manager.get_async_session() as session:
                    result = await session.execute(text(query))
                    await result.fetchone()
                
                end_time = time.perf_counter()
                execution_time_ms = (end_time - start_time) * 1000
                times.append(execution_time_ms)
                
            except Exception as e:
                logger.error(f"Error in {operation_name} iteration {i}: {e}")
                errors += 1
                times.append(1000.0)  # Penalty for errors
            
            # Small delay between iterations
            await asyncio.sleep(0.01)
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        success_rate = ((iterations - errors) / iterations) * 100
        
        return BasicPerformanceResult(
            operation=operation_name,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            iterations=iterations,
            success_rate=success_rate,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def _test_connection_establishment(self, iterations: int) -> BasicPerformanceResult:
        """Test connection establishment performance."""
        times = []
        errors = 0
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            try:
                async with self.manager.get_async_session() as session:
                    # Simple query to ensure connection is established
                    result = await session.execute(text("SELECT 1"))
                    await result.fetchone()
                
                end_time = time.perf_counter()
                connection_time_ms = (end_time - start_time) * 1000
                times.append(connection_time_ms)
                
            except Exception as e:
                logger.error(f"Connection error in iteration {i}: {e}")
                errors += 1
                times.append(100.0)  # Penalty for errors
            
            # Small delay between iterations
            await asyncio.sleep(0.02)
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        success_rate = ((iterations - errors) / iterations) * 100
        
        return BasicPerformanceResult(
            operation="CONNECTION_ESTABLISHMENT",
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            iterations=iterations,
            success_rate=success_rate,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def _save_results(self, results: Dict[str, BasicPerformanceResult]) -> None:
        """Save test results to file."""
        report_data = {
            "test_timestamp": datetime.now(timezone.utc).isoformat(),
            "test_type": "basic_asyncpg_performance",
            "results": {
                name: {
                    "operation": result.operation,
                    "avg_time_ms": result.avg_time_ms,
                    "min_time_ms": result.min_time_ms,
                    "max_time_ms": result.max_time_ms,
                    "iterations": result.iterations,
                    "success_rate": result.success_rate,
                    "timestamp": result.timestamp.isoformat()
                }
                for name, result in results.items()
            },
            "summary": {
                "total_operations": len(results),
                "avg_response_time_ms": np.mean([r.avg_time_ms for r in results.values()]),
                "overall_success_rate": np.mean([r.success_rate for r in results.values()]),
                "fastest_operation": min(results.items(), key=lambda x: x[1].avg_time_ms)[0],
                "slowest_operation": max(results.items(), key=lambda x: x[1].avg_time_ms)[0]
            }
        }
        
        # Save to file
        report_file = Path("basic_performance_results.json")
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"📄 Basic performance results saved to {report_file}")

async def main():
    """Run basic performance test."""
    test = BasicAsyncPGPerformanceTest()
    results = await test.run_basic_performance_test()
    
    print("\n" + "="*60)
    print("🎯 BASIC ASYNCPG PERFORMANCE TEST RESULTS")
    print("="*60)
    
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Average: {result.avg_time_ms:.2f}ms")
        print(f"  Range: {result.min_time_ms:.2f}ms - {result.max_time_ms:.2f}ms")
        print(f"  Success Rate: {result.success_rate:.1f}%")
        print()
    
    # Calculate overall metrics
    avg_response_time = np.mean([r.avg_time_ms for r in results.values()])
    overall_success_rate = np.mean([r.success_rate for r in results.values()])
    
    print(f"Overall Average Response Time: {avg_response_time:.2f}ms")
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    
    # Performance evaluation
    if avg_response_time < 50.0 and overall_success_rate > 95.0:
        print("✅ Performance targets met!")
    else:
        print("❌ Performance targets not met")
        if avg_response_time >= 50.0:
            print(f"   - Response time {avg_response_time:.2f}ms exceeds 50ms target")
        if overall_success_rate <= 95.0:
            print(f"   - Success rate {overall_success_rate:.1f}% below 95% target")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
