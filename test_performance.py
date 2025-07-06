#!/usr/bin/env python3
"""
Performance validation script for APES MCP Server.
Tests response times for MCP tools to ensure <200ms target.
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prompt_improver.mcp_server.mcp_server import improve_prompt, store_prompt, get_rule_status
from prompt_improver.database import get_session


class PerformanceTester:
    """Performance testing suite for MCP tools"""
    
    def __init__(self):
        self.results = {}
        
    async def test_improve_prompt_performance(self, iterations: int = 10) -> Dict[str, Any]:
        """Test improve_prompt tool performance"""
        print(f"🧪 Testing improve_prompt tool performance ({iterations} iterations)")
        
        test_prompts = [
            "Please analyze this thing and make it better",
            "Write a summary of the document",
            "Create a list of items",
            "Explain how this works",
            "Help me with this stuff"
        ]
        
        response_times = []
        
        for i in range(iterations):
            prompt = test_prompts[i % len(test_prompts)]
            
            start_time = time.time()
            try:
                # Test the tool directly
                result = await improve_prompt(
                    prompt=prompt,
                    context={"domain": "testing"},
                    session_id=f"test_{i}"
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                response_times.append(response_time)
                
                print(f"   Iteration {i+1}: {response_time:.2f}ms")
                
            except Exception as e:
                print(f"   Iteration {i+1}: ERROR - {e}")
                response_times.append(float('inf'))
        
        # Calculate statistics
        valid_times = [t for t in response_times if t != float('inf')]
        
        if valid_times:
            stats = {
                "avg_response_time": statistics.mean(valid_times),
                "median_response_time": statistics.median(valid_times),
                "min_response_time": min(valid_times),
                "max_response_time": max(valid_times),
                "std_dev": statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
                "success_rate": len(valid_times) / iterations * 100,
                "under_200ms_rate": len([t for t in valid_times if t < 200]) / len(valid_times) * 100
            }
        else:
            stats = {
                "avg_response_time": float('inf'),
                "median_response_time": float('inf'),
                "min_response_time": float('inf'),
                "max_response_time": float('inf'),
                "std_dev": 0,
                "success_rate": 0,
                "under_200ms_rate": 0
            }
        
        return stats
    
    async def test_store_prompt_performance(self, iterations: int = 5) -> Dict[str, Any]:
        """Test store_prompt tool performance"""
        print(f"\n🧪 Testing store_prompt tool performance ({iterations} iterations)")
        
        response_times = []
        
        for i in range(iterations):
            start_time = time.time()
            try:
                result = await store_prompt(
                    original=f"Test prompt {i}",
                    enhanced=f"Enhanced test prompt {i}",
                    metrics={"improvement_score": 0.8, "processing_time": 50},
                    session_id=f"perf_test_{i}"
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                response_times.append(response_time)
                
                print(f"   Iteration {i+1}: {response_time:.2f}ms")
                
            except Exception as e:
                print(f"   Iteration {i+1}: ERROR - {e}")
                response_times.append(float('inf'))
        
        # Calculate statistics
        valid_times = [t for t in response_times if t != float('inf')]
        
        if valid_times:
            stats = {
                "avg_response_time": statistics.mean(valid_times),
                "median_response_time": statistics.median(valid_times),
                "min_response_time": min(valid_times),
                "max_response_time": max(valid_times),
                "std_dev": statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
                "success_rate": len(valid_times) / iterations * 100,
                "under_200ms_rate": len([t for t in valid_times if t < 200]) / len(valid_times) * 100
            }
        else:
            stats = {
                "avg_response_time": float('inf'),
                "median_response_time": float('inf'),
                "min_response_time": float('inf'),
                "max_response_time": float('inf'),
                "std_dev": 0,
                "success_rate": 0,
                "under_200ms_rate": 0
            }
        
        return stats
    
    async def test_rule_status_performance(self, iterations: int = 5) -> Dict[str, Any]:
        """Test rule_status resource performance"""
        print(f"\n🧪 Testing rule_status resource performance ({iterations} iterations)")
        
        response_times = []
        
        for i in range(iterations):
            start_time = time.time()
            try:
                result = await get_rule_status()
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                response_times.append(response_time)
                
                print(f"   Iteration {i+1}: {response_time:.2f}ms")
                
            except Exception as e:
                print(f"   Iteration {i+1}: ERROR - {e}")
                response_times.append(float('inf'))
        
        # Calculate statistics
        valid_times = [t for t in response_times if t != float('inf')]
        
        if valid_times:
            stats = {
                "avg_response_time": statistics.mean(valid_times),
                "median_response_time": statistics.median(valid_times),
                "min_response_time": min(valid_times),
                "max_response_time": max(valid_times),
                "std_dev": statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
                "success_rate": len(valid_times) / iterations * 100,
                "under_200ms_rate": len([t for t in valid_times if t < 200]) / len(valid_times) * 100
            }
        else:
            stats = {
                "avg_response_time": float('inf'),
                "median_response_time": float('inf'),
                "min_response_time": float('inf'),
                "max_response_time": float('inf'),
                "std_dev": 0,
                "success_rate": 0,
                "under_200ms_rate": 0
            }
        
        return stats
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite"""
        print("🚀 Starting APES MCP Performance Validation")
        print("=" * 60)
        
        # Test each MCP endpoint
        improve_prompt_stats = await self.test_improve_prompt_performance(10)
        store_prompt_stats = await self.test_store_prompt_performance(5)
        rule_status_stats = await self.test_rule_status_performance(5)
        
        # Compile results
        results = {
            "improve_prompt": improve_prompt_stats,
            "store_prompt": store_prompt_stats,
            "rule_status": rule_status_stats
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE VALIDATION RESULTS")
        print("=" * 60)
        
        for tool_name, stats in results.items():
            print(f"\n🔧 {tool_name.upper()} TOOL:")
            print(f"   Average Response Time: {stats['avg_response_time']:.2f}ms")
            print(f"   Median Response Time:  {stats['median_response_time']:.2f}ms")
            print(f"   Min Response Time:     {stats['min_response_time']:.2f}ms")
            print(f"   Max Response Time:     {stats['max_response_time']:.2f}ms")
            print(f"   Standard Deviation:    {stats['std_dev']:.2f}ms")
            print(f"   Success Rate:          {stats['success_rate']:.1f}%")
            print(f"   Under 200ms Rate:      {stats['under_200ms_rate']:.1f}%")
            
            # Performance validation
            if stats['avg_response_time'] < 200 and stats['under_200ms_rate'] >= 90:
                print(f"   ✅ PERFORMANCE TARGET MET")
            elif stats['avg_response_time'] < 200:
                print(f"   ⚠️  MOSTLY MEETS TARGET (some outliers)")
            else:
                print(f"   ❌ PERFORMANCE TARGET NOT MET")
        
        # Overall assessment
        avg_response_times = [stats['avg_response_time'] for stats in results.values() if stats['avg_response_time'] != float('inf')]
        overall_under_200 = [stats['under_200ms_rate'] for stats in results.values()]
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if avg_response_times:
            overall_avg = statistics.mean(avg_response_times)
            overall_under_200_avg = statistics.mean(overall_under_200)
            
            print(f"   Overall Average Response Time: {overall_avg:.2f}ms")
            print(f"   Overall Under 200ms Rate:      {overall_under_200_avg:.1f}%")
            
            if overall_avg < 200 and overall_under_200_avg >= 90:
                print(f"   ✅ PHASE 1 PERFORMANCE TARGET ACHIEVED")
                print(f"   🎉 MCP response times validated under 200ms")
            else:
                print(f"   ⚠️  PHASE 1 PERFORMANCE TARGET NEEDS OPTIMIZATION")
        else:
            print(f"   ❌ UNABLE TO VALIDATE PERFORMANCE - ALL TESTS FAILED")
        
        return results


async def main():
    """Main performance testing function"""
    tester = PerformanceTester()
    results = await tester.run_comprehensive_test()
    return results


if __name__ == "__main__":
    asyncio.run(main())