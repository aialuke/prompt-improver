#!/usr/bin/env python3
"""
Comprehensive Performance Baseline Capture Script
Measures API response times, memory usage, CPU utilization, database query performance, 
ML inference latency, and Redis operations to establish performance baselines.
"""

import asyncio
import gc
import json
import logging
import os
import psutil
import time
import tracemalloc
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiohttp
import asyncpg
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import statistics
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Set PYTHONPATH for proper module discovery
os.environ['PYTHONPATH'] = str(project_root / "src")

# Performance monitoring imports
from prompt_improver.database.connection import get_session_context
from prompt_improver.utils.redis_cache import get_redis_client
from prompt_improver.performance.monitoring.performance_monitor import PerformanceMonitor
DATABASE_AVAILABLE = True

# Suppress noisy logging for cleaner output
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)

@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    load_average: Optional[List[float]]
    process_count: int
    thread_count: int

@dataclass
class DatabaseMetrics:
    """Database performance metrics."""
    connection_time_ms: float
    query_time_ms: float
    pool_active_connections: int
    pool_idle_connections: int
    transaction_time_ms: float
    insert_time_ms: Optional[float] = None
    select_time_ms: Optional[float] = None

@dataclass
class RedisMetrics:
    """Redis cache performance metrics."""
    ping_time_ms: float
    set_time_ms: float
    get_time_ms: float
    delete_time_ms: float
    memory_usage_mb: float
    connected_clients: int
    used_memory_human: str
    keyspace_hits: int
    keyspace_misses: int

@dataclass
class APIMetrics:
    """API endpoint performance metrics."""
    endpoint: str
    response_time_ms: float
    status_code: int
    payload_size_bytes: int
    memory_before_mb: float
    memory_after_mb: float
    cpu_before_percent: float
    cpu_after_percent: float

@dataclass
class MLMetrics:
    """Machine learning inference performance metrics."""
    model_name: str
    initialization_time_ms: float
    inference_time_ms: float
    memory_usage_mb: float
    batch_size: int
    features_extracted: int
    model_size_mb: Optional[float] = None

@dataclass
class BaselineReport:
    """Complete performance baseline report."""
    timestamp: str
    system_metrics: SystemMetrics
    database_metrics: Optional[DatabaseMetrics]
    redis_metrics: Optional[RedisMetrics]
    api_metrics: List[APIMetrics]
    ml_metrics: List[MLMetrics]
    cpu_benchmarks: Dict[str, Any]
    io_benchmarks: Dict[str, Any]
    summary: Dict[str, Any]

class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.results = {}
        self.start_time = time.time()
        
    def get_system_metrics(self) -> SystemMetrics:
        """Capture current system performance metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get load average if available (Unix systems)
        load_avg = None
        if hasattr(psutil, 'getloadavg'):
            try:
                load_avg = list(psutil.getloadavg())
            except:
                pass
        
        return SystemMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=memory.percent,
            memory_used_gb=round(memory.used / (1024**3), 2),
            memory_available_gb=round(memory.available / (1024**3), 2),
            disk_percent=disk.percent,
            disk_free_gb=round(disk.free / (1024**3), 2),
            load_average=load_avg,
            process_count=len(psutil.pids()),
            thread_count=psutil.cpu_count()
        )
    
    async def benchmark_database(self) -> Optional[DatabaseMetrics]:
        """Benchmark database performance."""
        if not DATABASE_AVAILABLE:
            print("⚠️  Database components not available, skipping database benchmark")
            return None
            
        print("📊 Benchmarking database performance...")
        
        try:
            # Connection time
            start_time = time.time()
            async with get_session_context() as session:
                connection_time = (time.time() - start_time) * 1000
                
                # Simple query
                start_time = time.time()
                result = await session.execute("SELECT version(), current_timestamp")
                query_time = (time.time() - start_time) * 1000
                
                # Transaction test
                start_time = time.time()
                await session.begin()
                await session.execute("SELECT 1")
                await session.commit()
                transaction_time = (time.time() - start_time) * 1000
                
                # Simulate insert test
                start_time = time.time()
                await session.execute(
                    "SELECT pg_sleep(0.001)"  # Simulate small operation
                )
                insert_time = (time.time() - start_time) * 1000
                
                # Select test with joins (simulate complex query)
                start_time = time.time()
                await session.execute("""
                    SELECT 
                        t1.table_name, 
                        t2.column_name 
                    FROM information_schema.tables t1
                    LEFT JOIN information_schema.columns t2 ON t1.table_name = t2.table_name
                    WHERE t1.table_schema = 'public'
                    LIMIT 10
                """)
                select_time = (time.time() - start_time) * 1000
                
            return DatabaseMetrics(
                connection_time_ms=round(connection_time, 2),
                query_time_ms=round(query_time, 2),
                pool_active_connections=1,  # Simplified for this test
                pool_idle_connections=0,
                transaction_time_ms=round(transaction_time, 2),
                insert_time_ms=round(insert_time, 2),
                select_time_ms=round(select_time, 2)
            )
            
        except Exception as e:
            print(f"❌ Database benchmark failed: {e}")
            return None
    
    async def benchmark_redis(self) -> Optional[RedisMetrics]:
        """Benchmark Redis cache performance."""
        print("📊 Benchmarking Redis performance...")
        
        try:
            # Get Redis client
            client = await get_redis_client()
            
            # Ping test
            start_time = time.time()
            await client.ping()
            ping_time = (time.time() - start_time) * 1000
            
            # Set operation
            test_key = "benchmark_test_key"
            test_value = json.dumps({"test": "data", "timestamp": time.time()})
            
            start_time = time.time()
            await client.setex(test_key, 60, test_value)
            set_time = (time.time() - start_time) * 1000
            
            # Get operation
            start_time = time.time()
            result = await client.get(test_key)
            get_time = (time.time() - start_time) * 1000
            
            # Delete operation
            start_time = time.time()
            await client.delete(test_key)
            delete_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = await client.info()
            
            return RedisMetrics(
                ping_time_ms=round(ping_time, 2),
                set_time_ms=round(set_time, 2),
                get_time_ms=round(get_time, 2),
                delete_time_ms=round(delete_time, 2),
                memory_usage_mb=round(info.get('used_memory', 0) / (1024*1024), 2),
                connected_clients=info.get('connected_clients', 0),
                used_memory_human=info.get('used_memory_human', 'unknown'),
                keyspace_hits=info.get('keyspace_hits', 0),
                keyspace_misses=info.get('keyspace_misses', 0)
            )
            
        except Exception as e:
            print(f"❌ Redis benchmark failed: {e}")
            return None
    
    async def benchmark_api_endpoints(self, base_url: str = "http://localhost:8000") -> List[APIMetrics]:
        """Benchmark API endpoint performance."""
        print("📊 Benchmarking API endpoints...")
        
        endpoints = [
            "/health/live",
            "/health/ready", 
            "/health/startup",
            "/health"
        ]
        
        metrics = []
        
        # Use unified HTTP client for health checks
        from prompt_improver.monitoring.unified_http_client import make_health_check_request
        
        for endpoint in endpoints:
            try:
                # Capture metrics before request
                memory_before = self.process.memory_info().rss / (1024*1024)
                cpu_before = self.process.cpu_percent()
                
                start_time = time.time()
                async with make_health_check_request(f"{base_url}{endpoint}") as response:
                    content = await response.read()
                    response_time = (time.time() - start_time) * 1000
                        
                        # Capture metrics after request
                        memory_after = self.process.memory_info().rss / (1024*1024)
                        cpu_after = self.process.cpu_percent()
                        
                        metrics.append(APIMetrics(
                            endpoint=endpoint,
                            response_time_ms=round(response_time, 2),
                            status_code=response.status,
                            payload_size_bytes=len(content),
                            memory_before_mb=round(memory_before, 2),
                            memory_after_mb=round(memory_after, 2),
                            cpu_before_percent=cpu_before,
                            cpu_after_percent=cpu_after
                        ))
                        
                except Exception as e:
                    print(f"❌ Failed to benchmark {endpoint}: {e}")
                    metrics.append(APIMetrics(
                        endpoint=endpoint,
                        response_time_ms=-1,
                        status_code=-1,
                        payload_size_bytes=0,
                        memory_before_mb=0,
                        memory_after_mb=0,
                        cpu_before_percent=0,
                        cpu_after_percent=0
                    ))
        
        return metrics
    
    async def benchmark_ml_inference(self) -> List[MLMetrics]:
        """Benchmark ML model inference performance."""
        print("📊 Benchmarking synthetic ML workloads...")
        
        metrics = []
        
        # Synthetic text processing benchmark
        try:
            memory_before = self.process.memory_info().rss / (1024*1024)
            
            # Initialize synthetic text processor
            start_time = time.time()
            import re
            import hashlib
            from collections import Counter
            
            # Simulate feature extraction initialization
            patterns = [
                r'\b\w+\b',  # words
                r'[A-Z][a-z]+',  # capitalized words
                r'\d+',  # numbers
                r'[.!?]+',  # sentence endings
            ]
            compiled_patterns = [re.compile(p) for p in patterns]
            init_time = (time.time() - start_time) * 1000
            
            # Test synthetic inference
            test_texts = [
                "Create a machine learning model for text classification using Python and scikit-learn with cross-validation and hyperparameter tuning.",
                "Develop a web application with React and Node.js for real-time data visualization and analytics dashboard.",
                "Implement a microservices architecture using Docker containers and Kubernetes orchestration platform.",
                "Design a database schema for e-commerce platform with user management and order processing capabilities.",
                "Build a neural network for image recognition using TensorFlow and convolutional neural networks."
            ]
            
            start_time = time.time()
            features_extracted = 0
            
            for text in test_texts:
                # Simulate feature extraction
                word_count = len(text.split())
                char_count = len(text)
                hash_features = hashlib.md5(text.encode()).hexdigest()[:8]
                
                # Pattern matching (simulating linguistic features)
                pattern_counts = {}
                for i, pattern in enumerate(compiled_patterns):
                    matches = pattern.findall(text)
                    pattern_counts[f'pattern_{i}'] = len(matches)
                
                # Word frequency (simulating domain features)
                word_freq = Counter(text.lower().split())
                top_words = dict(word_freq.most_common(5))
                
                features_extracted += len(pattern_counts) + len(top_words) + 3  # +3 for basic counts
            
            inference_time = (time.time() - start_time) * 1000
            
            memory_after = self.process.memory_info().rss / (1024*1024)
            memory_usage = memory_after - memory_before
            
            metrics.append(MLMetrics(
                model_name="synthetic_text_processor",
                initialization_time_ms=round(init_time, 2),
                inference_time_ms=round(inference_time, 2),
                memory_usage_mb=round(memory_usage, 2),
                batch_size=len(test_texts),
                features_extracted=features_extracted
            ))
            
        except Exception as e:
            print(f"❌ Synthetic text processing benchmark failed: {e}")
        
        # Synthetic clustering benchmark
        try:
            memory_before = self.process.memory_info().rss / (1024*1024)
            
            start_time = time.time()
            
            # Simulate clustering initialization
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            init_time = (time.time() - start_time) * 1000
            
            # Create test data for clustering
            test_docs = [
                "machine learning artificial intelligence data science",
                "web development frontend backend javascript python",
                "database sql nosql postgresql mongodb",
                "cloud computing aws azure kubernetes docker",
                "mobile development ios android react native",
                "data analysis pandas numpy matplotlib visualization",
                "security encryption authentication authorization oauth",
                "testing unit integration performance automated",
                "devops continuous integration deployment monitoring",
                "algorithms data structures complexity optimization"
            ]
            
            start_time = time.time()
            
            # Vectorize text
            X = vectorizer.fit_transform(test_docs)
            
            # Perform clustering
            clusters = kmeans.fit_predict(X)
            
            inference_time = (time.time() - start_time) * 1000
            
            memory_after = self.process.memory_info().rss / (1024*1024)
            memory_usage = memory_after - memory_before
            
            metrics.append(MLMetrics(
                model_name="synthetic_clustering_engine",
                initialization_time_ms=round(init_time, 2),
                inference_time_ms=round(inference_time, 2),
                memory_usage_mb=round(memory_usage, 2),
                batch_size=len(test_docs),
                features_extracted=X.shape[1],  # Number of TF-IDF features
                model_size_mb=round(sys.getsizeof(kmeans) / (1024*1024), 2)
            ))
            
        except Exception as e:
            print(f"❌ Synthetic clustering benchmark failed: {e}")
        
        return metrics
    
    async def benchmark_cpu_intensive_tasks(self) -> Dict[str, Any]:
        """Benchmark CPU-intensive computational tasks."""
        print("📊 Benchmarking CPU-intensive workloads...")
        
        results = {}
        
        # Mathematical computation benchmark
        try:
            start_time = time.time()
            cpu_before = psutil.cpu_percent()
            
            # Prime number calculation (CPU intensive)
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            
            primes = [n for n in range(2, 1000) if is_prime(n)]
            prime_time = (time.time() - start_time) * 1000
            cpu_after = psutil.cpu_percent()
            
            results['prime_calculation'] = {
                'execution_time_ms': round(prime_time, 2),
                'cpu_before_percent': cpu_before,
                'cpu_after_percent': cpu_after,
                'primes_found': len(primes)
            }
            
        except Exception as e:
            print(f"❌ Prime calculation benchmark failed: {e}")
        
        # String processing benchmark
        try:
            start_time = time.time()
            memory_before = self.process.memory_info().rss / (1024*1024)
            
            # Generate and process large text
            text_data = []
            for i in range(1000):
                text = f"This is sample text number {i} for benchmarking string processing performance. " * 10
                # String manipulation operations
                processed = text.upper().replace('SAMPLE', 'TEST').split()[:50]
                text_data.append(' '.join(processed))
            
            string_time = (time.time() - start_time) * 1000
            memory_after = self.process.memory_info().rss / (1024*1024)
            
            results['string_processing'] = {
                'execution_time_ms': round(string_time, 2),
                'memory_usage_mb': round(memory_after - memory_before, 2),
                'strings_processed': len(text_data)
            }
            
        except Exception as e:
            print(f"❌ String processing benchmark failed: {e}")
        
        # JSON serialization benchmark
        try:
            start_time = time.time()
            
            # Create complex nested data structure
            test_data = {
                'users': [
                    {
                        'id': i,
                        'name': f'User {i}',
                        'email': f'user{i}@example.com',
                        'metadata': {
                            'created_at': '2025-01-25T12:00:00Z',
                            'preferences': {
                                'theme': 'dark' if i % 2 else 'light',
                                'notifications': True,
                                'features': [f'feature_{j}' for j in range(5)]
                            }
                        }
                    }
                    for i in range(100)
                ]
            }
            
            # Serialize and deserialize
            serialized = json.dumps(test_data)
            deserialized = json.loads(serialized)
            
            json_time = (time.time() - start_time) * 1000
            
            results['json_processing'] = {
                'execution_time_ms': round(json_time, 2),
                'data_size_bytes': len(serialized),
                'objects_processed': len(test_data['users'])
            }
            
        except Exception as e:
            print(f"❌ JSON processing benchmark failed: {e}")
        
        return results
    
    async def benchmark_io_operations(self) -> Dict[str, Any]:
        """Benchmark file I/O operations."""
        print("📊 Benchmarking I/O operations...")
        
        results = {}
        test_file_path = "/tmp/benchmark_test_file.txt"
        
        try:
            # File write benchmark
            start_time = time.time()
            
            test_content = "Sample benchmark data line\n" * 1000
            with open(test_file_path, 'w') as f:
                f.write(test_content)
            
            write_time = (time.time() - start_time) * 1000
            file_size = os.path.getsize(test_file_path)
            
            results['file_write'] = {
                'execution_time_ms': round(write_time, 2),
                'file_size_bytes': file_size,
                'write_speed_mb_per_sec': round((file_size / (1024*1024)) / (write_time / 1000), 2)
            }
            
            # File read benchmark
            start_time = time.time()
            
            with open(test_file_path, 'r') as f:
                content = f.read()
            
            read_time = (time.time() - start_time) * 1000
            
            results['file_read'] = {
                'execution_time_ms': round(read_time, 2),
                'bytes_read': len(content),
                'read_speed_mb_per_sec': round((len(content) / (1024*1024)) / (read_time / 1000), 2)
            }
            
            # Directory operations benchmark
            start_time = time.time()
            test_dir = "/tmp/benchmark_test_dir"
            os.makedirs(test_dir, exist_ok=True)
            
            # Create multiple files
            for i in range(10):
                with open(f"{test_dir}/test_{i}.txt", 'w') as f:
                    f.write(f"Test file {i}")
            
            # List directory
            files = os.listdir(test_dir)
            
            # Clean up
            for file in files:
                os.remove(f"{test_dir}/{file}")
            os.rmdir(test_dir)
            
            dir_time = (time.time() - start_time) * 1000
            
            results['directory_operations'] = {
                'execution_time_ms': round(dir_time, 2),
                'files_created': 10,
                'files_listed': len(files)
            }
            
            # Clean up main test file
            os.remove(test_file_path)
            
        except Exception as e:
            print(f"❌ I/O operations benchmark failed: {e}")
        
        return results
    
    def calculate_summary_statistics(self, metrics_list: List[Any], field: str) -> Dict[str, float]:
        """Calculate summary statistics for a numeric field across metrics."""
        values = [getattr(m, field) for m in metrics_list if hasattr(m, field) and getattr(m, field) is not None and getattr(m, field) >= 0]
        
        if not values:
            return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}
        
        return {
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "std": round(statistics.stdev(values) if len(values) > 1 else 0, 2)
        }
    
    async def run_comprehensive_benchmark(self) -> BaselineReport:
        """Run the complete performance benchmark suite."""
        print("🚀 Starting Comprehensive Performance Baseline Capture")
        print("="*60)
        
        # Capture system metrics
        print("📊 Capturing system metrics...")
        system_metrics = self.get_system_metrics()
        
        # Benchmark database
        database_metrics = await self.benchmark_database()
        
        # Benchmark Redis
        redis_metrics = await self.benchmark_redis()
        
        # Benchmark API endpoints (skip if no server running)
        api_metrics = []
        try:
            api_metrics = await self.benchmark_api_endpoints()
        except Exception as e:
            print(f"⚠️  API benchmark skipped (server may not be running): {e}")
        
        # Benchmark ML inference
        ml_metrics = await self.benchmark_ml_inference()
        
        # Benchmark CPU intensive tasks
        cpu_benchmarks = await self.benchmark_cpu_intensive_tasks()
        
        # Benchmark I/O operations  
        io_benchmarks = await self.benchmark_io_operations()
        
        # Calculate summary statistics
        summary = {
            "benchmark_duration_seconds": round(time.time() - self.start_time, 2),
            "total_memory_used_mb": round(system_metrics.memory_used_gb * 1024, 2),
            "cpu_utilization_percent": system_metrics.cpu_percent,
            "disk_usage_percent": system_metrics.disk_percent
        }
        
        if api_metrics:
            summary["api_response_times"] = self.calculate_summary_statistics(api_metrics, "response_time_ms")
        
        if ml_metrics:
            summary["ml_inference_times"] = self.calculate_summary_statistics(ml_metrics, "inference_time_ms")
            summary["ml_initialization_times"] = self.calculate_summary_statistics(ml_metrics, "initialization_time_ms")
        
        if database_metrics:
            summary["database_performance"] = {
                "connection_time_ms": database_metrics.connection_time_ms,
                "query_time_ms": database_metrics.query_time_ms,
                "transaction_time_ms": database_metrics.transaction_time_ms
            }
        
        if redis_metrics:
            summary["redis_performance"] = {
                "ping_time_ms": redis_metrics.ping_time_ms,
                "avg_operation_time_ms": round((redis_metrics.set_time_ms + redis_metrics.get_time_ms + redis_metrics.delete_time_ms) / 3, 2)
            }
        
        # Add CPU and I/O benchmark summaries
        if cpu_benchmarks:
            summary["cpu_performance"] = {
                "prime_calculation_ms": cpu_benchmarks.get('prime_calculation', {}).get('execution_time_ms', 0),
                "string_processing_ms": cpu_benchmarks.get('string_processing', {}).get('execution_time_ms', 0),
                "json_processing_ms": cpu_benchmarks.get('json_processing', {}).get('execution_time_ms', 0)
            }
        
        if io_benchmarks:
            summary["io_performance"] = {
                "file_write_speed_mb_per_sec": io_benchmarks.get('file_write', {}).get('write_speed_mb_per_sec', 0),
                "file_read_speed_mb_per_sec": io_benchmarks.get('file_read', {}).get('read_speed_mb_per_sec', 0),
                "directory_ops_ms": io_benchmarks.get('directory_operations', {}).get('execution_time_ms', 0)
            }
        
        return BaselineReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            system_metrics=system_metrics,
            database_metrics=database_metrics,
            redis_metrics=redis_metrics,
            api_metrics=api_metrics,
            ml_metrics=ml_metrics,
            cpu_benchmarks=cpu_benchmarks,
            io_benchmarks=io_benchmarks,
            summary=summary
        )

def save_baseline_report(report: BaselineReport, output_path: str = None):
    """Save the baseline report to JSON file."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/Users/lukemckenzie/prompt-improver/performance_baseline_{timestamp}.json"
    
    # Convert dataclass to dict
    report_dict = asdict(report)
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)
    
    print(f"📄 Baseline report saved to: {output_path}")
    return output_path

def print_baseline_summary(report: BaselineReport):
    """Print a human-readable summary of the baseline report."""
    print("\n" + "="*60)
    print("📊 PERFORMANCE BASELINE SUMMARY")
    print("="*60)
    
    # System metrics
    print(f"\n🖥️  SYSTEM METRICS")
    print(f"   CPU Usage:        {report.system_metrics.cpu_percent}%")
    print(f"   Memory Usage:     {report.system_metrics.memory_percent}% ({report.system_metrics.memory_used_gb}GB)")
    print(f"   Disk Usage:       {report.system_metrics.disk_percent}% ({report.system_metrics.disk_free_gb}GB free)")
    print(f"   Process Count:    {report.system_metrics.process_count}")
    
    # Database metrics
    if report.database_metrics:
        print(f"\n🗄️  DATABASE PERFORMANCE")
        print(f"   Connection Time:  {report.database_metrics.connection_time_ms:.2f}ms")
        print(f"   Query Time:       {report.database_metrics.query_time_ms:.2f}ms")
        print(f"   Transaction Time: {report.database_metrics.transaction_time_ms:.2f}ms")
        if report.database_metrics.select_time_ms:
            print(f"   Complex Query:    {report.database_metrics.select_time_ms:.2f}ms")
    
    # Redis metrics
    if report.redis_metrics:
        print(f"\n🔄 REDIS PERFORMANCE")
        print(f"   Ping Time:        {report.redis_metrics.ping_time_ms:.2f}ms")
        print(f"   Set Operation:    {report.redis_metrics.set_time_ms:.2f}ms")
        print(f"   Get Operation:    {report.redis_metrics.get_time_ms:.2f}ms")
        print(f"   Memory Usage:     {report.redis_metrics.memory_usage_mb:.2f}MB")
        print(f"   Connected Clients: {report.redis_metrics.connected_clients}")
    
    # API metrics
    if report.api_metrics:
        print(f"\n🌐 API ENDPOINT PERFORMANCE")
        successful_apis = [m for m in report.api_metrics if m.status_code == 200]
        if successful_apis:
            avg_response_time = statistics.mean([m.response_time_ms for m in successful_apis])
            print(f"   Average Response: {avg_response_time:.2f}ms")
            print(f"   Successful Calls: {len(successful_apis)}/{len(report.api_metrics)}")
            
            for api in successful_apis:
                print(f"   {api.endpoint:15} {api.response_time_ms:6.2f}ms ({api.status_code})")
    
    # ML metrics
    if report.ml_metrics:
        print(f"\n🤖 MACHINE LEARNING PERFORMANCE")
        for ml in report.ml_metrics:
            print(f"   {ml.model_name}:")
            print(f"     Initialization: {ml.initialization_time_ms:.2f}ms")
            print(f"     Inference:      {ml.inference_time_ms:.2f}ms")
            print(f"     Memory Usage:   {ml.memory_usage_mb:.2f}MB")
            print(f"     Features:       {ml.features_extracted}")
    
    # CPU benchmarks
    if report.cpu_benchmarks:
        print(f"\n⚡ CPU PERFORMANCE BENCHMARKS")
        if 'prime_calculation' in report.cpu_benchmarks:
            pc = report.cpu_benchmarks['prime_calculation']
            print(f"   Prime Calculation: {pc['execution_time_ms']:.2f}ms ({pc['primes_found']} primes)")
        if 'string_processing' in report.cpu_benchmarks:
            sp = report.cpu_benchmarks['string_processing']
            print(f"   String Processing: {sp['execution_time_ms']:.2f}ms ({sp['strings_processed']} strings)")
        if 'json_processing' in report.cpu_benchmarks:
            jp = report.cpu_benchmarks['json_processing']
            print(f"   JSON Processing:   {jp['execution_time_ms']:.2f}ms ({jp['objects_processed']} objects)")
    
    # I/O benchmarks
    if report.io_benchmarks:
        print(f"\n💾 I/O PERFORMANCE BENCHMARKS")
        if 'file_write' in report.io_benchmarks:
            fw = report.io_benchmarks['file_write']
            print(f"   File Write:       {fw['execution_time_ms']:.2f}ms ({fw['write_speed_mb_per_sec']:.2f} MB/s)")
        if 'file_read' in report.io_benchmarks:
            fr = report.io_benchmarks['file_read']
            print(f"   File Read:        {fr['execution_time_ms']:.2f}ms ({fr['read_speed_mb_per_sec']:.2f} MB/s)")
        if 'directory_operations' in report.io_benchmarks:
            do = report.io_benchmarks['directory_operations']
            print(f"   Directory Ops:    {do['execution_time_ms']:.2f}ms ({do['files_created']} files)")
    
    # Summary statistics
    print(f"\n📈 PERFORMANCE TARGETS")
    print(f"   Target Response Time:    <200ms")
    print(f"   Target Memory Usage:     <1000MB")
    print(f"   Target CPU Usage:        <80%")
    print(f"   Target DB Query Time:    <50ms")
    
    # Performance assessment
    print(f"\n✅ BASELINE ASSESSMENT")
    if report.database_metrics and report.database_metrics.query_time_ms < 50:
        print("   Database Performance: ✅ Excellent (<50ms)")
    elif report.database_metrics and report.database_metrics.query_time_ms < 200:
        print("   Database Performance: ⚠️  Good (50-200ms)")
    elif report.database_metrics:
        print("   Database Performance: ❌ Needs improvement (>200ms)")
    
    if report.system_metrics.cpu_percent < 50:
        print("   CPU Utilization:     ✅ Excellent (<50%)")
    elif report.system_metrics.cpu_percent < 80:
        print("   CPU Utilization:     ⚠️  Good (50-80%)")
    else:
        print("   CPU Utilization:     ❌ High (>80%)")
    
    if report.system_metrics.memory_percent < 70:
        print("   Memory Utilization:  ✅ Excellent (<70%)")
    elif report.system_metrics.memory_percent < 90:
        print("   Memory Utilization:  ⚠️  Good (70-90%)")
    else:
        print("   Memory Utilization:  ❌ High (>90%)")

async def main():
    """Main execution function."""
    benchmark = PerformanceBenchmark()
    
    try:
        # Run comprehensive benchmark
        report = await benchmark.run_comprehensive_benchmark()
        
        # Save report
        output_path = save_baseline_report(report)
        
        # Print summary
        print_baseline_summary(report)
        
        print(f"\n✅ Performance baseline capture completed!")
        print(f"📄 Full report saved to: {output_path}")
        print(f"⏱️  Total benchmark time: {report.summary['benchmark_duration_seconds']}s")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set up event loop policy for better performance on Unix systems
    if sys.platform != 'win32':
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ImportError:
            pass
    
    # Run the benchmark
    result = asyncio.run(main())
    
    if result:
        print(f"\n🎯 Use this baseline for monitoring performance regressions")
        print(f"📊 Compare future benchmarks against: {result}")
    else:
        sys.exit(1)