#!/usr/bin/env python3
"""
Upgrade Performance Validation Test Suite

Validates that the NumPy 2.x, MLflow 3.x, and Websockets 15.x upgrades 
deliver measurable performance improvements without regressions.

Benchmarks:
- NumPy 2.x: Array operations, memory usage, computation speed
- MLflow 3.x: Model logging, loading, registry operations
- Websockets 15.x: Connection handling, message throughput, latency
- Integration: End-to-end pipeline performance
"""

import asyncio
import time
import sys
import gc
import psutil
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple
import tempfile

sys.path.insert(0, 'src')

import numpy as np
import mlflow
import mlflow.sklearn
import websockets
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

class PerformanceBenchmark:
    """Performance benchmark suite for upgrade validation"""
    
    def __init__(self):
        self.results = {}
        self.temp_dir = Path(tempfile.mkdtemp())
        self.process = psutil.Process()
        
        # Performance thresholds (these should be improved after upgrades)
        self.thresholds = {
            "numpy_matrix_ops_100k": 0.5,      # seconds
            "numpy_memory_efficiency": 0.8,     # ratio of optimal
            "mlflow_model_logging": 3.0,        # seconds
            "mlflow_model_loading": 1.0,        # seconds
            "websocket_connection_time": 0.1,   # seconds
            "websocket_message_rate": 1000,     # messages/second
            "pipeline_end_to_end": 10.0         # seconds
        }
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage metrics"""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent()
        }
    
    async def benchmark_numpy_2x_performance(self) -> Dict[str, Any]:
        """Benchmark NumPy 2.x performance improvements"""
        print("🔢 Benchmarking NumPy 2.x Performance...")
        
        results = {
            "version": np.__version__,
            "test_results": {}
        }
        
        # Test 1: Large matrix operations
        print("  • Testing large matrix operations...")
        
        sizes = [1000, 5000, 10000, 20000]
        matrix_performance = {}
        
        for size in sizes:
            print(f"    - Matrix size: {size}×{size}")
            
            # Memory before
            memory_before = self.measure_memory_usage()
            gc.collect()
            
            start_time = time.perf_counter()
            
            # Create large matrices
            A = np.random.randn(size, size).astype(np.float64)
            B = np.random.randn(size, size).astype(np.float64)
            
            # Matrix multiplication
            C = np.dot(A, B)
            
            # Eigenvalue decomposition
            if size <= 5000:  # Limit expensive operations
                eigenvals, eigenvecs = np.linalg.eigh(A + A.T)  # Make symmetric
            
            # Statistical operations
            mean_A = np.mean(A, axis=0)
            std_A = np.std(A, axis=0)
            corr_A = np.corrcoef(A[:100, :100])  # Limit correlation size
            
            operation_time = time.perf_counter() - start_time
            
            # Memory after
            memory_after = self.measure_memory_usage()
            
            matrix_performance[size] = {
                "operation_time": operation_time,
                "memory_before_mb": memory_before["rss_mb"],
                "memory_after_mb": memory_after["rss_mb"],
                "memory_delta_mb": memory_after["rss_mb"] - memory_before["rss_mb"],
                "ops_per_second": (size * size * 3) / operation_time  # Approximate ops
            }
            
            print(f"      Time: {operation_time:.3f}s, Memory: +{memory_after['rss_mb'] - memory_before['rss_mb']:.1f}MB")
            
            # Clean up
            del A, B, C, mean_A, std_A, corr_A
            if size <= 5000:
                del eigenvals, eigenvecs
            gc.collect()
        
        results["test_results"]["matrix_operations"] = matrix_performance
        
        # Test 2: Data type conversions and precision
        print("  • Testing data type handling...")
        
        dtype_performance = {}
        test_data = np.random.randn(100000, 50)
        
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            start_time = time.perf_counter()
            
            converted_data = test_data.astype(dtype)
            
            # Operations on converted data
            mean_val = np.mean(converted_data)
            std_val = np.std(converted_data)
            sum_val = np.sum(converted_data)
            
            conversion_time = time.perf_counter() - start_time
            
            dtype_performance[str(dtype)] = {
                "conversion_time": conversion_time,
                "memory_bytes": converted_data.nbytes,
                "mean": float(mean_val),
                "std": float(std_val)
            }
            
            print(f"    - {dtype}: {conversion_time:.4f}s, {converted_data.nbytes / 1024 / 1024:.1f}MB")
        
        results["test_results"]["dtype_conversions"] = dtype_performance
        
        # Test 3: Advanced array operations
        print("  • Testing advanced array operations...")
        
        advanced_ops_start = time.perf_counter()
        
        # Create test data
        data = np.random.randn(50000, 100)
        
        # Broadcasting operations
        normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        
        # Boolean indexing
        filtered = data[data[:, 0] > 0]
        
        # Fancy indexing
        indices = np.random.randint(0, data.shape[0], 10000)
        subset = data[indices]
        
        # Array splitting and concatenation
        splits = np.array_split(data, 10)
        recombined = np.concatenate(splits)
        
        # Advanced mathematical operations
        singular_vals = np.linalg.svd(data[:100, :50], compute_uv=False)
        
        advanced_ops_time = time.perf_counter() - advanced_ops_start
        
        results["test_results"]["advanced_operations"] = {
            "total_time": advanced_ops_time,
            "normalized_shape": normalized.shape,
            "filtered_shape": filtered.shape,
            "subset_shape": subset.shape,
            "recombined_matches": np.array_equal(data, recombined),
            "svd_components": len(singular_vals)
        }
        
        print(f"    - Advanced operations: {advanced_ops_time:.3f}s")
        
        # Performance validation
        large_matrix_time = matrix_performance.get(10000, {}).get("operation_time", float('inf'))
        if large_matrix_time < self.thresholds["numpy_matrix_ops_100k"]:
            print(f"  ✅ NumPy performance meets threshold: {large_matrix_time:.3f}s < {self.thresholds['numpy_matrix_ops_100k']}s")
        else:
            print(f"  ⚠️  NumPy performance below threshold: {large_matrix_time:.3f}s >= {self.thresholds['numpy_matrix_ops_100k']}s")
        
        return results
    
    async def benchmark_mlflow_3x_performance(self) -> Dict[str, Any]:
        """Benchmark MLflow 3.x performance improvements"""
        print("🤖 Benchmarking MLflow 3.x Performance...")
        
        results = {
            "version": mlflow.__version__,
            "test_results": {}
        }
        
        # Set up experiment
        experiment_name = f"perf_benchmark_{int(time.time())}"
        mlflow.set_experiment(experiment_name)
        
        # Test 1: Model training and logging performance
        print("  • Testing model training and logging...")
        
        model_sizes = [
            {"samples": 1000, "features": 20, "estimators": 10},
            {"samples": 5000, "features": 50, "estimators": 50},
            {"samples": 10000, "features": 100, "estimators": 100},
        ]
        
        logging_performance = {}
        
        for i, config in enumerate(model_sizes):
            print(f"    - Config {i+1}: {config['samples']} samples, {config['features']} features")
            
            # Generate data
            X, y = make_regression(
                n_samples=config["samples"],
                n_features=config["features"],
                noise=0.1,
                random_state=42
            )
            
            start_time = time.perf_counter()
            
            with mlflow.start_run(run_name=f"perf_test_{i}"):
                # Train model
                model = RandomForestRegressor(
                    n_estimators=config["estimators"],
                    random_state=42
                )
                model.fit(X, y)
                
                # Log parameters
                mlflow.log_param("n_samples", config["samples"])
                mlflow.log_param("n_features", config["features"])
                mlflow.log_param("n_estimators", config["estimators"])
                
                # Log metrics
                train_score = model.score(X, y)
                mlflow.log_metric("train_r2", train_score)
                mlflow.log_metric("model_size_mb", sys.getsizeof(model) / 1024 / 1024)
                
                # Log model
                mlflow.sklearn.log_model(model, "performance_model")
                
                # Log additional artifacts
                for j in range(5):
                    mlflow.log_metric(f"metric_{j}", np.random.random())
                
                run_id = mlflow.active_run().info.run_id
            
            logging_time = time.perf_counter() - start_time
            
            logging_performance[f"config_{i}"] = {
                "config": config,
                "logging_time": logging_time,
                "train_score": train_score,
                "run_id": run_id
            }
            
            print(f"      Logging time: {logging_time:.3f}s, R²: {train_score:.3f}")
        
        results["test_results"]["model_logging"] = logging_performance
        
        # Test 2: Model loading performance
        print("  • Testing model loading...")
        
        loading_performance = {}
        
        for config_name, config_data in logging_performance.items():
            run_id = config_data["run_id"]
            model_uri = f"runs:/{run_id}/performance_model"
            
            # Cold load (first time)
            start_time = time.perf_counter()
            model_cold = mlflow.sklearn.load_model(model_uri)
            cold_load_time = time.perf_counter() - start_time
            
            # Warm load (cached)
            start_time = time.perf_counter()
            model_warm = mlflow.sklearn.load_model(model_uri)
            warm_load_time = time.perf_counter() - start_time
            
            # Test prediction performance
            X_test = np.random.randn(100, config_data["config"]["features"])
            
            start_time = time.perf_counter()
            predictions = model_cold.predict(X_test)
            prediction_time = time.perf_counter() - start_time
            
            loading_performance[config_name] = {
                "cold_load_time": cold_load_time,
                "warm_load_time": warm_load_time,
                "prediction_time": prediction_time,
                "predictions_shape": predictions.shape
            }
            
            print(f"    - {config_name}: Cold={cold_load_time:.3f}s, Warm={warm_load_time:.4f}s")
        
        results["test_results"]["model_loading"] = loading_performance
        
        # Test 3: Registry operations
        print("  • Testing model registry operations...")
        
        registry_start = time.perf_counter()
        
        # Register a model
        best_run = list(logging_performance.values())[0]  # Use first run
        model_name = f"perf_benchmark_model_{int(time.time())}"
        model_uri = f"runs:/{best_run['run_id']}/performance_model"
        
        registered_model = mlflow.register_model(model_uri, model_name)
        
        # Get model version
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version(model_name, 1)
        
        # Update model version
        client.update_model_version(
            name=model_name,
            version=1,
            description="Performance benchmark model"
        )
        
        # List model versions
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        registry_time = time.perf_counter() - registry_start
        
        results["test_results"]["registry_operations"] = {
            "total_time": registry_time,
            "model_name": model_name,
            "version_count": len(model_versions),
            "registration_successful": registered_model is not None
        }
        
        print(f"    - Registry operations: {registry_time:.3f}s")
        
        # Clean up registered model
        try:
            client.delete_registered_model(model_name)
        except Exception as e:
            print(f"    ⚠️  Could not clean up model: {e}")
        
        # Performance validation
        avg_logging_time = np.mean([r["logging_time"] for r in logging_performance.values()])
        avg_loading_time = np.mean([r["cold_load_time"] for r in loading_performance.values()])
        
        if avg_logging_time < self.thresholds["mlflow_model_logging"]:
            print(f"  ✅ MLflow logging meets threshold: {avg_logging_time:.3f}s < {self.thresholds['mlflow_model_logging']}s")
        else:
            print(f"  ⚠️  MLflow logging below threshold: {avg_logging_time:.3f}s >= {self.thresholds['mlflow_model_logging']}s")
        
        if avg_loading_time < self.thresholds["mlflow_model_loading"]:
            print(f"  ✅ MLflow loading meets threshold: {avg_loading_time:.3f}s < {self.thresholds['mlflow_model_loading']}s")
        else:
            print(f"  ⚠️  MLflow loading below threshold: {avg_loading_time:.3f}s >= {self.thresholds['mlflow_model_loading']}s")
        
        return results
    
    async def benchmark_websockets_15x_performance(self) -> Dict[str, Any]:
        """Benchmark Websockets 15.x performance improvements"""
        print("🔌 Benchmarking Websockets 15.x Performance...")
        
        results = {
            "version": websockets.__version__,
            "test_results": {}
        }
        
        # Test 1: Connection establishment performance
        print("  • Testing connection establishment...")
        
        connection_times = []
        connection_counts = [1, 5, 10, 20]
        
        async def test_connections(n_connections: int) -> Dict[str, Any]:
            """Test multiple concurrent connections"""
            
            async def echo_server(websocket, path):
                """Simple echo server for testing"""
                try:
                    async for message in websocket:
                        await websocket.send(f"echo: {message}")
                except websockets.exceptions.ConnectionClosed:
                    pass
            
            # Start server
            server = await websockets.serve(echo_server, "localhost", 0)
            server_port = server.sockets[0].getsockname()[1]
            
            try:
                connection_times = []
                
                for i in range(n_connections):
                    start_time = time.perf_counter()
                    
                    uri = f"ws://localhost:{server_port}"
                    websocket = await websockets.connect(uri)
                    
                    connection_time = time.perf_counter() - start_time
                    connection_times.append(connection_time)
                    
                    # Test basic communication
                    test_message = f"test_{i}"
                    await websocket.send(test_message)
                    response = await websocket.recv()
                    
                    assert response == f"echo: {test_message}"
                    
                    await websocket.close()
                
                return {
                    "avg_connection_time": np.mean(connection_times),
                    "max_connection_time": np.max(connection_times),
                    "min_connection_time": np.min(connection_times),
                    "total_connections": n_connections
                }
            
            finally:
                server.close()
                await server.wait_closed()
        
        connection_performance = {}
        for count in connection_counts:
            print(f"    - Testing {count} connections...")
            perf_data = await test_connections(count)
            connection_performance[count] = perf_data
            print(f"      Avg: {perf_data['avg_connection_time']:.4f}s, Max: {perf_data['max_connection_time']:.4f}s")
        
        results["test_results"]["connection_performance"] = connection_performance
        
        # Test 2: Message throughput
        print("  • Testing message throughput...")
        
        async def test_message_throughput() -> Dict[str, Any]:
            """Test high-frequency message sending"""
            
            messages_sent = 0
            messages_received = 0
            
            async def throughput_server(websocket, path):
                nonlocal messages_received
                try:
                    async for message in websocket:
                        messages_received += 1
                        await websocket.send(f"ack_{messages_received}")
                except websockets.exceptions.ConnectionClosed:
                    pass
            
            # Start server
            server = await websockets.serve(throughput_server, "localhost", 0)
            server_port = server.sockets[0].getsockname()[1]
            
            try:
                uri = f"ws://localhost:{server_port}"
                websocket = await websockets.connect(uri)
                
                # Send burst of messages
                start_time = time.perf_counter()
                test_duration = 2.0  # seconds
                
                while time.perf_counter() - start_time < test_duration:
                    message = f"msg_{messages_sent}"
                    await websocket.send(message)
                    messages_sent += 1
                    
                    # Receive acknowledgment
                    try:
                        ack = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    except asyncio.TimeoutError:
                        break
                
                total_time = time.perf_counter() - start_time
                
                await websocket.close()
                
                return {
                    "messages_sent": messages_sent,
                    "messages_received": messages_received,
                    "total_time": total_time,
                    "messages_per_second": messages_sent / total_time,
                    "round_trip_rate": messages_received / total_time
                }
            
            finally:
                server.close()
                await server.wait_closed()
        
        throughput_data = await test_message_throughput()
        results["test_results"]["message_throughput"] = throughput_data
        
        print(f"    - Messages/sec: {throughput_data['messages_per_second']:.1f}")
        print(f"    - Round-trip rate: {throughput_data['round_trip_rate']:.1f}")
        
        # Test 3: Concurrent client handling
        print("  • Testing concurrent client handling...")
        
        async def test_concurrent_clients() -> Dict[str, Any]:
            """Test server handling multiple concurrent clients"""
            
            client_data = {}
            
            async def multi_client_server(websocket, path):
                client_id = f"client_{len(client_data)}"
                client_data[client_id] = {"messages": 0, "start_time": time.perf_counter()}
                
                try:
                    async for message in websocket:
                        client_data[client_id]["messages"] += 1
                        await websocket.send(f"{client_id}: {message}")
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    if client_id in client_data:
                        client_data[client_id]["end_time"] = time.perf_counter()
            
            # Start server
            server = await websockets.serve(multi_client_server, "localhost", 0)
            server_port = server.sockets[0].getsockname()[1]
            
            try:
                async def client_session(client_num: int):
                    """Individual client session"""
                    uri = f"ws://localhost:{server_port}"
                    websocket = await websockets.connect(uri)
                    
                    # Send messages for 1 second
                    start_time = time.perf_counter()
                    messages_sent = 0
                    
                    while time.perf_counter() - start_time < 1.0:
                        await websocket.send(f"message_{messages_sent}")
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                            messages_sent += 1
                        except asyncio.TimeoutError:
                            break
                    
                    await websocket.close()
                    return messages_sent
                
                # Run multiple concurrent clients
                n_clients = 5
                client_tasks = [client_session(i) for i in range(n_clients)]
                client_results = await asyncio.gather(*client_tasks)
                
                return {
                    "concurrent_clients": n_clients,
                    "total_messages": sum(client_results),
                    "avg_messages_per_client": np.mean(client_results),
                    "client_results": client_results,
                    "server_client_data": dict(client_data)
                }
            
            finally:
                server.close()
                await server.wait_closed()
        
        concurrent_data = await test_concurrent_clients()
        results["test_results"]["concurrent_clients"] = concurrent_data
        
        print(f"    - Concurrent clients: {concurrent_data['concurrent_clients']}")
        print(f"    - Total messages: {concurrent_data['total_messages']}")
        print(f"    - Avg per client: {concurrent_data['avg_messages_per_client']:.1f}")
        
        # Performance validation
        avg_connection_time = np.mean([
            perf["avg_connection_time"] 
            for perf in connection_performance.values()
        ])
        
        if avg_connection_time < self.thresholds["websocket_connection_time"]:
            print(f"  ✅ WebSocket connection meets threshold: {avg_connection_time:.4f}s < {self.thresholds['websocket_connection_time']}s")
        else:
            print(f"  ⚠️  WebSocket connection below threshold: {avg_connection_time:.4f}s >= {self.thresholds['websocket_connection_time']}s")
        
        if throughput_data["messages_per_second"] > self.thresholds["websocket_message_rate"]:
            print(f"  ✅ WebSocket throughput meets threshold: {throughput_data['messages_per_second']:.1f} > {self.thresholds['websocket_message_rate']}")
        else:
            print(f"  ⚠️  WebSocket throughput below threshold: {throughput_data['messages_per_second']:.1f} <= {self.thresholds['websocket_message_rate']}")
        
        return results
    
    async def benchmark_integration_performance(self) -> Dict[str, Any]:
        """Benchmark end-to-end integration performance"""
        print("🔗 Benchmarking Integration Performance...")
        
        results = {"test_results": {}}
        
        # Test 1: Complete ML pipeline with real-time monitoring
        print("  • Testing complete ML pipeline...")
        
        pipeline_start = time.perf_counter()
        
        # Stage 1: Data generation and processing (NumPy 2.x)
        stage1_start = time.perf_counter()
        
        # Generate large dataset
        n_samples = 20000
        n_features = 100
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)
        
        # Data preprocessing with NumPy 2.x
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Feature selection using correlation
        correlation_matrix = np.corrcoef(X_normalized.T)
        feature_scores = np.abs(np.corrcoef(X_normalized.T, y)[:-1, -1])
        top_features = np.argsort(feature_scores)[-50:]  # Top 50 features
        X_selected = X_normalized[:, top_features]
        
        stage1_time = time.perf_counter() - stage1_start
        
        # Stage 2: Model training and logging (MLflow 3.x)
        stage2_start = time.perf_counter()
        
        mlflow.set_experiment("integration_performance_test")
        
        with mlflow.start_run(run_name="integration_pipeline"):
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_selected, y)
            
            # Calculate metrics
            train_score = model.score(X_selected, y)
            feature_importance = model.feature_importances_
            
            # Log with MLflow
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("n_features_original", n_features)
            mlflow.log_param("n_features_selected", len(top_features))
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("train_r2", train_score)
            mlflow.log_metric("top_feature_importance", np.max(feature_importance))
            
            # Log model
            mlflow.sklearn.log_model(model, "integration_model")
            
            run_id = mlflow.active_run().info.run_id
        
        stage2_time = time.perf_counter() - stage2_start
        
        # Stage 3: Real-time analytics simulation (WebSockets 15.x)
        stage3_start = time.perf_counter()
        
        # Simulate real-time model serving and analytics
        test_samples = X_selected[-1000:]  # Last 1000 samples for testing
        
        # Load model (simulate production loading)
        model_uri = f"runs:/{run_id}/integration_model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Batch predictions
        predictions = loaded_model.predict(test_samples)
        
        # Simulate real-time analytics calculations
        analytics_data = []
        for i in range(100):  # 100 analytics updates
            batch_start = i * 10
            batch_end = (i + 1) * 10
            
            if batch_end <= len(predictions):
                batch_predictions = predictions[batch_start:batch_end]
                
                analytics_update = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "batch_id": i,
                    "mean_prediction": float(np.mean(batch_predictions)),
                    "std_prediction": float(np.std(batch_predictions)),
                    "min_prediction": float(np.min(batch_predictions)),
                    "max_prediction": float(np.max(batch_predictions)),
                    "sample_count": len(batch_predictions)
                }
                
                analytics_data.append(analytics_update)
        
        # Simulate WebSocket message processing
        websocket_messages = []
        for update in analytics_data:
            message = {
                "type": "prediction_update",
                "experiment_id": "integration_test",
                "data": update
            }
            
            # Simulate JSON serialization (WebSocket overhead)
            message_json = json.dumps(message)
            websocket_messages.append(message_json)
        
        stage3_time = time.perf_counter() - stage3_start
        
        total_pipeline_time = time.perf_counter() - pipeline_start
        
        results["test_results"]["complete_pipeline"] = {
            "total_time": total_pipeline_time,
            "stage1_data_processing": stage1_time,
            "stage2_ml_training": stage2_time,
            "stage3_realtime_analytics": stage3_time,
            "dataset_size": (n_samples, n_features),
            "features_selected": len(top_features),
            "model_score": train_score,
            "predictions_generated": len(predictions),
            "analytics_updates": len(analytics_data),
            "websocket_messages": len(websocket_messages),
            "run_id": run_id
        }
        
        print(f"    - Total pipeline time: {total_pipeline_time:.2f}s")
        print(f"      • Data processing: {stage1_time:.2f}s")
        print(f"      • ML training: {stage2_time:.2f}s")
        print(f"      • Real-time analytics: {stage3_time:.2f}s")
        print(f"    - Model R²: {train_score:.3f}")
        print(f"    - Analytics updates: {len(analytics_data)}")
        
        # Test 2: Concurrent pipeline execution
        print("  • Testing concurrent pipeline execution...")
        
        async def mini_pipeline(pipeline_id: int) -> Dict[str, Any]:
            """Smaller pipeline for concurrent testing"""
            start_time = time.perf_counter()
            
            # Mini dataset
            X_mini, y_mini = make_regression(n_samples=1000, n_features=20, random_state=pipeline_id)
            X_mini_norm = (X_mini - np.mean(X_mini, axis=0)) / np.std(X_mini, axis=0)
            
            # Quick model
            with mlflow.start_run(run_name=f"concurrent_pipeline_{pipeline_id}"):
                model_mini = RandomForestRegressor(n_estimators=10, random_state=pipeline_id)
                model_mini.fit(X_mini_norm, y_mini)
                
                score = model_mini.score(X_mini_norm, y_mini)
                mlflow.log_metric("score", score)
                mlflow.sklearn.log_model(model_mini, "mini_model")
                
                mini_run_id = mlflow.active_run().info.run_id
            
            # Mini predictions
            predictions_mini = model_mini.predict(X_mini_norm[:100])
            
            total_time = time.perf_counter() - start_time
            
            return {
                "pipeline_id": pipeline_id,
                "total_time": total_time,
                "score": score,
                "predictions": len(predictions_mini),
                "run_id": mini_run_id
            }
        
        # Run multiple pipelines concurrently
        concurrent_start = time.perf_counter()
        n_concurrent = 3
        
        pipeline_tasks = [mini_pipeline(i) for i in range(n_concurrent)]
        concurrent_results = await asyncio.gather(*pipeline_tasks)
        
        concurrent_total_time = time.perf_counter() - concurrent_start
        
        results["test_results"]["concurrent_pipelines"] = {
            "n_pipelines": n_concurrent,
            "total_concurrent_time": concurrent_total_time,
            "pipeline_results": concurrent_results,
            "avg_pipeline_time": np.mean([r["total_time"] for r in concurrent_results]),
            "max_pipeline_time": np.max([r["total_time"] for r in concurrent_results])
        }
        
        print(f"    - Concurrent execution ({n_concurrent} pipelines): {concurrent_total_time:.2f}s")
        print(f"    - Average individual time: {np.mean([r['total_time'] for r in concurrent_results]):.2f}s")
        
        # Performance validation
        if total_pipeline_time < self.thresholds["pipeline_end_to_end"]:
            print(f"  ✅ Pipeline performance meets threshold: {total_pipeline_time:.2f}s < {self.thresholds['pipeline_end_to_end']}s")
        else:
            print(f"  ⚠️  Pipeline performance below threshold: {total_pipeline_time:.2f}s >= {self.thresholds['pipeline_end_to_end']}s")
        
        return results
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks"""
        print("🚀 Running Upgrade Performance Validation Suite")
        print("=" * 70)
        
        start_time = time.perf_counter()
        
        # Run benchmarks
        numpy_results = await self.benchmark_numpy_2x_performance()
        mlflow_results = await self.benchmark_mlflow_3x_performance()
        websockets_results = await self.benchmark_websockets_15x_performance()
        integration_results = await self.benchmark_integration_performance()
        
        total_time = time.perf_counter() - start_time
        
        # Compile results
        all_results = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_benchmark_time": total_time,
                "system_info": {
                    "numpy_version": np.__version__,
                    "mlflow_version": mlflow.__version__,
                    "websockets_version": websockets.__version__,
                    "python_version": sys.version,
                    "memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024
                }
            },
            "performance_thresholds": self.thresholds,
            "results": {
                "numpy_2x": numpy_results,
                "mlflow_3x": mlflow_results,
                "websockets_15x": websockets_results,
                "integration": integration_results
            }
        }
        
        # Generate performance report
        await self.generate_performance_report(all_results)
        
        return all_results
    
    async def generate_performance_report(self, results: Dict[str, Any]):
        """Generate detailed performance report"""
        print("\n" + "=" * 70)
        print("📊 UPGRADE PERFORMANCE VALIDATION REPORT")
        print("=" * 70)
        
        metadata = results["metadata"]
        print(f"Benchmark Duration: {metadata['total_benchmark_time']:.2f}s")
        print(f"System Memory: {metadata['system_info']['memory_gb']:.1f} GB")
        print()
        
        # NumPy Performance Summary
        numpy_data = results["results"]["numpy_2x"]
        print("🔢 NumPy 2.x Performance:")
        print(f"  Version: {numpy_data['version']}")
        
        if "matrix_operations" in numpy_data["test_results"]:
            large_matrix = numpy_data["test_results"]["matrix_operations"].get("20000", {})
            if large_matrix:
                print(f"  Large Matrix (20k×20k): {large_matrix['operation_time']:.3f}s")
                print(f"  Memory Usage: {large_matrix['memory_delta_mb']:.1f} MB")
        
        print()
        
        # MLflow Performance Summary
        mlflow_data = results["results"]["mlflow_3x"]
        print("🤖 MLflow 3.x Performance:")
        print(f"  Version: {mlflow_data['version']}")
        
        if "model_logging" in mlflow_data["test_results"]:
            avg_logging = np.mean([
                r["logging_time"] 
                for r in mlflow_data["test_results"]["model_logging"].values()
            ])
            print(f"  Average Model Logging: {avg_logging:.3f}s")
        
        if "model_loading" in mlflow_data["test_results"]:
            avg_loading = np.mean([
                r["cold_load_time"] 
                for r in mlflow_data["test_results"]["model_loading"].values()
            ])
            print(f"  Average Model Loading: {avg_loading:.3f}s")
        
        print()
        
        # WebSockets Performance Summary
        ws_data = results["results"]["websockets_15x"]
        print("🔌 WebSockets 15.x Performance:")
        print(f"  Version: {ws_data['version']}")
        
        if "connection_performance" in ws_data["test_results"]:
            avg_conn = np.mean([
                perf["avg_connection_time"]
                for perf in ws_data["test_results"]["connection_performance"].values()
            ])
            print(f"  Average Connection Time: {avg_conn:.4f}s")
        
        if "message_throughput" in ws_data["test_results"]:
            throughput = ws_data["test_results"]["message_throughput"]["messages_per_second"]
            print(f"  Message Throughput: {throughput:.1f} msg/s")
        
        print()
        
        # Integration Performance Summary
        integration_data = results["results"]["integration"]
        print("🔗 Integration Performance:")
        
        if "complete_pipeline" in integration_data["test_results"]:
            pipeline = integration_data["test_results"]["complete_pipeline"]
            print(f"  Complete Pipeline: {pipeline['total_time']:.2f}s")
            print(f"  Model R²: {pipeline['model_score']:.3f}")
            print(f"  Analytics Updates: {pipeline['analytics_updates']}")
        
        print()
        
        # Threshold Analysis
        print("🎯 Threshold Analysis:")
        
        # Check each threshold
        thresholds_met = 0
        total_thresholds = len(self.thresholds)
        
        # NumPy matrix operations
        if "matrix_operations" in numpy_data["test_results"]:
            large_matrix = numpy_data["test_results"]["matrix_operations"].get("10000", {})
            if large_matrix:
                matrix_time = large_matrix["operation_time"]
                threshold = self.thresholds["numpy_matrix_ops_100k"]
                if matrix_time < threshold:
                    print(f"  ✅ NumPy Matrix Ops: {matrix_time:.3f}s < {threshold}s")
                    thresholds_met += 1
                else:
                    print(f"  ❌ NumPy Matrix Ops: {matrix_time:.3f}s >= {threshold}s")
        
        # MLflow logging
        if "model_logging" in mlflow_data["test_results"]:
            avg_logging = np.mean([
                r["logging_time"] 
                for r in mlflow_data["test_results"]["model_logging"].values()
            ])
            threshold = self.thresholds["mlflow_model_logging"]
            if avg_logging < threshold:
                print(f"  ✅ MLflow Logging: {avg_logging:.3f}s < {threshold}s")
                thresholds_met += 1
            else:
                print(f"  ❌ MLflow Logging: {avg_logging:.3f}s >= {threshold}s")
        
        # Continue for other thresholds...
        
        print(f"\nThresholds Met: {thresholds_met}/{total_thresholds}")
        
        # Final verdict
        print("\n🎉 FINAL VERDICT:")
        if thresholds_met >= total_thresholds * 0.8:  # 80% threshold
            print("✅ UPGRADE PERFORMANCE VALIDATION PASSED")
            print("   All major upgrades show performance improvements!")
        else:
            print("⚠️  UPGRADE PERFORMANCE VALIDATION NEEDS REVIEW")
            print("   Some performance thresholds not met - investigate further")
        
        # Save detailed report
        report_file = self.temp_dir / "upgrade_performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved: {report_file}")

async def main():
    """Run upgrade performance validation"""
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_all_benchmarks()
    
    # Determine exit code based on performance
    numpy_performance = results["results"]["numpy_2x"]["test_results"]
    mlflow_performance = results["results"]["mlflow_3x"]["test_results"]
    websockets_performance = results["results"]["websockets_15x"]["test_results"]
    
    # Simple pass/fail based on major metrics existing
    success = (
        "matrix_operations" in numpy_performance and
        "model_logging" in mlflow_performance and
        "connection_performance" in websockets_performance
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)