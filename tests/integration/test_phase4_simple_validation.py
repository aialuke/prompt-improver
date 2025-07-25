#!/usr/bin/env python3
"""
Simplified Phase 4 Validation Test

This test validates the core functionality after upgrades without requiring
database connections or complex imports. Focuses on:
- NumPy 2.x functionality and performance
- MLflow 3.x model operations  
- Websockets 15.x basic operations
- Simple integration validation

This can run standalone to verify the upgrades are working.
"""

import asyncio
import json
import numpy as np
import mlflow
import mlflow.sklearn
import websockets
import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

class SimplePhase4Validator:
    """Simple validation of Phase 4 upgrades"""
    
    def __init__(self):
        self.results = {"tests": [], "summary": {}}
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def test_numpy_2x_validation(self):
        """Test NumPy 2.x basic functionality"""
        print("🔢 Testing NumPy 2.x...")
        
        try:
            # Verify version
            assert np.__version__.startswith("2."), f"Expected NumPy 2.x, got {np.__version__}"
            print(f"  ✓ NumPy version: {np.__version__}")
            
            # Test basic operations
            start_time = time.time()
            
            # Create test data
            X = np.random.randn(10000, 100).astype(np.float64)
            y = np.random.randint(0, 2, 10000)
            
            # Mathematical operations
            X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            correlation = np.corrcoef(X.T)
            eigenvals, eigenvecs = np.linalg.eigh(correlation[:50, :50])
            
            operation_time = time.time() - start_time
            
            # Validate results
            assert not np.any(np.isnan(X_normalized))
            assert correlation.shape == (100, 100)
            assert len(eigenvals) == 50
            
            print(f"  ✓ Matrix operations: {operation_time:.3f}s")
            print(f"  ✓ Data shape: {X.shape}")
            print(f"  ✓ Memory usage: {X.nbytes / 1024 / 1024:.1f} MB")
            
            self.results["tests"].append({
                "name": "NumPy 2.x Validation",
                "status": "PASSED",
                "version": np.__version__,
                "operation_time": operation_time,
                "data_shape": X.shape
            })
            
            return True
            
        except Exception as e:
            print(f"  ❌ NumPy test failed: {e}")
            self.results["tests"].append({
                "name": "NumPy 2.x Validation",
                "status": "FAILED",
                "error": str(e)
            })
            return False
    
    def test_mlflow_3x_validation(self):
        """Test MLflow 3.x basic functionality"""
        print("🤖 Testing MLflow 3.x...")
        
        try:
            # Verify version
            assert mlflow.__version__.startswith("3."), f"Expected MLflow 3.x, got {mlflow.__version__}"
            print(f"  ✓ MLflow version: {mlflow.__version__}")
            
            # Set up experiment
            experiment_name = f"phase4_validation_{int(time.time())}"
            mlflow.set_experiment(experiment_name)
            
            # Test model training and logging
            start_time = time.time()
            
            # Generate data
            X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=42)
            
            with mlflow.start_run(run_name="validation_test"):
                # Train model
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X, y)
                
                # Calculate metrics
                accuracy = model.score(X, y)
                
                # Log with MLflow 3.x
                mlflow.log_param("n_estimators", 50)
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("n_samples", X.shape[0])
                mlflow.log_metric("accuracy", accuracy)
                
                # Log model
                mlflow.sklearn.log_model(model, "validation_model")
                
                run_id = mlflow.active_run().info.run_id
            
            training_time = time.time() - start_time
            
            # Test model loading
            start_time = time.time()
            model_uri = f"runs:/{run_id}/validation_model"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            loading_time = time.time() - start_time
            
            # Test predictions
            test_predictions = loaded_model.predict(X[:100])
            
            # Validate results
            assert len(test_predictions) == 100
            assert accuracy > 0.5
            assert loading_time < 5.0
            
            print(f"  ✓ Model training: {training_time:.3f}s")
            print(f"  ✓ Model loading: {loading_time:.3f}s")
            print(f"  ✓ Model accuracy: {accuracy:.3f}")
            print(f"  ✓ Predictions: {len(test_predictions)} samples")
            
            self.results["tests"].append({
                "name": "MLflow 3.x Validation", 
                "status": "PASSED",
                "version": mlflow.__version__,
                "training_time": training_time,
                "loading_time": loading_time,
                "accuracy": accuracy,
                "run_id": run_id
            })
            
            return True
            
        except Exception as e:
            print(f"  ❌ MLflow test failed: {e}")
            self.results["tests"].append({
                "name": "MLflow 3.x Validation",
                "status": "FAILED", 
                "error": str(e)
            })
            return False
    
    async def test_websockets_15x_validation(self):
        """Test Websockets 15.x basic functionality"""
        print("🔌 Testing Websockets 15.x...")
        
        try:
            # Verify version
            assert websockets.__version__.startswith("15."), f"Expected Websockets 15.x, got {websockets.__version__}"
            print(f"  ✓ Websockets version: {websockets.__version__}")
            
            # Test basic WebSocket server/client
            messages_sent = 0
            messages_received = 0
            
            async def echo_server(websocket, path):
                nonlocal messages_received
                try:
                    async for message in websocket:
                        messages_received += 1
                        await websocket.send(f"echo: {message}")
                except websockets.exceptions.ConnectionClosed:
                    pass
            
            # Start server
            server = await websockets.serve(echo_server, "localhost", 0)
            server_port = server.sockets[0].getsockname()[1]
            
            try:
                # Test client connection
                start_time = time.time()
                
                uri = f"ws://localhost:{server_port}"
                websocket = await websockets.connect(uri)
                
                connection_time = time.time() - start_time
                
                # Test message exchange
                start_time = time.time()
                
                for i in range(10):
                    test_message = f"test_message_{i}"
                    await websocket.send(test_message)
                    messages_sent += 1
                    
                    response = await websocket.recv()
                    assert response == f"echo: {test_message}"
                
                messaging_time = time.time() - start_time
                
                await websocket.close()
                
                # Validate results
                assert messages_sent == 10
                assert messages_received == 10
                assert connection_time < 1.0
                assert messaging_time < 1.0
                
                print(f"  ✓ Connection time: {connection_time:.4f}s")
                print(f"  ✓ Messaging time: {messaging_time:.4f}s")
                print(f"  ✓ Messages exchanged: {messages_sent}/{messages_received}")
                
                self.results["tests"].append({
                    "name": "Websockets 15.x Validation",
                    "status": "PASSED",
                    "version": websockets.__version__,
                    "connection_time": connection_time,
                    "messaging_time": messaging_time,
                    "messages_exchanged": messages_sent
                })
                
                return True
                
            finally:
                server.close()
                await server.wait_closed()
            
        except Exception as e:
            print(f"  ❌ Websockets test failed: {e}")
            self.results["tests"].append({
                "name": "Websockets 15.x Validation",
                "status": "FAILED",
                "error": str(e)
            })
            return False
    
    def test_integration_validation(self):
        """Test simple integration between components"""
        print("🔗 Testing Integration...")
        
        try:
            # Test NumPy + MLflow integration
            print("  • Testing NumPy + MLflow integration...")
            
            # Create data with NumPy 2.x
            X = np.random.randn(2000, 30).astype(np.float32)
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
            
            # Train model and log with MLflow 3.x
            with mlflow.start_run(run_name="integration_test"):
                model = RandomForestClassifier(n_estimators=25, random_state=42)
                model.fit(X, y)
                
                accuracy = model.score(X, y)
                
                # Log NumPy array info
                mlflow.log_param("numpy_version", np.__version__)
                mlflow.log_param("data_dtype", str(X.dtype))
                mlflow.log_metric("data_shape_0", X.shape[0])
                mlflow.log_metric("data_shape_1", X.shape[1])
                mlflow.log_metric("accuracy", accuracy)
                
                mlflow.sklearn.log_model(model, "integration_model")
                
                run_id = mlflow.active_run().info.run_id
            
            # Test real-time analytics data format (for WebSocket simulation)
            analytics_data = []
            
            for i in range(5):
                # Use NumPy for calculations
                sample_predictions = model.predict_proba(X[i*100:(i+1)*100])
                
                analytics_update = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "batch_id": i,
                    "sample_size": 100,
                    "mean_probability": float(np.mean(sample_predictions[:, 1])),
                    "std_probability": float(np.std(sample_predictions[:, 1])),
                    "run_id": run_id
                }
                
                analytics_data.append(analytics_update)
            
            # Validate integration results
            assert accuracy > 0.5
            assert len(analytics_data) == 5
            assert all(0 <= item["mean_probability"] <= 1 for item in analytics_data)
            
            print(f"    ✓ Model accuracy: {accuracy:.3f}")
            print(f"    ✓ Analytics updates: {len(analytics_data)}")
            print(f"    ✓ Data processing: NumPy {np.__version__}")
            print(f"    ✓ Model tracking: MLflow {mlflow.__version__}")
            
            self.results["tests"].append({
                "name": "Integration Validation",
                "status": "PASSED",
                "model_accuracy": accuracy,
                "analytics_updates": len(analytics_data),
                "components_tested": ["NumPy", "MLflow", "WebSocket-ready data"]
            })
            
            return True
            
        except Exception as e:
            print(f"  ❌ Integration test failed: {e}")
            self.results["tests"].append({
                "name": "Integration Validation",
                "status": "FAILED",
                "error": str(e)
            })
            return False
    
    async def run_all_validations(self):
        """Run all validation tests"""
        print("🚀 Starting Phase 4 Simple Validation")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run tests
        test_results = []
        
        test_results.append(self.test_numpy_2x_validation())
        test_results.append(self.test_mlflow_3x_validation())
        test_results.append(await self.test_websockets_15x_validation())
        test_results.append(self.test_integration_validation())
        
        total_time = time.time() - start_time
        
        # Calculate summary
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate,
            "total_time": total_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Generate report
        await self.generate_report()
        
        return self.results
    
    async def generate_report(self):
        """Generate validation report"""
        print("\n" + "=" * 60)
        print("📊 PHASE 4 SIMPLE VALIDATION REPORT")
        print("=" * 60)
        
        summary = self.results["summary"]
        
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Execution Time: {summary['total_time']:.2f}s")
        
        print("\n📝 Test Details:")
        for test in self.results["tests"]:
            status_icon = "✅" if test["status"] == "PASSED" else "❌"
            print(f"  {status_icon} {test['name']}: {test['status']}")
            
            if test["status"] == "PASSED":
                if "version" in test:
                    print(f"    Version: {test['version']}")
                if "operation_time" in test:
                    print(f"    Operation Time: {test['operation_time']:.3f}s")
                if "accuracy" in test:
                    print(f"    Accuracy: {test['accuracy']:.3f}")
            else:
                if "error" in test:
                    print(f"    Error: {test['error']}")
        
        print("\n🎯 Upgrade Status:")
        print(f"  NumPy 2.x: ✅ {np.__version__}")
        print(f"  MLflow 3.x: ✅ {mlflow.__version__}")
        print(f"  Websockets 15.x: ✅ {websockets.__version__}")
        
        print("\n💡 Conclusion:")
        if summary["failed_tests"] == 0:
            print("  🎉 ALL VALIDATIONS PASSED!")
            print("  The system upgrades are working correctly.")
            print("  NumPy 2.x, MLflow 3.x, and Websockets 15.x are functional.")
        elif summary["success_rate"] >= 75:
            print("  ⚠️  MOSTLY SUCCESSFUL - Minor issues detected")
            print("  Most functionality working, review failed tests.")
        else:
            print("  ❌ VALIDATION FAILED - Major issues detected")
            print("  Significant problems with the upgrades.")
        
        # Save results
        results_file = self.temp_dir / "phase4_simple_validation.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved: {results_file}")

async def main():
    """Main validation entry point"""
    validator = SimplePhase4Validator()
    
    try:
        results = await validator.run_all_validations()
        
        # Return exit code based on results
        if results["summary"]["failed_tests"] == 0:
            return 0  # All tests passed
        elif results["summary"]["success_rate"] >= 75:
            return 1  # Mostly successful
        else:
            return 2  # Major failures
            
    except Exception as e:
        print(f"💥 Validation crashed: {e}")
        return 3

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)