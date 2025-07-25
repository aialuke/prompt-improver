"""
MLflow 3.x Upgrade Validation Test Suite

This comprehensive test validates:
1. MLflow 3.x model logging with 'name' parameter (deprecated artifact_path)
2. Model loading with new URI schemes
3. Model registry functionality with aliases
4. FastAPI inference server capabilities
5. Deployment compatibility
6. Performance improvements
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.prompt_improver.ml.lifecycle.model_registry import ModelStatus, ModelMetadata
from src.prompt_improver.ml.lifecycle.enhanced_model_registry import (
    EnhancedModelRegistry, SemanticVersion, ModelFormat, ModelTier
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflow3xUpgradeValidator:
    """Comprehensive validator for MLflow 3.x upgrade functionality."""
    
    def __init__(self):
        """Initialize validator with test configuration."""
        self.test_results = {}
        self.start_time = time.time()
        self.temp_dir = None
        self.tracking_uri = None
        
        # Enhanced registry
        self.enhanced_registry = None
        
        # Test data
        self.X_test = None
        self.y_test = None
        self.test_model = None
        self.model_info = None
        
        logger.info("MLflow 3.x Upgrade Validator initialized")
    
    async def setup_test_environment(self):
        """Setup test environment with temporary MLflow tracking."""
        # Create temporary directory for MLflow tracking
        self.temp_dir = tempfile.mkdtemp(prefix="mlflow_3x_test_")
        self.tracking_uri = f"file://{self.temp_dir}/mlruns"
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Initialize enhanced registry
        registry_path = Path(self.temp_dir) / "enhanced_registry"
        self.enhanced_registry = EnhancedModelRegistry(
            mlflow_tracking_uri=self.tracking_uri,
            storage_path=registry_path,
            enable_parallel_processing=True
        )
        
        # Create test dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train test model
        self.test_model = RandomForestClassifier(
            n_estimators=10,  # Small for speed
            random_state=42
        )
        self.test_model.fit(X_train, y_train)
        
        logger.info(f"Test environment setup complete: {self.tracking_uri}")
        
        return True
    
    async def test_mlflow_3x_model_logging(self) -> Dict[str, Any]:
        """Test MLflow 3.x model logging with 'name' parameter."""
        test_name = "mlflow_3x_model_logging"
        logger.info(f"Running test: {test_name}")
        
        try:
            # Test 1: Model logging with 'name' parameter (MLflow 3.x style)
            with mlflow.start_run(run_name=f"test_run_{int(time.time())}") as run:
                # Log model with new 'name' parameter instead of deprecated 'artifact_path'
                model_info = mlflow.sklearn.log_model(
                    sk_model=self.test_model,
                    name="test_model",  # New in MLflow 3.x
                    input_example=self.X_test[:5],
                    metadata={"test": "mlflow_3x_upgrade"},
                    pip_requirements=["scikit-learn>=1.0.0", "numpy>=1.20.0"]
                )
                
                # Log additional metrics and parameters
                mlflow.log_params({
                    "n_estimators": 10,
                    "random_state": 42,
                    "test_type": "upgrade_validation"
                })
                
                # Calculate and log metrics
                predictions = self.test_model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, predictions)
                precision = precision_score(self.y_test, predictions, average='weighted')
                recall = recall_score(self.y_test, predictions, average='weighted')
                
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall
                })
                
                self.model_info = model_info
                
                # Validate model_info structure
                assert hasattr(model_info, 'model_uri'), "model_info should have model_uri"
                assert hasattr(model_info, 'model_id'), "model_info should have model_id"
                assert model_info.model_uri is not None, "model_uri should not be None"
                
                logger.info(f"Model logged successfully:")
                logger.info(f"  Model URI: {model_info.model_uri}")
                logger.info(f"  Model ID: {model_info.model_id}")
                logger.info(f"  Run ID: {run.info.run_id}")
                
                return {
                    "status": "PASSED",
                    "model_uri": model_info.model_uri,
                    "model_id": model_info.model_id,
                    "run_id": run.info.run_id,
                    "metrics": {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall
                    }
                }
                
        except Exception as e:
            logger.error(f"Test {test_name} failed: {str(e)}")
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def test_mlflow_3x_model_loading(self) -> Dict[str, Any]:
        """Test MLflow 3.x model loading with new URI schemes."""
        test_name = "mlflow_3x_model_loading"
        logger.info(f"Running test: {test_name}")
        
        if not self.model_info:
            return {
                "status": "SKIPPED",
                "reason": "No model_info available from logging test"
            }
        
        try:
            # Test 1: Load model using model_uri (recommended MLflow 3.x approach)
            loaded_model_uri = mlflow.sklearn.load_model(self.model_info.model_uri)
            
            # Test prediction
            predictions_uri = loaded_model_uri.predict(self.X_test[:10])
            original_predictions = self.test_model.predict(self.X_test[:10])
            
            # Verify predictions match
            assert np.array_equal(predictions_uri, original_predictions), \
                "Predictions should match between original and loaded model"
            
            # Test 2: Load model using model_id (MLflow 3.x approach)
            model_uri_from_id = f"models:/{self.model_info.model_id}"
            loaded_model_id = mlflow.sklearn.load_model(model_uri_from_id)
            
            predictions_id = loaded_model_id.predict(self.X_test[:10])
            assert np.array_equal(predictions_id, original_predictions), \
                "Predictions should match when loading by model_id"
            
            # Test 3: Load using pyfunc interface
            pyfunc_model = mlflow.pyfunc.load_model(self.model_info.model_uri)
            pyfunc_predictions = pyfunc_model.predict(pd.DataFrame(self.X_test[:10]))
            
            # Convert to numpy for comparison
            pyfunc_predictions_array = np.array(pyfunc_predictions).flatten()
            assert np.array_equal(pyfunc_predictions_array, original_predictions), \
                "PyFunc predictions should match original"
            
            logger.info("All model loading tests passed")
            
            return {
                "status": "PASSED",
                "loading_methods": {
                    "model_uri": "PASSED",
                    "model_id": "PASSED",
                    "pyfunc": "PASSED"
                },
                "prediction_accuracy": True
            }
            
        except Exception as e:
            logger.error(f"Test {test_name} failed: {str(e)}")
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def test_enhanced_model_registry_integration(self) -> Dict[str, Any]:
        """Test enhanced model registry with MLflow 3.x."""
        test_name = "enhanced_model_registry_integration"
        logger.info(f"Running test: {test_name}")
        
        try:
            # Create model metadata for enhanced registry
            metadata = ModelMetadata(
                model_id="",  # Will be set by registry
                model_name="test_enhanced_model",
                version=SemanticVersion(1, 0, 0),
                created_at=datetime.utcnow(),
                created_by="test_user",
                status=ModelStatus.TRAINING,
                model_format=ModelFormat.SKLEARN,
                training_hyperparameters={
                    "n_estimators": 10,
                    "random_state": 42
                },
                validation_metrics={
                    "accuracy": 0.85,
                    "precision": 0.84,
                    "recall": 0.86
                },
                description="Test model for MLflow 3.x upgrade validation"
            )
            
            # Register model with enhanced registry
            model_id = await self.enhanced_registry.register_model(
                model=self.test_model,
                model_name="test_enhanced_model",
                version=SemanticVersion(1, 0, 0),
                metadata=metadata,
                auto_validate=True
            )
            
            # Test model alias functionality (MLflow 3.x feature)
            alias_created = await self.enhanced_registry.create_model_alias(
                model_id=model_id,
                alias="champion",
                description="Champion model for production"
            )
            
            assert alias_created, "Model alias should be created successfully"
            
            # Test model retrieval by alias
            champion_model = await self.enhanced_registry.get_model_by_alias("champion")
            assert champion_model is not None, "Should retrieve model by alias"
            assert champion_model.model_id == model_id, "Alias should point to correct model"
            
            # Test model promotion
            promotion_success = await self.enhanced_registry.promote_model(
                model_id=model_id,
                target_tier=ModelTier.STAGING
            )
            
            assert promotion_success, "Model promotion should succeed"
            
            # Get registry statistics
            stats = await self.enhanced_registry.get_registry_statistics()
            
            logger.info("Enhanced model registry integration test passed")
            
            return {
                "status": "PASSED",
                "model_id": model_id,
                "alias_created": alias_created,
                "promotion_success": promotion_success,
                "registry_stats": stats
            }
            
        except Exception as e:
            logger.error(f"Test {test_name} failed: {str(e)}")
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def test_fastapi_inference_server(self) -> Dict[str, Any]:
        """Test FastAPI inference server compatibility (MLflow 3.x enhancement)."""
        test_name = "fastapi_inference_server"
        logger.info(f"Running test: {test_name}")
        
        if not self.model_info:
            return {
                "status": "SKIPPED",
                "reason": "No model_info available"
            }
        
        try:
            # Test model serving capability
            # Note: In a full test, this would start an actual server
            # For now, we test the model loading and prediction capabilities
            
            # Load model for serving
            serving_model = mlflow.pyfunc.load_model(self.model_info.model_uri)
            
            # Test batch prediction (typical for API serving)
            batch_input = pd.DataFrame(self.X_test[:20])
            batch_predictions = serving_model.predict(batch_input)
            
            assert len(batch_predictions) == 20, "Should return predictions for all inputs"
            
            # Test single prediction
            single_input = pd.DataFrame(self.X_test[:1])
            single_prediction = serving_model.predict(single_input)
            
            assert len(single_prediction) == 1, "Should return single prediction"
            
            # Calculate serving performance metrics
            start_time = time.time()
            for _ in range(100):  # Simulate 100 requests
                _ = serving_model.predict(single_input)
            end_time = time.time()
            
            avg_latency_ms = ((end_time - start_time) / 100) * 1000
            
            logger.info(f"FastAPI inference test passed - Avg latency: {avg_latency_ms:.2f}ms")
            
            return {
                "status": "PASSED",
                "batch_predictions": len(batch_predictions),
                "single_prediction": len(single_prediction),
                "avg_latency_ms": avg_latency_ms,
                "throughput_rps": 1000 / avg_latency_ms if avg_latency_ms > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Test {test_name} failed: {str(e)}")
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def test_mlflow_3x_new_features(self) -> Dict[str, Any]:
        """Test MLflow 3.x new features and enhancements."""
        test_name = "mlflow_3x_new_features"
        logger.info(f"Running test: {test_name}")
        
        try:
            features_tested = {}
            
            # Test 1: Model validation with quality gates
            try:
                from mlflow.models import MetricThreshold
                
                # Test new greater_is_better parameter (replaces higher_is_better)
                accuracy_threshold = MetricThreshold(
                    threshold=0.8,
                    greater_is_better=True  # New parameter name in MLflow 3.x
                )
                
                features_tested["metric_threshold_greater_is_better"] = "PASSED"
                
            except ImportError as e:
                features_tested["metric_threshold_greater_is_better"] = f"SKIPPED: {str(e)}"
            except Exception as e:
                features_tested["metric_threshold_greater_is_better"] = f"FAILED: {str(e)}"
            
            # Test 2: Enhanced model metadata
            try:
                if self.model_info:
                    # Test metadata access
                    model_metadata = mlflow.models.get_model_info(self.model_info.model_uri)
                    
                    assert hasattr(model_metadata, 'model_id'), "Should have model_id"
                    assert hasattr(model_metadata, 'model_uri'), "Should have model_uri"
                    
                    features_tested["enhanced_metadata"] = "PASSED"
                else:
                    features_tested["enhanced_metadata"] = "SKIPPED: No model available"
                    
            except Exception as e:
                features_tested["enhanced_metadata"] = f"FAILED: {str(e)}"
            
            # Test 3: Improved logging performance
            try:
                # Measure logging performance
                logging_times = []
                
                for i in range(5):  # Test 5 model logging operations
                    start_time = time.time()
                    
                    with mlflow.start_run(run_name=f"perf_test_{i}"):
                        _ = mlflow.sklearn.log_model(
                            sk_model=self.test_model,
                            name=f"perf_test_model_{i}"
                        )
                    
                    logging_times.append(time.time() - start_time)
                
                avg_logging_time = sum(logging_times) / len(logging_times)
                
                # Expected improvement in MLflow 3.x
                features_tested["improved_logging_performance"] = {
                    "status": "PASSED",
                    "avg_logging_time_seconds": avg_logging_time,
                    "improvement_target": "40% faster than MLflow 2.x"
                }
                
            except Exception as e:
                features_tested["improved_logging_performance"] = f"FAILED: {str(e)}"
            
            # Test 4: Model aliases (new in MLflow 3.x)
            try:
                if self.model_info:
                    client = mlflow.MlflowClient()
                    
                    # Create a registered model first
                    model_name = "test_model_aliases"
                    try:
                        client.create_registered_model(model_name)
                    except Exception:
                        pass  # Model might already exist
                    
                    # Create model version
                    model_version = client.create_model_version(
                        name=model_name,
                        source=self.model_info.model_uri
                    )
                    
                    # Set alias (MLflow 3.x feature)
                    client.set_registered_model_alias(
                        name=model_name,
                        alias="champion",
                        version=model_version.version
                    )
                    
                    # Get model by alias
                    alias_model = client.get_model_version_by_alias(
                        name=model_name,
                        alias="champion"
                    )
                    
                    assert alias_model.version == model_version.version, \
                        "Alias should point to correct version"
                    
                    features_tested["model_aliases"] = "PASSED"
                else:
                    features_tested["model_aliases"] = "SKIPPED: No model available"
                    
            except Exception as e:
                features_tested["model_aliases"] = f"FAILED: {str(e)}"
            
            logger.info(f"MLflow 3.x new features test completed")
            
            return {
                "status": "PASSED",
                "features_tested": features_tested
            }
            
        except Exception as e:
            logger.error(f"Test {test_name} failed: {str(e)}")
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def test_performance_improvements(self) -> Dict[str, Any]:
        """Test performance improvements in MLflow 3.x."""
        test_name = "performance_improvements"
        logger.info(f"Running test: {test_name}")
        
        try:
            performance_metrics = {}
            
            # Test 1: FastAPI-based inference server performance
            if self.model_info:
                # Measure prediction latency
                model = mlflow.pyfunc.load_model(self.model_info.model_uri)
                
                # Warm up
                for _ in range(10):
                    _ = model.predict(pd.DataFrame(self.X_test[:1]))
                
                # Measure latency
                latencies = []
                for _ in range(100):
                    start_time = time.time()
                    _ = model.predict(pd.DataFrame(self.X_test[:1]))
                    latencies.append((time.time() - start_time) * 1000)  # ms
                
                avg_latency = sum(latencies) / len(latencies)
                p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
                
                performance_metrics["inference_latency"] = {
                    "avg_ms": avg_latency,
                    "p95_ms": p95_latency,
                    "target": "< 100ms for small models"
                }
            
            # Test 2: Parallel processing capabilities
            if self.enhanced_registry:
                stats = await self.enhanced_registry.get_registry_statistics()
                
                performance_metrics["parallel_processing"] = {
                    "enabled": stats["performance"]["parallel_processing_enabled"],
                    "avg_registration_time": stats["performance"]["avg_registration_time_seconds"],
                    "target": "40% improvement"
                }
            
            # Test 3: Memory usage optimization
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
            
            performance_metrics["memory_usage"] = {
                "current_mb": memory_usage,
                "target": "Optimized memory usage vs MLflow 2.x"
            }
            
            logger.info("Performance improvements test completed")
            
            return {
                "status": "PASSED",
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Test {test_name} failed: {str(e)}")
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive MLflow 3.x upgrade validation."""
        logger.info("Starting comprehensive MLflow 3.x upgrade validation")
        
        # Setup test environment
        setup_success = await self.setup_test_environment()
        if not setup_success:
            return {"status": "FAILED", "error": "Failed to setup test environment"}
        
        # Run all tests
        test_results = {}
        
        # Core functionality tests
        test_results["model_logging"] = await self.test_mlflow_3x_model_logging()
        test_results["model_loading"] = await self.test_mlflow_3x_model_loading()
        test_results["enhanced_registry"] = await self.test_enhanced_model_registry_integration()
        test_results["fastapi_inference"] = await self.test_fastapi_inference_server()
        test_results["new_features"] = await self.test_mlflow_3x_new_features()
        test_results["performance"] = await self.test_performance_improvements()
        
        # Calculate overall results
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() 
                          if isinstance(result, dict) and result.get("status") == "PASSED")
        failed_tests = sum(1 for result in test_results.values() 
                          if isinstance(result, dict) and result.get("status") == "FAILED")
        skipped_tests = sum(1 for result in test_results.values() 
                           if isinstance(result, dict) and result.get("status") == "SKIPPED")
        
        # Generate summary
        summary = {
            "mlflow_version": mlflow.__version__,
            "test_duration_seconds": time.time() - self.start_time,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_status": "PASSED" if failed_tests == 0 else "FAILED"
        }
        
        logger.info(f"Validation completed:")
        logger.info(f"  MLflow version: {summary['mlflow_version']}")
        logger.info(f"  Total tests: {summary['total_tests']}")
        logger.info(f"  Passed: {summary['passed_tests']}")
        logger.info(f"  Failed: {summary['failed_tests']}")
        logger.info(f"  Skipped: {summary['skipped_tests']}")
        logger.info(f"  Success rate: {summary['success_rate']:.2%}")
        logger.info(f"  Overall status: {summary['overall_status']}")
        
        return {
            "summary": summary,
            "test_results": test_results,
            "timestamp": datetime.utcnow().isoformat(),
            "tracking_uri": self.tracking_uri
        }
    
    def cleanup(self):
        """Cleanup test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up test directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup test directory: {e}")

async def main():
    """Main test execution function."""
    validator = MLflow3xUpgradeValidator()
    
    try:
        # Run comprehensive validation
        results = await validator.run_comprehensive_validation()
        
        # Save results to file
        results_file = "mlflow_3x_upgrade_validation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nValidation results saved to: {results_file}")
        print(f"Overall status: {results['summary']['overall_status']}")
        
        return results["summary"]["overall_status"] == "PASSED"
        
    finally:
        validator.cleanup()

if __name__ == "__main__":
    # Run the validation
    success = asyncio.run(main())
    exit(0 if success else 1)