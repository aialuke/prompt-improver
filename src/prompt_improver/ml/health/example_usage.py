"""
ML Health Monitoring Example Usage - 2025 Best Practices

This example demonstrates how to integrate ML health monitoring
into your ML inference pipeline.
"""

import asyncio
import logging
import time
from typing import Any, Dict

# Example sklearn model for demonstration
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from .integration_manager import get_ml_health_integration_manager

logger = logging.getLogger(__name__)


async def example_ml_pipeline_with_health_monitoring():
    """Example ML pipeline with comprehensive health monitoring"""
    
    # Initialize health monitoring
    health_manager = await get_ml_health_integration_manager()
    
    # Create example model
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Register model for health monitoring
    model_id = "example_classifier_v1"
    await health_manager.register_model_for_monitoring(
        model_id=model_id,
        model=model,
        model_type="RandomForestClassifier",
        version="1.0.0",
        metadata={"features": 20, "classes": 2}
    )
    
    print(f"✅ Registered model {model_id} for health monitoring")
    
    # Simulate inference requests
    print("\n🔄 Simulating inference requests...")
    for i in range(50):
        # Generate sample data
        sample_features = X[i:i+1]
        
        # Start inference tracking
        request_id = f"req_{i:03d}"
        await health_manager.track_inference(
            model_id=model_id,
            request_id=request_id,
            features=sample_features[0].tolist()
        )
        
        # Simulate model inference
        start_time = time.time()
        try:
            # Actual prediction
            prediction = model.predict(sample_features)[0]
            prediction_proba = model.predict_proba(sample_features)[0]
            confidence = max(prediction_proba)
            
            # Simulate some processing time
            await asyncio.sleep(0.001 + (i % 10) * 0.001)  # Variable latency
            
            # Complete tracking (successful)
            latency_ms = (time.time() - start_time) * 1000
            await health_manager.complete_inference_tracking(
                model_id=model_id,
                request_id=request_id,
                success=True,
                latency_ms=latency_ms
            )
            
            # Also track prediction for drift detection
            await health_manager.track_inference(
                model_id=model_id,
                request_id=request_id + "_drift",
                prediction=float(prediction),
                confidence=confidence,
                features=sample_features[0].tolist(),
                start_performance_tracking=False  # Already tracked above
            )
            
        except Exception as e:
            # Complete tracking (failed)
            await health_manager.complete_inference_tracking(
                model_id=model_id,
                request_id=request_id,
                success=False,
                error_type=type(e).__name__
            )
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/50 requests")
    
    # Get health summary
    print("\n📊 Getting model health summary...")
    health_summary = await health_manager.get_model_health_summary(model_id)
    
    if health_summary:
        print(f"\n🏥 Health Summary for {model_id}:")
        print(f"   Overall Healthy: {health_summary['overall_healthy']}")
        
        # Health component
        health_comp = health_summary['components'].get('health', {})
        if health_comp:
            print(f"   Success Rate: {health_comp['success_rate']:.1%}")
            print(f"   Total Predictions: {health_comp['total_predictions']}")
            print(f"   Memory Usage: {health_comp['memory_mb']:.1f} MB")
            print(f"   P95 Latency: {health_comp.get('latency_p95', 0):.1f} ms")
        
        # Performance component
        perf_comp = health_summary['components'].get('performance', {})
        if perf_comp:
            latency_summary = perf_comp.get('latency_summary', {})
            print(f"   Avg Latency: {latency_summary.get('avg_ms', 0):.1f} ms")
            print(f"   Max Latency: {latency_summary.get('max_ms', 0):.1f} ms")
        
        # Drift component
        drift_comp = health_summary['components'].get('drift', {})
        if drift_comp:
            print(f"   Drift Risk: {drift_comp['risk_level']}")
            print(f"   Drift Detected: {drift_comp['drift_detected']}")
    
    # Get system-wide health dashboard
    print("\n🌐 Getting system health dashboard...")
    dashboard = await health_manager.get_system_health_dashboard()
    
    if 'error' not in dashboard:
        print(f"\n🎯 System Health Dashboard:")
        
        system_health = dashboard.get('system_health', {})
        print(f"   System Healthy: {system_health.get('healthy', 'Unknown')}")
        print(f"   Health Score: {system_health.get('health_score', 0):.2f}")
        
        # Model summaries
        model_summaries = dashboard.get('model_summaries', [])
        print(f"   Models Monitored: {len(model_summaries)}")
        
        # Alerts
        alerts = dashboard.get('alerts', [])
        if alerts:
            print(f"   🚨 Active Alerts: {len(alerts)}")
            for alert in alerts:
                print(f"      - {alert['level'].upper()}: {alert['message']}")
        
        # Recommendations
        recommendations = dashboard.get('recommendations', [])
        if recommendations:
            print(f"   💡 Recommendations:")
            for rec in recommendations:
                print(f"      - {rec}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up model registration...")
    await health_manager.unregister_model(model_id)
    print(f"   Unregistered model {model_id}")
    
    print("\n✅ Example completed successfully!")


async def example_health_check_api():
    """Example of how health checks would be used in API endpoints"""
    
    print("\n🔍 Example: Checking ML health for API endpoint...")
    
    try:
        # This simulates what happens in the health check API
        from ..health.ml_health_monitor import get_ml_health_monitor
        from ..health.resource_monitor import get_resource_monitor
        
        # Get ML system health
        ml_monitor = await get_ml_health_monitor()
        system_health = await ml_monitor.get_system_health()
        
        print(f"   ML System Health: {system_health.get('healthy', 'Unknown')}")
        print(f"   Total Models: {system_health.get('models', {}).get('total_loaded', 0)}")
        print(f"   Total Memory: {system_health.get('models', {}).get('total_memory_mb', 0):.1f} MB")
        
        # Get resource utilization
        resource_monitor = await get_resource_monitor()
        resources = await resource_monitor.get_current_resources()
        
        print(f"   CPU Usage: {resources.cpu_percent:.1f}%")
        print(f"   Memory Usage: {resources.memory_percent:.1f}%")
        print(f"   GPU Available: {resources.gpu_count > 0}")
        
        if resources.gpu_count > 0:
            print(f"   GPU Utilization: {resources.gpu_utilization_percent:.1f}%")
            print(f"   GPU Memory: {resources.gpu_used_memory_gb:.1f}/{resources.gpu_total_memory_gb:.1f} GB")
        
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("🚀 ML Health Monitoring Example")
    print("=" * 50)
    
    # Run the example
    asyncio.run(example_ml_pipeline_with_health_monitoring())
    
    # Run health check example
    asyncio.run(example_health_check_api())
    
    print("\n" + "=" * 50)
    print("✨ All examples completed!")