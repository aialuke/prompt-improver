#!/usr/bin/env python3
"""
Debug script for ML components validation issues.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_improver.ml.learning.algorithms.analysis_orchestrator import AnalysisOrchestrator
from prompt_improver.ml.learning.algorithms.failure_classifier import FailureClassifier


class MinimalFailureConfig:
    """Minimal configuration for ML components validation."""

    def __init__(self):
        # Basic configuration
        self.failure_threshold = 0.3
        self.confidence_threshold = 0.7
        self.max_failures_to_analyze = 100

        # Feature flags
        self.enable_prometheus_monitoring = False  # Disabled for OTel migration
        self.enable_robustness_validation = False  # Simplified for validation
        self.enable_anomaly_detection = True
        self.enable_clustering = False  # Simplified for validation

        # Alerting configuration
        self.alert_cooldown_seconds = 300
        self.high_failure_rate_threshold = 0.8
        self.anomaly_threshold = 0.9

        # Analysis configuration
        self.min_samples_for_analysis = 5
        self.max_analysis_time_seconds = 30

        # Pattern detection configuration (required by PatternDetector)
        self.min_pattern_size = 2
        self.max_pattern_depth = 3
        self.max_patterns = 20  # Maximum number of patterns to return
        self.max_root_causes = 10  # Maximum number of root causes to return
        self.similarity_threshold = 0.8
        self.context_window_size = 5
        self.outlier_threshold = 2.0  # Z-score threshold for outlier detection

        # Clustering configuration (required by AnalysisOrchestrator)
        self.min_cluster_size = 2
        self.max_clusters = 10
        self.cluster_similarity_threshold = 0.7

        # Robustness validation configuration
        self.noise_levels = [0.1, 0.2]  # Simplified for validation
        self.adversarial_epsilon = 0.1
        self.edge_case_threshold = 0.05
        self.robustness_test_samples = 10  # Number of samples for robustness testing

        # Additional feature flags (required by AnalysisOrchestrator)
        self.ensemble_anomaly_detection = True
        self.adversarial_testing = False  # Disabled for validation (requires ART library)

        # Additional thresholds and limits
        self.max_analysis_iterations = 5
        self.convergence_threshold = 0.01


async def test_analysis_orchestrator():
    """Test AnalysisOrchestrator initialization and basic functionality."""
    print("🧪 Testing AnalysisOrchestrator...")
    
    try:
        config = MinimalFailureConfig()
        print(f"✅ Config created: {config}")
        
        analyzer = AnalysisOrchestrator(config)
        print(f"✅ AnalysisOrchestrator created: {analyzer}")
        
        # Create test data
        test_results = [
            {
                "overallImprovement": 0.2,  # Below failure threshold
                "error": "Test validation error",
                "prompt": "Test prompt",
                "response": "Test response",
                "metadata": {"test": True}
            }
        ]
        
        print(f"📊 Test data: {test_results}")
        
        # Test analyze_failures method
        analysis_result = await analyzer.analyze_failures(test_results)
        print(f"✅ Analysis result: {analysis_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ AnalysisOrchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_failure_classifier():
    """Test FailureClassifier initialization and basic functionality."""
    print("\n🧪 Testing FailureClassifier...")
    
    try:
        config = MinimalFailureConfig()
        print(f"✅ Config created: {config}")
        
        classifier = FailureClassifier(config)
        print(f"✅ FailureClassifier created: {classifier}")
        
        # Test FMEA analysis
        test_failures = [
            {"error": "timeout_error", "type": "timeout", "severity": "high"}
        ]
        test_results = [
            {"overallImprovement": 0.2, "error": "timeout_error", "prompt": "test"}
        ]
        
        print(f"📊 Test failures: {test_failures}")
        print(f"📊 Test results: {test_results}")
        
        fmea_result = await classifier.perform_ml_fmea_analysis(test_failures, test_results)
        print(f"✅ FMEA result: {fmea_result}")
        
        # Test anomaly detection
        anomaly_result = await classifier.perform_ensemble_anomaly_detection(test_failures)
        print(f"✅ Anomaly result: {anomaly_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ FailureClassifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Starting ML Components Debug Test")
    print("=" * 50)
    
    # Test AnalysisOrchestrator
    orchestrator_success = await test_analysis_orchestrator()
    
    # Test FailureClassifier
    classifier_success = await test_failure_classifier()
    
    print("\n" + "=" * 50)
    print("📊 DEBUG TEST RESULTS")
    print("=" * 50)
    print(f"AnalysisOrchestrator: {'✅ PASS' if orchestrator_success else '❌ FAIL'}")
    print(f"FailureClassifier: {'✅ PASS' if classifier_success else '❌ FAIL'}")
    
    overall_success = orchestrator_success and classifier_success
    print(f"\nOverall: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
