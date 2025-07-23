#!/usr/bin/env python3
"""
File Existence Validation Test

Verifies that all 69 components listed in ALL_COMPONENTS.md actually exist as files
in the codebase.
"""

import os
from pathlib import Path

def test_file_existence():
    """Test file existence for all components in ALL_COMPONENTS.md."""
    
    print("📁 Testing File Existence for All 69 Components")
    print("=" * 60)
    
    # All 69 components from ALL_COMPONENTS.md with their file paths
    all_components = {
        # Tier 1: Core ML Pipeline Components (11 components)
        "TrainingDataLoader": "src/prompt_improver/ml/core/training_data_loader.py",
        "MLModelService": "src/prompt_improver/ml/core/ml_integration.py",
        "RuleOptimizer": "src/prompt_improver/ml/optimization/algorithms/rule_optimizer.py",
        "MultiarmedBanditFramework": "src/prompt_improver/ml/optimization/algorithms/multi_armed_bandit.py",
        "AprioriAnalyzer": "src/prompt_improver/ml/learning/patterns/apriori_analyzer.py",
        "BatchProcessor": "src/prompt_improver/ml/optimization/batch/batch_processor.py",
        "ProductionModelRegistry": "src/prompt_improver/ml/models/production_registry.py",
        "ContextLearner": "src/prompt_improver/ml/learning/algorithms/context_learner.py",
        "ClusteringOptimizer": "src/prompt_improver/ml/optimization/algorithms/clustering_optimizer.py",
        "AdvancedDimensionalityReducer": "src/prompt_improver/ml/optimization/algorithms/dimensionality_reducer.py",
        "ProductionSyntheticDataGenerator": "src/prompt_improver/ml/preprocessing/synthetic_data_generator.py",
        
        # Tier 2: Optimization & Learning Components (14 components)
        "InsightGenerationEngine": "src/prompt_improver/ml/learning/algorithms/insight_engine.py",
        "FailureModeAnalyzer": "src/prompt_improver/ml/learning/algorithms/failure_analyzer.py",
        "ContextLearner": "src/prompt_improver/ml/learning/algorithms/context_learner.py",
        "RuleEffectivenessAnalyzer": "src/prompt_improver/ml/learning/algorithms/rule_analyzer.py",
        "ContextAwareFeatureWeighter": "src/prompt_improver/ml/learning/algorithms/context_aware_weighter.py",
        "EnhancedOptimizationValidator": "src/prompt_improver/ml/optimization/validation/optimization_validator.py",
        "AdvancedPatternDiscovery": "src/prompt_improver/ml/learning/patterns/advanced_pattern_discovery.py",
        "LLMTransformerService": "src/prompt_improver/ml/preprocessing/llm_transformer.py",
        "AutoMLOrchestrator": "src/prompt_improver/ml/automl/orchestrator.py",
        "AutoMLCallbacks": "src/prompt_improver/ml/automl/callbacks.py",
        "AdvancedEarlyStoppingFramework": "src/prompt_improver/ml/optimization/algorithms/early_stopping.py",
        "ContextCacheManager": "src/prompt_improver/ml/learning/algorithms/context_cache_manager.py",
        "LinguisticAnalyzer": "src/prompt_improver/ml/analysis/linguistic_analyzer.py",
        "DomainDetector": "src/prompt_improver/ml/analysis/domain_detector.py",
        
        # Tier 3: Evaluation & Analysis Components (14 components)
        "CausalInferenceAnalyzer": "src/prompt_improver/ml/evaluation/causal_inference_analyzer.py",
        "AdvancedStatisticalValidator": "src/prompt_improver/ml/evaluation/advanced_statistical_validator.py",
        "PatternSignificanceAnalyzer": "src/prompt_improver/ml/evaluation/pattern_significance_analyzer.py",
        "StructuralAnalyzer": "src/prompt_improver/ml/evaluation/structural_analyzer.py",
        "ExperimentOrchestrator": "src/prompt_improver/ml/evaluation/experiment_orchestrator.py",
        "StatisticalAnalyzer": "src/prompt_improver/ml/evaluation/statistical_analyzer.py",
        "EnhancedQualityScorer": "src/prompt_improver/ml/learning/quality/enhanced_scorer.py",
        "DomainFeatureExtractor": "src/prompt_improver/ml/analysis/domain_feature_extractor.py",
        "NERExtractor": "src/prompt_improver/ml/analysis/ner_extractor.py",
        "DependencyParser": "src/prompt_improver/ml/analysis/dependency_parser.py",
        "CompositeFeatureExtractor": "src/prompt_improver/ml/learning/features/composite_extractor.py",
        "LinguisticFeatureExtractor": "src/prompt_improver/ml/learning/features/linguistic_extractor.py",

        "ContextFeatureExtractor": "src/prompt_improver/ml/learning/features/context_extractor.py",
        
        # Tier 4: Performance & Infrastructure Components (30 components)
        "RealTimeMonitor": "src/prompt_improver/performance/monitoring/monitoring.py",
        "PerformanceMonitor": "src/prompt_improver/performance/monitoring/performance_monitor.py",
        "HealthMonitor": "src/prompt_improver/performance/monitoring/health_monitor.py",
        "RealTimeAnalyticsService": "src/prompt_improver/performance/analytics/real_time_analytics.py",
        "AnalyticsService": "src/prompt_improver/performance/analytics/analytics.py",
        "MCPPerformanceBenchmark": "src/prompt_improver/performance/monitoring/performance_benchmark.py",
        "PerformanceMetricsWidget": "src/prompt_improver/tui/widgets/performance_metrics.py",
        "ModernABTestingService": "src/prompt_improver/performance/testing/ab_testing_service.py",
        "CanaryTestingService": "src/prompt_improver/performance/testing/canary_testing.py",
        "AsyncBatchProcessor": "src/prompt_improver/performance/optimization/async_optimizer.py",
        "ResponseOptimizer": "src/prompt_improver/performance/optimization/response_optimizer.py",
        "PerformanceOptimizer": "src/prompt_improver/performance/optimization/performance_optimizer.py",
        "PerformanceValidator": "src/prompt_improver/performance/validation/performance_validation.py",
        "MLServiceHealthChecker": "src/prompt_improver/performance/monitoring/health/ml_service_checkers.py",
        "MLOrchestratorHealthChecker": "src/prompt_improver/performance/monitoring/health/ml_orchestration_checkers.py",
        "RedisHealthMonitor": "src/prompt_improver/performance/monitoring/health/redis_monitor.py",
        "AnalyticsServiceHealthChecker": "src/prompt_improver/performance/monitoring/health/analytics_checkers.py",
        "BackgroundTaskManager": "src/prompt_improver/performance/monitoring/health/background_manager.py",
        "HealthService": "src/prompt_improver/performance/monitoring/health/service.py",
        "MLResourceManagerHealthChecker": "src/prompt_improver/performance/monitoring/health/ml_orchestration_checkers.py",
        "ConnectionPoolManager": "src/prompt_improver/performance/optimization/async_optimizer.py",
        "MultiLevelCache": "src/prompt_improver/utils/multi_level_cache.py",
        "ResourceManager": "src/prompt_improver/ml/orchestration/core/resource_manager.py",
        "APESServiceManager": "src/prompt_improver/core/services/manager.py",
        "UnifiedRetryManager": "src/prompt_improver/ml/orchestration/core/unified_retry_manager.py",
        "InputSanitizer": "src/prompt_improver/security/input_sanitization.py",
        "MemoryGuard": "src/prompt_improver/security/memory_guard.py",
        "SecureKeyManager": "src/prompt_improver/security/key_manager.py",
        "FernetKeyManager": "src/prompt_improver/security/key_manager.py",
        "RobustnessEvaluator": "src/prompt_improver/security/adversarial_defense.py",
        "DatabasePerformanceMonitor": "src/prompt_improver/database/performance_monitor.py",
        "DatabaseConnectionOptimizer": "src/prompt_improver/database/query_optimizer.py",
        "PreparedStatementCache": "src/prompt_improver/database/query_optimizer.py",
        "TypeSafePsycopgClient": "src/prompt_improver/database/psycopg_client.py",
        "RetryManager": "src/prompt_improver/database/error_handling.py",
        "ABTestingWidget": "src/prompt_improver/tui/widgets/ab_testing.py",
        "ServiceControlWidget": "src/prompt_improver/tui/widgets/service_control.py",
        "SystemOverviewWidget": "src/prompt_improver/tui/widgets/system_overview.py",
        
        # Tier 6: Advanced Security Components (5 components)
        "AdversarialDefenseSystem": "src/prompt_improver/security/adversarial_defense.py",
        "DifferentialPrivacyService": "src/prompt_improver/security/differential_privacy.py",
        "FederatedLearningService": "src/prompt_improver/security/federated_learning.py",
        "AutoMLStatusWidget": "src/prompt_improver/tui/widgets/automl_status.py",
        "PromptDataProtection": "src/prompt_improver/core/services/security.py",
    }
    
    # Test file existence
    existing_files = []
    missing_files = []
    
    for component_name, file_path in all_components.items():
        if os.path.exists(file_path):
            existing_files.append((component_name, file_path))
        else:
            missing_files.append((component_name, file_path))
    
    # Print results
    print(f"\n✅ EXISTING FILES: {len(existing_files)}/69")
    print("-" * 50)
    for name, path in existing_files:
        print(f"  ✅ {name}")
    
    print(f"\n❌ MISSING FILES: {len(missing_files)}/69")
    print("-" * 50)
    for name, path in missing_files:
        print(f"  ❌ {name}: {path}")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total components: 69")
    print(f"  Files exist: {len(existing_files)}")
    print(f"  Files missing: {len(missing_files)}")
    print(f"  Existence rate: {len(existing_files)/69*100:.1f}%")
    
    return {
        'total': 69,
        'existing': len(existing_files),
        'missing': len(missing_files),
        'existing_files': existing_files,
        'missing_files': missing_files
    }


if __name__ == "__main__":
    result = test_file_existence()
