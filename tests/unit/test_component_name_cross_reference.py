#!/usr/bin/env python3
"""
Component Name Cross-Reference Test

Cross-references component names between ALL_COMPONENTS.md and orchestrator files
to identify any naming discrepancies or mismatches.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_component_name_cross_reference():
    """Cross-reference component names between ALL_COMPONENTS.md and orchestrator."""
    
    print("🔍 Cross-Referencing Component Names")
    print("=" * 60)
    
    # Components from ALL_COMPONENTS.md (69 total)
    all_components_md = {
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
    
    # Get orchestrator component mappings
    try:
        from prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader
        from prompt_improver.ml.orchestration.core.component_registry import ComponentTier
        
        loader = DirectComponentLoader()
        
        # Extract all orchestrator component names
        orchestrator_components = {}
        for tier, components in loader.component_paths.items():
            for comp_name, module_path in components.items():
                orchestrator_components[comp_name] = {
                    'tier': tier,
                    'module_path': module_path,
                    'file_path': module_path.replace('.', '/') + '.py'
                }
        
        print(f"📊 Component Counts:")
        print(f"  ALL_COMPONENTS.md: {len(all_components_md)} components")
        print(f"  Orchestrator loader: {len(orchestrator_components)} components")
        
        # Cross-reference analysis
        print(f"\n🔍 Cross-Reference Analysis:")
        print("-" * 50)
        
        # Find components in ALL_COMPONENTS.md but not in orchestrator
        missing_from_orchestrator = []
        for comp_name, file_path in all_components_md.items():
            # Convert to snake_case for comparison
            snake_case_name = ''.join(['_' + c.lower() if c.isupper() else c for c in comp_name]).lstrip('_')
            
            # Check various possible names
            possible_names = [
                snake_case_name,
                comp_name.lower(),
                comp_name.replace('Service', '').replace('System', '').replace('Framework', '').lower(),
            ]
            
            found = False
            for possible_name in possible_names:
                if possible_name in orchestrator_components:
                    found = True
                    break
            
            if not found:
                missing_from_orchestrator.append((comp_name, snake_case_name))
        
        # Find components in orchestrator but not in ALL_COMPONENTS.md
        extra_in_orchestrator = []
        for orch_name in orchestrator_components.keys():
            # Convert to PascalCase for comparison
            pascal_case_name = ''.join(word.capitalize() for word in orch_name.split('_'))
            
            found = False
            for comp_name in all_components_md.keys():
                if (comp_name == pascal_case_name or 
                    comp_name.lower() == orch_name.lower() or
                    comp_name.replace('Service', '').replace('System', '').replace('Framework', '') == pascal_case_name):
                    found = True
                    break
            
            if not found:
                extra_in_orchestrator.append(orch_name)
        
        # Print results
        print(f"\n❌ Components in ALL_COMPONENTS.md but NOT in orchestrator: {len(missing_from_orchestrator)}")
        for comp_name, snake_name in missing_from_orchestrator:
            print(f"  - {comp_name} (expected: {snake_name})")
        
        print(f"\n➕ Components in orchestrator but NOT in ALL_COMPONENTS.md: {len(extra_in_orchestrator)}")
        for orch_name in extra_in_orchestrator:
            print(f"  - {orch_name}")
        
        # Path mismatches
        print(f"\n🔍 Path Verification:")
        print("-" * 50)
        
        path_mismatches = []
        for comp_name, file_path in all_components_md.items():
            snake_case_name = ''.join(['_' + c.lower() if c.isupper() else c for c in comp_name]).lstrip('_')
            
            if snake_case_name in orchestrator_components:
                orch_path = "src/" + orchestrator_components[snake_case_name]['file_path']
                if orch_path != file_path:
                    path_mismatches.append((comp_name, file_path, orch_path))
        
        if path_mismatches:
            print(f"⚠️ Path mismatches found: {len(path_mismatches)}")
            for comp_name, md_path, orch_path in path_mismatches:
                print(f"  - {comp_name}:")
                print(f"    ALL_COMPONENTS.md: {md_path}")
                print(f"    Orchestrator:      {orch_path}")
        else:
            print("✅ No path mismatches found")
        
        return {
            'all_components_count': len(all_components_md),
            'orchestrator_count': len(orchestrator_components),
            'missing_from_orchestrator': missing_from_orchestrator,
            'extra_in_orchestrator': extra_in_orchestrator,
            'path_mismatches': path_mismatches
        }
        
    except Exception as e:
        print(f"❌ Error in cross-reference analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = test_component_name_cross_reference()
    if result:
        print(f"\n🎯 CROSS-REFERENCE SUMMARY:")
        print(f"  Missing from orchestrator: {len(result['missing_from_orchestrator'])}")
        print(f"  Extra in orchestrator: {len(result['extra_in_orchestrator'])}")
        print(f"  Path mismatches: {len(result['path_mismatches'])}")
    else:
        print("\n❌ Cross-reference analysis failed")
