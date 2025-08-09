"""
Component Integration Discrepancy Analysis

Investigates the 24-component gap between expected (77) and actual (53) 
integrated components in the ML Pipeline Orchestrator.
"""
import asyncio
from pathlib import Path
import re
import sys
from typing import Dict, List, Set, Tuple
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def analyze_component_integration_discrepancy():
    """Comprehensive analysis of component integration discrepancy."""
    print('🔍 Component Integration Discrepancy Analysis')
    print('=' * 60)
    expected_components = parse_all_components_md()
    print(f'📋 Expected components from ALL_COMPONENTS.md: {len(expected_components)}')
    actual_components = await get_orchestrator_components()
    print(f'📊 Actual components from orchestrator: {len(actual_components)}')
    missing_components = expected_components - actual_components
    print(f'❌ Missing components: {len(missing_components)}')
    await analyze_by_tier(expected_components, actual_components, missing_components)
    await check_component_definitions(missing_components)
    await check_direct_loader_paths(missing_components)
    matched, unmatched_expected, unmatched_actual = create_name_mapping_analysis(expected_components, actual_components)
    generate_recommendations(missing_components)

def parse_all_components_md() -> Set[str]:
    """Parse ALL_COMPONENTS.md to extract all component names."""
    components = set()
    try:
        with open('ALL_COMPONENTS.md', 'r') as f:
            content = f.read()
        pattern = '^(\\w+)\\s+✅\\s+\\*\\*Integrated\\*\\*'
        for line in content.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                component_name = match.group(1)
                components.add(component_name)
        print(f'📋 Parsed {len(components)} components from ALL_COMPONENTS.md')
        sample_components = list(components)[:10]
        print(f'📝 Sample components: {sample_components}')
    except Exception as e:
        print(f'❌ Error parsing ALL_COMPONENTS.md: {e}')
    return components

async def get_orchestrator_components() -> Set[str]:
    """Get actual components from the orchestrator."""
    components = set()
    try:
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        config = OrchestratorConfig(debug_mode=True)
        orchestrator = MLPipelineOrchestrator(config)
        await orchestrator.initialize()
        component_health = await orchestrator.get_component_health()
        components = set(component_health.keys())
        print(f'📊 Retrieved {len(components)} components from orchestrator')
        sample_components = list(components)[:10]
        print(f'📝 Sample orchestrator components: {sample_components}')
        await orchestrator.shutdown()
    except Exception as e:
        print(f'❌ Error getting orchestrator components: {e}')
    return components

async def analyze_by_tier(expected: Set[str], actual: Set[str], missing: Set[str]):
    """Analyze missing components by tier."""
    print(f'\n🏗️ Tier-Based Analysis')
    print('-' * 40)
    tier_patterns = {'Tier 1 (Core ML Pipeline)': ['TrainingDataLoader', 'MLModelService', 'RuleOptimizer', 'MultiarmedBanditFramework', 'AprioriAnalyzer', 'BatchProcessor', 'ProductionModelRegistry', 'ContextLearner', 'ClusteringOptimizer', 'AdvancedDimensionalityReducer', 'ProductionSyntheticDataGenerator', 'FailureModeAnalyzer'], 'Tier 2 (Optimization & Learning)': ['InsightGenerationEngine', 'RuleEffectivenessAnalyzer', 'ContextAwareFeatureWeighter', 'EnhancedOptimizationValidator', 'AdvancedPatternDiscovery', 'LLMTransformerService', 'AutoMLOrchestrator', 'AutoMLCallbacks', 'ContextCacheManager'], 'Tier 3 (Evaluation & Analysis)': ['CausalInferenceAnalyzer', 'AdvancedStatisticalValidator', 'PatternSignificanceAnalyzer', 'StructuralAnalyzer', 'ExperimentOrchestrator', 'StatisticalAnalyzer', 'DomainFeatureExtractor', 'LinguisticAnalyzer', 'DependencyParser', 'DomainDetector', 'NERExtractor'], 'Tier 4 (Performance & Infrastructure)': ['RealTimeMonitor', 'PerformanceMonitor', 'RealTimeAnalyticsService', 'AnalyticsService', 'PerformanceMetricsWidget', 'ModernABTestingService', 'CanaryTestingService', 'AsyncBatchProcessor', 'AdvancedEarlyStoppingFramework', 'BackgroundTaskManager', 'MultiLevelCache', 'ResourceManager', 'HealthService', 'MLResourceManagerHealthChecker', 'RedisHealthMonitor', 'DatabasePerformanceMonitor', 'DatabaseConnectionOptimizer', 'PreparedStatementCache', 'TypeSafePsycopgClient', 'APESServiceManager', 'UnifiedRetryManager', 'UnifiedKeyManager', 'RobustnessEvaluator', 'RetryManager', 'ABTestingWidget', 'ServiceControlWidget', 'SystemOverviewWidget'], 'Tier 5 (Infrastructure & Model Management)': ['ModelManager', 'EnhancedQualityScorer', 'PromptEnhancement', 'RedisCache', 'PerformanceValidator', 'PerformanceOptimizer'], 'Tier 6 (Security & Advanced)': ['InputSanitizer', 'MemoryGuard', 'AdversarialDefenseSystem', 'RobustnessEvaluator', 'DifferentialPrivacyService', 'FederatedLearningService', 'PerformanceBenchmark', 'ResponseOptimizer', 'AutoMLStatusWidget', 'PromptDataProtection'], 'Tier 7 (Feature Engineering)': ['CompositeFeatureExtractor', 'LinguisticFeatureExtractor', 'ContextFeatureExtractor']}
    for tier_name, tier_components in tier_patterns.items():
        tier_expected = set(tier_components) & expected
        tier_actual = set(tier_components) & actual
        tier_missing = set(tier_components) & missing
        if tier_expected:
            coverage = len(tier_actual) / len(tier_expected) * 100
            print(f'\n{tier_name}:')
            print(f'  Expected: {len(tier_expected)}')
            print(f'  Actual: {len(tier_actual)}')
            print(f'  Missing: {len(tier_missing)}')
            print(f'  Coverage: {coverage:.1f}%')
            if tier_missing:
                print(f'  Missing components: {list(tier_missing)[:5]}...')

async def check_component_definitions(missing_components: Set[str]):
    """Check if missing components are defined in component definitions."""
    print(f'\n📋 Component Definitions Analysis')
    print('-' * 40)
    try:
        from prompt_improver.ml.orchestration.config.component_definitions import ComponentDefinitions
        from prompt_improver.ml.orchestration.core.component_registry import ComponentTier
        component_defs = ComponentDefinitions()
        tiers = [ComponentTier.TIER_1_CORE, ComponentTier.TIER_2_OPTIMIZATION, ComponentTier.TIER_3_EVALUATION, ComponentTier.TIER_4_PERFORMANCE, ComponentTier.TIER_6_SECURITY]
        defined_components = set()
        for tier in tiers:
            tier_components = component_defs.get_tier_components(tier)
            defined_components.update(tier_components.keys())
            print(f'  {tier.value}: {len(tier_components)} components defined')
        print(f'\n📊 Total defined components: {len(defined_components)}')
        missing_in_definitions = missing_components & defined_components
        missing_not_in_definitions = missing_components - defined_components
        print(f'❌ Missing components that ARE in definitions: {len(missing_in_definitions)}')
        print(f'❌ Missing components NOT in definitions: {len(missing_not_in_definitions)}')
        if missing_not_in_definitions:
            print(f'📝 Components not in definitions: {list(missing_not_in_definitions)[:10]}')
    except Exception as e:
        print(f'❌ Error checking component definitions: {e}')

async def check_direct_loader_paths(missing_components: Set[str]):
    """Check if missing components have paths in DirectComponentLoader."""
    print(f'\n🔗 Direct Component Loader Analysis')
    print('-' * 40)
    try:
        from prompt_improver.ml.orchestration.integration.direct_component_loader import DirectComponentLoader
        loader = DirectComponentLoader()
        all_loader_components = set()
        for tier, components in loader.component_paths.items():
            all_loader_components.update(components.keys())
            print(f'  {tier.value}: {len(components)} components in loader')
        print(f'\n📊 Total components in loader: {len(all_loader_components)}')
        missing_in_loader = missing_components & all_loader_components
        missing_not_in_loader = missing_components - all_loader_components
        print(f'❌ Missing components that ARE in loader: {len(missing_in_loader)}')
        print(f'❌ Missing components NOT in loader: {len(missing_not_in_loader)}')
        if missing_not_in_loader:
            print(f'📝 Components not in loader: {list(missing_not_in_loader)[:10]}')
    except Exception as e:
        print(f'❌ Error checking direct component loader: {e}')

def generate_recommendations(missing_components: Set[str]):
    """Generate recommendations to fix the integration discrepancy."""
    print(f'\n💡 Recommendations')
    print('-' * 40)
    print('1. **Component Name Mapping Issue**:')
    print("   - ALL_COMPONENTS.md uses class names (e.g., 'TrainingDataLoader')")
    print("   - Orchestrator uses snake_case names (e.g., 'training_data_loader')")
    print('   - Need to create proper mapping between the two naming conventions')
    print('\n2. **Missing Component Definitions**:')
    print('   - Some components in ALL_COMPONENTS.md may not be in component_definitions.py')
    print('   - Need to add missing components to appropriate tier definitions')
    print('\n3. **DirectComponentLoader Gaps**:')
    print('   - Some components may not have import paths in DirectComponentLoader')
    print('   - Need to add missing component paths to component_paths mapping')
    print('\n4. **Conditional Loading**:')
    print('   - Some components may only load under specific conditions')
    print('   - Need to review initialization logic for conditional components')
    print('\n5. **Component Registry Discovery**:')
    print('   - Component discovery may not be finding all components')
    print('   - Need to enhance discovery mechanism to find all 77 components')

def create_name_mapping_analysis(expected_components: Set[str], actual_components: Set[str]):
    """Create mapping analysis between PascalCase and snake_case names."""
    print(f'\n🔄 Name Mapping Analysis')
    print('-' * 40)

    def pascal_to_snake(name: str) -> str:
        """Convert PascalCase to snake_case."""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', name)
        return re.sub('([a-z0-9])([A-Z])', '\\1_\\2', s1).lower()
    expected_to_snake = {}
    for component in expected_components:
        snake_name = pascal_to_snake(component)
        expected_to_snake[component] = snake_name
    matched_components = []
    unmatched_expected = []
    unmatched_actual = set(actual_components)
    for pascal_name, snake_name in expected_to_snake.items():
        if snake_name in actual_components:
            matched_components.append((pascal_name, snake_name))
            unmatched_actual.discard(snake_name)
        else:
            unmatched_expected.append((pascal_name, snake_name))
    print(f'✅ Matched components: {len(matched_components)}')
    print(f'❌ Unmatched expected: {len(unmatched_expected)}')
    print(f'❌ Unmatched actual: {len(unmatched_actual)}')
    print(f'\n📋 Sample matches:')
    for pascal, snake in matched_components[:10]:
        print(f'  {pascal} → {snake}')
    if unmatched_expected:
        print(f'\n❌ Unmatched expected (first 10):')
        for pascal, snake in unmatched_expected[:10]:
            print(f'  {pascal} → {snake} (not found)')
    if unmatched_actual:
        print(f'\n❌ Unmatched actual (first 10):')
        for snake in list(unmatched_actual)[:10]:
            print(f'  {snake} (no PascalCase equivalent found)')
    return (matched_components, unmatched_expected, unmatched_actual)
if __name__ == '__main__':
    asyncio.run(analyze_component_integration_discrepancy())
