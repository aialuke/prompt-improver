"""
Direct Component Loader for ML Pipeline Orchestrator.

Loads actual ML components and makes them available for orchestration.
"""

import importlib
import inspect
import logging
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from pathlib import Path

from ..core.component_registry import ComponentInfo, ComponentTier

@dataclass
class LoadedComponent:
    """Represents a loaded ML component with its metadata."""
    name: str
    component_class: Type
    instance: Optional[Any] = None
    module_path: str = ""
    dependencies: List[str] = None
    is_initialized: bool = False

class DirectComponentLoader:
    """
    Loads ML components directly from the codebase for orchestration.
    
    This provides actual component integration rather than placeholder connectors.
    """
    
    def __init__(self):
        """Initialize the direct component loader."""
        self.logger = logging.getLogger(__name__)
        self.loaded_components: Dict[str, LoadedComponent] = {}
        
        # Component path mappings for each tier
        self.component_paths = {
            ComponentTier.TIER_1_CORE: {
                "training_data_loader": "prompt_improver.ml.core.training_data_loader",
                "ml_integration": "prompt_improver.ml.core.ml_integration", 
                "rule_optimizer": "prompt_improver.ml.optimization.algorithms.rule_optimizer",
                "multi_armed_bandit": "prompt_improver.ml.optimization.algorithms.multi_armed_bandit",
                "apriori_analyzer": "prompt_improver.ml.learning.patterns.apriori_analyzer",
                "batch_processor": "prompt_improver.ml.optimization.batch.batch_processor",
                "production_registry": "prompt_improver.ml.models.production_registry",
                "context_learner": "prompt_improver.ml.learning.algorithms.context_learner",
                "clustering_optimizer": "prompt_improver.ml.optimization.algorithms.clustering_optimizer",
                "failure_analyzer": "prompt_improver.ml.learning.algorithms.failure_analyzer",
                "dimensionality_reducer": "prompt_improver.ml.optimization.algorithms.dimensionality_reducer",
                "synthetic_data_generator": "prompt_improver.ml.preprocessing.orchestrator",
            },
            ComponentTier.TIER_2_OPTIMIZATION: {
                "insight_engine": "prompt_improver.ml.learning.algorithms.insight_engine",
                "rule_analyzer": "prompt_improver.ml.learning.algorithms.rule_analyzer",
                "context_aware_weighter": "prompt_improver.ml.learning.algorithms.context_aware_weighter",
                "optimization_validator": "prompt_improver.ml.optimization.validation.optimization_validator",
                "advanced_pattern_discovery": "prompt_improver.ml.learning.patterns.advanced_pattern_discovery",
                "llm_transformer": "prompt_improver.ml.preprocessing.llm_transformer",
                "automl_orchestrator": "prompt_improver.ml.automl.orchestrator",
                "automl_callbacks": "prompt_improver.ml.automl.callbacks",
                "context_cache_manager": "prompt_improver.ml.learning.algorithms.context_cache_manager",
            },
            ComponentTier.TIER_3_EVALUATION: {
                "experiment_orchestrator": "prompt_improver.ml.evaluation.experiment_orchestrator",
                "advanced_statistical_validator": "prompt_improver.ml.evaluation.advanced_statistical_validator",
                "causal_inference_analyzer": "prompt_improver.ml.evaluation.causal_inference_analyzer",
                "pattern_significance_analyzer": "prompt_improver.ml.evaluation.pattern_significance_analyzer",
                "statistical_analyzer": "prompt_improver.ml.evaluation.statistical_analyzer",
                "structural_analyzer": "prompt_improver.ml.evaluation.structural_analyzer",
                "domain_feature_extractor": "prompt_improver.ml.analysis.domain_feature_extractor",
                "linguistic_analyzer": "prompt_improver.ml.analysis.linguistic_analyzer",
                "dependency_parser": "prompt_improver.ml.analysis.dependency_parser",
                "domain_detector": "prompt_improver.ml.analysis.domain_detector",
                "ner_extractor": "prompt_improver.ml.analysis.ner_extractor",
            },
            ComponentTier.TIER_4_PERFORMANCE: {
                "advanced_ab_testing": "prompt_improver.performance.testing.ab_testing_service",
                "canary_testing": "prompt_improver.performance.testing.canary_testing",
                "real_time_analytics": "prompt_improver.performance.analytics.real_time_analytics",
                "analytics": "prompt_improver.performance.analytics.analytics",
                "monitoring": "prompt_improver.performance.monitoring.monitoring",
                "performance_monitor": "prompt_improver.performance.monitoring.performance_monitor",
                "unified_retry_manager": "prompt_improver.core.retry_manager",
                "async_optimizer": "prompt_improver.performance.optimization.async_optimizer",
                "early_stopping": "prompt_improver.ml.optimization.algorithms.early_stopping",
                "background_manager": "prompt_improver.performance.monitoring.health.background_manager",
                "multi_level_cache": "prompt_improver.utils.multi_level_cache",
                "resource_manager": "prompt_improver.ml.orchestration.core.resource_manager",
                "health_service": "prompt_improver.performance.monitoring.health.service",
                "ml_resource_manager_health_checker": "prompt_improver.performance.monitoring.health.ml_orchestration_checkers",
                "redis_health_monitor": "prompt_improver.performance.monitoring.health.redis_monitor",
                "database_performance_monitor": "prompt_improver.database.performance_monitor",
                "database_connection_optimizer": "prompt_improver.database.query_optimizer",
                "prepared_statement_cache": "prompt_improver.database.query_optimizer",
                # "type_safe_psycopg_client": REMOVED - eliminated per DATABASE_CONSOLIDATION.md
                "apes_service_manager": "prompt_improver.core.services.manager",
                "unified_retry_manager": "prompt_improver.core.retry_manager",
                "secure_key_manager": "prompt_improver.security.key_manager",
                "fernet_key_manager": "prompt_improver.security.key_manager",
                "robustness_evaluator": "prompt_improver.security.adversarial_defense",
                "retry_manager": "prompt_improver.database.error_handling",
                "performance_metrics_widget": "prompt_improver.tui.widgets.performance_metrics",
                "ab_testing_widget": "prompt_improver.tui.widgets.ab_testing",
                "service_control_widget": "prompt_improver.tui.widgets.service_control",
                "system_overview_widget": "prompt_improver.tui.widgets.system_overview",
            },
            ComponentTier.TIER_5_INFRASTRUCTURE: {
                "model_manager": "prompt_improver.ml.models.model_manager",
                "enhanced_scorer": "prompt_improver.ml.learning.quality.enhanced_scorer",
                "prompt_enhancement": "prompt_improver.ml.models.prompt_enhancement",
                "redis_cache": "prompt_improver.utils.redis_cache",
                "performance_validation": "prompt_improver.performance.validation.performance_validation",
                "performance_optimizer": "prompt_improver.performance.optimization.performance_optimizer",
            },
            ComponentTier.TIER_6_SECURITY: {
                "input_sanitizer": "prompt_improver.security.input_sanitization",
                "memory_guard": "prompt_improver.security.memory_guard",
                "adversarial_defense": "prompt_improver.security.adversarial_defense",
                "robustness_evaluator": "prompt_improver.security.adversarial_defense",
                "differential_privacy": "prompt_improver.security.differential_privacy",
                "federated_learning": "prompt_improver.security.federated_learning",
                "performance_benchmark": "prompt_improver.performance.monitoring.performance_benchmark",
                "response_optimizer": "prompt_improver.performance.optimization.response_optimizer",
                "automl_status": "prompt_improver.tui.widgets.automl_status",
                "prompt_data_protection": "prompt_improver.core.services.security",
            },
            ComponentTier.TIER_7_FEATURE_ENGINEERING: {
                "composite_feature_extractor": "prompt_improver.ml.learning.features.composite_feature_extractor",
                "linguistic_feature_extractor": "prompt_improver.ml.learning.features.linguistic_feature_extractor",
                "context_feature_extractor": "prompt_improver.ml.learning.features.context_feature_extractor",
            }
        }
    
    async def load_component(self, component_name: str, tier: ComponentTier) -> Optional[LoadedComponent]:
        """
        Load a specific ML component by name and tier.
        
        Args:
            component_name: Name of the component to load
            tier: Component tier for path resolution
            
        Returns:
            LoadedComponent instance if successful, None if failed
        """
        if component_name in self.loaded_components:
            return self.loaded_components[component_name]
        
        if tier not in self.component_paths:
            self.logger.error(f"Unknown tier: {tier}")
            return None
            
        if component_name not in self.component_paths[tier]:
            self.logger.error(f"Component {component_name} not found in tier {tier}")
            return None
        
        module_path = self.component_paths[tier][component_name]
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Find the main class in the module
            component_class = self._find_main_class(module, component_name)
            
            if component_class is None:
                self.logger.error(f"No suitable class found in {module_path}")
                return None
            
            # Create loaded component
            loaded_component = LoadedComponent(
                name=component_name,
                component_class=component_class,
                module_path=module_path,
                dependencies=self._extract_dependencies(component_class),
                is_initialized=False
            )
            
            self.loaded_components[component_name] = loaded_component
            self.logger.info(f"Successfully loaded component: {component_name}")
            
            return loaded_component
            
        except ImportError as e:
            self.logger.error(f"Failed to import {module_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading component {component_name}: {e}")
            return None
    
    async def load_tier_components(self, tier: ComponentTier) -> Dict[str, LoadedComponent]:
        """
        Load all components in a specific tier.
        
        Args:
            tier: Component tier to load
            
        Returns:
            Dictionary of component name to LoadedComponent
        """
        loaded_tier_components = {}
        
        if tier not in self.component_paths:
            self.logger.error(f"Unknown tier: {tier}")
            return loaded_tier_components
        
        for component_name in self.component_paths[tier]:
            loaded_component = await self.load_component(component_name, tier)
            if loaded_component:
                loaded_tier_components[component_name] = loaded_component
        
        self.logger.info(f"Loaded {len(loaded_tier_components)}/{len(self.component_paths[tier])} components for {tier}")
        return loaded_tier_components
    
    async def load_all_components(self) -> Dict[str, LoadedComponent]:
        """
        Load all components across all tiers.
        
        Returns:
            Dictionary of all loaded components
        """
        all_loaded = {}
        
        for tier in ComponentTier:
            tier_components = await self.load_tier_components(tier)
            all_loaded.update(tier_components)
        
        self.logger.info(f"Loaded {len(all_loaded)} total components")
        return all_loaded
    
    def _find_main_class(self, module: Any, component_name: str) -> Optional[Type]:
        """
        Find the main class in a module for a given component.
        
        Args:
            module: The imported module
            component_name: Name of the component
            
        Returns:
            The main class type if found
        """
        # Get all classes from the module
        module_classes = [obj for name, obj in inspect.getmembers(module) 
                         if inspect.isclass(obj) and obj.__module__ == module.__name__]
        
        if not module_classes:
            return None
        
        # Specific mappings for problematic components
        specific_mappings = {
            "enhanced_scorer": "EnhancedQualityScorer",
            "monitoring": "RealTimeMonitor",
            "performance_monitor": "PerformanceMonitor",
            "performance_validation": "PerformanceValidator",
            "multi_armed_bandit": "MultiarmedBanditFramework",
            "context_learner": "ContextLearner",
            "failure_analyzer": "FailureModeAnalyzer",
            "insight_engine": "InsightGenerationEngine",
            "rule_analyzer": "RuleEffectivenessAnalyzer",
            "automl_orchestrator": "AutoMLOrchestrator",
            "ner_extractor": "NERExtractor",
            "background_manager": "BackgroundTaskManager",
            "automl_status": "AutoMLStatusWidget",
            "dimensionality_reducer": "AdvancedDimensionalityReducer",
            "synthetic_data_generator": "ProductionSyntheticDataGenerator",
            "unified_retry_manager": "RetryManager",
            "multi_level_cache": "MultiLevelCache",
            "resource_manager": "ResourceManager",
            "health_service": "EnhancedHealthService",
            "ml_resource_manager_health_checker": "MLResourceManagerHealthChecker",
            "redis_health_monitor": "RedisHealthMonitor",
            "prepared_statement_cache": "PreparedStatementCache",
            # "type_safe_psycopg_client": REMOVED - eliminated per DATABASE_CONSOLIDATION.md
            "context_cache_manager": "ContextCacheManager",
            "apes_service_manager": "APESServiceManager",
            "unified_retry_manager": "RetryManager",
            "secure_key_manager": "UnifiedKeyManager",
            "fernet_key_manager": "UnifiedKeyManager",
            "robustness_evaluator": "RobustnessEvaluator",
            "retry_manager": "RetryManager",
            "performance_metrics_widget": "PerformanceMetricsWidget",
            "ab_testing_widget": "ABTestingWidget",
            "service_control_widget": "ServiceControlWidget",
            "system_overview_widget": "SystemOverviewWidget",
            "input_sanitizer": "InputSanitizer",
            "memory_guard": "MemoryGuard",
            "prompt_data_protection": "PromptDataProtection",
            "composite_feature_extractor": "CompositeFeatureExtractor",
            "linguistic_feature_extractor": "LinguisticFeatureExtractor",
            "context_feature_extractor": "ContextFeatureExtractor",
        }
        
        # Check specific mappings first
        if component_name in specific_mappings:
            target_class_name = specific_mappings[component_name]
            for cls in module_classes:
                if cls.__name__ == target_class_name:
                    return cls
        
        # Common class name patterns
        possible_names = [
            component_name.title().replace("_", ""),  # training_data_loader -> TrainingDataLoader
            component_name.replace("_", " ").title().replace(" ", ""),  # Same but with spaces
            f"{component_name.title().replace('_', '')}Service",  # Add Service suffix
            f"{component_name.title().replace('_', '')}Manager",  # Add Manager suffix
            f"{component_name.title().replace('_', '')}Analyzer",  # Add Analyzer suffix
            f"{component_name.title().replace('_', '')}Optimizer",  # Add Optimizer suffix
            f"{component_name.title().replace('_', '')}Framework",  # Add Framework suffix
            f"{component_name.title().replace('_', '')}Engine",  # Add Engine suffix
            f"{component_name.title().replace('_', '')}Validator",  # Add Validator suffix
            f"{component_name.title().replace('_', '')}Monitor",  # Add Monitor suffix
            f"{component_name.title().replace('_', '')}Extractor",  # Add Extractor suffix
            f"{component_name.title().replace('_', '')}Widget",  # Add Widget suffix
        ]
        
        # Try to find by name pattern
        for class_name in possible_names:
            for cls in module_classes:
                if cls.__name__ == class_name:
                    return cls
        
        # Score classes based on how likely they are to be the main class
        def score_class(cls):
            score = 0
            class_name = cls.__name__
            
            # Prefer classes that are not data classes or models
            if not hasattr(cls, '__dataclass_fields__'):
                score += 10
            
            # Prefer classes with multiple methods
            methods = [m for m in dir(cls) if not m.startswith('_') and callable(getattr(cls, m, None))]
            score += len(methods)
            
            # Prefer classes that end with service-like suffixes
            service_suffixes = ['Service', 'Manager', 'Analyzer', 'Optimizer', 'Framework', 
                              'Engine', 'Validator', 'Monitor', 'Extractor', 'Widget', 'Orchestrator']
            for suffix in service_suffixes:
                if class_name.endswith(suffix):
                    score += 20
                    break
            
            # Avoid data classes, results, configs
            avoid_patterns = ['Result', 'Config', 'Metrics', 'Alert', 'Task', 'Status']
            for pattern in avoid_patterns:
                if pattern in class_name:
                    score -= 5
            
            # Avoid very simple classes
            if len(class_name) < 4:
                score -= 5
                
            # Avoid ABC, BaseModel, Enum
            if class_name in ['ABC', 'BaseModel', 'Enum']:
                score -= 20
            
            return score
        
        # Sort classes by score and return the best one
        scored_classes = [(score_class(cls), cls) for cls in module_classes]
        scored_classes.sort(key=lambda x: x[0], reverse=True)
        
        return scored_classes[0][1] if scored_classes else None
    
    def _extract_dependencies(self, component_class: Type) -> List[str]:
        """
        Extract dependencies from a component class.
        
        Args:
            component_class: The component class to analyze
            
        Returns:
            List of dependency names
        """
        dependencies = []
        
        # Check __init__ signature for dependencies
        try:
            init_signature = inspect.signature(component_class.__init__)
            for param_name, param in init_signature.parameters.items():
                if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                    # Extract simple type names as dependencies
                    dep_name = getattr(param.annotation, '__name__', str(param.annotation))
                    if dep_name not in ['str', 'int', 'float', 'bool', 'dict', 'list', 'Optional']:
                        dependencies.append(dep_name)
        except Exception:
            pass
        
        return dependencies
    
    async def initialize_component(self, component_name: str, **kwargs) -> bool:
        """
        Initialize a loaded component with given parameters.
        
        Args:
            component_name: Name of the component to initialize
            **kwargs: Initialization parameters
            
        Returns:
            True if initialization successful
        """
        if component_name not in self.loaded_components:
            self.logger.error(f"Component {component_name} not loaded")
            return False
        
        loaded_component = self.loaded_components[component_name]
        
        if loaded_component.is_initialized:
            return True
        
        try:
            # Create instance
            loaded_component.instance = loaded_component.component_class(**kwargs)
            loaded_component.is_initialized = True
            
            self.logger.info(f"Initialized component: {component_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {component_name}: {e}")
            return False
    
    def get_loaded_component(self, component_name: str) -> Optional[LoadedComponent]:
        """Get a loaded component by name."""
        return self.loaded_components.get(component_name)
    
    def get_all_loaded_components(self) -> Dict[str, LoadedComponent]:
        """Get all loaded components."""
        return self.loaded_components.copy()
    
    def is_component_loaded(self, component_name: str) -> bool:
        """Check if a component is loaded."""
        return component_name in self.loaded_components
    
    def is_component_initialized(self, component_name: str) -> bool:
        """Check if a component is loaded and initialized."""
        loaded_component = self.loaded_components.get(component_name)
        return loaded_component is not None and loaded_component.is_initialized