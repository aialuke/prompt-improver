"""
Component Registry for ML Pipeline orchestration.

Manages registration, discovery, and health monitoring of all ML components.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

from ..config.orchestrator_config import OrchestratorConfig

class ComponentTier(Enum):
    """Component tier classifications."""
    TIER_1_CORE = "tier1_core"  # Core ML Pipeline (11 components)
    TIER_2_OPTIMIZATION = "tier2_optimization"  # Optimization & Learning (8 components)
    TIER_3_EVALUATION = "tier3_evaluation"  # Evaluation & Analysis (10 components)
    TIER_4_PERFORMANCE = "tier4_performance"  # Performance & Testing (8 components)
    TIER_5_INFRASTRUCTURE = "tier5_infrastructure"  # Model & Infrastructure (6 components)
    TIER_6_SECURITY = "tier6_security"  # Security & Advanced (7+ components)
    TIER_7_FEATURE_ENGINEERING = "tier7_feature_engineering"  # Feature Engineering Components (3 components)

class ComponentStatus(Enum):
    """Component status states."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    unhealthy = "unhealthy"
    starting = "starting"
    stopping = "stopping"
    ERROR = "error"

@dataclass
class ComponentCapability:
    """Represents a capability that a component provides."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComponentInfo:
    """Information about a registered ML component."""
    name: str
    tier: ComponentTier
    description: str
    version: str
    capabilities: List[ComponentCapability]
    health_check_endpoint: Optional[str] = None
    api_endpoints: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime status
    status: ComponentStatus = ComponentStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    error_message: Optional[str] = None
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class ComponentRegistry:
    """
    Registry for all ML pipeline components.
    
    Manages the 50+ components across 6 tiers:
    - Discovery and registration
    - Health monitoring
    - Capability tracking
    - Dependency management
    """
    
    def __init__(self, config: OrchestratorConfig):
        """Initialize the component registry."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Registry storage
        self.components: Dict[str, ComponentInfo] = {}
        self.components_by_tier: Dict[ComponentTier, List[str]] = {
            tier: [] for tier in ComponentTier
        }
        
        # Health monitoring
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
    
    async def initialize(self) -> None:
        """Initialize the component registry."""
        self.logger.info("Initializing component registry")
        
        # Load component definitions
        await self._load_component_definitions()
        
        # Start health monitoring
        await self._start_health_monitoring()
        
        self.logger.info(f"Component registry initialized with {len(self.components)} components")
    
    async def shutdown(self) -> None:
        """Shutdown the component registry."""
        self.logger.info("Shutting down component registry")
        
        # Stop health monitoring
        await self._stop_health_monitoring()
        
        self.logger.info("Component registry shutdown complete")
    
    async def register_component(self, component_info: ComponentInfo) -> None:
        """
        Register a new ML component.
        
        Args:
            component_info: Component information and capabilities
        """
        name = component_info.name
        
        if name in self.components:
            self.logger.warning(f"Component {name} already registered, updating")
        
        # Register component
        self.components[name] = component_info
        self.components_by_tier[component_info.tier].append(name)
        
        self.logger.info(f"Registered component {name} in {component_info.tier.value}")
    
    async def unregister_component(self, component_name: str) -> bool:
        """
        Unregister a component.
        
        Args:
            component_name: Name of component to unregister
            
        Returns:
            True if component was found and removed
        """
        if component_name not in self.components:
            return False
        
        component_info = self.components[component_name]
        
        # Remove from tier list
        if component_name in self.components_by_tier[component_info.tier]:
            self.components_by_tier[component_info.tier].remove(component_name)
        
        # Remove from main registry
        del self.components[component_name]
        
        self.logger.info(f"Unregistered component {component_name}")
        return True
    
    async def get_component(self, component_name: str) -> Optional[ComponentInfo]:
        """Get component information by name."""
        return self.components.get(component_name)
    
    async def list_components(self, tier: Optional[ComponentTier] = None) -> List[ComponentInfo]:
        """
        List registered components.
        
        Args:
            tier: Filter by specific tier (optional)
            
        Returns:
            List of component information
        """
        if tier:
            component_names = self.components_by_tier[tier]
            return [self.components[name] for name in component_names]
        
        return list(self.components.values())
    
    async def get_components_by_capability(self, capability_name: str) -> List[ComponentInfo]:
        """
        Find components that provide a specific capability.
        
        Args:
            capability_name: Name of the capability to search for
            
        Returns:
            List of components that provide the capability
        """
        matching_components = []
        
        for component in self.components.values():
            for capability in component.capabilities:
                if capability.name == capability_name:
                    matching_components.append(component)
                    break
        
        return matching_components
    
    async def check_component_health(self, component_name: str) -> ComponentStatus:
        """
        Check the health of a specific component.
        
        Args:
            component_name: Name of component to check
            
        Returns:
            Current health status
        """
        component = self.components.get(component_name)
        if not component:
            return ComponentStatus.UNKNOWN
        
        try:
            # Perform health check
            status = await self._perform_health_check(component)
            
            # Update component status
            component.status = status
            component.last_health_check = datetime.now(timezone.utc)
            component.error_message = None
            
            return status
            
        except Exception as e:
            component.status = ComponentStatus.ERROR
            component.error_message = str(e)
            component.last_health_check = datetime.now(timezone.utc)
            
            self.logger.error(f"Health check failed for {component_name}: {e}")
            return ComponentStatus.ERROR
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """
        Get overall health summary of all components.
        
        Returns:
            Health summary with statistics
        """
        total_components = len(self.components)
        status_counts = {status.value: 0 for status in ComponentStatus}
        tier_health = {tier.value: {"total": 0, "healthy": 0} for tier in ComponentTier}
        
        for component in self.components.values():
            status_counts[component.status.value] += 1
            tier_health[component.tier.value]["total"] += 1
            
            if component.status == ComponentStatus.HEALTHY:
                tier_health[component.tier.value]["healthy"] += 1
        
        return {
            "total_components": total_components,
            "status_distribution": status_counts,
            "tier_health": tier_health,
            "overall_health_percentage": (
                status_counts["healthy"] / total_components * 100 
                if total_components > 0 else 0
            )
        }
    
    async def discover_components(self) -> List[ComponentInfo]:
        """
        Discover components from the codebase.

        This method loads components from component definitions and registers them.

        Returns:
            List of discovered components
        """
        self.logger.info("Discovering ML components")

        discovered_components = []

        try:
            # Create predefined components to avoid circular imports
            predefined_components = self._get_predefined_components()

            # Create ComponentInfo objects for each predefined component
            for component_name, component_data in predefined_components.items():
                try:
                    # Create ComponentInfo
                    component_info = ComponentInfo(
                        name=component_name,
                        tier=component_data["tier"],
                        capabilities=component_data["capabilities"],
                        dependencies=component_data.get("dependencies", []),
                        description=component_data.get("description", ""),
                        version=component_data.get("version", "1.0.0"),
                        metadata={
                            "module_path": component_data.get("module_path", ""),
                            "class_name": component_data.get("class_name", "")
                        }
                    )

                    discovered_components.append(component_info)
                    self.logger.debug(f"Discovered component: {component_name} (Tier: {component_data['tier'].value})")

                except Exception as e:
                    self.logger.warning(f"Failed to create ComponentInfo for {component_name}: {e}")
                    continue

            self.logger.info(f"Discovered {len(discovered_components)} components")

        except Exception as e:
            self.logger.error(f"Component discovery failed: {e}")

        return discovered_components

    def _get_predefined_components(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined components to avoid circular imports."""

        return {
            "rule_analyzer": {
                "tier": ComponentTier.TIER_2_OPTIMIZATION,
                "capabilities": [
                    ComponentCapability(
                        name="bayesian_analysis",
                        description="Bayesian effectiveness analysis using PyMC",
                        input_types=["performance_data"],
                        output_types=["analysis_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.learning.algorithms.rule_analyzer",
                "class_name": "RuleEffectivenessAnalyzer",
                "description": "Enhanced rule analyzer with Bayesian capabilities",
                "version": "1.1.0"
            },
            "experiment_orchestrator": {
                "tier": ComponentTier.TIER_3_EVALUATION,
                "capabilities": [
                    ComponentCapability(
                        name="bayesian_ab_testing",
                        description="Bayesian A/B testing with real-time monitoring",
                        input_types=["experiment_config"],
                        output_types=["experiment_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.evaluation.experiment_orchestrator",
                "class_name": "ExperimentOrchestrator",
                "description": "Enhanced experiment orchestrator with Bayesian A/B testing",
                "version": "1.1.0"
            },
            "automl_orchestrator": {
                "tier": ComponentTier.TIER_2_OPTIMIZATION,
                "capabilities": [
                    ComponentCapability(
                        name="automated_optimization",
                        description="Automated ML optimization and hyperparameter tuning",
                        input_types=["optimization_config"],
                        output_types=["optimization_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.orchestration.coordinators.optimization_controller",
                "class_name": "OptimizationController",
                "description": "Enhanced optimization controller with Bayesian optimization",
                "version": "1.1.0"
            },
            "rule_optimizer": {
                "tier": ComponentTier.TIER_2_OPTIMIZATION,
                "capabilities": [
                    ComponentCapability(
                        name="rule_optimization",
                        description="Rule optimization using various algorithms",
                        input_types=["rule_config"],
                        output_types=["optimized_rule"]
                    )
                ],
                "module_path": "prompt_improver.ml.optimization.algorithms.rule_optimizer",
                "class_name": "RuleOptimizer",
                "description": "Rule optimization with Gaussian Process support",
                "version": "1.0.0"
            },
            "insight_engine": {
                "tier": ComponentTier.TIER_2_OPTIMIZATION,
                "capabilities": [
                    ComponentCapability(
                        name="insight_generation",
                        description="Causal discovery and automated insights generation",
                        input_types=["performance_data"],
                        output_types=["insights_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.learning.algorithms.insight_engine",
                "class_name": "InsightGenerationEngine",
                "description": "Advanced insight generation with causal discovery",
                "version": "1.0.0"
            },
            "synthetic_data_generator": {
                "tier": ComponentTier.TIER_2_OPTIMIZATION,
                "capabilities": [
                    ComponentCapability(
                        name="synthetic_data_generation",
                        description="Production synthetic data generation with modern generative models",
                        input_types=["generation_config"],
                        output_types=["synthetic_data_result"]
                    ),
                    ComponentCapability(
                        name="neural_generation",
                        description="Neural network-based synthetic data generation (VAE, GAN)",
                        input_types=["neural_config"],
                        output_types=["neural_synthetic_data"]
                    ),
                    ComponentCapability(
                        name="diffusion_generation",
                        description="Diffusion model-based synthetic data generation",
                        input_types=["diffusion_config"],
                        output_types=["diffusion_synthetic_data"]
                    )
                ],
                "module_path": "prompt_improver.ml.preprocessing.synthetic_data_generator",
                "class_name": "ProductionSyntheticDataGenerator",
                "description": "Production synthetic data generator with modern generative models",
                "version": "1.0.0"
            },
            "ml_model_service": {
                "tier": ComponentTier.TIER_2_OPTIMIZATION,
                "capabilities": [
                    ComponentCapability(
                        name="model_training",
                        description="ML model training with production deployment capabilities",
                        input_types=["training_config"],
                        output_types=["training_result"]
                    ),
                    ComponentCapability(
                        name="model_deployment",
                        description="Production model deployment with blue-green strategies",
                        input_types=["deployment_config"],
                        output_types=["deployment_result"]
                    ),
                    ComponentCapability(
                        name="model_serving",
                        description="ML model serving with caching and monitoring",
                        input_types=["prediction_request"],
                        output_types=["prediction_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.core.ml_integration",
                "class_name": "MLModelService",
                "description": "Enhanced ML service with production deployment capabilities",
                "version": "1.0.0"
            },
            "failure_analyzer": {
                "tier": ComponentTier.TIER_2_OPTIMIZATION,
                "capabilities": [
                    ComponentCapability(
                        name="failure_analysis",
                        description="Automated failure pattern analysis with robustness testing",
                        input_types=["test_results"],
                        output_types=["failure_analysis_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.learning.algorithms.failure_analyzer",
                "class_name": "FailureModeAnalyzer",
                "description": "Advanced failure mode analysis with adversarial robustness",
                "version": "1.0.0"
            },
            "context_learner": {
                "tier": ComponentTier.TIER_2_OPTIMIZATION,
                "capabilities": [
                    ComponentCapability(
                        name="context_learning",
                        description="Context-specific learning with adaptive clustering",
                        input_types=["context_data"],
                        output_types=["learning_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.learning.algorithms.context_learner",
                "class_name": "ContextLearner",
                "description": "Context learner with modular architecture",
                "version": "1.0.0"
            },
            "enhanced_quality_scorer": {
                "tier": ComponentTier.TIER_3_EVALUATION,
                "capabilities": [
                    ComponentCapability(
                        name="quality_assessment",
                        description="Multi-dimensional quality assessment with statistical validation",
                        input_types=["features", "effectiveness_scores"],
                        output_types=["quality_metrics"]
                    )
                ],
                "module_path": "prompt_improver.ml.learning.quality.enhanced_scorer",
                "class_name": "EnhancedQualityScorer",
                "description": "Advanced quality scoring with multi-dimensional assessment",
                "version": "1.0.0"
            },
            "enhanced_structural_analyzer": {
                "tier": ComponentTier.TIER_3_EVALUATION,
                "capabilities": [
                    ComponentCapability(
                        name="structural_analysis",
                        description="2025 enhanced structural analysis with graph-based representation and semantic understanding",
                        input_types=["text"],
                        output_types=["structural_analysis_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.evaluation.structural_analyzer",
                "class_name": "EnhancedStructuralAnalyzer",
                "description": "Enhanced structural analyzer with graph analysis, semantic understanding, and automated pattern discovery",
                "version": "2025.1.0"
            },
            "causal_inference_analyzer": {
                "tier": ComponentTier.TIER_3_EVALUATION,
                "capabilities": [
                    ComponentCapability(
                        name="causal_analysis",
                        description="Advanced causal inference analysis with multiple methods and assumption testing",
                        input_types=["outcome_data", "treatment_data", "covariates"],
                        output_types=["causal_analysis_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.evaluation.causal_inference_analyzer",
                "class_name": "CausalInferenceAnalyzer",
                "description": "Advanced causal inference analyzer with 2025 best practices",
                "version": "1.0.0"
            },
            "advanced_statistical_validator": {
                "tier": ComponentTier.TIER_3_EVALUATION,
                "capabilities": [
                    ComponentCapability(
                        name="statistical_validation",
                        description="Advanced statistical validation with multiple testing corrections and effect size analysis",
                        input_types=["control_data", "treatment_data"],
                        output_types=["validation_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.evaluation.advanced_statistical_validator",
                "class_name": "AdvancedStatisticalValidator",
                "description": "Advanced statistical validator with 2025 best practices",
                "version": "1.0.0"
            },
            "pattern_significance_analyzer": {
                "tier": ComponentTier.TIER_3_EVALUATION,
                "capabilities": [
                    ComponentCapability(
                        name="pattern_analysis",
                        description="Pattern significance analysis with multiple pattern types and business insights",
                        input_types=["patterns_data", "control_data", "treatment_data"],
                        output_types=["pattern_analysis_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.evaluation.pattern_significance_analyzer",
                "class_name": "PatternSignificanceAnalyzer",
                "description": "Pattern significance analyzer with 2025 best practices",
                "version": "1.0.0"
            },
            "multiarmed_bandit_framework": {
                "tier": ComponentTier.TIER_2_OPTIMIZATION,
                "capabilities": [
                    ComponentCapability(
                        name="bandit_optimization",
                        description="Multi-armed bandit optimization with contextual bandits and Thompson sampling",
                        input_types=["arms", "context_data", "reward_data"],
                        output_types=["bandit_optimization_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.optimization.algorithms.multi_armed_bandit",
                "class_name": "MultiarmedBanditFramework",
                "description": "Advanced multi-armed bandit framework with 2025 best practices",
                "version": "1.0.0"
            },
            "clustering_optimizer": {
                "tier": ComponentTier.TIER_2_OPTIMIZATION,
                "capabilities": [
                    ComponentCapability(
                        name="clustering_optimization",
                        description="Advanced clustering optimization with UMAP and HDBSCAN",
                        input_types=["features", "labels", "sample_weights"],
                        output_types=["clustering_optimization_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.optimization.algorithms.clustering_optimizer",
                "class_name": "ClusteringOptimizer",
                "description": "Advanced clustering optimizer with 2025 best practices",
                "version": "1.0.0"
            },
            "advanced_early_stopping_framework": {
                "tier": ComponentTier.TIER_2_OPTIMIZATION,
                "capabilities": [
                    ComponentCapability(
                        name="early_stopping",
                        description="Advanced early stopping with group sequential design and alpha spending",
                        input_types=["control_data", "treatment_data", "stopping_criteria"],
                        output_types=["early_stopping_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.optimization.algorithms.early_stopping",
                "class_name": "AdvancedEarlyStoppingFramework",
                "description": "Advanced early stopping framework with 2025 best practices",
                "version": "1.0.0"
            },
            "enhanced_optimization_validator": {
                "tier": ComponentTier.TIER_2_OPTIMIZATION,
                "capabilities": [
                    ComponentCapability(
                        name="optimization_validation",
                        description="Enhanced optimization validation with Bayesian methods, robust statistics, and causal inference",
                        input_types=["baseline_data", "optimized_data", "validation_criteria"],
                        output_types=["validation_result"]
                    )
                ],
                "module_path": "prompt_improver.ml.optimization.validation.optimization_validator",
                "class_name": "EnhancedOptimizationValidator",
                "description": "Enhanced optimization validator with 2025 best practices",
                "version": "2025.1.0"
            }
        }
    
    async def _load_component_definitions(self) -> None:
        """Load component definitions from configuration."""
        from ..config.component_definitions import ComponentDefinitions
        
        component_defs = ComponentDefinitions()
        
        # Load Tier 1 components
        tier1_defs = component_defs.get_tier_components(ComponentTier.TIER_1_CORE)
        for name, definition in tier1_defs.items():
            component_info = component_defs.create_component_info(name, definition, ComponentTier.TIER_1_CORE)
            await self.register_component(component_info)
        
        # Load Tier 2 components
        tier2_defs = component_defs.get_tier_components(ComponentTier.TIER_2_OPTIMIZATION)
        for name, definition in tier2_defs.items():
            component_info = component_defs.create_component_info(name, definition, ComponentTier.TIER_2_OPTIMIZATION)
            await self.register_component(component_info)
        
        # Load Tier 3 components
        tier3_defs = component_defs.get_tier_components(ComponentTier.TIER_3_EVALUATION)
        for name, definition in tier3_defs.items():
            component_info = component_defs.create_component_info(name, definition, ComponentTier.TIER_3_EVALUATION)
            await self.register_component(component_info)
        
        # Load Tier 4 components
        tier4_defs = component_defs.get_tier_components(ComponentTier.TIER_4_PERFORMANCE)
        for name, definition in tier4_defs.items():
            component_info = component_defs.create_component_info(name, definition, ComponentTier.TIER_4_PERFORMANCE)
            await self.register_component(component_info)
        
        # Load Tier 6 security components
        tier6_defs = component_defs.get_tier_components(ComponentTier.TIER_6_SECURITY)
        for name, definition in tier6_defs.items():
            component_info = component_defs.create_component_info(name, definition, ComponentTier.TIER_6_SECURITY)
            await self.register_component(component_info)
        
        total_components = len(tier1_defs) + len(tier2_defs) + len(tier3_defs) + len(tier4_defs) + len(tier6_defs)
        self.logger.info(f"Loaded definitions for {total_components} components ({len(tier1_defs)} Tier 1, {len(tier2_defs)} Tier 2, {len(tier3_defs)} Tier 3, {len(tier4_defs)} Tier 4, {len(tier6_defs)} Tier 6 Security)")
    
    async def _start_health_monitoring(self) -> None:
        """Start periodic health monitoring of components."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
        
        self.logger.info("Started component health monitoring")
    
    async def _stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped component health monitoring")
    
    async def _health_monitoring_loop(self) -> None:
        """Periodic health monitoring loop."""
        while self.is_monitoring:
            try:
                # Check health of all components
                for component_name in self.components:
                    await self.check_component_health(component_name)
                
                # Wait for next check interval
                await asyncio.sleep(self.config.component_health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _perform_health_check(self, component: ComponentInfo) -> ComponentStatus:
        """
        Perform health check for a component.
        
        Args:
            component: Component to check
            
        Returns:
            Health status
        """
        # For Phase 1, we'll implement basic health checks
        # Later phases will implement actual component-specific checks
        
        if component.health_check_endpoint:
            # Would perform HTTP health check
            # For now, return HEALTHY as placeholder
            return ComponentStatus.HEALTHY
        
        # Basic check - assume component is healthy if recently registered
        time_since_registration = datetime.now(timezone.utc) - component.registered_at
        if time_since_registration.total_seconds() < 300:  # 5 minutes
            return ComponentStatus.HEALTHY
        
        return ComponentStatus.UNKNOWN