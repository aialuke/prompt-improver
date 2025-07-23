"""
Workflow templates for ML Pipeline orchestration.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field

from ..core.workflow_types import WorkflowDefinition, WorkflowStep


@dataclass
class WorkflowTemplates:
    """Collection of predefined workflow templates."""
    
    @staticmethod
    def get_tier1_training_workflow() -> WorkflowDefinition:
        """Core ML training workflow using Tier 1 components."""
        return WorkflowDefinition(
            workflow_type="tier1_training",
            name="Core ML Training Workflow", 
            description="Basic training workflow using core ML components",
            steps=[
                WorkflowStep(
                    step_id="load_data",
                    name="Load Training Data",
                    component_name="training_data_loader",
                    parameters={"batch_size": 1000},
                    timeout=300
                ),
                WorkflowStep(
                    step_id="train_model",
                    name="Train ML Model",
                    component_name="ml_integration",
                    parameters={"epochs": 10},
                    dependencies=["load_data"],
                    timeout=1800
                ),
                WorkflowStep(
                    step_id="optimize_rules",
                    name="Optimize Rules",
                    component_name="rule_optimizer",
                    parameters={"iterations": 100},
                    dependencies=["train_model"],
                    timeout=600
                )
            ],
            global_timeout=3600
        )
    
    @staticmethod
    def get_tier2_optimization_workflow() -> WorkflowDefinition:
        """Advanced optimization workflow using Tier 2 components."""
        return WorkflowDefinition(
            workflow_type="tier2_optimization",
            name="ML Optimization Workflow",
            description="Advanced optimization using multi-armed bandit and pattern analysis",
            steps=[
                WorkflowStep(
                    step_id="bandit_optimization",
                    name="Multi-Armed Bandit Optimization",
                    component_name="multi_armed_bandit",
                    parameters={"n_arms": 10, "exploration_rate": 0.1},
                    timeout=300
                ),
                WorkflowStep(
                    step_id="pattern_analysis",
                    name="Apriori Pattern Analysis", 
                    component_name="apriori_analyzer",
                    parameters={"min_support": 0.1, "min_confidence": 0.5},
                    dependencies=["bandit_optimization"],
                    timeout=600
                ),
                WorkflowStep(
                    step_id="insight_generation",
                    name="Generate Insights",
                    component_name="insight_engine", 
                    parameters={"max_insights": 50},
                    dependencies=["pattern_analysis"],
                    timeout=300
                )
            ],
            global_timeout=1800
        )
    
    @staticmethod
    def get_end_to_end_workflow() -> WorkflowDefinition:
        """Complete end-to-end ML workflow across multiple tiers."""
        return WorkflowDefinition(
            workflow_type="end_to_end_ml",
            name="End-to-End ML Pipeline",
            description="Complete ML pipeline from data loading to deployment",
            steps=[
                # Data preparation (Tier 1)
                WorkflowStep(
                    step_id="load_data",
                    name="Load Training Data",
                    component_name="training_data_loader",
                    parameters={"batch_size": 1000},
                    timeout=300
                ),
                # Model training (Tier 1)
                WorkflowStep(
                    step_id="train_model",
                    name="Train ML Model",
                    component_name="ml_integration",
                    parameters={"epochs": 20},
                    dependencies=["load_data"],
                    timeout=1800
                ),
                # Optimization (Tier 2)
                WorkflowStep(
                    step_id="optimize_model",
                    name="Optimize Model Parameters",
                    component_name="rule_optimizer",
                    parameters={"iterations": 200},
                    dependencies=["train_model"],
                    timeout=600
                ),
                # Pattern discovery (Tier 2)
                WorkflowStep(
                    step_id="discover_patterns",
                    name="Discover Patterns",
                    component_name="advanced_pattern_discovery",
                    parameters={"max_patterns": 100},
                    dependencies=["train_model"],
                    timeout=400
                ),
                # Model registration (Tier 1)
                WorkflowStep(
                    step_id="register_model",
                    name="Register Model",
                    component_name="production_registry",
                    parameters={"version": "auto"},
                    dependencies=["optimize_model", "discover_patterns"],
                    timeout=120
                )
            ],
            global_timeout=7200,
            parallel_execution=True
        )
    
    @staticmethod
    def get_training_workflow() -> WorkflowDefinition:
        """Standard training workflow."""
        return WorkflowDefinition(
            workflow_type="training",
            name="Standard Training Workflow",
            description="Standard ML training workflow",
            steps=[
                WorkflowStep(
                    step_id="load_data",
                    name="Load Training Data",
                    component_name="training_data_loader",
                    parameters={"batch_size": 1000},
                    timeout=300
                ),
                WorkflowStep(
                    step_id="train_model",
                    name="Train ML Model",
                    component_name="ml_integration",
                    parameters={"epochs": 10},
                    dependencies=["load_data"],
                    timeout=1800
                ),
                WorkflowStep(
                    step_id="optimize_rules",
                    name="Optimize Rules",
                    component_name="rule_optimizer",
                    parameters={"iterations": 100},
                    dependencies=["train_model"],
                    timeout=600
                )
            ],
            global_timeout=3600
        )
    
    @staticmethod
    def get_all_workflow_templates() -> List[WorkflowDefinition]:
        """Get all predefined workflow templates."""
        return [
            WorkflowTemplates.get_training_workflow(),
            WorkflowTemplates.get_tier1_training_workflow(),
            WorkflowTemplates.get_tier2_optimization_workflow(), 
            WorkflowTemplates.get_end_to_end_workflow()
        ]