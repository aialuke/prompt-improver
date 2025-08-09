"""
Workflow templates for ML Pipeline orchestration.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List
from ..core.workflow_types import WorkflowDefinition, WorkflowStep

@dataclass
class WorkflowTemplates:
    """Collection of predefined workflow templates."""

    @staticmethod
    def get_tier1_training_workflow() -> WorkflowDefinition:
        """Core ML training workflow using Tier 1 components."""
        return WorkflowDefinition(workflow_type='tier1_training', name='Core ML Training Workflow', description='Basic training workflow using core ML components', steps=[WorkflowStep(step_id='load_data', name='Load Training Data', component_name='training_data_loader', parameters={'batch_size': 1000}, timeout=300), WorkflowStep(step_id='train_model', name='Train ML Model', component_name='ml_integration', parameters={'epochs': 10}, dependencies=['load_data'], timeout=1800), WorkflowStep(step_id='optimize_rules', name='Optimize Rules', component_name='rule_optimizer', parameters={'iterations': 100}, dependencies=['train_model'], timeout=600)], global_timeout=3600)

    @staticmethod
    def get_tier2_optimization_workflow() -> WorkflowDefinition:
        """Advanced optimization workflow using Tier 2 components."""
        return WorkflowDefinition(workflow_type='tier2_optimization', name='ML Optimization Workflow', description='Advanced optimization using multi-armed bandit and pattern analysis', steps=[WorkflowStep(step_id='bandit_optimization', name='Multi-Armed Bandit Optimization', component_name='multi_armed_bandit', parameters={'n_arms': 10, 'exploration_rate': 0.1}, timeout=300), WorkflowStep(step_id='pattern_analysis', name='Apriori Pattern Analysis', component_name='apriori_analyzer', parameters={'min_support': 0.1, 'min_confidence': 0.5}, dependencies=['bandit_optimization'], timeout=600), WorkflowStep(step_id='insight_generation', name='Generate Insights', component_name='insight_engine', parameters={'max_insights': 50}, dependencies=['pattern_analysis'], timeout=300)], global_timeout=1800)

    @staticmethod
    def get_end_to_end_workflow() -> WorkflowDefinition:
        """Complete end-to-end ML workflow across multiple tiers."""
        return WorkflowDefinition(workflow_type='end_to_end_ml', name='End-to-End ML Pipeline', description='Complete ML pipeline from data loading to deployment', steps=[WorkflowStep(step_id='load_data', name='Load Training Data', component_name='training_data_loader', parameters={'batch_size': 1000}, timeout=300), WorkflowStep(step_id='train_model', name='Train ML Model', component_name='ml_integration', parameters={'epochs': 20}, dependencies=['load_data'], timeout=1800), WorkflowStep(step_id='optimize_model', name='Optimize Model Parameters', component_name='rule_optimizer', parameters={'iterations': 200}, dependencies=['train_model'], timeout=600), WorkflowStep(step_id='discover_patterns', name='Discover Patterns', component_name='advanced_pattern_discovery', parameters={'max_patterns': 100}, dependencies=['train_model'], timeout=400), WorkflowStep(step_id='register_model', name='Register Model', component_name='production_registry', parameters={'version': 'auto'}, dependencies=['optimize_model', 'discover_patterns'], timeout=120)], global_timeout=7200, parallel_execution=True)

    @staticmethod
    def get_training_workflow() -> WorkflowDefinition:
        """Standard training workflow."""
        return WorkflowDefinition(workflow_type='training', name='Standard Training Workflow', description='Standard ML training workflow', steps=[WorkflowStep(step_id='load_data', name='Load Training Data', component_name='training_data_loader', parameters={'batch_size': 1000}, timeout=300), WorkflowStep(step_id='train_model', name='Train ML Model', component_name='ml_integration', parameters={'epochs': 10}, dependencies=['load_data'], timeout=1800), WorkflowStep(step_id='optimize_rules', name='Optimize Rules', component_name='rule_optimizer', parameters={'iterations': 100}, dependencies=['train_model'], timeout=600)], global_timeout=3600)

    @staticmethod
    def get_continuous_training_workflow() -> WorkflowDefinition:
        """
        Continuous adaptive training workflow with performance gap analysis.

        Implements 2025 best practices for self-improving ML training loops:
        - Performance gap analysis with correlation-driven stopping
        - Targeted synthetic data generation
        - Incremental model training
        - Intelligent rule optimization
        - Automated validation and session management
        """
        return WorkflowDefinition(workflow_type='continuous_training', name='Continuous Adaptive Training', description='Self-improving training loop with performance gap analysis and intelligent stopping', steps=[WorkflowStep(step_id='assess_performance', name='Assess Current Performance', component_name='performance_analyzer', parameters={'baseline_required': True, 'metrics': ['effectiveness', 'consistency', 'coverage'], 'window_size': 10}, timeout=300), WorkflowStep(step_id='analyze_gaps', name='Analyze Performance Gaps', component_name='performance_gap_analyzer', parameters={'improvement_threshold': 0.02, 'correlation_threshold': 0.95, 'plateau_detection': True, 'confidence_interval': 0.95}, dependencies=['assess_performance'], timeout=300), WorkflowStep(step_id='evaluate_stopping_criteria', name='Evaluate Stopping Criteria', component_name='stopping_criteria_evaluator', parameters={'correlation_driven': True, 'plateau_threshold': 0.05, 'min_improvement': 0.02}, dependencies=['analyze_gaps'], timeout=120), WorkflowStep(step_id='generate_targeted_data', name='Generate Targeted Synthetic Data', component_name='synthetic_data_orchestrator', parameters={'target_gaps': True, 'batch_size': 200, 'quality_threshold': 0.8, 'diversity_factor': 0.7}, dependencies=['evaluate_stopping_criteria'], timeout=600), WorkflowStep(step_id='incremental_training', name='Incremental Model Training', component_name='ml_integration', parameters={'incremental': True, 'epochs': 5, 'learning_rate': 0.001, 'batch_size': 32, 'validation_split': 0.2}, dependencies=['generate_targeted_data'], timeout=900), WorkflowStep(step_id='optimize_rules', name='Optimize Rules with Gaussian Process', component_name='rule_optimizer', parameters={'method': 'gaussian_process', 'acquisition_function': 'expected_improvement', 'n_iterations': 50, 'parallel_jobs': 2}, dependencies=['incremental_training'], timeout=600), WorkflowStep(step_id='validate_improvement', name='Validate Performance Improvement', component_name='performance_validator', parameters={'validation_threshold': 0.02, 'statistical_significance': 0.05, 'cross_validation': True, 'holdout_test': True}, dependencies=['optimize_rules'], timeout=300), WorkflowStep(step_id='update_session', name='Update Training Session', component_name='session_manager', parameters={'save_progress': True, 'update_metrics': True, 'checkpoint_model': True, 'log_iteration': True}, dependencies=['validate_improvement'], timeout=120)], global_timeout=3600, max_iterations=None, continuous=True, retry_policy={'max_retries': 3, 'backoff_factor': 2.0, 'retry_on_failure': ['generate_targeted_data', 'incremental_training']}, metadata={'workflow_version': '2025.1', 'best_practices': 'correlation_driven_stopping', 'optimization_method': 'gaussian_process', 'data_generation': 'gap_targeted'})

    @staticmethod
    def get_all_workflow_templates() -> List[WorkflowDefinition]:
        """Get all predefined workflow templates."""
        return [WorkflowTemplates.get_training_workflow(), WorkflowTemplates.get_tier1_training_workflow(), WorkflowTemplates.get_tier2_optimization_workflow(), WorkflowTemplates.get_end_to_end_workflow(), WorkflowTemplates.get_continuous_training_workflow()]
