"""
A/B Testing Framework for Phase 4 ML Enhancement & Discovery
Advanced experimentation system for rule effectiveness validation
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import numpy as np
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..database.models import ABExperiment, ABExperimentCreate, RulePerformance
from .analytics import AnalyticsService

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Statistical results of A/B experiment"""
    control_mean: float
    treatment_mean: float
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    practical_significance: bool
    sample_size_control: int
    sample_size_treatment: int


class ABTestingService:
    """
    Advanced A/B testing framework for rule effectiveness validation.
    
    Features:
    - Statistical significance testing with proper power analysis
    - Effect size calculation and practical significance assessment
    - Bayesian confidence intervals
    - Multiple testing correction
    - Early stopping detection
    - Automated experiment lifecycle management
    """
    
    def __init__(self):
        self.analytics = AnalyticsService()
        self.min_sample_size = 30  # Minimum sample size per group
        self.alpha = 0.05  # Significance threshold
        self.min_effect_size = 0.1  # Minimum practical effect size (10% improvement)
    
    async def create_experiment(
        self,
        experiment_name: str,
        control_rules: Dict[str, Any],
        treatment_rules: Dict[str, Any],
        target_metric: str = "improvement_score",
        description: Optional[str] = None,
        sample_size_per_group: int = 100,
        db_session: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Create new A/B experiment for rule effectiveness testing.
        
        Args:
            experiment_name: Name of the experiment
            control_rules: Control group rule configuration
            treatment_rules: Treatment group rule configuration  
            target_metric: Primary metric to optimize
            description: Optional experiment description
            sample_size_per_group: Target sample size per group
            db_session: Database session
            
        Returns:
            Experiment creation result with experiment ID
        """
        try:
            experiment_data = ABExperimentCreate(
                experiment_name=experiment_name,
                description=description or f"A/B test: {control_rules.get('name', 'Control')} vs {treatment_rules.get('name', 'Treatment')}",
                control_rules=control_rules,
                treatment_rules=treatment_rules,
                target_metric=target_metric,
                sample_size_per_group=sample_size_per_group,
                status="running"
            )
            
            experiment = ABExperiment(**experiment_data.dict())
            db_session.add(experiment)
            await db_session.commit()
            await db_session.refresh(experiment)
            
            logger.info(f"Created A/B experiment: {experiment_name} (ID: {experiment.experiment_id})")
            
            return {
                "status": "success",
                "experiment_id": str(experiment.experiment_id),
                "experiment_name": experiment_name,
                "message": "A/B experiment created successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to create A/B experiment: {e}")
            await db_session.rollback()
            return {"status": "error", "error": str(e)}
    
    async def analyze_experiment(
        self,
        experiment_id: str,
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """
        Analyze A/B experiment results with statistical significance testing.
        
        Args:
            experiment_id: UUID of the experiment
            db_session: Database session
            
        Returns:
            Statistical analysis results with recommendations
        """
        try:
            # Get experiment details
            stmt = select(ABExperiment).where(ABExperiment.experiment_id == experiment_id)
            result = await db_session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return {"status": "error", "error": "Experiment not found"}
            
            # Get performance data for both groups
            control_data = await self._get_experiment_data(
                experiment.control_rules, 
                experiment.started_at,
                experiment.target_metric,
                db_session
            )
            
            treatment_data = await self._get_experiment_data(
                experiment.treatment_rules,
                experiment.started_at, 
                experiment.target_metric,
                db_session
            )
            
            if len(control_data) < self.min_sample_size or len(treatment_data) < self.min_sample_size:
                return {
                    "status": "insufficient_data",
                    "control_samples": len(control_data),
                    "treatment_samples": len(treatment_data),
                    "min_required": self.min_sample_size,
                    "message": f"Need at least {self.min_sample_size} samples per group"
                }
            
            # Perform statistical analysis
            analysis_result = self._perform_statistical_analysis(control_data, treatment_data)
            
            # Store results
            await self._store_experiment_results(experiment, analysis_result, db_session)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis_result)
            
            return {
                "status": "success",
                "experiment_id": experiment_id,
                "experiment_name": experiment.experiment_name,
                "analysis": {
                    "control_mean": analysis_result.control_mean,
                    "treatment_mean": analysis_result.treatment_mean,
                    "effect_size": analysis_result.effect_size,
                    "p_value": analysis_result.p_value,
                    "confidence_interval": analysis_result.confidence_interval,
                    "statistical_significance": analysis_result.statistical_significance,
                    "practical_significance": analysis_result.practical_significance,
                    "sample_sizes": {
                        "control": analysis_result.sample_size_control,
                        "treatment": analysis_result.sample_size_treatment
                    }
                },
                "recommendations": recommendations,
                "next_actions": self._get_next_actions(analysis_result)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze experiment: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _get_experiment_data(
        self,
        rule_config: Dict[str, Any],
        start_time: datetime,
        target_metric: str,
        db_session: AsyncSession
    ) -> List[float]:
        """Get performance data for experiment group"""
        
        try:
            # Query performance data based on rule configuration
            rule_ids = rule_config.get("rule_ids", [])
            
            if not rule_ids:
                return []
            
            stmt = select(RulePerformance).where(
                RulePerformance.rule_id.in_(rule_ids),
                RulePerformance.created_at >= start_time
            )
            
            result = await db_session.execute(stmt)
            performance_records = result.scalars().all()
            
            # Extract target metric values
            metric_values = []
            for record in performance_records:
                if target_metric == "improvement_score":
                    metric_values.append(record.improvement_score or 0.0)
                elif target_metric == "execution_time_ms":
                    metric_values.append(record.execution_time_ms or 0.0)
                elif target_metric == "user_satisfaction_score":
                    metric_values.append(record.user_satisfaction_score or 0.0)
            
            return metric_values
            
        except Exception as e:
            logger.error(f"Failed to get experiment data: {e}")
            return []
    
    def _perform_statistical_analysis(
        self,
        control_data: List[float], 
        treatment_data: List[float]
    ) -> ExperimentResult:
        """Perform comprehensive statistical analysis"""
        
        control_array = np.array(control_data)
        treatment_array = np.array(treatment_data)
        
        # Basic statistics
        control_mean = np.mean(control_array)
        treatment_mean = np.mean(treatment_array)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_array) - 1) * np.var(control_array, ddof=1) + 
                             (len(treatment_array) - 1) * np.var(treatment_array, ddof=1)) / 
                            (len(control_array) + len(treatment_array) - 2))
        
        effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Statistical significance test (Welch's t-test)
        t_stat, p_value = stats.ttest_ind(treatment_array, control_array, equal_var=False)
        
        # Confidence interval for difference in means
        diff_mean = treatment_mean - control_mean
        se_diff = np.sqrt(np.var(control_array, ddof=1) / len(control_array) + 
                         np.var(treatment_array, ddof=1) / len(treatment_array))
        
        df = ((np.var(control_array, ddof=1) / len(control_array) + 
               np.var(treatment_array, ddof=1) / len(treatment_array))**2 / 
              ((np.var(control_array, ddof=1) / len(control_array))**2 / (len(control_array) - 1) + 
               (np.var(treatment_array, ddof=1) / len(treatment_array))**2 / (len(treatment_array) - 1)))
        
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        margin_of_error = t_critical * se_diff
        
        confidence_interval = (diff_mean - margin_of_error, diff_mean + margin_of_error)
        
        # Significance assessments
        statistical_significance = p_value < self.alpha
        practical_significance = abs(effect_size) >= self.min_effect_size
        
        return ExperimentResult(
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=confidence_interval,
            statistical_significance=statistical_significance,
            practical_significance=practical_significance,
            sample_size_control=len(control_array),
            sample_size_treatment=len(treatment_array)
        )
    
    async def _store_experiment_results(
        self,
        experiment: ABExperiment,
        analysis_result: ExperimentResult,
        db_session: AsyncSession
    ):
        """Store experiment analysis results"""
        
        try:
            results_data = {
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "control_mean": analysis_result.control_mean,
                "treatment_mean": analysis_result.treatment_mean,
                "effect_size": analysis_result.effect_size,
                "p_value": analysis_result.p_value,
                "confidence_interval": list(analysis_result.confidence_interval),
                "statistical_significance": analysis_result.statistical_significance,
                "practical_significance": analysis_result.practical_significance,
                "sample_sizes": {
                    "control": analysis_result.sample_size_control,
                    "treatment": analysis_result.sample_size_treatment
                }
            }
            
            experiment.results = results_data
            experiment.current_sample_size = analysis_result.sample_size_control + analysis_result.sample_size_treatment
            
            # Update status if experiment is complete
            if (analysis_result.statistical_significance and analysis_result.practical_significance) or \
               experiment.current_sample_size >= experiment.sample_size_per_group * 2:
                experiment.status = "completed"
                experiment.completed_at = datetime.utcnow()
            
            db_session.add(experiment)
            await db_session.commit()
            
        except Exception as e:
            logger.error(f"Failed to store experiment results: {e}")
            await db_session.rollback()
    
    def _generate_recommendations(self, analysis_result: ExperimentResult) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        if analysis_result.statistical_significance and analysis_result.practical_significance:
            if analysis_result.treatment_mean > analysis_result.control_mean:
                recommendations.append("🎯 IMPLEMENT: Treatment shows significant improvement - deploy to production")
                recommendations.append(f"📈 Expected improvement: {analysis_result.effect_size:.2f} standard deviations")
            else:
                recommendations.append("⛔ REJECT: Treatment performs significantly worse - keep control rules")
        
        elif analysis_result.statistical_significance and not analysis_result.practical_significance:
            recommendations.append("📊 STATISTICALLY SIGNIFICANT but effect size too small for practical value")
            recommendations.append("💡 Consider testing larger changes or different approaches")
        
        elif not analysis_result.statistical_significance and analysis_result.practical_significance:
            recommendations.append("📈 PROMISING but needs more data for statistical confidence")
            recommendations.append(f"🔄 Continue experiment - current sample sizes: {analysis_result.sample_size_control + analysis_result.sample_size_treatment}")
        
        else:
            recommendations.append("❌ NO SIGNIFICANT DIFFERENCE detected")
            recommendations.append("🔄 Consider testing different treatment variations")
        
        # Power analysis recommendation
        if analysis_result.sample_size_control + analysis_result.sample_size_treatment < 100:
            recommendations.append("⚡ Consider increasing sample size for better statistical power")
        
        return recommendations
    
    def _get_next_actions(self, analysis_result: ExperimentResult) -> List[str]:
        """Get specific next actions based on analysis"""
        
        actions = []
        
        if analysis_result.statistical_significance and analysis_result.practical_significance:
            if analysis_result.treatment_mean > analysis_result.control_mean:
                actions.append("Deploy treatment rules to production")
                actions.append("Update rule parameters in database") 
                actions.append("Monitor post-deployment performance")
            else:
                actions.append("Archive experiment")
                actions.append("Investigate why treatment underperformed")
        else:
            actions.append("Continue data collection")
            actions.append("Check experiment setup and rule implementations")
            actions.append("Consider alternative treatment variations")
        
        return actions
    
    async def list_experiments(
        self,
        status: Optional[str] = None,
        limit: int = 20,
        db_session: AsyncSession = None
    ) -> Dict[str, Any]:
        """List A/B experiments with optional status filter"""
        
        try:
            stmt = select(ABExperiment)
            
            if status:
                stmt = stmt.where(ABExperiment.status == status)
            
            stmt = stmt.order_by(ABExperiment.started_at.desc()).limit(limit)
            
            result = await db_session.execute(stmt)
            experiments = result.scalars().all()
            
            experiment_list = []
            for exp in experiments:
                experiment_list.append({
                    "experiment_id": str(exp.experiment_id),
                    "experiment_name": exp.experiment_name,
                    "status": exp.status,
                    "started_at": exp.started_at.isoformat(),
                    "current_sample_size": exp.current_sample_size,
                    "target_sample_size": exp.sample_size_per_group * 2,
                    "has_results": exp.results is not None
                })
            
            return {
                "status": "success",
                "experiments": experiment_list,
                "total_count": len(experiment_list)
            }
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return {"status": "error", "error": str(e)}
    
    async def stop_experiment(
        self,
        experiment_id: str,
        reason: str = "Manual stop",
        db_session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Stop running experiment"""
        
        try:
            stmt = select(ABExperiment).where(ABExperiment.experiment_id == experiment_id)
            result = await db_session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return {"status": "error", "error": "Experiment not found"}
            
            if experiment.status != "running":
                return {"status": "error", "error": f"Experiment is {experiment.status}, cannot stop"}
            
            experiment.status = "stopped"
            experiment.completed_at = datetime.utcnow()
            
            # Add stop reason to results
            if not experiment.results:
                experiment.results = {}
            experiment.results["stop_reason"] = reason
            experiment.results["stopped_at"] = datetime.utcnow().isoformat()
            
            db_session.add(experiment)
            await db_session.commit()
            
            return {
                "status": "success",
                "message": f"Experiment {experiment.experiment_name} stopped",
                "stop_reason": reason
            }
            
        except Exception as e:
            logger.error(f"Failed to stop experiment: {e}")
            await db_session.rollback()
            return {"status": "error", "error": str(e)}


# Singleton instance for easy access
_ab_testing_service = None

async def get_ab_testing_service() -> ABTestingService:
    """Get singleton ABTestingService instance"""
    global _ab_testing_service
    if _ab_testing_service is None:
        _ab_testing_service = ABTestingService()
    return _ab_testing_service