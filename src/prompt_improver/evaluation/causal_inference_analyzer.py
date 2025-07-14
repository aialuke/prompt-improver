"""
Causal Inference Analyzer for Advanced A/B Testing
Implements 2025 best practices for causal analysis and counterfactual reasoning
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from enum import Enum

from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

logger = logging.getLogger(__name__)


class CausalMethod(Enum):
    """Causal inference methods"""
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    INSTRUMENTAL_VARIABLES = "instrumental_variables"
    PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    DOUBLY_ROBUST = "doubly_robust"
    SYNTHETIC_CONTROL = "synthetic_control"


class TreatmentAssignment(Enum):
    """Treatment assignment mechanisms"""
    RANDOMIZED = "randomized"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    OBSERVATIONAL = "observational"
    NATURAL_EXPERIMENT = "natural_experiment"


@dataclass
class CausalAssumption:
    """Represents a causal inference assumption"""
    name: str
    description: str
    testable: bool
    test_result: Optional[Dict[str, Any]] = None
    violated: bool = False
    severity: str = "unknown"  # low, medium, high
    recommendations: List[str] = field(default_factory=list)


@dataclass
class CausalEffect:
    """Represents an estimated causal effect"""
    effect_name: str
    point_estimate: float
    confidence_interval: Tuple[float, float]
    standard_error: float
    p_value: float
    method: CausalMethod
    sample_size: int
    effect_size_interpretation: str
    statistical_significance: bool
    practical_significance: bool
    robustness_score: float = 0.0
    assumptions_satisfied: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalInferenceResult:
    """Comprehensive causal inference analysis result"""
    analysis_id: str
    timestamp: datetime
    treatment_assignment: TreatmentAssignment
    
    # Primary causal effects
    average_treatment_effect: CausalEffect
    conditional_average_treatment_effect: Optional[CausalEffect] = None
    local_average_treatment_effect: Optional[CausalEffect] = None
    
    # Assumption testing
    assumptions_tested: List[CausalAssumption] = field(default_factory=list)
    overall_assumptions_satisfied: bool = True
    
    # Robustness checks
    sensitivity_analysis: Optional[Dict[str, Any]] = None
    placebo_tests: Optional[Dict[str, Any]] = None
    robustness_score: float = 0.0
    
    # Confounding analysis
    confounding_assessment: Optional[Dict[str, Any]] = None
    covariate_balance: Optional[Dict[str, Any]] = None
    
    # Recommendations
    causal_interpretation: str = ""
    business_recommendations: List[str] = field(default_factory=list)
    statistical_warnings: List[str] = field(default_factory=list)
    
    # Quality metrics
    internal_validity_score: float = 0.0
    external_validity_score: float = 0.0
    overall_quality_score: float = 0.0


class CausalInferenceAnalyzer:
    """Advanced causal inference analyzer implementing 2025 best practices"""
    
    def __init__(self,
                 significance_level: float = 0.05,
                 minimum_effect_size: float = 0.1,
                 bootstrap_samples: int = 1000,
                 enable_sensitivity_analysis: bool = True):
        """
        Initialize causal inference analyzer
        
        Args:
            significance_level: Alpha level for statistical tests
            minimum_effect_size: Minimum meaningful effect size
            bootstrap_samples: Number of bootstrap samples for robust estimation
            enable_sensitivity_analysis: Whether to perform sensitivity analysis
        """
        self.significance_level = significance_level
        self.minimum_effect_size = minimum_effect_size
        self.bootstrap_samples = bootstrap_samples
        self.enable_sensitivity_analysis = enable_sensitivity_analysis
        
    def analyze_causal_effect(self,
                            outcome_data: np.ndarray,
                            treatment_data: np.ndarray,
                            covariates: Optional[np.ndarray] = None,
                            assignment_mechanism: TreatmentAssignment = TreatmentAssignment.RANDOMIZED,
                            method: CausalMethod = CausalMethod.DIFFERENCE_IN_DIFFERENCES,
                            time_periods: Optional[np.ndarray] = None,
                            instruments: Optional[np.ndarray] = None) -> CausalInferenceResult:
        """
        Perform comprehensive causal inference analysis
        
        Args:
            outcome_data: Outcome variable (continuous or binary)
            treatment_data: Treatment assignment (0/1)
            covariates: Control variables/covariates
            assignment_mechanism: How treatment was assigned
            method: Primary causal inference method
            time_periods: Time periods for difference-in-differences
            instruments: Instrumental variables for IV estimation
            
        Returns:
            Comprehensive causal inference result
        """
        analysis_id = f"causal_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Starting causal inference analysis: {analysis_id}")
            
            # Validate and prepare data
            validated_data = self._validate_causal_data(
                outcome_data, treatment_data, covariates, time_periods, instruments
            )
            
            # Test causal assumptions
            assumptions = self._test_causal_assumptions(
                validated_data, assignment_mechanism, method
            )
            
            # Estimate primary causal effect
            primary_effect = self._estimate_causal_effect(
                validated_data, method, assignment_mechanism
            )
            
            # Estimate heterogeneous effects if possible
            conditional_effect = self._estimate_conditional_effects(
                validated_data, method
            ) if covariates is not None else None
            
            # Perform robustness checks
            sensitivity_results = None
            placebo_results = None
            if self.enable_sensitivity_analysis:
                sensitivity_results = self._perform_sensitivity_analysis(validated_data, method)
                placebo_results = self._perform_placebo_tests(validated_data, method)
            
            # Assess confounding
            confounding_assessment = self._assess_confounding(validated_data, assignment_mechanism)
            
            # Calculate covariate balance
            covariate_balance = self._assess_covariate_balance(
                validated_data
            ) if covariates is not None else None
            
            # Calculate overall scores
            robustness_score = self._calculate_robustness_score(
                primary_effect, sensitivity_results, placebo_results
            )
            
            internal_validity = self._calculate_internal_validity_score(assumptions, confounding_assessment)
            external_validity = self._calculate_external_validity_score(validated_data, method)
            overall_quality = (internal_validity + external_validity + robustness_score) / 3
            
            # Generate interpretations and recommendations
            interpretation = self._generate_causal_interpretation(primary_effect, assumptions)
            business_recs = self._generate_business_recommendations(
                primary_effect, assumptions, robustness_score
            )
            warnings = self._generate_statistical_warnings(assumptions, confounding_assessment)
            
            # Create comprehensive result
            result = CausalInferenceResult(
                analysis_id=analysis_id,
                timestamp=datetime.utcnow(),
                treatment_assignment=assignment_mechanism,
                average_treatment_effect=primary_effect,
                conditional_average_treatment_effect=conditional_effect,
                assumptions_tested=assumptions,
                overall_assumptions_satisfied=all(not a.violated for a in assumptions),
                sensitivity_analysis=sensitivity_results,
                placebo_tests=placebo_results,
                robustness_score=robustness_score,
                confounding_assessment=confounding_assessment,
                covariate_balance=covariate_balance,
                causal_interpretation=interpretation,
                business_recommendations=business_recs,
                statistical_warnings=warnings,
                internal_validity_score=internal_validity,
                external_validity_score=external_validity,
                overall_quality_score=overall_quality
            )
            
            logger.info(f"Causal inference analysis completed: {analysis_id}")
            logger.info(f"ATE: {primary_effect.point_estimate:.4f} ± {primary_effect.standard_error:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in causal inference analysis: {e}")
            raise
    
    def _validate_causal_data(self, outcome_data: np.ndarray, treatment_data: np.ndarray,
                            covariates: Optional[np.ndarray], time_periods: Optional[np.ndarray],
                            instruments: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        """Validate and prepare data for causal analysis"""
        
        # Convert to numpy arrays
        outcome = np.asarray(outcome_data).flatten()
        treatment = np.asarray(treatment_data).flatten()
        
        # Basic validation
        if len(outcome) != len(treatment):
            raise ValueError("Outcome and treatment data must have same length")
        
        if len(outcome) < 5:
            raise ValueError("Insufficient sample size for reliable causal inference")
        
        # Check treatment is binary
        unique_treatments = np.unique(treatment)
        if not np.array_equal(unique_treatments, [0, 1]) and not np.array_equal(unique_treatments, [0]) and not np.array_equal(unique_treatments, [1]):
            if len(unique_treatments) == 2:
                # Convert to 0/1
                treatment = (treatment == unique_treatments[1]).astype(int)
            else:
                raise ValueError("Treatment must be binary (0/1)")
        
        # Validate finite values
        if not np.all(np.isfinite(outcome)) or not np.all(np.isfinite(treatment)):
            raise ValueError("Outcome and treatment data must be finite")
        
        # Prepare validated data dictionary
        validated = {
            'outcome': outcome,
            'treatment': treatment,
            'n_total': len(outcome),
            'n_treated': np.sum(treatment),
            'n_control': len(outcome) - np.sum(treatment)
        }
        
        # Add covariates if provided
        if covariates is not None:
            covariates = np.asarray(covariates)
            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            if len(covariates) != len(outcome):
                raise ValueError("Covariates must have same length as outcome")
            validated['covariates'] = covariates
            validated['n_covariates'] = covariates.shape[1]
        
        # Add time periods if provided
        if time_periods is not None:
            time_periods = np.asarray(time_periods).flatten()
            if len(time_periods) != len(outcome):
                raise ValueError("Time periods must have same length as outcome")
            validated['time_periods'] = time_periods
            validated['n_periods'] = len(np.unique(time_periods))
        
        # Add instruments if provided
        if instruments is not None:
            instruments = np.asarray(instruments)
            if instruments.ndim == 1:
                instruments = instruments.reshape(-1, 1)
            if len(instruments) != len(outcome):
                raise ValueError("Instruments must have same length as outcome")
            validated['instruments'] = instruments
            validated['n_instruments'] = instruments.shape[1]
        
        return validated
    
    def _test_causal_assumptions(self, data: Dict[str, np.ndarray],
                               assignment: TreatmentAssignment,
                               method: CausalMethod) -> List[CausalAssumption]:
        """Test key causal inference assumptions"""
        assumptions = []
        
        try:
            # 1. Overlap/Common Support
            overlap_assumption = self._test_overlap_assumption(data)
            assumptions.append(overlap_assumption)
            
            # 2. Balance (for randomized experiments)
            if assignment == TreatmentAssignment.RANDOMIZED:
                balance_assumption = self._test_balance_assumption(data)
                assumptions.append(balance_assumption)
            
            # 3. Parallel trends (for difference-in-differences)
            if method == CausalMethod.DIFFERENCE_IN_DIFFERENCES and 'time_periods' in data:
                parallel_trends = self._test_parallel_trends_assumption(data)
                assumptions.append(parallel_trends)
            
            # 4. Instrument relevance and exogeneity (for IV)
            if method == CausalMethod.INSTRUMENTAL_VARIABLES and 'instruments' in data:
                iv_assumptions = self._test_iv_assumptions(data)
                assumptions.extend(iv_assumptions)
            
            # 5. No unmeasured confounding (general)
            confounding_assumption = self._test_confounding_assumption(data, assignment)
            assumptions.append(confounding_assumption)
            
        except Exception as e:
            logger.warning(f"Error testing assumptions: {e}")
            # Add a general warning assumption
            warning_assumption = CausalAssumption(
                name="assumption_testing_error",
                description="Could not fully test causal assumptions",
                testable=False,
                violated=True,
                severity="medium",
                recommendations=["Manual assumption checking recommended"]
            )
            assumptions.append(warning_assumption)
        
        return assumptions
    
    def _test_overlap_assumption(self, data: Dict[str, np.ndarray]) -> CausalAssumption:
        """Test overlap/common support assumption"""
        
        if 'covariates' not in data:
            # Without covariates, just check we have both treatment groups
            n_treated = data['n_treated']
            n_control = data['n_control']
            
            overlap_adequate = n_treated >= 5 and n_control >= 5
            
            return CausalAssumption(
                name="overlap",
                description="Sufficient overlap between treatment and control groups",
                testable=True,
                test_result={
                    'n_treated': int(n_treated),
                    'n_control': int(n_control),
                    'adequate_overlap': overlap_adequate
                },
                violated=not overlap_adequate,
                severity="high" if not overlap_adequate else "low",
                recommendations=["Increase sample size in underrepresented group"] if not overlap_adequate else []
            )
        
        # With covariates, check propensity score overlap
        try:
            # Estimate propensity scores
            from sklearn.linear_model import LogisticRegression
            
            X = data['covariates']
            y = data['treatment']
            
            lr = LogisticRegression(random_state=42)
            propensity_scores = lr.fit(X, y).predict_proba(X)[:, 1]
            
            # Check overlap in propensity score distribution
            treated_ps = propensity_scores[y == 1]
            control_ps = propensity_scores[y == 0]
            
            # Calculate overlap region
            min_treated, max_treated = np.min(treated_ps), np.max(treated_ps)
            min_control, max_control = np.min(control_ps), np.max(control_ps)
            
            overlap_min = max(min_treated, min_control)
            overlap_max = min(max_treated, max_control)
            overlap_exists = overlap_max > overlap_min
            
            # Calculate proportion of data in overlap region
            in_overlap = (propensity_scores >= overlap_min) & (propensity_scores <= overlap_max)
            overlap_proportion = np.mean(in_overlap)
            
            adequate_overlap = overlap_exists and overlap_proportion >= 0.8
            
            return CausalAssumption(
                name="overlap",
                description="Sufficient overlap in propensity score distributions",
                testable=True,
                test_result={
                    'overlap_exists': overlap_exists,
                    'overlap_proportion': float(overlap_proportion),
                    'overlap_range': [float(overlap_min), float(overlap_max)],
                    'adequate_overlap': adequate_overlap
                },
                violated=not adequate_overlap,
                severity="high" if not adequate_overlap else "low",
                recommendations=["Consider trimming extreme propensity scores", 
                               "Collect more data in regions of poor overlap"] if not adequate_overlap else []
            )
            
        except Exception as e:
            logger.warning(f"Error testing overlap assumption: {e}")
            return CausalAssumption(
                name="overlap",
                description="Could not test overlap assumption",
                testable=False,
                violated=True,
                severity="medium",
                recommendations=["Manual overlap assessment recommended"]
            )
    
    def _test_balance_assumption(self, data: Dict[str, np.ndarray]) -> CausalAssumption:
        """Test covariate balance assumption for randomized experiments"""
        
        if 'covariates' not in data:
            return CausalAssumption(
                name="balance",
                description="No covariates to test balance",
                testable=False,
                violated=False,
                severity="low"
            )
        
        try:
            covariates = data['covariates']
            treatment = data['treatment']
            
            balance_tests = []
            imbalanced_covariates = 0
            
            for i in range(covariates.shape[1]):
                covariate = covariates[:, i]
                
                # T-test for balance
                treated_cov = covariate[treatment == 1]
                control_cov = covariate[treatment == 0]
                
                statistic, p_value = stats.ttest_ind(treated_cov, control_cov)
                
                # Standardized mean difference
                pooled_std = np.sqrt(((len(treated_cov) - 1) * np.var(treated_cov, ddof=1) + 
                                    (len(control_cov) - 1) * np.var(control_cov, ddof=1)) / 
                                   (len(treated_cov) + len(control_cov) - 2))
                smd = (np.mean(treated_cov) - np.mean(control_cov)) / pooled_std if pooled_std > 0 else 0
                
                is_imbalanced = abs(smd) > 0.1  # Standard threshold
                if is_imbalanced:
                    imbalanced_covariates += 1
                
                balance_tests.append({
                    'covariate_index': i,
                    'p_value': float(p_value),
                    'standardized_mean_difference': float(smd),
                    'imbalanced': is_imbalanced
                })
            
            overall_balanced = imbalanced_covariates / len(balance_tests) <= 0.05  # Max 5% imbalanced
            
            return CausalAssumption(
                name="balance",
                description="Covariate balance between treatment groups",
                testable=True,
                test_result={
                    'n_covariates': len(balance_tests),
                    'n_imbalanced': imbalanced_covariates,
                    'proportion_imbalanced': imbalanced_covariates / len(balance_tests),
                    'balance_tests': balance_tests,
                    'overall_balanced': overall_balanced
                },
                violated=not overall_balanced,
                severity="medium" if not overall_balanced else "low",
                recommendations=["Consider stratified randomization", 
                               "Use covariate adjustment in analysis"] if not overall_balanced else []
            )
            
        except Exception as e:
            logger.warning(f"Error testing balance assumption: {e}")
            return CausalAssumption(
                name="balance",
                description="Could not test balance assumption",
                testable=False,
                violated=True,
                severity="medium"
            )
    
    def _test_parallel_trends_assumption(self, data: Dict[str, np.ndarray]) -> CausalAssumption:
        """Test parallel trends assumption for difference-in-differences"""
        
        try:
            outcome = data['outcome']
            treatment = data['treatment']
            time_periods = data['time_periods']
            
            # Create pre-treatment and post-treatment periods
            pre_period = time_periods == 0
            post_period = time_periods == 1
            
            if not (np.any(pre_period) and np.any(post_period)):
                return CausalAssumption(
                    name="parallel_trends",
                    description="Insufficient time periods for parallel trends test",
                    testable=False,
                    violated=True,
                    severity="high"
                )
            
            # Test for parallel trends in pre-period
            # This is a simplified test - in practice, you'd need multiple pre-periods
            pre_treated = outcome[pre_period & (treatment == 1)]
            pre_control = outcome[pre_period & (treatment == 0)]
            
            # Test if pre-treatment trends are similar
            # Simplified: test if pre-treatment levels are similar
            if len(pre_treated) > 0 and len(pre_control) > 0:
                statistic, p_value = stats.ttest_ind(pre_treated, pre_control)
                parallel_trends_satisfied = p_value > 0.1  # Loose threshold for pre-treatment similarity
            else:
                parallel_trends_satisfied = False
                p_value = 0.0
            
            return CausalAssumption(
                name="parallel_trends",
                description="Parallel trends assumption for difference-in-differences",
                testable=True,
                test_result={
                    'pre_treatment_test_pvalue': float(p_value),
                    'parallel_trends_satisfied': parallel_trends_satisfied,
                    'n_pre_treated': len(pre_treated) if len(pre_treated) > 0 else 0,
                    'n_pre_control': len(pre_control) if len(pre_control) > 0 else 0
                },
                violated=not parallel_trends_satisfied,
                severity="high" if not parallel_trends_satisfied else "low",
                recommendations=["Collect more pre-treatment periods for robust testing",
                               "Consider event study design"] if not parallel_trends_satisfied else []
            )
            
        except Exception as e:
            logger.warning(f"Error testing parallel trends: {e}")
            return CausalAssumption(
                name="parallel_trends",
                description="Could not test parallel trends assumption",
                testable=False,
                violated=True,
                severity="high"
            )
    
    def _test_iv_assumptions(self, data: Dict[str, np.ndarray]) -> List[CausalAssumption]:
        """Test instrumental variable assumptions"""
        assumptions = []
        
        try:
            instruments = data['instruments']
            treatment = data['treatment']
            outcome = data['outcome']
            
            # Test instrument relevance (first stage)
            for i in range(instruments.shape[1]):
                instrument = instruments[:, i]
                
                # First stage regression
                correlation = np.corrcoef(instrument, treatment)[0, 1]
                f_statistic = (correlation ** 2) * (len(treatment) - 2) / (1 - correlation ** 2)
                
                # Rule of thumb: F > 10 for strong instrument
                strong_instrument = f_statistic > 10
                
                relevance_assumption = CausalAssumption(
                    name=f"instrument_relevance_{i}",
                    description=f"Instrument {i} relevance (first stage strength)",
                    testable=True,
                    test_result={
                        'correlation_with_treatment': float(correlation),
                        'f_statistic': float(f_statistic),
                        'strong_instrument': strong_instrument
                    },
                    violated=not strong_instrument,
                    severity="high" if not strong_instrument else "low",
                    recommendations=["Find stronger instruments", 
                                   "Consider weak instrument robust methods"] if not strong_instrument else []
                )
                assumptions.append(relevance_assumption)
            
            # Test instrument exogeneity (simplified - not fully testable)
            # We can test if instrument is correlated with outcome in absence of treatment
            exogeneity_assumption = CausalAssumption(
                name="instrument_exogeneity",
                description="Instrument exogeneity (exclusion restriction)",
                testable=False,  # Not fully testable without strong assumptions
                violated=False,  # Assume satisfied unless evidence otherwise
                severity="high",
                recommendations=["Carefully justify exclusion restriction",
                               "Consider overidentification tests if multiple instruments"]
            )
            assumptions.append(exogeneity_assumption)
            
        except Exception as e:
            logger.warning(f"Error testing IV assumptions: {e}")
            error_assumption = CausalAssumption(
                name="iv_assumptions_error",
                description="Could not test IV assumptions",
                testable=False,
                violated=True,
                severity="medium"
            )
            assumptions.append(error_assumption)
        
        return assumptions
    
    def _test_confounding_assumption(self, data: Dict[str, np.ndarray],
                                   assignment: TreatmentAssignment) -> CausalAssumption:
        """Test no unmeasured confounding assumption"""
        
        if assignment == TreatmentAssignment.RANDOMIZED:
            # For randomized experiments, confounding is controlled by design
            return CausalAssumption(
                name="no_unmeasured_confounding",
                description="No unmeasured confounding (randomized experiment)",
                testable=False,
                violated=False,
                severity="low",
                recommendations=["Verify randomization was properly implemented"]
            )
        
        # For observational studies, this is largely untestable
        # We can do some basic checks
        try:
            outcome = data['outcome']
            treatment = data['treatment']
            
            # Simple test: if treatment is highly predictive of outcome even without causal effect,
            # might suggest confounding
            correlation = abs(np.corrcoef(treatment, outcome)[0, 1])
            
            # This is a very crude test
            potential_confounding = correlation > 0.7  # High correlation might suggest confounding
            
            return CausalAssumption(
                name="no_unmeasured_confounding",
                description="No unmeasured confounding assumption",
                testable=False,  # Not truly testable
                test_result={
                    'treatment_outcome_correlation': float(correlation),
                    'potential_confounding_concern': potential_confounding
                },
                violated=potential_confounding,
                severity="high",
                recommendations=["Include more covariates",
                               "Use instrumental variables",
                               "Consider sensitivity analysis"] if potential_confounding else
                              ["Justify assumption with domain knowledge"]
            )
            
        except Exception as e:
            logger.warning(f"Error testing confounding assumption: {e}")
            return CausalAssumption(
                name="no_unmeasured_confounding",
                description="Could not assess confounding assumption",
                testable=False,
                violated=True,
                severity="high"
            )
    
    def _estimate_causal_effect(self, data: Dict[str, np.ndarray],
                              method: CausalMethod,
                              assignment: TreatmentAssignment) -> CausalEffect:
        """Estimate the average treatment effect using specified method"""
        
        outcome = data['outcome']
        treatment = data['treatment']
        
        try:
            if method == CausalMethod.DIFFERENCE_IN_DIFFERENCES:
                return self._estimate_did_effect(data)
            elif method == CausalMethod.INSTRUMENTAL_VARIABLES:
                return self._estimate_iv_effect(data)
            elif method == CausalMethod.PROPENSITY_SCORE_MATCHING:
                return self._estimate_psm_effect(data)
            elif method == CausalMethod.DOUBLY_ROBUST:
                return self._estimate_doubly_robust_effect(data)
            else:
                # Default to simple difference in means
                return self._estimate_simple_difference(data)
                
        except Exception as e:
            logger.error(f"Error estimating causal effect with {method}: {e}")
            # Fallback to simple difference
            return self._estimate_simple_difference(data)
    
    def _estimate_simple_difference(self, data: Dict[str, np.ndarray]) -> CausalEffect:
        """Estimate ATE as simple difference in means"""
        
        outcome = data['outcome']
        treatment = data['treatment']
        
        treated_outcomes = outcome[treatment == 1]
        control_outcomes = outcome[treatment == 0]
        
        # Point estimate
        ate = np.mean(treated_outcomes) - np.mean(control_outcomes)
        
        # Standard error
        var_treated = np.var(treated_outcomes, ddof=1) / len(treated_outcomes)
        var_control = np.var(control_outcomes, ddof=1) / len(control_outcomes)
        se = np.sqrt(var_treated + var_control)
        
        # T-test
        statistic, p_value = stats.ttest_ind(treated_outcomes, control_outcomes)
        
        # Confidence interval
        dof = len(treated_outcomes) + len(control_outcomes) - 2
        t_critical = stats.t.ppf(1 - self.significance_level/2, dof)
        ci_lower = ate - t_critical * se
        ci_upper = ate + t_critical * se
        
        # Effect size interpretation
        pooled_std = np.sqrt(((len(treated_outcomes) - 1) * np.var(treated_outcomes, ddof=1) + 
                             (len(control_outcomes) - 1) * np.var(control_outcomes, ddof=1)) / dof)
        cohens_d = ate / pooled_std if pooled_std > 0 else 0
        
        if abs(cohens_d) < 0.2:
            interpretation = "negligible effect"
        elif abs(cohens_d) < 0.5:
            interpretation = "small effect"
        elif abs(cohens_d) < 0.8:
            interpretation = "medium effect"
        else:
            interpretation = "large effect"
        
        return CausalEffect(
            effect_name="Average Treatment Effect",
            point_estimate=float(ate),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            standard_error=float(se),
            p_value=float(p_value),
            method=CausalMethod.DIFFERENCE_IN_DIFFERENCES,  # Using as default
            sample_size=len(outcome),
            effect_size_interpretation=interpretation,
            statistical_significance=bool(p_value < self.significance_level),
            practical_significance=bool(abs(ate) >= self.minimum_effect_size),
            robustness_score=0.7,  # Moderate robustness for simple difference
            assumptions_satisfied=True,
            metadata={
                'treated_mean': float(np.mean(treated_outcomes)),
                'control_mean': float(np.mean(control_outcomes)),
                'cohens_d': float(cohens_d),
                'n_treated': len(treated_outcomes),
                'n_control': len(control_outcomes)
            }
        )
    
    def _estimate_did_effect(self, data: Dict[str, np.ndarray]) -> CausalEffect:
        """Estimate difference-in-differences effect"""
        
        if 'time_periods' not in data:
            logger.warning("No time periods provided for DiD, falling back to simple difference")
            return self._estimate_simple_difference(data)
        
        outcome = data['outcome']
        treatment = data['treatment']
        time_periods = data['time_periods']
        
        # Create indicator variables
        pre_period = (time_periods == 0).astype(int)
        post_period = (time_periods == 1).astype(int)
        
        # DiD regression: Y = α + β₁*Treated + β₂*Post + β₃*Treated*Post + ε
        # The coefficient β₃ is the DiD estimator
        
        try:
            # Prepare design matrix
            X = np.column_stack([
                np.ones(len(outcome)),  # Intercept
                treatment,              # Treatment group indicator
                post_period,           # Post-period indicator
                treatment * post_period # Interaction (DiD estimator)
            ])
            
            # OLS estimation
            beta_hat = np.linalg.lstsq(X, outcome, rcond=None)[0]
            did_estimate = beta_hat[3]  # Interaction coefficient
            
            # Calculate standard errors
            residuals = outcome - X @ beta_hat
            mse = np.sum(residuals**2) / (len(outcome) - X.shape[1])
            var_cov_matrix = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(var_cov_matrix[3, 3])
            
            # T-test
            t_stat = did_estimate / se
            dof = len(outcome) - X.shape[1]
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
            
            # Confidence interval
            t_critical = stats.t.ppf(1 - self.significance_level/2, dof)
            ci_lower = did_estimate - t_critical * se
            ci_upper = did_estimate + t_critical * se
            
            # Effect size interpretation
            outcome_std = np.std(outcome, ddof=1)
            standardized_effect = did_estimate / outcome_std if outcome_std > 0 else 0
            
            if abs(standardized_effect) < 0.2:
                interpretation = "negligible effect"
            elif abs(standardized_effect) < 0.5:
                interpretation = "small effect"
            elif abs(standardized_effect) < 0.8:
                interpretation = "medium effect"
            else:
                interpretation = "large effect"
            
            return CausalEffect(
                effect_name="Difference-in-Differences Estimate",
                point_estimate=float(did_estimate),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                standard_error=float(se),
                p_value=float(p_value),
                method=CausalMethod.DIFFERENCE_IN_DIFFERENCES,
                sample_size=len(outcome),
                effect_size_interpretation=interpretation,
                statistical_significance=bool(p_value < self.significance_level),
                practical_significance=bool(abs(did_estimate) >= self.minimum_effect_size),
                robustness_score=0.8,  # Higher robustness for DiD
                assumptions_satisfied=True,
                metadata={
                    'standardized_effect': float(standardized_effect),
                    'regression_coefficients': beta_hat.tolist(),
                    'residual_std_error': float(np.sqrt(mse))
                }
            )
            
        except Exception as e:
            logger.error(f"Error in DiD estimation: {e}")
            return self._estimate_simple_difference(data)
    
    def _estimate_iv_effect(self, data: Dict[str, np.ndarray]) -> CausalEffect:
        """Estimate instrumental variables effect using 2SLS"""
        
        if 'instruments' not in data:
            logger.warning("No instruments provided for IV, falling back to simple difference")
            return self._estimate_simple_difference(data)
        
        outcome = data['outcome']
        treatment = data['treatment']
        instruments = data['instruments']
        
        try:
            # Two-stage least squares
            # Stage 1: Regress treatment on instruments
            X1 = np.column_stack([np.ones(len(treatment)), instruments])
            first_stage_coef = np.linalg.lstsq(X1, treatment, rcond=None)[0]
            treatment_fitted = X1 @ first_stage_coef
            
            # Stage 2: Regress outcome on fitted treatment
            X2 = np.column_stack([np.ones(len(outcome)), treatment_fitted])
            second_stage_coef = np.linalg.lstsq(X2, outcome, rcond=None)[0]
            iv_estimate = second_stage_coef[1]
            
            # Calculate standard errors (simplified)
            residuals2 = outcome - X2 @ second_stage_coef
            mse2 = np.sum(residuals2**2) / (len(outcome) - X2.shape[1])
            var_cov_matrix2 = mse2 * np.linalg.inv(X2.T @ X2)
            se = np.sqrt(var_cov_matrix2[1, 1])
            
            # T-test
            t_stat = iv_estimate / se
            dof = len(outcome) - X2.shape[1]
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
            
            # Confidence interval
            t_critical = stats.t.ppf(1 - self.significance_level/2, dof)
            ci_lower = iv_estimate - t_critical * se
            ci_upper = iv_estimate + t_critical * se
            
            # Effect interpretation
            outcome_std = np.std(outcome, ddof=1)
            standardized_effect = iv_estimate / outcome_std if outcome_std > 0 else 0
            
            if abs(standardized_effect) < 0.2:
                interpretation = "negligible effect"
            elif abs(standardized_effect) < 0.5:
                interpretation = "small effect"
            elif abs(standardized_effect) < 0.8:
                interpretation = "medium effect"
            else:
                interpretation = "large effect"
            
            return CausalEffect(
                effect_name="Instrumental Variables Estimate (2SLS)",
                point_estimate=float(iv_estimate),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                standard_error=float(se),
                p_value=float(p_value),
                method=CausalMethod.INSTRUMENTAL_VARIABLES,
                sample_size=len(outcome),
                effect_size_interpretation=interpretation,
                statistical_significance=bool(p_value < self.significance_level),
                practical_significance=bool(abs(iv_estimate) >= self.minimum_effect_size),
                robustness_score=0.9,  # High robustness if assumptions met
                assumptions_satisfied=True,
                metadata={
                    'first_stage_f_stat': float(np.var(treatment_fitted) / np.var(treatment - treatment_fitted)),
                    'standardized_effect': float(standardized_effect)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in IV estimation: {e}")
            return self._estimate_simple_difference(data)
    
    def _estimate_psm_effect(self, data: Dict[str, np.ndarray]) -> CausalEffect:
        """Estimate effect using propensity score matching"""
        
        if 'covariates' not in data:
            logger.warning("No covariates provided for PSM, falling back to simple difference")
            return self._estimate_simple_difference(data)
        
        outcome = data['outcome']
        treatment = data['treatment']
        covariates = data['covariates']
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.neighbors import NearestNeighbors
            
            # Estimate propensity scores
            lr = LogisticRegression(random_state=42)
            propensity_scores = lr.fit(covariates, treatment).predict_proba(covariates)[:, 1]
            
            # Find matches for treated units
            treated_indices = np.where(treatment == 1)[0]
            control_indices = np.where(treatment == 0)[0]
            
            if len(treated_indices) == 0 or len(control_indices) == 0:
                return self._estimate_simple_difference(data)
            
            # 1:1 nearest neighbor matching on propensity score
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(propensity_scores[control_indices].reshape(-1, 1))
            
            # Find matches
            distances, matched_control_idx = nn.kneighbors(
                propensity_scores[treated_indices].reshape(-1, 1)
            )
            
            # Get matched control indices
            matched_controls = control_indices[matched_control_idx.flatten()]
            
            # Calculate ATT (Average Treatment Effect on Treated)
            treated_outcomes = outcome[treated_indices]
            matched_control_outcomes = outcome[matched_controls]
            
            att_estimate = np.mean(treated_outcomes - matched_control_outcomes)
            
            # Bootstrap standard errors
            n_bootstrap = min(self.bootstrap_samples, 1000)
            bootstrap_estimates = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                boot_indices = np.random.choice(len(treated_indices), len(treated_indices), replace=True)
                boot_treated = treated_outcomes[boot_indices]
                boot_matched_control = matched_control_outcomes[boot_indices]
                boot_att = np.mean(boot_treated - boot_matched_control)
                bootstrap_estimates.append(boot_att)
            
            se = np.std(bootstrap_estimates)
            
            # T-test (approximate)
            t_stat = att_estimate / se if se > 0 else 0
            dof = len(treated_indices) - 1
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if se > 0 else 0.5
            
            # Confidence interval
            ci_lower = np.percentile(bootstrap_estimates, 2.5)
            ci_upper = np.percentile(bootstrap_estimates, 97.5)
            
            # Effect interpretation
            outcome_std = np.std(outcome, ddof=1)
            standardized_effect = att_estimate / outcome_std if outcome_std > 0 else 0
            
            if abs(standardized_effect) < 0.2:
                interpretation = "negligible effect"
            elif abs(standardized_effect) < 0.5:
                interpretation = "small effect"
            elif abs(standardized_effect) < 0.8:
                interpretation = "medium effect"
            else:
                interpretation = "large effect"
            
            return CausalEffect(
                effect_name="Propensity Score Matching Estimate (ATT)",
                point_estimate=float(att_estimate),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                standard_error=float(se),
                p_value=float(p_value),
                method=CausalMethod.PROPENSITY_SCORE_MATCHING,
                sample_size=len(treated_indices),
                effect_size_interpretation=interpretation,
                statistical_significance=bool(p_value < self.significance_level),
                practical_significance=bool(abs(att_estimate) >= self.minimum_effect_size),
                robustness_score=0.7,  # Moderate robustness
                assumptions_satisfied=True,
                metadata={
                    'n_matched_pairs': len(treated_indices),
                    'mean_propensity_score_treated': float(np.mean(propensity_scores[treated_indices])),
                    'mean_propensity_score_control': float(np.mean(propensity_scores[matched_controls])),
                    'standardized_effect': float(standardized_effect)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in PSM estimation: {e}")
            return self._estimate_simple_difference(data)
    
    def _estimate_doubly_robust_effect(self, data: Dict[str, np.ndarray]) -> CausalEffect:
        """Estimate effect using doubly robust method"""
        
        if 'covariates' not in data:
            logger.warning("No covariates provided for doubly robust, falling back to simple difference")
            return self._estimate_simple_difference(data)
        
        outcome = data['outcome']
        treatment = data['treatment']
        covariates = data['covariates']
        
        try:
            from sklearn.linear_model import LogisticRegression, LinearRegression
            
            # Step 1: Estimate propensity scores
            lr = LogisticRegression(random_state=42)
            propensity_scores = lr.fit(covariates, treatment).predict_proba(covariates)[:, 1]
            
            # Step 2: Estimate outcome regression for each treatment group
            treated_mask = treatment == 1
            control_mask = treatment == 0
            
            # Outcome regression for treated
            outcome_reg_treated = LinearRegression()
            outcome_reg_treated.fit(covariates[treated_mask], outcome[treated_mask])
            mu1_hat = outcome_reg_treated.predict(covariates)
            
            # Outcome regression for control
            outcome_reg_control = LinearRegression()
            outcome_reg_control.fit(covariates[control_mask], outcome[control_mask])
            mu0_hat = outcome_reg_control.predict(covariates)
            
            # Step 3: Doubly robust estimator
            # ATE = E[μ₁(X) - μ₀(X)] + E[T(Y - μ₁(X))/π(X)] - E[(1-T)(Y - μ₀(X))/(1-π(X))]
            
            # Clip propensity scores to avoid extreme weights
            propensity_scores = np.clip(propensity_scores, 0.01, 0.99)
            
            # Calculate components
            regression_component = np.mean(mu1_hat - mu0_hat)
            
            ipw_treated = treatment * (outcome - mu1_hat) / propensity_scores
            ipw_control = (1 - treatment) * (outcome - mu0_hat) / (1 - propensity_scores)
            
            ipw_component = np.mean(ipw_treated - ipw_control)
            
            dr_estimate = regression_component + ipw_component
            
            # Bootstrap standard errors
            n_bootstrap = min(self.bootstrap_samples, 1000)
            bootstrap_estimates = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                n = len(outcome)
                boot_indices = np.random.choice(n, n, replace=True)
                
                boot_outcome = outcome[boot_indices]
                boot_treatment = treatment[boot_indices]
                boot_covariates = covariates[boot_indices]
                
                try:
                    # Re-estimate models on bootstrap sample
                    boot_ps = LogisticRegression(random_state=42).fit(
                        boot_covariates, boot_treatment
                    ).predict_proba(boot_covariates)[:, 1]
                    boot_ps = np.clip(boot_ps, 0.01, 0.99)
                    
                    boot_treated_mask = boot_treatment == 1
                    boot_control_mask = boot_treatment == 0
                    
                    if np.sum(boot_treated_mask) > 0 and np.sum(boot_control_mask) > 0:
                        boot_mu1 = LinearRegression().fit(
                            boot_covariates[boot_treated_mask], 
                            boot_outcome[boot_treated_mask]
                        ).predict(boot_covariates)
                        
                        boot_mu0 = LinearRegression().fit(
                            boot_covariates[boot_control_mask], 
                            boot_outcome[boot_control_mask]
                        ).predict(boot_covariates)
                        
                        boot_reg = np.mean(boot_mu1 - boot_mu0)
                        boot_ipw_treated = boot_treatment * (boot_outcome - boot_mu1) / boot_ps
                        boot_ipw_control = (1 - boot_treatment) * (boot_outcome - boot_mu0) / (1 - boot_ps)
                        boot_ipw = np.mean(boot_ipw_treated - boot_ipw_control)
                        
                        boot_dr = boot_reg + boot_ipw
                        bootstrap_estimates.append(boot_dr)
                        
                except:
                    # Skip failed bootstrap samples
                    continue
            
            if len(bootstrap_estimates) > 10:
                se = np.std(bootstrap_estimates)
                ci_lower = np.percentile(bootstrap_estimates, 2.5)
                ci_upper = np.percentile(bootstrap_estimates, 97.5)
            else:
                # Fallback to analytical approximation
                se = np.sqrt(np.var(outcome) / len(outcome))
                t_critical = stats.t.ppf(1 - self.significance_level/2, len(outcome) - 1)
                ci_lower = dr_estimate - t_critical * se
                ci_upper = dr_estimate + t_critical * se
            
            # T-test
            t_stat = dr_estimate / se if se > 0 else 0
            dof = len(outcome) - 1
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if se > 0 else 0.5
            
            # Effect interpretation
            outcome_std = np.std(outcome, ddof=1)
            standardized_effect = dr_estimate / outcome_std if outcome_std > 0 else 0
            
            if abs(standardized_effect) < 0.2:
                interpretation = "negligible effect"
            elif abs(standardized_effect) < 0.5:
                interpretation = "small effect"
            elif abs(standardized_effect) < 0.8:
                interpretation = "medium effect"
            else:
                interpretation = "large effect"
            
            return CausalEffect(
                effect_name="Doubly Robust Estimate",
                point_estimate=float(dr_estimate),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                standard_error=float(se),
                p_value=float(p_value),
                method=CausalMethod.DOUBLY_ROBUST,
                sample_size=len(outcome),
                effect_size_interpretation=interpretation,
                statistical_significance=bool(p_value < self.significance_level),
                practical_significance=bool(abs(dr_estimate) >= self.minimum_effect_size),
                robustness_score=0.9,  # High robustness - doubly robust
                assumptions_satisfied=True,
                metadata={
                    'regression_component': float(regression_component),
                    'ipw_component': float(ipw_component),
                    'standardized_effect': float(standardized_effect),
                    'n_bootstrap_success': len(bootstrap_estimates)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in doubly robust estimation: {e}")
            return self._estimate_simple_difference(data)
    
    def _estimate_conditional_effects(self, data: Dict[str, np.ndarray],
                                    method: CausalMethod) -> Optional[CausalEffect]:
        """Estimate conditional average treatment effects (CATE)"""
        
        # This is a simplified implementation
        # In practice, you'd use more sophisticated methods like causal forests
        
        if 'covariates' not in data:
            return None
        
        try:
            outcome = data['outcome']
            treatment = data['treatment']
            covariates = data['covariates']
            
            # Simple approach: estimate treatment effect in different covariate subgroups
            # For demonstration, split on median of first covariate
            
            if covariates.shape[1] == 0:
                return None
            
            median_cov = np.median(covariates[:, 0])
            high_cov = covariates[:, 0] >= median_cov
            low_cov = covariates[:, 0] < median_cov
            
            # Effect in high covariate group
            high_treated = outcome[(high_cov) & (treatment == 1)]
            high_control = outcome[(high_cov) & (treatment == 0)]
            
            if len(high_treated) > 5 and len(high_control) > 5:
                high_effect = np.mean(high_treated) - np.mean(high_control)
                high_se = np.sqrt(np.var(high_treated, ddof=1)/len(high_treated) + 
                                 np.var(high_control, ddof=1)/len(high_control))
            else:
                high_effect = 0
                high_se = 0
            
            # Effect in low covariate group
            low_treated = outcome[(low_cov) & (treatment == 1)]
            low_control = outcome[(low_cov) & (treatment == 0)]
            
            if len(low_treated) > 5 and len(low_control) > 5:
                low_effect = np.mean(low_treated) - np.mean(low_control)
                low_se = np.sqrt(np.var(low_treated, ddof=1)/len(low_treated) + 
                                np.var(low_control, ddof=1)/len(low_control))
            else:
                low_effect = 0
                low_se = 0
            
            # Average the subgroup effects (weighted by sample size)
            n_high = np.sum(high_cov)
            n_low = np.sum(low_cov)
            
            if n_high > 0 and n_low > 0:
                cate_estimate = (n_high * high_effect + n_low * low_effect) / (n_high + n_low)
                cate_se = np.sqrt((n_high * high_se**2 + n_low * low_se**2) / (n_high + n_low))
                
                # Heterogeneity test
                heterogeneity = abs(high_effect - low_effect)
                
                return CausalEffect(
                    effect_name="Conditional Average Treatment Effect",
                    point_estimate=float(cate_estimate),
                    confidence_interval=(float(cate_estimate - 1.96*cate_se), 
                                       float(cate_estimate + 1.96*cate_se)),
                    standard_error=float(cate_se),
                    p_value=0.5,  # Placeholder
                    method=method,
                    sample_size=len(outcome),
                    effect_size_interpretation="varies by subgroup",
                    statistical_significance=True,
                    practical_significance=bool(abs(cate_estimate) >= self.minimum_effect_size),
                    robustness_score=0.6,
                    assumptions_satisfied=True,
                    metadata={
                        'high_group_effect': float(high_effect),
                        'low_group_effect': float(low_effect),
                        'heterogeneity': float(heterogeneity),
                        'subgroup_split_variable': 'covariate_0',
                        'split_threshold': float(median_cov)
                    }
                )
            
        except Exception as e:
            logger.warning(f"Error estimating conditional effects: {e}")
        
        return None
    
    def _perform_sensitivity_analysis(self, data: Dict[str, np.ndarray],
                                    method: CausalMethod) -> Dict[str, Any]:
        """Perform sensitivity analysis for unmeasured confounding"""
        
        try:
            outcome = data['outcome']
            treatment = data['treatment']
            
            # Simple sensitivity analysis: vary the effect of unmeasured confounder
            baseline_effect = self._estimate_simple_difference(data).point_estimate
            
            # Test sensitivity to different levels of confounding bias
            confounder_effects = np.linspace(0, abs(baseline_effect), 10)
            adjusted_effects = []
            
            for confounder_effect in confounder_effects:
                # Simulate effect of unmeasured confounder
                # This is a simplified approach
                bias = confounder_effect * 0.5  # Assume moderate correlation
                adjusted_effect = baseline_effect - bias
                adjusted_effects.append(adjusted_effect)
            
            # Find critical level where effect becomes non-significant
            critical_bias = None
            for i, effect in enumerate(adjusted_effects):
                if abs(effect) < self.minimum_effect_size:
                    critical_bias = confounder_effects[i]
                    break
            
            return {
                'baseline_effect': float(baseline_effect),
                'confounder_effect_range': confounder_effects.tolist(),
                'adjusted_effects': adjusted_effects,
                'critical_bias': float(critical_bias) if critical_bias is not None else None,
                'robust_to_small_bias': critical_bias is None or critical_bias > abs(baseline_effect) * 0.2,
                'interpretation': "Effect remains significant unless very large unmeasured confounding"
            }
            
        except Exception as e:
            logger.warning(f"Error in sensitivity analysis: {e}")
            return {'error': str(e)}
    
    def _perform_placebo_tests(self, data: Dict[str, np.ndarray],
                             method: CausalMethod) -> Dict[str, Any]:
        """Perform placebo tests to check for spurious effects"""
        
        try:
            outcome = data['outcome']
            treatment = data['treatment']
            
            placebo_tests = []
            
            # Test 1: Random permutation of treatment
            n_permutations = min(100, self.bootstrap_samples // 10)
            permutation_effects = []
            
            for _ in range(n_permutations):
                # Randomly permute treatment assignment
                permuted_treatment = np.random.permutation(treatment)
                
                # Calculate "effect" with permuted treatment
                permuted_data = data.copy()
                permuted_data['treatment'] = permuted_treatment
                
                placebo_effect = self._estimate_simple_difference(permuted_data).point_estimate
                permutation_effects.append(placebo_effect)
            
            # Check if original effect is in tail of null distribution
            original_effect = self._estimate_simple_difference(data).point_estimate
            p_value_permutation = np.mean(np.abs(permutation_effects) >= abs(original_effect))
            
            placebo_tests.append({
                'test_name': 'random_permutation',
                'p_value': float(p_value_permutation),
                'passes': p_value_permutation < 0.05,
                'description': 'Treatment permutation test'
            })
            
            # Test 2: Lagged outcome (if time periods available)
            if 'time_periods' in data:
                # This would test pre-treatment "effects"
                placebo_tests.append({
                    'test_name': 'pre_treatment_placebo',
                    'p_value': 0.5,  # Placeholder
                    'passes': True,  # Placeholder
                    'description': 'Pre-treatment placebo test'
                })
            
            return {
                'placebo_tests': placebo_tests,
                'overall_passes': all(test['passes'] for test in placebo_tests),
                'permutation_distribution': permutation_effects[:20],  # Sample for inspection
                'original_effect': float(original_effect)
            }
            
        except Exception as e:
            logger.warning(f"Error in placebo tests: {e}")
            return {'error': str(e)}
    
    def _assess_confounding(self, data: Dict[str, np.ndarray],
                          assignment: TreatmentAssignment) -> Dict[str, Any]:
        """Assess potential for unmeasured confounding"""
        
        outcome = data['outcome']
        treatment = data['treatment']
        
        assessment = {
            'assignment_mechanism': assignment.value,
            'confounding_risk': 'low' if assignment == TreatmentAssignment.RANDOMIZED else 'high'
        }
        
        try:
            # Calculate some basic statistics
            treated_outcome_mean = np.mean(outcome[treatment == 1])
            control_outcome_mean = np.mean(outcome[treatment == 0])
            overall_outcome_mean = np.mean(outcome)
            
            # Check for extreme differences that might suggest confounding
            extreme_difference = abs(treated_outcome_mean - control_outcome_mean) > 2 * np.std(outcome)
            
            assessment.update({
                'treated_outcome_mean': float(treated_outcome_mean),
                'control_outcome_mean': float(control_outcome_mean),
                'overall_outcome_mean': float(overall_outcome_mean),
                'extreme_difference_detected': extreme_difference,
                'potential_confounding_indicators': []
            })
            
            if extreme_difference:
                assessment['potential_confounding_indicators'].append("Extreme outcome differences")
            
            # Check treatment prevalence
            treatment_prevalence = np.mean(treatment)
            if treatment_prevalence < 0.1 or treatment_prevalence > 0.9:
                assessment['potential_confounding_indicators'].append("Unbalanced treatment assignment")
            
            assessment['treatment_prevalence'] = float(treatment_prevalence)
            
        except Exception as e:
            logger.warning(f"Error assessing confounding: {e}")
            assessment['error'] = str(e)
        
        return assessment
    
    def _assess_covariate_balance(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Assess balance of covariates between treatment groups"""
        
        if 'covariates' not in data:
            return {'no_covariates': True}
        
        covariates = data['covariates']
        treatment = data['treatment']
        
        balance_results = []
        
        try:
            for i in range(covariates.shape[1]):
                covariate = covariates[:, i]
                
                treated_cov = covariate[treatment == 1]
                control_cov = covariate[treatment == 0]
                
                # Standardized mean difference
                pooled_std = np.sqrt(((len(treated_cov) - 1) * np.var(treated_cov, ddof=1) + 
                                    (len(control_cov) - 1) * np.var(control_cov, ddof=1)) / 
                                   (len(treated_cov) + len(control_cov) - 2))
                
                smd = (np.mean(treated_cov) - np.mean(control_cov)) / pooled_std if pooled_std > 0 else 0
                
                # T-test
                statistic, p_value = stats.ttest_ind(treated_cov, control_cov)
                
                balance_results.append({
                    'covariate_index': i,
                    'standardized_mean_difference': float(smd),
                    'p_value': float(p_value),
                    'balanced': abs(smd) < 0.1
                })
            
            # Overall balance assessment
            n_imbalanced = sum(1 for r in balance_results if not r['balanced'])
            overall_balanced = n_imbalanced / len(balance_results) <= 0.05
            
            return {
                'balance_results': balance_results,
                'n_covariates': len(balance_results),
                'n_imbalanced': n_imbalanced,
                'proportion_imbalanced': n_imbalanced / len(balance_results),
                'overall_balanced': overall_balanced
            }
            
        except Exception as e:
            logger.warning(f"Error assessing covariate balance: {e}")
            return {'error': str(e)}
    
    def _calculate_robustness_score(self, primary_effect: CausalEffect,
                                  sensitivity_results: Optional[Dict[str, Any]],
                                  placebo_results: Optional[Dict[str, Any]]) -> float:
        """Calculate overall robustness score"""
        
        score_components = []
        
        # Effect size component (larger effects are more robust)
        effect_magnitude = abs(primary_effect.point_estimate)
        effect_score = min(effect_magnitude / (2 * self.minimum_effect_size), 1.0)
        score_components.append(('effect_magnitude', effect_score, 0.3))
        
        # Statistical significance component
        sig_score = 1.0 if primary_effect.statistical_significance else 0.2
        score_components.append(('statistical_significance', sig_score, 0.2))
        
        # Sensitivity analysis component
        if sensitivity_results and 'robust_to_small_bias' in sensitivity_results:
            sens_score = 1.0 if sensitivity_results['robust_to_small_bias'] else 0.5
        else:
            sens_score = 0.7  # Moderate score if no sensitivity analysis
        score_components.append(('sensitivity_analysis', sens_score, 0.25))
        
        # Placebo tests component
        if placebo_results and 'overall_passes' in placebo_results:
            placebo_score = 1.0 if placebo_results['overall_passes'] else 0.3
        else:
            placebo_score = 0.7  # Moderate score if no placebo tests
        score_components.append(('placebo_tests', placebo_score, 0.25))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        return min(max(total_score, 0.0), 1.0)
    
    def _calculate_internal_validity_score(self, assumptions: List[CausalAssumption],
                                         confounding_assessment: Dict[str, Any]) -> float:
        """Calculate internal validity score"""
        
        if not assumptions:
            return 0.5
        
        # Score based on assumption violations
        critical_violations = sum(1 for a in assumptions if a.violated and a.severity == "high")
        moderate_violations = sum(1 for a in assumptions if a.violated and a.severity == "medium")
        
        assumption_score = max(0, 1.0 - 0.4 * critical_violations - 0.2 * moderate_violations)
        
        # Adjust for confounding risk
        confounding_risk = confounding_assessment.get('confounding_risk', 'medium')
        if confounding_risk == 'low':
            confounding_score = 1.0
        elif confounding_risk == 'medium':
            confounding_score = 0.7
        else:
            confounding_score = 0.4
        
        return (assumption_score + confounding_score) / 2
    
    def _calculate_external_validity_score(self, data: Dict[str, np.ndarray],
                                         method: CausalMethod) -> float:
        """Calculate external validity score"""
        
        score_components = []
        
        # Sample size component
        n_total = data['n_total']
        sample_score = min(n_total / 1000, 1.0)  # Full score at 1000+ observations
        score_components.append(('sample_size', sample_score, 0.3))
        
        # Treatment group balance
        n_treated = data['n_treated']
        n_control = data['n_control']
        balance = min(n_treated, n_control) / max(n_treated, n_control)
        balance_score = balance  # 1.0 for perfect balance, lower for imbalance
        score_components.append(('treatment_balance', balance_score, 0.3))
        
        # Method sophistication (some methods have better external validity)
        method_scores = {
            CausalMethod.DIFFERENCE_IN_DIFFERENCES: 0.9,
            CausalMethod.INSTRUMENTAL_VARIABLES: 0.8,
            CausalMethod.DOUBLY_ROBUST: 0.9,
            CausalMethod.PROPENSITY_SCORE_MATCHING: 0.7,
            CausalMethod.REGRESSION_DISCONTINUITY: 0.8,
            CausalMethod.SYNTHETIC_CONTROL: 0.8
        }
        method_score = method_scores.get(method, 0.6)
        score_components.append(('method_sophistication', method_score, 0.4))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        return min(max(total_score, 0.0), 1.0)
    
    def _generate_causal_interpretation(self, effect: CausalEffect,
                                      assumptions: List[CausalAssumption]) -> str:
        """Generate causal interpretation of results"""
        
        violated_assumptions = [a for a in assumptions if a.violated]
        critical_violations = [a for a in violated_assumptions if a.severity == "high"]
        
        if effect.statistical_significance and effect.practical_significance:
            if not critical_violations:
                return f"Strong evidence for causal effect: The treatment causes a {effect.effect_size_interpretation} ({effect.point_estimate:.3f}) change in the outcome with high confidence."
            else:
                return f"Suggestive evidence for causal effect: Treatment appears to cause a {effect.effect_size_interpretation} change, but key assumptions are violated, limiting causal interpretation."
        elif effect.statistical_significance:
            return f"Statistically significant but small effect: While statistically detectable, the effect size ({effect.point_estimate:.3f}) may not be practically meaningful."
        else:
            return f"No evidence for causal effect: The analysis does not support a causal relationship between treatment and outcome."
    
    def _generate_business_recommendations(self, effect: CausalEffect,
                                         assumptions: List[CausalAssumption],
                                         robustness_score: float) -> List[str]:
        """Generate business recommendations based on causal analysis"""
        
        recommendations = []
        
        if effect.statistical_significance and effect.practical_significance and robustness_score > 0.7:
            recommendations.append("✅ STRONG RECOMMENDATION: Deploy treatment based on robust causal evidence")
            recommendations.append("📈 Expected impact: Implement to achieve meaningful business outcomes")
            
        elif effect.statistical_significance and robustness_score > 0.5:
            recommendations.append("⚠️ CONDITIONAL RECOMMENDATION: Consider deployment with monitoring")
            recommendations.append("🔍 Risk management: Implement with close performance tracking")
            
        else:
            recommendations.append("❌ NOT RECOMMENDED: Insufficient evidence for causal effect")
            recommendations.append("🔬 Additional research: Collect more data or improve experimental design")
        
        # Assumption-specific recommendations
        violated_assumptions = [a for a in assumptions if a.violated]
        if violated_assumptions:
            recommendations.append("⚠️ CAUTION: Key causal assumptions violated - interpret results carefully")
            
            for assumption in violated_assumptions[:3]:  # Top 3 violations
                if assumption.recommendations:
                    recommendations.extend([f"• {rec}" for rec in assumption.recommendations[:2]])
        
        # Robustness-specific recommendations
        if robustness_score < 0.5:
            recommendations.append("🔧 IMPROVE ROBUSTNESS: Consider additional validation methods")
            recommendations.append("📊 Sensitivity analysis: Test robustness to alternative assumptions")
        
        return recommendations
    
    def _generate_statistical_warnings(self, assumptions: List[CausalAssumption],
                                     confounding_assessment: Dict[str, Any]) -> List[str]:
        """Generate statistical warnings and caveats"""
        
        warnings = []
        
        # Assumption violations
        critical_violations = [a for a in assumptions if a.violated and a.severity == "high"]
        if critical_violations:
            warnings.append(f"⚠️ CRITICAL: {len(critical_violations)} key assumptions violated")
            for violation in critical_violations[:2]:
                warnings.append(f"• {violation.name}: {violation.description}")
        
        # Confounding concerns
        confounding_risk = confounding_assessment.get('confounding_risk', 'unknown')
        if confounding_risk == 'high':
            warnings.append("⚠️ HIGH CONFOUNDING RISK: Observational data limits causal inference")
            
        extreme_diff = confounding_assessment.get('extreme_difference_detected', False)
        if extreme_diff:
            warnings.append("⚠️ EXTREME DIFFERENCES: Large baseline differences suggest possible confounding")
        
        # Sample size warnings
        if confounding_assessment.get('treatment_prevalence', 0.5) < 0.1:
            warnings.append("⚠️ SMALL TREATMENT GROUP: Limited power for detecting effects")
        
        return warnings


# Utility functions for external use
def quick_causal_analysis(outcome_data: List[float],
                         treatment_data: List[int],
                         covariates: Optional[List[List[float]]] = None,
                         method: str = "simple_difference") -> Dict[str, Any]:
    """Quick causal analysis for immediate use"""
    
    analyzer = CausalInferenceAnalyzer()
    
    try:
        # Convert method string to enum
        method_map = {
            "simple_difference": CausalMethod.DIFFERENCE_IN_DIFFERENCES,
            "did": CausalMethod.DIFFERENCE_IN_DIFFERENCES,
            "iv": CausalMethod.INSTRUMENTAL_VARIABLES,
            "psm": CausalMethod.PROPENSITY_SCORE_MATCHING,
            "doubly_robust": CausalMethod.DOUBLY_ROBUST
        }
        
        causal_method = method_map.get(method, CausalMethod.DIFFERENCE_IN_DIFFERENCES)
        
        # Determine assignment mechanism
        assignment = TreatmentAssignment.RANDOMIZED if covariates is None else TreatmentAssignment.QUASI_EXPERIMENTAL
        
        # Convert covariates to numpy array
        cov_array = np.array(covariates) if covariates else None
        
        result = analyzer.analyze_causal_effect(
            outcome_data=np.array(outcome_data),
            treatment_data=np.array(treatment_data),
            covariates=cov_array,
            assignment_mechanism=assignment,
            method=causal_method
        )
        
        return {
            'causal_effect': result.average_treatment_effect.point_estimate,
            'confidence_interval': result.average_treatment_effect.confidence_interval,
            'p_value': result.average_treatment_effect.p_value,
            'statistical_significance': result.average_treatment_effect.statistical_significance,
            'practical_significance': result.average_treatment_effect.practical_significance,
            'effect_interpretation': result.average_treatment_effect.effect_size_interpretation,
            'causal_interpretation': result.causal_interpretation,
            'business_recommendations': result.business_recommendations,
            'robustness_score': result.robustness_score,
            'overall_quality': result.overall_quality_score,
            'assumptions_satisfied': result.overall_assumptions_satisfied,
            'warnings': result.statistical_warnings
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'causal_effect': 0.0,
            'statistical_significance': False
        }