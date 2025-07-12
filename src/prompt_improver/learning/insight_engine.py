"""Insight Generation Engine

Generates actionable insights from performance data and learning patterns.
Combines statistical analysis with pattern recognition to provide strategic
recommendations for system improvement.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from scipy import stats
import warnings

# Phase 2 enhancement imports for causal discovery
try:
    import networkx as nx
    from pgmpy.estimators import PC
    from pgmpy.models import BayesianNetwork
    import pandas as pd
    CAUSAL_DISCOVERY_AVAILABLE = True
except ImportError:
    import pandas as pd  # pandas is available separately
    CAUSAL_DISCOVERY_AVAILABLE = False
    warnings.warn("Causal discovery libraries not available. Install with: pip install networkx pgmpy")

logger = logging.getLogger(__name__)


@dataclass 
class InsightConfig:
    """Configuration for insight generation"""
    min_confidence: float = 0.7
    min_sample_size: int = 10
    significance_threshold: float = 0.05
    trend_window_days: int = 30
    max_insights: int = 20
    
    # Phase 2 enhancements - Causal Discovery
    enable_causal_discovery: bool = True
    causal_significance_level: float = 0.05
    min_causal_samples: int = 20
    max_causal_variables: int = 15
    intervention_confidence_threshold: float = 0.8


@dataclass
class Insight:
    """Generated insight with supporting evidence"""
    insight_id: str
    type: str  # 'performance', 'trend', 'pattern', 'opportunity', 'risk', 'causal'
    title: str
    description: str
    confidence: float
    impact: str  # 'high', 'medium', 'low'
    evidence: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalRelationship:
    """Discovered causal relationship"""
    cause: str
    effect: str
    strength: float
    confidence: float
    intervention_effect: Optional[float] = None
    statistical_tests: Dict[str, float] = field(default_factory=dict)


class InsightGenerationEngine:
    """Engine for generating actionable insights from performance data"""
    
    def __init__(self, config: Optional[InsightConfig] = None):
        self.config = config or InsightConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def generate_insights(self, performance_data: Dict[str, Any]) -> List[Insight]:
        """Generate insights from performance data"""
        insights = []
        
        # Performance insights
        performance_insights = await self._analyze_performance_patterns(performance_data)
        insights.extend(performance_insights)
        
        # Trend insights
        trend_insights = await self._analyze_trends(performance_data)
        insights.extend(trend_insights)
        
        # Opportunity insights
        opportunity_insights = await self._identify_opportunities(performance_data)
        insights.extend(opportunity_insights)
        
        # Risk insights
        risk_insights = await self._identify_risks(performance_data)
        insights.extend(risk_insights)
        
        # Phase 2: Causal discovery insights
        if self.config.enable_causal_discovery and CAUSAL_DISCOVERY_AVAILABLE:
            causal_insights = await self._discover_causal_relationships(performance_data)
            insights.extend(causal_insights)
        
        # Filter and sort by confidence and impact
        insights = [i for i in insights if i.confidence >= self.config.min_confidence]
        insights.sort(key=lambda x: (x.impact == 'high', x.confidence), reverse=True)
        
        return insights[:self.config.max_insights]
    
    async def _analyze_performance_patterns(self, data: Dict[str, Any]) -> List[Insight]:
        """Analyze performance patterns"""
        insights = []
        
        # Rule performance analysis
        if 'rule_performance' in data:
            rule_data = data['rule_performance']
            
            # Find top and bottom performers
            sorted_rules = sorted(
                rule_data.items(),
                key=lambda x: x[1].get('avg_score', 0),
                reverse=True
            )
            
            if len(sorted_rules) >= 3:
                best_rule = sorted_rules[0]
                worst_rule = sorted_rules[-1]
                
                score_gap = best_rule[1].get('avg_score', 0) - worst_rule[1].get('avg_score', 0)
                
                if score_gap > 0.2:  # Significant performance gap
                    insights.append(Insight(
                        insight_id="performance_gap",
                        type="performance",
                        title="Significant Rule Performance Gap",
                        description=f"Large performance difference between best ({best_rule[0]}: {best_rule[1].get('avg_score', 0):.3f}) and worst ({worst_rule[0]}: {worst_rule[1].get('avg_score', 0):.3f}) performing rules",
                        confidence=0.9,
                        impact="high",
                        evidence=[
                            f"Performance gap: {score_gap:.3f}",
                            f"Best rule sample size: {best_rule[1].get('sample_size', 0)}",
                            f"Worst rule sample size: {worst_rule[1].get('sample_size', 0)}"
                        ],
                        recommendations=[
                            f"Investigate why {best_rule[0]} performs better",
                            f"Consider deprecating or improving {worst_rule[0]}",
                            "Apply successful patterns from top performers to bottom performers"
                        ]
                    ))
        
        return insights
    
    async def _analyze_trends(self, data: Dict[str, Any]) -> List[Insight]:
        """Analyze performance trends"""
        insights = []
        
        if 'temporal_data' in data:
            temporal_data = data['temporal_data']
            
            # Analyze overall system trend
            if len(temporal_data) >= 7:  # At least a week of data
                timestamps = [d['timestamp'] for d in temporal_data]
                scores = [d['avg_score'] for d in temporal_data]
                
                # Calculate trend
                x = np.arange(len(scores))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
                
                if p_value < self.config.significance_threshold:
                    trend_direction = "improving" if slope > 0 else "declining"
                    confidence = min(0.95, abs(r_value))
                    
                    insights.append(Insight(
                        insight_id="system_trend",
                        type="trend", 
                        title=f"System Performance {trend_direction.title()}",
                        description=f"Overall system performance shows {trend_direction} trend over time",
                        confidence=confidence,
                        impact="high" if abs(slope) > 0.01 else "medium",
                        evidence=[
                            f"Trend slope: {slope:.4f}",
                            f"R-squared: {r_value**2:.3f}",
                            f"Statistical significance: p={p_value:.3f}"
                        ],
                        recommendations=[
                            "Continue monitoring trend",
                            "Identify factors driving the trend" if slope > 0 else "Investigate causes of decline"
                        ]
                    ))
        
        return insights
    
    async def _identify_opportunities(self, data: Dict[str, Any]) -> List[Insight]:
        """Identify improvement opportunities"""
        insights = []
        
        # Context-specific opportunities
        if 'context_performance' in data:
            context_data = data['context_performance']
            
            # Find underperforming contexts with sufficient volume
            underperforming = [
                (context, perf) for context, perf in context_data.items()
                if perf.get('avg_score', 1) < 0.7 and perf.get('sample_size', 0) >= self.config.min_sample_size
            ]
            
            if underperforming:
                underperforming.sort(key=lambda x: x[1].get('sample_size', 0), reverse=True)
                top_opportunity = underperforming[0]
                
                insights.append(Insight(
                    insight_id="context_opportunity",
                    type="opportunity",
                    title="High-Volume Improvement Opportunity",
                    description=f"Context '{top_opportunity[0]}' has high volume but low performance",
                    confidence=0.8,
                    impact="high",
                    evidence=[
                        f"Average score: {top_opportunity[1].get('avg_score', 0):.3f}",
                        f"Sample size: {top_opportunity[1].get('sample_size', 0)}",
                        f"Improvement potential: {0.8 - top_opportunity[1].get('avg_score', 0):.3f}"
                    ],
                    recommendations=[
                        f"Create specialized rules for {top_opportunity[0]} context",
                        "Analyze successful patterns from other contexts",
                        "Increase focus on this high-volume area"
                    ]
                ))
        
        return insights
    
    async def _identify_risks(self, data: Dict[str, Any]) -> List[Insight]:
        """Identify potential risks"""
        insights = []
        
        # Consistency risks
        if 'rule_performance' in data:
            rule_data = data['rule_performance']
            
            inconsistent_rules = [
                (rule_id, perf) for rule_id, perf in rule_data.items()
                if perf.get('std_score', 0) > 0.3 and perf.get('sample_size', 0) >= self.config.min_sample_size
            ]
            
            if inconsistent_rules:
                most_inconsistent = max(inconsistent_rules, key=lambda x: x[1].get('std_score', 0))
                
                insights.append(Insight(
                    insight_id="consistency_risk",
                    type="risk",
                    title="High Performance Variability",
                    description=f"Rule '{most_inconsistent[0]}' shows high performance variability",
                    confidence=0.85,
                    impact="medium",
                    evidence=[
                        f"Standard deviation: {most_inconsistent[1].get('std_score', 0):.3f}",
                        f"Average score: {most_inconsistent[1].get('avg_score', 0):.3f}",
                        f"Coefficient of variation: {most_inconsistent[1].get('std_score', 0) / most_inconsistent[1].get('avg_score', 1):.3f}"
                    ],
                    recommendations=[
                        "Investigate causes of inconsistency",
                        "Add rule stability checks",
                        "Consider rule refinement or replacement"
                    ]
                ))
        
        return insights
    
    async def _discover_causal_relationships(self, data: Dict[str, Any]) -> List[Insight]:
        """Phase 2: Discover causal relationships using PC algorithm"""
        insights = []
        
        if not CAUSAL_DISCOVERY_AVAILABLE:
            self.logger.warning("Causal discovery libraries not available")
            return insights
        
        try:
            # Prepare data for causal discovery
            causal_data = self._prepare_causal_data(data)
            if causal_data is None or len(causal_data) < self.config.min_causal_samples:
                self.logger.info(f"Insufficient data for causal discovery: {len(causal_data) if causal_data is not None else 0} samples")
                return insights
            
            # Apply PC algorithm for causal structure learning
            causal_graph = self._apply_pc_algorithm(causal_data)
            if causal_graph is None:
                return insights
            
            # Extract causal relationships
            relationships = self._extract_causal_relationships(causal_graph, causal_data)
            
            # Generate intervention analysis
            intervention_insights = self._analyze_interventions(relationships, causal_data, causal_graph)
            
            # Convert to insights
            for relationship in relationships:
                if relationship.confidence >= self.config.intervention_confidence_threshold:
                    insight = self._create_causal_insight(relationship, intervention_insights.get(relationship.cause))
                    if insight:
                        insights.append(insight)
            
            self.logger.info(f"Generated {len(insights)} causal insights from {len(relationships)} relationships")
            
        except Exception as e:
            self.logger.error(f"Causal discovery failed: {e}")
        
        return insights
    
    def _prepare_causal_data(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Prepare data for causal discovery analysis"""
        try:
            # Extract relevant features for causal analysis
            causal_features = []
            
            # Rule performance features
            if 'rule_performance' in data:
                rule_data = data['rule_performance']
                for rule_id, perf in rule_data.items():
                    if perf.get('sample_size', 0) >= self.config.min_sample_size:
                        feature_row = {
                            'rule_avg_score': perf.get('avg_score', 0),
                            'rule_consistency': 1 - (perf.get('std_score', 0) / max(perf.get('avg_score', 1), 0.01)),
                            'rule_sample_size': min(perf.get('sample_size', 0) / 100, 1.0),  # Normalize
                            'rule_category': self._encode_categorical(rule_id.split('_')[0] if '_' in rule_id else 'general')
                        }
                        causal_features.append(feature_row)
            
            # Context performance features
            if 'context_performance' in data:
                context_data = data['context_performance']
                for i, (context, perf) in enumerate(context_data.items()):
                    if perf.get('sample_size', 0) >= self.config.min_sample_size:
                        feature_row = {
                            'context_avg_score': perf.get('avg_score', 0),
                            'context_consistency': 1 - (perf.get('std_score', 0) / max(perf.get('avg_score', 1), 0.01)),
                            'context_complexity': self._estimate_context_complexity(context),
                            'context_volume': min(perf.get('sample_size', 0) / 100, 1.0)  # Normalize
                        }
                        causal_features.append(feature_row)
            
            # System-level features
            if 'system_metrics' in data:
                system_data = data['system_metrics']
                feature_row = {
                    'system_overall_score': system_data.get('overall_score', 0),
                    'system_efficiency': system_data.get('efficiency_score', 0),
                    'system_reliability': system_data.get('reliability_score', 0),
                    'system_load': min(system_data.get('processing_load', 0) / 100, 1.0)
                }
                causal_features.append(feature_row)
            
            if not causal_features:
                return None
            
            # Create DataFrame and limit variables
            df = pd.DataFrame(causal_features)
            
            # Limit to max variables for computational efficiency
            if len(df.columns) > self.config.max_causal_variables:
                # Select most variable features
                variances = df.var()
                top_vars = variances.nlargest(self.config.max_causal_variables).index
                df = df[top_vars]
            
            # Remove columns with insufficient variance
            df = df.loc[:, df.var() > 0.01]
            
            return df if len(df) >= self.config.min_causal_samples else None
            
        except Exception as e:
            self.logger.error(f"Failed to prepare causal data: {e}")
            return None
    
    def _apply_pc_algorithm(self, causal_data: pd.DataFrame) -> Optional[nx.DiGraph]:
        """Apply PC algorithm for causal structure learning"""
        try:
            # Initialize PC estimator
            pc_estimator = PC(data=causal_data)
            
            # Learn causal structure with statistical tests
            skeleton, separating_sets = pc_estimator.build_skeleton(
                significance_level=self.config.causal_significance_level
            )
            
            # Orient edges to create DAG
            dag = pc_estimator.skeleton_to_pdag(skeleton, separating_sets)
            
            # Convert to NetworkX DiGraph for easier manipulation
            nx_graph = nx.DiGraph()
            for edge in dag.edges():
                nx_graph.add_edge(edge[0], edge[1])
            
            self.logger.info(f"PC algorithm discovered DAG with {len(nx_graph.nodes)} nodes and {len(nx_graph.edges)} edges")
            
            return nx_graph
            
        except Exception as e:
            self.logger.error(f"PC algorithm failed: {e}")
            return None
    
    def _extract_causal_relationships(self, causal_graph: nx.DiGraph, 
                                    causal_data: pd.DataFrame) -> List[CausalRelationship]:
        """Extract causal relationships from discovered DAG"""
        relationships = []
        
        try:
            for cause, effect in causal_graph.edges():
                # Calculate relationship strength using correlation
                if cause in causal_data.columns and effect in causal_data.columns:
                    correlation = causal_data[cause].corr(causal_data[effect])
                    strength = abs(correlation)
                    
                    # Statistical tests for relationship confidence
                    stat_tests = self._perform_causal_tests(
                        causal_data[cause], causal_data[effect], causal_graph, cause, effect
                    )
                    
                    # Calculate confidence based on statistical evidence
                    confidence = self._calculate_causal_confidence(stat_tests)
                    
                    relationship = CausalRelationship(
                        cause=cause,
                        effect=effect,
                        strength=strength,
                        confidence=confidence,
                        statistical_tests=stat_tests
                    )
                    relationships.append(relationship)
            
            # Sort by confidence and strength
            relationships.sort(key=lambda r: (r.confidence, r.strength), reverse=True)
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Failed to extract causal relationships: {e}")
            return []
    
    def _perform_causal_tests(self, cause_data: pd.Series, effect_data: pd.Series,
                            causal_graph: nx.DiGraph, cause: str, effect: str) -> Dict[str, float]:
        """Perform statistical tests for causal relationship validation"""
        tests = {}
        
        try:
            # Correlation test
            correlation, corr_p_value = stats.pearsonr(cause_data, effect_data)
            tests['correlation'] = abs(correlation)
            tests['correlation_p_value'] = corr_p_value
            
            # Granger causality test (simplified version using lag correlation)
            if len(cause_data) > 10:
                # Create lagged versions
                cause_lagged = cause_data.shift(1).dropna()
                effect_current = effect_data[1:len(cause_lagged)+1]
                
                if len(cause_lagged) > 5:
                    lag_correlation, lag_p_value = stats.pearsonr(cause_lagged, effect_current)
                    tests['granger_approximation'] = abs(lag_correlation)
                    tests['granger_p_value'] = lag_p_value
            
            # Partial correlation (controlling for common causes)
            common_parents = set(causal_graph.predecessors(cause)).intersection(
                set(causal_graph.predecessors(effect))
            )
            
            if common_parents:
                # Simplified partial correlation
                tests['has_confounders'] = 1.0
                tests['n_confounders'] = float(len(common_parents))
            else:
                tests['has_confounders'] = 0.0
                tests['n_confounders'] = 0.0
            
        except Exception as e:
            self.logger.warning(f"Causal tests failed for {cause} -> {effect}: {e}")
        
        return tests
    
    def _calculate_causal_confidence(self, stat_tests: Dict[str, float]) -> float:
        """Calculate confidence in causal relationship based on statistical tests"""
        confidence = 0.0
        
        # Base confidence from correlation strength
        if 'correlation' in stat_tests:
            confidence += min(stat_tests['correlation'] * 0.5, 0.4)
        
        # Boost confidence with significant correlation
        if stat_tests.get('correlation_p_value', 1.0) < self.config.causal_significance_level:
            confidence += 0.2
        
        # Granger causality approximation
        if 'granger_approximation' in stat_tests:
            confidence += min(stat_tests['granger_approximation'] * 0.3, 0.2)
        
        # Penalize for confounders
        if stat_tests.get('has_confounders', 0) > 0:
            confounder_penalty = min(stat_tests.get('n_confounders', 0) * 0.1, 0.3)
            confidence -= confounder_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _analyze_interventions(self, relationships: List[CausalRelationship],
                             causal_data: pd.DataFrame, causal_graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """Analyze potential interventions and their effects"""
        intervention_analysis = {}
        
        try:
            for relationship in relationships:
                if relationship.confidence >= self.config.intervention_confidence_threshold:
                    # Simulate intervention effect
                    intervention_effect = self._simulate_intervention(
                        relationship, causal_data, causal_graph
                    )
                    
                    intervention_analysis[relationship.cause] = {
                        'target_effect': relationship.effect,
                        'estimated_improvement': intervention_effect,
                        'confidence': relationship.confidence,
                        'intervention_feasibility': self._assess_intervention_feasibility(relationship.cause)
                    }
        
        except Exception as e:
            self.logger.error(f"Intervention analysis failed: {e}")
        
        return intervention_analysis
    
    def _simulate_intervention(self, relationship: CausalRelationship,
                             causal_data: pd.DataFrame, causal_graph: nx.DiGraph) -> float:
        """Simulate the effect of intervening on a causal variable"""
        try:
            cause_data = causal_data[relationship.cause]
            effect_data = causal_data[relationship.effect]
            
            # Simple intervention simulation: increase cause by one standard deviation
            cause_std = cause_data.std()
            intervention_magnitude = cause_std
            
            # Estimate effect using linear approximation
            correlation = cause_data.corr(effect_data)
            effect_std = effect_data.std()
            
            # Linear relationship assumption
            estimated_effect_change = correlation * (intervention_magnitude / cause_std) * effect_std
            
            # Normalize to percentage improvement
            current_effect_mean = effect_data.mean()
            if current_effect_mean > 0:
                percentage_improvement = estimated_effect_change / current_effect_mean
                return min(abs(percentage_improvement), 1.0)  # Cap at 100% improvement
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Intervention simulation failed: {e}")
            return 0.0
    
    def _assess_intervention_feasibility(self, variable: str) -> float:
        """Assess how feasible it is to intervene on a variable"""
        # Simple heuristic based on variable type
        feasibility_map = {
            'rule': 0.9,      # Can modify rules easily
            'context': 0.3,   # Context is largely external
            'system': 0.7,    # System parameters can be tuned
        }
        
        for var_type, feasibility in feasibility_map.items():
            if var_type in variable.lower():
                return feasibility
        
        return 0.5  # Default moderate feasibility
    
    def _create_causal_insight(self, relationship: CausalRelationship,
                             intervention_info: Optional[Dict[str, float]]) -> Optional[Insight]:
        """Create an insight from a causal relationship"""
        try:
            # Determine impact level
            impact = "high" if relationship.strength > 0.6 and relationship.confidence > 0.8 else \
                    "medium" if relationship.strength > 0.4 and relationship.confidence > 0.6 else "low"
            
            # Build description
            description = f"Causal analysis reveals that {relationship.cause} has a {relationship.strength:.2f} strength causal effect on {relationship.effect}"
            
            # Build evidence
            evidence = [
                f"Causal strength: {relationship.strength:.3f}",
                f"Statistical confidence: {relationship.confidence:.3f}",
                f"Correlation: {relationship.statistical_tests.get('correlation', 0):.3f}"
            ]
            
            if 'granger_approximation' in relationship.statistical_tests:
                evidence.append(f"Temporal causality: {relationship.statistical_tests['granger_approximation']:.3f}")
            
            # Build recommendations
            recommendations = [
                f"Monitor the relationship between {relationship.cause} and {relationship.effect}",
                "Consider targeted interventions to optimize causal effects"
            ]
            
            if intervention_info:
                estimated_improvement = intervention_info.get('estimated_improvement', 0)
                feasibility = intervention_info.get('intervention_feasibility', 0)
                
                if estimated_improvement > 0.1 and feasibility > 0.5:
                    recommendations.append(
                        f"High-impact intervention: Optimizing {relationship.cause} could improve {relationship.effect} by {estimated_improvement*100:.1f}%"
                    )
                    evidence.append(f"Estimated intervention effect: {estimated_improvement*100:.1f}% improvement")
                    evidence.append(f"Intervention feasibility: {feasibility:.2f}")
            
            return Insight(
                insight_id=f"causal_{relationship.cause}_{relationship.effect}",
                type="causal",
                title=f"Causal Relationship: {relationship.cause} → {relationship.effect}",
                description=description,
                confidence=relationship.confidence,
                impact=impact,
                evidence=evidence,
                recommendations=recommendations,
                metadata={
                    'causal_strength': relationship.strength,
                    'statistical_tests': relationship.statistical_tests,
                    'intervention_analysis': intervention_info
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create causal insight: {e}")
            return None
    
    def _encode_categorical(self, category: str) -> float:
        """Simple categorical encoding for causal analysis"""
        category_map = {
            'clarity': 0.1,
            'specificity': 0.2,
            'completeness': 0.3,
            'actionability': 0.4,
            'context': 0.5,
            'performance': 0.6,
            'system': 0.7,
            'general': 0.8
        }
        return category_map.get(category.lower(), 0.5)
    
    def _estimate_context_complexity(self, context: str) -> float:
        """Estimate context complexity for causal analysis"""
        # Simple heuristic based on context characteristics
        complexity_indicators = ['enterprise', 'large', 'complex', 'multi', 'distributed']
        simplicity_indicators = ['simple', 'basic', 'small', 'single']
        
        context_lower = context.lower()
        complexity_score = 0.5  # Default medium complexity
        
        for indicator in complexity_indicators:
            if indicator in context_lower:
                complexity_score += 0.1
        
        for indicator in simplicity_indicators:
            if indicator in context_lower:
                complexity_score -= 0.1
        
        return max(0.0, min(1.0, complexity_score))