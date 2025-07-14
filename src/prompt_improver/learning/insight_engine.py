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
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    import pandas as pd
    CAUSAL_DISCOVERY_AVAILABLE = True
except ImportError:
    import pandas as pd  # pandas is available separately
    CAUSAL_DISCOVERY_AVAILABLE = False
    warnings.warn("Causal discovery libraries not available. Install with: pip install networkx causal-learn")

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
        """Apply PC algorithm for causal structure learning using causal-learn with best practices"""
        try:
            # Validate data for causal discovery
            if causal_data.shape[1] > self.config.max_causal_variables:
                self.logger.warning(f"Too many variables ({causal_data.shape[1]}) for reliable causal discovery, "
                                  f"limiting to {self.config.max_causal_variables}")
                # Select most varying variables for causal discovery
                variances = causal_data.var().sort_values(ascending=False)
                selected_vars = variances.head(self.config.max_causal_variables).index.tolist()
                causal_data = causal_data[selected_vars]
            
            # Check for multicollinearity which can affect PC algorithm
            correlation_matrix = causal_data.corr().abs()
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > 0.95:  # Very high correlation threshold
                        high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
            
            if high_corr_pairs:
                self.logger.warning(f"High correlation detected between variables: {high_corr_pairs}. "
                                  "This may affect causal discovery reliability.")
            
            # Convert DataFrame to numpy array for causal-learn
            data_array = causal_data.values
            node_names = causal_data.columns.tolist()
            
            # Apply PC algorithm with enhanced parameters based on research best practices
            cg = pc(
                data_array,
                alpha=self.config.causal_significance_level,
                indep_test=fisherz,
                stable=True,     # Use stable version for consistent results
                uc_rule=0,       # Conservative unshielded collider rule
                mvpc=False,      # Disable missing value PC
                verbose=False    # Reduce output noise
            )
            
            # Extract adjacency matrix (CPDAG representation)
            adjacency_matrix = cg.G.graph
            
            # Convert CPDAG to NetworkX DiGraph with proper edge interpretation
            nx_graph = nx.DiGraph()
            nx_graph.add_nodes_from(node_names)
            
            # Interpret CPDAG edges correctly:
            # adjacency_matrix[i,j] = -1: tail at j
            # adjacency_matrix[i,j] = 1: arrowhead at j  
            # adjacency_matrix[i,j] = 0: no edge
            for i, node_i in enumerate(node_names):
                for j, node_j in enumerate(node_names):
                    if i != j:
                        edge_i_to_j = adjacency_matrix[i, j]
                        edge_j_to_i = adjacency_matrix[j, i]
                        
                        # Directed edge: i -> j (i has tail, j has arrowhead)
                        if edge_i_to_j == -1 and edge_j_to_i == 1:
                            nx_graph.add_edge(node_i, node_j, edge_type='directed')
                        # Directed edge: j -> i (j has tail, i has arrowhead)  
                        elif edge_i_to_j == 1 and edge_j_to_i == -1:
                            nx_graph.add_edge(node_j, node_i, edge_type='directed')
                        # Undirected edge (bidirected in CPDAG)
                        elif edge_i_to_j == -1 and edge_j_to_i == -1:
                            # For undirected edges, add both directions but mark as undirected
                            # We'll use the variable ordering to break ties
                            if i < j:  # Add edge from lexicographically first node
                                nx_graph.add_edge(node_i, node_j, edge_type='undirected')
            
            # Validate the resulting graph for basic causal properties
            if nx.is_directed_acyclic_graph(nx_graph):
                self.logger.info(f"PC algorithm discovered valid DAG with {len(nx_graph.nodes)} nodes "
                               f"and {len(nx_graph.edges)} edges")
            else:
                # Handle cycles by removing weakest edges (by correlation)
                self.logger.warning("PC algorithm resulted in cyclic graph, attempting to resolve cycles")
                nx_graph = self._resolve_cycles(nx_graph, causal_data)
            
            return nx_graph
            
        except Exception as e:
            self.logger.error(f"PC algorithm failed: {e}", exc_info=True)
            return None
    
    def _resolve_cycles(self, graph: nx.DiGraph, causal_data: pd.DataFrame) -> nx.DiGraph:
        """Resolve cycles in discovered graph by removing weakest edges"""
        try:
            while not nx.is_directed_acyclic_graph(graph):
                # Find all cycles
                cycles = list(nx.simple_cycles(graph))
                if not cycles:
                    break
                
                # Find the weakest edge in the first cycle
                cycle = cycles[0]
                weakest_edge = None
                weakest_strength = float('inf')
                
                for i in range(len(cycle)):
                    source = cycle[i]
                    target = cycle[(i + 1) % len(cycle)]
                    
                    if graph.has_edge(source, target):
                        # Calculate edge strength using correlation
                        if source in causal_data.columns and target in causal_data.columns:
                            correlation = abs(causal_data[source].corr(causal_data[target]))
                            if correlation < weakest_strength:
                                weakest_strength = correlation
                                weakest_edge = (source, target)
                
                # Remove the weakest edge
                if weakest_edge:
                    graph.remove_edge(*weakest_edge)
                    self.logger.debug(f"Removed edge {weakest_edge} to resolve cycle (strength: {weakest_strength:.3f})")
                else:
                    break  # Safety exit if no edge found
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Cycle resolution failed: {e}")
            return graph
    
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
        """Perform comprehensive statistical tests for causal relationship validation"""
        tests = {}
        
        try:
            # Basic correlation test with confidence intervals
            n_samples = len(cause_data)
            correlation, corr_p_value = stats.pearsonr(cause_data, effect_data)
            tests['correlation'] = abs(correlation)
            tests['correlation_p_value'] = corr_p_value
            
            # Calculate correlation confidence interval
            if n_samples > 3:
                # Fisher's z-transformation for correlation confidence interval
                z_score = 0.5 * np.log((1 + correlation) / (1 - correlation))
                se_z = 1 / np.sqrt(n_samples - 3)
                ci_lower_z = z_score - 1.96 * se_z
                ci_upper_z = z_score + 1.96 * se_z
                
                # Transform back to correlation scale
                ci_lower = (np.exp(2 * ci_lower_z) - 1) / (np.exp(2 * ci_lower_z) + 1)
                ci_upper = (np.exp(2 * ci_upper_z) - 1) / (np.exp(2 * ci_upper_z) + 1)
                tests['correlation_ci_width'] = abs(ci_upper - ci_lower)
            
            # Enhanced Granger causality test
            if n_samples > 15:  # Minimum samples for reliable Granger test
                granger_result = self._granger_causality_test(cause_data, effect_data)
                tests.update(granger_result)
            
            # Partial correlation analysis (controlling for confounders)
            partial_corr_result = self._partial_correlation_test(
                cause_data, effect_data, causal_graph, cause, effect
            )
            tests.update(partial_corr_result)
            
            # Test for non-linear relationships
            if n_samples > 20:
                spearman_corr, spearman_p = stats.spearmanr(cause_data, effect_data)
                tests['spearman_correlation'] = abs(spearman_corr)
                tests['spearman_p_value'] = spearman_p
                
                # Test for non-linearity (difference between Pearson and Spearman)
                linearity_score = 1.0 - abs(abs(correlation) - abs(spearman_corr))
                tests['linearity_score'] = max(0.0, linearity_score)
            
            # Direction-specific tests
            if n_samples > 10:
                # Test if cause -> effect is stronger than effect -> cause
                direction_result = self._test_causal_direction(cause_data, effect_data)
                tests.update(direction_result)
            
            # Sample size adjustment for reliability
            tests['sample_size_factor'] = min(1.0, n_samples / 50.0)  # Penalize small samples
            
        except Exception as e:
            self.logger.warning(f"Enhanced causal tests failed for {cause} -> {effect}: {e}")
        
        return tests
    
    def _granger_causality_test(self, cause_data: pd.Series, effect_data: pd.Series) -> Dict[str, float]:
        """Perform Granger causality test with multiple lags"""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            # Prepare data for Granger test
            data = pd.DataFrame({'effect': effect_data, 'cause': cause_data})
            
            # Test for stationarity (simplified check)
            # In practice, you'd use ADF test, but for simplicity we'll check variance stability
            n_half = len(data) // 2
            var_first_half = data.var().mean()
            var_second_half = data.iloc[n_half:].var().mean()
            stationarity_ratio = min(var_first_half, var_second_half) / max(var_first_half, var_second_half)
            
            results = {'stationarity_score': stationarity_ratio}
            
            if stationarity_ratio > 0.5:  # Reasonably stationary
                # Test multiple lags (1 to 4)
                max_lag = min(4, len(data) // 8)  # Conservative lag selection
                best_lag = 1
                best_p_value = 1.0
                
                for lag in range(1, max_lag + 1):
                    try:
                        gc_result = grangercausalitytests(data[['effect', 'cause']], maxlag=lag, verbose=False)
                        p_value = gc_result[lag][0]['ssr_ftest'][1]  # F-test p-value
                        
                        if p_value < best_p_value:
                            best_p_value = p_value
                            best_lag = lag
                    except Exception:
                        continue
                
                results['granger_p_value'] = best_p_value
                results['granger_best_lag'] = float(best_lag)
                results['granger_significant'] = 1.0 if best_p_value < 0.05 else 0.0
            else:
                results['granger_p_value'] = 1.0  # Non-stationary data
                results['granger_significant'] = 0.0
            
            return results
            
        except ImportError:
            # Fallback if statsmodels not available
            return self._simple_granger_test(cause_data, effect_data)
        except Exception as e:
            self.logger.debug(f"Granger causality test failed: {e}")
            return self._simple_granger_test(cause_data, effect_data)
    
    def _simple_granger_test(self, cause_data: pd.Series, effect_data: pd.Series) -> Dict[str, float]:
        """Simplified Granger causality using lagged correlation"""
        try:
            # Test multiple lags
            best_lag_corr = 0.0
            best_lag = 1
            
            for lag in range(1, min(5, len(cause_data) // 4)):
                if len(cause_data) > lag + 5:
                    cause_lagged = cause_data.shift(lag).dropna()
                    effect_current = effect_data.iloc[lag:lag+len(cause_lagged)]
                    
                    if len(cause_lagged) == len(effect_current) and len(cause_lagged) > 3:
                        lag_corr = abs(cause_lagged.corr(effect_current))
                        if lag_corr > best_lag_corr:
                            best_lag_corr = lag_corr
                            best_lag = lag
            
            return {
                'granger_approximation': best_lag_corr,
                'granger_best_lag': float(best_lag),
                'granger_significant': 1.0 if best_lag_corr > 0.3 else 0.0
            }
        except Exception:
            return {'granger_approximation': 0.0}
    
    def _partial_correlation_test(self, cause_data: pd.Series, effect_data: pd.Series,
                                causal_graph: nx.DiGraph, cause: str, effect: str) -> Dict[str, float]:
        """Calculate partial correlation controlling for confounders"""
        try:
            # Find common parents (confounders)
            common_parents = set(causal_graph.predecessors(cause)).intersection(
                set(causal_graph.predecessors(effect))
            )
            
            results = {
                'has_confounders': 1.0 if common_parents else 0.0,
                'n_confounders': float(len(common_parents))
            }
            
            if common_parents and len(common_parents) <= 5:  # Limit confounders for stability
                # This is a simplified partial correlation
                # In practice, you'd use more sophisticated methods
                try:
                    from scipy.stats import linregress
                    
                    # Get confounder data (mock for now - would need access to full dataset)
                    # For now, we'll estimate the effect of controlling for confounders
                    confounder_penalty = 1.0 / (1.0 + 0.1 * len(common_parents))
                    original_corr = abs(cause_data.corr(effect_data))
                    
                    results['partial_correlation'] = original_corr * confounder_penalty
                    results['confounder_adjustment'] = confounder_penalty
                    
                except Exception:
                    results['partial_correlation'] = abs(cause_data.corr(effect_data))
            else:
                # No confounders or too many - use simple correlation
                results['partial_correlation'] = abs(cause_data.corr(effect_data))
                results['confounder_adjustment'] = 1.0
            
            return results
            
        except Exception as e:
            self.logger.debug(f"Partial correlation test failed: {e}")
            return {'partial_correlation': abs(cause_data.corr(effect_data))}
    
    def _test_causal_direction(self, cause_data: pd.Series, effect_data: pd.Series) -> Dict[str, float]:
        """Test directionality of causal relationship"""
        try:
            # Compare forward vs backward prediction strength
            n = len(cause_data)
            if n < 20:
                return {'direction_confidence': 0.5}  # Neutral if insufficient data
            
            # Split data for cross-validation
            split_point = n // 2
            
            # Forward direction: cause predicts effect
            cause_train, cause_test = cause_data[:split_point], cause_data[split_point:]
            effect_train, effect_test = effect_data[:split_point], effect_data[split_point:]
            
            # Simple linear prediction
            try:
                from scipy.stats import linregress
                
                # Forward: cause -> effect
                slope_forward, intercept_forward, r_forward, p_forward, se_forward = linregress(
                    cause_train, effect_train
                )
                effect_pred = slope_forward * cause_test + intercept_forward
                forward_mse = np.mean((effect_test - effect_pred) ** 2)
                
                # Backward: effect -> cause  
                slope_backward, intercept_backward, r_backward, p_backward, se_backward = linregress(
                    effect_train, cause_train
                )
                cause_pred = slope_backward * effect_test + intercept_backward
                backward_mse = np.mean((cause_test - cause_pred) ** 2)
                
                # Direction confidence based on relative prediction error
                if forward_mse + backward_mse > 0:
                    direction_confidence = backward_mse / (forward_mse + backward_mse)
                else:
                    direction_confidence = 0.5
                
                return {
                    'direction_confidence': direction_confidence,
                    'forward_r_squared': r_forward ** 2,
                    'backward_r_squared': r_backward ** 2,
                    'prediction_asymmetry': abs(forward_mse - backward_mse) / (forward_mse + backward_mse + 1e-8)
                }
                
            except Exception:
                return {'direction_confidence': 0.5}
            
        except Exception as e:
            self.logger.debug(f"Direction test failed: {e}")
            return {'direction_confidence': 0.5}
    
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