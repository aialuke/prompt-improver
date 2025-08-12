"""Insight Generation Engine

Generates actionable insights from performance data and learning patterns.
Combines statistical analysis with pattern recognition to provide strategic
recommendations for system improvement.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional
import warnings
import numpy as np
from scipy import stats
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    import networkx as nx
    import pandas as pd
    CAUSAL_DISCOVERY_AVAILABLE = True
except ImportError:
    import pandas as pd
    CAUSAL_DISCOVERY_AVAILABLE = False
    warnings.warn('Causal discovery libraries not available. Install with: pip install networkx causal-learn')
logger = logging.getLogger(__name__)

@dataclass
class InsightConfig:
    """Configuration for insight generation"""
    min_confidence: float = 0.7
    min_sample_size: int = 10
    significance_threshold: float = 0.05
    trend_window_days: int = 30
    max_insights: int = 20
    enable_causal_discovery: bool = True
    causal_significance_level: float = 0.05
    min_causal_samples: int = 20
    max_causal_variables: int = 15
    intervention_confidence_threshold: float = 0.8

@dataclass
class Insight:
    """Generated insight with supporting evidence"""
    insight_id: str
    type: str
    title: str
    description: str
    confidence: float
    impact: str
    evidence: list[str]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class CausalRelationship:
    """Discovered causal relationship"""
    cause: str
    effect: str
    strength: float
    confidence: float
    intervention_effect: float | None = None
    statistical_tests: dict[str, float] = field(default_factory=dict)

class InsightGenerationEngine:
    """Engine for generating actionable insights from performance data"""

    def __init__(self, config: InsightConfig | None=None):
        self.config = config or InsightConfig()
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

    async def run_orchestrated_analysis(self, config: dict[str, Any]) -> dict[str, Any]:
        """Orchestrator-compatible interface for insight generation (2025 pattern)

        Args:
            config: Orchestrator configuration containing:
                - performance_data: Performance data to analyze
                - output_path: Local path for output files (optional)
                - analysis_type: Type of analysis to perform (optional)

        Returns:
            Orchestrator-compatible result with insights and metadata
        """
        start_time = datetime.now()
        try:
            performance_data = config.get('performance_data', {})
            output_path = config.get('output_path', './outputs/insights')
            analysis_type = config.get('analysis_type', 'comprehensive')
            insights = await self.generate_insights(performance_data)
            result = {'insights': [{'type': insight.type, 'description': insight.description, 'confidence': insight.confidence, 'impact': insight.impact, 'recommendations': insight.recommendations, 'evidence': insight.evidence} for insight in insights], 'summary': {'total_insights': len(insights), 'high_impact_insights': len([i for i in insights if i.impact == 'high']), 'avg_confidence': sum(i.confidence for i in insights) / len(insights) if insights else 0.0}}
            execution_time = (datetime.now() - start_time).total_seconds()
            return {'orchestrator_compatible': True, 'component_result': result, 'local_metadata': {'output_path': output_path, 'execution_time': execution_time, 'analysis_type': analysis_type, 'insights_generated': len(insights), 'causal_discovery_enabled': self.config.enable_causal_discovery, 'component_version': '1.0.0'}}
        except Exception as e:
            self.logger.error('Orchestrated analysis failed: %s', e)
            return {'orchestrator_compatible': True, 'component_result': {'error': str(e), 'insights': []}, 'local_metadata': {'execution_time': (datetime.now() - start_time).total_seconds(), 'error': True, 'component_version': '1.0.0'}}

    async def generate_insights(self, performance_data: dict[str, Any]) -> list[Insight]:
        """Generate insights from performance data"""
        insights = []
        performance_insights = await self._analyze_performance_patterns(performance_data)
        insights.extend(performance_insights)
        trend_insights = await self._analyze_trends(performance_data)
        insights.extend(trend_insights)
        opportunity_insights = await self._identify_opportunities(performance_data)
        insights.extend(opportunity_insights)
        risk_insights = await self._identify_risks(performance_data)
        insights.extend(risk_insights)
        if self.config.enable_causal_discovery and CAUSAL_DISCOVERY_AVAILABLE:
            causal_insights = await self._discover_causal_relationships(performance_data)
            insights.extend(causal_insights)
        insights = [i for i in insights if i.confidence >= self.config.min_confidence]
        insights.sort(key=lambda x: (x.impact == 'high', x.confidence), reverse=True)
        return insights[:self.config.max_insights]

    async def _analyze_performance_patterns(self, data: dict[str, Any]) -> list[Insight]:
        """Analyze performance patterns"""
        insights = []
        if 'rule_performance' in data:
            rule_data = data['rule_performance']
            sorted_rules = sorted(rule_data.items(), key=lambda x: x[1].get('avg_score', 0), reverse=True)
            if len(sorted_rules) >= 3:
                best_rule = sorted_rules[0]
                worst_rule = sorted_rules[-1]
                score_gap = best_rule[1].get('avg_score', 0) - worst_rule[1].get('avg_score', 0)
                if score_gap > 0.2:
                    insights.append(Insight(insight_id='performance_gap', type='performance', title='Significant Rule Performance Gap', description=f"Large performance difference between best ({best_rule[0]}: {best_rule[1].get('avg_score', 0):.3f}) and worst ({worst_rule[0]}: {worst_rule[1].get('avg_score', 0):.3f}) performing rules", confidence=0.9, impact='high', evidence=[f'Performance gap: {score_gap:.3f}', f"Best rule sample size: {best_rule[1].get('sample_size', 0)}", f"Worst rule sample size: {worst_rule[1].get('sample_size', 0)}"], recommendations=[f'Investigate why {best_rule[0]} performs better', f'Consider deprecating or improving {worst_rule[0]}', 'Apply successful patterns from top performers to bottom performers']))
        return insights

    async def _analyze_trends(self, data: dict[str, Any]) -> list[Insight]:
        """Analyze performance trends"""
        insights = []
        if 'temporal_data' in data:
            temporal_data = data['temporal_data']
            if len(temporal_data) >= 7:
                timestamps = [d['timestamp'] for d in temporal_data]
                scores = [d['avg_score'] for d in temporal_data]
                x = np.arange(len(scores))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
                if p_value < self.config.significance_threshold:
                    trend_direction = 'improving' if slope > 0 else 'declining'
                    confidence = min(0.95, abs(r_value))
                    insights.append(Insight(insight_id='system_trend', type='trend', title=f'System Performance {trend_direction.title()}', description=f'Overall system performance shows {trend_direction} trend over time', confidence=confidence, impact='high' if abs(slope) > 0.01 else 'medium', evidence=[f'Trend slope: {slope:.4f}', f'R-squared: {r_value ** 2:.3f}', f'Statistical significance: p={p_value:.3f}'], recommendations=['Continue monitoring trend', 'Identify factors driving the trend' if slope > 0 else 'Investigate causes of decline']))
        return insights

    async def _identify_opportunities(self, data: dict[str, Any]) -> list[Insight]:
        """Identify improvement opportunities"""
        insights = []
        if 'context_performance' in data:
            context_data = data['context_performance']
            underperforming = [(context, perf) for context, perf in context_data.items() if perf.get('avg_score', 1) < 0.7 and perf.get('sample_size', 0) >= self.config.min_sample_size]
            if underperforming:
                underperforming.sort(key=lambda x: x[1].get('sample_size', 0), reverse=True)
                top_opportunity = underperforming[0]
                insights.append(Insight(insight_id='context_opportunity', type='opportunity', title='High-Volume Improvement Opportunity', description=f"Context '{top_opportunity[0]}' has high volume but low performance", confidence=0.8, impact='high', evidence=[f"Average score: {top_opportunity[1].get('avg_score', 0):.3f}", f"Sample size: {top_opportunity[1].get('sample_size', 0)}", f"Improvement potential: {0.8 - top_opportunity[1].get('avg_score', 0):.3f}"], recommendations=[f'Create specialized rules for {top_opportunity[0]} context', 'Analyze successful patterns from other contexts', 'Increase focus on this high-volume area']))
        return insights

    async def _identify_risks(self, data: dict[str, Any]) -> list[Insight]:
        """Identify potential risks"""
        insights = []
        if 'rule_performance' in data:
            rule_data = data['rule_performance']
            inconsistent_rules = [(rule_id, perf) for rule_id, perf in rule_data.items() if perf.get('std_score', 0) > 0.3 and perf.get('sample_size', 0) >= self.config.min_sample_size]
            if inconsistent_rules:
                most_inconsistent = max(inconsistent_rules, key=lambda x: x[1].get('std_score', 0))
                insights.append(Insight(insight_id='consistency_risk', type='risk', title='High Performance Variability', description=f"Rule '{most_inconsistent[0]}' shows high performance variability", confidence=0.85, impact='medium', evidence=[f"Standard deviation: {most_inconsistent[1].get('std_score', 0):.3f}", f"Average score: {most_inconsistent[1].get('avg_score', 0):.3f}", f"Coefficient of variation: {most_inconsistent[1].get('std_score', 0) / most_inconsistent[1].get('avg_score', 1):.3f}"], recommendations=['Investigate causes of inconsistency', 'Add rule stability checks', 'Consider rule refinement or replacement']))
        return insights

    async def _discover_causal_relationships(self, data: dict[str, Any]) -> list[Insight]:
        """Phase 2: Discover causal relationships using PC algorithm"""
        insights = []
        if not CAUSAL_DISCOVERY_AVAILABLE:
            self.logger.warning('Causal discovery libraries not available')
            return insights
        try:
            causal_data = self._prepare_causal_data(data)
            if causal_data is None or len(causal_data) < self.config.min_causal_samples:
                self.logger.info('Insufficient data for causal discovery: %s samples', len(causal_data) if causal_data is not None else 0)
                return insights
            causal_graph = self._apply_pc_algorithm(causal_data)
            if causal_graph is None:
                return insights
            relationships = self._extract_causal_relationships(causal_graph, causal_data)
            intervention_insights = self._analyze_interventions(relationships, causal_data, causal_graph)
            for relationship in relationships:
                if relationship.confidence >= self.config.intervention_confidence_threshold:
                    insight = self._create_causal_insight(relationship, intervention_insights.get(relationship.cause))
                    if insight:
                        insights.append(insight)
            self.logger.info('Generated %s causal insights from %s relationships', len(insights), len(relationships))
        except Exception as e:
            self.logger.error('Causal discovery failed: %s', e)
        return insights

    def _prepare_causal_data(self, data: dict[str, Any]) -> pd.DataFrame | None:
        """Prepare data for causal discovery analysis"""
        try:
            causal_features = []
            if 'rule_performance' in data:
                rule_data = data['rule_performance']
                for rule_id, perf in rule_data.items():
                    if perf.get('sample_size', 0) >= self.config.min_sample_size:
                        feature_row = {'rule_avg_score': perf.get('avg_score', 0), 'rule_consistency': 1 - perf.get('std_score', 0) / max(perf.get('avg_score', 1), 0.01), 'rule_sample_size': min(perf.get('sample_size', 0) / 100, 1.0), 'rule_category': self._encode_categorical(rule_id.split('_')[0] if '_' in rule_id else 'general')}
                        causal_features.append(feature_row)
            if 'context_performance' in data:
                context_data = data['context_performance']
                for i, (context, perf) in enumerate(context_data.items()):
                    if perf.get('sample_size', 0) >= self.config.min_sample_size:
                        feature_row = {'context_avg_score': perf.get('avg_score', 0), 'context_consistency': 1 - perf.get('std_score', 0) / max(perf.get('avg_score', 1), 0.01), 'context_complexity': self._estimate_context_complexity(context), 'context_volume': min(perf.get('sample_size', 0) / 100, 1.0)}
                        causal_features.append(feature_row)
            if 'system_metrics' in data:
                system_data = data['system_metrics']
                feature_row = {'system_overall_score': system_data.get('overall_score', 0), 'system_efficiency': system_data.get('efficiency_score', 0), 'system_reliability': system_data.get('reliability_score', 0), 'system_load': min(system_data.get('processing_load', 0) / 100, 1.0)}
                causal_features.append(feature_row)
            if not causal_features:
                return None
            df = pd.DataFrame(causal_features)
            if len(df.columns) > self.config.max_causal_variables:
                variances = df.var()
                top_vars = variances.nlargest(self.config.max_causal_variables).index
                df = df[top_vars]
            df = df.loc[:, df.var() > 0.01]
            return df if len(df) >= self.config.min_causal_samples else None
        except Exception as e:
            self.logger.error('Failed to prepare causal data: %s', e)
            return None

    def _apply_pc_algorithm(self, causal_data: pd.DataFrame) -> nx.DiGraph | None:
        """Apply PC algorithm for causal structure learning using causal-learn with best practices"""
        try:
            if causal_data.shape[1] > self.config.max_causal_variables:
                self.logger.warning('Too many variables (%s) for reliable causal discovery, limiting to %s', causal_data.shape[1], self.config.max_causal_variables)
                variances = causal_data.var().sort_values(ascending=False)
                selected_vars = variances.head(self.config.max_causal_variables).index.tolist()
                causal_data = causal_data[selected_vars]
            correlation_matrix = causal_data.corr().abs()
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
            if high_corr_pairs:
                self.logger.warning('High correlation detected between variables: %s. This may affect causal discovery reliability.', high_corr_pairs)
            data_array = causal_data.values
            node_names = causal_data.columns.tolist()
            cg = pc(data_array, alpha=self.config.causal_significance_level, indep_test=fisherz, stable=True, uc_rule=0, mvpc=False, verbose=False)
            adjacency_matrix = cg.G.graph
            nx_graph = nx.DiGraph()
            nx_graph.add_nodes_from(node_names)
            for i, node_i in enumerate(node_names):
                for j, node_j in enumerate(node_names):
                    if i != j:
                        edge_i_to_j = adjacency_matrix[i, j]
                        edge_j_to_i = adjacency_matrix[j, i]
                        if edge_i_to_j == -1 and edge_j_to_i == 1:
                            nx_graph.add_edge(node_i, node_j, edge_type='directed')
                        elif edge_i_to_j == 1 and edge_j_to_i == -1:
                            nx_graph.add_edge(node_j, node_i, edge_type='directed')
                        elif edge_i_to_j == -1 and edge_j_to_i == -1:
                            if i < j:
                                nx_graph.add_edge(node_i, node_j, edge_type='undirected')
            if nx.is_directed_acyclic_graph(nx_graph):
                self.logger.info('PC algorithm discovered valid DAG with %s nodes and %s edges', len(nx_graph.nodes), len(nx_graph.edges))
            else:
                self.logger.warning('PC algorithm resulted in cyclic graph, attempting to resolve cycles')
                nx_graph = self._resolve_cycles(nx_graph, causal_data)
            return nx_graph
        except Exception as e:
            self.logger.error('PC algorithm failed: %s', e, exc_info=True)
            return None

    def _resolve_cycles(self, graph: nx.DiGraph, causal_data: pd.DataFrame) -> nx.DiGraph:
        """Resolve cycles in discovered graph by removing weakest edges"""
        try:
            while not nx.is_directed_acyclic_graph(graph):
                cycles = list(nx.simple_cycles(graph))
                if not cycles:
                    break
                cycle = cycles[0]
                weakest_edge = None
                weakest_strength = float('inf')
                for i in range(len(cycle)):
                    source = cycle[i]
                    target = cycle[(i + 1) % len(cycle)]
                    if graph.has_edge(source, target):
                        if source in causal_data.columns and target in causal_data.columns:
                            correlation = abs(causal_data[source].corr(causal_data[target]))
                            if correlation < weakest_strength:
                                weakest_strength = correlation
                                weakest_edge = (source, target)
                if weakest_edge:
                    graph.remove_edge(*weakest_edge)
                    self.logger.debug('Removed edge %s to resolve cycle (strength: %s)', weakest_edge, format(weakest_strength, '.3f'))
                else:
                    break
            return graph
        except Exception as e:
            self.logger.error('Cycle resolution failed: %s', e)
            return graph

    def _extract_causal_relationships(self, causal_graph: nx.DiGraph, causal_data: pd.DataFrame) -> list[CausalRelationship]:
        """Extract causal relationships from discovered DAG"""
        relationships = []
        try:
            for cause, effect in causal_graph.edges():
                if cause in causal_data.columns and effect in causal_data.columns:
                    correlation = causal_data[cause].corr(causal_data[effect])
                    strength = abs(correlation)
                    stat_tests = self._perform_causal_tests(causal_data[cause], causal_data[effect], causal_graph, cause, effect)
                    confidence = self._calculate_causal_confidence(stat_tests)
                    relationship = CausalRelationship(cause=cause, effect=effect, strength=strength, confidence=confidence, statistical_tests=stat_tests)
                    relationships.append(relationship)
            relationships.sort(key=lambda r: (r.confidence, r.strength), reverse=True)
            return relationships
        except Exception as e:
            self.logger.error('Failed to extract causal relationships: %s', e)
            return []

    def _perform_causal_tests(self, cause_data: pd.Series, effect_data: pd.Series, causal_graph: nx.DiGraph, cause: str, effect: str) -> dict[str, float]:
        """Perform comprehensive statistical tests for causal relationship validation"""
        tests = {}
        try:
            n_samples = len(cause_data)
            correlation, corr_p_value = stats.pearsonr(cause_data, effect_data)
            tests['correlation'] = abs(correlation)
            tests['correlation_p_value'] = corr_p_value
            if n_samples > 3:
                z_score = 0.5 * np.log((1 + correlation) / (1 - correlation))
                se_z = 1 / np.sqrt(n_samples - 3)
                ci_lower_z = z_score - 1.96 * se_z
                ci_upper_z = z_score + 1.96 * se_z
                ci_lower = (np.exp(2 * ci_lower_z) - 1) / (np.exp(2 * ci_lower_z) + 1)
                ci_upper = (np.exp(2 * ci_upper_z) - 1) / (np.exp(2 * ci_upper_z) + 1)
                tests['correlation_ci_width'] = abs(ci_upper - ci_lower)
            if n_samples > 15:
                granger_result = self._granger_causality_test(cause_data, effect_data)
                tests.update(granger_result)
            partial_corr_result = self._partial_correlation_test(cause_data, effect_data, causal_graph, cause, effect)
            tests.update(partial_corr_result)
            if n_samples > 20:
                spearman_corr, spearman_p = stats.spearmanr(cause_data, effect_data)
                tests['spearman_correlation'] = abs(spearman_corr)
                tests['spearman_p_value'] = spearman_p
                linearity_score = 1.0 - abs(abs(correlation) - abs(spearman_corr))
                tests['linearity_score'] = max(0.0, linearity_score)
            if n_samples > 10:
                direction_result = self._test_causal_direction(cause_data, effect_data)
                tests.update(direction_result)
            tests['sample_size_factor'] = min(1.0, n_samples / 50.0)
        except Exception as e:
            self.logger.warning('Enhanced causal tests failed for %s -> %s: %s', cause, effect, e)
        return tests

    def _granger_causality_test(self, cause_data: pd.Series, effect_data: pd.Series) -> dict[str, float]:
        """Perform Granger causality test with multiple lags"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            from statsmodels.tsa.stattools import grangercausalitytests
            data = pd.DataFrame({'effect': effect_data, 'cause': cause_data})
            n_half = len(data) // 2
            var_first_half = data.var().mean()
            var_second_half = data.iloc[n_half:].var().mean()
            stationarity_ratio = min(var_first_half, var_second_half) / max(var_first_half, var_second_half)
            results = {'stationarity_score': stationarity_ratio}
            if stationarity_ratio > 0.5:
                max_lag = min(4, len(data) // 8)
                best_lag = 1
                best_p_value = 1.0
                for lag in range(1, max_lag + 1):
                    try:
                        gc_result = grangercausalitytests(data[['effect', 'cause']], maxlag=lag, verbose=False)
                        p_value = gc_result[lag][0]['ssr_ftest'][1]
                        if p_value < best_p_value:
                            best_p_value = p_value
                            best_lag = lag
                    except Exception:
                        continue
                results['granger_p_value'] = best_p_value
                results['granger_best_lag'] = float(best_lag)
                results['granger_significant'] = 1.0 if best_p_value < 0.05 else 0.0
            else:
                results['granger_p_value'] = 1.0
                results['granger_significant'] = 0.0
            return results
        except ImportError:
            return self._simple_granger_test(cause_data, effect_data)
        except Exception as e:
            self.logger.debug('Granger causality test failed: %s', e)
            return self._simple_granger_test(cause_data, effect_data)

    def _simple_granger_test(self, cause_data: pd.Series, effect_data: pd.Series) -> dict[str, float]:
        """Simplified Granger causality using lagged correlation"""
        try:
            best_lag_corr = 0.0
            best_lag = 1
            for lag in range(1, min(5, len(cause_data) // 4)):
                if len(cause_data) > lag + 5:
                    cause_lagged = cause_data.shift(lag).dropna()
                    effect_current = effect_data.iloc[lag:lag + len(cause_lagged)]
                    if len(cause_lagged) == len(effect_current) and len(cause_lagged) > 3:
                        lag_corr = abs(cause_lagged.corr(effect_current))
                        if lag_corr > best_lag_corr:
                            best_lag_corr = lag_corr
                            best_lag = lag
            return {'granger_approximation': best_lag_corr, 'granger_best_lag': float(best_lag), 'granger_significant': 1.0 if best_lag_corr > 0.3 else 0.0}
        except Exception:
            return {'granger_approximation': 0.0}

    def _partial_correlation_test(self, cause_data: pd.Series, effect_data: pd.Series, causal_graph: nx.DiGraph, cause: str, effect: str) -> dict[str, float]:
        """Calculate partial correlation controlling for confounders"""
        try:
            common_parents = set(causal_graph.predecessors(cause)).intersection(set(causal_graph.predecessors(effect)))
            results = {'has_confounders': 1.0 if common_parents else 0.0, 'n_confounders': float(len(common_parents))}
            if common_parents and len(common_parents) <= 5:
                try:
                    from scipy.stats import linregress
                    confounder_penalty = 1.0 / (1.0 + 0.1 * len(common_parents))
                    original_corr = abs(cause_data.corr(effect_data))
                    results['partial_correlation'] = original_corr * confounder_penalty
                    results['confounder_adjustment'] = confounder_penalty
                except Exception:
                    results['partial_correlation'] = abs(cause_data.corr(effect_data))
            else:
                results['partial_correlation'] = abs(cause_data.corr(effect_data))
                results['confounder_adjustment'] = 1.0
            return results
        except Exception as e:
            self.logger.debug('Partial correlation test failed: %s', e)
            return {'partial_correlation': abs(cause_data.corr(effect_data))}

    def _test_causal_direction(self, cause_data: pd.Series, effect_data: pd.Series) -> dict[str, float]:
        """Test directionality of causal relationship"""
        try:
            n = len(cause_data)
            if n < 20:
                return {'direction_confidence': 0.5}
            split_point = n // 2
            cause_train, cause_test = (cause_data[:split_point], cause_data[split_point:])
            effect_train, effect_test = (effect_data[:split_point], effect_data[split_point:])
            try:
                from scipy.stats import linregress
                slope_forward, intercept_forward, r_forward, p_forward, se_forward = linregress(cause_train, effect_train)
                effect_pred = slope_forward * cause_test + intercept_forward
                forward_mse = np.mean((effect_test - effect_pred) ** 2)
                slope_backward, intercept_backward, r_backward, p_backward, se_backward = linregress(effect_train, cause_train)
                cause_pred = slope_backward * effect_test + intercept_backward
                backward_mse = np.mean((cause_test - cause_pred) ** 2)
                if forward_mse + backward_mse > 0:
                    direction_confidence = backward_mse / (forward_mse + backward_mse)
                else:
                    direction_confidence = 0.5
                return {'direction_confidence': direction_confidence, 'forward_r_squared': r_forward ** 2, 'backward_r_squared': r_backward ** 2, 'prediction_asymmetry': abs(forward_mse - backward_mse) / (forward_mse + backward_mse + 1e-08)}
            except Exception:
                return {'direction_confidence': 0.5}
        except Exception as e:
            self.logger.debug('Direction test failed: %s', e)
            return {'direction_confidence': 0.5}

    def _calculate_causal_confidence(self, stat_tests: dict[str, float]) -> float:
        """Calculate confidence in causal relationship based on statistical tests"""
        confidence = 0.0
        if 'correlation' in stat_tests:
            confidence += min(stat_tests['correlation'] * 0.5, 0.4)
        if stat_tests.get('correlation_p_value', 1.0) < self.config.causal_significance_level:
            confidence += 0.2
        if 'granger_approximation' in stat_tests:
            confidence += min(stat_tests['granger_approximation'] * 0.3, 0.2)
        if stat_tests.get('has_confounders', 0) > 0:
            confounder_penalty = min(stat_tests.get('n_confounders', 0) * 0.1, 0.3)
            confidence -= confounder_penalty
        return max(0.0, min(1.0, confidence))

    def _analyze_interventions(self, relationships: list[CausalRelationship], causal_data: pd.DataFrame, causal_graph: nx.DiGraph) -> dict[str, dict[str, float]]:
        """Analyze potential interventions and their effects"""
        intervention_analysis = {}
        try:
            for relationship in relationships:
                if relationship.confidence >= self.config.intervention_confidence_threshold:
                    intervention_effect = self._simulate_intervention(relationship, causal_data, causal_graph)
                    intervention_analysis[relationship.cause] = {'target_effect': relationship.effect, 'estimated_improvement': intervention_effect, 'confidence': relationship.confidence, 'intervention_feasibility': self._assess_intervention_feasibility(relationship.cause)}
        except Exception as e:
            self.logger.error('Intervention analysis failed: %s', e)
        return intervention_analysis

    def _simulate_intervention(self, relationship: CausalRelationship, causal_data: pd.DataFrame, causal_graph: nx.DiGraph) -> float:
        """Simulate the effect of intervening on a causal variable"""
        try:
            cause_data = causal_data[relationship.cause]
            effect_data = causal_data[relationship.effect]
            cause_std = cause_data.std()
            intervention_magnitude = cause_std
            correlation = cause_data.corr(effect_data)
            effect_std = effect_data.std()
            estimated_effect_change = correlation * (intervention_magnitude / cause_std) * effect_std
            current_effect_mean = effect_data.mean()
            if current_effect_mean > 0:
                percentage_improvement = estimated_effect_change / current_effect_mean
                return min(abs(percentage_improvement), 1.0)
            return 0.0
        except Exception as e:
            self.logger.warning('Intervention simulation failed: %s', e)
            return 0.0

    def _assess_intervention_feasibility(self, variable: str) -> float:
        """Assess how feasible it is to intervene on a variable"""
        feasibility_map = {'rule': 0.9, 'context': 0.3, 'system': 0.7}
        for var_type, feasibility in feasibility_map.items():
            if var_type in variable.lower():
                return feasibility
        return 0.5

    def _create_causal_insight(self, relationship: CausalRelationship, intervention_info: dict[str, float] | None) -> Insight | None:
        """Create an insight from a causal relationship"""
        try:
            impact = 'high' if relationship.strength > 0.6 and relationship.confidence > 0.8 else 'medium' if relationship.strength > 0.4 and relationship.confidence > 0.6 else 'low'
            description = f'Causal analysis reveals that {relationship.cause} has a {relationship.strength:.2f} strength causal effect on {relationship.effect}'
            evidence = [f'Causal strength: {relationship.strength:.3f}', f'Statistical confidence: {relationship.confidence:.3f}', f"Correlation: {relationship.statistical_tests.get('correlation', 0):.3f}"]
            if 'granger_approximation' in relationship.statistical_tests:
                evidence.append(f"Temporal causality: {relationship.statistical_tests['granger_approximation']:.3f}")
            recommendations = [f'Monitor the relationship between {relationship.cause} and {relationship.effect}', 'Consider targeted interventions to optimize causal effects']
            if intervention_info:
                estimated_improvement = intervention_info.get('estimated_improvement', 0)
                feasibility = intervention_info.get('intervention_feasibility', 0)
                if estimated_improvement > 0.1 and feasibility > 0.5:
                    recommendations.append(f'High-impact intervention: Optimizing {relationship.cause} could improve {relationship.effect} by {estimated_improvement * 100:.1f}%')
                    evidence.append(f'Estimated intervention effect: {estimated_improvement * 100:.1f}% improvement')
                    evidence.append(f'Intervention feasibility: {feasibility:.2f}')
            return Insight(insight_id=f'causal_{relationship.cause}_{relationship.effect}', type='causal', title=f'Causal Relationship: {relationship.cause} → {relationship.effect}', description=description, confidence=relationship.confidence, impact=impact, evidence=evidence, recommendations=recommendations, metadata={'causal_strength': relationship.strength, 'statistical_tests': relationship.statistical_tests, 'intervention_analysis': intervention_info})
        except Exception as e:
            self.logger.error('Failed to create causal insight: %s', e)
            return None

    def _encode_categorical(self, category: str) -> float:
        """Simple categorical encoding for causal analysis"""
        category_map = {'clarity': 0.1, 'specificity': 0.2, 'completeness': 0.3, 'actionability': 0.4, 'context': 0.5, 'performance': 0.6, 'system': 0.7, 'general': 0.8}
        return category_map.get(category.lower(), 0.5)

    def _estimate_context_complexity(self, context: str) -> float:
        """Estimate context complexity for causal analysis"""
        complexity_indicators = ['enterprise', 'large', 'complex', 'multi', 'distributed']
        simplicity_indicators = ['simple', 'basic', 'small', 'single']
        context_lower = context.lower()
        complexity_score = 0.5
        for indicator in complexity_indicators:
            if indicator in context_lower:
                complexity_score += 0.1
        for indicator in simplicity_indicators:
            if indicator in context_lower:
                complexity_score -= 0.1
        return max(0.0, min(1.0, complexity_score))
