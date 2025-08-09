"""Statistical Analysis Framework

Significance testing and validation for evaluation results.
Provides comprehensive statistical analysis including hypothesis testing,
effect size analysis, and reliability validation.
"""
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy import stats
from scipy.stats import normaltest, pearsonr, shapiro, spearmanr
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import resample
logger = logging.getLogger(__name__)

@dataclass
class StatisticalConfig:
    """Configuration for statistical analysis"""
    significance_level: float = 0.05
    confidence_level: float = 0.95
    minimum_sample_size: int = 5
    recommended_sample_size: int = 30
    effect_sizes: dict[str, float] = None
    validation_thresholds: dict[str, float] = None

    def __post_init__(self):
        if self.effect_sizes is None:
            self.effect_sizes = {'small': 0.2, 'medium': 0.5, 'large': 0.8}
        if self.validation_thresholds is None:
            self.validation_thresholds = {'reliability_threshold': 0.7, 'consistency_threshold': 0.8, 'validity_threshold': 0.6}

@dataclass
class DescriptiveStats:
    """Descriptive statistics for a metric"""
    count: int
    mean: float
    median: float
    mode: float | list[float]
    standard_deviation: float
    variance: float
    min_value: float
    max_value: float
    range_value: float
    quartiles: dict[str, float]
    skewness: float
    kurtosis: float
    coefficient_of_variation: float

@dataclass
class DistributionAnalysis:
    """Distribution analysis results"""
    normality_test: dict[str, Any]
    histogram: dict[str, Any]
    outliers: list[float]
    distribution_type: str

@dataclass
class CorrelationResult:
    """Correlation analysis result"""
    correlation: float
    p_value: float
    significance: str
    strength: str
    direction: str

class StatisticalAnalyzer:
    """Statistical Analysis Framework for evaluation results"""

    def __init__(self, config: StatisticalConfig | None=None):
        """Initialize the statistical analyzer

        Args:
            config: Statistical analysis configuration
        """
        self.config = config or StatisticalConfig()
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

    async def perform_statistical_analysis(self, results: list[dict[str, Any]], options: dict[str, Any] | None=None) -> dict[str, Any]:
        """Perform comprehensive statistical analysis on evaluation results

        Args:
            results: Array of evaluation results
            options: Analysis options

        Returns:
            Statistical analysis results
        """
        start_time = datetime.now()
        options = options or {}
        self.logger.info('Starting statistical analysis', extra={'sample_size': len(results), 'analysis_type': options.get('analysisType', 'comprehensive')})
        try:
            self._validate_inputs(results, options)
            analysis_data = self._prepare_analysis_data(results, options)
            analysis = {'metadata': {'sample_size': len(results), 'analysis_type': options.get('analysisType', 'comprehensive'), 'significance_level': self.config.significance_level, 'confidence_level': self.config.confidence_level, 'analyzed_at': datetime.now().isoformat()}, 'descriptive_stats': self._calculate_descriptive_statistics(analysis_data), 'distribution_analysis': self._analyze_distributions(analysis_data), 'hypothesis_tests': await self._perform_hypothesis_tests(analysis_data, options), 'effect_size_analysis': self._analyze_effect_sizes(analysis_data, options), 'reliability_analysis': self._analyze_reliability(analysis_data, options), 'validity_analysis': self._analyze_validity(analysis_data, options), 'correlation_analysis': self._analyze_correlations(analysis_data), 'confidence_intervals': self._calculate_confidence_intervals(analysis_data), 'summary': {}, 'recommendations': []}
            analysis['summary'] = self._generate_statistical_summary(analysis)
            analysis['recommendations'] = self._generate_statistical_recommendations(analysis)
            analysis_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info('Statistical analysis completed', extra={'analysis_time_ms': analysis_time, 'significant_findings': analysis['summary'].get('significant_findings', 0)})
            return analysis
        except Exception as error:
            self.logger.error('Statistical analysis failed: %s', error)
            raise Exception(f'Statistical analysis failed: {error}')

    def _validate_inputs(self, results: list[dict[str, Any]], options: dict[str, Any]):
        """Validate analysis inputs"""
        if not results:
            raise ValueError('Results array cannot be empty')
        if len(results) < self.config.minimum_sample_size:
            self.logger.warning('Sample size (%s) below minimum (%s)', len(results), self.config.minimum_sample_size)
        if len(results) < self.config.recommended_sample_size:
            self.logger.warning('Sample size (%s) below recommended (%s)', len(results), self.config.recommended_sample_size)

    def _prepare_analysis_data(self, results: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        """Prepare data for statistical analysis"""
        data = {'raw': results, 'overall_scores': [self._extract_score(r, 'overallScore') or self._extract_score(r, 'overallQuality') or 0 for r in results], 'clarity_scores': [self._extract_score(r, 'clarity') for r in results], 'completeness_scores': [self._extract_score(r, 'completeness') for r in results], 'actionability_scores': [self._extract_score(r, 'actionability') for r in results], 'effectiveness_scores': [self._extract_score(r, 'effectiveness') for r in results], 'strategies': [r.get('strategy') or r.get('metadata', {}).get('strategy', 'unknown') for r in results], 'models': [r.get('model') or r.get('metadata', {}).get('model', 'unknown') for r in results], 'complexities': [r.get('complexity', 'unknown') for r in results], 'timestamps': [r.get('timestamp') or r.get('metadata', {}).get('timestamp', datetime.now().isoformat()) for r in results], 'sample_sizes': [r.get('sampleSize', 1) for r in results], 'groups': self._group_data_for_analysis(results, options)}
        data['score_differences'] = self._calculate_score_differences(data)
        data['consistency_metrics'] = self._calculate_consistency_metrics(data)
        data['quality_trends'] = self._calculate_quality_trends(data)
        return data

    def _extract_score(self, result: dict[str, Any], metric: str) -> float:
        """Extract score from result object"""
        metric_value = result.get(metric)
        if isinstance(metric_value, (int, float)):
            return float(metric_value)
        if isinstance(metric_value, dict):
            if metric_value.get('score') is not None:
                return float(metric_value['score'])
            if metric_value.get('mean') is not None:
                return float(metric_value['mean'])
        scores = result.get('scores', {})
        if isinstance(scores, dict) and scores.get(metric) is not None:
            return float(scores[metric])
        metrics = result.get('metrics', {})
        if isinstance(metrics, dict) and isinstance(metrics.get(metric), dict):
            metric_obj = metrics[metric]
            if metric_obj.get('score') is not None:
                return float(metric_obj['score'])
        return 0.0

    def _group_data_for_analysis(self, results: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        """Group data for comparative analysis"""
        groups = {}
        groups['by_strategy'] = self._group_by(results, lambda r: r.get('strategy', 'default'))
        groups['by_model'] = self._group_by(results, lambda r: r.get('model', 'default'))
        groups['by_complexity'] = self._group_by(results, lambda r: r.get('complexity', 'unknown'))
        groups['by_quality_grade'] = self._group_by(results, lambda r: self._get_quality_grade(r.get('overallScore') or r.get('overallQuality', 0)))
        if options.get('groupBy'):
            groups['custom'] = self._group_by(results, options['groupBy'])
        return groups

    def _group_by(self, array: list[Any], key_fn) -> dict[str, list[Any]]:
        """Group array by key function"""
        groups = {}
        for item in array:
            key = key_fn(item)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        return groups

    def _get_quality_grade(self, score: float) -> str:
        """Get quality grade from score"""
        if score >= 0.8:
            return 'excellent'
        if score >= 0.7:
            return 'good'
        if score >= 0.6:
            return 'fair'
        if score >= 0.5:
            return 'poor'
        return 'very_poor'

    def _calculate_descriptive_statistics(self, data: dict[str, Any]) -> dict[str, DescriptiveStats]:
        """Calculate descriptive statistics for all metrics"""
        metrics = ['overall_scores', 'clarity_scores', 'completeness_scores', 'actionability_scores', 'effectiveness_scores']
        stats_dict = {}
        for metric in metrics:
            values = [v for v in data[metric] if v is not None and (not np.isnan(v))]
            if len(values) > 0:
                stats_dict[metric] = DescriptiveStats(count=len(values), mean=np.mean(values), median=np.median(values), mode=self._calculate_mode(values), standard_deviation=np.std(values, ddof=1) if len(values) > 1 else 0, variance=np.var(values, ddof=1) if len(values) > 1 else 0, min_value=np.min(values), max_value=np.max(values), range_value=np.max(values) - np.min(values), quartiles=self._calculate_quartiles(values), skewness=stats.skew(values) if len(values) > 2 else 0, kurtosis=stats.kurtosis(values) if len(values) > 3 else 0, coefficient_of_variation=np.std(values) / np.mean(values) if np.mean(values) != 0 else 0)
            else:
                stats_dict[metric] = self._get_empty_stats()
        return stats_dict

    def _calculate_mode(self, values: list[float]) -> float | list[float]:
        """Calculate mode of values"""
        try:
            mode_result = stats.mode(values, keepdims=True)
            modes = mode_result.mode
            return modes[0] if len(modes) == 1 else modes.tolist()
        except:
            return 0.0

    def _calculate_quartiles(self, values: list[float]) -> dict[str, float]:
        """Calculate quartiles"""
        return {'q1': np.percentile(values, 25), 'q2': np.percentile(values, 50), 'q3': np.percentile(values, 75), 'iqr': np.percentile(values, 75) - np.percentile(values, 25)}

    def _get_empty_stats(self) -> DescriptiveStats:
        """Get empty statistics object"""
        return DescriptiveStats(count=0, mean=0, median=0, mode=0, standard_deviation=0, variance=0, min_value=0, max_value=0, range_value=0, quartiles={'q1': 0, 'q2': 0, 'q3': 0, 'iqr': 0}, skewness=0, kurtosis=0, coefficient_of_variation=0)

    def _analyze_distributions(self, data: dict[str, Any]) -> dict[str, DistributionAnalysis]:
        """Analyze distributions of metrics"""
        metrics = ['overall_scores', 'clarity_scores', 'completeness_scores', 'actionability_scores', 'effectiveness_scores']
        distributions = {}
        for metric in metrics:
            values = [v for v in data[metric] if v is not None and (not np.isnan(v))]
            if len(values) >= self.config.minimum_sample_size:
                distributions[metric] = DistributionAnalysis(normality_test=self._test_normality(values), histogram=self._create_histogram(values), outliers=self._detect_outliers(values), distribution_type=self._classify_distribution(values))
        return distributions

    def _test_normality(self, values: list[float]) -> dict[str, Any]:
        """Test normality of distribution"""
        try:
            if len(values) < 3:
                return {'test': 'insufficient_data', 'p_value': None, 'is_normal': False}
            if len(values) <= 50:
                statistic, p_value = shapiro(values)
                test_name = 'shapiro_wilk'
            else:
                statistic, p_value = normaltest(values)
                test_name = 'dagostino_pearson'
            return {'test': test_name, 'statistic': float(statistic), 'p_value': float(p_value), 'is_normal': p_value > self.config.significance_level}
        except Exception as e:
            self.logger.warning('Normality test failed: %s', e)
            return {'test': 'failed', 'p_value': None, 'is_normal': False}

    def _create_histogram(self, values: list[float]) -> dict[str, Any]:
        """Create histogram data"""
        try:
            hist, bin_edges = np.histogram(values, bins='auto')
            return {'counts': hist.tolist(), 'bin_edges': bin_edges.tolist(), 'bins': len(hist)}
        except:
            return {'counts': [], 'bin_edges': [], 'bins': 0}

    def _detect_outliers(self, values: list[float]) -> list[float]:
        """Detect outliers using IQR method"""
        try:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [v for v in values if v < lower_bound or v > upper_bound]
            return outliers
        except:
            return []

    def _classify_distribution(self, values: list[float]) -> str:
        """Classify distribution type"""
        try:
            skewness = stats.skew(values)
            kurtosis = stats.kurtosis(values)
            if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
                return 'normal'
            if skewness > 0.5:
                return 'right_skewed'
            if skewness < -0.5:
                return 'left_skewed'
            if kurtosis > 0.5:
                return 'heavy_tailed'
            if kurtosis < -0.5:
                return 'light_tailed'
            return 'unknown'
        except:
            return 'unknown'

    async def _perform_hypothesis_tests(self, data: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        """Perform hypothesis tests"""
        tests = {}
        if options.get('compareGroups'):
            tests['group_comparisons'] = await self._perform_group_comparisons(data, options)
        if options.get('baseline'):
            tests['baseline_comparison'] = self._test_improvement_significance(data, options['baseline'])
        tests['consistency_tests'] = self._test_metric_consistency(data)
        return tests

    async def _perform_group_comparisons(self, data: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        """Perform statistical comparisons between groups"""
        comparisons = {}
        for group_type, groups in data['groups'].items():
            if len(groups) >= 2:
                group_names = list(groups.keys())
                for i, group1 in enumerate(group_names):
                    for group2 in group_names[i + 1:]:
                        comparison_key = f'{group1}_vs_{group2}'
                        comparisons[comparison_key] = self._compare_two_groups(groups[group1], groups[group2])
        return comparisons

    def _compare_two_groups(self, group1: list[dict], group2: list[dict]) -> dict[str, Any]:
        """Compare two groups statistically"""
        try:
            scores1 = [self._extract_score(r, 'overallScore') for r in group1]
            scores2 = [self._extract_score(r, 'overallScore') for r in group2]
            scores1 = [s for s in scores1 if s is not None and (not np.isnan(s))]
            scores2 = [s for s in scores2 if s is not None and (not np.isnan(s))]
            if len(scores1) < 3 or len(scores2) < 3:
                return {'error': 'insufficient_sample_size'}
            statistic, p_value = stats.ttest_ind(scores1, scores2)
            pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + (len(scores2) - 1) * np.var(scores2, ddof=1)) / (len(scores1) + len(scores2) - 2))
            cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std != 0 else 0
            return {'statistic': float(statistic), 'p_value': float(p_value), 'significant': p_value < self.config.significance_level, 'effect_size': float(cohens_d), 'effect_magnitude': self._interpret_effect_size(abs(cohens_d)), 'group1_mean': float(np.mean(scores1)), 'group2_mean': float(np.mean(scores2)), 'group1_n': len(scores1), 'group2_n': len(scores2)}
        except Exception as e:
            return {'error': str(e)}

    def _test_improvement_significance(self, data: dict[str, Any], baseline: float) -> dict[str, Any]:
        """Test if improvement over baseline is significant"""
        try:
            scores = [s for s in data['overall_scores'] if s is not None and (not np.isnan(s))]
            if len(scores) < 3:
                return {'error': 'insufficient_sample_size'}
            statistic, p_value = stats.ttest_1samp(scores, baseline)
            return {'baseline': float(baseline), 'sample_mean': float(np.mean(scores)), 'statistic': float(statistic), 'p_value': float(p_value), 'significant_improvement': p_value < self.config.significance_level and np.mean(scores) > baseline, 'effect_size': (np.mean(scores) - baseline) / np.std(scores, ddof=1) if np.std(scores, ddof=1) != 0 else 0}
        except Exception as e:
            return {'error': str(e)}

    def _test_metric_consistency(self, data: dict[str, Any]) -> dict[str, Any]:
        """Test consistency across metrics"""
        metrics = ['clarity_scores', 'completeness_scores', 'actionability_scores', 'effectiveness_scores']
        consistency_tests = {}
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i + 1:]:
                values1 = [v for v in data[metric1] if v is not None and (not np.isnan(v))]
                values2 = [v for v in data[metric2] if v is not None and (not np.isnan(v))]
                if len(values1) == len(values2) and len(values1) >= 3:
                    correlation, p_value = pearsonr(values1, values2)
                    consistency_tests[f'{metric1}_vs_{metric2}'] = {'correlation': float(correlation), 'p_value': float(p_value), 'significant': p_value < self.config.significance_level, 'consistent': correlation > 0.3 and p_value < self.config.significance_level}
        return consistency_tests

    def _analyze_effect_sizes(self, data: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        """Analyze effect sizes for comparisons"""
        effect_sizes = {}
        if 'groups' in data:
            for group_type, groups in data['groups'].items():
                if len(groups) >= 2:
                    group_names = list(groups.keys())
                    for i, group1 in enumerate(group_names):
                        for group2 in group_names[i + 1:]:
                            effect_key = f'{group1}_vs_{group2}_effect'
                            effect_sizes[effect_key] = self._calculate_cohens_d(groups[group1], groups[group2])
        return effect_sizes

    def _calculate_cohens_d(self, group1: list[dict], group2: list[dict]) -> dict[str, Any]:
        """Calculate Cohen's d effect size"""
        try:
            scores1 = [self._extract_score(r, 'overallScore') for r in group1]
            scores2 = [self._extract_score(r, 'overallScore') for r in group2]
            scores1 = [s for s in scores1 if s is not None and (not np.isnan(s))]
            scores2 = [s for s in scores2 if s is not None and (not np.isnan(s))]
            if len(scores1) < 2 or len(scores2) < 2:
                return {'error': 'insufficient_sample_size'}
            pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + (len(scores2) - 1) * np.var(scores2, ddof=1)) / (len(scores1) + len(scores2) - 2))
            if pooled_std == 0:
                return {'cohens_d': 0, 'magnitude': 'none'}
            cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
            return {'cohens_d': float(cohens_d), 'magnitude': self._interpret_effect_size(abs(cohens_d))}
        except Exception as e:
            return {'error': str(e)}

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude"""
        if effect_size >= self.config.effect_sizes['large']:
            return 'large'
        if effect_size >= self.config.effect_sizes['medium']:
            return 'medium'
        if effect_size >= self.config.effect_sizes['small']:
            return 'small'
        return 'negligible'

    def _analyze_reliability(self, data: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        """Analyze reliability of measurements"""
        reliability = {}
        metrics = ['clarity_scores', 'completeness_scores', 'actionability_scores', 'effectiveness_scores']
        metric_matrix = []
        for metric in metrics:
            values = [v for v in data[metric] if v is not None and (not np.isnan(v))]
            if len(values) > 0:
                metric_matrix.append(values)
        if len(metric_matrix) >= 2 and all((len(row) == len(metric_matrix[0]) for row in metric_matrix)):
            reliability['cronbachs_alpha'] = self._calculate_cronbachs_alpha(np.array(metric_matrix).T)
        if len(data.get('timestamps', [])) > 1:
            reliability['temporal_consistency'] = self._analyze_temporal_consistency(data)
        return reliability

    def _calculate_cronbachs_alpha(self, item_matrix: np.ndarray) -> dict[str, Any]:
        """Calculate Cronbach's alpha for internal consistency"""
        try:
            n_items = item_matrix.shape[1]
            item_variances = np.var(item_matrix, axis=0, ddof=1)
            total_variance = np.var(np.sum(item_matrix, axis=1), ddof=1)
            alpha = n_items / (n_items - 1) * (1 - np.sum(item_variances) / total_variance)
            if alpha >= 0.9:
                interpretation = 'excellent'
            elif alpha >= 0.8:
                interpretation = 'good'
            elif alpha >= 0.7:
                interpretation = 'acceptable'
            elif alpha >= 0.6:
                interpretation = 'questionable'
            else:
                interpretation = 'poor'
            return {'alpha': float(alpha), 'interpretation': interpretation, 'acceptable': alpha >= self.config.validation_thresholds['reliability_threshold']}
        except Exception as e:
            return {'error': str(e)}

    def _analyze_temporal_consistency(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze consistency over time"""
        try:
            scores = data['overall_scores']
            timestamps = data['timestamps']
            if len(scores) != len(timestamps):
                return {'error': 'mismatched_data'}
            window_size = min(5, len(scores) // 2)
            if window_size < 2:
                return {'error': 'insufficient_temporal_data'}
            variances = []
            for i in range(len(scores) - window_size + 1):
                window_scores = scores[i:i + window_size]
                variances.append(np.var(window_scores, ddof=1))
            temporal_consistency = 1 - np.mean(variances) / np.var(scores, ddof=1) if np.var(scores, ddof=1) != 0 else 1
            return {'temporal_consistency': float(max(0, temporal_consistency)), 'acceptable': temporal_consistency >= self.config.validation_thresholds['consistency_threshold']}
        except Exception as e:
            return {'error': str(e)}

    def _analyze_validity(self, data: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        """Analyze validity of measurements"""
        validity = {}
        related_pairs = [('clarity_scores', 'completeness_scores'), ('actionability_scores', 'effectiveness_scores'), ('clarity_scores', 'overall_scores'), ('effectiveness_scores', 'overall_scores')]
        convergent_validity = {}
        for metric1, metric2 in related_pairs:
            values1 = [v for v in data[metric1] if v is not None and (not np.isnan(v))]
            values2 = [v for v in data[metric2] if v is not None and (not np.isnan(v))]
            if len(values1) == len(values2) and len(values1) >= 3:
                correlation, p_value = pearsonr(values1, values2)
                convergent_validity[f'{metric1}_{metric2}'] = {'correlation': float(correlation), 'p_value': float(p_value), 'valid': correlation > self.config.validation_thresholds['validity_threshold'] and p_value < self.config.significance_level}
        validity['convergent_validity'] = convergent_validity
        validity['face_validity'] = self._assess_face_validity(data)
        return validity

    def _assess_face_validity(self, data: dict[str, Any]) -> dict[str, Any]:
        """Assess face validity of score distributions"""
        assessment = {}
        metrics = ['overall_scores', 'clarity_scores', 'completeness_scores', 'actionability_scores', 'effectiveness_scores']
        for metric in metrics:
            values = [v for v in data[metric] if v is not None and (not np.isnan(v))]
            if len(values) > 0:
                min_val, max_val = (np.min(values), np.max(values))
                reasonable_range = 0 <= min_val <= 1 and 0 <= max_val <= 1 or (0 <= min_val <= 100 and 0 <= max_val <= 100)
                distribution_variance = np.var(values) > 0.01
                mean_val = np.mean(values)
                reasonable_mean = 0.1 <= mean_val <= 0.9 or 10 <= mean_val <= 90
                assessment[metric] = {'reasonable_range': reasonable_range, 'has_variance': distribution_variance, 'reasonable_mean': reasonable_mean, 'face_valid': reasonable_range and distribution_variance and reasonable_mean}
        return assessment

    def _analyze_correlations(self, data: dict[str, Any]) -> dict[str, CorrelationResult]:
        """Analyze correlations between metrics"""
        metrics = ['overall_scores', 'clarity_scores', 'completeness_scores', 'actionability_scores', 'effectiveness_scores']
        correlations = {}
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i + 1:]:
                values1 = [v for v in data[metric1] if v is not None and (not np.isnan(v))]
                values2 = [v for v in data[metric2] if v is not None and (not np.isnan(v))]
                if len(values1) == len(values2) and len(values1) >= 3:
                    correlation, p_value = pearsonr(values1, values2)
                    correlations[f'{metric1}_{metric2}'] = CorrelationResult(correlation=float(correlation), p_value=float(p_value), significance='significant' if p_value < self.config.significance_level else 'not_significant', strength=self._interpret_correlation_strength(abs(correlation)), direction='positive' if correlation > 0 else 'negative')
        return correlations

    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength"""
        if correlation >= 0.7:
            return 'strong'
        if correlation >= 0.5:
            return 'moderate'
        if correlation >= 0.3:
            return 'weak'
        return 'negligible'

    def _calculate_confidence_intervals(self, data: dict[str, Any]) -> dict[str, dict[str, float]]:
        """Calculate confidence intervals for metrics with bootstrap support"""
        metrics = ['overall_scores', 'clarity_scores', 'completeness_scores', 'actionability_scores', 'effectiveness_scores']
        intervals = {}
        for metric in metrics:
            values = [v for v in data[metric] if v is not None and (not np.isnan(v))]
            if len(values) >= 3:
                mean = np.mean(values)
                sem = stats.sem(values)
                ci = stats.t.interval(self.config.confidence_level, len(values) - 1, loc=mean, scale=sem)
                bootstrap_ci = self._calculate_bootstrap_ci(values)
                intervals[metric] = {'mean': float(mean), 'lower_bound': float(ci[0]), 'upper_bound': float(ci[1]), 'margin_of_error': float(ci[1] - mean), 'bootstrap_lower': float(bootstrap_ci[0]), 'bootstrap_upper': float(bootstrap_ci[1]), 'bootstrap_method': 'BCa'}
        return intervals

    def _calculate_score_differences(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate score differences and improvements"""
        differences = {}
        scores = data['overall_scores']
        if len(scores) > 1:
            differences['score_variance'] = float(np.var(scores, ddof=1))
            differences['score_range'] = float(np.max(scores) - np.min(scores))
            differences['improvement_trend'] = self._calculate_trend(scores)
        return differences

    def _calculate_consistency_metrics(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate consistency metrics across evaluations"""
        consistency = {}
        metrics = ['overall_scores', 'clarity_scores', 'completeness_scores', 'actionability_scores', 'effectiveness_scores']
        for metric in metrics:
            values = [v for v in data[metric] if v is not None and (not np.isnan(v))]
            if len(values) > 1:
                mean_val = np.mean(values)
                if mean_val != 0:
                    cv = np.std(values, ddof=1) / mean_val
                    consistency[f'{metric}_cv'] = float(cv)
        return consistency

    def _calculate_quality_trends(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate quality trends over time"""
        trends = {}
        scores = data['overall_scores']
        if len(scores) > 2:
            trends['linear_trend'] = self._calculate_trend(scores)
            trends['trend_strength'] = self._calculate_trend_strength(scores)
        return trends

    def _calculate_trend(self, values: list[float]) -> float:
        """Calculate linear trend slope"""
        try:
            x = np.arange(len(values))
            slope, _, _, _, _ = stats.linregress(x, values)
            return float(slope)
        except:
            return 0.0

    def _calculate_trend_strength(self, values: list[float]) -> float:
        """Calculate strength of trend (R-squared)"""
        try:
            x = np.arange(len(values))
            _, _, r_value, _, _ = stats.linregress(x, values)
            return float(r_value ** 2)
        except:
            return 0.0

    def _generate_statistical_summary(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Generate summary of statistical analysis"""
        summary = {'total_metrics_analyzed': 0, 'significant_findings': 0, 'reliability_assessment': 'unknown', 'validity_assessment': 'unknown', 'overall_quality': 'unknown', 'key_insights': []}
        if 'descriptive_stats' in analysis:
            summary['total_metrics_analyzed'] = len(analysis['descriptive_stats'])
        if 'reliability_analysis' in analysis and 'cronbachs_alpha' in analysis['reliability_analysis']:
            alpha_info = analysis['reliability_analysis']['cronbachs_alpha']
            if isinstance(alpha_info, dict) and 'acceptable' in alpha_info:
                summary['reliability_assessment'] = 'acceptable' if alpha_info['acceptable'] else 'questionable'
        if 'validity_analysis' in analysis and 'convergent_validity' in analysis['validity_analysis']:
            valid_correlations = sum((1 for v in analysis['validity_analysis']['convergent_validity'].values() if isinstance(v, dict) and v.get('valid', False)))
            total_correlations = len(analysis['validity_analysis']['convergent_validity'])
            if total_correlations > 0:
                validity_ratio = valid_correlations / total_correlations
                summary['validity_assessment'] = 'good' if validity_ratio > 0.7 else 'questionable'
        summary['key_insights'] = self._generate_key_insights(analysis)
        return summary

    def _generate_statistical_recommendations(self, analysis: dict[str, Any]) -> list[str]:
        """Generate statistical recommendations based on analysis"""
        recommendations = []
        sample_size = analysis['metadata']['sample_size']
        if sample_size < self.config.recommended_sample_size:
            recommendations.append(f'Increase sample size to at least {self.config.recommended_sample_size} for more reliable results')
        if 'reliability_analysis' in analysis:
            reliability = analysis['reliability_analysis']
            if 'cronbachs_alpha' in reliability and isinstance(reliability['cronbachs_alpha'], dict):
                alpha = reliability['cronbachs_alpha'].get('alpha', 0)
                if alpha < self.config.validation_thresholds['reliability_threshold']:
                    recommendations.append('Improve measurement reliability by refining evaluation criteria')
        if 'distribution_analysis' in analysis:
            for metric, dist_info in analysis['distribution_analysis'].items():
                if isinstance(dist_info, DistributionAnalysis):
                    metric_stats = analysis['descriptive_stats'].get(metric)
                    if metric_stats and hasattr(metric_stats, 'count'):
                        if len(dist_info.outliers) > metric_stats.count * 0.1:
                            recommendations.append(f'Investigate outliers in {metric} - may indicate measurement issues')
        if 'correlation_analysis' in analysis:
            weak_correlations = [k for k, v in analysis['correlation_analysis'].items() if isinstance(v, CorrelationResult) and v.strength == 'negligible']
            if len(weak_correlations) > 0:
                recommendations.append('Consider revising metrics that show weak correlations with overall scores')
        return recommendations

    def _calculate_bootstrap_ci(self, values: list[float], n_bootstrap: int=10000, method: str='bca') -> tuple[float, float]:
        """Calculate bootstrap confidence intervals with bias-corrected and accelerated (BCa) method.

        Best practice 2025: BCa provides more accurate intervals than percentile method.
        Uses 10,000+ iterations for stability with bias correction.
        """
        try:
            if len(values) < 3:
                return (0.0, 0.0)
            values_array = np.array(values)
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = resample(values_array, n_samples=len(values_array), replace=True)
                bootstrap_means.append(np.mean(sample))
            bootstrap_means = np.array(bootstrap_means)
            observed_mean = np.mean(values_array)
            if method == 'percentile':
                alpha = 1 - self.config.confidence_level
                return (np.percentile(bootstrap_means, 100 * alpha / 2), np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
            if method == 'bca':
                alpha = 1 - self.config.confidence_level
                bias_correction = stats.norm.ppf(np.mean(bootstrap_means < observed_mean))
                jackknife_means = []
                for i in range(len(values_array)):
                    jack_sample = np.delete(values_array, i)
                    jackknife_means.append(np.mean(jack_sample))
                jackknife_mean = np.mean(jackknife_means)
                numerator = np.sum((jackknife_mean - jackknife_means) ** 3)
                denominator = 6 * np.sum((jackknife_mean - jackknife_means) ** 2) ** 1.5
                if denominator == 0:
                    acceleration = 0
                else:
                    acceleration = numerator / denominator
                z_alpha_2 = stats.norm.ppf(alpha / 2)
                z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
                alpha_1_numerator = bias_correction + z_alpha_2
                alpha_1_denominator = 1 - acceleration * (bias_correction + z_alpha_2)
                alpha_1 = stats.norm.cdf(bias_correction + alpha_1_numerator / alpha_1_denominator)
                alpha_2_numerator = bias_correction + z_1_alpha_2
                alpha_2_denominator = 1 - acceleration * (bias_correction + z_1_alpha_2)
                alpha_2 = stats.norm.cdf(bias_correction + alpha_2_numerator / alpha_2_denominator)
                alpha_1 = max(0.001, min(0.999, alpha_1))
                alpha_2 = max(0.001, min(0.999, alpha_2))
                return (np.percentile(bootstrap_means, 100 * alpha_1), np.percentile(bootstrap_means, 100 * alpha_2))
        except Exception as e:
            self.logger.warning('Bootstrap CI calculation failed: %s', e)
            alpha = 1 - self.config.confidence_level
            return (np.percentile(bootstrap_means, 100 * alpha / 2), np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    def _calculate_effect_sizes(self, control: list[float], treatment: list[float]) -> dict[str, Any]:
        """Calculate multiple effect size measures following 2025 statistical guidelines.

        Hedges' g preferred for small samples, Cohen's d acceptable for large samples.
        """
        if len(control) < 2 or len(treatment) < 2:
            return {'error': 'Insufficient sample size'}
        n1, n2 = (len(control), len(treatment))
        pooled_std = np.sqrt(((n1 - 1) * np.var(control, ddof=1) + (n2 - 1) * np.var(treatment, ddof=1)) / (n1 + n2 - 2))
        if pooled_std == 0:
            return {'cohens_d': 0, 'hedges_g': 0, 'recommended_measure': 'none'}
        cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std
        j = 1 - 3 / (4 * (n1 + n2) - 9)
        hedges_g = cohens_d * j
        glass_delta = (np.mean(treatment) - np.mean(control)) / np.std(control, ddof=1)

        def interpret_effect_size(effect_size, field='psychology'):
            if field == 'psychology':
                if abs(effect_size) < 0.15:
                    return 'negligible'
                if abs(effect_size) < 0.4:
                    return 'small'
                if abs(effect_size) < 0.75:
                    return 'medium'
                return 'large'
            if field == 'gerontology':
                if abs(effect_size) < 0.15:
                    return 'small'
                if abs(effect_size) < 0.4:
                    return 'medium'
                return 'large'
            return 'unknown'
        return {'cohens_d': float(cohens_d), 'hedges_g': float(hedges_g), 'glass_delta': float(glass_delta), 'recommended_measure': 'hedges_g' if min(n1, n2) < 50 else 'cohens_d', 'interpretation': interpret_effect_size(hedges_g), 'sample_sizes': {'control': n1, 'treatment': n2}, 'bias_correction_factor': float(j)}

    def _apply_multiple_testing_correction(self, p_values: list[float], fdr_level: float=0.05, method: str='adaptive') -> dict[str, Any]:
        """Apply FDR correction following 2025 best practices.

        Key insights from 2025 research:
        - BH procedure valid for 10-20 tests (small scale) and large scale studies
        - Adaptive method (Benjamini-Krieger-Yekutieli) has more power than standard BH
        - FDR controls expected proportion of false discoveries among all discoveries
        """
        if not p_values or len(p_values) == 0:
            return {'error': 'No p-values provided'}
        try:
            if method == 'adaptive':
                try:
                    rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(p_values, alpha=fdr_level, method='fdr_by')
                    method_used = 'fdr_by_adaptive'
                except:
                    rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(p_values, alpha=fdr_level, method='fdr_bh')
                    method_used = 'fdr_bh_fallback'
            else:
                rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(p_values, alpha=fdr_level, method='fdr_bh')
                method_used = 'fdr_bh_standard'
            num_discoveries = np.sum(rejected)
            if num_discoveries > 0:
                expected_false_discoveries = np.sum(p_corrected[rejected])
                actual_fdr = expected_false_discoveries / num_discoveries
            else:
                actual_fdr = 0.0
            results = {'original_p_values': list(p_values), 'adjusted_p_values': list(p_corrected), 'rejected_hypotheses': list(rejected), 'num_discoveries': int(num_discoveries), 'expected_fdr': float(actual_fdr), 'target_fdr': float(fdr_level), 'method_used': method_used, 'interpretation': self._interpret_fdr_results(num_discoveries, actual_fdr, fdr_level)}
            return results
        except Exception as e:
            self.logger.error('Multiple testing correction failed: %s', e)
            return {'error': str(e)}

    def _interpret_fdr_results(self, num_discoveries: int, actual_fdr: float, target_fdr: float) -> str:
        """Provide clear interpretation of FDR results for researchers"""
        if num_discoveries == 0:
            return 'No significant results after FDR correction'
        interpretation = f'Found {num_discoveries} significant results. '
        interpretation += f'Expected false discovery rate: {actual_fdr:.1%} '
        interpretation += f'(target: {target_fdr:.1%}). '
        if actual_fdr <= target_fdr:
            interpretation += 'FDR control successful - results are reliable.'
        else:
            interpretation += 'FDR slightly exceeded target - interpret with caution.'
        return interpretation

    def _generate_key_insights(self, analysis: dict[str, Any]) -> list[str]:
        """Generate key insights from statistical analysis"""
        insights = []
        if 'descriptive_stats' in analysis and 'overall_scores' in analysis['descriptive_stats']:
            overall_stats = analysis['descriptive_stats']['overall_scores']
            if isinstance(overall_stats, DescriptiveStats):
                insights.append(f'Average performance: {overall_stats.mean:.3f} (±{overall_stats.standard_deviation:.3f})')
                if overall_stats.skewness > 0.5:
                    insights.append('Performance distribution is right-skewed - most results below average')
                elif overall_stats.skewness < -0.5:
                    insights.append('Performance distribution is left-skewed - most results above average')
        if 'correlation_analysis' in analysis:
            strong_correlations = [(k, v) for k, v in analysis['correlation_analysis'].items() if isinstance(v, CorrelationResult) and v.strength in ['strong', 'moderate']]
            if strong_correlations:
                strongest = max(strong_correlations, key=lambda x: abs(x[1].correlation))
                insights.append(f'Strongest correlation: {strongest[0]} (r={strongest[1].correlation:.3f})')
        if 'effect_size_analysis' in analysis:
            large_effects = [k for k, v in analysis['effect_size_analysis'].items() if isinstance(v, dict) and v.get('magnitude') == 'large']
            if large_effects:
                insights.append(f'Found {len(large_effects)} large effect sizes indicating meaningful differences')
        return insights
