"""
Real Behavior Tests for Difficulty Distribution Analysis - 2025 Best Practices
Tests verify actual difficulty distribution algorithms with real data.
"""
from typing import Any, Dict, List
import pytest
import numpy as np
from prompt_improver.ml.analysis.difficulty_distribution_analyzer import DifficultyDistributionAnalyzer, DifficultyProfile, FocusAreaTarget
from prompt_improver.ml.analysis.performance_gap_analyzer import PerformanceGap

@pytest.fixture
def real_difficulty_analyzer():
    """Create real difficulty distribution analyzer."""
    return DifficultyDistributionAnalyzer()

@pytest.fixture
def real_performance_gaps():
    """Create real performance gaps for testing."""
    gaps = []
    for i in range(3):
        gap = PerformanceGap(rule_id=f'critical_rule_{i}', gap_type='effectiveness', severity=0.8 + i * 0.05, current_performance=0.3 + i * 0.1, target_performance=0.8, gap_magnitude=0.5 - i * 0.1, improvement_potential=0.4 - i * 0.05, confidence=0.7 + i * 0.1, metadata={'sample_count': 25 + i * 5, 'std_dev': 0.15 - i * 0.02, 'trend': ['declining', 'stable', 'improving'][i]})
        gaps.append(gap)
    for i in range(4):
        gap = PerformanceGap(rule_id=f'moderate_rule_{i}', gap_type='consistency', severity=0.4 + i * 0.05, current_performance=0.5 + i * 0.05, target_performance=0.8, gap_magnitude=0.3 - i * 0.05, improvement_potential=0.25 - i * 0.02, confidence=0.8 + i * 0.02, metadata={'sample_count': 30 + i * 3, 'std_dev': 0.12 - i * 0.01, 'variability': 0.2 + i * 0.02})
        gaps.append(gap)
    for i in range(2):
        gap = PerformanceGap(rule_id=f'minor_rule_{i}', gap_type='coverage', severity=0.2 + i * 0.05, current_performance=0.7 + i * 0.05, target_performance=0.85, gap_magnitude=0.15 - i * 0.05, improvement_potential=0.12 - i * 0.02, confidence=0.9, metadata={'sample_count': 40 + i * 5, 'std_dev': 0.08, 'trend': 'stable'})
        gaps.append(gap)
    return gaps

@pytest.fixture
def real_hardness_analysis():
    """Create real hardness analysis data."""
    return {'optimal_threshold': 0.65, 'distribution': {'mean': 0.55, 'std': 0.22, 'median': 0.52, 'q25': 0.38, 'q75': 0.71, 'hard_examples_ratio': 0.35}, 'rule_hardness': {'critical_rule_0': [0.8, 0.75, 0.82, 0.78], 'critical_rule_1': [0.7, 0.73, 0.69, 0.71], 'moderate_rule_0': [0.5, 0.52, 0.48, 0.51]}, 'recommendations': ['High proportion of hard examples - focus on neural generation methods', 'High hardness variance - use adaptive difficulty distribution']}

class TestRealDifficultyDistributionAnalysis:
    """Test real behavior of difficulty distribution analysis."""

    async def test_real_difficulty_distribution_analysis(self, real_difficulty_analyzer: DifficultyDistributionAnalyzer, real_performance_gaps: list[PerformanceGap], real_hardness_analysis: dict[str, Any]):
        """Test difficulty distribution analysis with real performance gaps."""
        difficulty_profile = await real_difficulty_analyzer.analyze_optimal_difficulty_distribution(performance_gaps=real_performance_gaps, hardness_analysis=real_hardness_analysis, focus_areas=['clarity', 'specificity', 'effectiveness'])
        assert isinstance(difficulty_profile, DifficultyProfile)
        assert isinstance(difficulty_profile.distribution_weights, dict)
        assert isinstance(difficulty_profile.hardness_threshold, float)
        assert isinstance(difficulty_profile.focus_areas, list)
        assert isinstance(difficulty_profile.complexity_factors, dict)
        assert isinstance(difficulty_profile.adaptive_parameters, dict)
        weights = difficulty_profile.distribution_weights
        assert 'easy' in weights
        assert 'medium' in weights
        assert 'hard' in weights
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
        assert all(w >= 0.0 for w in weights.values())
        assert 0.0 <= difficulty_profile.hardness_threshold <= 1.0
        assert set(difficulty_profile.focus_areas) == {'clarity', 'specificity', 'effectiveness'}
        adaptive_params = difficulty_profile.adaptive_parameters
        assert 'learning_rate' in adaptive_params
        assert 'adjustment_frequency' in adaptive_params
        assert 'performance_window' in adaptive_params
        print(f'Difficulty distribution: {weights}')
        print(f'Hardness threshold: {difficulty_profile.hardness_threshold:.3f}')
        critical_ratio = 3 / 9
        assert weights['hard'] > 0.3

    async def test_real_focus_area_targeting(self, real_difficulty_analyzer: DifficultyDistributionAnalyzer, real_performance_gaps: list[PerformanceGap]):
        """Test focus area targeting with real performance data."""
        focus_areas = ['clarity', 'specificity', 'effectiveness', 'consistency']
        current_performance = {'clarity_effectiveness': 0.6, 'specificity_effectiveness': 0.5, 'effectiveness_effectiveness': 0.4, 'consistency_effectiveness': 0.7}
        focus_targets = await real_difficulty_analyzer.generate_focus_area_targets(performance_gaps=real_performance_gaps, focus_areas=focus_areas, current_performance=current_performance, target_samples=1000)
        assert len(focus_targets) == len(focus_areas)
        assert all(isinstance(target, FocusAreaTarget) for target in focus_targets)
        priorities = [target.priority for target in focus_targets]
        assert priorities == sorted(priorities, reverse=True)
        total_allocation = sum(target.sample_allocation for target in focus_targets)
        assert abs(total_allocation - 1.0) < 0.01
        effectiveness_target = next(target for target in focus_targets if target.area_name == 'effectiveness')
        assert effectiveness_target.priority == max(target.priority for target in focus_targets)
        for target in focus_targets:
            assert 0.0 <= target.target_improvement <= 0.5
            assert 0.0 <= target.current_performance <= 1.0
            assert target.estimated_samples > 0
        print('Focus area priorities:')
        for target in focus_targets:
            print(f"  {target.area_name}: priority={target.priority:.3f}, allocation={target.sample_allocation:.3f}, samples={target.metadata['estimated_samples']}")

    async def test_real_adaptive_distribution_calculation(self, real_difficulty_analyzer: DifficultyDistributionAnalyzer, real_performance_gaps: list[PerformanceGap], real_hardness_analysis: dict[str, Any]):
        """Test adaptive distribution calculation with varying gap profiles."""
        high_severity_gaps = [gap for gap in real_performance_gaps if gap.severity >= 0.7]
        high_severity_profile = await real_difficulty_analyzer.analyze_optimal_difficulty_distribution(performance_gaps=high_severity_gaps, hardness_analysis=real_hardness_analysis, focus_areas=['effectiveness'])
        assert high_severity_profile.distribution_weights['hard'] > 0.4
        low_severity_gaps = [gap for gap in real_performance_gaps if gap.severity < 0.3]
        low_severity_profile = await real_difficulty_analyzer.analyze_optimal_difficulty_distribution(performance_gaps=low_severity_gaps, hardness_analysis={**real_hardness_analysis, 'distribution': {**real_hardness_analysis['distribution'], 'hard_examples_ratio': 0.15}}, focus_areas=['coverage'])
        assert low_severity_profile.distribution_weights['easy'] > 0.3
        high_hard_ratio = high_severity_profile.distribution_weights['hard']
        low_hard_ratio = low_severity_profile.distribution_weights['hard']
        assert high_hard_ratio > low_hard_ratio
        print(f'High severity distribution: {high_severity_profile.distribution_weights}')
        print(f'Low severity distribution: {low_severity_profile.distribution_weights}')

    async def test_real_complexity_factor_application(self, real_difficulty_analyzer: DifficultyDistributionAnalyzer, real_performance_gaps: list[PerformanceGap], real_hardness_analysis: dict[str, Any]):
        """Test complexity factor application in difficulty distribution."""
        complex_gaps = []
        for i, gap in enumerate(real_performance_gaps[:3]):
            complex_gap = PerformanceGap(rule_id=gap.rule_id, gap_type=gap.gap_type, severity=0.8, current_performance=gap.current_performance, target_performance=gap.target_performance, gap_magnitude=gap.gap_magnitude, improvement_potential=gap.improvement_potential, confidence=0.5 - i * 0.1, metadata={**gap.metadata, 'complexity_indicators': {'semantic_complexity': 0.7 + i * 0.1, 'rule_interactions': i + 1, 'pattern_variability': 0.6 + i * 0.15}})
            complex_gaps.append(complex_gap)
        complex_profile = await real_difficulty_analyzer.analyze_optimal_difficulty_distribution(performance_gaps=complex_gaps, hardness_analysis=real_hardness_analysis, focus_areas=['effectiveness'])
        assert 'semantic_complexity' in complex_profile.complexity_factors
        assert 'rule_complexity' in complex_profile.complexity_factors
        assert complex_profile.distribution_weights['hard'] > 0.5
        adaptive_params = complex_profile.adaptive_parameters
        assert adaptive_params['hardness_sensitivity'] > 0.0
        print(f'Complex gaps distribution: {complex_profile.distribution_weights}')
        print(f'Complexity factors: {complex_profile.complexity_factors}')

    async def test_real_distribution_validation_and_normalization(self, real_difficulty_analyzer: DifficultyDistributionAnalyzer):
        """Test distribution validation and normalization with edge cases."""
        minimal_gaps = [PerformanceGap(rule_id='minimal_rule', gap_type='effectiveness', severity=0.1, current_performance=0.8, target_performance=0.85, gap_magnitude=0.05, improvement_potential=0.04, confidence=0.9, metadata={'sample_count': 50})]
        minimal_profile = await real_difficulty_analyzer.analyze_optimal_difficulty_distribution(performance_gaps=minimal_gaps, hardness_analysis={'optimal_threshold': 0.7, 'distribution': {'hard_examples_ratio': 0.1, 'std': 0.1, 'median': 0.6}}, focus_areas=['effectiveness'])
        weights = minimal_profile.distribution_weights
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= 0.0 for w in weights.values())
        empty_profile = await real_difficulty_analyzer.analyze_optimal_difficulty_distribution(performance_gaps=[], hardness_analysis={'optimal_threshold': 0.7, 'distribution': {'hard_examples_ratio': 0.3}}, focus_areas=[])
        empty_weights = empty_profile.distribution_weights
        assert abs(sum(empty_weights.values()) - 1.0) < 0.01
        print(f'Minimal gaps distribution: {weights}')
        print(f'Empty gaps distribution: {empty_weights}')
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
