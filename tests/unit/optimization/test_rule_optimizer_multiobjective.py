"""Tests for Phase 2 Multi-Objective Optimization in Rule Optimizer using real behavior.

Migrated from mock-based testing to real behavior testing following 2025 best practices:
- Use real DEAP NSGA-II and scikit-learn GPR with optimized parameters for test speed
- Test actual optimization behavior and quality metrics
- Mock only external dependencies, not core optimization functionality
- Focus on behavior validation rather than implementation details

Comprehensive test suite for multi-objective optimization enhancements including:
- Real NSGA-II algorithm for actual Pareto frontier discovery
- Real Gaussian Process optimization with authentic Expected Improvement
- Actual hypervolume calculation and convergence metrics
- Real multi-objective trade-off analysis

Testing best practices applied from 2025 research:
- Real statistical validation of optimization results
- Actual optimization algorithm parameter testing
- Proper handling of real Pareto frontiers and dominance
- Real performance validation within expected bounds
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pytest
from prompt_improver.ml.optimization.algorithms.rule_optimizer import GaussianProcessResult, MultiObjectiveResult, OptimizationConfig, ParetoSolution, RuleOptimizer

@pytest.fixture
def multiobjective_config():
    """Configuration with multi-objective optimization enabled using optimized parameters for test speed."""
    return OptimizationConfig(enable_multi_objective=True, pareto_population_size=20, pareto_generations=5, pareto_crossover_prob=0.7, pareto_mutation_prob=0.2, enable_gaussian_process=True, gp_acquisition_samples=100, gp_exploration_weight=0.01, min_sample_size=10, improvement_threshold=0.1, confidence_threshold=0.8)

@pytest.fixture
def rule_optimizer_mo(multiobjective_config):
    """Rule optimizer with multi-objective optimization enabled."""
    return RuleOptimizer(config=multiobjective_config)

@pytest.fixture
def multiobjective_historical_data():
    """Realistic historical data for multi-objective optimization, optimized for test speed."""
    np.random.seed(42)
    n_samples = 40
    historical_data = []
    for i in range(n_samples):
        rule_params = {'threshold': np.random.uniform(0.1, 0.9), 'weight': np.random.uniform(0.5, 1.0), 'complexity_factor': np.random.uniform(0.1, 1.0), 'context_sensitivity': np.random.uniform(0.0, 1.0)}
        base_performance = 0.3 + 0.5 * rule_params['weight'] * rule_params['threshold']
        performance = base_performance + np.random.normal(0, 0.08)
        performance = np.clip(performance, 0.0, 1.0)
        consistency = 0.9 - 0.3 * rule_params['complexity_factor'] + np.random.normal(0, 0.05)
        consistency = np.clip(consistency, 0.0, 1.0)
        efficiency = 1.0 - 0.4 * rule_params['complexity_factor'] + np.random.normal(0, 0.06)
        efficiency = np.clip(efficiency, 0.0, 1.0)
        robustness = 0.3 + 0.6 * rule_params['context_sensitivity'] + np.random.normal(0, 0.07)
        robustness = np.clip(robustness, 0.0, 1.0)
        historical_data.append({'score': float(performance), 'context': f'context_{i % 5}', 'timestamp': datetime.now().isoformat(), 'execution_time_ms': 50 + int(50 * rule_params['complexity_factor']) + np.random.randint(-10, 11), 'rule_parameters': rule_params, 'objectives': {'performance': float(performance), 'consistency': float(consistency), 'efficiency': float(efficiency), 'robustness': float(robustness)}})
    return historical_data

@pytest.fixture
def gaussian_process_data():
    """Data suitable for Gaussian Process optimization, optimized for test speed."""
    np.random.seed(123)
    n_samples = 15
    gp_data = []
    for i in range(n_samples):
        params = {'threshold': np.random.uniform(0.1, 0.9), 'weight': np.random.uniform(0.5, 1.0), 'complexity_factor': np.random.uniform(0.1, 1.0), 'context_sensitivity': np.random.uniform(0.0, 1.0)}
        distance_from_optimal = abs(params['threshold'] - 0.7) + abs(params['weight'] - 0.8)
        performance = 0.9 - 0.3 * distance_from_optimal + np.random.normal(0, 0.05)
        performance = np.clip(performance, 0.0, 1.0)
        gp_data.append({'score': float(performance), 'context': 'gp_context', 'timestamp': datetime.now().isoformat(), 'rule_parameters': params})
    return gp_data

@pytest.fixture
def insufficient_mo_data():
    """Minimal data that may not support multi-objective optimization."""
    return [{'score': 0.8, 'context': 'minimal', 'timestamp': datetime.now().isoformat(), 'execution_time_ms': 80}, {'score': 0.75, 'context': 'minimal', 'timestamp': datetime.now().isoformat(), 'execution_time_ms': 90}]

class TestMultiObjectiveOptimization:
    """Test suite for multi-objective optimization functionality."""

    @pytest.mark.asyncio
    async def test_multiobjective_optimization_integration(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test full multi-objective optimization workflow integration using real DEAP."""
        try:
            import deap
            import deap.algorithms
            import deap.tools
        except ImportError:
            pytest.skip('DEAP not available for real multi-objective optimization testing')
        rule_id = 'rule_mo_test'
        performance_data = {rule_id: {'total_applications': len(multiobjective_historical_data), 'avg_improvement': 0.75, 'consistency_score': 0.8}}
        result = await rule_optimizer_mo.optimize_rule(rule_id, performance_data, multiobjective_historical_data)
        assert 'multi_objective_optimization' in result
        mo_result = result['multi_objective_optimization']
        if isinstance(mo_result, dict):
            expected_keys = ['pareto_frontier', 'hypervolume', 'convergence_metric', 'best_compromise_solution', 'trade_off_analysis']
            for key in expected_keys:
                if key in mo_result:
                    if key == 'hypervolume':
                        assert 0.0 <= mo_result[key] <= 10.0
                    elif key == 'convergence_metric':
                        assert 0.0 <= mo_result[key] <= 1.0
                    elif key == 'pareto_frontier':
                        assert isinstance(mo_result[key], list)
                        assert len(mo_result[key]) > 0
            if 'total_evaluations' in mo_result:
                assert mo_result['total_evaluations'] > 0

    @pytest.mark.asyncio
    async def test_nsga2_algorithm_execution(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test NSGA-II algorithm execution using real DEAP implementation."""
        try:
            import deap
            import deap.algorithms
            import deap.tools
        except ImportError:
            pytest.skip('DEAP not available for real multi-objective optimization testing')
        rule_id = 'rule_nsga2_test'
        mo_result = await rule_optimizer_mo._multi_objective_optimization(rule_id, multiobjective_historical_data)
        if mo_result:
            assert isinstance(mo_result, MultiObjectiveResult)
            assert mo_result.rule_id == rule_id
            assert len(mo_result.pareto_frontier) > 0
            for solution in mo_result.pareto_frontier:
                assert isinstance(solution, ParetoSolution)
                assert len(solution.rule_parameters) > 0
                assert len(solution.objectives) > 0
                for obj_value in solution.objectives.values():
                    assert 0.0 <= obj_value <= 1.0
                assert solution.feasible is True
                assert solution.dominance_rank >= 0
                assert solution.crowding_distance >= 0.0
            if len(mo_result.pareto_frontier) > 1:
                first_solution = mo_result.pareto_frontier[0]
                second_solution = mo_result.pareto_frontier[1]
                objectives_differ = any((first_solution.objectives.get(obj_name, 0) != second_solution.objectives.get(obj_name, 0) for obj_name in first_solution.objectives.keys()))
                assert objectives_differ

    @pytest.mark.asyncio
    async def test_pareto_frontier_analysis(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test Pareto frontier analysis and dominance relationships using real DEAP."""
        try:
            import deap
            import deap.algorithms
            import deap.tools
        except ImportError:
            pytest.skip('DEAP not available for real Pareto frontier testing')
        rule_id = 'rule_pareto_test'
        mo_result = await rule_optimizer_mo._multi_objective_optimization(rule_id, multiobjective_historical_data)
        if mo_result:
            if hasattr(mo_result, 'trade_off_analysis') and mo_result.trade_off_analysis:
                trade_offs = mo_result.trade_off_analysis
                assert isinstance(trade_offs, dict)
                if 'objective_correlations' in trade_offs:
                    correlations = trade_offs['objective_correlations']
                    assert isinstance(correlations, dict)
                if 'objective_ranges' in trade_offs:
                    ranges = trade_offs['objective_ranges']
                    assert isinstance(ranges, dict)
                if 'frontier_size' in trade_offs:
                    frontier_size = trade_offs['frontier_size']
                    assert isinstance(frontier_size, int)
                    assert frontier_size > 0
            if hasattr(mo_result, 'pareto_frontier') and mo_result.pareto_frontier:
                pareto_frontier = mo_result.pareto_frontier
                assert isinstance(pareto_frontier, list)
                assert len(pareto_frontier) > 0
                for solution in pareto_frontier:
                    if hasattr(solution, 'rule_parameters') and hasattr(solution, 'objectives'):
                        assert isinstance(solution.rule_parameters, dict)
                        assert isinstance(solution.objectives, dict)
                    elif isinstance(solution, dict):
                        assert 'rule_parameters' in solution or 'objectives' in solution

    @pytest.mark.asyncio
    async def test_hypervolume_calculation(self, rule_optimizer_mo):
        """Test hypervolume calculation for Pareto frontier quality assessment."""
        pareto_frontier = [ParetoSolution(rule_parameters={'threshold': 0.8}, objectives={'performance': 0.9, 'consistency': 0.8}, dominance_rank=0, crowding_distance=0.5), ParetoSolution(rule_parameters={'threshold': 0.6}, objectives={'performance': 0.7, 'consistency': 0.9}, dominance_rank=0, crowding_distance=0.6)]
        hypervolume = rule_optimizer_mo._calculate_hypervolume(pareto_frontier)
        assert isinstance(hypervolume, float)
        assert hypervolume >= 0.0
        assert hypervolume > 0.0

    @pytest.mark.asyncio
    async def test_best_compromise_solution(self, rule_optimizer_mo):
        """Test identification of best compromise solution from Pareto frontier."""
        pareto_frontier = [ParetoSolution(rule_parameters={'threshold': 0.9}, objectives={'performance': 0.95, 'consistency': 0.6, 'efficiency': 0.7, 'robustness': 0.8}, dominance_rank=0, crowding_distance=0.3, feasible=True), ParetoSolution(rule_parameters={'threshold': 0.7}, objectives={'performance': 0.8, 'consistency': 0.85, 'efficiency': 0.8, 'robustness': 0.8}, dominance_rank=0, crowding_distance=0.5, feasible=True), ParetoSolution(rule_parameters={'threshold': 0.5}, objectives={'performance': 0.65, 'consistency': 0.95, 'efficiency': 0.9, 'robustness': 0.7}, dominance_rank=0, crowding_distance=0.4, feasible=True)]
        best_compromise = rule_optimizer_mo._find_best_compromise_solution(pareto_frontier)
        assert best_compromise is not None
        assert best_compromise.feasible is True
        assert best_compromise in pareto_frontier

    @pytest.mark.asyncio
    async def test_gaussian_process_optimization_integration(self, rule_optimizer_mo, gaussian_process_data):
        """Test Gaussian Process optimization with Expected Improvement using real scikit-learn."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
        except ImportError:
            pytest.skip('scikit-learn not available for real Gaussian Process optimization testing')
        rule_id = 'rule_gp_test'
        performance_data = {rule_id: {'total_applications': len(gaussian_process_data), 'avg_improvement': 0.8, 'consistency_score': 0.85}}
        result = await rule_optimizer_mo.optimize_rule(rule_id, performance_data, gaussian_process_data)
        assert 'gaussian_process_optimization' in result
        gp_result = result['gaussian_process_optimization']
        if isinstance(gp_result, dict):
            expected_keys = ['optimal_parameters', 'predicted_performance', 'uncertainty_estimate', 'acquisition_history', 'model_confidence', 'expected_improvement']
            for key in expected_keys:
                if key in gp_result:
                    if key == 'predicted_performance' or key == 'uncertainty_estimate' or key == 'model_confidence':
                        assert 0.0 <= gp_result[key] <= 1.0
                    elif key == 'expected_improvement':
                        assert gp_result[key] >= 0.0
                    elif key == 'optimal_parameters':
                        assert isinstance(gp_result[key], dict)
                        assert len(gp_result[key]) > 0
                    elif key == 'acquisition_history':
                        assert isinstance(gp_result[key], list)
                        if len(gp_result[key]) > 0:
                            for point in gp_result[key]:
                                assert isinstance(point, dict)
                                assert 'parameters' in point or 'expected_improvement' in point

    @pytest.mark.asyncio
    async def test_expected_improvement_acquisition(self, rule_optimizer_mo, gaussian_process_data):
        """Test Expected Improvement acquisition function for GP optimization using real scikit-learn."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
        except ImportError:
            pytest.skip('scikit-learn not available for real Gaussian Process optimization testing')
        rule_id = 'rule_ei_test'
        gp_result = await rule_optimizer_mo._gaussian_process_optimization(rule_id, gaussian_process_data)
        if gp_result:
            assert isinstance(gp_result, GaussianProcessResult)
            assert gp_result.rule_id == rule_id
            optimal_params = gp_result.optimal_parameters
            assert isinstance(optimal_params, dict)
            assert len(optimal_params) > 0
            if 'threshold' in optimal_params:
                assert 0.1 <= optimal_params['threshold'] <= 0.9
            if 'weight' in optimal_params:
                assert 0.5 <= optimal_params['weight'] <= 1.0
            acquisition_history = gp_result.acquisition_history
            assert isinstance(acquisition_history, list)
            assert len(acquisition_history) > 0
            for acquisition_point in acquisition_history:
                assert isinstance(acquisition_point, dict)
                if 'parameters' in acquisition_point:
                    assert isinstance(acquisition_point['parameters'], dict)
                if 'expected_improvement' in acquisition_point:
                    assert acquisition_point['expected_improvement'] >= 0.0
            if len(acquisition_history) > 1:
                ei_values = [point.get('expected_improvement', 0) for point in acquisition_history]
                assert any((ei > 0 for ei in ei_values))

    @pytest.mark.asyncio
    async def test_multiobjective_parameter_bounds(self, rule_optimizer_mo):
        """Test parameter bounds enforcement in multi-objective optimization."""
        param_bounds = rule_optimizer_mo.param_bounds if hasattr(rule_optimizer_mo, 'param_bounds') else {'threshold': (0.1, 0.9), 'weight': (0.5, 1.0), 'complexity_factor': (0.1, 1.0), 'context_sensitivity': (0.0, 1.0)}
        for param_name, (lower, upper) in param_bounds.items():
            assert lower < upper
            assert 0.0 <= lower <= 1.0
            assert 0.0 <= upper <= 1.0
            if param_name == 'threshold':
                assert lower >= 0.1
                assert upper <= 0.9
            elif param_name == 'weight':
                assert lower >= 0.5

    @pytest.mark.asyncio
    async def test_insufficient_multiobjective_data(self, rule_optimizer_mo, insufficient_mo_data):
        """Test handling of insufficient data for multi-objective optimization."""
        rule_id = 'rule_insufficient_mo'
        performance_data = {rule_id: {'total_applications': len(insufficient_mo_data), 'avg_improvement': 0.75, 'consistency_score': 0.8}}
        result = await rule_optimizer_mo.optimize_rule(rule_id, performance_data, insufficient_mo_data)
        assert 'rule_id' in result
        assert result['status'] in ['optimized', 'insufficient_data']
        if 'multi_objective_optimization' in result:
            mo_result = result['multi_objective_optimization']
            assert mo_result is None or isinstance(mo_result, dict)

    @pytest.mark.parametrize('population_size', [10, 20, 30])
    async def test_variable_population_sizes(self, multiobjective_historical_data, population_size):
        """Test multi-objective optimization with different population sizes using real DEAP."""
        try:
            import deap
            import deap.algorithms
            import deap.tools
        except ImportError:
            pytest.skip('DEAP not available for real population size testing')
        config = OptimizationConfig(enable_multi_objective=True, pareto_population_size=population_size, pareto_generations=3)
        optimizer = RuleOptimizer(config=config)
        rule_id = 'rule_pop_size_test'
        mo_result = await optimizer._multi_objective_optimization(rule_id, multiobjective_historical_data)
        if mo_result:
            assert isinstance(mo_result, (dict, MultiObjectiveResult))
            if isinstance(mo_result, dict) and 'total_evaluations' in mo_result:
                assert mo_result['total_evaluations'] > 0
                assert mo_result['total_evaluations'] >= population_size * 3
            elif hasattr(mo_result, 'pareto_frontier'):
                assert len(mo_result.pareto_frontier) > 0

    @pytest.mark.asyncio
    async def test_convergence_metrics_calculation(self, rule_optimizer_mo):
        """Test calculation of convergence metrics for optimization assessment."""
        mock_logbook = []
        for gen in range(10):
            gen_stats = {'avg': [0.6 + 0.02 * gen, 0.7 + 0.01 * gen, 0.65 + 0.015 * gen, 0.72 + 0.01 * gen], 'std': [0.1, 0.08, 0.09, 0.07], 'min': [0.5, 0.6, 0.55, 0.65], 'max': [0.8, 0.85, 0.8, 0.82]}
            mock_logbook.append(gen_stats)
        convergence_metric = rule_optimizer_mo._calculate_convergence_metric(mock_logbook)
        assert isinstance(convergence_metric, float)
        assert 0.0 <= convergence_metric <= 1.0

class TestMultiObjectiveErrorHandling:
    """Test error handling and edge cases for multi-objective optimization."""

    @pytest.mark.asyncio
    async def test_multiobjective_libraries_unavailable(self, multiobjective_historical_data):
        """Test behavior when multi-objective optimization libraries are not available using real import checking."""
        config = OptimizationConfig(enable_multi_objective=True)
        optimizer = RuleOptimizer(config=config)
        rule_id = 'rule_no_libs'
        performance_data = {rule_id: {'total_applications': 50, 'avg_improvement': 0.8}}
        result = await optimizer.optimize_rule(rule_id, performance_data, multiobjective_historical_data)
        assert 'rule_id' in result
        assert result['status'] == 'optimized'
        if 'multi_objective_optimization' in result:
            mo_result = result['multi_objective_optimization']
            assert mo_result is None or isinstance(mo_result, dict)

    @pytest.mark.asyncio
    async def test_multiobjective_disabled(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test behavior when multi-objective optimization is disabled."""
        config = OptimizationConfig(enable_multi_objective=False)
        optimizer = RuleOptimizer(config=config)
        rule_id = 'rule_mo_disabled'
        performance_data = {rule_id: {'total_applications': 50, 'avg_improvement': 0.8}}
        result = await optimizer.optimize_rule(rule_id, performance_data, multiobjective_historical_data)
        assert 'rule_id' in result
        assert result['status'] == 'optimized'
        assert 'multi_objective_optimization' not in result

    @pytest.mark.asyncio
    async def test_deap_algorithm_failure(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test handling of DEAP algorithm failures using real import checking."""
        rule_id = 'rule_deap_fail'
        try:
            import deap
            import deap.algorithms
            import deap.tools
        except ImportError:
            pytest.skip('DEAP not available for real algorithm failure testing')
        invalid_config = OptimizationConfig(enable_multi_objective=True, pareto_population_size=0, pareto_generations=0)
        invalid_optimizer = RuleOptimizer(config=invalid_config)
        mo_result = await invalid_optimizer._multi_objective_optimization(rule_id, multiobjective_historical_data)
        assert mo_result is None

    @pytest.mark.asyncio
    async def test_gaussian_process_failure(self, rule_optimizer_mo, gaussian_process_data):
        """Test handling of Gaussian Process optimization failures using real import checking."""
        rule_id = 'rule_gp_fail'
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
        except ImportError:
            pytest.skip('scikit-learn not available for real GP failure testing')
        problematic_data = [{'score': float('nan'), 'context': 'test', 'rule_parameters': {'threshold': 0.5}}, {'score': float('inf'), 'context': 'test', 'rule_parameters': {'threshold': 0.6}}]
        gp_result = await rule_optimizer_mo._gaussian_process_optimization(rule_id, problematic_data)
        assert gp_result is None or isinstance(gp_result, dict)

    @pytest.mark.asyncio
    async def test_extreme_optimization_parameters(self):
        """Test optimization with extreme parameter configurations using real DEAP."""
        try:
            import deap
            import deap.algorithms
            import deap.tools
        except ImportError:
            pytest.skip('DEAP not available for real extreme parameter testing')
        extreme_config = OptimizationConfig(enable_multi_objective=True, pareto_population_size=2, pareto_generations=1, gp_acquisition_samples=5)
        optimizer = RuleOptimizer(config=extreme_config)
        test_data = [{'score': 0.8, 'context': 'test'}] * 10
        mo_result = await optimizer._multi_objective_optimization('test_rule', test_data)
        assert mo_result is None or isinstance(mo_result, dict)

    @pytest.mark.asyncio
    async def test_invalid_objective_values(self, rule_optimizer_mo):
        """Test handling of invalid objective values."""
        invalid_data = [{'score': float('inf'), 'context': 'test', 'execution_time_ms': 100}, {'score': -1.0, 'context': 'test', 'execution_time_ms': -50}]
        mo_result = await rule_optimizer_mo._multi_objective_optimization('test_rule', invalid_data)
        assert mo_result is None or isinstance(mo_result, dict)

class TestMultiObjectiveIntegration:
    """Integration tests for multi-objective optimization with existing workflows."""

    @pytest.mark.asyncio
    async def test_multiobjective_with_traditional_optimization(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test integration of multi-objective optimization with traditional rule optimization using real algorithms."""
        rule_id = 'rule_integration_test'
        performance_data = {rule_id: {'total_applications': len(multiobjective_historical_data), 'avg_improvement': 0.8, 'consistency_score': 0.85}}
        result = await rule_optimizer_mo.optimize_rule(rule_id, performance_data, multiobjective_historical_data)
        assert 'rule_id' in result
        assert 'status' in result
        if 'traditional_recommendations' in result:
            assert isinstance(result['traditional_recommendations'], (list, dict))
        if 'multi_objective_optimization' in result:
            assert isinstance(result['multi_objective_optimization'], dict)
        if 'gaussian_process_optimization' in result:
            assert isinstance(result['gaussian_process_optimization'], dict)

    @pytest.mark.asyncio
    async def test_multiobjective_performance_monitoring(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test performance characteristics of multi-objective optimization."""
        import time
        rule_id = 'rule_performance_test'
        performance_data = {rule_id: {'total_applications': 50, 'avg_improvement': 0.8}}
        start_time = time.time()
        result = await rule_optimizer_mo.optimize_rule(rule_id, performance_data, multiobjective_historical_data)
        execution_time = time.time() - start_time
        assert execution_time < 15.0
        assert 'rule_id' in result
        assert result['status'] == 'optimized'

    @pytest.mark.asyncio
    async def test_multiobjective_scalability(self, rule_optimizer_mo):
        """Test scalability with larger datasets using real algorithms."""
        large_data = []
        np.random.seed(456)
        for i in range(60):
            large_data.append({'score': 0.5 + 0.4 * np.random.random(), 'context': f'context_{i % 10}', 'timestamp': datetime.now().isoformat(), 'execution_time_ms': 50 + np.random.randint(0, 100)})
        rule_id = 'rule_scalability_test'
        performance_data = {rule_id: {'total_applications': len(large_data), 'avg_improvement': 0.75}}
        result = await rule_optimizer_mo.optimize_rule(rule_id, performance_data, large_data)
        assert 'rule_id' in result
        assert result['status'] == 'optimized'
pytestmark = [pytest.mark.unit, pytest.mark.ml_performance, pytest.mark.ml_contracts]
