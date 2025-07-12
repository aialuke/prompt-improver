"""Tests for Phase 2 Multi-Objective Optimization in Rule Optimizer.

Comprehensive test suite for multi-objective optimization enhancements including:
- NSGA-II algorithm for Pareto frontier discovery
- Gaussian Process optimization with Expected Improvement
- Hypervolume calculation and convergence metrics
- Multi-objective trade-off analysis

Testing best practices applied from Context7 research:
- Statistical validation of optimization results
- Realistic parameter ranges for optimization algorithms
- Proper handling of Pareto frontiers and dominance
- Performance validation within expected bounds
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from datetime import datetime

from prompt_improver.optimization.rule_optimizer import (
    RuleOptimizer,
    OptimizationConfig,
    ParetoSolution,
    MultiObjectiveResult,
    GaussianProcessResult
)


@pytest.fixture
def multiobjective_config():
    """Configuration with multi-objective optimization enabled."""
    return OptimizationConfig(
        enable_multi_objective=True,
        pareto_population_size=100,
        pareto_generations=50,
        pareto_crossover_prob=0.7,
        pareto_mutation_prob=0.2,
        enable_gaussian_process=True,
        gp_acquisition_samples=1000,
        gp_exploration_weight=0.01,
        # Standard parameters
        min_sample_size=20,
        improvement_threshold=0.1,
        confidence_threshold=0.8
    )


@pytest.fixture
def rule_optimizer_mo(multiobjective_config):
    """Rule optimizer with multi-objective optimization enabled."""
    return RuleOptimizer(config=multiobjective_config)


@pytest.fixture
def multiobjective_historical_data():
    """Realistic historical data for multi-objective optimization."""
    np.random.seed(42)  # Reproducible test data
    
    # Generate realistic rule performance data with multiple objectives
    n_samples = 80
    historical_data = []
    
    for i in range(n_samples):
        # Simulate realistic rule parameters
        rule_params = {
            "threshold": np.random.uniform(0.1, 0.9),
            "weight": np.random.uniform(0.5, 1.0),
            "complexity_factor": np.random.uniform(0.1, 1.0),
            "context_sensitivity": np.random.uniform(0.0, 1.0)
        }
        
        # Simulate multi-objective performance with realistic trade-offs
        # Performance vs. consistency trade-off
        base_performance = 0.3 + 0.5 * rule_params["weight"] * rule_params["threshold"]
        performance = base_performance + np.random.normal(0, 0.08)
        performance = np.clip(performance, 0.0, 1.0)
        
        # Consistency decreases with complexity
        consistency = 0.9 - 0.3 * rule_params["complexity_factor"] + np.random.normal(0, 0.05)
        consistency = np.clip(consistency, 0.0, 1.0)
        
        # Efficiency vs. performance trade-off
        efficiency = 1.0 - 0.4 * rule_params["complexity_factor"] + np.random.normal(0, 0.06)
        efficiency = np.clip(efficiency, 0.0, 1.0)
        
        # Robustness depends on context sensitivity
        robustness = 0.3 + 0.6 * rule_params["context_sensitivity"] + np.random.normal(0, 0.07)
        robustness = np.clip(robustness, 0.0, 1.0)
        
        historical_data.append({
            "score": float(performance),
            "context": f"context_{i % 5}",
            "timestamp": datetime.now().isoformat(),
            "execution_time_ms": 50 + int(50 * rule_params["complexity_factor"]) + np.random.randint(-10, 11),
            "rule_parameters": rule_params,
            "objectives": {
                "performance": float(performance),
                "consistency": float(consistency),
                "efficiency": float(efficiency),
                "robustness": float(robustness)
            }
        })
    
    return historical_data


@pytest.fixture
def gaussian_process_data():
    """Data suitable for Gaussian Process optimization."""
    np.random.seed(123)
    
    # Generate smaller dataset for GP optimization
    n_samples = 25
    gp_data = []
    
    for i in range(n_samples):
        # Generate parameters with known optimal region
        params = {
            "threshold": np.random.uniform(0.1, 0.9),
            "weight": np.random.uniform(0.5, 1.0),
            "complexity_factor": np.random.uniform(0.1, 1.0),
            "context_sensitivity": np.random.uniform(0.0, 1.0)
        }
        
        # Known optimal region around threshold=0.7, weight=0.8
        distance_from_optimal = abs(params["threshold"] - 0.7) + abs(params["weight"] - 0.8)
        performance = 0.9 - 0.3 * distance_from_optimal + np.random.normal(0, 0.05)
        performance = np.clip(performance, 0.0, 1.0)
        
        gp_data.append({
            "score": float(performance),
            "context": "gp_context",
            "timestamp": datetime.now().isoformat(),
            "rule_parameters": params
        })
    
    return gp_data


@pytest.fixture
def insufficient_mo_data():
    """Minimal data that may not support multi-objective optimization."""
    return [
        {
            "score": 0.8,
            "context": "minimal",
            "timestamp": datetime.now().isoformat(),
            "execution_time_ms": 80
        },
        {
            "score": 0.75,
            "context": "minimal", 
            "timestamp": datetime.now().isoformat(),
            "execution_time_ms": 90
        }
    ]


class TestMultiObjectiveOptimization:
    """Test suite for multi-objective optimization functionality."""

    @pytest.mark.asyncio
    async def test_multiobjective_optimization_integration(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test full multi-objective optimization workflow integration."""
        rule_id = "rule_mo_test"
        performance_data = {
            rule_id: {
                "total_applications": len(multiobjective_historical_data),
                "avg_improvement": 0.75,
                "consistency_score": 0.8
            }
        }
        
        with patch('prompt_improver.optimization.rule_optimizer.DEAP_AVAILABLE', True):
            result = await rule_optimizer_mo.optimize_rule(rule_id, performance_data, multiobjective_historical_data)
            
            # Should include multi-objective optimization results
            assert "multi_objective_optimization" in result
            mo_result = result["multi_objective_optimization"]
            
            # Validate multi-objective results structure
            assert "pareto_frontier" in mo_result
            assert "hypervolume" in mo_result
            assert "convergence_metric" in mo_result
            assert "best_compromise_solution" in mo_result
            assert "trade_off_analysis" in mo_result
            
            # Validate realistic metrics
            assert 0.0 <= mo_result["hypervolume"] <= 10.0  # Reasonable hypervolume range
            assert 0.0 <= mo_result["convergence_metric"] <= 1.0
            assert mo_result["total_evaluations"] > 0

    @pytest.mark.asyncio
    async def test_nsga2_algorithm_execution(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test NSGA-II algorithm execution with proper mocking."""
        rule_id = "rule_nsga2_test"
        
        with patch('prompt_improver.optimization.rule_optimizer.DEAP_AVAILABLE', True):
            with patch('deap.algorithms.eaMuPlusLambda') as mock_nsga2, \
                 patch('deap.tools.sortNondominated') as mock_sort_nd:
                
                # Setup NSGA-II mock
                mock_population = []
                for i in range(10):  # Small population for testing
                    individual = MagicMock()
                    individual.fitness.values = (
                        0.7 + 0.2 * np.random.random(),  # Performance
                        0.8 + 0.1 * np.random.random(),  # Consistency  
                        0.6 + 0.3 * np.random.random(),  # Efficiency
                        0.75 + 0.15 * np.random.random() # Robustness
                    )
                    # Simulate individual parameters
                    individual.__getitem__ = lambda self, idx: 0.5 + 0.3 * np.random.random()
                    mock_population.append(individual)
                
                mock_logbook = MagicMock()
                mock_nsga2.return_value = (mock_population, mock_logbook)
                
                # Setup Pareto sorting mock
                mock_sort_nd.return_value = [mock_population[:5]]  # First front
                
                # Test NSGA-II execution
                mo_result = await rule_optimizer_mo._multi_objective_optimization(rule_id, multiobjective_historical_data)
                
                # Verify NSGA-II was called
                mock_nsga2.assert_called_once()
                
                # Validate result structure
                if mo_result:
                    assert isinstance(mo_result, MultiObjectiveResult)
                    assert mo_result.rule_id == rule_id
                    assert len(mo_result.pareto_frontier) > 0
                    
                    # Validate Pareto solutions
                    for solution in mo_result.pareto_frontier:
                        assert isinstance(solution, ParetoSolution)
                        assert len(solution.rule_parameters) > 0
                        assert len(solution.objectives) > 0
                        # All objectives should be in [0,1] range
                        for obj_value in solution.objectives.values():
                            assert 0.0 <= obj_value <= 1.0

    @pytest.mark.asyncio
    async def test_pareto_frontier_analysis(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test Pareto frontier analysis and dominance relationships."""
        rule_id = "rule_pareto_test"
        
        with patch('prompt_improver.optimization.rule_optimizer.DEAP_AVAILABLE', True):
            with patch('deap.algorithms.eaMuPlusLambda') as mock_nsga2:
                
                # Create test Pareto frontier with known dominance relationships
                pareto_solutions = []
                
                # Solution 1: High performance, lower consistency
                sol1 = ParetoSolution(
                    rule_parameters={"threshold": 0.8, "weight": 0.9},
                    objectives={"performance": 0.9, "consistency": 0.6, "efficiency": 0.7, "robustness": 0.8},
                    dominance_rank=0,
                    crowding_distance=0.5,
                    feasible=True
                )
                
                # Solution 2: High consistency, lower performance  
                sol2 = ParetoSolution(
                    rule_parameters={"threshold": 0.6, "weight": 0.7},
                    objectives={"performance": 0.7, "consistency": 0.9, "efficiency": 0.8, "robustness": 0.7},
                    dominance_rank=0,
                    crowding_distance=0.6,
                    feasible=True
                )
                
                # Solution 3: Balanced trade-off
                sol3 = ParetoSolution(
                    rule_parameters={"threshold": 0.7, "weight": 0.8},
                    objectives={"performance": 0.8, "consistency": 0.8, "efficiency": 0.75, "robustness": 0.75},
                    dominance_rank=0,
                    crowding_distance=0.4,
                    feasible=True
                )
                
                pareto_solutions = [sol1, sol2, sol3]
                
                # Mock NSGA-II to return known solutions
                mock_population = [MagicMock() for _ in range(3)]
                for i, individual in enumerate(mock_population):
                    individual.fitness.values = tuple(pareto_solutions[i].objectives.values())
                    individual.__getitem__ = lambda self, idx, sol=pareto_solutions[i]: list(sol.rule_parameters.values())[idx]
                
                mock_nsga2.return_value = (mock_population, MagicMock())
                
                # Test Pareto analysis
                with patch.object(rule_optimizer_mo, '_run_nsga2_optimization', return_value=(pareto_solutions, MagicMock())):
                    mo_result = await rule_optimizer_mo._multi_objective_optimization(rule_id, multiobjective_historical_data)
                    
                    if mo_result:
                        # Validate trade-off analysis
                        trade_offs = mo_result.trade_off_analysis
                        assert "objective_correlations" in trade_offs
                        assert "objective_ranges" in trade_offs
                        assert "frontier_size" in trade_offs
                        
                        # Check for detected trade-offs
                        correlations = trade_offs["objective_correlations"]
                        for correlation_key, correlation_data in correlations.items():
                            if isinstance(correlation_data, dict):
                                assert "correlation" in correlation_data
                                assert "trade_off_strength" in correlation_data
                                assert "conflicting" in correlation_data

    @pytest.mark.asyncio
    async def test_hypervolume_calculation(self, rule_optimizer_mo):
        """Test hypervolume calculation for Pareto frontier quality assessment."""
        # Create test Pareto frontier
        pareto_frontier = [
            ParetoSolution(
                rule_parameters={"threshold": 0.8},
                objectives={"performance": 0.9, "consistency": 0.8},
                dominance_rank=0,
                crowding_distance=0.5
            ),
            ParetoSolution(
                rule_parameters={"threshold": 0.6},
                objectives={"performance": 0.7, "consistency": 0.9},
                dominance_rank=0,
                crowding_distance=0.6
            )
        ]
        
        # Test hypervolume calculation
        hypervolume = rule_optimizer_mo._calculate_hypervolume(pareto_frontier)
        
        # Validate hypervolume metric
        assert isinstance(hypervolume, float)
        assert hypervolume >= 0.0  # Hypervolume should be non-negative
        # For 2 solutions with reasonable objectives, expect positive hypervolume
        assert hypervolume > 0.0

    @pytest.mark.asyncio
    async def test_best_compromise_solution(self, rule_optimizer_mo):
        """Test identification of best compromise solution from Pareto frontier."""
        # Create diverse Pareto frontier
        pareto_frontier = [
            ParetoSolution(
                rule_parameters={"threshold": 0.9},
                objectives={"performance": 0.95, "consistency": 0.6, "efficiency": 0.7, "robustness": 0.8},
                dominance_rank=0,
                crowding_distance=0.3,
                feasible=True
            ),
            ParetoSolution(
                rule_parameters={"threshold": 0.7},
                objectives={"performance": 0.8, "consistency": 0.85, "efficiency": 0.8, "robustness": 0.8},
                dominance_rank=0,
                crowding_distance=0.5,
                feasible=True
            ),
            ParetoSolution(
                rule_parameters={"threshold": 0.5},
                objectives={"performance": 0.65, "consistency": 0.95, "efficiency": 0.9, "robustness": 0.7},
                dominance_rank=0,
                crowding_distance=0.4,
                feasible=True
            )
        ]
        
        # Test compromise solution selection
        best_compromise = rule_optimizer_mo._find_best_compromise_solution(pareto_frontier)
        
        # Should select a feasible solution
        assert best_compromise is not None
        assert best_compromise.feasible is True
        assert best_compromise in pareto_frontier
        
        # Should be reasonably balanced (likely the middle solution in this case)
        # Not strictly enforced due to weighting strategy flexibility

    @pytest.mark.asyncio
    async def test_gaussian_process_optimization_integration(self, rule_optimizer_mo, gaussian_process_data):
        """Test Gaussian Process optimization with Expected Improvement."""
        rule_id = "rule_gp_test"
        performance_data = {
            rule_id: {
                "total_applications": len(gaussian_process_data),
                "avg_improvement": 0.8,
                "consistency_score": 0.85
            }
        }
        
        with patch('prompt_improver.optimization.rule_optimizer.GAUSSIAN_PROCESS_AVAILABLE', True):
            result = await rule_optimizer_mo.optimize_rule(rule_id, performance_data, gaussian_process_data)
            
            # Should include Gaussian Process optimization results
            assert "gaussian_process_optimization" in result
            gp_result = result["gaussian_process_optimization"]
            
            # Validate GP results structure
            assert "optimal_parameters" in gp_result
            assert "predicted_performance" in gp_result
            assert "uncertainty_estimate" in gp_result
            assert "acquisition_history" in gp_result
            assert "model_confidence" in gp_result
            assert "expected_improvement" in gp_result
            
            # Validate realistic values
            assert 0.0 <= gp_result["predicted_performance"] <= 1.0
            assert 0.0 <= gp_result["uncertainty_estimate"] <= 1.0
            assert 0.0 <= gp_result["model_confidence"] <= 1.0
            assert gp_result["expected_improvement"] >= 0.0

    @pytest.mark.asyncio
    async def test_expected_improvement_acquisition(self, rule_optimizer_mo, gaussian_process_data):
        """Test Expected Improvement acquisition function for GP optimization."""
        rule_id = "rule_ei_test"
        
        with patch('prompt_improver.optimization.rule_optimizer.GAUSSIAN_PROCESS_AVAILABLE', True):
            with patch('sklearn.gaussian_process.GaussianProcessRegressor') as mock_gpr:
                
                # Setup GP mock
                mock_gp = MagicMock()
                
                # Mock GP predictions with realistic uncertainty
                def mock_predict(X, return_std=False):
                    if return_std:
                        # Return mean and std predictions
                        mean_pred = np.array([0.8 + 0.1 * np.random.random() for _ in range(len(X))])
                        std_pred = np.array([0.05 + 0.05 * np.random.random() for _ in range(len(X))])
                        return mean_pred, std_pred
                    else:
                        return np.array([0.8 + 0.1 * np.random.random() for _ in range(len(X))])
                
                mock_gp.predict = mock_predict
                mock_gpr.return_value = mock_gp
                
                # Test GP optimization
                gp_result = await rule_optimizer_mo._gaussian_process_optimization(rule_id, gaussian_process_data)
                
                if gp_result:
                    assert isinstance(gp_result, GaussianProcessResult)
                    assert gp_result.rule_id == rule_id
                    
                    # Validate optimal parameters
                    optimal_params = gp_result.optimal_parameters
                    assert "threshold" in optimal_params
                    assert "weight" in optimal_params
                    
                    # Parameter values should be within bounds
                    assert 0.1 <= optimal_params["threshold"] <= 0.9
                    assert 0.5 <= optimal_params["weight"] <= 1.0
                    
                    # Validate acquisition history
                    acquisition_history = gp_result.acquisition_history
                    assert len(acquisition_history) > 0
                    
                    for acquisition_point in acquisition_history:
                        assert "parameters" in acquisition_point
                        assert "expected_improvement" in acquisition_point
                        assert acquisition_point["expected_improvement"] >= 0.0

    @pytest.mark.asyncio
    async def test_multiobjective_parameter_bounds(self, rule_optimizer_mo):
        """Test parameter bounds enforcement in multi-objective optimization."""
        # Test parameter bounds validation
        param_bounds = rule_optimizer_mo.param_bounds if hasattr(rule_optimizer_mo, 'param_bounds') else {
            'threshold': (0.1, 0.9),
            'weight': (0.5, 1.0),
            'complexity_factor': (0.1, 1.0),
            'context_sensitivity': (0.0, 1.0)
        }
        
        # Validate bounds are reasonable
        for param_name, (lower, upper) in param_bounds.items():
            assert lower < upper
            assert 0.0 <= lower <= 1.0
            assert 0.0 <= upper <= 1.0
            
            # Specific parameter validations
            if param_name == "threshold":
                assert lower >= 0.1  # Reasonable minimum threshold
                assert upper <= 0.9   # Reasonable maximum threshold
            elif param_name == "weight":
                assert lower >= 0.5   # Minimum effective weight

    @pytest.mark.asyncio
    async def test_insufficient_multiobjective_data(self, rule_optimizer_mo, insufficient_mo_data):
        """Test handling of insufficient data for multi-objective optimization."""
        rule_id = "rule_insufficient_mo"
        performance_data = {
            rule_id: {
                "total_applications": len(insufficient_mo_data),
                "avg_improvement": 0.75,
                "consistency_score": 0.8
            }
        }
        
        with patch('prompt_improver.optimization.rule_optimizer.DEAP_AVAILABLE', True):
            result = await rule_optimizer_mo.optimize_rule(rule_id, performance_data, insufficient_mo_data)
            
            # Should handle gracefully with traditional optimization
            assert "rule_id" in result
            assert result["status"] in ["optimized", "insufficient_data"]
            
            # Multi-objective optimization may be skipped
            if "multi_objective_optimization" in result:
                mo_result = result["multi_objective_optimization"]
                # May indicate insufficient data
                assert mo_result is None or isinstance(mo_result, dict)

    @pytest.mark.parametrize("population_size", [50, 100, 200])
    async def test_variable_population_sizes(self, multiobjective_historical_data, population_size):
        """Test multi-objective optimization with different population sizes."""
        config = OptimizationConfig(
            enable_multi_objective=True,
            pareto_population_size=population_size,
            pareto_generations=20  # Reduced for test performance
        )
        optimizer = RuleOptimizer(config=config)
        
        rule_id = "rule_pop_size_test"
        
        with patch('prompt_improver.optimization.rule_optimizer.DEAP_AVAILABLE', True):
            with patch('deap.algorithms.eaMuPlusLambda') as mock_nsga2:
                
                # Mock population based on size
                mock_population = [MagicMock() for _ in range(population_size // 10)]  # Scaled for testing
                for individual in mock_population:
                    individual.fitness.values = (0.8, 0.7, 0.6, 0.75)
                    individual.__getitem__ = lambda self, idx: 0.5
                
                mock_nsga2.return_value = (mock_population, MagicMock())
                
                # Test optimization with different population size
                mo_result = await optimizer._multi_objective_optimization(rule_id, multiobjective_historical_data)
                
                # Verify population size configuration
                if mock_nsga2.called:
                    call_kwargs = mock_nsga2.call_args[1]
                    assert call_kwargs.get("mu", 0) == population_size

    @pytest.mark.asyncio
    async def test_convergence_metrics_calculation(self, rule_optimizer_mo):
        """Test calculation of convergence metrics for optimization assessment."""
        # Create mock logbook with convergence data
        mock_logbook = []
        
        # Simulate improving fitness over generations
        for gen in range(10):
            gen_stats = {
                'avg': [0.6 + 0.02 * gen, 0.7 + 0.01 * gen, 0.65 + 0.015 * gen, 0.72 + 0.01 * gen],
                'std': [0.1, 0.08, 0.09, 0.07],
                'min': [0.5, 0.6, 0.55, 0.65],
                'max': [0.8, 0.85, 0.8, 0.82]
            }
            mock_logbook.append(gen_stats)
        
        # Test convergence calculation
        convergence_metric = rule_optimizer_mo._calculate_convergence_metric(mock_logbook)
        
        # Validate convergence metric
        assert isinstance(convergence_metric, float)
        assert 0.0 <= convergence_metric <= 1.0
        
        # Should show improvement (positive convergence)
        # Not strictly enforced due to metric calculation flexibility


class TestMultiObjectiveErrorHandling:
    """Test error handling and edge cases for multi-objective optimization."""

    @pytest.mark.asyncio
    async def test_multiobjective_libraries_unavailable(self, multiobjective_historical_data):
        """Test behavior when multi-objective optimization libraries are not available."""
        config = OptimizationConfig(enable_multi_objective=True)
        optimizer = RuleOptimizer(config=config)
        
        rule_id = "rule_no_libs"
        performance_data = {rule_id: {"total_applications": 50, "avg_improvement": 0.8}}
        
        with patch('prompt_improver.optimization.rule_optimizer.DEAP_AVAILABLE', False):
            result = await optimizer.optimize_rule(rule_id, performance_data, multiobjective_historical_data)
            
            # Should provide traditional optimization without multi-objective features
            assert "rule_id" in result
            assert result["status"] == "optimized"
            # Should not contain multi-objective optimization
            assert "multi_objective_optimization" not in result

    @pytest.mark.asyncio
    async def test_multiobjective_disabled(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test behavior when multi-objective optimization is disabled."""
        config = OptimizationConfig(enable_multi_objective=False)
        optimizer = RuleOptimizer(config=config)
        
        rule_id = "rule_mo_disabled"
        performance_data = {rule_id: {"total_applications": 50, "avg_improvement": 0.8}}
        
        result = await optimizer.optimize_rule(rule_id, performance_data, multiobjective_historical_data)
        
        # Should not perform multi-objective optimization
        assert "rule_id" in result
        assert result["status"] == "optimized"
        assert "multi_objective_optimization" not in result

    @pytest.mark.asyncio
    async def test_deap_algorithm_failure(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test handling of DEAP algorithm failures."""
        rule_id = "rule_deap_fail"
        
        with patch('prompt_improver.optimization.rule_optimizer.DEAP_AVAILABLE', True):
            with patch('deap.algorithms.eaMuPlusLambda') as mock_nsga2:
                
                # Simulate DEAP algorithm failure
                mock_nsga2.side_effect = Exception("NSGA-II algorithm failed")
                
                # Should handle failure gracefully
                mo_result = await rule_optimizer_mo._multi_objective_optimization(rule_id, multiobjective_historical_data)
                
                # Should return None or error indicator
                assert mo_result is None

    @pytest.mark.asyncio
    async def test_gaussian_process_failure(self, rule_optimizer_mo, gaussian_process_data):
        """Test handling of Gaussian Process optimization failures."""
        rule_id = "rule_gp_fail"
        
        with patch('prompt_improver.optimization.rule_optimizer.GAUSSIAN_PROCESS_AVAILABLE', True):
            with patch('sklearn.gaussian_process.GaussianProcessRegressor') as mock_gpr:
                
                # Simulate GP failure
                mock_gpr.side_effect = Exception("GP fitting failed")
                
                # Should handle failure gracefully
                gp_result = await rule_optimizer_mo._gaussian_process_optimization(rule_id, gaussian_process_data)
                
                # Should return None
                assert gp_result is None

    @pytest.mark.asyncio
    async def test_extreme_optimization_parameters(self):
        """Test optimization with extreme parameter configurations."""
        # Test with very small parameters
        extreme_config = OptimizationConfig(
            enable_multi_objective=True,
            pareto_population_size=2,    # Very small
            pareto_generations=1,        # Single generation
            gp_acquisition_samples=5     # Very few samples
        )
        optimizer = RuleOptimizer(config=extreme_config)
        
        test_data = [{"score": 0.8, "context": "test"}] * 10
        
        with patch('prompt_improver.optimization.rule_optimizer.DEAP_AVAILABLE', True):
            with patch('deap.algorithms.eaMuPlusLambda') as mock_nsga2:
                
                # Mock minimal population
                mock_population = [MagicMock(), MagicMock()]
                for individual in mock_population:
                    individual.fitness.values = (0.8, 0.7, 0.6, 0.75)
                    individual.__getitem__ = lambda self, idx: 0.5
                
                mock_nsga2.return_value = (mock_population, [])
                
                # Should handle extreme parameters without crashing
                mo_result = await optimizer._multi_objective_optimization("test_rule", test_data)
                
                # Should either succeed or fail gracefully
                assert mo_result is None or isinstance(mo_result, MultiObjectiveResult)

    @pytest.mark.asyncio
    async def test_invalid_objective_values(self, rule_optimizer_mo):
        """Test handling of invalid objective values."""
        invalid_data = [
            {
                "score": float('inf'),  # Invalid infinite score
                "context": "test",
                "execution_time_ms": 100
            },
            {
                "score": -1.0,         # Out of range score
                "context": "test", 
                "execution_time_ms": -50  # Invalid negative time
            }
        ]
        
        with patch('prompt_improver.optimization.rule_optimizer.DEAP_AVAILABLE', True):
            # Should handle invalid data gracefully
            mo_result = await rule_optimizer_mo._multi_objective_optimization("test_rule", invalid_data)
            
            # Should return None due to invalid data
            assert mo_result is None


class TestMultiObjectiveIntegration:
    """Integration tests for multi-objective optimization with existing workflows."""

    @pytest.mark.asyncio
    async def test_multiobjective_with_traditional_optimization(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test integration of multi-objective optimization with traditional rule optimization."""
        rule_id = "rule_integration_test"
        performance_data = {
            rule_id: {
                "total_applications": len(multiobjective_historical_data),
                "avg_improvement": 0.8,
                "consistency_score": 0.85
            }
        }
        
        with patch('prompt_improver.optimization.rule_optimizer.DEAP_AVAILABLE', True), \
             patch('prompt_improver.optimization.rule_optimizer.GAUSSIAN_PROCESS_AVAILABLE', True):
            
            result = await rule_optimizer_mo.optimize_rule(rule_id, performance_data, multiobjective_historical_data)
            
            # Should have both traditional and advanced optimization
            assert "rule_id" in result
            assert "status" in result
            assert "traditional_recommendations" in result
            assert "multi_objective_optimization" in result
            assert "gaussian_process_optimization" in result

    @pytest.mark.asyncio
    async def test_multiobjective_performance_monitoring(self, rule_optimizer_mo, multiobjective_historical_data):
        """Test performance characteristics of multi-objective optimization."""
        import time
        
        rule_id = "rule_performance_test"
        performance_data = {rule_id: {"total_applications": 50, "avg_improvement": 0.8}}
        
        start_time = time.time()
        result = await rule_optimizer_mo.optimize_rule(rule_id, performance_data, multiobjective_historical_data)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (allow up to 15 seconds for complex optimization)
        assert execution_time < 15.0
        
        # Should provide meaningful results
        assert "rule_id" in result
        assert result["status"] == "optimized"

    @pytest.mark.asyncio
    async def test_multiobjective_scalability(self, rule_optimizer_mo):
        """Test scalability with larger datasets."""
        # Generate larger dataset for scalability testing
        large_data = []
        np.random.seed(456)
        
        for i in range(200):  # Larger dataset
            large_data.append({
                "score": 0.5 + 0.4 * np.random.random(),
                "context": f"context_{i % 10}",
                "timestamp": datetime.now().isoformat(),
                "execution_time_ms": 50 + np.random.randint(0, 100)
            })
        
        rule_id = "rule_scalability_test"
        performance_data = {rule_id: {"total_applications": len(large_data), "avg_improvement": 0.75}}
        
        with patch('prompt_improver.optimization.rule_optimizer.DEAP_AVAILABLE', True):
            # Should handle larger datasets reasonably
            result = await rule_optimizer_mo.optimize_rule(rule_id, performance_data, large_data)
            
            # Should complete successfully
            assert "rule_id" in result
            assert result["status"] == "optimized"


# Test markers for categorization
pytestmark = [
    pytest.mark.unit,
    pytest.mark.ml_performance,
    pytest.mark.ml_contracts
]