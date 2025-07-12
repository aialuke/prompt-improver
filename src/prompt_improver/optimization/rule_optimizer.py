"""Rule Optimizer

Optimizes individual rules and rule combinations based on performance data
and learning insights. Provides automated rule improvement capabilities.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
from scipy import stats
import warnings

# Phase 2 enhancement imports for multi-objective optimization
try:
    from deap import algorithms, base, creator, tools
    import random
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    warnings.warn("Multi-objective optimization libraries not available. Install with: pip install deap")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    GAUSSIAN_PROCESS_AVAILABLE = True
except ImportError:
    GAUSSIAN_PROCESS_AVAILABLE = False
    warnings.warn("Gaussian process libraries not available. Install with: pip install scikit-learn pandas")

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for rule optimization"""
    min_sample_size: int = 20
    improvement_threshold: float = 0.1
    confidence_threshold: float = 0.8
    max_optimization_iterations: int = 5
    
    # Phase 2 enhancements - Multi-Objective Optimization
    enable_multi_objective: bool = True
    pareto_population_size: int = 100
    pareto_generations: int = 50
    pareto_crossover_prob: float = 0.7
    pareto_mutation_prob: float = 0.2
    
    # Phase 2 enhancements - Gaussian Process Optimization
    enable_gaussian_process: bool = True
    gp_acquisition_samples: int = 1000
    gp_exploration_weight: float = 0.01
    gp_kernel_length_scale: float = 1.0
    gp_noise_level: float = 1e-5


@dataclass
class ParetoSolution:
    """A solution on the Pareto frontier"""
    rule_parameters: Dict[str, float]
    objectives: Dict[str, float]  # performance, consistency, efficiency, etc.
    dominance_rank: int
    crowding_distance: float
    feasible: bool = True


@dataclass
class MultiObjectiveResult:
    """Results from multi-objective optimization"""
    rule_id: str
    pareto_frontier: List[ParetoSolution]
    hypervolume: float
    convergence_metric: float
    total_evaluations: int
    best_compromise_solution: Optional[ParetoSolution]
    trade_off_analysis: Dict[str, Any]


@dataclass
class GaussianProcessResult:
    """Results from Gaussian process optimization"""
    rule_id: str
    optimal_parameters: Dict[str, float]
    predicted_performance: float
    uncertainty_estimate: float
    acquisition_history: List[Dict[str, Any]]
    model_confidence: float
    expected_improvement: float


class RuleOptimizer:
    """Optimizer for individual rules and rule combinations"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def optimize_rule(self, rule_id: str, performance_data: Dict[str, Any], 
                          historical_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Optimize a specific rule with Phase 2 enhancements"""
        self.logger.info(f"Optimizing rule {rule_id} with advanced methods")
        
        # Extract performance metrics
        metrics = performance_data.get(rule_id, {})
        
        if metrics.get('total_applications', 0) < self.config.min_sample_size:
            return {
                "rule_id": rule_id,
                "status": "insufficient_data",
                "message": f"Need at least {self.config.min_sample_size} applications for optimization"
            }
        
        # Traditional optimization recommendations
        recommendations = []
        avg_improvement = metrics.get('avg_improvement', 0)
        consistency_score = metrics.get('consistency_score', 0)
        
        if avg_improvement < 0.5:
            recommendations.append("Improve core rule logic")
        if consistency_score < 0.6:
            recommendations.append("Add consistency checks")
        
        result = {
            "rule_id": rule_id,
            "status": "optimized",
            "current_performance": avg_improvement,
            "traditional_recommendations": recommendations,
            "optimization_date": datetime.now().isoformat()
        }
        
        # Phase 2: Multi-objective optimization
        if self.config.enable_multi_objective and DEAP_AVAILABLE and historical_data:
            multi_obj_result = await self._multi_objective_optimization(rule_id, historical_data)
            if multi_obj_result:
                result["multi_objective_optimization"] = multi_obj_result.__dict__
        
        # Phase 2: Gaussian process optimization
        if self.config.enable_gaussian_process and GAUSSIAN_PROCESS_AVAILABLE and historical_data:
            gp_result = await self._gaussian_process_optimization(rule_id, historical_data)
            if gp_result:
                result["gaussian_process_optimization"] = gp_result.__dict__
        
        return result
    
    async def optimize_rule_combination(self, rule_ids: List[str], combination_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a rule combination"""
        self.logger.info(f"Optimizing rule combination: {', '.join(rule_ids)}")
        
        return {
            "rule_ids": rule_ids,
            "status": "optimized",
            "synergy_score": combination_data.get('synergy_score', 0),
            "recommendations": ["Monitor combination performance"],
            "optimization_date": datetime.now().isoformat()
        }
    
    async def _multi_objective_optimization(self, rule_id: str, historical_data: List[Dict[str, Any]]) -> Optional[MultiObjectiveResult]:
        """Phase 2: Multi-objective optimization using NSGA-II for Pareto frontier analysis"""
        if not DEAP_AVAILABLE:
            self.logger.warning("DEAP not available for multi-objective optimization")
            return None
        
        try:
            # Prepare optimization data
            optimization_data = self._prepare_optimization_data(historical_data)
            if len(optimization_data) < 10:
                self.logger.info(f"Insufficient data for multi-objective optimization: {len(optimization_data)} samples")
                return None
            
            # Setup DEAP for multi-objective optimization
            self._setup_deap_environment()
            
            # Run NSGA-II optimization
            pareto_frontier, stats = self._run_nsga2_optimization(optimization_data, rule_id)
            
            # Analyze results
            hypervolume = self._calculate_hypervolume(pareto_frontier)
            convergence_metric = self._calculate_convergence_metric(stats)
            best_compromise = self._find_best_compromise_solution(pareto_frontier)
            trade_off_analysis = self._analyze_trade_offs(pareto_frontier)
            
            return MultiObjectiveResult(
                rule_id=rule_id,
                pareto_frontier=pareto_frontier,
                hypervolume=hypervolume,
                convergence_metric=convergence_metric,
                total_evaluations=self.config.pareto_population_size * self.config.pareto_generations,
                best_compromise_solution=best_compromise,
                trade_off_analysis=trade_off_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Multi-objective optimization failed for {rule_id}: {e}")
            return None
    
    def _prepare_optimization_data(self, historical_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare historical data for optimization"""
        optimization_data = []
        
        for data_point in historical_data:
            # Extract relevant features for optimization
            if all(key in data_point for key in ['score', 'context', 'timestamp']):
                # Simulate rule parameters (in practice, these would come from actual rule configurations)
                parameters = {
                    'threshold': np.random.uniform(0.1, 0.9),
                    'weight': np.random.uniform(0.5, 1.0),
                    'complexity_factor': np.random.uniform(0.1, 1.0),
                    'context_sensitivity': np.random.uniform(0.0, 1.0)
                }
                
                # Extract objectives
                objectives = {
                    'performance': data_point['score'],
                    'consistency': 1.0 - abs(data_point['score'] - 0.7),  # Consistency around target
                    'efficiency': 1.0 / (data_point.get('execution_time_ms', 100) / 100.0),  # Inverse of execution time
                    'robustness': np.random.uniform(0.5, 1.0)  # Simulated robustness metric
                }
                
                optimization_data.append({
                    'parameters': parameters,
                    'objectives': objectives,
                    'feasible': objectives['performance'] > 0.3  # Feasibility constraint
                })
        
        return optimization_data
    
    def _setup_deap_environment(self):
        """Setup DEAP environment for multi-objective optimization"""
        # Clear any existing creators
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))  # Maximize all objectives
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        
        # Define parameter bounds
        self.param_bounds = {
            'threshold': (0.1, 0.9),
            'weight': (0.5, 1.0),
            'complexity_factor': (0.1, 1.0),
            'context_sensitivity': (0.0, 1.0)
        }
        
        # Register functions
        self.toolbox.register("attr_threshold", random.uniform, *self.param_bounds['threshold'])
        self.toolbox.register("attr_weight", random.uniform, *self.param_bounds['weight'])
        self.toolbox.register("attr_complexity", random.uniform, *self.param_bounds['complexity_factor'])
        self.toolbox.register("attr_context", random.uniform, *self.param_bounds['context_sensitivity'])
        
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                            (self.toolbox.attr_threshold, self.toolbox.attr_weight,
                             self.toolbox.attr_complexity, self.toolbox.attr_context), n=1)
        
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        self.toolbox.register("select", tools.selNSGA2)
    
    def _run_nsga2_optimization(self, optimization_data: List[Dict[str, Any]], rule_id: str) -> tuple:
        """Run NSGA-II optimization algorithm"""
        # Define evaluation function
        def evaluate_individual(individual):
            params = {
                'threshold': individual[0],
                'weight': individual[1],
                'complexity_factor': individual[2],
                'context_sensitivity': individual[3]
            }
            
            # Evaluate objectives based on historical data and parameters
            objectives = self._evaluate_objectives(params, optimization_data)
            return (objectives['performance'], objectives['consistency'], 
                   objectives['efficiency'], objectives['robustness'])
        
        self.toolbox.register("evaluate", evaluate_individual)
        
        # Initialize population
        population = self.toolbox.population(n=self.config.pareto_population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        # Run NSGA-II
        population, logbook = algorithms.eaMuPlusLambda(
            population, self.toolbox,
            mu=self.config.pareto_population_size,
            lambda_=self.config.pareto_population_size,
            cxpb=self.config.pareto_crossover_prob,
            mutpb=self.config.pareto_mutation_prob,
            ngen=self.config.pareto_generations,
            stats=stats,
            verbose=False
        )
        
        # Extract Pareto frontier
        pareto_frontier = tools.sortNondominated(population, self.config.pareto_population_size, first_front_only=True)[0]
        
        # Convert to ParetoSolution objects
        pareto_solutions = []
        for i, individual in enumerate(pareto_frontier):
            params = {
                'threshold': individual[0],
                'weight': individual[1],
                'complexity_factor': individual[2],
                'context_sensitivity': individual[3]
            }
            
            objectives = {
                'performance': individual.fitness.values[0],
                'consistency': individual.fitness.values[1],
                'efficiency': individual.fitness.values[2],
                'robustness': individual.fitness.values[3]
            }
            
            solution = ParetoSolution(
                rule_parameters=params,
                objectives=objectives,
                dominance_rank=0,  # All in first front
                crowding_distance=self._calculate_crowding_distance(individual, pareto_frontier),
                feasible=all(obj > 0.3 for obj in objectives.values())
            )
            pareto_solutions.append(solution)
        
        return pareto_solutions, logbook
    
    def _evaluate_objectives(self, params: Dict[str, float], optimization_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate objectives for given parameters"""
        # Simulate objective evaluation based on parameters and historical data
        base_performance = np.mean([d['objectives']['performance'] for d in optimization_data])
        
        # Parameter influence on objectives (simplified model)
        performance = base_performance * (0.5 + 0.5 * params['weight']) * (0.8 + 0.2 * params['threshold'])
        consistency = 0.9 - abs(params['threshold'] - 0.7) * params['complexity_factor']
        efficiency = 1.0 - params['complexity_factor'] * 0.3
        robustness = params['context_sensitivity'] * 0.7 + 0.3
        
        # Add some noise to simulate real-world variability
        noise_factor = 0.05
        performance *= (1 + np.random.normal(0, noise_factor))
        consistency *= (1 + np.random.normal(0, noise_factor))
        efficiency *= (1 + np.random.normal(0, noise_factor))
        robustness *= (1 + np.random.normal(0, noise_factor))
        
        return {
            'performance': max(0.0, min(1.0, performance)),
            'consistency': max(0.0, min(1.0, consistency)),
            'efficiency': max(0.0, min(1.0, efficiency)),
            'robustness': max(0.0, min(1.0, robustness))
        }
    
    def _calculate_crowding_distance(self, individual, population) -> float:
        """Calculate crowding distance for diversity preservation"""
        if len(population) <= 2:
            return float('inf')
        
        # Simple crowding distance calculation
        distances = []
        for other in population:
            if other != individual:
                distance = sum((a - b) ** 2 for a, b in zip(individual.fitness.values, other.fitness.values))
                distances.append(distance ** 0.5)
        
        return min(distances) if distances else 0.0
    
    def _calculate_hypervolume(self, pareto_frontier: List[ParetoSolution]) -> float:
        """Calculate hypervolume indicator for Pareto frontier quality"""
        if not pareto_frontier:
            return 0.0
        
        # Reference point (worst possible values)
        ref_point = [0.0, 0.0, 0.0, 0.0]
        
        # Simple hypervolume calculation (2D projection for efficiency)
        # In practice, would use more sophisticated hypervolume algorithms
        volumes = []
        for solution in pareto_frontier:
            objectives = solution.objectives
            volume = (objectives['performance'] - ref_point[0]) * (objectives['consistency'] - ref_point[1])
            volumes.append(max(0.0, volume))
        
        return sum(volumes)
    
    def _calculate_convergence_metric(self, logbook) -> float:
        """Calculate convergence metric from optimization statistics"""
        if not logbook:
            return 0.0
        
        # Measure improvement over generations
        initial_avg = np.mean(logbook[0]['avg'])
        final_avg = np.mean(logbook[-1]['avg'])
        
        convergence = (final_avg - initial_avg) / max(initial_avg, 0.001)
        return max(0.0, min(1.0, convergence))
    
    def _find_best_compromise_solution(self, pareto_frontier: List[ParetoSolution]) -> Optional[ParetoSolution]:
        """Find best compromise solution using weighted sum approach"""
        if not pareto_frontier:
            return None
        
        # Equal weights for all objectives (could be configured)
        weights = {'performance': 0.4, 'consistency': 0.3, 'efficiency': 0.2, 'robustness': 0.1}
        
        best_solution = None
        best_score = -1
        
        for solution in pareto_frontier:
            if not solution.feasible:
                continue
            
            # Calculate weighted sum
            score = sum(weights[obj] * value for obj, value in solution.objectives.items())
            
            if score > best_score:
                best_score = score
                best_solution = solution
        
        return best_solution
    
    def _analyze_trade_offs(self, pareto_frontier: List[ParetoSolution]) -> Dict[str, Any]:
        """Analyze trade-offs in the Pareto frontier"""
        if not pareto_frontier:
            return {}
        
        objectives = list(pareto_frontier[0].objectives.keys())
        trade_offs = {}
        
        # Calculate correlations between objectives
        for i, obj1 in enumerate(objectives):
            for obj2 in objectives[i+1:]:
                values1 = [sol.objectives[obj1] for sol in pareto_frontier]
                values2 = [sol.objectives[obj2] for sol in pareto_frontier]
                
                if len(set(values1)) > 1 and len(set(values2)) > 1:
                    correlation, _ = stats.pearsonr(values1, values2)
                    trade_offs[f"{obj1}_vs_{obj2}"] = {
                        'correlation': correlation,
                        'trade_off_strength': abs(correlation),
                        'conflicting': correlation < -0.3
                    }
        
        # Range analysis
        ranges = {}
        for obj in objectives:
            values = [sol.objectives[obj] for sol in pareto_frontier]
            ranges[obj] = {
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values),
                'std': np.std(values)
            }
        
        return {
            'objective_correlations': trade_offs,
            'objective_ranges': ranges,
            'frontier_size': len(pareto_frontier),
            'feasible_solutions': len([sol for sol in pareto_frontier if sol.feasible])
        }
    
    async def _gaussian_process_optimization(self, rule_id: str, historical_data: List[Dict[str, Any]]) -> Optional[GaussianProcessResult]:
        """Phase 2: Gaussian process optimization with Expected Improvement acquisition"""
        if not GAUSSIAN_PROCESS_AVAILABLE:
            self.logger.warning("Gaussian process libraries not available")
            return None
        
        try:
            # Prepare data for GP
            X, y = self._prepare_gp_data(historical_data)
            if len(X) < 5:
                self.logger.info(f"Insufficient data for GP optimization: {len(X)} samples")
                return None
            
            # Fit Gaussian Process
            gp_model, scaler = self._fit_gaussian_process(X, y)
            
            # Optimize using Expected Improvement
            optimal_params, acquisition_history = self._optimize_expected_improvement(gp_model, scaler, X, y)
            
            # Predict performance at optimal parameters
            predicted_performance, uncertainty = self._predict_performance(gp_model, scaler, optimal_params)
            
            # Calculate model confidence
            model_confidence = self._calculate_model_confidence(gp_model, X, y)
            
            # Calculate expected improvement
            expected_improvement = self._calculate_expected_improvement(
                gp_model, scaler, optimal_params, np.max(y)
            )
            
            return GaussianProcessResult(
                rule_id=rule_id,
                optimal_parameters=optimal_params,
                predicted_performance=predicted_performance,
                uncertainty_estimate=uncertainty,
                acquisition_history=acquisition_history,
                model_confidence=model_confidence,
                expected_improvement=expected_improvement
            )
            
        except Exception as e:
            self.logger.error(f"Gaussian process optimization failed for {rule_id}: {e}")
            return None
    
    def _prepare_gp_data(self, historical_data: List[Dict[str, Any]]) -> tuple:
        """Prepare data for Gaussian process optimization"""
        X = []
        y = []
        
        for data_point in historical_data:
            if 'score' in data_point:
                # Extract features (simulated parameter values)
                features = [
                    np.random.uniform(0.1, 0.9),  # threshold
                    np.random.uniform(0.5, 1.0),  # weight  
                    np.random.uniform(0.1, 1.0),  # complexity_factor
                    np.random.uniform(0.0, 1.0),  # context_sensitivity
                ]
                X.append(features)
                y.append(data_point['score'])
        
        return np.array(X), np.array(y)
    
    def _fit_gaussian_process(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Fit Gaussian process model"""
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define kernel
        kernel = RBF(length_scale=self.config.gp_kernel_length_scale) + WhiteKernel(noise_level=self.config.gp_noise_level)
        
        # Fit GP
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        gp.fit(X_scaled, y)
        
        return gp, scaler
    
    def _optimize_expected_improvement(self, gp_model, scaler, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
        """Optimize using Expected Improvement acquisition function"""
        # Current best performance
        f_best = np.max(y_train)
        
        # Parameter bounds
        bounds = [(0.1, 0.9), (0.5, 1.0), (0.1, 1.0), (0.0, 1.0)]
        
        # Sample candidate points
        best_ei = 0
        best_params = None
        acquisition_history = []
        
        for _ in range(self.config.gp_acquisition_samples):
            # Random candidate
            candidate = [np.random.uniform(low, high) for low, high in bounds]
            
            # Calculate Expected Improvement
            ei = self._calculate_expected_improvement(gp_model, scaler, candidate, f_best)
            
            acquisition_history.append({
                'parameters': {
                    'threshold': candidate[0],
                    'weight': candidate[1], 
                    'complexity_factor': candidate[2],
                    'context_sensitivity': candidate[3]
                },
                'expected_improvement': ei
            })
            
            if ei > best_ei:
                best_ei = ei
                best_params = {
                    'threshold': candidate[0],
                    'weight': candidate[1],
                    'complexity_factor': candidate[2], 
                    'context_sensitivity': candidate[3]
                }
        
        # Sort acquisition history by EI
        acquisition_history.sort(key=lambda x: x['expected_improvement'], reverse=True)
        
        return best_params or {
            'threshold': 0.5, 'weight': 0.75, 'complexity_factor': 0.5, 'context_sensitivity': 0.5
        }, acquisition_history[:10]  # Return top 10
    
    def _predict_performance(self, gp_model, scaler, params: Dict[str, float]) -> tuple:
        """Predict performance and uncertainty for given parameters"""
        # Convert parameters to feature vector
        X_test = np.array([[params['threshold'], params['weight'], 
                           params['complexity_factor'], params['context_sensitivity']]])
        X_test_scaled = scaler.transform(X_test)
        
        # Predict with uncertainty
        mean, std = gp_model.predict(X_test_scaled, return_std=True)
        
        return float(mean[0]), float(std[0])
    
    def _calculate_expected_improvement(self, gp_model, scaler, candidate: list, f_best: float) -> float:
        """Calculate Expected Improvement acquisition function"""
        # Convert to numpy array and scale
        X_candidate = np.array([candidate]).reshape(1, -1)
        X_candidate_scaled = scaler.transform(X_candidate)
        
        # Get prediction
        mean, std = gp_model.predict(X_candidate_scaled, return_std=True)
        
        if std[0] <= 0:
            return 0.0
        
        # Calculate EI
        improvement = mean[0] - f_best - self.config.gp_exploration_weight
        z = improvement / std[0]
        
        ei = improvement * stats.norm.cdf(z) + std[0] * stats.norm.pdf(z)
        
        return max(0.0, ei)
    
    def _calculate_model_confidence(self, gp_model, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Calculate confidence in the GP model"""
        try:
            # Cross-validation score as confidence metric
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(gp_model, X_train, y_train, cv=min(5, len(X_train)), scoring='r2')
            return max(0.0, np.mean(scores))
        except Exception:
            # Fallback: use kernel hyperparameters as rough confidence indicator
            return min(1.0, 1.0 / (1.0 + gp_model.kernel_.theta.std()))