"""Multi-Armed Bandit Framework for Rule Optimization

This module implements research-validated multi-armed bandit algorithms for
intelligent rule selection and optimization in the APES system.

Supports:
- Epsilon-Greedy with decay scheduling
- Upper Confidence Bound (ucb) with configurable exploration
- Thompson Sampling with Bayesian updating
- Contextual bandits for rule-specific optimization
- Integration with A/B testing framework

Based on:
- Auer et al. (2002) "Finite-time Analysis of the Multiarmed Bandit Problem"
- Agrawal & Goyal (2013) "Thompson Sampling for Contextual Bandits with Linear Payoffs"
- Li et al. (2010) "A Contextual-Bandit Approach to Personalized News Article Recommendation"
"""

import asyncio
import logging
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class BanditAlgorithm(Enum):
    """Multi-armed bandit algorithm types"""

    EPSILON_GREEDY = "epsilon_greedy"
    EPSILON_DECAY = "epsilon_decay"
    ucb = "ucb"
    THOMPSON_SAMPLING = "thompson_sampling"
    CONTEXTUAL_UCB = "contextual_ucb"
    CONTEXTUAL_THOMPSON = "contextual_thompson"

class ExplorationStrategy(Enum):
    """Exploration strategy types"""

    random = "random"
    optimistic = "optimistic"
    UNCERTAINTY_BASED = "uncertainty_based"

@dataclass
class BanditConfig:
    """Configuration for multi-armed bandit algorithms"""

    # Algorithm selection
    algorithm: BanditAlgorithm = BanditAlgorithm.THOMPSON_SAMPLING

    # Epsilon-Greedy parameters
    epsilon: float = 0.1
    epsilon_decay: float = 0.99
    min_epsilon: float = 0.01

    # ucb parameters
    ucb_confidence: float = 2.0
    ucb_exploration_bonus: float = 1.0

    # Thompson Sampling parameters
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    # Contextual bandit parameters
    context_dim: int = 10
    ridge_alpha: float = 1.0
    context_exploration: float = 0.1

    # General parameters
    min_samples_per_arm: int = 5
    warmup_trials: int = 100
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.UNCERTAINTY_BASED

    # Performance tracking
    enable_regret_tracking: bool = True
    enable_confidence_intervals: bool = True
    confidence_level: float = 0.95

@dataclass
class ArmResult:
    """Result from pulling a bandit arm"""

    arm_id: str
    reward: float
    context: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0
    uncertainty: float = 0.0

@dataclass
class BanditState:
    """Current state of a bandit arm"""

    arm_id: str
    pulls: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    variance: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    # Thompson Sampling parameters
    alpha: float = 1.0
    beta: float = 1.0

    # Contextual parameters
    context_features: np.ndarray | None = None
    last_update: datetime = field(default_factory=datetime.utcnow)

@dataclass
class BanditExperiment:
    """Multi-armed bandit experiment tracking"""

    experiment_id: str
    arms: list[str]
    algorithm: BanditAlgorithm
    config: BanditConfig
    start_time: datetime = field(default_factory=datetime.utcnow)
    total_trials: int = 0
    total_reward: float = 0.0
    regret_history: list[float] = field(default_factory=list)
    arm_states: dict[str, BanditState] = field(default_factory=dict)
    is_active: bool = True

class BaseBandit(ABC):
    """Abstract base class for multi-armed bandit algorithms"""

    def __init__(self, arms: list[str], config: BanditConfig):
        """Initialize bandit with arms and configuration

        Args:
            arms: List of arm identifiers
            config: Bandit configuration
        """
        self.arms = arms
        self.config = config
        self.arm_states = {arm: BanditState(arm_id=arm) for arm in arms}
        self.total_trials = 0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize arm states with priors
        for arm in arms:
            self.arm_states[arm].alpha = config.prior_alpha
            self.arm_states[arm].beta = config.prior_beta

    @abstractmethod
    async def select_arm(self, context: np.ndarray | None = None) -> str:
        """Select an arm to pull

        Args:
            context: Optional contextual information

        Returns:
            Selected arm identifier
        """

    async def update(
        self, arm_id: str, reward: float, context: np.ndarray | None = None
    ) -> None:
        """Update arm statistics with observed reward

        Args:
            arm_id: Arm that was pulled
            reward: Observed reward (0.0 to 1.0)
            context: Optional contextual information
        """
        if arm_id not in self.arm_states:
            raise ValueError(f"Unknown arm: {arm_id}")

        state = self.arm_states[arm_id]
        state.pulls += 1
        state.total_reward += reward
        state.mean_reward = state.total_reward / state.pulls

        # Update variance (Welford's online algorithm)
        if state.pulls == 1:
            state.variance = 0.0
        else:
            delta = reward - state.mean_reward
            state.variance = ((state.pulls - 2) * state.variance + delta * delta) / (
                state.pulls - 1
            )

        # Update confidence interval
        if state.pulls > 1:
            std_error = math.sqrt(state.variance / state.pulls)
            z_score = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
            margin = z_score * std_error
            state.confidence_interval = (
                state.mean_reward - margin,
                state.mean_reward + margin,
            )

        state.last_update = datetime.utcnow()
        self.total_trials += 1

        self.logger.debug(
            f"Updated arm {arm_id}: pulls={state.pulls}, mean_reward={state.mean_reward:.3f}"
        )

    def get_arm_statistics(self) -> dict[str, dict[str, Any]]:
        """Get current statistics for all arms"""
        return {
            arm_id: {
                "pulls": state.pulls,
                "mean_reward": state.mean_reward,
                "total_reward": state.total_reward,
                "variance": state.variance,
                "confidence_interval": state.confidence_interval,
                "last_update": state.last_update,
            }
            for arm_id, state in self.arm_states.items()
        }

    def get_best_arm(self) -> str:
        """Get the arm with highest mean reward"""
        return max(
            self.arm_states.keys(), key=lambda arm: self.arm_states[arm].mean_reward
        )

    def calculate_regret(self, optimal_reward: float) -> float:
        """Calculate cumulative regret compared to optimal strategy"""
        if self.total_trials == 0:
            return 0.0

        total_optimal_reward = optimal_reward * self.total_trials
        total_actual_reward = sum(
            state.total_reward for state in self.arm_states.values()
        )
        return max(0.0, total_optimal_reward - total_actual_reward)

class EpsilonGreedyBandit(BaseBandit):
    """Epsilon-Greedy bandit with optional decay"""

    def __init__(self, arms: list[str], config: BanditConfig):
        super().__init__(arms, config)
        self.current_epsilon = config.epsilon

    async def select_arm(self, context: np.ndarray | None = None) -> str:
        """Select arm using epsilon-greedy strategy"""
        # Warmup phase: random selection
        if self.total_trials < self.config.warmup_trials:
            return np.random.choice(self.arms)

        # Epsilon decay
        if self.config.algorithm == BanditAlgorithm.EPSILON_DECAY:
            self.current_epsilon = max(
                self.config.min_epsilon,
                self.current_epsilon * self.config.epsilon_decay,
            )

        # Epsilon-greedy selection
        if np.random.random() < self.current_epsilon:
            # Explore: random selection
            return np.random.choice(self.arms)
        # Exploit: select best arm
        return self.get_best_arm()

class UCBBandit(BaseBandit):
    """Upper Confidence Bound bandit"""

    async def select_arm(self, context: np.ndarray | None = None) -> str:
        """Select arm using ucb strategy"""
        # Warmup phase: ensure each arm is pulled at least once
        unpulled_arms = [arm for arm in self.arms if self.arm_states[arm].pulls == 0]
        if unpulled_arms:
            return np.random.choice(unpulled_arms)

        # Calculate ucb values
        ucb_values = {}
        for arm in self.arms:
            state = self.arm_states[arm]
            if state.pulls == 0:
                ucb_values[arm] = float("inf")
            else:
                confidence_radius = math.sqrt(
                    (self.config.ucb_confidence * math.log(self.total_trials))
                    / state.pulls
                )
                ucb_values[arm] = state.mean_reward + confidence_radius

        # Select arm with highest ucb value
        return max(ucb_values.keys(), key=lambda arm: ucb_values[arm])

class ThompsonSamplingBandit(BaseBandit):
    """Thompson Sampling bandit using Beta distributions"""

    async def update(
        self, arm_id: str, reward: float, context: np.ndarray | None = None
    ) -> None:
        """Update arm with Bayesian updating"""
        await super().update(arm_id, reward, context)

        # Update Beta distribution parameters
        state = self.arm_states[arm_id]
        if reward > 0.5:  # Success
            state.alpha += 1
        else:  # Failure
            state.beta += 1

    async def select_arm(self, context: np.ndarray | None = None) -> str:
        """Select arm using Thompson Sampling"""
        # Sample from each arm's Beta distribution
        sampled_values = {}
        for arm in self.arms:
            state = self.arm_states[arm]
            sampled_values[arm] = np.random.beta(state.alpha, state.beta)

        # Select arm with highest sampled value
        return max(sampled_values.keys(), key=lambda arm: sampled_values[arm])

class ContextualBandit(BaseBandit):
    """Contextual bandit using linear models"""

    def __init__(self, arms: list[str], config: BanditConfig):
        super().__init__(arms, config)
        self.models = {arm: Ridge(alpha=config.ridge_alpha) for arm in arms}
        self.context_history = {arm: [] for arm in arms}
        self.reward_history = {arm: [] for arm in arms}
        self.is_fitted = dict.fromkeys(arms, False)

    async def update(
        self, arm_id: str, reward: float, context: np.ndarray | None = None
    ) -> None:
        """Update contextual model with observed reward"""
        await super().update(arm_id, reward, context)

        if context is not None:
            self.context_history[arm_id].append(context)
            self.reward_history[arm_id].append(reward)

            # Refit model if we have enough samples
            if len(self.context_history[arm_id]) >= self.config.min_samples_per_arm:
                X = np.array(self.context_history[arm_id])
                y = np.array(self.reward_history[arm_id])
                self.models[arm_id].fit(X, y)
                self.is_fitted[arm_id] = True

    async def select_arm(self, context: np.ndarray | None = None) -> str:
        """Select arm using contextual information"""
        if context is None:
            # Fallback to epsilon-greedy without context
            if np.random.random() < self.config.epsilon:
                return np.random.choice(self.arms)
            return self.get_best_arm()

        # Warmup phase: random selection
        if self.total_trials < self.config.warmup_trials:
            return np.random.choice(self.arms)

        # Calculate predicted rewards with confidence
        predictions = {}
        for arm in self.arms:
            if self.is_fitted[arm]:
                pred_reward = self.models[arm].predict(context.reshape(1, -1))[0]

                # Add exploration bonus based on uncertainty
                if self.config.algorithm == BanditAlgorithm.CONTEXTUAL_UCB:
                    # Simple uncertainty estimation
                    uncertainty = self.config.context_exploration / math.sqrt(
                        max(1, self.arm_states[arm].pulls)
                    )
                    predictions[arm] = pred_reward + uncertainty
                else:
                    predictions[arm] = pred_reward
            else:
                # High uncertainty for unfitted models
                predictions[arm] = 1.0

        # Select arm with highest prediction
        return max(predictions.keys(), key=lambda arm: predictions[arm])

class MultiarmedBanditFramework:
    """Main framework for multi-armed bandit experiments"""

    def __init__(self, config: BanditConfig = None):
        """Initialize bandit framework

        Args:
            config: Bandit configuration
        """
        self.config = config or BanditConfig()
        self.experiments: dict[str, BanditExperiment] = {}
        self.bandits: dict[str, BaseBandit] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrator-compatible interface for multi-armed bandit optimization (2025 pattern)

        Args:
            config: Orchestrator configuration containing:
                - experiment_name: Name of the bandit experiment
                - arms: List of arm identifiers (e.g., rule IDs, strategies)
                - algorithm: Bandit algorithm ('epsilon_greedy', 'ucb', 'thompson_sampling', etc.)
                - context_data: Optional contextual information for contextual bandits
                - num_trials: Number of trials to simulate/run
                - reward_data: Optional historical reward data for initialization
                - output_path: Local path for output files (optional)

        Returns:
            Orchestrator-compatible result with bandit optimization and metadata
        """
        start_time = datetime.now()

        try:
            # Extract configuration from orchestrator
            experiment_name = config.get("experiment_name", f"bandit_experiment_{uuid.uuid4().hex[:8]}")
            arms = config.get("arms", [])
            algorithm_str = config.get("algorithm", "thompson_sampling")
            context_data = config.get("context_data", None)
            num_trials = config.get("num_trials", 100)
            reward_data = config.get("reward_data", {})
            output_path = config.get("output_path", "./outputs/bandit_optimization")

            # Validate input data
            if not arms or len(arms) < 2:
                raise ValueError("At least 2 arms are required for bandit optimization")

            # Convert algorithm string to enum
            try:
                algorithm = BanditAlgorithm(algorithm_str)
            except ValueError:
                algorithm = BanditAlgorithm.THOMPSON_SAMPLING
                self.logger.warning(f"Unknown algorithm '{algorithm_str}', using thompson_sampling")

            # Create bandit experiment
            experiment_id = await self.create_experiment(
                experiment_name=experiment_name,
                arms=arms,
                algorithm=algorithm
            )

            # Initialize with historical reward data if provided
            if reward_data:
                await self._initialize_with_historical_data(experiment_id, reward_data)

            # Run bandit optimization simulation
            optimization_results = await self._run_bandit_simulation(
                experiment_id, num_trials, context_data
            )

            # Get final experiment statistics
            experiment_stats = self._get_experiment_statistics(experiment_id)

            # Prepare orchestrator-compatible result
            result = {
                "experiment_summary": {
                    "experiment_id": experiment_id,
                    "experiment_name": experiment_name,
                    "algorithm_used": algorithm.value,
                    "total_arms": len(arms),
                    "total_trials": num_trials,
                    "best_arm": optimization_results["best_arm"],
                    "best_arm_reward": optimization_results["best_arm_reward"],
                    "convergence_trial": optimization_results.get("convergence_trial", num_trials)
                },
                "arm_performance": {
                    arm_id: {
                        "total_pulls": stats["pulls"],
                        "mean_reward": stats["mean_reward"],
                        "confidence_interval": stats.get("confidence_interval", [0, 0]),
                        "regret": stats.get("regret", 0.0),
                        "selection_probability": stats["pulls"] / num_trials if num_trials > 0 else 0
                    }
                    for arm_id, stats in experiment_stats.items()
                },
                "optimization_metrics": {
                    "total_regret": optimization_results.get("total_regret", 0.0),
                    "cumulative_regret": optimization_results.get("cumulative_regret", []),
                    "exploration_rate": optimization_results.get("exploration_rate", 0.0),
                    "convergence_achieved": optimization_results.get("convergence_achieved", False),
                    "algorithm_efficiency": optimization_results.get("algorithm_efficiency", 0.0)
                },
                "recommendations": {
                    "recommended_arm": optimization_results["best_arm"],
                    "confidence_level": optimization_results.get("confidence_level", 0.0),
                    "continue_exploration": optimization_results.get("continue_exploration", False),
                    "suggested_next_trials": optimization_results.get("suggested_next_trials", 0)
                }
            }

            # Calculate execution metadata
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "orchestrator_compatible": True,
                "component_result": result,
                "local_metadata": {
                    "output_path": output_path,
                    "execution_time": execution_time,
                    "experiment_id": experiment_id,
                    "algorithm_used": algorithm.value,
                    "arms_count": len(arms),
                    "trials_completed": num_trials,
                    "contextual_bandit": context_data is not None,
                    "component_version": "1.0.0"
                }
            }

        except ValueError as e:
            self.logger.error(f"Validation error in orchestrated bandit optimization: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": f"Validation error: {str(e)}", "experiment_summary": {}},
                "local_metadata": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "error": True,
                    "error_type": "validation",
                    "component_version": "1.0.0"
                }
            }
        except Exception as e:
            self.logger.error(f"Orchestrated bandit optimization failed: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": str(e), "experiment_summary": {}},
                "local_metadata": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "error": True,
                    "component_version": "1.0.0"
                }
            }

    async def create_experiment(
        self,
        experiment_name: str,
        arms: list[str],
        algorithm: BanditAlgorithm = None,
        config: BanditConfig = None,
    ) -> str:
        """Create a new bandit experiment

        Args:
            experiment_name: Name of the experiment
            arms: List of arm identifiers
            algorithm: Bandit algorithm to use
            config: Optional experiment-specific config

        Returns:
            Experiment ID
        """
        experiment_id = f"bandit_{experiment_name}_{uuid.uuid4().hex[:8]}"
        algorithm = algorithm or self.config.algorithm
        config = config or self.config

        # Create bandit instance
        bandit = self._create_bandit(arms, algorithm, config)

        # Create experiment record
        experiment = BanditExperiment(
            experiment_id=experiment_id,
            arms=arms,
            algorithm=algorithm,
            config=config,
            arm_states={arm: BanditState(arm_id=arm) for arm in arms},
        )

        self.experiments[experiment_id] = experiment
        self.bandits[experiment_id] = bandit

        self.logger.info(
            f"Created bandit experiment {experiment_id} with {len(arms)} arms using {algorithm.value}"
        )

        return experiment_id

    def _create_bandit(
        self, arms: list[str], algorithm: BanditAlgorithm, config: BanditConfig
    ) -> BaseBandit:
        """Create bandit instance based on algorithm"""
        if algorithm in [BanditAlgorithm.EPSILON_GREEDY, BanditAlgorithm.EPSILON_DECAY]:
            return EpsilonGreedyBandit(arms, config)
        if algorithm == BanditAlgorithm.ucb:
            return UCBBandit(arms, config)
        if algorithm == BanditAlgorithm.THOMPSON_SAMPLING:
            return ThompsonSamplingBandit(arms, config)
        if algorithm in [
            BanditAlgorithm.CONTEXTUAL_UCB,
            BanditAlgorithm.CONTEXTUAL_THOMPSON,
        ]:
            return ContextualBandit(arms, config)
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    async def select_arm(
        self, experiment_id: str, context: dict[str, Any] | None = None
    ) -> ArmResult:
        """Select an arm for the given experiment

        Args:
            experiment_id: Experiment identifier
            context: Optional contextual information

        Returns:
            Selected arm result
        """
        if experiment_id not in self.bandits:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        bandit = self.bandits[experiment_id]
        experiment = self.experiments[experiment_id]

        # Convert context to numpy array if provided
        context_array = None
        if context:
            context_array = self._context_to_array(context)

        # Select arm
        selected_arm = await bandit.select_arm(context_array)

        # Calculate confidence and uncertainty
        state = bandit.arm_states[selected_arm]
        confidence = self._calculate_confidence(state)
        uncertainty = self._calculate_uncertainty(state)

        return ArmResult(
            arm_id=selected_arm,
            reward=0.0,  # Will be updated when reward is observed
            context=context_array,
            metadata={"experiment_id": experiment_id},
            confidence=confidence,
            uncertainty=uncertainty,
        )

    async def update_reward(
        self, experiment_id: str, arm_result: ArmResult, reward: float
    ) -> None:
        """Update bandit with observed reward

        Args:
            experiment_id: Experiment identifier
            arm_result: Previous arm selection result
            reward: Observed reward (0.0 to 1.0)
        """
        if experiment_id not in self.bandits:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        # Normalize reward to [0, 1]
        normalized_reward = max(0.0, min(1.0, reward))

        bandit = self.bandits[experiment_id]
        experiment = self.experiments[experiment_id]

        # Update bandit
        await bandit.update(arm_result.arm_id, normalized_reward, arm_result.context)

        # Update experiment tracking
        experiment.total_trials += 1
        experiment.total_reward += normalized_reward
        experiment.arm_states[arm_result.arm_id] = bandit.arm_states[arm_result.arm_id]

        # Track regret if enabled
        if self.config.enable_regret_tracking:
            best_possible_reward = max(
                state.mean_reward for state in bandit.arm_states.values()
            )
            current_regret = bandit.calculate_regret(best_possible_reward)
            experiment.regret_history.append(current_regret)

        self.logger.debug(
            f"Updated experiment {experiment_id}: total_trials={experiment.total_trials}, total_reward={experiment.total_reward:.3f}"
        )

    def _context_to_array(self, context: dict[str, Any]) -> np.ndarray:
        """Convert context dictionary to numpy array"""
        # Simple conversion - in practice this would be more sophisticated
        features = []

        # Extract numeric features
        for key, value in context.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Simple string hashing
                features.append(float(hash(value) % 1000) / 1000.0)

        # Pad or truncate to expected dimension
        while len(features) < self.config.context_dim:
            features.append(0.0)

        return np.array(features[: self.config.context_dim])

    def _calculate_confidence(self, state: BanditState) -> float:
        """Calculate confidence in arm selection"""
        if state.pulls == 0:
            return 0.0

        # Use confidence interval width as inverse of confidence
        ci_width = state.confidence_interval[1] - state.confidence_interval[0]
        return max(0.0, 1.0 - ci_width)

    def _calculate_uncertainty(self, state: BanditState) -> float:
        """Calculate uncertainty in arm selection"""
        if state.pulls == 0:
            return 1.0

        # Use standard error as uncertainty measure
        std_error = (
            math.sqrt(state.variance / state.pulls) if state.variance > 0 else 0.0
        )
        return min(1.0, std_error)

    def get_experiment_summary(self, experiment_id: str) -> dict[str, Any]:
        """Get summary of experiment performance"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        experiment = self.experiments[experiment_id]
        bandit = self.bandits[experiment_id]

        # Calculate performance metrics
        best_arm = bandit.get_best_arm()
        total_regret = (
            experiment.regret_history[-1] if experiment.regret_history else 0.0
        )

        arm_stats = bandit.get_arm_statistics()

        return {
            "experiment_id": experiment_id,
            "algorithm": experiment.algorithm.value,
            "total_trials": experiment.total_trials,
            "total_reward": experiment.total_reward,
            "average_reward": experiment.total_reward / max(1, experiment.total_trials),
            "best_arm": best_arm,
            "best_arm_reward": arm_stats[best_arm]["mean_reward"],
            "total_regret": total_regret,
            "arm_statistics": arm_stats,
            "start_time": experiment.start_time,
            "is_active": experiment.is_active,
        }

    def stop_experiment(self, experiment_id: str) -> None:
        """Stop a running experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].is_active = False
            self.logger.info(f"Stopped experiment {experiment_id}")

    def cleanup_old_experiments(self, days_old: int = 30) -> int:
        """Clean up old completed experiments"""
        cutoff_time = datetime.utcnow() - timedelta(days=days_old)

        to_remove = []
        for exp_id, experiment in self.experiments.items():
            if not experiment.is_active and experiment.start_time < cutoff_time:
                to_remove.append(exp_id)

        for exp_id in to_remove:
            del self.experiments[exp_id]
            if exp_id in self.bandits:
                del self.bandits[exp_id]

        self.logger.info(f"Cleaned up {len(to_remove)} old experiments")
        return len(to_remove)

    async def _initialize_with_historical_data(self, experiment_id: str, reward_data: Dict[str, List[float]]) -> None:
        """Initialize bandit with historical reward data"""
        if experiment_id not in self.bandits:
            return

        bandit = self.bandits[experiment_id]

        for arm_id, rewards in reward_data.items():
            if arm_id in bandit.arms:
                for reward in rewards:
                    await bandit.update(arm_id, reward)

    async def _run_bandit_simulation(self, experiment_id: str, num_trials: int, context_data: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Run bandit optimization simulation"""
        if experiment_id not in self.bandits:
            raise ValueError(f"Experiment {experiment_id} not found")

        bandit = self.bandits[experiment_id]

        # Track optimization metrics
        cumulative_regret = []
        total_regret = 0.0
        arm_selections = []
        rewards_received = []

        # Simulate trials
        for trial in range(num_trials):
            # Get context for this trial if available
            context = None
            if context_data and trial < len(context_data):
                context_dict = context_data[trial]
                if context_dict:
                    # Convert context dict to numpy array (simple implementation)
                    context = np.array(list(context_dict.values()), dtype=float)

            # Select arm
            selected_arm = await bandit.select_arm(context)
            arm_selections.append(selected_arm)

            # Simulate reward (in real usage, this would come from actual system)
            simulated_reward = self._simulate_reward(selected_arm, context)
            rewards_received.append(simulated_reward)

            # Update bandit
            await bandit.update(selected_arm, simulated_reward, context)

            # Calculate regret (assuming we know the optimal arm for simulation)
            optimal_reward = max(self._simulate_reward(arm, context) for arm in bandit.arms)
            regret = optimal_reward - simulated_reward
            total_regret += regret
            cumulative_regret.append(total_regret)

        # Determine best arm
        best_arm = bandit.get_best_arm()
        best_arm_stats = bandit.arm_states[best_arm]

        # Calculate convergence metrics
        convergence_trial = self._calculate_convergence_trial(arm_selections, best_arm)
        convergence_achieved = convergence_trial < num_trials * 0.8  # Converged in first 80%

        return {
            "best_arm": best_arm,
            "best_arm_reward": best_arm_stats.mean_reward,
            "total_regret": total_regret,
            "cumulative_regret": cumulative_regret,
            "convergence_trial": convergence_trial,
            "convergence_achieved": convergence_achieved,
            "exploration_rate": self._calculate_exploration_rate(arm_selections),
            "algorithm_efficiency": 1.0 - (total_regret / (num_trials * 1.0)),  # Normalized efficiency
            "confidence_level": self._calculate_confidence_level(bandit, best_arm),
            "continue_exploration": not convergence_achieved,
            "suggested_next_trials": max(0, int(num_trials * 0.2)) if not convergence_achieved else 0
        }

    def _simulate_reward(self, arm_id: str, context: Optional[np.ndarray] = None) -> float:
        """Simulate reward for an arm (for testing/simulation purposes)"""
        # Simple simulation: different arms have different base rewards
        arm_rewards = {
            "arm_0": 0.3, "arm_1": 0.5, "arm_2": 0.7, "arm_3": 0.4,
            "rule_1": 0.6, "rule_2": 0.8, "rule_3": 0.4, "rule_4": 0.7,
            "strategy_a": 0.5, "strategy_b": 0.7, "strategy_c": 0.6
        }

        base_reward = arm_rewards.get(arm_id, 0.5)

        # Add context influence if available
        if context is not None and len(context) > 0:
            context_influence = np.mean(context) * 0.1  # Small context effect
            base_reward += context_influence

        # Add noise
        noise = np.random.normal(0, 0.1)
        reward = np.clip(base_reward + noise, 0, 1)

        return float(reward)

    def _calculate_convergence_trial(self, arm_selections: List[str], best_arm: str) -> int:
        """Calculate when the algorithm converged to the best arm"""
        if not arm_selections:
            return 0

        # Find the last trial where a non-best arm was selected
        for i in range(len(arm_selections) - 1, -1, -1):
            if arm_selections[i] != best_arm:
                return i + 1

        return 0  # Converged immediately

    def _calculate_exploration_rate(self, arm_selections: List[str]) -> float:
        """Calculate the exploration rate (diversity of arm selections)"""
        if not arm_selections:
            return 0.0

        unique_arms = len(set(arm_selections))
        total_arms = len(set(arm_selections))  # This should be total possible arms, but we'll use unique for now

        return unique_arms / max(1, len(arm_selections)) * len(arm_selections)

    def _calculate_confidence_level(self, bandit: BaseBandit, best_arm: str) -> float:
        """Calculate confidence level in the best arm selection"""
        if best_arm not in bandit.arm_states:
            return 0.0

        best_stats = bandit.arm_states[best_arm]
        if best_stats.pulls == 0:
            return 0.0

        # Simple confidence based on number of pulls and reward variance
        confidence = min(1.0, best_stats.pulls / 50.0)  # More pulls = more confidence

        # Adjust for reward consistency
        if hasattr(best_stats, 'variance') and best_stats.variance is not None:
            consistency_bonus = max(0, 1.0 - best_stats.variance)
            confidence = (confidence + consistency_bonus) / 2

        return confidence

    def _get_experiment_statistics(self, experiment_id: str) -> Dict[str, Dict[str, Any]]:
        """Get experiment statistics for all arms"""
        if experiment_id not in self.bandits:
            return {}

        bandit = self.bandits[experiment_id]
        stats = {}

        for arm_id in bandit.arms:
            arm_state = bandit.arm_states[arm_id]
            stats[arm_id] = {
                "pulls": arm_state.pulls,
                "mean_reward": arm_state.mean_reward,
                "total_reward": arm_state.total_reward,
                "confidence_interval": [
                    max(0, arm_state.mean_reward - 1.96 * np.sqrt(arm_state.variance / max(1, arm_state.pulls))),
                    min(1, arm_state.mean_reward + 1.96 * np.sqrt(arm_state.variance / max(1, arm_state.pulls)))
                ] if hasattr(arm_state, 'variance') and arm_state.variance is not None else [0, 0],
                "regret": 0.0  # Will be calculated relative to best arm
            }

        # Calculate regret relative to best arm
        best_mean = max(stats[arm]["mean_reward"] for arm in stats.keys()) if stats else 0
        for arm_id in stats:
            stats[arm_id]["regret"] = best_mean - stats[arm_id]["mean_reward"]

        return stats

# Utility functions for easy integration

async def create_rule_optimization_bandit(
    rule_ids: list[str],
    algorithm: BanditAlgorithm = BanditAlgorithm.THOMPSON_SAMPLING,
    config: BanditConfig = None,
) -> tuple[MultiarmedBanditFramework, str]:
    """Create a bandit experiment for rule optimization

    Args:
        rule_ids: List of rule identifiers to optimize
        algorithm: Bandit algorithm to use
        config: Optional configuration

    Returns:
        (framework, experiment_id) tuple
    """
    framework = MultiarmedBanditFramework(config)
    experiment_id = await framework.create_experiment(
        experiment_name="rule_optimization",
        arms=rule_ids,
        algorithm=algorithm,
        config=config,
    )

    return framework, experiment_id

async def intelligent_rule_selection(
    framework: MultiarmedBanditFramework,
    experiment_id: str,
    prompt_context: dict[str, Any] = None,
) -> tuple[str, float]:
    """Intelligently select a rule using bandit algorithm

    Args:
        framework: Bandit framework instance
        experiment_id: Experiment identifier
        prompt_context: Optional context about the prompt

    Returns:
        (selected_rule_id, confidence) tuple
    """
    result = await framework.select_arm(experiment_id, prompt_context)
    return result.arm_id, result.confidence
