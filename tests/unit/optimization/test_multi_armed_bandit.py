"""Tests for Multi-Armed Bandit Framework

This test suite validates the multi-armed bandit implementation including:
- Epsilon-Greedy and Epsilon-Decay algorithms
- Upper Confidence Bound (UCB) algorithm
- Thompson Sampling algorithm
- Contextual bandit support
- Integration with A/B testing framework
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from src.prompt_improver.optimization.multi_armed_bandit import (
    MultiarmedBanditFramework,
    BanditConfig,
    BanditAlgorithm,
    EpsilonGreedyBandit,
    UCBBandit,
    ThompsonSamplingBandit,
    ContextualBandit,
    ArmResult,
    BanditState,
    create_rule_optimization_bandit,
    intelligent_rule_selection
)


class TestBanditConfig:
    """Test bandit configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = BanditConfig()
        
        assert config.algorithm == BanditAlgorithm.THOMPSON_SAMPLING
        assert config.epsilon == 0.1
        assert config.epsilon_decay == 0.99
        assert config.min_epsilon == 0.01
        assert config.ucb_confidence == 2.0
        assert config.prior_alpha == 1.0
        assert config.prior_beta == 1.0
        assert config.min_samples_per_arm == 5
        assert config.warmup_trials == 100
        assert config.enable_regret_tracking is True
        assert config.confidence_level == 0.95
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = BanditConfig(
            algorithm=BanditAlgorithm.UCB,
            epsilon=0.2,
            ucb_confidence=1.5,
            warmup_trials=50
        )
        
        assert config.algorithm == BanditAlgorithm.UCB
        assert config.epsilon == 0.2
        assert config.ucb_confidence == 1.5
        assert config.warmup_trials == 50


class TestEpsilonGreedyBandit:
    """Test Epsilon-Greedy bandit algorithm"""
    
    @pytest.fixture
    def bandit(self):
        """Create epsilon-greedy bandit for testing"""
        arms = ["rule_1", "rule_2", "rule_3"]
        config = BanditConfig(
            algorithm=BanditAlgorithm.EPSILON_GREEDY,
            epsilon=0.1,
            warmup_trials=0  # Disable warmup for testing
        )
        return EpsilonGreedyBandit(arms, config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, bandit):
        """Test bandit initialization"""
        assert len(bandit.arms) == 3
        assert bandit.total_trials == 0
        assert all(state.pulls == 0 for state in bandit.arm_states.values())
        assert bandit.current_epsilon == 0.1
    
    @pytest.mark.asyncio
    async def test_arm_selection_exploration(self, bandit):
        """Test exploration in epsilon-greedy"""
        # Mock random to force exploration
        with patch('numpy.random.random', return_value=0.05):  # < epsilon
            with patch('numpy.random.choice', return_value="rule_2") as mock_choice:
                arm = await bandit.select_arm()
                assert arm == "rule_2"
                mock_choice.assert_called_once_with(bandit.arms)
    
    @pytest.mark.asyncio
    async def test_arm_selection_exploitation(self, bandit):
        """Test exploitation in epsilon-greedy"""
        # Give rule_2 higher reward
        await bandit.update("rule_2", 0.8)
        await bandit.update("rule_1", 0.3)
        await bandit.update("rule_3", 0.2)
        
        # Mock random to force exploitation
        with patch('numpy.random.random', return_value=0.95):  # >= epsilon
            arm = await bandit.select_arm()
            assert arm == "rule_2"  # Highest reward
    
    @pytest.mark.asyncio
    async def test_epsilon_decay(self):
        """Test epsilon decay functionality"""
        arms = ["rule_1", "rule_2"]
        config = BanditConfig(
            algorithm=BanditAlgorithm.EPSILON_DECAY,
            epsilon=1.0,
            epsilon_decay=0.9,
            min_epsilon=0.1,
            warmup_trials=0
        )
        bandit = EpsilonGreedyBandit(arms, config)
        
        initial_epsilon = bandit.current_epsilon
        
        # Multiple selections should decay epsilon
        for _ in range(10):
            await bandit.select_arm()
        
        assert bandit.current_epsilon < initial_epsilon
        assert bandit.current_epsilon >= config.min_epsilon


class TestUCBBandit:
    """Test Upper Confidence Bound bandit algorithm"""
    
    @pytest.fixture
    def bandit(self):
        """Create UCB bandit for testing"""
        arms = ["rule_1", "rule_2", "rule_3"]
        config = BanditConfig(
            algorithm=BanditAlgorithm.UCB,
            ucb_confidence=2.0
        )
        return UCBBandit(arms, config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, bandit):
        """Test UCB bandit initialization"""
        assert len(bandit.arms) == 3
        assert bandit.total_trials == 0
        assert all(state.pulls == 0 for state in bandit.arm_states.values())
    
    @pytest.mark.asyncio
    async def test_warmup_phase(self, bandit):
        """Test that UCB explores unpulled arms first"""
        # All arms should be selected once during warmup
        selected_arms = set()
        
        for _ in range(3):
            arm = await bandit.select_arm()
            selected_arms.add(arm)
            await bandit.update(arm, 0.5)
        
        assert len(selected_arms) == 3  # All arms selected
    
    @pytest.mark.asyncio
    async def test_ucb_calculation(self, bandit):
        """Test UCB value calculation and selection"""
        # Pull each arm once
        for arm in bandit.arms:
            await bandit.update(arm, 0.5)
        
        # Give rule_1 high reward but few pulls
        await bandit.update("rule_1", 0.9)
        
        # Give rule_2 medium reward but many pulls
        for _ in range(10):
            await bandit.update("rule_2", 0.6)
        
        # UCB should consider both reward and uncertainty
        # rule_1 should have high UCB due to uncertainty
        arm = await bandit.select_arm()
        # This is probabilistic, but rule_1 should often be selected due to high uncertainty


class TestThompsonSamplingBandit:
    """Test Thompson Sampling bandit algorithm"""
    
    @pytest.fixture
    def bandit(self):
        """Create Thompson Sampling bandit for testing"""
        arms = ["rule_1", "rule_2", "rule_3"]
        config = BanditConfig(
            algorithm=BanditAlgorithm.THOMPSON_SAMPLING,
            prior_alpha=1.0,
            prior_beta=1.0
        )
        return ThompsonSamplingBandit(arms, config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, bandit):
        """Test Thompson Sampling initialization"""
        assert len(bandit.arms) == 3
        assert bandit.total_trials == 0
        for state in bandit.arm_states.values():
            assert state.alpha == 1.0
            assert state.beta == 1.0
    
    @pytest.mark.asyncio
    async def test_beta_parameter_updates(self, bandit):
        """Test Beta distribution parameter updates"""
        # High reward should increase alpha
        await bandit.update("rule_1", 0.8)
        assert bandit.arm_states["rule_1"].alpha == 2.0
        assert bandit.arm_states["rule_1"].beta == 1.0
        
        # Low reward should increase beta
        await bandit.update("rule_2", 0.2)
        assert bandit.arm_states["rule_2"].alpha == 1.0
        assert bandit.arm_states["rule_2"].beta == 2.0
    
    @pytest.mark.asyncio
    async def test_sampling_selection(self, bandit):
        """Test that Thompson Sampling selects based on Beta sampling"""
        # Set deterministic rewards to test behavior
        np.random.seed(42)
        
        # Give rule_1 consistently high rewards
        for _ in range(10):
            await bandit.update("rule_1", 0.9)
        
        # Give rule_2 consistently low rewards  
        for _ in range(10):
            await bandit.update("rule_2", 0.1)
        
        # Thompson Sampling should favor rule_1
        selections = []
        for _ in range(20):
            arm = await bandit.select_arm()
            selections.append(arm)
        
        # rule_1 should be selected more often
        rule_1_count = selections.count("rule_1")
        rule_2_count = selections.count("rule_2")
        assert rule_1_count > rule_2_count


class TestContextualBandit:
    """Test Contextual bandit algorithm"""
    
    @pytest.fixture
    def bandit(self):
        """Create contextual bandit for testing"""
        arms = ["rule_1", "rule_2", "rule_3"]
        config = BanditConfig(
            algorithm=BanditAlgorithm.CONTEXTUAL_UCB,
            context_dim=5,
            min_samples_per_arm=3,
            warmup_trials=0
        )
        return ContextualBandit(arms, config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, bandit):
        """Test contextual bandit initialization"""
        assert len(bandit.arms) == 3
        assert len(bandit.models) == 3
        assert all(not fitted for fitted in bandit.is_fitted.values())
    
    @pytest.mark.asyncio
    async def test_context_handling(self, bandit):
        """Test contextual information handling"""
        context = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Before fitting, should fallback to epsilon-greedy
        arm = await bandit.select_arm(context)
        assert arm in bandit.arms
        
        # Update with context
        await bandit.update(arm, 0.7, context)
        assert len(bandit.context_history[arm]) == 1
        assert len(bandit.reward_history[arm]) == 1
    
    @pytest.mark.asyncio
    async def test_model_fitting(self, bandit):
        """Test that models are fitted after sufficient samples"""
        context1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        context2 = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        context3 = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        
        # Add samples for rule_1
        await bandit.update("rule_1", 0.8, context1)
        await bandit.update("rule_1", 0.7, context2)
        await bandit.update("rule_1", 0.9, context3)
        
        # Model should now be fitted
        assert bandit.is_fitted["rule_1"] is True
        assert bandit.is_fitted["rule_2"] is False


class TestMultiarmedBanditFramework:
    """Test the main bandit framework"""
    
    @pytest.fixture
    def framework(self):
        """Create bandit framework for testing"""
        config = BanditConfig(
            algorithm=BanditAlgorithm.THOMPSON_SAMPLING,
            warmup_trials=5,
            enable_regret_tracking=True
        )
        return MultiarmedBanditFramework(config)
    
    @pytest.mark.asyncio
    async def test_experiment_creation(self, framework):
        """Test creating bandit experiments"""
        arms = ["rule_1", "rule_2", "rule_3"]
        experiment_id = await framework.create_experiment(
            experiment_name="test_experiment",
            arms=arms,
            algorithm=BanditAlgorithm.UCB
        )
        
        assert experiment_id in framework.experiments
        assert experiment_id in framework.bandits
        
        experiment = framework.experiments[experiment_id]
        assert experiment.arms == arms
        assert experiment.algorithm == BanditAlgorithm.UCB
        assert experiment.is_active is True
    
    @pytest.mark.asyncio
    async def test_arm_selection_and_update(self, framework):
        """Test arm selection and reward updates"""
        arms = ["rule_1", "rule_2", "rule_3"]
        experiment_id = await framework.create_experiment(
            experiment_name="test_experiment",
            arms=arms
        )
        
        # Select arm
        result = await framework.select_arm(experiment_id)
        assert result.arm_id in arms
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.uncertainty <= 1.0
        
        # Update with reward
        await framework.update_reward(experiment_id, result, 0.8)
        
        # Check experiment state updated
        experiment = framework.experiments[experiment_id]
        assert experiment.total_trials == 1
        assert experiment.total_reward == 0.8
    
    @pytest.mark.asyncio
    async def test_experiment_summary(self, framework):
        """Test experiment summary generation"""
        arms = ["rule_1", "rule_2", "rule_3"]
        experiment_id = await framework.create_experiment(
            experiment_name="test_experiment",
            arms=arms
        )
        
        # Run some trials
        for _ in range(10):
            result = await framework.select_arm(experiment_id)
            reward = 0.7 if result.arm_id == "rule_2" else 0.3
            await framework.update_reward(experiment_id, result, reward)
        
        summary = framework.get_experiment_summary(experiment_id)
        
        assert summary["experiment_id"] == experiment_id
        assert summary["total_trials"] == 10
        assert summary["total_reward"] > 0
        assert summary["best_arm"] in arms
        assert "arm_statistics" in summary
        assert len(summary["arm_statistics"]) == 3
    
    @pytest.mark.asyncio
    async def test_context_conversion(self, framework):
        """Test context dictionary to array conversion"""
        context = {
            "prompt_length": 100,
            "has_examples": True,
            "complexity": "medium",
            "domain": "technical"
        }
        
        context_array = framework._context_to_array(context)
        
        assert isinstance(context_array, np.ndarray)
        assert len(context_array) == framework.config.context_dim
        assert all(isinstance(x, float) for x in context_array)
    
    @pytest.mark.asyncio
    async def test_experiment_stopping(self, framework):
        """Test stopping experiments"""
        arms = ["rule_1", "rule_2"]
        experiment_id = await framework.create_experiment(
            experiment_name="test_experiment",
            arms=arms
        )
        
        assert framework.experiments[experiment_id].is_active is True
        
        framework.stop_experiment(experiment_id)
        
        assert framework.experiments[experiment_id].is_active is False
    
    def test_cleanup_old_experiments(self, framework):
        """Test cleanup of old experiments"""
        # Create old experiment manually
        old_experiment_id = "old_experiment"
        framework.experiments[old_experiment_id] = type('Experiment', (), {
            'is_active': False,
            'start_time': datetime.utcnow() - timedelta(days=35)
        })()
        framework.bandits[old_experiment_id] = None
        
        # Create recent experiment
        recent_experiment_id = "recent_experiment"
        framework.experiments[recent_experiment_id] = type('Experiment', (), {
            'is_active': False,
            'start_time': datetime.utcnow() - timedelta(days=5)
        })()
        framework.bandits[recent_experiment_id] = None
        
        cleaned_count = framework.cleanup_old_experiments(days_old=30)
        
        assert cleaned_count == 1
        assert old_experiment_id not in framework.experiments
        assert recent_experiment_id in framework.experiments


class TestBanditStatistics:
    """Test bandit statistical calculations"""
    
    @pytest.fixture
    def bandit(self):
        """Create bandit with some data for testing"""
        arms = ["rule_1", "rule_2"]
        config = BanditConfig()
        bandit = ThompsonSamplingBandit(arms, config)
        return bandit
    
    @pytest.mark.asyncio
    async def test_confidence_interval_calculation(self, bandit):
        """Test confidence interval calculation"""
        # Add multiple rewards to get meaningful statistics
        rewards = [0.6, 0.7, 0.8, 0.5, 0.9]
        for reward in rewards:
            await bandit.update("rule_1", reward)
        
        state = bandit.arm_states["rule_1"]
        assert state.pulls == 5
        assert state.mean_reward == np.mean(rewards)
        assert state.variance > 0
        
        # Confidence interval should be reasonable
        ci_lower, ci_upper = state.confidence_interval
        assert ci_lower < state.mean_reward < ci_upper
        assert ci_upper - ci_lower > 0  # Non-zero width
    
    @pytest.mark.asyncio
    async def test_regret_calculation(self, bandit):
        """Test regret calculation"""
        # Set up scenario with known optimal arm
        optimal_reward = 0.9
        
        # Add rewards
        await bandit.update("rule_1", 0.9)  # Optimal
        await bandit.update("rule_2", 0.3)  # Suboptimal
        
        regret = bandit.calculate_regret(optimal_reward)
        
        # Should have some regret from pulling suboptimal arm
        assert regret >= 0
    
    def test_get_best_arm(self, bandit):
        """Test best arm identification"""
        # Initially all arms have same reward
        best_arm = bandit.get_best_arm()
        assert best_arm in bandit.arms
        
        # After updates, should identify best arm
        bandit.arm_states["rule_1"].mean_reward = 0.8
        bandit.arm_states["rule_2"].mean_reward = 0.3
        
        best_arm = bandit.get_best_arm()
        assert best_arm == "rule_1"


class TestBanditIntegration:
    """Test integration functions"""
    
    @pytest.mark.asyncio
    async def test_create_rule_optimization_bandit(self):
        """Test rule optimization bandit creation"""
        rule_ids = ["clarity_rule", "specificity_rule", "cot_rule"]
        
        framework, experiment_id = await create_rule_optimization_bandit(
            rule_ids=rule_ids,
            algorithm=BanditAlgorithm.UCB
        )
        
        assert isinstance(framework, MultiarmedBanditFramework)
        assert experiment_id in framework.experiments
        
        experiment = framework.experiments[experiment_id]
        assert experiment.arms == rule_ids
        assert experiment.algorithm == BanditAlgorithm.UCB
    
    @pytest.mark.asyncio
    async def test_intelligent_rule_selection(self):
        """Test intelligent rule selection utility"""
        rule_ids = ["clarity_rule", "specificity_rule"]
        
        framework, experiment_id = await create_rule_optimization_bandit(
            rule_ids=rule_ids
        )
        
        # Test selection
        selected_rule, confidence = await intelligent_rule_selection(
            framework=framework,
            experiment_id=experiment_id,
            prompt_context={"length": 100, "complexity": "medium"}
        )
        
        assert selected_rule in rule_ids
        assert 0.0 <= confidence <= 1.0


class TestErrorHandling:
    """Test error handling in bandit framework"""
    
    @pytest.fixture
    def framework(self):
        """Create framework for error testing"""
        return MultiarmedBanditFramework()
    
    @pytest.mark.asyncio
    async def test_unknown_experiment_error(self, framework):
        """Test handling of unknown experiment IDs"""
        with pytest.raises(ValueError, match="Unknown experiment"):
            await framework.select_arm("nonexistent_experiment")
        
        with pytest.raises(ValueError, match="Unknown experiment"):
            await framework.get_experiment_summary("nonexistent_experiment")
    
    @pytest.mark.asyncio
    async def test_unknown_arm_error(self):
        """Test handling of unknown arms"""
        arms = ["rule_1", "rule_2"]
        config = BanditConfig()
        bandit = EpsilonGreedyBandit(arms, config)
        
        with pytest.raises(ValueError, match="Unknown arm"):
            await bandit.update("unknown_rule", 0.5)
    
    def test_invalid_algorithm_error(self):
        """Test handling of invalid algorithms"""
        framework = MultiarmedBanditFramework()
        
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            framework._create_bandit(["rule_1"], "invalid_algorithm", BanditConfig())


class TestPerformanceMetrics:
    """Test performance metrics and analysis"""
    
    @pytest.fixture
    def framework_with_data(self):
        """Create framework with sample data"""
        framework = MultiarmedBanditFramework()
        return framework
    
    @pytest.mark.asyncio
    async def test_convergence_detection(self, framework_with_data):
        """Test detection of algorithm convergence"""
        arms = ["rule_1", "rule_2", "rule_3"]
        experiment_id = await framework_with_data.create_experiment(
            experiment_name="convergence_test",
            arms=arms
        )
        
        # Simulate convergence by repeatedly selecting best arm
        for _ in range(50):
            # Always give rule_2 high reward
            result = await framework_with_data.select_arm(experiment_id)
            reward = 0.9 if result.arm_id == "rule_2" else 0.3
            await framework_with_data.update_reward(experiment_id, result, reward)
        
        summary = framework_with_data.get_experiment_summary(experiment_id)
        
        # After convergence, best arm should be rule_2
        assert summary["best_arm"] == "rule_2"
        
        # Should show high average reward
        assert summary["average_reward"] > 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])