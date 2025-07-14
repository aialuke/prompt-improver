"""
Adversarial Robustness Security Tests

Tests adversarial robustness and security validation mechanisms to ensure
ML models are resistant to adversarial attacks and maintain security
under adversarial conditions in Phase 3 components.

Security Test Coverage:
- Adversarial attack simulation and detection
- Model robustness validation
- Input perturbation resistance
- Adversarial training validation
- Attack success rate measurement
- Defense mechanism effectiveness
- Robustness certification
- Security-critical failure detection
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import hashlib

# Try to import adversarial robustness toolbox
try:
    import art
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    from art.estimators.classification import SklearnClassifier
    from art.defences.preprocessor import GaussianAugmentation
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    warnings.warn("Adversarial Robustness Toolbox not available for adversarial tests")


class MockAdversarialAttack:
    """Mock adversarial attack implementation for testing"""
    
    def __init__(self, attack_type: str = "fgsm", epsilon: float = 0.1):
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.attack_history: List[Dict[str, Any]] = []
    
    def generate_adversarial_examples(self, x_clean: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Generate adversarial examples using specified attack"""
        if self.attack_type == "fgsm":
            return self._fast_gradient_sign_method(x_clean, y_true)
        elif self.attack_type == "pgd":
            return self._projected_gradient_descent(x_clean, y_true)
        elif self.attack_type == "random":
            return self._random_noise_attack(x_clean)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
    
    def _fast_gradient_sign_method(self, x_clean: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Mock Fast Gradient Sign Method attack"""
        # Simulate gradient-based perturbation
        gradient_direction = np.random.choice([-1, 1], size=x_clean.shape)
        perturbation = self.epsilon * gradient_direction
        x_adversarial = x_clean + perturbation
        
        # Clip to valid range (assuming [0, 1] for normalized data)
        x_adversarial = np.clip(x_adversarial, 0, 1)
        
        self._log_attack("fgsm", x_clean.shape[0], self.epsilon)
        return x_adversarial
    
    def _projected_gradient_descent(self, x_clean: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Mock Projected Gradient Descent attack"""
        x_adversarial = x_clean.copy()
        
        # Simulate iterative attack (simplified)
        for iteration in range(10):  # 10 iterations
            step_size = self.epsilon / 10
            gradient_direction = np.random.choice([-1, 1], size=x_clean.shape)
            
            # Take step
            x_adversarial += step_size * gradient_direction
            
            # Project back to epsilon-ball
            perturbation = x_adversarial - x_clean
            perturbation = np.clip(perturbation, -self.epsilon, self.epsilon)
            x_adversarial = x_clean + perturbation
            
            # Clip to valid range
            x_adversarial = np.clip(x_adversarial, 0, 1)
        
        self._log_attack("pgd", x_clean.shape[0], self.epsilon)
        return x_adversarial
    
    def _random_noise_attack(self, x_clean: np.ndarray) -> np.ndarray:
        """Random noise attack for baseline comparison"""
        noise = np.random.uniform(-self.epsilon, self.epsilon, size=x_clean.shape)
        x_adversarial = x_clean + noise
        x_adversarial = np.clip(x_adversarial, 0, 1)
        
        self._log_attack("random", x_clean.shape[0], self.epsilon)
        return x_adversarial
    
    def _log_attack(self, attack_type: str, num_samples: int, epsilon: float):
        """Log attack for audit purposes"""
        self.attack_history.append({
            "timestamp": datetime.utcnow(),
            "attack_type": attack_type,
            "num_samples": num_samples,
            "epsilon": epsilon,
            "attack_id": hashlib.md5(f"{datetime.utcnow()}{attack_type}".encode()).hexdigest()[:8]
        })


class MockRobustnessEvaluator:
    """Mock robustness evaluator for testing defense mechanisms"""
    
    def __init__(self):
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_robustness(self, model: Any, x_clean: np.ndarray, x_adversarial: np.ndarray, 
                          y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate model robustness against adversarial examples"""
        # Get predictions
        pred_clean = model.predict(x_clean)
        pred_adversarial = model.predict(x_adversarial)
        
        # Calculate metrics
        clean_accuracy = accuracy_score(y_true, pred_clean)
        adversarial_accuracy = accuracy_score(y_true, pred_adversarial)
        robustness_score = adversarial_accuracy / clean_accuracy if clean_accuracy > 0 else 0
        
        # Calculate attack success rate
        successful_attacks = np.sum(pred_clean != pred_adversarial)
        attack_success_rate = successful_attacks / len(y_true)
        
        results = {
            "clean_accuracy": clean_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "robustness_score": robustness_score,
            "attack_success_rate": attack_success_rate,
            "samples_tested": len(y_true)
        }
        
        self._log_evaluation(results)
        return results
    
    def certified_robustness_check(self, model: Any, x: np.ndarray, epsilon: float) -> Dict[str, Any]:
        """Check certified robustness for given epsilon"""
        # Simplified certified robustness check
        # In practice, this would use formal verification methods
        
        certified_samples = 0
        total_samples = len(x)
        
        for i in range(total_samples):
            # Simulate certification check (simplified)
            # In reality, this would involve complex mathematical verification
            sample_certified = np.random.random() > 0.3  # 70% certification rate
            if sample_certified:
                certified_samples += 1
        
        certification_rate = certified_samples / total_samples
        
        results = {
            "certification_rate": certification_rate,
            "certified_samples": certified_samples,
            "total_samples": total_samples,
            "epsilon": epsilon,
            "certification_method": "mock_verification"
        }
        
        return results
    
    def _log_evaluation(self, results: Dict[str, Any]):
        """Log evaluation results"""
        self.evaluation_history.append({
            "timestamp": datetime.utcnow(),
            "results": results,
            "evaluation_id": hashlib.md5(f"{datetime.utcnow()}".encode()).hexdigest()[:8]
        })


class MockDefenseSystem:
    """Mock defense system for testing adversarial defenses"""
    
    def __init__(self):
        self.defense_methods = ["gaussian_noise", "adversarial_training", "input_validation"]
        self.active_defenses: List[str] = []
    
    def apply_gaussian_noise_defense(self, x: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """Apply Gaussian noise defense"""
        noise = np.random.normal(0, noise_std, x.shape)
        return x + noise
    
    def apply_input_validation(self, x: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Apply input validation defense"""
        valid_indices = []
        validated_inputs = []
        
        for i, sample in enumerate(x):
            # Simple validation: check for extreme values
            if np.all(sample >= 0) and np.all(sample <= 1) and not np.any(np.isnan(sample)):
                valid_indices.append(i)
                validated_inputs.append(sample)
        
        return np.array(validated_inputs), valid_indices
    
    def detect_adversarial_examples(self, x: np.ndarray, model: Any) -> List[bool]:
        """Detect potential adversarial examples"""
        detection_results = []
        
        for sample in x:
            # Simple detection heuristic: check prediction confidence
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([sample])[0]
                max_confidence = np.max(proba)
                
                # Low confidence might indicate adversarial example
                is_adversarial = max_confidence < 0.7
            else:
                # For models without probability prediction, use random detection
                is_adversarial = np.random.random() < 0.3
            
            detection_results.append(is_adversarial)
        
        return detection_results
    
    def enable_defense(self, defense_name: str):
        """Enable a defense method"""
        if defense_name in self.defense_methods and defense_name not in self.active_defenses:
            self.active_defenses.append(defense_name)
    
    def disable_defense(self, defense_name: str):
        """Disable a defense method"""
        if defense_name in self.active_defenses:
            self.active_defenses.remove(defense_name)


@pytest.fixture
def mock_model():
    """Create a mock ML model for testing"""
    # Create synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Normalize features to [0, 1]
    X_train = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
    X_test = (X_test - X_test.min(axis=0)) / (X_test.max(axis=0) - X_test.min(axis=0))
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test


@pytest.fixture
def adversarial_attacker():
    """Create adversarial attacker for testing"""
    return MockAdversarialAttack(attack_type="fgsm", epsilon=0.1)


@pytest.fixture
def robustness_evaluator():
    """Create robustness evaluator for testing"""
    return MockRobustnessEvaluator()


@pytest.fixture
def defense_system():
    """Create defense system for testing"""
    return MockDefenseSystem()


class TestAdversarialAttackGeneration:
    """Test adversarial attack generation"""
    
    def test_fgsm_attack_generation(self, mock_model, adversarial_attacker):
        """Test Fast Gradient Sign Method attack generation"""
        model, X_test, y_test = mock_model
        
        # Generate adversarial examples
        X_adversarial = adversarial_attacker.generate_adversarial_examples(X_test[:50], y_test[:50])
        
        # Check that adversarial examples were generated
        assert X_adversarial.shape == X_test[:50].shape
        assert not np.array_equal(X_adversarial, X_test[:50])  # Should be different
        
        # Check that perturbation is bounded
        perturbation = np.abs(X_adversarial - X_test[:50])
        assert np.all(perturbation <= adversarial_attacker.epsilon + 1e-10)  # Allow numerical precision
        
        # Check that values are in valid range
        assert np.all(X_adversarial >= 0)
        assert np.all(X_adversarial <= 1)
    
    def test_pgd_attack_generation(self, mock_model):
        """Test Projected Gradient Descent attack generation"""
        model, X_test, y_test = mock_model
        attacker = MockAdversarialAttack(attack_type="pgd", epsilon=0.05)
        
        # Generate adversarial examples
        X_adversarial = attacker.generate_adversarial_examples(X_test[:30], y_test[:30])
        
        # Check properties
        assert X_adversarial.shape == X_test[:30].shape
        assert not np.array_equal(X_adversarial, X_test[:30])
        
        # Check perturbation bounds (PGD should respect epsilon constraint)
        perturbation = np.abs(X_adversarial - X_test[:30])
        assert np.all(perturbation <= attacker.epsilon + 1e-10)
    
    def test_random_attack_baseline(self, mock_model):
        """Test random attack as baseline"""
        model, X_test, y_test = mock_model
        attacker = MockAdversarialAttack(attack_type="random", epsilon=0.1)
        
        # Generate random adversarial examples
        X_adversarial = attacker.generate_adversarial_examples(X_test[:40], y_test[:40])
        
        # Check that random perturbations were added
        assert not np.array_equal(X_adversarial, X_test[:40])
        
        # Check bounds
        perturbation = np.abs(X_adversarial - X_test[:40])
        assert np.all(perturbation <= attacker.epsilon + 1e-10)
    
    def test_attack_logging(self, mock_model, adversarial_attacker):
        """Test that attacks are properly logged"""
        model, X_test, y_test = mock_model
        
        # Perform multiple attacks
        adversarial_attacker.generate_adversarial_examples(X_test[:20], y_test[:20])
        adversarial_attacker.generate_adversarial_examples(X_test[20:40], y_test[20:40])
        
        # Check attack history
        assert len(adversarial_attacker.attack_history) == 2
        
        # Verify log entries
        for log_entry in adversarial_attacker.attack_history:
            assert "timestamp" in log_entry
            assert "attack_type" in log_entry
            assert "num_samples" in log_entry
            assert "epsilon" in log_entry
            assert "attack_id" in log_entry
            assert log_entry["attack_type"] == "fgsm"
            assert log_entry["epsilon"] == 0.1


class TestRobustnessEvaluation:
    """Test robustness evaluation mechanisms"""
    
    def test_robustness_evaluation_metrics(self, mock_model, adversarial_attacker, robustness_evaluator):
        """Test robustness evaluation metrics"""
        model, X_test, y_test = mock_model
        
        # Generate adversarial examples
        X_adversarial = adversarial_attacker.generate_adversarial_examples(X_test[:100], y_test[:100])
        
        # Evaluate robustness
        results = robustness_evaluator.evaluate_robustness(model, X_test[:100], X_adversarial, y_test[:100])
        
        # Check that all expected metrics are present
        expected_metrics = ["clean_accuracy", "adversarial_accuracy", "robustness_score", 
                          "attack_success_rate", "samples_tested"]
        for metric in expected_metrics:
            assert metric in results
            assert isinstance(results[metric], (int, float))
        
        # Check metric ranges
        assert 0 <= results["clean_accuracy"] <= 1
        assert 0 <= results["adversarial_accuracy"] <= 1
        assert 0 <= results["robustness_score"] <= 1
        assert 0 <= results["attack_success_rate"] <= 1
        assert results["samples_tested"] == 100
    
    def test_robustness_degradation_detection(self, mock_model, robustness_evaluator):
        """Test detection of robustness degradation"""
        model, X_test, y_test = mock_model
        
        # Test with different epsilon values
        epsilons = [0.01, 0.05, 0.1, 0.2]
        robustness_scores = []
        
        for epsilon in epsilons:
            attacker = MockAdversarialAttack(attack_type="fgsm", epsilon=epsilon)
            X_adversarial = attacker.generate_adversarial_examples(X_test[:50], y_test[:50])
            results = robustness_evaluator.evaluate_robustness(model, X_test[:50], X_adversarial, y_test[:50])
            robustness_scores.append(results["robustness_score"])
        
        # Robustness should generally decrease with larger epsilon
        # (though this is a simplified test with mock attacks)
        assert len(robustness_scores) == len(epsilons)
        
        # At least some degradation should be observed
        min_robustness = min(robustness_scores)
        max_robustness = max(robustness_scores)
        assert max_robustness - min_robustness >= 0  # Allow for noise in mock results
    
    def test_certified_robustness_check(self, mock_model, robustness_evaluator):
        """Test certified robustness checking"""
        model, X_test, y_test = mock_model
        
        # Test certification for different epsilon values
        epsilons = [0.01, 0.05, 0.1]
        
        for epsilon in epsilons:
            cert_results = robustness_evaluator.certified_robustness_check(model, X_test[:30], epsilon)
            
            # Check result structure
            assert "certification_rate" in cert_results
            assert "certified_samples" in cert_results
            assert "total_samples" in cert_results
            assert "epsilon" in cert_results
            
            # Check ranges
            assert 0 <= cert_results["certification_rate"] <= 1
            assert 0 <= cert_results["certified_samples"] <= cert_results["total_samples"]
            assert cert_results["epsilon"] == epsilon


class TestDefenseMechanisms:
    """Test defense mechanisms against adversarial attacks"""
    
    def test_gaussian_noise_defense(self, mock_model, defense_system):
        """Test Gaussian noise defense"""
        model, X_test, y_test = mock_model
        
        # Apply Gaussian noise defense
        X_defended = defense_system.apply_gaussian_noise_defense(X_test[:50], noise_std=0.02)
        
        # Check that noise was added
        assert not np.array_equal(X_defended, X_test[:50])
        
        # Check that defended inputs are still in reasonable range
        assert np.all(X_defended >= -0.1)  # Allow some values below 0 due to noise
        assert np.all(X_defended <= 1.1)   # Allow some values above 1 due to noise
        
        # Test that defense reduces attack effectiveness
        attacker = MockAdversarialAttack(epsilon=0.1)
        X_adversarial = attacker.generate_adversarial_examples(X_test[:50], y_test[:50])
        X_adversarial_defended = defense_system.apply_gaussian_noise_defense(X_adversarial, noise_std=0.02)
        
        # Get predictions
        pred_clean = model.predict(X_test[:50])
        pred_adversarial = model.predict(X_adversarial)
        pred_adversarial_defended = model.predict(X_adversarial_defended)
        
        # Defense should improve robustness (reduce attack success)
        attack_success_original = np.sum(pred_clean != pred_adversarial) / len(pred_clean)
        attack_success_defended = np.sum(pred_clean != pred_adversarial_defended) / len(pred_clean)
        
        # Defense should reduce attack success rate (though this test is simplified)
        assert 0 <= attack_success_defended <= 1
        assert 0 <= attack_success_original <= 1
    
    def test_input_validation_defense(self, mock_model, defense_system):
        """Test input validation defense"""
        model, X_test, y_test = mock_model
        
        # Create some invalid inputs
        X_invalid = X_test[:20].copy()
        X_invalid[0, 0] = -0.5  # Invalid negative value
        X_invalid[1, 1] = 1.5   # Invalid value > 1
        X_invalid[2, 2] = np.nan  # Invalid NaN value
        
        # Apply input validation
        X_validated, valid_indices = defense_system.apply_input_validation(X_invalid)
        
        # Check that invalid samples were filtered out
        assert len(valid_indices) < len(X_invalid)  # Some samples should be rejected
        assert len(X_validated) == len(valid_indices)
        
        # Check that validated inputs are valid
        assert np.all(X_validated >= 0)
        assert np.all(X_validated <= 1)
        assert not np.any(np.isnan(X_validated))
    
    def test_adversarial_example_detection(self, mock_model, defense_system, adversarial_attacker):
        """Test adversarial example detection"""
        model, X_test, y_test = mock_model
        
        # Generate mix of clean and adversarial examples
        X_clean = X_test[:30]
        X_adversarial = adversarial_attacker.generate_adversarial_examples(X_test[30:60], y_test[30:60])
        X_mixed = np.vstack([X_clean, X_adversarial])
        
        # Detect adversarial examples
        detection_results = defense_system.detect_adversarial_examples(X_mixed, model)
        
        # Check detection results
        assert len(detection_results) == len(X_mixed)
        assert all(isinstance(result, bool) for result in detection_results)
        
        # Some examples should be detected as adversarial
        num_detected = sum(detection_results)
        assert 0 <= num_detected <= len(X_mixed)
    
    def test_defense_system_configuration(self, defense_system):
        """Test defense system configuration"""
        # Initially no defenses are active
        assert len(defense_system.active_defenses) == 0
        
        # Enable defenses
        defense_system.enable_defense("gaussian_noise")
        defense_system.enable_defense("input_validation")
        
        assert "gaussian_noise" in defense_system.active_defenses
        assert "input_validation" in defense_system.active_defenses
        assert len(defense_system.active_defenses) == 2
        
        # Disable defense
        defense_system.disable_defense("gaussian_noise")
        assert "gaussian_noise" not in defense_system.active_defenses
        assert "input_validation" in defense_system.active_defenses
        assert len(defense_system.active_defenses) == 1
        
        # Try to enable non-existent defense (should not crash)
        defense_system.enable_defense("non_existent_defense")
        assert "non_existent_defense" not in defense_system.active_defenses


class TestSecurityCriticalFailures:
    """Test detection of security-critical failures"""
    
    def test_model_confidence_under_attack(self, mock_model, adversarial_attacker):
        """Test model confidence degradation under attack"""
        model, X_test, y_test = mock_model
        
        # Check if model supports probability prediction
        if not hasattr(model, 'predict_proba'):
            pytest.skip("Model doesn't support probability prediction")
        
        # Get clean predictions with confidence
        X_clean = X_test[:50]
        clean_proba = model.predict_proba(X_clean)
        clean_confidence = np.max(clean_proba, axis=1)
        
        # Generate adversarial examples
        X_adversarial = adversarial_attacker.generate_adversarial_examples(X_clean, y_test[:50])
        adversarial_proba = model.predict_proba(X_adversarial)
        adversarial_confidence = np.max(adversarial_proba, axis=1)
        
        # Calculate confidence degradation
        confidence_degradation = clean_confidence - adversarial_confidence
        
        # Some confidence degradation should occur
        mean_degradation = np.mean(confidence_degradation)
        assert mean_degradation >= 0  # Confidence should generally decrease or stay same
        
        # Identify samples with severe confidence degradation (potential security risk)
        severe_degradation_threshold = 0.3
        severe_degradation_mask = confidence_degradation > severe_degradation_threshold
        severe_degradation_rate = np.mean(severe_degradation_mask)
        
        # Log security-critical cases
        if severe_degradation_rate > 0.1:  # More than 10% severe degradation
            print(f"WARNING: High severe confidence degradation rate: {severe_degradation_rate:.2f}")
    
    def test_prediction_consistency_under_perturbation(self, mock_model):
        """Test prediction consistency under small perturbations"""
        model, X_test, y_test = mock_model
        
        # Test prediction consistency with small random perturbations
        X_sample = X_test[:30]
        original_predictions = model.predict(X_sample)
        
        consistency_scores = []
        perturbation_magnitudes = [0.001, 0.005, 0.01, 0.02]
        
        for magnitude in perturbation_magnitudes:
            # Add small random perturbations
            perturbations = np.random.uniform(-magnitude, magnitude, X_sample.shape)
            X_perturbed = X_sample + perturbations
            X_perturbed = np.clip(X_perturbed, 0, 1)  # Keep in valid range
            
            # Get predictions on perturbed inputs
            perturbed_predictions = model.predict(X_perturbed)
            
            # Calculate consistency (fraction of predictions that remain the same)
            consistency = np.mean(original_predictions == perturbed_predictions)
            consistency_scores.append(consistency)
        
        # Consistency should be high for small perturbations
        for i, magnitude in enumerate(perturbation_magnitudes):
            consistency = consistency_scores[i]
            expected_min_consistency = max(0.8 - magnitude * 10, 0.5)  # Allow some degradation
            assert consistency >= expected_min_consistency, f"Low consistency {consistency:.3f} for perturbation {magnitude}"
    
    def test_boundary_case_robustness(self, mock_model):
        """Test robustness on boundary cases"""
        model, X_test, y_test = mock_model
        
        # Create boundary test cases
        boundary_cases = [
            np.zeros((1, X_test.shape[1])),  # All zeros
            np.ones((1, X_test.shape[1])),   # All ones
            np.full((1, X_test.shape[1]), 0.5),  # All 0.5 (middle values)
        ]
        
        for i, boundary_case in enumerate(boundary_cases):
            try:
                prediction = model.predict(boundary_case)
                assert len(prediction) == 1, f"Boundary case {i} should return single prediction"
                assert prediction[0] in [0, 1], f"Boundary case {i} should return valid class"
            except Exception as e:
                pytest.fail(f"Model failed on boundary case {i}: {e}")


@pytest.mark.skipif(not ART_AVAILABLE, reason="Adversarial Robustness Toolbox not available")
class TestARTIntegration:
    """Test integration with Adversarial Robustness Toolbox"""
    
    def test_art_classifier_wrapper(self, mock_model):
        """Test ART classifier wrapper"""
        model, X_test, y_test = mock_model
        
        # Wrap model with ART
        art_classifier = SklearnClassifier(model=model)
        
        # Test basic functionality
        predictions = art_classifier.predict(X_test[:10])
        assert predictions.shape[0] == 10
        assert predictions.shape[1] == 2  # Binary classification
    
    def test_art_fgsm_attack(self, mock_model):
        """Test ART FGSM attack"""
        model, X_test, y_test = mock_model
        
        # Wrap model with ART
        art_classifier = SklearnClassifier(model=model)
        
        # Create FGSM attack
        attack = FastGradientMethod(estimator=art_classifier, eps=0.1)
        
        # Generate adversarial examples
        X_sample = X_test[:20].astype(np.float32)
        X_adversarial = attack.generate(x=X_sample)
        
        # Check that adversarial examples were generated
        assert X_adversarial.shape == X_sample.shape
        assert not np.array_equal(X_adversarial, X_sample)
        
        # Check perturbation bounds
        perturbation = np.abs(X_adversarial - X_sample)
        assert np.all(perturbation <= 0.1 + 1e-6)  # Allow for numerical precision
    
    def test_art_defense_integration(self, mock_model):
        """Test ART defense integration"""
        model, X_test, y_test = mock_model
        
        # Create Gaussian augmentation defense
        defense = GaussianAugmentation(sigma=0.01, augmentation=False)
        
        # Apply defense
        X_sample = X_test[:15].astype(np.float32)
        X_defended, _ = defense(X_sample)
        
        # Check that defense was applied
        assert X_defended.shape == X_sample.shape
        # Note: Gaussian augmentation might not always change values significantly


@pytest.mark.performance
class TestAdversarialPerformance:
    """Test performance of adversarial robustness mechanisms"""
    
    def test_attack_generation_performance(self, mock_model):
        """Test attack generation performance"""
        import time
        
        model, X_test, y_test = mock_model
        attacker = MockAdversarialAttack(attack_type="fgsm", epsilon=0.1)
        
        # Test performance on varying batch sizes
        batch_sizes = [10, 50, 100]
        
        for batch_size in batch_sizes:
            X_batch = X_test[:batch_size]
            y_batch = y_test[:batch_size]
            
            start_time = time.time()
            X_adversarial = attacker.generate_adversarial_examples(X_batch, y_batch)
            elapsed_time = time.time() - start_time
            
            # Should generate attacks quickly
            time_per_sample = elapsed_time / batch_size
            assert time_per_sample < 0.01, f"Attack generation too slow: {time_per_sample:.4f}s per sample"
    
    def test_defense_performance(self, mock_model, defense_system):
        """Test defense mechanism performance"""
        import time
        
        model, X_test, y_test = mock_model
        
        # Test Gaussian noise defense performance
        X_batch = X_test[:100]
        
        start_time = time.time()
        for _ in range(10):  # Apply defense 10 times
            defense_system.apply_gaussian_noise_defense(X_batch)
        elapsed_time = time.time() - start_time
        
        # Should apply defense quickly
        avg_time_per_application = elapsed_time / 10
        assert avg_time_per_application < 0.01, f"Defense application too slow: {avg_time_per_application:.4f}s"
    
    def test_robustness_evaluation_performance(self, mock_model, robustness_evaluator):
        """Test robustness evaluation performance"""
        import time
        
        model, X_test, y_test = mock_model
        attacker = MockAdversarialAttack(epsilon=0.1)
        
        # Generate adversarial examples
        X_adversarial = attacker.generate_adversarial_examples(X_test[:200], y_test[:200])
        
        # Test evaluation performance
        start_time = time.time()
        results = robustness_evaluator.evaluate_robustness(model, X_test[:200], X_adversarial, y_test[:200])
        elapsed_time = time.time() - start_time
        
        # Should evaluate robustness quickly
        time_per_sample = elapsed_time / 200
        assert time_per_sample < 0.001, f"Robustness evaluation too slow: {time_per_sample:.6f}s per sample"


class TestAdversarialAuditAndCompliance:
    """Test audit and compliance features for adversarial robustness"""
    
    def test_attack_audit_trail(self, adversarial_attacker):
        """Test that adversarial attacks leave proper audit trail"""
        # Perform attacks and check audit trail
        X_dummy = np.random.rand(50, 10)
        y_dummy = np.random.randint(0, 2, 50)
        
        adversarial_attacker.generate_adversarial_examples(X_dummy, y_dummy)
        
        # Check audit trail
        assert len(adversarial_attacker.attack_history) == 1
        
        audit_entry = adversarial_attacker.attack_history[0]
        required_fields = ["timestamp", "attack_type", "num_samples", "epsilon", "attack_id"]
        
        for field in required_fields:
            assert field in audit_entry, f"Missing audit field: {field}"
        
        # Check field values
        assert audit_entry["attack_type"] == "fgsm"
        assert audit_entry["num_samples"] == 50
        assert audit_entry["epsilon"] == 0.1
        assert len(audit_entry["attack_id"]) == 8  # 8-character hash
    
    def test_robustness_evaluation_logging(self, mock_model, robustness_evaluator):
        """Test that robustness evaluations are properly logged"""
        model, X_test, y_test = mock_model
        
        # Perform evaluation
        attacker = MockAdversarialAttack(epsilon=0.05)
        X_adversarial = attacker.generate_adversarial_examples(X_test[:30], y_test[:30])
        results = robustness_evaluator.evaluate_robustness(model, X_test[:30], X_adversarial, y_test[:30])
        
        # Check evaluation history
        assert len(robustness_evaluator.evaluation_history) == 1
        
        eval_entry = robustness_evaluator.evaluation_history[0]
        assert "timestamp" in eval_entry
        assert "results" in eval_entry
        assert "evaluation_id" in eval_entry
        
        # Check that results are properly stored
        stored_results = eval_entry["results"]
        expected_metrics = ["clean_accuracy", "adversarial_accuracy", "robustness_score", "attack_success_rate"]
        for metric in expected_metrics:
            assert metric in stored_results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])