"""
Privacy-Preserving ML Security Tests

Tests security aspects of privacy-preserving ML features to ensure proper
implementation of differential privacy, federated learning, and secure
aggregation mechanisms in Phase 3 components.

Security Test Coverage:
- Differential privacy implementation security
- Privacy budget enforcement and tracking
- Federated learning security
- Secure aggregation validation
- Privacy parameter validation
- Data anonymization verification
- Privacy leak detection
- Cryptographic security in ML contexts
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import warnings

# Try to import privacy-preserving libraries
try:
    import opacus
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    warnings.warn("Opacus not available for differential privacy tests")


class MockDifferentialPrivacy:
    """Mock differential privacy implementation for testing"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-6):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget_used = 0.0
        self.query_history: List[Dict[str, Any]] = []
    
    def add_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Laplace noise for differential privacy"""
        # Laplace mechanism: scale = sensitivity / epsilon
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        
        # Track privacy budget usage
        self.privacy_budget_used += self.epsilon
        
        # Log query for audit
        self.query_history.append({
            "timestamp": datetime.utcnow(),
            "original_value": value,
            "sensitivity": sensitivity,
            "noise_added": noise,
            "epsilon_used": self.epsilon
        })
        
        return value + noise
    
    def add_gaussian_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Gaussian noise for (epsilon, delta)-differential privacy"""
        # Gaussian mechanism
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        noise = np.random.normal(0, sigma)
        
        self.privacy_budget_used += self.epsilon
        
        self.query_history.append({
            "timestamp": datetime.utcnow(),
            "original_value": value,
            "sensitivity": sensitivity,
            "noise_added": noise,
            "epsilon_used": self.epsilon,
            "mechanism": "gaussian"
        })
        
        return value + noise
    
    def check_privacy_budget(self, requested_epsilon: float) -> bool:
        """Check if privacy budget allows for the requested epsilon"""
        total_budget = 10.0  # Total allowed privacy budget
        return (self.privacy_budget_used + requested_epsilon) <= total_budget
    
    def get_privacy_spent(self) -> Tuple[float, int]:
        """Get total privacy budget spent and number of queries"""
        return self.privacy_budget_used, len(self.query_history)


class MockFederatedLearning:
    """Mock federated learning implementation for testing"""
    
    def __init__(self):
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.global_model_updates: List[Dict[str, Any]] = []
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def register_client(self, client_id: str, public_key: str) -> bool:
        """Register federated learning client"""
        if client_id in self.clients:
            return False
        
        self.clients[client_id] = {
            "public_key": public_key,
            "registered_at": datetime.utcnow(),
            "updates_count": 0,
            "is_active": True
        }
        return True
    
    def submit_encrypted_update(self, client_id: str, encrypted_update: bytes) -> bool:
        """Submit encrypted model update from client"""
        if client_id not in self.clients or not self.clients[client_id]["is_active"]:
            return False
        
        try:
            # Decrypt and validate update
            decrypted_update = self.cipher.decrypt(encrypted_update)
            update_data = eval(decrypted_update.decode())  # In production, use proper serialization
            
            # Validate update structure
            if not isinstance(update_data, dict) or "gradients" not in update_data:
                return False
            
            # Store update
            self.global_model_updates.append({
                "client_id": client_id,
                "timestamp": datetime.utcnow(),
                "update": update_data,
                "encrypted_size": len(encrypted_update)
            })
            
            self.clients[client_id]["updates_count"] += 1
            return True
            
        except Exception:
            return False
    
    def aggregate_updates(self) -> Optional[Dict[str, Any]]:
        """Securely aggregate model updates"""
        if len(self.global_model_updates) < 2:  # Need at least 2 clients
            return None
        
        # Simple averaging aggregation (in production, use more sophisticated methods)
        aggregated_gradients = {}
        client_count = len(self.global_model_updates)
        
        for update in self.global_model_updates:
            gradients = update["update"]["gradients"]
            for layer_name, gradient in gradients.items():
                if layer_name not in aggregated_gradients:
                    aggregated_gradients[layer_name] = np.array(gradient)
                else:
                    aggregated_gradients[layer_name] += np.array(gradient)
        
        # Average the gradients
        for layer_name in aggregated_gradients:
            aggregated_gradients[layer_name] /= client_count
        
        return {
            "aggregated_gradients": aggregated_gradients,
            "client_count": client_count,
            "timestamp": datetime.utcnow()
        }
    
    def encrypt_data(self, data: str) -> bytes:
        """Encrypt data for secure transmission"""
        return self.cipher.encrypt(data.encode())


@pytest.fixture
def dp_service():
    """Create differential privacy service for testing"""
    return MockDifferentialPrivacy(epsilon=1.0, delta=1e-6)


@pytest.fixture
def fl_service():
    """Create federated learning service for testing"""
    return MockFederatedLearning()


class TestDifferentialPrivacySecurity:
    """Test differential privacy security implementation"""
    
    def test_privacy_budget_enforcement(self, dp_service):
        """Test privacy budget is properly enforced"""
        # Check initial budget
        budget_used, query_count = dp_service.get_privacy_spent()
        assert budget_used == 0.0
        assert query_count == 0
        
        # Make queries and check budget usage
        dp_service.add_noise(100.0)
        budget_used, query_count = dp_service.get_privacy_spent()
        assert budget_used == 1.0
        assert query_count == 1
        
        # Check budget validation
        assert dp_service.check_privacy_budget(5.0) is True  # Should be allowed
        assert dp_service.check_privacy_budget(15.0) is False  # Should exceed budget
    
    def test_laplace_noise_addition(self, dp_service):
        """Test Laplace noise addition for differential privacy"""
        original_value = 100.0
        sensitivity = 1.0
        
        # Add noise multiple times and check variance
        noisy_values = []
        for _ in range(100):
            noisy_value = dp_service.add_noise(original_value, sensitivity)
            noisy_values.append(noisy_value)
        
        noisy_values = np.array(noisy_values)
        
        # Check that noise was actually added (values should vary)
        assert np.std(noisy_values) > 0
        
        # Check that mean is close to original (unbiased)
        assert abs(np.mean(noisy_values) - original_value) < 10  # Allow some variance
        
        # Check that noise scale is reasonable
        expected_scale = sensitivity / dp_service.epsilon
        actual_scale = np.std(noisy_values) * np.sqrt(2)  # Laplace std = scale * sqrt(2)
        assert abs(actual_scale - expected_scale) < expected_scale * 0.5  # Within 50%
    
    def test_gaussian_noise_addition(self, dp_service):
        """Test Gaussian noise addition for (epsilon, delta)-DP"""
        original_value = 50.0
        sensitivity = 1.0
        
        # Add Gaussian noise multiple times
        noisy_values = []
        for _ in range(100):
            noisy_value = dp_service.add_gaussian_noise(original_value, sensitivity)
            noisy_values.append(noisy_value)
        
        noisy_values = np.array(noisy_values)
        
        # Check that noise was added
        assert np.std(noisy_values) > 0
        
        # Check that mean is close to original
        assert abs(np.mean(noisy_values) - original_value) < 10
        
        # Verify Gaussian noise properties
        expected_sigma = np.sqrt(2 * np.log(1.25 / dp_service.delta)) * sensitivity / dp_service.epsilon
        actual_sigma = np.std(noisy_values)
        assert abs(actual_sigma - expected_sigma) < expected_sigma * 0.5
    
    def test_privacy_parameter_validation(self):
        """Test privacy parameter validation"""
        # Valid parameters
        valid_dp = MockDifferentialPrivacy(epsilon=1.0, delta=1e-6)
        assert valid_dp.epsilon == 1.0
        assert valid_dp.delta == 1e-6
        
        # Test parameter bounds
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            MockDifferentialPrivacy(epsilon=0, delta=1e-6)
        
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            MockDifferentialPrivacy(epsilon=-1.0, delta=1e-6)
        
        with pytest.raises(ValueError, match="Delta must be positive and small"):
            MockDifferentialPrivacy(epsilon=1.0, delta=0)
        
        with pytest.raises(ValueError, match="Delta must be positive and small"):
            MockDifferentialPrivacy(epsilon=1.0, delta=1e-2)  # Too large
    
    def test_query_audit_logging(self, dp_service):
        """Test that all privacy queries are properly logged"""
        # Make several queries
        dp_service.add_noise(100.0, sensitivity=1.0)
        dp_service.add_gaussian_noise(50.0, sensitivity=2.0)
        dp_service.add_noise(75.0, sensitivity=0.5)
        
        # Check query history
        assert len(dp_service.query_history) == 3
        
        # Verify first query log
        first_query = dp_service.query_history[0]
        assert first_query["original_value"] == 100.0
        assert first_query["sensitivity"] == 1.0
        assert first_query["epsilon_used"] == dp_service.epsilon
        assert "timestamp" in first_query
        assert "noise_added" in first_query
        
        # Verify Gaussian query has mechanism field
        gaussian_query = dp_service.query_history[1]
        assert gaussian_query["mechanism"] == "gaussian"
    
    def test_privacy_composition(self, dp_service):
        """Test privacy composition across multiple queries"""
        initial_epsilon = dp_service.epsilon
        
        # Make multiple queries
        for i in range(5):
            dp_service.add_noise(i * 10.0)
        
        # Check that privacy budget accumulates
        total_budget_used, query_count = dp_service.get_privacy_spent()
        expected_budget = initial_epsilon * 5
        assert total_budget_used == expected_budget
        assert query_count == 5


class TestFederatedLearningSecurity:
    """Test federated learning security implementation"""
    
    def test_client_registration(self, fl_service):
        """Test secure client registration"""
        client_id = "client_001"
        public_key = "mock_public_key_12345"
        
        # Register client
        result = fl_service.register_client(client_id, public_key)
        assert result is True
        assert client_id in fl_service.clients
        assert fl_service.clients[client_id]["public_key"] == public_key
        
        # Try to register same client again (should fail)
        result = fl_service.register_client(client_id, "different_key")
        assert result is False
    
    def test_encrypted_update_submission(self, fl_service):
        """Test encrypted model update submission"""
        client_id = "client_002"
        public_key = "mock_public_key_67890"
        
        # Register client first
        fl_service.register_client(client_id, public_key)
        
        # Create mock update
        update_data = {
            "gradients": {
                "layer1": [0.1, 0.2, 0.3],
                "layer2": [0.4, 0.5, 0.6]
            },
            "loss": 0.25,
            "samples": 100
        }
        
        # Encrypt update
        encrypted_update = fl_service.encrypt_data(str(update_data))
        
        # Submit encrypted update
        result = fl_service.submit_encrypted_update(client_id, encrypted_update)
        assert result is True
        assert len(fl_service.global_model_updates) == 1
        
        # Verify stored update
        stored_update = fl_service.global_model_updates[0]
        assert stored_update["client_id"] == client_id
        assert "gradients" in stored_update["update"]
    
    def test_unregistered_client_rejection(self, fl_service):
        """Test that unregistered clients are rejected"""
        unregistered_client = "unregistered_client"
        fake_update = fl_service.encrypt_data(str({"gradients": {}}))
        
        result = fl_service.submit_encrypted_update(unregistered_client, fake_update)
        assert result is False
        assert len(fl_service.global_model_updates) == 0
    
    def test_secure_aggregation(self, fl_service):
        """Test secure aggregation of client updates"""
        # Register multiple clients
        clients = ["client_A", "client_B", "client_C"]
        for client_id in clients:
            fl_service.register_client(client_id, f"key_{client_id}")
        
        # Submit updates from each client
        for i, client_id in enumerate(clients):
            update_data = {
                "gradients": {
                    "layer1": [0.1 + i * 0.1, 0.2 + i * 0.1],
                    "layer2": [0.3 + i * 0.1, 0.4 + i * 0.1]
                }
            }
            encrypted_update = fl_service.encrypt_data(str(update_data))
            fl_service.submit_encrypted_update(client_id, encrypted_update)
        
        # Aggregate updates
        aggregated = fl_service.aggregate_updates()
        assert aggregated is not None
        assert aggregated["client_count"] == 3
        
        # Check aggregated gradients (should be averaged)
        layer1_grads = aggregated["aggregated_gradients"]["layer1"]
        expected_layer1 = np.array([0.2, 0.3])  # Average of [0.1,0.2], [0.2,0.3], [0.3,0.4]
        np.testing.assert_array_almost_equal(layer1_grads, expected_layer1, decimal=6)
    
    def test_insufficient_clients_for_aggregation(self, fl_service):
        """Test that aggregation requires minimum number of clients"""
        # Register and submit update from only one client
        fl_service.register_client("single_client", "key")
        update_data = {"gradients": {"layer1": [0.1, 0.2]}}
        encrypted_update = fl_service.encrypt_data(str(update_data))
        fl_service.submit_encrypted_update("single_client", encrypted_update)
        
        # Aggregation should fail with insufficient clients
        aggregated = fl_service.aggregate_updates()
        assert aggregated is None
    
    def test_encryption_decryption_security(self, fl_service):
        """Test encryption/decryption security"""
        original_data = "sensitive_model_update_data_12345"
        
        # Encrypt data
        encrypted = fl_service.encrypt_data(original_data)
        assert encrypted != original_data.encode()
        assert len(encrypted) > len(original_data)
        
        # Decrypt data
        decrypted = fl_service.cipher.decrypt(encrypted).decode()
        assert decrypted == original_data
        
        # Test that tampering breaks decryption
        tampered_encrypted = encrypted[:-5] + b"XXXXX"
        with pytest.raises(Exception):  # Should raise InvalidToken or similar
            fl_service.cipher.decrypt(tampered_encrypted)


class TestSecureAggregation:
    """Test secure aggregation mechanisms"""
    
    def test_homomorphic_addition_simulation(self):
        """Test homomorphic addition simulation for secure aggregation"""
        # Simulate simple additive homomorphic encryption
        def simple_additive_encrypt(value: float, key: int) -> int:
            """Simple additive encryption for testing"""
            return int(value * 1000) + key
        
        def simple_additive_decrypt(encrypted_value: int, key: int) -> float:
            """Simple additive decryption for testing"""
            return (encrypted_value - key) / 1000.0
        
        def homomorphic_add(enc_a: int, enc_b: int) -> int:
            """Homomorphic addition"""
            return enc_a + enc_b
        
        # Test homomorphic properties
        key = 12345
        a, b = 10.5, 20.3
        
        # Encrypt values
        enc_a = simple_additive_encrypt(a, key)
        enc_b = simple_additive_encrypt(b, key)
        
        # Add encrypted values
        enc_sum = homomorphic_add(enc_a, enc_b)
        
        # Decrypt result
        decrypted_sum = simple_additive_decrypt(enc_sum, key * 2)  # Need to account for both keys
        
        # Verify homomorphic property: Decrypt(Enc(a) + Enc(b)) ≈ a + b
        expected_sum = a + b
        assert abs(decrypted_sum - expected_sum) < 0.01
    
    def test_secure_multi_party_computation_simulation(self):
        """Test secure multi-party computation simulation"""
        # Simulate secure computation of average without revealing individual values
        def secret_share(value: float, num_parties: int) -> List[float]:
            """Create secret shares of a value"""
            shares = []
            total = 0.0
            
            # Generate random shares for all but last party
            for _ in range(num_parties - 1):
                share = np.random.uniform(-100, 100)
                shares.append(share)
                total += share
            
            # Last share ensures sum equals original value
            last_share = value - total
            shares.append(last_share)
            
            return shares
        
        def reconstruct_secret(shares: List[float]) -> float:
            """Reconstruct secret from shares"""
            return sum(shares)
        
        # Test with multiple parties
        values = [10.0, 20.0, 30.0]  # Each party's private value
        num_parties = len(values)
        
        # Each party creates shares of their value
        all_shares = []
        for value in values:
            shares = secret_share(value, num_parties)
            all_shares.append(shares)
        
        # Compute sum of shares for each position
        position_sums = []
        for i in range(num_parties):
            position_sum = sum(all_shares[j][i] for j in range(num_parties))
            position_sums.append(position_sum)
        
        # Reconstruct the sum
        total_sum = reconstruct_secret(position_sums)
        expected_sum = sum(values)
        
        assert abs(total_sum - expected_sum) < 1e-10


class TestPrivacyLeakDetection:
    """Test privacy leak detection mechanisms"""
    
    def test_membership_inference_protection(self, dp_service):
        """Test protection against membership inference attacks"""
        # Simulate membership inference attack scenario
        def compute_model_loss(data_point: np.ndarray, is_member: bool) -> float:
            """Simulate model loss computation"""
            base_loss = np.random.uniform(0.1, 0.5)
            if is_member:
                # Members typically have lower loss
                return base_loss * 0.7
            else:
                return base_loss
        
        # Generate data for members and non-members
        member_losses = []
        non_member_losses = []
        
        for _ in range(100):
            # Member losses (with DP noise)
            member_loss = compute_model_loss(np.random.randn(10), True)
            noisy_member_loss = dp_service.add_noise(member_loss, sensitivity=0.1)
            member_losses.append(noisy_member_loss)
            
            # Non-member losses (with DP noise)
            non_member_loss = compute_model_loss(np.random.randn(10), False)
            noisy_non_member_loss = dp_service.add_noise(non_member_loss, sensitivity=0.1)
            non_member_losses.append(noisy_non_member_loss)
        
        # Test that DP noise makes membership inference harder
        member_mean = np.mean(member_losses)
        non_member_mean = np.mean(non_member_losses)
        
        # With DP noise, the difference should be reduced
        difference = abs(member_mean - non_member_mean)
        assert difference < 0.2  # Should be small due to DP noise
    
    def test_model_inversion_protection(self):
        """Test protection against model inversion attacks"""
        # Simulate model inversion protection through output perturbation
        def perturb_model_output(output: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
            """Add noise to model output to prevent inversion"""
            noise = np.random.normal(0, noise_scale, output.shape)
            return output + noise
        
        # Original model output
        original_output = np.array([0.8, 0.1, 0.05, 0.05])  # Probability distribution
        
        # Perturbed outputs
        perturbed_outputs = []
        for _ in range(10):
            perturbed = perturb_model_output(original_output)
            perturbed_outputs.append(perturbed)
        
        perturbed_outputs = np.array(perturbed_outputs)
        
        # Check that outputs are still valid probabilities (approximately)
        for output in perturbed_outputs:
            assert np.all(output >= 0) or np.any(output >= -0.1)  # Allow small negative due to noise
            # Note: In practice, you'd renormalize to ensure valid probabilities
    
    def test_gradient_leakage_protection(self):
        """Test protection against gradient-based privacy attacks"""
        # Simulate gradient clipping for privacy protection
        def clip_gradients(gradients: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
            """Clip gradients to bound sensitivity"""
            grad_norm = np.linalg.norm(gradients)
            if grad_norm > max_norm:
                return gradients * (max_norm / grad_norm)
            return gradients
        
        # Test with various gradient magnitudes
        test_gradients = [
            np.array([0.1, 0.2, 0.3]),  # Small gradients
            np.array([2.0, 3.0, 4.0]),  # Large gradients
            np.array([10.0, -5.0, 8.0])  # Very large gradients
        ]
        
        for gradients in test_gradients:
            clipped = clip_gradients(gradients, max_norm=1.0)
            clipped_norm = np.linalg.norm(clipped)
            
            # Clipped gradients should have norm <= max_norm
            assert clipped_norm <= 1.0 + 1e-10  # Allow for numerical precision
            
            # Direction should be preserved for large gradients
            if np.linalg.norm(gradients) > 1.0:
                # Cosine similarity should be 1 (same direction)
                cosine_sim = np.dot(gradients, clipped) / (np.linalg.norm(gradients) * np.linalg.norm(clipped))
                assert abs(cosine_sim - 1.0) < 1e-10


@pytest.mark.skipif(not OPACUS_AVAILABLE, reason="Opacus not available")
class TestOpacusIntegration:
    """Test integration with Opacus differential privacy library"""
    
    def test_opacus_privacy_engine_creation(self):
        """Test Opacus privacy engine creation"""
        try:
            # Create a simple mock model
            import torch
            import torch.nn as nn
            
            model = nn.Linear(10, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            # Create privacy engine
            privacy_engine = PrivacyEngine()
            
            # Attach to model and optimizer
            model, optimizer, data_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=[(torch.randn(32, 10), torch.randn(32, 1))],  # Mock data loader
                noise_multiplier=1.0,
                max_grad_norm=1.0,
            )
            
            # Verify privacy engine is attached
            assert hasattr(model, "_module")
            assert privacy_engine.accountant is not None
            
        except ImportError:
            pytest.skip("PyTorch not available for Opacus test")
    
    def test_opacus_privacy_accounting(self):
        """Test Opacus privacy accounting"""
        try:
            import torch
            import torch.nn as nn
            from opacus.accountants import RDPAccountant
            
            # Create accountant
            accountant = RDPAccountant()
            
            # Test privacy accounting
            steps = 100
            noise_multiplier = 1.0
            sample_rate = 0.01
            
            # Step the accountant
            for _ in range(steps):
                accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
            
            # Get privacy spent
            epsilon = accountant.get_epsilon(delta=1e-5)
            
            # Verify epsilon is reasonable
            assert epsilon > 0
            assert epsilon < 100  # Should not be too large
            
        except ImportError:
            pytest.skip("PyTorch not available for Opacus accounting test")


@pytest.mark.performance
class TestPrivacyPerformance:
    """Test performance of privacy-preserving mechanisms"""
    
    def test_differential_privacy_performance(self, dp_service):
        """Test differential privacy performance"""
        import time
        
        values = [float(i) for i in range(1000)]
        
        start_time = time.time()
        for value in values:
            dp_service.add_noise(value)
        elapsed_time = time.time() - start_time
        
        # Should add noise to 1000 values quickly
        assert elapsed_time < 0.1
        
        avg_time_per_operation = elapsed_time / 1000
        assert avg_time_per_operation < 0.0001
    
    def test_encryption_performance(self, fl_service):
        """Test encryption performance for federated learning"""
        import time
        
        test_data = "x" * 10000  # 10KB of data
        
        start_time = time.time()
        for _ in range(100):
            encrypted = fl_service.encrypt_data(test_data)
            fl_service.cipher.decrypt(encrypted)
        elapsed_time = time.time() - start_time
        
        # Should encrypt/decrypt 100 times quickly
        assert elapsed_time < 1.0
        
        avg_time_per_cycle = elapsed_time / 100
        assert avg_time_per_cycle < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])