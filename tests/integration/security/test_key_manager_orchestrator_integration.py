"""
Key Manager Orchestrator Integration Tests

Tests integration of SecureKeyManager and FernetKeyManager components with the 
ML orchestrator system, ensuring they work correctly as Tier 6 security components
in the ML pipeline. Follows 2025 best practices with real behavior testing.

Integration Test Coverage:
- Component registration and discovery through orchestrator
- Real orchestrator workflow execution with key management operations
- Integration with actual ML pipeline components
- Security component tier validation (Tier 6)
- Cross-component communication and data flow
- Error handling and recovery in orchestrated scenarios
- Performance validation in integrated environment
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Import real orchestrator components for authentic integration testing
from src.prompt_improver.ml.orchestration.core.component_registry import ComponentRegistry, ComponentTier
from src.prompt_improver.ml.orchestration.config.component_definitions import ComponentDefinitions
from src.prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from src.prompt_improver.security.key_manager import SecureKeyManager, FernetKeyManager


class TestKeyManagerComponentRegistration:
    """Test key manager component registration in orchestrator"""

    @pytest.fixture
    def temp_key_dir(self):
        """Create temporary directory for key storage"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def component_registry(self):
        """Create component registry for testing"""
        config = OrchestratorConfig(
            component_health_check_interval=60,
            max_concurrent_workflows=10,
            debug_mode=True  # Enable debug for testing
        )
        return ComponentRegistry(config)

    def test_security_components_are_defined(self):
        """Test that security components are properly defined"""
        component_definitions = ComponentDefinitions()
        security_components = component_definitions.get_tier_components(ComponentTier.TIER_6_SECURITY)
        
        # Check specific components
        assert "secure_key_manager" in security_components
        assert "fernet_key_manager" in security_components
        
        # Validate component definitions
        secure_key_def = security_components["secure_key_manager"]
        assert "key_management" in secure_key_def["capabilities"]
        assert "key_rotation" in secure_key_def["capabilities"]
        assert "orchestrator_compatible" in secure_key_def["capabilities"]
        
        fernet_key_def = security_components["fernet_key_manager"]
        assert "fernet_encryption" in fernet_key_def["capabilities"]
        assert "data_encryption" in fernet_key_def["capabilities"]
        assert "orchestrator_compatible" in fernet_key_def["capabilities"]

    async def test_component_registry_loads_security_components(self, component_registry):
        """Test that component registry loads security components"""
        # Initialize registry (this loads all component definitions)
        await component_registry._load_component_definitions()
        
        # Check that security components are registered
        registered_components = await component_registry.list_components()
        
        # Should include security components
        component_names = [comp.name for comp in registered_components]
        assert "secure_key_manager" in component_names
        assert "fernet_key_manager" in component_names
        
        # Check component tiers
        security_components = [
            comp for comp in registered_components
            if comp.tier == ComponentTier.TIER_6_SECURITY
        ]
        assert len(security_components) >= 2

    async def test_component_capabilities_registration(self, component_registry):
        """Test that component capabilities are properly registered"""
        await component_registry._load_component_definitions()
        
        # Get security components
        secure_key_comp = await component_registry.get_component("secure_key_manager")
        fernet_comp = await component_registry.get_component("fernet_key_manager")
        
        assert secure_key_comp is not None
        assert fernet_comp is not None
        
        # Check capabilities
        capability_names_secure = [cap.name for cap in secure_key_comp.capabilities]
        capability_names_fernet = [cap.name for cap in fernet_comp.capabilities]
        
        assert any("key_management" in name for name in capability_names_secure)
        assert any("key_rotation" in name for name in capability_names_secure)
        assert any("fernet_encryption" in name for name in capability_names_fernet)
        assert any("data_encryption" in name for name in capability_names_fernet)

    async def test_component_dependencies_validation(self, component_registry):
        """Test that component dependencies are properly validated"""
        await component_registry._load_component_definitions()
        
        # Get components
        secure_key_comp = await component_registry.get_component("secure_key_manager")
        fernet_comp = await component_registry.get_component("fernet_key_manager")
        
        # FernetKeyManager should depend on SecureKeyManager
        fernet_deps = getattr(fernet_comp, 'dependencies', [])
        assert "secure_key_manager" in fernet_deps or "SecureKeyManager" in str(fernet_deps)


class TestOrchestoratedKeyOperations:
    """Test orchestrated key management operations"""

    @pytest.fixture
    def temp_key_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def key_managers(self, temp_key_dir):
        """Create key managers for testing"""
        # SecureKeyManager uses in-memory storage, no key_directory parameter
        secure_manager = SecureKeyManager()
        return {
            "secure": secure_manager,
            "fernet": FernetKeyManager(key_manager=secure_manager)
        }

    def test_orchestrated_key_lifecycle(self, key_managers):
        """Test complete key lifecycle through orchestrator interface"""
        secure_mgr = key_managers["secure"]
        
        key_id = "lifecycle_test_key"
        
        # Step 1: Generate key through orchestrator
        gen_result = secure_mgr.run_orchestrated_analysis(
            operation="get_key",
            parameters={"key_id": key_id}
        )
        
        assert gen_result["orchestrator_compatible"] is True
        assert gen_result["operation"] == "get_key"
        assert "timestamp" in gen_result
        
        # If key doesn't exist, create it
        if gen_result["result"] is None:
            create_result = secure_mgr.run_orchestrated_analysis(
                operation="generate_key",
                parameters={"key_id": key_id}
            )
            assert create_result["orchestrator_compatible"] is True
            assert create_result["result"] is not None
            generated_key = create_result["result"]
        else:
            generated_key = gen_result["result"]
        
        # Step 2: Rotate key through orchestrator
        rotate_result = secure_mgr.run_orchestrated_analysis(
            operation="rotate_key",
            parameters={"key_id": key_id}
        )
        
        assert rotate_result["orchestrator_compatible"] is True
        assert rotate_result["result"] != generated_key
        
        # Step 3: Get status through orchestrator
        status_result = secure_mgr.run_orchestrated_analysis(
            operation="get_status",
            parameters={}
        )
        
        assert status_result["orchestrator_compatible"] is True
        assert isinstance(status_result["result"], dict)
        
        # Step 4: Cleanup through orchestrator
        cleanup_result = secure_mgr.run_orchestrated_analysis(
            operation="cleanup",
            parameters={"key_id": key_id}
        )
        
        assert cleanup_result["orchestrator_compatible"] is True

    def test_orchestrated_encryption_workflow(self, key_managers):
        """Test complete encryption workflow through orchestrator"""
        fernet_mgr = key_managers["fernet"]
        
        test_data = "Orchestrated encryption test data"
        key_id = "orchestration_encryption_key"
        
        # Step 1: Encrypt through orchestrator
        encrypt_result = fernet_mgr.run_orchestrated_analysis(
            operation="encrypt",
            parameters={"data": test_data, "key_id": key_id}
        )
        
        assert encrypt_result["orchestrator_compatible"] is True
        assert "result" in encrypt_result
        assert "encrypted_data" in encrypt_result["result"]
        assert "key_id" in encrypt_result["result"]
        
        encrypted_data = encrypt_result["result"]["encrypted_data"]
        used_key_id = encrypt_result["result"]["key_id"]
        
        # Step 2: Decrypt through orchestrator
        decrypt_result = fernet_mgr.run_orchestrated_analysis(
            operation="decrypt",
            parameters={"encrypted_data": encrypted_data, "key_id": used_key_id}
        )
        
        assert decrypt_result["orchestrator_compatible"] is True
        assert decrypt_result["result"]["success"] is True
        assert decrypt_result["result"]["decrypted_data"] == test_data
        
        # Step 3: Test encryption cycle through orchestrator
        test_result = fernet_mgr.run_orchestrated_analysis(
            operation="test_encryption",
            parameters={}
        )
        
        assert test_result["orchestrator_compatible"] is True
        assert test_result["result"]["success"] is True

    def test_orchestrated_error_handling(self, key_managers):
        """Test error handling in orchestrated operations"""
        secure_mgr = key_managers["secure"]
        fernet_mgr = key_managers["fernet"]
        
        # Test invalid operation
        invalid_result = secure_mgr.run_orchestrated_analysis(
            operation="invalid_operation",
            parameters={}
        )
        
        assert invalid_result["orchestrator_compatible"] is True
        assert invalid_result["result"] is None
        assert "error" in invalid_result
        
        # Test missing parameters
        missing_params_result = fernet_mgr.run_orchestrated_analysis(
            operation="encrypt",
            parameters={}  # Missing required 'data' parameter
        )
        
        assert missing_params_result["orchestrator_compatible"] is True
        # Should handle gracefully with error information
        
        # Test invalid encrypted data
        invalid_decrypt_result = fernet_mgr.run_orchestrated_analysis(
            operation="decrypt",
            parameters={"encrypted_data": b"invalid_data", "key_id": "test_key"}
        )
        
        assert invalid_decrypt_result["orchestrator_compatible"] is True
        if "result" in invalid_decrypt_result and invalid_decrypt_result["result"]:
            assert invalid_decrypt_result["result"]["success"] is False

    def test_orchestrated_operations_metadata(self, key_managers):
        """Test metadata in orchestrated operations"""
        secure_mgr = key_managers["secure"]
        
        result = secure_mgr.run_orchestrated_analysis(
            operation="get_status",
            parameters={}
        )
        
        # Check required metadata fields
        assert result["orchestrator_compatible"] is True
        assert "operation" in result
        assert "timestamp" in result
        assert "result" in result
        
        # Check timestamp format (should be ISO format)
        timestamp = result["timestamp"]
        assert "T" in timestamp  # ISO format includes 'T'
        assert timestamp.endswith("Z") or "+" in timestamp  # UTC indicator


class TestMLPipelineIntegration:
    """Test integration with ML pipeline components"""

    @pytest.fixture
    def temp_key_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def integrated_setup(self, temp_key_dir):
        """Set up integrated ML pipeline with key management"""
        config = OrchestratorConfig(debug_mode=True)
        secure_manager = SecureKeyManager()
        return {
            "key_manager": secure_manager,
            "fernet_manager": FernetKeyManager(key_manager=secure_manager),
            "component_registry": ComponentRegistry(config)
        }

    def test_ml_data_encryption_pipeline(self, integrated_setup):
        """Test ML data encryption in integrated pipeline"""
        fernet_mgr = integrated_setup["fernet_manager"]
        
        # Simulate ML training data
        ml_training_data = {
            "features": np.random.rand(100, 10).tolist(),
            "labels": np.random.randint(0, 2, 100).tolist(),
            "metadata": {
                "dataset_id": "training_set_001",
                "preprocessing_steps": ["normalization", "feature_scaling"],
                "sensitive_info": "This dataset contains PII"
            }
        }
        
        # Convert to JSON for encryption
        data_json = json.dumps(ml_training_data)
        
        # Encrypt through orchestrator interface
        encrypt_result = fernet_mgr.run_orchestrated_analysis(
            operation="encrypt",
            parameters={"data": data_json, "key_id": "ml_training_data_key"}
        )
        
        assert encrypt_result["orchestrator_compatible"] is True
        encrypted_data = encrypt_result["result"]["encrypted_data"]
        key_id = encrypt_result["result"]["key_id"]
        
        # Later in pipeline: decrypt data for processing
        decrypt_result = fernet_mgr.run_orchestrated_analysis(
            operation="decrypt",
            parameters={"encrypted_data": encrypted_data, "key_id": key_id}
        )
        
        assert decrypt_result["orchestrator_compatible"] is True
        assert decrypt_result["result"]["success"] is True
        
        # Verify data integrity
        decrypted_json = decrypt_result["result"]["decrypted_data"]
        restored_data = json.loads(decrypted_json)
        
        assert restored_data["metadata"]["dataset_id"] == ml_training_data["metadata"]["dataset_id"]
        assert len(restored_data["features"]) == len(ml_training_data["features"])
        assert restored_data["metadata"]["sensitive_info"] == ml_training_data["metadata"]["sensitive_info"]

    def test_model_weights_encryption(self, integrated_setup):
        """Test encryption of ML model weights"""
        fernet_mgr = integrated_setup["fernet_manager"]
        
        # Simulate trained model weights
        model_weights = {
            "layer_1": np.random.rand(784, 128).tolist(),
            "layer_1_bias": np.random.rand(128).tolist(),
            "layer_2": np.random.rand(128, 10).tolist(),
            "layer_2_bias": np.random.rand(10).tolist(),
            "model_metadata": {
                "architecture": "dense_neural_network",
                "accuracy": 0.95,
                "training_epochs": 100,
                "optimizer": "adam"
            }
        }
        
        weights_json = json.dumps(model_weights)
        
        # Encrypt model weights
        encrypt_result = fernet_mgr.run_orchestrated_analysis(
            operation="encrypt",
            parameters={"data": weights_json, "key_id": "model_weights_key"}
        )
        
        assert encrypt_result["orchestrator_compatible"] is True
        
        # Simulate model deployment: decrypt weights
        decrypt_result = fernet_mgr.run_orchestrated_analysis(
            operation="decrypt",
            parameters={
                "encrypted_data": encrypt_result["result"]["encrypted_data"],
                "key_id": encrypt_result["result"]["key_id"]
            }
        )
        
        assert decrypt_result["result"]["success"] is True
        
        # Verify model integrity
        restored_weights = json.loads(decrypt_result["result"]["decrypted_data"])
        assert restored_weights["model_metadata"]["accuracy"] == 0.95
        assert len(restored_weights["layer_1"]) == len(model_weights["layer_1"])

    def test_secure_model_versioning(self, integrated_setup):
        """Test secure model versioning with key rotation"""
        key_mgr = integrated_setup["key_manager"]
        fernet_mgr = integrated_setup["fernet_manager"]
        
        model_id = "secure_model_v1"
        
        # Version 1: Initial model
        v1_data = {"version": "1.0", "weights": [1, 2, 3, 4, 5]}
        v1_json = json.dumps(v1_data)
        
        v1_encrypted = fernet_mgr.run_orchestrated_analysis(
            operation="encrypt",
            parameters={"data": v1_json, "key_id": f"{model_id}_key"}
        )
        
        # Version 2: Updated model with key rotation
        rotate_result = key_mgr.run_orchestrated_analysis(
            operation="rotate_key",
            parameters={"key_id": f"{model_id}_key"}
        )
        
        v2_data = {"version": "2.0", "weights": [6, 7, 8, 9, 10]}
        v2_json = json.dumps(v2_data)
        
        # Use new encryption without specifying old key_id (let system use current key)
        v2_encrypted = fernet_mgr.run_orchestrated_analysis(
            operation="encrypt",
            parameters={"data": v2_json}  # Don't specify key_id, let it use current
        )
        
        # Both operations should succeed
        assert v1_encrypted["orchestrator_compatible"] is True
        assert v2_encrypted["orchestrator_compatible"] is True
        
        # Version 2 should decrypt correctly with the key used for encryption
        v2_decrypted = fernet_mgr.run_orchestrated_analysis(
            operation="decrypt",
            parameters={
                "encrypted_data": v2_encrypted["result"]["encrypted_data"],
                "key_id": v2_encrypted["result"]["key_id"]  # Use actual key that was used
            }
        )
        
        assert v2_decrypted["result"]["success"] is True
        restored_v2 = json.loads(v2_decrypted["result"]["decrypted_data"])
        assert restored_v2["version"] == "2.0"

    def test_cross_component_security_validation(self, integrated_setup):
        """Test security validation across multiple components"""
        registry = integrated_setup["component_registry"]
        fernet_mgr = integrated_setup["fernet_manager"]
        
        # Simulate cross-component data flow with encryption
        sensitive_data = "Cross-component sensitive information"
        
        # Component A encrypts data
        component_a_result = fernet_mgr.run_orchestrated_analysis(
            operation="encrypt",
            parameters={"data": sensitive_data, "key_id": "cross_component_key"}
        )
        
        encrypted_payload = component_a_result["result"]
        
        # Simulate passing encrypted data through orchestrator
        # Component B receives and decrypts data
        component_b_result = fernet_mgr.run_orchestrated_analysis(
            operation="decrypt",
            parameters={
                "encrypted_data": encrypted_payload["encrypted_data"],
                "key_id": encrypted_payload["key_id"]
            }
        )
        
        # Verify secure data transfer
        assert component_b_result["result"]["success"] is True
        assert component_b_result["result"]["decrypted_data"] == sensitive_data
        
        # Test security validation
        test_result = fernet_mgr.run_orchestrated_analysis(
            operation="test_encryption",
            parameters={}
        )
        
        assert test_result["result"]["success"] is True
        assert test_result["result"]["performance_ms"] < 1000  # Should be fast


class TestSecurityComplianceIntegration:
    """Test security compliance and audit in integrated environment"""

    @pytest.fixture
    def temp_key_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def audit_setup(self, temp_key_dir):
        """Set up components with audit capabilities"""
        secure_manager = SecureKeyManager()
        return {
            "key_manager": secure_manager,
            "fernet_manager": FernetKeyManager(key_manager=secure_manager),
            "operations_log": []
        }

    def test_security_operations_audit_trail(self, audit_setup):
        """Test audit trail for security operations"""
        key_mgr = audit_setup["key_manager"]
        fernet_mgr = audit_setup["fernet_manager"]
        ops_log = audit_setup["operations_log"]
        
        # Perform multiple security operations
        operations = [
            ("generate_key", {"key_id": "audit_test_key"}),
            ("encrypt", {"data": "audit test data", "key_id": "audit_test_key"}),
            ("get_status", {}),
            ("test_encryption", {}),
            ("rotate_key", {"key_id": "audit_test_key"}),
        ]
        
        for i, (operation, params) in enumerate(operations[:2]):
            # Use key manager for key operations
            result = key_mgr.run_orchestrated_analysis(operation=operation, parameters=params)
            ops_log.append({
                "step": i + 1,
                "operation": operation,
                "timestamp": result.get("timestamp"),
                "success": result.get("result") is not None,
                "orchestrator_compatible": result.get("orchestrator_compatible")
            })
        
        for i, (operation, params) in enumerate(operations[1:], start=2):
            # Use fernet manager for encryption operations
            if operation in ["encrypt", "test_encryption"]:
                result = fernet_mgr.run_orchestrated_analysis(operation=operation, parameters=params)
                ops_log.append({
                    "step": i + 1,
                    "operation": operation,
                    "timestamp": result.get("timestamp"),
                    "success": result.get("result") is not None,
                    "orchestrator_compatible": result.get("orchestrator_compatible")
                })
        
        # Verify audit trail
        assert len(ops_log) >= 2
        
        for log_entry in ops_log:
            assert "timestamp" in log_entry
            assert log_entry["orchestrator_compatible"] is True
            assert isinstance(log_entry["success"], bool)

    def test_security_compliance_validation(self, audit_setup):
        """Test security compliance validation"""
        fernet_mgr = audit_setup["fernet_manager"]
        
        # Test encryption strength validation
        test_data = "Security compliance test data"
        
        encrypt_result = fernet_mgr.run_orchestrated_analysis(
            operation="encrypt",
            parameters={"data": test_data, "key_id": "compliance_key"}
        )
        
        # Validate encryption result
        assert encrypt_result["orchestrator_compatible"] is True
        encrypted_data = encrypt_result["result"]["encrypted_data"]
        
        # Encrypted data should be significantly different from original
        assert encrypted_data != test_data.encode()
        assert len(encrypted_data) > len(test_data)  # Encryption adds overhead
        
        # Test system validation
        validation_result = fernet_mgr.run_orchestrated_analysis(
            operation="test_encryption",
            parameters={}
        )
        
        assert validation_result["result"]["success"] is True
        
        # Performance should meet compliance standards (< 100ms for test data)
        assert validation_result["result"]["performance_ms"] < 100

    def test_key_rotation_compliance(self, audit_setup):
        """Test key rotation compliance requirements"""
        key_mgr = audit_setup["key_manager"]
        
        compliance_key_id = "compliance_rotation_key"
        
        # Initial key generation
        gen_result = key_mgr.run_orchestrated_analysis(
            operation="get_key",
            parameters={"key_id": compliance_key_id}
        )
        
        if gen_result["result"] is None:
            # Generate initial key
            initial_result = key_mgr.run_orchestrated_analysis(
                operation="generate_key",
                parameters={"key_id": compliance_key_id}
            )
            original_key = initial_result["result"]
        else:
            original_key = gen_result["result"]
        
        # Perform rotation (compliance requirement)
        rotation_result = key_mgr.run_orchestrated_analysis(
            operation="rotate_key",
            parameters={"key_id": compliance_key_id}
        )
        
        assert rotation_result["orchestrator_compatible"] is True
        rotated_key = rotation_result["result"]
        
        # Verify compliance requirements
        assert rotated_key != original_key  # Key must change
        assert isinstance(rotated_key, bytes)  # Must be proper key format
        assert len(rotated_key) == 44  # Fernet key length requirement
        
        # Verify rotation was logged with timestamp
        assert "timestamp" in rotation_result
        
        # Test cleanup compliance
        cleanup_result = key_mgr.run_orchestrated_analysis(
            operation="cleanup",
            parameters={"key_id": compliance_key_id}
        )
        
        assert cleanup_result["orchestrator_compatible"] is True


@pytest.mark.performance
class TestIntegratedPerformance:
    """Test performance in integrated orchestrator environment"""

    @pytest.fixture
    def temp_key_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def perf_setup(self, temp_key_dir):
        """Set up performance testing environment"""
        config = OrchestratorConfig(debug_mode=True)
        secure_manager = SecureKeyManager()
        return {
            "key_manager": secure_manager,
            "fernet_manager": FernetKeyManager(key_manager=secure_manager),
            "registry": ComponentRegistry(config)
        }

    def test_orchestrated_operations_performance(self, perf_setup):
        """Test performance of orchestrated operations"""
        fernet_mgr = perf_setup["fernet_manager"]
        
        test_data = "Performance test data" * 100  # ~2KB of data
        
        # Measure encryption performance
        start_time = time.time()
        
        for i in range(20):
            key_id = f"perf_key_{i}"
            encrypt_result = fernet_mgr.run_orchestrated_analysis(
                operation="encrypt",
                parameters={"data": f"{test_data}_{i}", "key_id": key_id}
            )
            assert encrypt_result["orchestrator_compatible"] is True
        
        encrypt_time = time.time() - start_time
        
        # Should complete 20 encryptions quickly (< 2 seconds)
        assert encrypt_time < 2.0
        
        avg_encrypt_time = encrypt_time / 20
        assert avg_encrypt_time < 0.1  # < 100ms per encryption

    def test_concurrent_orchestrated_operations(self, perf_setup):
        """Test concurrent orchestrated operations performance"""
        import threading
        
        fernet_mgr = perf_setup["fernet_manager"]
        results = []
        errors = []
        
        def concurrent_operations(thread_id):
            try:
                for i in range(5):
                    key_id = f"concurrent_{thread_id}_{i}"
                    data = f"Thread {thread_id} data {i}"
                    
                    # Encrypt
                    encrypt_result = fernet_mgr.run_orchestrated_analysis(
                        operation="encrypt",
                        parameters={"data": data, "key_id": key_id}
                    )
                    
                    # Decrypt
                    decrypt_result = fernet_mgr.run_orchestrated_analysis(
                        operation="decrypt",
                        parameters={
                            "encrypted_data": encrypt_result["result"]["encrypted_data"],
                            "key_id": encrypt_result["result"]["key_id"]
                        }
                    )
                    
                    assert decrypt_result["result"]["success"] is True
                    assert decrypt_result["result"]["decrypted_data"] == data
                    results.append((thread_id, i))
                    
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run concurrent operations
        threads = []
        start_time = time.time()
        
        for thread_id in range(4):
            thread = threading.Thread(target=concurrent_operations, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        elapsed_time = time.time() - start_time
        
        # Verify results
        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 20  # 4 threads × 5 operations
        
        # Should complete concurrent operations efficiently (< 3 seconds)
        assert elapsed_time < 3.0

    def test_large_data_orchestrated_encryption(self, perf_setup):
        """Test orchestrated encryption with large data sets"""
        fernet_mgr = perf_setup["fernet_manager"]
        
        # Test with larger data sizes
        data_sizes = [1024, 10240, 51200]  # 1KB, 10KB, 50KB
        
        for size in data_sizes:
            large_data = "X" * size
            
            start_time = time.time()
            
            encrypt_result = fernet_mgr.run_orchestrated_analysis(
                operation="encrypt",
                parameters={"data": large_data, "key_id": f"large_data_{size}"}
            )
            
            encrypt_time = time.time() - start_time
            
            assert encrypt_result["orchestrator_compatible"] is True
            
            # Performance should scale reasonably
            max_time = 0.5 + (size / 100000)  # Allow scaling
            assert encrypt_time < max_time, f"Too slow for {size}B: {encrypt_time:.3f}s"
            
            # Verify decryption works
            decrypt_result = fernet_mgr.run_orchestrated_analysis(
                operation="decrypt",
                parameters={
                    "encrypted_data": encrypt_result["result"]["encrypted_data"],
                    "key_id": encrypt_result["result"]["key_id"]
                }
            )
            
            assert decrypt_result["result"]["success"] is True
            assert len(decrypt_result["result"]["decrypted_data"]) == size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])