"""Real Behavior Tests for Facade Implementations

This test suite validates that all facade implementations work correctly
with real service integrations, ensuring no functionality regression
during the coupling reduction refactoring.

Key test areas:
- Facade initialization and lifecycle management
- Service resolution through facades  
- Protocol compliance validation
- Performance impact measurement
- Integration with existing services
- Error handling and recovery
"""

import asyncio
import pytest
import time
from typing import Any, Dict

from prompt_improver.core.facades import (
    get_di_facade,
    get_config_facade,
    get_repository_facade,
    get_cli_facade,
    get_api_facade,
    get_performance_facade,
)
from prompt_improver.core.protocols.facade_protocols import (
    DIFacadeProtocol,
    ConfigFacadeProtocol,
    RepositoryFacadeProtocol,
    CLIFacadeProtocol,
    APIFacadeProtocol,
    PerformanceFacadeProtocol,
)


@pytest.mark.asyncio
class TestFacadeBehavior:
    """Real behavior tests for facade implementations."""

    async def test_di_facade_protocol_compliance(self):
        """Test DI facade implements protocol correctly."""
        facade = get_di_facade()
        assert isinstance(facade, DIFacadeProtocol)
        
        # Test facade methods exist and are callable
        assert hasattr(facade, "get_service")
        assert hasattr(facade, "get_core_service")  
        assert hasattr(facade, "get_ml_service")
        assert hasattr(facade, "initialize")
        assert hasattr(facade, "shutdown")
        assert hasattr(facade, "health_check")
        
        # Test basic functionality
        await facade.initialize()
        health = await facade.health_check()
        assert isinstance(health, dict)
        await facade.shutdown()

    async def test_config_facade_protocol_compliance(self):
        """Test config facade implements protocol correctly."""
        facade = get_config_facade()
        assert isinstance(facade, ConfigFacadeProtocol)
        
        # Test facade methods exist and are callable
        assert hasattr(facade, "get_config")
        assert hasattr(facade, "get_database_config")
        assert hasattr(facade, "get_security_config")
        assert hasattr(facade, "get_monitoring_config")
        assert hasattr(facade, "get_ml_config")
        assert hasattr(facade, "validate_configuration")
        assert hasattr(facade, "reload_config")
        
        # Test basic functionality
        config = facade.get_config()
        assert config is not None
        
        # Test environment detection
        env_name = facade.get_environment_name()
        assert isinstance(env_name, str)
        assert env_name in ["development", "testing", "production"]

    async def test_repository_facade_protocol_compliance(self):
        """Test repository facade implements protocol correctly."""
        facade = get_repository_facade()
        assert isinstance(facade, RepositoryFacadeProtocol)
        
        # Test facade methods exist and are callable
        assert hasattr(facade, "get_analytics_repository")
        assert hasattr(facade, "get_ml_repository")
        assert hasattr(facade, "get_repository")
        assert hasattr(facade, "health_check")
        assert hasattr(facade, "cleanup")
        
        # Test available repositories listing
        available_repos = facade.get_available_repositories()
        assert isinstance(available_repos, list)
        assert len(available_repos) > 0

    async def test_cli_facade_protocol_compliance(self):
        """Test CLI facade implements protocol correctly."""
        facade = get_cli_facade()
        assert isinstance(facade, CLIFacadeProtocol)
        
        # Test facade methods exist and are callable
        assert hasattr(facade, "get_orchestrator")
        assert hasattr(facade, "get_signal_handler")
        assert hasattr(facade, "initialize_all")
        assert hasattr(facade, "shutdown_all")
        assert hasattr(facade, "create_emergency_checkpoint")
        
        # Test component status
        status = facade.get_component_status()
        assert isinstance(status, dict)
        assert "initialized" in status

    async def test_api_facade_protocol_compliance(self):
        """Test API facade implements protocol correctly."""
        facade = get_api_facade()
        assert isinstance(facade, APIFacadeProtocol)
        
        # Test facade methods exist and are callable
        assert hasattr(facade, "get_config")
        assert hasattr(facade, "get_container")
        assert hasattr(facade, "get_database_services")
        assert hasattr(facade, "initialize_all_services")
        assert hasattr(facade, "shutdown_all_services")
        
        # Test service status
        status = facade.get_service_status()
        assert isinstance(status, dict)
        assert "initialized" in status

    async def test_performance_facade_protocol_compliance(self):
        """Test performance facade implements protocol correctly."""
        facade = get_performance_facade()
        assert isinstance(facade, PerformanceFacadeProtocol)
        
        # Test facade methods exist and are callable
        assert hasattr(facade, "get_baseline_system")
        assert hasattr(facade, "get_profiler")
        assert hasattr(facade, "record_operation")
        assert hasattr(facade, "analyze_trends")
        assert hasattr(facade, "check_regressions")
        assert hasattr(facade, "generate_report")
        assert hasattr(facade, "initialize_system")
        assert hasattr(facade, "shutdown_system")
        
        # Test system status
        status = facade.get_system_status()
        assert isinstance(status, dict)
        assert "initialized" in status

    async def test_facade_initialization_performance(self):
        """Test facade initialization has minimal performance impact."""
        start_time = time.time()
        
        # Initialize all facades
        facades = [
            get_di_facade(),
            get_config_facade(), 
            get_repository_facade(),
            get_cli_facade(),
            get_api_facade(),
            get_performance_facade(),
        ]
        
        initialization_time = time.time() - start_time
        
        # Facade creation should be very fast (< 10ms)
        assert initialization_time < 0.01, f"Facade initialization took {initialization_time:.3f}s"
        
        # All facades should be created
        assert len(facades) == 6
        for facade in facades:
            assert facade is not None

    async def test_facade_lazy_loading_behavior(self):
        """Test facades only load dependencies when actually used."""
        facade = get_config_facade()
        
        # Facade creation should not load heavy dependencies
        start_time = time.time()
        facade = get_config_facade()
        creation_time = time.time() - start_time
        assert creation_time < 0.001, "Facade creation should be instantaneous"
        
        # First actual usage should be slightly slower due to lazy loading
        start_time = time.time()
        config = facade.get_config()
        first_usage_time = time.time() - start_time
        
        # Second usage should be much faster due to caching
        start_time = time.time()
        config2 = facade.get_config()
        second_usage_time = time.time() - start_time
        
        assert config is not None
        assert config2 is not None
        assert second_usage_time < first_usage_time, "Caching should improve performance"

    async def test_facade_error_handling(self):
        """Test facades handle errors gracefully."""
        facade = get_config_facade()
        
        # Test with invalid configuration access
        try:
            # This should not raise an exception
            result = facade.validate_configuration()
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"Facade error handling failed: {e}")

    async def test_facade_lifecycle_management(self):
        """Test facade lifecycle management works correctly."""
        di_facade = get_di_facade()
        
        # Test initialization
        await di_facade.initialize()
        health = await di_facade.health_check()
        assert health is not None
        
        # Test shutdown
        await di_facade.shutdown()
        
        # Test re-initialization
        await di_facade.initialize()
        health2 = await di_facade.health_check()
        assert health2 is not None
        
        await di_facade.shutdown()

    async def test_facade_integration_with_existing_services(self):
        """Test facades integrate correctly with existing services."""
        config_facade = get_config_facade()
        
        # Test config facade provides same results as direct imports
        facade_config = config_facade.get_config()
        
        # Import directly for comparison (only in test)
        from prompt_improver.core.config import get_config as direct_get_config
        direct_config = direct_get_config()
        
        # Should have same environment
        assert facade_config.environment.environment == direct_config.environment.environment

    def test_facade_coupling_reduction_verification(self):
        """Verify facades actually reduce coupling as claimed."""
        import ast
        import inspect
        from pathlib import Path
        
        # Check facade files have minimal imports
        facade_files = [
            "src/prompt_improver/core/facades/di_facade.py",
            "src/prompt_improver/core/facades/config_facade.py", 
            "src/prompt_improver/core/facades/repository_facade.py",
            "src/prompt_improver/core/facades/cli_facade.py",
            "src/prompt_improver/core/facades/api_facade.py",
            "src/prompt_improver/core/facades/performance_facade.py",
        ]
        
        coupling_results = {}
        
        for facade_file in facade_files:
            if Path(facade_file).exists():
                with open(facade_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                prompt_improver_imports = 0
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith('prompt_improver'):
                            prompt_improver_imports += 1
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith('prompt_improver'):
                                prompt_improver_imports += 1
                
                coupling_results[facade_file] = prompt_improver_imports
        
        # Log coupling results
        print("\nFacade Coupling Results:")
        for file, count in coupling_results.items():
            print(f"{file}: {count} internal imports")
        
        # Verify significant coupling reduction
        max_allowed_imports = 5  # Should be much less than original 10-12
        for file, count in coupling_results.items():
            assert count <= max_allowed_imports, f"{file} has {count} imports, expected <={max_allowed_imports}"

    async def test_facade_performance_vs_direct_imports(self):
        """Test facade performance impact vs direct imports."""
        # Test facade approach
        start_time = time.time()
        config_facade = get_config_facade()
        config = config_facade.get_config()
        facade_time = time.time() - start_time
        
        # Test direct import approach
        start_time = time.time()
        from prompt_improver.core.config import get_config as direct_get_config
        direct_config = direct_get_config()
        direct_time = time.time() - start_time
        
        # Facade should not be significantly slower (< 2x)
        performance_ratio = facade_time / direct_time if direct_time > 0 else 1.0
        assert performance_ratio < 2.0, f"Facade is {performance_ratio:.2f}x slower than direct import"
        
        # Results should be equivalent
        assert config.environment.environment == direct_config.environment.environment


@pytest.mark.integration
class TestFacadeIntegration:
    """Integration tests for facade coordination."""

    async def test_cross_facade_coordination(self):
        """Test facades work together correctly."""
        # Get all facades
        di_facade = get_di_facade()
        config_facade = get_config_facade() 
        api_facade = get_api_facade()
        
        # Test coordination
        await di_facade.initialize()
        config = config_facade.get_config()
        api_config = api_facade.get_config()
        
        # Should provide consistent configuration
        assert config.environment.environment == api_config.environment.environment
        
        await di_facade.shutdown()

    async def test_facade_replacement_compatibility(self):
        """Test facades can replace original high-coupling modules."""
        # Test config facade can replace core.config.__init__
        config_facade = get_config_facade()
        
        # Should provide all expected functions
        assert config_facade.get_config() is not None
        assert config_facade.get_database_config() is not None
        assert config_facade.get_security_config() is not None
        assert config_facade.get_monitoring_config() is not None
        assert config_facade.get_ml_config() is not None
        
        # Should support testing environment detection
        assert isinstance(config_facade.is_testing(), bool)
        assert isinstance(config_facade.is_development(), bool)
        assert isinstance(config_facade.is_production(), bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])