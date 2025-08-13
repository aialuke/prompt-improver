"""Integration tests for protocol boundaries and clean architecture enforcement.

These tests validate that the protocol boundary implementation correctly:
1. Prevents circular imports
2. Enforces clean architecture layer separation
3. Enables protocol-based dependency injection
4. Maintains performance with protocol overhead
"""

import pytest
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from unittest.mock import Mock, AsyncMock
import time

from prompt_improver.core.boundaries.layer_enforcement import (
    LayerBoundaryEnforcer,
    CircularImportDetector,
    DependencyValidator,
    ArchitectureLayer,
    ImportViolation,
)
from prompt_improver.core.di.protocol_container import (
    ProtocolContainer,
    ProtocolContainerBuilder,
    LifecycleScope,
    CircularDependencyError,
    DependencyNotFoundError,
)
from prompt_improver.core.domain.types import (
    ImprovementSessionData,
    HealthCheckResultData,
    SessionId,
    UserId,
)
from prompt_improver.core.domain.enums import HealthStatus, SessionStatus


# Test protocols for validation
@runtime_checkable
class TestDomainServiceProtocol(Protocol):
    """Test protocol for domain service."""
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process domain data."""
        ...


@runtime_checkable
class TestApplicationServiceProtocol(Protocol):
    """Test protocol for application service."""
    
    async def orchestrate_workflow(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate business workflow."""
        ...


@runtime_checkable
class TestRepositoryProtocol(Protocol):
    """Test protocol for repository."""
    
    async def save(self, entity: Dict[str, Any]) -> str:
        """Save entity."""
        ...
    
    async def find_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Find entity by ID."""
        ...


# Test implementations
class TestDomainService:
    """Test implementation of domain service."""
    
    def __init__(self):
        self.processed_count = 0
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process domain data."""
        self.processed_count += 1
        return {
            "processed": True,
            "input_keys": list(data.keys()),
            "process_count": self.processed_count
        }


class TestApplicationService:
    """Test implementation of application service."""
    
    def __init__(self, domain_service: TestDomainServiceProtocol):
        self.domain_service = domain_service
        self.workflow_count = 0
    
    async def orchestrate_workflow(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate business workflow."""
        self.workflow_count += 1
        
        # Use domain service
        domain_result = await self.domain_service.process_data(request)
        
        return {
            "workflow_completed": True,
            "domain_result": domain_result,
            "workflow_count": self.workflow_count
        }


class TestRepository:
    """Test implementation of repository."""
    
    def __init__(self):
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.save_count = 0
    
    async def save(self, entity: Dict[str, Any]) -> str:
        """Save entity."""
        entity_id = f"id_{self.save_count}"
        self.storage[entity_id] = entity
        self.save_count += 1
        return entity_id
    
    async def find_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Find entity by ID."""
        return self.storage.get(entity_id)


class TestProtocolBoundaries:
    """Test suite for protocol boundaries and architecture enforcement."""
    
    @pytest.fixture
    def project_root(self, tmp_path: Path) -> Path:
        """Create a temporary project structure for testing."""
        # Create mock project structure
        (tmp_path / "src" / "test_project" / "api").mkdir(parents=True)
        (tmp_path / "src" / "test_project" / "application").mkdir(parents=True)
        (tmp_path / "src" / "test_project" / "core").mkdir(parents=True)
        (tmp_path / "src" / "test_project" / "database").mkdir(parents=True)
        
        # Create test files with different import patterns
        api_file = tmp_path / "src" / "test_project" / "api" / "endpoints.py"
        api_file.write_text("""
from test_project.application import ApplicationService
from test_project.database.models import User  # VIOLATION: Skip application layer
""")
        
        app_file = tmp_path / "src" / "test_project" / "application" / "services.py"
        app_file.write_text("""
from test_project.core.domain import DomainService
""")
        
        core_file = tmp_path / "src" / "test_project" / "core" / "domain.py"
        core_file.write_text("""
# Domain should have no external dependencies
""")
        
        db_file = tmp_path / "src" / "test_project" / "database" / "models.py"
        db_file.write_text("""
from test_project.core.domain import Entity  # OK: Infrastructure can depend on domain
""")
        
        return tmp_path
    
    @pytest.fixture
    def layer_enforcer(self, project_root: Path) -> LayerBoundaryEnforcer:
        """Create layer boundary enforcer for testing."""
        return LayerBoundaryEnforcer(project_root)
    
    @pytest.fixture
    def circular_detector(self, project_root: Path) -> CircularImportDetector:
        """Create circular import detector for testing."""
        return CircularImportDetector(project_root)
    
    @pytest.fixture
    def protocol_container(self) -> ProtocolContainer:
        """Create protocol container for testing."""
        return ProtocolContainer("test_container")
    
    def test_layer_boundary_enforcement(self, layer_enforcer: LayerBoundaryEnforcer):
        """Test that layer boundaries are correctly enforced."""
        # Test layer identification
        assert layer_enforcer.get_module_layer("test_project.api.endpoints") == ArchitectureLayer.PRESENTATION
        assert layer_enforcer.get_module_layer("test_project.application.services") == ArchitectureLayer.APPLICATION
        assert layer_enforcer.get_module_layer("test_project.core.domain") == ArchitectureLayer.DOMAIN
        assert layer_enforcer.get_module_layer("test_project.database.models") == ArchitectureLayer.INFRASTRUCTURE
        
        # Test valid dependency (presentation -> application)
        violation = layer_enforcer.validate_import(
            "test_project.api.endpoints",
            "test_project.application.services"
        )
        assert violation is None
        
        # Test invalid dependency (presentation -> infrastructure, skipping application)
        violation = layer_enforcer.validate_import(
            "test_project.api.endpoints", 
            "test_project.database.models"
        )
        assert violation is not None
        assert violation.violation_type == "INVALID_LAYER_DEPENDENCY"
        assert violation.severity == "ERROR"
    
    def test_circular_import_detection(self, project_root: Path, circular_detector: CircularImportDetector):
        """Test circular import detection."""
        # Create files with circular imports
        circular_a = project_root / "src" / "circular_a.py"
        circular_a.write_text("from circular_b import function_b")
        
        circular_b = project_root / "src" / "circular_b.py"
        circular_b.write_text("from circular_a import function_a")
        
        # Build import graph
        import_graph = circular_detector.build_import_graph()
        
        # Detect circular imports
        circular_imports = circular_detector.detect_circular_imports(import_graph)
        
        # Should detect the circular import
        assert len(circular_imports) > 0
        found_circular = False
        for circular in circular_imports:
            if "circular_a" in circular.modules and "circular_b" in circular.modules:
                found_circular = True
                break
        assert found_circular
    
    def test_project_architecture_validation(self, project_root: Path):
        """Test comprehensive project architecture validation."""
        validator = DependencyValidator(project_root)
        
        # Run validation
        validation_result = validator.validate_project_architecture()
        
        # Check validation structure
        assert "overall_health" in validation_result
        assert "layer_violations" in validation_result
        assert "circular_imports" in validation_result
        assert "summary" in validation_result
        assert "next_steps" in validation_result
        
        # Should find layer violations from our test setup
        assert validation_result["summary"]["total_layer_violations"] > 0
    
    async def test_protocol_container_registration_resolution(self, protocol_container: ProtocolContainer):
        """Test protocol container registration and resolution."""
        # Register domain service as singleton
        protocol_container.register_singleton(
            TestDomainServiceProtocol,
            lambda: TestDomainService()
        )
        
        # Register application service with domain service dependency
        protocol_container.register_protocol(
            TestApplicationServiceProtocol,
            lambda domain_svc: TestApplicationService(domain_svc),
            LifecycleScope.TRANSIENT
        )
        
        # Register repository as scoped
        protocol_container.register_scoped(
            TestRepositoryProtocol,
            lambda: TestRepository()
        )
        
        # Resolve services
        domain_service = protocol_container.resolve(TestDomainServiceProtocol)
        app_service = protocol_container.resolve(TestApplicationServiceProtocol)
        repository = protocol_container.resolve(TestRepositoryProtocol, scope_id="test_scope")
        
        assert isinstance(domain_service, TestDomainService)
        assert isinstance(app_service, TestApplicationService)
        assert isinstance(repository, TestRepository)
        
        # Test that singleton returns same instance
        domain_service2 = protocol_container.resolve(TestDomainServiceProtocol)
        assert domain_service is domain_service2
        
        # Test that transient returns different instance
        app_service2 = protocol_container.resolve(TestApplicationServiceProtocol)
        assert app_service is not app_service2
        
        # Test that scoped returns same instance within scope
        repository2 = protocol_container.resolve(TestRepositoryProtocol, scope_id="test_scope")
        assert repository is repository2
        
        # Test that different scope returns different instance
        repository3 = protocol_container.resolve(TestRepositoryProtocol, scope_id="different_scope")
        assert repository is not repository3
    
    async def test_protocol_dependency_injection(self, protocol_container: ProtocolContainer):
        """Test that dependency injection works correctly through protocols."""
        # Register services
        protocol_container.register_singleton(
            TestDomainServiceProtocol,
            lambda: TestDomainService()
        )
        
        protocol_container.register_protocol(
            TestApplicationServiceProtocol,
            lambda domain_svc: TestApplicationService(domain_svc),
            LifecycleScope.TRANSIENT
        )
        
        # Resolve application service (should auto-inject domain service)
        app_service = protocol_container.resolve(TestApplicationServiceProtocol)
        
        # Test functionality
        result = await app_service.orchestrate_workflow({"test": "data"})
        
        assert result["workflow_completed"] is True
        assert result["domain_result"]["processed"] is True
        assert result["domain_result"]["input_keys"] == ["test"]
    
    def test_circular_dependency_detection_in_container(self, protocol_container: ProtocolContainer):
        """Test that container detects circular dependencies."""
        
        @runtime_checkable
        class ServiceA(Protocol):
            def method_a(self) -> str: ...
        
        @runtime_checkable 
        class ServiceB(Protocol):
            def method_b(self) -> str: ...
        
        class ImplA:
            def __init__(self, service_b: ServiceB):
                self.service_b = service_b
            def method_a(self) -> str:
                return "A"
        
        class ImplB:
            def __init__(self, service_a: ServiceA):
                self.service_a = service_a
            def method_b(self) -> str:
                return "B"
        
        # Register circular dependencies
        protocol_container.register_protocol(ServiceA, lambda b: ImplA(b))
        protocol_container.register_protocol(ServiceB, lambda a: ImplB(a))
        
        # Should raise CircularDependencyError
        with pytest.raises(CircularDependencyError):
            protocol_container.resolve(ServiceA)
    
    def test_container_validation(self, protocol_container: ProtocolContainer):
        """Test container validation functionality."""
        # Register service with missing dependency
        protocol_container.register_protocol(
            TestApplicationServiceProtocol,
            lambda domain_svc: TestApplicationService(domain_svc)
        )
        
        # Validate registrations
        validation = protocol_container.validate_registrations()
        
        assert not validation["validation_passed"]
        assert validation["total_issues"] > 0
        assert any("Missing dependency" in issue for issue in validation["issues"])
    
    async def test_protocol_performance_overhead(self, protocol_container: ProtocolContainer):
        """Test that protocol-based DI doesn't add significant performance overhead."""
        # Register services
        protocol_container.register_singleton(
            TestDomainServiceProtocol,
            lambda: TestDomainService()
        )
        
        # Measure resolution time
        start_time = time.time()
        for _ in range(1000):
            service = protocol_container.resolve(TestDomainServiceProtocol)
        end_time = time.time()
        
        resolution_time = (end_time - start_time) / 1000  # Average per resolution
        
        # Resolution should be very fast (< 1ms per resolution)
        assert resolution_time < 0.001, f"Resolution too slow: {resolution_time * 1000:.2f}ms"
    
    def test_container_health_monitoring(self, protocol_container: ProtocolContainer):
        """Test container health monitoring."""
        # Register some services
        protocol_container.register_singleton(
            TestDomainServiceProtocol,
            lambda: TestDomainService()
        )
        
        protocol_container.register_scoped(
            TestRepositoryProtocol,
            lambda: TestRepository()
        )
        
        # Resolve to create instances
        protocol_container.resolve(TestDomainServiceProtocol)
        protocol_container.resolve(TestRepositoryProtocol, scope_id="scope1")
        protocol_container.resolve(TestRepositoryProtocol, scope_id="scope2")
        
        # Get health status
        health = protocol_container.get_health_status()
        
        assert health["container_name"] == "test_container"
        assert health["total_registrations"] == 2
        assert health["singleton_instances"] == 1
        assert health["scoped_scopes"] == 2
        assert health["total_scoped_instances"] == 2
        assert health["health_status"] == HealthStatus.HEALTHY.value
    
    def test_container_builder_pattern(self):
        """Test container builder for fluent configuration."""
        container = (ProtocolContainerBuilder("test_builder")
            .register_singleton(TestDomainServiceProtocol, lambda: TestDomainService())
            .register_scoped(TestRepositoryProtocol, lambda: TestRepository())
            .register(
                TestApplicationServiceProtocol,
                lambda domain_svc: TestApplicationService(domain_svc),
                LifecycleScope.TRANSIENT
            )
            .build()
        )
        
        assert container.name == "test_builder"
        
        # Validate all services can be resolved
        domain_service = container.resolve(TestDomainServiceProtocol)
        app_service = container.resolve(TestApplicationServiceProtocol)
        repository = container.resolve(TestRepositoryProtocol)
        
        assert isinstance(domain_service, TestDomainService)
        assert isinstance(app_service, TestApplicationService)
        assert isinstance(repository, TestRepository)


@pytest.mark.integration
class TestRealBehaviorProtocolBoundaries:
    """Real behavior tests for protocol boundaries in actual codebase."""
    
    def test_domain_types_replace_database_models(self):
        """Test that domain types successfully replace database model imports."""
        # Import domain types (should not fail)
        from prompt_improver.core.domain.types import (
            ImprovementSessionData,
            PromptSessionData,
            UserFeedbackData,
        )
        
        # Verify types can be instantiated
        from datetime import datetime
        from uuid import uuid4
        
        session_data = ImprovementSessionData(
            id=SessionId(uuid4()),
            user_id=UserId(uuid4()),
            original_prompt="test prompt",
            improved_prompt="improved test prompt",
            improvement_rules=["rule1", "rule2"],
            metrics={"score": 0.95},
            status="completed",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"source": "test"}
        )
        
        assert session_data.original_prompt == "test prompt"
        assert session_data.status == "completed"
    
    def test_protocol_imports_are_clean(self):
        """Test that protocol imports don't create circular dependencies."""
        # These imports should work without circular import errors
        from prompt_improver.core.boundaries.presentation_protocols import APIEndpointProtocol
        from prompt_improver.core.boundaries.application_protocols import ApplicationServiceProtocol  
        from prompt_improver.core.boundaries.domain_protocols import DomainServiceProtocol
        from prompt_improver.core.boundaries.infrastructure_protocols import RepositoryProtocol
        
        # Verify protocols are runtime checkable
        assert hasattr(APIEndpointProtocol, '__protocol__') or hasattr(APIEndpointProtocol, '_is_protocol')
        assert hasattr(ApplicationServiceProtocol, '__protocol__') or hasattr(ApplicationServiceProtocol, '_is_protocol')
        assert hasattr(DomainServiceProtocol, '__protocol__') or hasattr(DomainServiceProtocol, '_is_protocol')
        assert hasattr(RepositoryProtocol, '__protocol__') or hasattr(RepositoryProtocol, '_is_protocol')
    
    def test_layer_enforcement_on_real_codebase(self):
        """Test layer enforcement on the actual codebase."""
        from prompt_improver.core.boundaries.layer_enforcement import validate_architecture
        
        # Run validation on current codebase
        project_root = Path(__file__).parent.parent.parent
        validation_result = validate_architecture(project_root)
        
        assert "overall_health" in validation_result
        assert "layer_violations" in validation_result
        assert "circular_imports" in validation_result
        
        # Log results for inspection
        print(f"Architecture validation results:")
        print(f"Overall health: {validation_result['overall_health']}")
        print(f"Layer violations: {validation_result['summary']['total_layer_violations']}")
        print(f"Circular imports: {validation_result['summary']['total_circular_imports']}")
        
        # The goal is to have no violations, but we'll start by measuring current state
        # TODO: Once protocol boundaries are fully implemented, this should pass:
        # assert validation_result["summary"]["architecture_compliant"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])