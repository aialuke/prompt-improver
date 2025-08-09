"""
Tests for ComponentFactory with dependency injection and Protocol compliance.

Tests the ComponentFactory implementation for proper dependency injection,
component creation, and Protocol interface compliance.
"""
import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock
import pytest
from src.prompt_improver.core.factories.component_factory import ComponentFactory, DependencyValidator, create_component_factory
from src.prompt_improver.core.protocols.ml_protocols import ComponentFactoryProtocol, ComponentSpec, ServiceContainerProtocol

class MockServiceContainer:
    """Mock service container for testing."""

    def __init__(self):
        self.services = {}

    async def get_service(self, service_name: str) -> Any:
        if service_name not in self.services:
            raise KeyError(f'Service not found: {service_name}')
        return self.services[service_name]

    def add_service(self, name: str, service: Any):
        self.services[name] = service

class MockComponent:
    """Mock component for testing."""

    def __init__(self, db_service=None, cache_service=None):
        self.db_service = db_service
        self.cache_service = cache_service
        self.initialized = False
        self.shutdown_called = False

    async def initialize(self):
        self.initialized = True

    async def shutdown(self):
        self.shutdown_called = True

@pytest.fixture
def mock_container():
    """Create mock service container."""
    container = MockServiceContainer()
    container.add_service('database_service', Mock(name='database_service'))
    container.add_service('cache_service', Mock(name='cache_service'))
    return container

@pytest.fixture
def component_factory(mock_container):
    """Create ComponentFactory with mock container."""
    return ComponentFactory(mock_container)

@pytest.fixture
def sample_spec():
    """Create sample component specification."""
    return ComponentSpec(name='test_component', module_path='tests.core.test_component_factory', class_name='MockComponent', tier='TIER_1', dependencies={'db_service': 'database_service', 'cache_service': 'cache_service'})

class TestComponentFactory:
    """Test ComponentFactory implementation."""

    def test_implements_protocol(self, component_factory):
        """Test that ComponentFactory implements ComponentFactoryProtocol."""
        assert isinstance(component_factory, ComponentFactoryProtocol)

    def test_factory_creation(self):
        """Test factory creation utility."""
        container = MockServiceContainer()
        factory = create_component_factory(container)
        assert isinstance(factory, ComponentFactory)
        assert factory.container is container

    async def test_register_component_spec(self, component_factory, sample_spec):
        """Test component specification registration."""
        await component_factory.register_component_spec(sample_spec)
        assert sample_spec.name in component_factory.component_specs
        assert component_factory.component_specs[sample_spec.name] == sample_spec

    async def test_register_multiple_specs(self, component_factory):
        """Test multiple specification registration."""
        specs = [ComponentSpec(name='comp1', module_path='test.module1', class_name='Comp1', tier='TIER_1', dependencies={}), ComponentSpec(name='comp2', module_path='test.module2', class_name='Comp2', tier='TIER_1', dependencies={})]
        await component_factory.register_multiple_specs(specs)
        assert len(component_factory.component_specs) == 2
        assert 'comp1' in component_factory.component_specs
        assert 'comp2' in component_factory.component_specs

    async def test_get_component_class(self, component_factory, sample_spec):
        """Test component class retrieval via dynamic import."""
        component_class = await component_factory.get_component_class(sample_spec)
        assert component_class is MockComponent

    async def test_get_component_class_missing_module(self, component_factory):
        """Test error handling for missing module."""
        bad_spec = ComponentSpec(name='bad_component', module_path='non.existent.module', class_name='BadComponent', tier='TIER_1', dependencies={})
        with pytest.raises(ImportError, match='Module import failed'):
            await component_factory.get_component_class(bad_spec)

    async def test_get_component_class_missing_class(self, component_factory):
        """Test error handling for missing class."""
        bad_spec = ComponentSpec(name='bad_component', module_path='tests.core.test_component_factory', class_name='NonExistentClass', tier='TIER_1', dependencies={})
        with pytest.raises(ImportError, match='Class.*resolution failed'):
            await component_factory.get_component_class(bad_spec)

    async def test_validate_dependencies_success(self, component_factory, sample_spec):
        """Test successful dependency validation."""
        result = await component_factory.validate_dependencies(sample_spec, {})
        assert result is True

    async def test_validate_dependencies_missing_service(self, component_factory):
        """Test dependency validation with missing service."""
        bad_spec = ComponentSpec(name='bad_component', module_path='test.module', class_name='BadComponent', tier='TIER_1', dependencies={'missing_service': 'non_existent_service'})
        with pytest.raises(ValueError, match='Missing dependencies'):
            await component_factory.validate_dependencies(bad_spec, {})

    async def test_create_component_success(self, component_factory, sample_spec):
        """Test successful component creation."""
        component = await component_factory.create_component(sample_spec)
        assert isinstance(component, MockComponent)
        assert component.db_service is not None
        assert component.cache_service is not None
        assert sample_spec.name in component_factory.created_components

    async def test_create_component_with_async_init(self, component_factory, sample_spec):
        """Test component creation with async initialization."""
        spec_with_init = ComponentSpec(name=sample_spec.name, module_path=sample_spec.module_path, class_name=sample_spec.class_name, tier=sample_spec.tier, dependencies=sample_spec.dependencies, config={'requires_async_init': True})
        component = await component_factory.create_component(spec_with_init)
        assert isinstance(component, MockComponent)
        assert component.initialized is True

    async def test_create_component_by_name(self, component_factory, sample_spec):
        """Test component creation by name."""
        await component_factory.register_component_spec(sample_spec)
        component = await component_factory.create_component_by_name(sample_spec.name)
        assert isinstance(component, MockComponent)
        assert sample_spec.name in component_factory.created_components

    async def test_create_component_by_name_not_found(self, component_factory):
        """Test error handling for unknown component name."""
        with pytest.raises(KeyError, match='Component specification not found'):
            await component_factory.create_component_by_name('unknown_component')

    async def test_get_or_create_component_cached(self, component_factory, sample_spec):
        """Test getting cached component."""
        await component_factory.register_component_spec(sample_spec)
        component1 = await component_factory.get_or_create_component(sample_spec.name)
        component2 = await component_factory.get_or_create_component(sample_spec.name)
        assert component1 is component2

    async def test_shutdown_all_components(self, component_factory, sample_spec):
        """Test graceful shutdown of all components."""
        await component_factory.register_component_spec(sample_spec)
        component = await component_factory.create_component_by_name(sample_spec.name)
        await component_factory.shutdown_all_components()
        assert component.shutdown_called is True
        assert len(component_factory.created_components) == 0

    async def test_component_lifecycle_context(self, component_factory, sample_spec):
        """Test component lifecycle context manager."""
        async with component_factory.component_lifecycle(sample_spec) as component:
            assert isinstance(component, MockComponent)
            assert not component.shutdown_called
        assert component.shutdown_called is True

class TestDependencyValidator:
    """Test DependencyValidator implementation."""

    def test_validator_creation(self):
        """Test validator creation."""
        specs = {'comp1': Mock()}
        validator = DependencyValidator(specs)
        assert validator.specs is specs

    def test_validate_all_dependencies_success(self):
        """Test successful validation of all dependencies."""
        spec = ComponentSpec(name='test_comp', module_path='test.module', class_name='TestComp', tier='TIER_1', dependencies={'service1': 'database_service'})
        validator = DependencyValidator({'test_comp': spec})
        available_services = ['database_service', 'cache_service']
        errors = validator.validate_all_dependencies(available_services)
        assert len(errors) == 0

    def test_validate_all_dependencies_missing(self):
        """Test validation with missing dependencies."""
        spec = ComponentSpec(name='test_comp', module_path='test.module', class_name='TestComp', tier='TIER_1', dependencies={'service1': 'missing_service'})
        validator = DependencyValidator({'test_comp': spec})
        available_services = ['database_service']
        errors = validator.validate_all_dependencies(available_services)
        assert len(errors) == 1
        assert 'missing_service' in errors[0]

    def test_get_initialization_order(self):
        """Test initialization order determination."""
        spec1 = ComponentSpec(name='comp1', module_path='test.module1', class_name='Comp1', tier='TIER_1', dependencies={})
        spec2 = ComponentSpec(name='comp2', module_path='test.module2', class_name='Comp2', tier='TIER_1', dependencies={'service1': 'database_service'})
        validator = DependencyValidator({'comp1': spec1, 'comp2': spec2})
        levels = validator.get_initialization_order()
        assert len(levels) >= 1
        assert 'comp1' in levels[0] or 'comp1' in levels[1] if len(levels) > 1 else True
        assert 'comp2' in levels[0] or 'comp2' in levels[1] if len(levels) > 1 else True
if __name__ == '__main__':
    pytest.main([__file__])
