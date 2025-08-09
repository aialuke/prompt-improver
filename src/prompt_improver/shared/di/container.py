"""Central dependency injection container following 2025 best practices
Uses python-dependency-injector for clean architecture implementation
"""
from dependency_injector import containers, providers
from prompt_improver.core.config import get_config

class DatabaseContainer(containers.DeclarativeContainer):
    """Container for database-related dependencies"""
    config = providers.Configuration()
    database_session = providers.Resource(providers.Factory)

class InfrastructureContainer(containers.DeclarativeContainer):
    """Container for infrastructure layer dependencies"""
    database = providers.DependenciesContainer()
    prompt_repository = providers.Factory()
    session_repository = providers.Factory()
    metrics_repository = providers.Factory()
    ml_service = providers.Singleton()
    automl_orchestrator = providers.Singleton()
    emergency_operations = providers.Singleton()

class ApplicationContainer(containers.DeclarativeContainer):
    """Main application container for dependency injection

    This container follows clean architecture principles:
    - Domain layer is dependency-free
    - Application layer depends on domain interfaces
    - Infrastructure layer implements the interfaces
    - Presentation layer depends on application layer
    """
    config = providers.Configuration()
    infrastructure = providers.DependenciesContainer()
    improve_prompt_use_case = providers.Factory()
    analyze_prompt_use_case = providers.Factory()
    optimize_rules_use_case = providers.Factory()
    mcp_server = providers.Factory()
    cli_application = providers.Factory()

class TestDIContainer(containers.DeclarativeContainer):
    """Dependency injection container for test dependencies with mocked implementations"""
    mock_prompt_repository = providers.Singleton()
    mock_ml_service = providers.Singleton()
    mock_automl_orchestrator = providers.Singleton()
    test_improve_prompt_use_case = providers.Factory()

def create_production_container() -> ApplicationContainer:
    """Create production container with all dependencies wired

    Returns:
        Configured application container for production use
    """
    container = ApplicationContainer()
    config = get_config()
    container.config.from_dict(config.__dict__)
    return container

def create_test_container() -> TestDIContainer:
    """Create test container with mocked dependencies

    Returns:
        Configured test container with mocks
    """
    container = TestDIContainer()
    container.config.from_dict({'database': {'url': 'sqlite:///:memory:'}, 'ml': {'model_config': {'test_mode': True}}, 'automl': {'n_trials': 5}})
    return container
_container: Optional[ApplicationContainer] = None

def get_container() -> ApplicationContainer:
    """Get the global application container

    Returns:
        Global application container instance

    Raises:
        RuntimeError: If container not initialized
    """
    if _container is None:
        raise RuntimeError('Container not initialized. Call init_container() first.')
    return _container

def init_container(test_mode: bool=False) -> ApplicationContainer:
    """Initialize the global application container

    Args:
        test_mode: Whether to use test configuration

    Returns:
        Initialized container
    """
    global _container
    if test_mode:
        _container = create_test_container()
    else:
        _container = create_production_container()
    return _container

def cleanup_container() -> None:
    """Cleanup the global container and release resources"""
    global _container
    if _container is not None:
        if hasattr(_container, 'shutdown_resources'):
            _container.shutdown_resources()
        _container = None
