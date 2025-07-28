"""Central dependency injection container following 2025 best practices
Uses python-dependency-injector for clean architecture implementation
"""
from dependency_injector import containers, providers

# Configuration
from ...core.config import get_config

# Interfaces

class DatabaseContainer(containers.DeclarativeContainer):
    """Container for database-related dependencies"""
    
    # Database configuration
    config = providers.Configuration()
    
    # Database session factory
    database_session = providers.Resource(
        providers.Factory,
        # Will be implemented in infrastructure layer
        # create_database_session,
        # connection_string=config.database.url,
    )

class InfrastructureContainer(containers.DeclarativeContainer):
    """Container for infrastructure layer dependencies"""
    
    # Include database container
    database = providers.DependenciesContainer()
    
    # Repository implementations (will be wired to concrete classes)
    prompt_repository = providers.Factory(
        # SQLPromptRepository,  # Will be imported from infrastructure
        # session=database.database_session,
    )
    
    session_repository = providers.Factory(
        # SQLSessionRepository,  # Will be imported from infrastructure
        # session=database.database_session,
    )
    
    metrics_repository = providers.Factory(
        # SQLMetricsRepository,  # Will be imported from infrastructure
        # session=database.database_session,
    )
    
    # ML service implementations
    ml_service = providers.Singleton(
        # MLModelService,  # Will be imported from infrastructure
        # model_config=providers.Configuration.ml.model_config,
    )
    
    # AutoML orchestrator
    automl_orchestrator = providers.Singleton(
        # AutoMLOrchestrator,  # Will be imported from infrastructure
        # config=providers.Configuration.automl,
    )
    
    # Emergency operations manager
    emergency_operations = providers.Singleton(
        # EmergencyOperationsManager,  # Will be imported from infrastructure
        # backup_dir=providers.Configuration.emergency.backup_dir,
    )

class ApplicationContainer(containers.DeclarativeContainer):
    """Main application container for dependency injection
    
    This container follows clean architecture principles:
    - Domain layer is dependency-free
    - Application layer depends on domain interfaces
    - Infrastructure layer implements the interfaces
    - Presentation layer depends on application layer
    """
    
    # Configuration from centralized config system
    config = providers.Configuration()
    
    # Wire infrastructure container
    infrastructure = providers.DependenciesContainer()
    
    # Domain services (pure business logic)
    # These will be implemented as concrete classes in domain layer
    
    # Application use cases
    improve_prompt_use_case = providers.Factory(
        # ImprovePromptUseCase,  # Will be imported from application layer
        # improvement_service=infrastructure.improvement_service,
        # prompt_repository=infrastructure.prompt_repository,
    )
    
    analyze_prompt_use_case = providers.Factory(
        # AnalyzePromptUseCase,  # Will be imported from application layer
        # ml_service=infrastructure.ml_service,
        # metrics_repository=infrastructure.metrics_repository,
    )
    
    optimize_rules_use_case = providers.Factory(
        # OptimizeRulesUseCase,  # Will be imported from application layer
        # automl_orchestrator=infrastructure.automl_orchestrator,
        # session_repository=infrastructure.session_repository,
    )
    
    # Presentation layer services
    mcp_server = providers.Factory(
        # MCPServer,  # Will be imported from presentation layer
        # improve_prompt_use_case=improve_prompt_use_case,
        # analyze_prompt_use_case=analyze_prompt_use_case,
        # config=config.mcp,
    )
    
    cli_application = providers.Factory(
        # CLIApplication,  # Will be imported from presentation layer
        # improve_prompt_use_case=improve_prompt_use_case,
        # optimize_rules_use_case=optimize_rules_use_case,
        # emergency_operations=infrastructure.emergency_operations,
        # config=config.cli,
    )

class TestContainer(containers.DeclarativeContainer):
    """Container for test dependencies with mocked implementations"""
    
    # Mock implementations for testing
    mock_prompt_repository = providers.Singleton(
        # MockPromptRepository,  # Will be implemented in tests
    )
    
    mock_ml_service = providers.Singleton(
        # MockMLService,  # Will be implemented in tests
    )
    
    mock_automl_orchestrator = providers.Singleton(
        # MockAutoMLOrchestrator,  # Will be implemented in tests
    )
    
    # Override application container for testing
    test_improve_prompt_use_case = providers.Factory(
        # ImprovePromptUseCase,
        # improvement_service=mock_improvement_service,
        # prompt_repository=mock_prompt_repository,
    )

def create_production_container() -> ApplicationContainer:
    """Create production container with all dependencies wired
    
    Returns:
        Configured application container for production use
    """
    container = ApplicationContainer()
    
    # Load configuration
    config = get_config()
    container.config.from_dict(config.__dict__)
    
    # Wire infrastructure dependencies
    # This will be completed when infrastructure implementations are ready
    
    return container

def create_test_container() -> TestContainer:
    """Create test container with mocked dependencies
    
    Returns:
        Configured test container with mocks
    """
    container = TestContainer()
    
    # Configure test settings
    container.config.from_dict({
        "database": {"url": "sqlite:///:memory:"},
        "ml": {"model_config": {"test_mode": True}},
        "automl": {"n_trials": 5},  # Fewer trials for tests
    })
    
    return container

# Global container instance (will be initialized in main)
_container: Optional[ApplicationContainer] = None

def get_container() -> ApplicationContainer:
    """Get the global application container
    
    Returns:
        Global application container instance
    
    Raises:
        RuntimeError: If container not initialized
    """
    if _container is None:
        raise RuntimeError("Container not initialized. Call init_container() first.")
    return _container

def init_container(test_mode: bool = False) -> ApplicationContainer:
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
        # Shutdown resources if container supports it
        if hasattr(_container, 'shutdown_resources'):
            _container.shutdown_resources()
        _container = None