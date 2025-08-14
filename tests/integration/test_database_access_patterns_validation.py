"""Integration tests validating repository pattern compliance and database access elimination.

Tests to ensure:
1. No direct database imports in business logic layers
2. All database access goes through repository protocols
3. Session management follows Clean Architecture principles
4. Performance characteristics are maintained
"""

import ast
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Set

import pytest

logger = logging.getLogger(__name__)


class DatabaseImportChecker:
    """Analyzes Python files for direct database import violations."""
    
    def __init__(self):
        self.violations: Dict[str, List[str]] = {}
        self.allowed_patterns = {
            # Test files can import database for setup
            "test_",
            "conftest.py",
            # Performance and monitoring files may need database access
            "performance",
            "monitoring",
            # Database layer itself can import database modules
            "/database/",
            # Repository implementations need database access
            "/repositories/impl/",
            # Infrastructure layer can access database
            "/infrastructure/",
        }
        
    def is_allowed_file(self, file_path: str) -> bool:
        """Check if a file is allowed to have direct database imports."""
        return any(pattern in file_path for pattern in self.allowed_patterns)
    
    def check_file(self, file_path: Path) -> List[str]:
        """Check a single file for database import violations."""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and 'prompt_improver.database' in node.module:
                        # Check if this is a direct database import
                        if not self.is_allowed_file(str(file_path)):
                            import_info = f"Line {node.lineno}: from {node.module}"
                            if node.names:
                                names = [alias.name for alias in node.names]
                                import_info += f" import {', '.join(names)}"
                            violations.append(import_info)
                            
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if 'prompt_improver.database' in alias.name:
                            if not self.is_allowed_file(str(file_path)):
                                violations.append(f"Line {node.lineno}: import {alias.name}")
                                
        except Exception as e:
            logger.warning(f"Could not parse {file_path}: {e}")
            
        return violations
    
    def scan_directory(self, directory: Path) -> Dict[str, List[str]]:
        """Scan entire directory for database import violations."""
        violations = {}
        
        for py_file in directory.rglob("*.py"):
            file_violations = self.check_file(py_file)
            if file_violations:
                relative_path = py_file.relative_to(directory)
                violations[str(relative_path)] = file_violations
                
        return violations


@pytest.fixture
def src_directory():
    """Get the source directory path."""
    current_file = Path(__file__)
    # Navigate from tests/integration/test_file.py to src/
    src_path = current_file.parent.parent.parent / "src"
    return src_path


@pytest.fixture
def database_import_checker():
    """Create database import checker instance."""
    return DatabaseImportChecker()


class TestRepositoryPatternCompliance:
    """Test repository pattern compliance and Clean Architecture principles."""
    
    def test_no_direct_database_imports_in_application_layer(self, src_directory, database_import_checker):
        """Test that application services don't have direct database imports."""
        app_services_dir = src_directory / "prompt_improver" / "application" / "services"
        
        if app_services_dir.exists():
            violations = database_import_checker.scan_directory(app_services_dir)
            
            if violations:
                error_message = "Direct database imports found in application services:\n"
                for file_path, file_violations in violations.items():
                    error_message += f"\n{file_path}:\n"
                    for violation in file_violations:
                        error_message += f"  - {violation}\n"
                        
                pytest.fail(error_message)
    
    def test_no_direct_database_imports_in_api_layer(self, src_directory, database_import_checker):
        """Test that API endpoints don't have direct database imports."""
        api_dir = src_directory / "prompt_improver" / "api"
        
        if api_dir.exists():
            violations = database_import_checker.scan_directory(api_dir)
            
            # Remove allowed files (health endpoints may need database for checks)
            filtered_violations = {
                file_path: file_violations 
                for file_path, file_violations in violations.items()
                if not any(allowed in file_path for allowed in ["health.py", "__init__.py"])
            }
            
            if filtered_violations:
                error_message = "Direct database imports found in API layer:\n"
                for file_path, file_violations in filtered_violations.items():
                    error_message += f"\n{file_path}:\n"
                    for violation in file_violations:
                        error_message += f"  - {violation}\n"
                        
                pytest.fail(error_message)
    
    def test_core_services_use_proper_abstractions(self, src_directory, database_import_checker):
        """Test that core services use repository patterns instead of direct DB access."""
        core_services_dir = src_directory / "prompt_improver" / "core" / "services"
        
        if core_services_dir.exists():
            violations = database_import_checker.scan_directory(core_services_dir)
            
            # Filter out TYPE_CHECKING imports (these are acceptable for type hints)
            filtered_violations = {}
            for file_path, file_violations in violations.items():
                critical_violations = []
                
                try:
                    full_path = core_services_dir / file_path
                    with open(full_path, 'r') as f:
                        content = f.read()
                        
                    # Check if violations are only in TYPE_CHECKING blocks
                    for violation in file_violations:
                        line_num = int(violation.split(':')[0].split()[-1])
                        lines = content.split('\n')
                        
                        # Look for TYPE_CHECKING context around the line
                        in_type_checking = False
                        for i in range(max(0, line_num - 10), min(len(lines), line_num + 5)):
                            if 'TYPE_CHECKING' in lines[i]:
                                in_type_checking = True
                                break
                                
                        if not in_type_checking:
                            critical_violations.append(violation)
                            
                except Exception as e:
                    logger.warning(f"Could not analyze TYPE_CHECKING context for {file_path}: {e}")
                    critical_violations = file_violations  # Be conservative
                    
                if critical_violations:
                    filtered_violations[file_path] = critical_violations
            
            if filtered_violations:
                error_message = "Critical database imports found in core services (outside TYPE_CHECKING):\n"
                for file_path, file_violations in filtered_violations.items():
                    error_message += f"\n{file_path}:\n"
                    for violation in file_violations:
                        error_message += f"  - {violation}\n"
                        
                pytest.fail(error_message)
    
    def test_session_manager_protocol_usage(self, src_directory):
        """Test that files use SessionManagerProtocol instead of direct DatabaseServices."""
        protocol_usage_files = []
        database_services_usage_files = []
        
        # Scan for files using the proper protocol
        for py_file in src_directory.rglob("*.py"):
            if any(skip in str(py_file) for skip in ["/database/", "test_", "__pycache__"]):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'SessionManagerProtocol' in content:
                    protocol_usage_files.append(str(py_file.relative_to(src_directory)))
                    
                if 'DatabaseServices' in content and 'TYPE_CHECKING' not in content:
                    database_services_usage_files.append(str(py_file.relative_to(src_directory)))
                    
            except Exception as e:
                logger.warning(f"Could not read {py_file}: {e}")
        
        # We should have some files using the protocol
        assert len(protocol_usage_files) > 0, "No files found using SessionManagerProtocol"
        
        # Files using DatabaseServices directly should be minimal and in allowed locations
        forbidden_usage = [
            f for f in database_services_usage_files 
            if not any(allowed in f for allowed in [
                "repository", "database", "performance", "monitoring", "test_"
            ])
        ]
        
        if forbidden_usage:
            pytest.fail(f"Files using DatabaseServices directly in business logic: {forbidden_usage}")


class TestArchitecturalCompliance:
    """Test overall architectural compliance with Clean Architecture principles."""
    
    def test_dependency_direction_compliance(self, src_directory):
        """Test that dependencies flow inward according to Clean Architecture."""
        # Application layer should not import from infrastructure
        app_dir = src_directory / "prompt_improver" / "application"
        if not app_dir.exists():
            pytest.skip("Application directory not found")
            
        violations = []
        
        for py_file in app_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for outward dependencies (violations)
                outward_imports = [
                    'from prompt_improver.database import',
                    'from prompt_improver.api import',
                    'from prompt_improver.mcp_server import',
                ]
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    for forbidden_import in outward_imports:
                        if forbidden_import in line and 'TYPE_CHECKING' not in line:
                            relative_path = py_file.relative_to(src_directory)
                            violations.append(f"{relative_path}:{line_num} - {line.strip()}")
                            
            except Exception as e:
                logger.warning(f"Could not check dependencies in {py_file}: {e}")
        
        if violations:
            error_msg = "Clean Architecture violations found:\n" + "\n".join(violations)
            pytest.fail(error_msg)
    
    def test_performance_characteristics_maintained(self, src_directory):
        """Test that architectural changes don't degrade performance characteristics."""
        # This is a basic smoke test - in a real system you'd have more sophisticated metrics
        
        start_time = time.time()
        
        # Import key modules to test import performance
        try:
            from prompt_improver.application.services.apriori_application_service import AprioriApplicationService
            from prompt_improver.application.services.health_application_service import HealthApplicationService
            from prompt_improver.application.services.prompt_application_service import PromptApplicationService
            from prompt_improver.application.services.training_application_service import TrainingApplicationService
        except ImportError as e:
            pytest.fail(f"Import performance test failed: {e}")
        
        import_time = time.time() - start_time
        
        # Imports should be fast (under 1 second for all application services)
        assert import_time < 1.0, f"Import time too slow: {import_time:.2f}s"


class TestIntegrationReadiness:
    """Test that the codebase is ready for integration testing."""
    
    def test_repository_protocols_available(self, src_directory):
        """Test that all required repository protocols are available."""
        protocol_dir = src_directory / "prompt_improver" / "repositories" / "protocols"
        
        required_protocols = [
            "session_manager_protocol.py",
            "apriori_repository_protocol.py",
            "ml_repository_protocol.py", 
            "analytics_repository_protocol.py",
            "prompt_repository_protocol.py",
            "health_repository_protocol.py",
        ]
        
        missing_protocols = []
        for protocol_file in required_protocols:
            protocol_path = protocol_dir / protocol_file
            if not protocol_path.exists():
                missing_protocols.append(protocol_file)
        
        if missing_protocols:
            pytest.fail(f"Missing repository protocol files: {missing_protocols}")
    
    def test_dependency_injection_ready(self, src_directory):
        """Test that dependency injection container is properly configured."""
        di_dir = src_directory / "prompt_improver" / "core" / "di"
        
        if not di_dir.exists():
            pytest.skip("DI directory not found")
        
        # Check that container files exist
        container_files = list(di_dir.glob("*container*.py"))
        assert len(container_files) > 0, "No container files found in DI directory"
        
        # Check for container protocol/interface
        protocol_files = list(di_dir.glob("*protocol*.py"))
        assert len(protocol_files) > 0, "No protocol files found in DI directory"


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v"])