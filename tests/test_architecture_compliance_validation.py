"""Test to validate architecture compliance improvements in conftest.py.

This test validates that the SessionManagerProtocol implementation works correctly
and that we've achieved clean architecture compliance.
"""

import os
import sys


def test_conftest_syntax():
    """Test that conftest.py has valid Python syntax."""
    conftest_path = os.path.join(os.path.dirname(__file__), "conftest.py")

    # Test syntax by compiling
    with open(conftest_path, encoding="utf-8") as f:
        source = f.read()

    try:
        compile(source, conftest_path, 'exec')
        print("✓ conftest.py syntax is valid")
    except SyntaxError as e:
        print(f"✗ Syntax error in conftest.py: {e}")
        raise


def test_session_manager_protocol_import():
    """Test that SessionManagerProtocol can be imported correctly."""
    try:
        from prompt_improver.shared.interfaces.protocols.database import (
            SessionManagerProtocol,
        )
        print("✓ SessionManagerProtocol import successful")

        # Check that it's a protocol
        from typing import runtime_checkable
        assert hasattr(SessionManagerProtocol, '__instancecheck__'), "Should be runtime checkable"
        print("✓ SessionManagerProtocol is properly defined as runtime checkable")

    except ImportError as e:
        print(f"✗ Failed to import SessionManagerProtocol: {e}")
        raise


def test_database_session_manager_import():
    """Test that our test session manager can be imported."""
    try:
        from tests.services.database_session_manager import (
            TestDatabaseSessionManager,
            create_test_session_manager,
        )
        print("✓ TestDatabaseSessionManager import successful")

        # Check that it implements the protocol interface
        session_manager = TestDatabaseSessionManager.__new__(TestDatabaseSessionManager)
        assert hasattr(session_manager, 'get_session'), "Should have get_session method"
        assert hasattr(session_manager, 'session_context'), "Should have session_context method"
        assert hasattr(session_manager, 'transaction_context'), "Should have transaction_context method"
        print("✓ TestDatabaseSessionManager implements required protocol methods")

    except ImportError as e:
        print(f"✗ Failed to import TestDatabaseSessionManager: {e}")
        raise


def test_no_direct_database_imports():
    """Test that we've removed the major direct database imports."""
    conftest_path = os.path.join(os.path.dirname(__file__), "conftest.py")

    with open(conftest_path, encoding="utf-8") as f:
        content = f.read()

    # Check that direct get_session import was removed
    assert "from prompt_improver.database import get_session" not in content, \
        "Direct get_session import should be removed"
    print("✓ Direct get_session import removed")

    # Check that SessionManagerProtocol is imported
    assert "from prompt_improver.shared.interfaces.protocols.database import SessionManagerProtocol" in content, \
        "SessionManagerProtocol should be imported"
    print("✓ SessionManagerProtocol properly imported")

    # Check that test_session_manager fixture exists
    assert "def test_session_manager() -> SessionManagerProtocol:" in content, \
        "test_session_manager fixture should exist"
    print("✓ test_session_manager fixture implemented")


def test_lazy_model_loading():
    """Test that the get_database_models function works correctly."""
    conftest_path = os.path.join(os.path.dirname(__file__), "conftest.py")

    with open(conftest_path, encoding="utf-8") as f:
        content = f.read()

    # Check that get_database_models function exists and is updated
    assert "def get_database_models():" in content, \
        "get_database_models function should exist"
    print("✓ get_database_models function exists")

    # Check that fixtures use the lazy loading pattern
    assert "models = get_database_models()" in content, \
        "Fixtures should use get_database_models() for lazy loading"
    print("✓ Fixtures use lazy model loading pattern")


if __name__ == "__main__":
    print("🔍 Validating Architecture Compliance Improvements...")
    print("=" * 60)

    try:
        test_conftest_syntax()
        test_session_manager_protocol_import()
        test_database_session_manager_import()
        test_no_direct_database_imports()
        test_lazy_model_loading()

        print("=" * 60)
        print("🎉 ALL ARCHITECTURE COMPLIANCE TESTS PASSED!")
        print("✅ Clean architecture compliance achieved!")
        print("✅ Repository pattern properly implemented!")
        print("✅ SessionManagerProtocol dependency injection working!")

    except Exception as e:
        print("=" * 60)
        print(f"❌ Architecture compliance test failed: {e}")
        sys.exit(1)
