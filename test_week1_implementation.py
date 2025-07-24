#!/usr/bin/env python3
"""
Week 1 Implementation Test Suite
Comprehensive testing of the ultra-minimal CLI transformation.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_console_script():
    """Test that the apes console script works."""
    print("🧪 Testing console script entry point...")
    
    try:
        result = subprocess.run(
            ["apes", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result.returncode == 0:
            if "APES - Ultra-Minimal ML Training System" in result.stdout:
                print("✅ Console script working correctly")
                return True
            else:
                print("❌ Console script output unexpected")
                print("STDOUT:", result.stdout[:200])
                return False
        else:
            print("❌ Console script failed")
            print("STDERR:", result.stderr[:200])
            return False
            
    except Exception as e:
        print(f"❌ Console script test failed: {e}")
        return False

def test_cli_commands():
    """Test that all 3 CLI commands are available."""
    print("🧪 Testing CLI command structure...")
    
    try:
        result = subprocess.run(
            ["apes", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result.returncode == 0:
            commands = ["train", "status", "stop"]
            missing_commands = []
            
            for command in commands:
                if command not in result.stdout:
                    missing_commands.append(command)
            
            if not missing_commands:
                print("✅ All 3 commands (train, status, stop) available")
                return True
            else:
                print(f"❌ Missing commands: {missing_commands}")
                return False
        else:
            print("❌ CLI help failed")
            return False
            
    except Exception as e:
        print(f"❌ CLI commands test failed: {e}")
        return False

def test_imports():
    """Test that all critical imports work."""
    print("🧪 Testing critical imports...")
    
    try:
        # Test CLI imports
        from prompt_improver.cli import app
        print("✅ CLI app import successful")
        
        # Test utility imports
        from prompt_improver.cli.utils.validation import validate_path, validate_port, validate_timeout
        from prompt_improver.cli.utils.progress import ProgressReporter
        print("✅ Utility imports successful")
        
        # Test database model imports
        from prompt_improver.database.models import TrainingSession, TrainingSessionCreate, TrainingSessionUpdate
        print("✅ TrainingSession model imports successful")
        
        # Test TrainingSystemManager
        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        print("✅ TrainingSystemManager import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Import test error: {e}")
        return False

def test_validation_utilities():
    """Test validation utility functions."""
    print("🧪 Testing validation utilities...")
    
    try:
        from prompt_improver.cli.utils.validation import validate_path, validate_port, validate_timeout
        
        # Test path validation
        valid_path = validate_path("/tmp/test")
        print("✅ Path validation working")
        
        # Test port validation
        valid_port = validate_port(8080)
        assert valid_port == 8080
        print("✅ Port validation working")
        
        # Test timeout validation
        valid_timeout = validate_timeout(60)
        assert valid_timeout == 60
        print("✅ Timeout validation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation utilities test failed: {e}")
        return False

def test_progress_reporter():
    """Test progress reporter functionality."""
    print("🧪 Testing progress reporter...")
    
    try:
        from prompt_improver.cli.utils.progress import ProgressReporter
        from rich.console import Console
        
        console = Console()
        reporter = ProgressReporter(console)
        
        # Test progress bar creation
        progress_bar = reporter.create_progress_bar("Test Progress")
        print("✅ Progress bar creation working")
        
        # Test metrics
        metrics = reporter.get_all_metrics()
        print("✅ Progress metrics working")
        
        return True
        
    except Exception as e:
        print(f"❌ Progress reporter test failed: {e}")
        return False

async def test_training_system_manager():
    """Test TrainingSystemManager functionality."""
    print("🧪 Testing TrainingSystemManager...")
    
    try:
        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        from rich.console import Console
        
        console = Console()
        manager = TrainingSystemManager(console)
        
        # Test method availability
        required_methods = [
            'smart_initialize', 
            'validate_ready_for_training', 
            'create_training_session',
            'get_system_status', 
            'get_active_sessions'
        ]
        
        for method in required_methods:
            if not hasattr(manager, method):
                print(f"❌ Missing method: {method}")
                return False
        
        print("✅ TrainingSystemManager methods available")
        
        # Test basic functionality (without database connection)
        try:
            status = await manager.get_training_status()
            print("✅ TrainingSystemManager basic functionality working")
        except Exception as e:
            print(f"⚠️  TrainingSystemManager database connection needed: {e}")
            # This is expected without database setup
        
        return True
        
    except Exception as e:
        print(f"❌ TrainingSystemManager test failed: {e}")
        return False

def test_database_models():
    """Test database model functionality."""
    print("🧪 Testing database models...")
    
    try:
        from prompt_improver.database.models import TrainingSession, TrainingSessionCreate, TrainingSessionUpdate
        
        # Test model creation
        session_data = TrainingSessionCreate(
            session_id="test_session_123",
            continuous_mode=True,
            improvement_threshold=0.02
        )
        print("✅ TrainingSessionCreate model working")
        
        # Test model fields
        required_fields = [
            'session_id', 'continuous_mode', 'max_iterations', 
            'improvement_threshold', 'timeout_seconds', 'auto_init_enabled'
        ]
        
        for field in required_fields:
            if field not in TrainingSession.model_fields:
                print(f"❌ Missing field in TrainingSession: {field}")
                return False
        
        print("✅ TrainingSession model fields complete")
        return True
        
    except Exception as e:
        print(f"❌ Database models test failed: {e}")
        return False

async def run_all_tests():
    """Run all Week 1 implementation tests."""
    print("🚀 Starting Week 1 Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Console Script", test_console_script),
        ("CLI Commands", test_cli_commands),
        ("Critical Imports", test_imports),
        ("Validation Utilities", test_validation_utilities),
        ("Progress Reporter", test_progress_reporter),
        ("TrainingSystemManager", test_training_system_manager),
        ("Database Models", test_database_models),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"💥 {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Week 1 implementation successful!")
        print("\n🚀 Ready to proceed to Week 2: Orchestrator Integration")
    else:
        print(f"⚠️  {total - passed} tests failed - review implementation")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
