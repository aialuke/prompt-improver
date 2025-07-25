#!/usr/bin/env python3
"""
Fixed Signal Handling Tests - Demonstrating proper test methodology

This script shows how the signal handling tests should be written to work
reliably in test environments while still testing the actual functionality.
"""

import asyncio
import signal
import sys
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prompt_improver.cli.core.signal_handler import AsyncSignalHandler, SignalOperation, SignalContext


async def test_status_signal_handling_fixed():
    """Test SIGUSR2 status signal handling - FIXED VERSION."""
    if os.name == 'nt':  # Skip on Windows
        print("❌ SIGUSR2 not available on Windows")
        return False

    # Create signal handler and event loop
    signal_handler = AsyncSignalHandler()
    loop = asyncio.get_event_loop()
    signal_handler.setup_signal_handlers(loop)

    # Register status handler
    status_handler = AsyncMock(return_value={"status": "running", "sessions": 2})
    signal_handler.register_operation_handler(SignalOperation.STATUS_REPORT, status_handler)

    # FIXED: Use direct signal handler call instead of os.kill
    # This tests the actual signal handling logic without relying on signal delivery
    signal_handler._handle_signal(signal.SIGUSR2, "SIGUSR2")

    # Wait for async execution to complete
    await asyncio.sleep(0.2)  # Give more time for async execution

    # Verify handler was called
    if status_handler.called:
        call_args = status_handler.call_args[0][0]
        assert isinstance(call_args, SignalContext)
        assert call_args.operation == SignalOperation.STATUS_REPORT
        assert call_args.signal_name == "SIGUSR2"
        print("✅ Status signal handling test PASSED")
        return True
    else:
        print("❌ Status signal handling test FAILED - handler not called")
        return False


async def test_config_reload_signal_handling_fixed():
    """Test SIGHUP config reload signal handling - FIXED VERSION."""
    if os.name == 'nt':  # Skip on Windows
        print("❌ SIGHUP not available on Windows")
        return False

    # Create signal handler and event loop
    signal_handler = AsyncSignalHandler()
    loop = asyncio.get_event_loop()
    signal_handler.setup_signal_handlers(loop)

    # Register config handler
    config_handler = AsyncMock(return_value={"config": "reloaded"})
    signal_handler.register_operation_handler(SignalOperation.CONFIG_RELOAD, config_handler)

    # FIXED: Use direct signal handler call
    signal_handler._handle_signal(signal.SIGHUP, "SIGHUP")

    # Wait for async execution to complete
    await asyncio.sleep(0.2)

    # Verify handler was called
    if config_handler.called:
        call_args = config_handler.call_args[0][0]
        assert isinstance(call_args, SignalContext)
        assert call_args.operation == SignalOperation.CONFIG_RELOAD
        assert call_args.signal_name == "SIGHUP"
        print("✅ Config reload signal handling test PASSED")
        return True
    else:
        print("❌ Config reload signal handling test FAILED - handler not called")
        return False


async def test_real_signal_delivery():
    """Test that real signal delivery works (for comparison)."""
    if os.name == 'nt':
        print("❌ Real signals not available on Windows")
        return False

    # Create signal handler and event loop
    signal_handler = AsyncSignalHandler()
    loop = asyncio.get_event_loop()
    signal_handler.setup_signal_handlers(loop)

    # Use a simple flag instead of AsyncMock for real signal test
    signal_received = {"status": False, "config": False}

    async def status_handler(context: SignalContext):
        signal_received["status"] = True
        return {"status": "running"}

    async def config_handler(context: SignalContext):
        signal_received["config"] = True
        return {"config": "reloaded"}

    signal_handler.register_operation_handler(SignalOperation.STATUS_REPORT, status_handler)
    signal_handler.register_operation_handler(SignalOperation.CONFIG_RELOAD, config_handler)

    # Send real signals
    try:
        os.kill(os.getpid(), signal.SIGUSR2)
        await asyncio.sleep(0.3)
        
        os.kill(os.getpid(), signal.SIGHUP)
        await asyncio.sleep(0.3)
        
        if signal_received["status"] and signal_received["config"]:
            print("✅ Real signal delivery test PASSED")
            return True
        else:
            print(f"❌ Real signal delivery test FAILED - Status: {signal_received['status']}, Config: {signal_received['config']}")
            return False
            
    except Exception as e:
        print(f"❌ Real signal delivery test ERROR: {e}")
        return False


async def main():
    """Run all fixed tests."""
    print("🔧 Fixed Signal Handling Tests")
    print("=" * 50)
    
    results = []
    
    print("\n1. Testing status signal handling (fixed method)...")
    results.append(await test_status_signal_handling_fixed())
    
    print("\n2. Testing config reload signal handling (fixed method)...")
    results.append(await test_config_reload_signal_handling_fixed())
    
    print("\n3. Testing real signal delivery (for comparison)...")
    results.append(await test_real_signal_delivery())
    
    print("\n📊 Test Results Summary:")
    print(f"   Fixed method tests passed: {sum(results[:2])}/2")
    print(f"   Real signal delivery works: {results[2]}")
    
    if all(results):
        print("✅ All tests passed! Signal handling is fully functional.")
    else:
        print("⚠️  Some tests failed, but this demonstrates the testing methodology.")
    
    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(main())
        sys.exit(0 if all(results) else 1)
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
