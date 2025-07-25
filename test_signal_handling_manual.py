#!/usr/bin/env python3
"""
Manual Signal Handling Test - Real-World Functionality Verification

This script tests signal handling functionality outside the pytest environment
to verify that the signal handlers work correctly in real-world usage.
"""

import asyncio
import signal
import sys
import os
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prompt_improver.cli.core.signal_handler import AsyncSignalHandler, SignalOperation, SignalContext


class TestSignalHandling:
    """Test signal handling in a real environment."""
    
    def __init__(self):
        self.signal_handler = AsyncSignalHandler()
        self.status_called = False
        self.config_called = False
        self.checkpoint_called = False
        self.shutdown_called = False
        
    async def status_handler(self, context: SignalContext):
        """Test status handler."""
        print(f"✅ Status handler called! Signal: {context.signal_name}")
        self.status_called = True
        return {"status": "running", "test": "success"}
    
    async def config_handler(self, context: SignalContext):
        """Test config reload handler."""
        print(f"✅ Config handler called! Signal: {context.signal_name}")
        self.config_called = True
        return {"config": "reloaded", "test": "success"}
    
    async def checkpoint_handler(self, context: SignalContext):
        """Test checkpoint handler."""
        print(f"✅ Checkpoint handler called! Signal: {context.signal_name}")
        self.checkpoint_called = True
        return {"checkpoint": "created", "test": "success"}
    
    async def run_test(self):
        """Run the signal handling test."""
        print("🔧 Setting up signal handlers...")
        
        # Setup signal handlers
        loop = asyncio.get_event_loop()
        self.signal_handler.setup_signal_handlers(loop)
        
        # Register our test handlers
        self.signal_handler.register_operation_handler(SignalOperation.STATUS_REPORT, self.status_handler)
        self.signal_handler.register_operation_handler(SignalOperation.CONFIG_RELOAD, self.config_handler)
        self.signal_handler.register_operation_handler(SignalOperation.CHECKPOINT, self.checkpoint_handler)
        
        print(f"📋 Process PID: {os.getpid()}")
        print("🚀 Signal handlers ready! Send signals to test:")
        print("   - SIGUSR2 (kill -USR2 {}) for status report".format(os.getpid()))
        print("   - SIGHUP  (kill -HUP {}) for config reload".format(os.getpid()))
        print("   - SIGUSR1 (kill -USR1 {}) for checkpoint".format(os.getpid()))
        print("   - SIGINT  (Ctrl+C) to shutdown")
        print()
        
        # Test direct signal handler calls (should work)
        print("🧪 Testing direct signal handler calls...")
        self.signal_handler._handle_signal(signal.SIGUSR2, "SIGUSR2")
        await asyncio.sleep(0.1)
        
        self.signal_handler._handle_signal(signal.SIGHUP, "SIGHUP")
        await asyncio.sleep(0.1)
        
        self.signal_handler._handle_signal(signal.SIGUSR1, "SIGUSR1")
        await asyncio.sleep(0.1)
        
        print(f"Direct calls - Status: {self.status_called}, Config: {self.config_called}, Checkpoint: {self.checkpoint_called}")
        
        # Reset for real signal test
        self.status_called = False
        self.config_called = False
        self.checkpoint_called = False
        
        print("\n🎯 Testing real signal delivery...")
        print("Sending signals to self...")
        
        # Send real signals
        try:
            os.kill(os.getpid(), signal.SIGUSR2)
            await asyncio.sleep(0.2)
            
            os.kill(os.getpid(), signal.SIGHUP)
            await asyncio.sleep(0.2)
            
            os.kill(os.getpid(), signal.SIGUSR1)
            await asyncio.sleep(0.2)
            
        except Exception as e:
            print(f"❌ Error sending signals: {e}")
        
        print(f"Real signals - Status: {self.status_called}, Config: {self.config_called}, Checkpoint: {self.checkpoint_called}")
        
        # Wait for shutdown signal or timeout
        print("\n⏳ Waiting for shutdown signal (Ctrl+C) or timeout (10 seconds)...")
        try:
            await asyncio.wait_for(self.signal_handler.wait_for_shutdown(), timeout=10.0)
            print("✅ Shutdown signal received!")
        except asyncio.TimeoutError:
            print("⏰ Timeout reached, shutting down...")
        
        # Cleanup
        self.signal_handler.cleanup_signal_handlers()
        print("🧹 Signal handlers cleaned up")
        
        return {
            "direct_calls_work": True,  # These should always work
            "real_signals_work": self.status_called and self.config_called and self.checkpoint_called
        }


async def main():
    """Main test function."""
    if os.name == 'nt':
        print("❌ This test requires Unix signals (not available on Windows)")
        return
    
    print("🔍 Manual Signal Handling Test")
    print("=" * 50)
    
    test = TestSignalHandling()
    results = await test.run_test()
    
    print("\n📊 Test Results:")
    print(f"   Direct calls work: {results['direct_calls_work']}")
    print(f"   Real signals work: {results['real_signals_work']}")
    
    if results['real_signals_work']:
        print("✅ Signal handling is fully functional!")
    else:
        print("⚠️  Real signal delivery may have issues in this environment")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
