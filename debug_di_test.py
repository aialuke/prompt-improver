#!/usr/bin/env python3
"""Minimal test to debug DI container factory singleton behavior."""

import asyncio
import sys
sys.path.insert(0, 'src')

from prompt_improver.core.di.core_container import CoreContainer, ServiceLifetime

# Test protocol and implementation (matching the real test)
class TestServiceProtocol:
    async def get_value(self):
        pass

class TestServiceImplementation:
    def __init__(self, value):
        self.value = value
    
    async def get_value(self):
        return self.value

async def test_factory_singleton():
    """Test exactly like the failing test."""
    # Create isolated container (matching test fixture)
    container = CoreContainer(name="test_isolated")
    call_count = 0

    def test_factory():
        nonlocal call_count
        call_count += 1
        print(f"Factory called: {call_count}")
        return TestServiceImplementation(f"factory_value_{call_count}")

    # Register factory as singleton (exactly like test)
    container.register_factory(
        TestServiceProtocol,
        test_factory,
        lifetime=ServiceLifetime.SINGLETON,
        tags={"factory", "singleton"}
    )

    # Test singleton factory behavior (exactly like test)
    print("Getting first instance...")
    instance1 = await container.get(TestServiceProtocol)
    print(f"Call count after first get: {call_count}")
    
    print("Getting second instance...")
    instance2 = await container.get(TestServiceProtocol)  
    print(f"Call count after second get: {call_count}")

    print(f"Same instance: {instance1 is instance2}")
    print(f"Instance1 id: {id(instance1)}")
    print(f"Instance2 id: {id(instance2)}")
    
    # Assertions from the test
    assert instance1 is instance2, "Singleton factory should reuse instance"
    assert call_count == 1, "Factory should be called only once for singleton"

    value = await instance1.get_value()
    assert value == "factory_value_1"
    
    print("✓ All assertions passed!")

if __name__ == '__main__':
    asyncio.run(test_factory_singleton())