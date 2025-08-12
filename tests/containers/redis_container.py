import asyncio
import os
from dataclasses import dataclass


@dataclass
class RedisContainer:
    """Minimal Redis container stub for tests.
    Provides host/port and start/stop no-ops to satisfy integration tests that expect this helper.
    """

    host: str = os.getenv("REDIS_HOST", "redis")
    port: int = int(os.getenv("REDIS_PORT", "6379"))

    async def start(self):
        # No-op: assume local redis or external service configured
        await asyncio.sleep(0)
        return self

    async def stop(self):
        await asyncio.sleep(0)
