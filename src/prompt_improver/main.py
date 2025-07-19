from prompt_improver.utils.redis_cache import (
    start_cache_subscriber,
    stop_cache_subscriber,
)


async def startup_event():
    await start_cache_subscriber()


async def shutdown_event():
    await stop_cache_subscriber()
