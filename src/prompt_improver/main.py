from prompt_improver.core.config import AppConfig  # Redis functionality (
    start_cache_subscriber,
    stop_cache_subscriber,
)

async def startup_event():
    await start_cache_subscriber()

async def shutdown_event():
    await stop_cache_subscriber()
