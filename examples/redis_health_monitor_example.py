"""Example demonstrating Redis health monitoring with periodic checks."""

import asyncio
import logging
from src.prompt_improver.services.health.redis_monitor import RedisHealthMonitor, start_redis_health_monitor
from src.prompt_improver.services.health.service import get_health_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_redis_health_monitor():
    """Demonstrate Redis health monitoring functionality."""
    logger.info("Starting Redis health monitor demonstration...")
    
    # Configuration for Redis health monitoring
    config = {
        'check_interval': 30,  # Check every 30 seconds
        'failure_threshold': 2,  # Trigger reconnection after 2 failures
        'latency_threshold': 200,  # 200ms latency threshold
        'reconnection': {
            'max_retries': 3,
            'backoff_factor': 1.5
        }
    }
    
    # Create and configure Redis health monitor
    redis_monitor = RedisHealthMonitor(config)
    
    logger.info("Running manual health checks...")
    
    # Perform manual health checks
    for i in range(5):
        logger.info(f"Health check #{i+1}")
        
        # Perform health check
        result = await redis_monitor.check()
        
        logger.info(f"Status: {result.status.value}")
        logger.info(f"Message: {result.message}")
        
        if result.error:
            logger.error(f"Error: {result.error}")
        
        if result.response_time_ms:
            logger.info(f"Response time: {result.response_time_ms}ms")
        
        logger.info("---")
        await asyncio.sleep(2)
    
    # Demonstrate integration with health service
    logger.info("Adding Redis monitor to health service...")
    health_service = get_health_service()
    health_service.add_checker(redis_monitor)
    
    # Run aggregated health check
    logger.info("Running aggregated health check...")
    aggregated_result = await health_service.run_health_check()
    
    logger.info(f"Overall status: {aggregated_result.overall_status.value}")
    logger.info(f"Redis check: {aggregated_result.checks['redis'].status.value}")
    logger.info(f"Redis message: {aggregated_result.checks['redis'].message}")
    
    logger.info("Demonstration completed!")


async def demonstrate_background_monitoring():
    """Demonstrate background Redis health monitoring."""
    logger.info("Starting background Redis health monitoring...")
    
    # Configuration for background monitoring
    config = {
        'check_interval': 15,  # Check every 15 seconds
        'failure_threshold': 3,
        'latency_threshold': 100,
        'reconnection': {
            'max_retries': 5,
            'backoff_factor': 2
        }
    }
    
    # Start background monitoring (this would run indefinitely)
    # In a real application, this would be started during application startup
    logger.info("Background monitoring would run indefinitely...")
    logger.info("Configuration:")
    logger.info(f"  - Check interval: {config['check_interval']} seconds")
    logger.info(f"  - Failure threshold: {config['failure_threshold']}")
    logger.info(f"  - Latency threshold: {config['latency_threshold']}ms")
    logger.info(f"  - Max retries: {config['reconnection']['max_retries']}")
    logger.info(f"  - Backoff factor: {config['reconnection']['backoff_factor']}")
    
    # Note: In production, you would call:
    # await start_redis_health_monitor(config)
    # But this runs indefinitely, so we'll just show the concept


if __name__ == "__main__":
    # Run the demonstrations
    asyncio.run(demonstrate_redis_health_monitor())
    asyncio.run(demonstrate_background_monitoring())
