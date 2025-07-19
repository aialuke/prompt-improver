# Redis Pattern Cache Rollout with Canary A/B Testing

## Overview
This documentation provides a comprehensive view of the Redis pattern cache rollout behind a feature flag using canary A/B testing. The rollout aims to enhance the caching mechanism of the Adaptive Prompt Enhancement System (APES) with controlled and measurable deployment.

## Feature Flag: ENABLE_PATTERN_CACHE
- **Description**: This flag controls the rollout of the Redis pattern cache.
- **Activation**: Managed via environment variables.
- **Purpose**: Ensures that the feature can be toggled without deploying new code across environments.

## Dedicated Cache DB Setup
- **Database**: DB 2
- **Eviction Policy**: `allkeys-lru` - Least Recently Used eviction is implemented for effective memory management.

## CLI Commands
### Cache Management
- `apes cache-stats`: Display detailed statistics of the Redis cache, including memory usage, fragmentation ratios, and hit/miss rates.
- `apes cache-clear`: Clear the Redis cache as needed.

### Canary Testing Management
- `apes canary-status`: Show current canary testing status and metrics as JSON.
- `apes canary-adjust`: Auto-adjust canary rollout percentage based on success evaluations.

## Canary A/B Testing
- **Service File**: `src/prompt_improver/services/canary_testing.py`
- **Class**: `CanaryTestingService`
- **Functionality**:
  - Reads config from `redis_config.yaml`.
  - Determines feature activation per user/session.
  - Stores metrics on requests, success/fail rates, response times, cache hits/misses.
  - Aggregates metrics and evaluates success/rollback criteria.
  - Auto-adjusts rollout percentage.

## Rollout Plan
- **Initial Phase**: Low percentage rollout to limited users.
- **Success Evaluation**: Monitor cache performance, error deltas under 1%.
- **Responsive Adjustments**: Increase or rollback based on real-time analytics.

## Conclusion
This rollout strategy involving a feature flag and canary A/B testing facilitates a robust deployment mechanism, allowing incremental cache feature rollouts with measurable impact analysis. The continued evaluation ensures system reliability and optimizes feature delivery.

## References
- `redis_config.yaml` for Redis configuration details.
- CLI commands in `src/prompt_improver/cli.py` for managing cache and canary testing.
