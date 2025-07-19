# Redis Pattern-Caching Architecture Design

## Key Schema
- **Format:** `apes:pattern:{hash(prompt_characteristics)}:{version}`
- **Purpose:** Ensures unique identification of each pattern cache using a hash of prompt characteristics. Versioning is appended to handle schema changes or updates.

## Data Format
- **Encoding:** MsgPack-encoded JSON
- **Reasoning:** This minimizes serialization overhead, providing faster data encoding/decoding compared to plain JSON.

## TTL Tiers
- **Stable Patterns:** 1 hour
- **Volatile Discovery Results:** 5 minutes
- **Rationale:** Differentiated TTLs maintain cache freshness while ensuring availability of stable patterns longer.

## Invalidation Hooks
- **Event:** Publish `pattern.invalidate` events on rule or parameter updates.
- **Mechanism:** Use Redis Pub/Sub to notify relevant systems of cache invalidation needs.

## Back-off Mutex
- **Implementation:** Singleflight Lua script
- **Purpose:** Prevents cache stampedes by ensuring only one process recomputes a result while others wait for the result to be cached.

## Sequence Diagram
1. **Cache Miss:**
   - Check Redis for `apes:pattern:{hash(prompt_characteristics)}`
   - If miss, run Singleflight Lua script
   - Load or compute pattern
   - MsgPack encode data and store in Redis with TTL
   - Other waiting requests use the updated cache

2. **Cache Invalidation:**
   - Rule/Parameter Update triggers `pattern.invalidate`
   - Invalidate relevant `apes:pattern:{hash(prompt_characteristics)}` keys

