# What Are the Impacts of the Redis Expiration Algorithm?

[Redis CE and Stack](https://redis.io/kb/public?cat=2o9qogwfm6)

[Redis Enterprise Software](https://redis.io/kb/public?cat=1cmzkcxg2e)

[Redis Cloud](https://redis.io/kb/public?cat=1x5z2h4q4d)

[memory](https://redis.io/kb/public?tag=memory)

Last updated 22, Mar 2024

## Goal

The capability to expire keys is a common feature of many caching technologies. The mechanism is simple: every key that at some point needs to be expired has a Time-To-Live (TTL) associated, and the database needs to detect those keys that are expired and remove them, adopting the least expensive strategy possible, which is not a trivial problem.

In this document, the expiration algorithm is detailed together with the tradeoffs to minimize memory consumption and CPU usage, thus maximizing the free space available and performance.

## Solution

Keys in a Redis database are stored in a hash table (the keyspace), and they point to the related data structure (Sets, Lists, Hashes, etc.). We don't save the TTL value in the key, because in those cases where the expiration is not required (especially when Redis is not used as a cache), there would be a memory overhead. Because of this, the TTL for volatile keys is stored in a **secondary hash table**, usually smaller than the main dictionary. This secondary hash table stores the pointer to the key, and the key points to the TTL value. The expiration mechanism works as follows:

- **Passive expiration**. When a key is accessed e.g. GET key, Redis checks if the key exists and if it is expired. If expired, the key is removed and `nil` returned. However, this is not enough, because there may exist keys that are no longer accessed by clients. So the second strategy addresses this situation.
- **Active expiration**. This happens by random sampling. Keys are not sorted by expiration time and the strategy to find a suitable candidate for expiration is sampling the secondary hash table.

### Before Redis 6

The original approach, (before Redis 6) was simply to remove those keys sampled by an algorithm and with expired TTL. The problem with this approach is that as the loop of sampling and deleting the keys progresses, it reaches fewer keys having an expired TTL, which is resource-intensive and does not produce a relevant number of evictions, causing the random sampling to waste CPU cycles. Because of this, the algorithm would stop sampling when the good candidates for eviction fell below a configurable threshold, set to the default of 25%. Meaning that in the worst case, 25% of the keys may be logically expired but still using memory.

### Improvements in Redis 6

After prototyping and benchmarking alternative solutions, an improvement has been added to Redis 6 to **reduce the amount of hidden memory**, that is, the memory allocated by expired keys that the previous implementation would not deallocate. The improvement consists of the introduction of a [radix tree](https://en.wikipedia.org/wiki/Radix_tree) in addition to the secondary hash. In this revision of the expiration algorithm:

1. The sampling of the secondary hash table for expired keys still happens
2. The samples are compared to the keys already in the radix tree
3. If a sample is expired, it is deleted right away; but if a sample is **about to** expire (read further on), it is stored in the radix tree. It is a hint for the next sampling iteration, so the iteration begins from the radix tree which stores keys that are likely to expire.

With the introduction of the radix tree, the information about expiration times of the samples is capitalized and achieves a reduction of hidden memory allocated. This solution helps drop the 25% hidden keys without big design changes and also reduces the amount of used memory.

* * *

[Redis 6.0 RC1](https://raw.githubusercontent.com/antirez/redis/6.0/00-RELEASENOTES) Released Thu Dec 19 09:58:24 CEST 2019

```hljs csharp
* The Redis active expire cycle was rewritten for much faster eviction of keys
  that are already expired. Now the effort is tunable.

Copy code
```

### Recommendations

Refer to the Redis [configuration file](https://raw.githubusercontent.com/redis/redis/7.0/redis.conf) to learn more about the configuration of this algorithm using the parameter `active-expire-effort,` which configures the tolerance to the expired keys still present in the system. As a general rule, when sizing a system in which keys have a TTL set, hidden memory must be taken into account and memory pressure tested accordingly.

## References

- The evolution of the Redis key expiration algorithm was presented by Salvatore Sanfilippo at RedisDay in 2019. Watch the presentation [here](https://www.youtube.com/watch?v=SyQTG0hXPxY).
- Review the source code of the file [expire.c](https://github.com/redis/redis/blob/unstable/src/expire.c) which details the algorithm behind keys expiration.
