# Connection pools and multiplexing

Manage Redis connections efficiently

Redis example code generally opens a connection, demonstrates
a command or feature, and then closes. Real-world code typically
has short bursts of communication with the server and periods of
inactivity in between. Opening and closing connections
involves some overhead and leads to inefficiency if you do
it frequently. This means that you can improve the performance of production
code by making as few separate connections as possible.

Managing connections in your own code can be tricky, so the Redis
client libraries give you some help. The two basic approaches to
connection management are called _connection pooling_ and _multiplexing_.
The [`redis-py`](https://redis.io/docs/latest/develop/clients/redis-py/),
[`jedis`](https://redis.io/docs/latest/develop/clients/jedis/), and
[`go-redis`](https://redis.io/docs/latest/develop/clients/go/) clients support
connection pooling, while
[`NRedisStack`](https://redis.io/docs/latest/develop/clients/dotnet/)
supports multiplexing.
[`Lettuce`](https://redis.io/docs/latest/develop/clients/lettuce/)
supports both approaches.

## Connection pooling

When you initialize a connection pool, the client opens a small number
of connections and adds them to the pool.

[![](https://redis.io/docs/latest/images/dev/connect/pool-and-mux/ConnPoolInit.drawio.svg)](https://redis.io/docs/latest/images/dev/connect/pool-and-mux/ConnPoolInit.drawio.svg)

Each time you "open" a connection
from the pool, the client returns one of these existing
connections and notes the fact that it is in use.

[![](https://redis.io/docs/latest/images/dev/connect/pool-and-mux/ConnPoolInUse.drawio.svg)](https://redis.io/docs/latest/images/dev/connect/pool-and-mux/ConnPoolInUse.drawio.svg)

When you later "close"
the connection, the client puts it back into the pool of available
connections without actually closing it.

[![](https://redis.io/docs/latest/images/dev/connect/pool-and-mux/ConnPoolDiscon.drawio.svg)](https://redis.io/docs/latest/images/dev/connect/pool-and-mux/ConnPoolDiscon.drawio.svg)

If all connections in the pool are in use but the app needs more, then
the client can simply open new connections as necessary. In this way, the client
eventually finds the right number of connections to satisfy your
app's demands.

## Multiplexing

Instead of pooling several connections, a multiplexer keeps a
single connection open and uses it for all traffic between the
client and the server. The "connections" returned to your code are
used to identify where to send the response data from your commands.

[![](https://redis.io/docs/latest/images/dev/connect/pool-and-mux/ConnMux.drawio.svg)](https://redis.io/docs/latest/images/dev/connect/pool-and-mux/ConnMux.drawio.svg)

Note that it is not a problem if the multiplexer receives several commands close
together in time. When this happens, the multiplexer can often combine the commands into a
[pipeline](https://redis.io/docs/latest/develop/using-commands/pipelining/), which
improves efficiency.

Multiplexing offers high efficiency but works transparently without requiring
any special code to enable it in your app. The main disadvantage of multiplexing compared to
connection pooling is that it can't support the blocking "pop" commands (such as
[`BLPOP`](https://redis.io/docs/latest/commands/blpop/)) since these would stall the
connection for all callers.

RATE THIS PAGE

★★★★★

[Back to top ↑](https://redis.io/docs/latest/develop/clients/pools-and-muxing/#)

Submit


## On this page
