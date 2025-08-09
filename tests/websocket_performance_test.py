"""
WebSocket 15.x Performance Test

Tests performance improvements and load handling capabilities of websockets 15.x
"""
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
project_root = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(project_root))
try:
    import coredis
    import websockets
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed
except ImportError as e:
    print(f'Missing required dependencies: {e}')
    sys.exit(1)

@dataclass
class PerformanceMetrics:
    """Performance test results"""
    test_name: str
    duration: float
    messages_sent: int
    messages_received: int
    messages_per_second: float
    avg_latency: float
    min_latency: float
    max_latency: float
    success_rate: float
    errors: list[str]
    memory_usage_mb: float = 0.0

class WebSocketPerformanceTester:
    """WebSocket performance testing suite"""

    def __init__(self, base_url: str='ws://localhost:8765'):
        self.base_url = base_url
        self.results: list[PerformanceMetrics] = []

    async def create_echo_server(self, port: int=8765):
        """Create a simple echo server for testing"""

        async def echo_handler(websocket, path):
            print(f'Client connected: {websocket.remote_address}')
            try:
                async for message in websocket:
                    response = {'original': json.loads(message), 'server_timestamp': time.time(), 'echo': True}
                    await websocket.send(json.dumps(response))
            except ConnectionClosed:
                print(f'Client disconnected: {websocket.remote_address}')
            except Exception as e:
                print(f'Error in echo handler: {e}')
        return await websockets.serve(echo_handler, 'localhost', port)

    async def test_message_throughput(self, num_messages: int=1000, concurrent_connections: int=1):
        """Test message throughput performance"""
        print(f'Testing message throughput: {num_messages} messages, {concurrent_connections} connections')

        async def single_connection_test(connection_id: int):
            latencies = []
            messages_sent = 0
            messages_received = 0
            errors = []
            try:
                async with connect(f'{self.base_url}') as websocket:
                    for i in range(num_messages // concurrent_connections):
                        try:
                            send_time = time.time()
                            message = {'connection_id': connection_id, 'sequence': i, 'timestamp': send_time, 'data': f'Performance test message {i} from connection {connection_id}'}
                            await websocket.send(json.dumps(message))
                            messages_sent += 1
                            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                            receive_time = time.time()
                            messages_received += 1
                            latency = (receive_time - send_time) * 1000
                            latencies.append(latency)
                        except TimeoutError:
                            errors.append(f'Timeout on message {i}')
                        except Exception as e:
                            errors.append(f'Error on message {i}: {e!s}')
            except Exception as e:
                errors.append(f'Connection error: {e!s}')
            return {'connection_id': connection_id, 'messages_sent': messages_sent, 'messages_received': messages_received, 'latencies': latencies, 'errors': errors}
        start_time = time.time()
        tasks = [single_connection_test(i) for i in range(concurrent_connections)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        total_sent = sum((r.get('messages_sent', 0) for r in results if isinstance(r, dict)))
        total_received = sum((r.get('messages_received', 0) for r in results if isinstance(r, dict)))
        all_latencies = []
        all_errors = []
        for r in results:
            if isinstance(r, dict):
                all_latencies.extend(r.get('latencies', []))
                all_errors.extend(r.get('errors', []))
            else:
                all_errors.append(f'Task exception: {r!s}')
        metrics = PerformanceMetrics(test_name=f'throughput_{concurrent_connections}conn_{num_messages}msg', duration=duration, messages_sent=total_sent, messages_received=total_received, messages_per_second=total_received / duration if duration > 0 else 0, avg_latency=statistics.mean(all_latencies) if all_latencies else 0, min_latency=min(all_latencies) if all_latencies else 0, max_latency=max(all_latencies) if all_latencies else 0, success_rate=total_received / total_sent * 100 if total_sent > 0 else 0, errors=all_errors)
        self.results.append(metrics)
        return metrics

    async def test_connection_stability(self, duration_seconds: int=30):
        """Test connection stability over time"""
        print(f'Testing connection stability for {duration_seconds} seconds')
        messages_sent = 0
        messages_received = 0
        errors = []
        latencies = []
        start_time = time.time()
        try:
            async with connect(f'{self.base_url}') as websocket:
                end_time = start_time + duration_seconds
                while time.time() < end_time:
                    try:
                        send_time = time.time()
                        message = {'type': 'stability_test', 'timestamp': send_time, 'sequence': messages_sent}
                        await websocket.send(json.dumps(message))
                        messages_sent += 1
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        receive_time = time.time()
                        messages_received += 1
                        latency = (receive_time - send_time) * 1000
                        latencies.append(latency)
                        await asyncio.sleep(0.1)
                    except TimeoutError:
                        errors.append(f'Timeout at {time.time() - start_time:.1f}s')
                    except Exception as e:
                        errors.append(f'Error at {time.time() - start_time:.1f}s: {e!s}')
        except Exception as e:
            errors.append(f'Connection error: {e!s}')
        duration = time.time() - start_time
        metrics = PerformanceMetrics(test_name='connection_stability', duration=duration, messages_sent=messages_sent, messages_received=messages_received, messages_per_second=messages_received / duration if duration > 0 else 0, avg_latency=statistics.mean(latencies) if latencies else 0, min_latency=min(latencies) if latencies else 0, max_latency=max(latencies) if latencies else 0, success_rate=messages_received / messages_sent * 100 if messages_sent > 0 else 0, errors=errors)
        self.results.append(metrics)
        return metrics

    async def test_burst_performance(self, burst_size: int=100, num_bursts: int=10):
        """Test burst message performance"""
        print(f'Testing burst performance: {num_bursts} bursts of {burst_size} messages')
        messages_sent = 0
        messages_received = 0
        errors = []
        latencies = []
        burst_times = []
        start_time = time.time()
        try:
            async with connect(f'{self.base_url}') as websocket:
                for burst_num in range(num_bursts):
                    burst_start = time.time()
                    send_tasks = []
                    for i in range(burst_size):
                        message = {'type': 'burst_test', 'burst': burst_num, 'sequence': i, 'timestamp': time.time()}
                        send_tasks.append(websocket.send(json.dumps(message)))
                        messages_sent += 1
                    await asyncio.gather(*send_tasks)
                    for i in range(burst_size):
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                            messages_received += 1
                        except TimeoutError:
                            errors.append(f'Timeout in burst {burst_num}, message {i}')
                        except Exception as e:
                            errors.append(f'Error in burst {burst_num}, message {i}: {e!s}')
                    burst_duration = time.time() - burst_start
                    burst_times.append(burst_duration)
                    await asyncio.sleep(0.1)
        except Exception as e:
            errors.append(f'Burst test error: {e!s}')
        duration = time.time() - start_time
        metrics = PerformanceMetrics(test_name='burst_performance', duration=duration, messages_sent=messages_sent, messages_received=messages_received, messages_per_second=messages_received / duration if duration > 0 else 0, avg_latency=statistics.mean(burst_times) * 1000 if burst_times else 0, min_latency=min(burst_times) * 1000 if burst_times else 0, max_latency=max(burst_times) * 1000 if burst_times else 0, success_rate=messages_received / messages_sent * 100 if messages_sent > 0 else 0, errors=errors)
        self.results.append(metrics)
        return metrics

    async def test_large_message_performance(self, message_sizes: list[int]=None):
        """Test performance with different message sizes"""
        if message_sizes is None:
            message_sizes = [1024, 10240, 102400, 1048576]
        print(f'Testing large message performance with sizes: {message_sizes}')
        for size in message_sizes:
            messages_sent = 0
            messages_received = 0
            errors = []
            latencies = []
            payload = 'x' * (size - 100)
            start_time = time.time()
            try:
                async with connect(f'{self.base_url}') as websocket:
                    for i in range(10):
                        try:
                            send_time = time.time()
                            message = {'type': 'large_message_test', 'size': size, 'sequence': i, 'payload': payload, 'timestamp': send_time}
                            await websocket.send(json.dumps(message))
                            messages_sent += 1
                            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                            receive_time = time.time()
                            messages_received += 1
                            latency = (receive_time - send_time) * 1000
                            latencies.append(latency)
                        except TimeoutError:
                            errors.append(f'Timeout on large message {i} (size: {size})')
                        except Exception as e:
                            errors.append(f'Error on large message {i} (size: {size}): {e!s}')
            except Exception as e:
                errors.append(f'Large message test error (size: {size}): {e!s}')
            duration = time.time() - start_time
            metrics = PerformanceMetrics(test_name=f'large_message_{size}_bytes', duration=duration, messages_sent=messages_sent, messages_received=messages_received, messages_per_second=messages_received / duration if duration > 0 else 0, avg_latency=statistics.mean(latencies) if latencies else 0, min_latency=min(latencies) if latencies else 0, max_latency=max(latencies) if latencies else 0, success_rate=messages_received / messages_sent * 100 if messages_sent > 0 else 0, errors=errors)
            self.results.append(metrics)

    def print_results(self):
        """Print performance test results"""
        print('\n' + '=' * 80)
        print('WEBSOCKET 15.X PERFORMANCE TEST RESULTS')
        print('=' * 80)
        for metrics in self.results:
            print(f'\n📊 {metrics.test_name.upper()}')
            print('-' * 60)
            print(f'Duration: {metrics.duration:.2f}s')
            print(f'Messages Sent: {metrics.messages_sent}')
            print(f'Messages Received: {metrics.messages_received}')
            print(f'Throughput: {metrics.messages_per_second:.1f} msg/sec')
            print(f'Success Rate: {metrics.success_rate:.1f}%')
            print(f'Average Latency: {metrics.avg_latency:.2f}ms')
            print(f'Min Latency: {metrics.min_latency:.2f}ms')
            print(f'Max Latency: {metrics.max_latency:.2f}ms')
            if metrics.errors:
                print(f'Errors ({len(metrics.errors)}):')
                for error in metrics.errors[:5]:
                    print(f'  - {error}')
                if len(metrics.errors) > 5:
                    print(f'  ... and {len(metrics.errors) - 5} more errors')

    def save_results(self, filename: str='websocket_performance_results.json'):
        """Save results to JSON file"""
        results_data = {'websocket_version': websockets.__version__, 'timestamp': time.time(), 'results': [{'test_name': m.test_name, 'duration': m.duration, 'messages_sent': m.messages_sent, 'messages_received': m.messages_received, 'messages_per_second': m.messages_per_second, 'avg_latency': m.avg_latency, 'min_latency': m.min_latency, 'max_latency': m.max_latency, 'success_rate': m.success_rate, 'error_count': len(m.errors), 'errors': m.errors[:10]} for m in self.results]}
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f'\n💾 Results saved to: {filename}')

async def main():
    """Main performance test runner"""
    print('WebSocket 15.x Performance Test Suite')
    print('=====================================')
    print(f'WebSocket version: {websockets.__version__}')
    tester = WebSocketPerformanceTester()
    print('Starting echo server...')
    server = await tester.create_echo_server(8765)
    try:
        await asyncio.sleep(1)
        print('\n🚀 Running performance tests...')
        await tester.test_message_throughput(num_messages=500, concurrent_connections=1)
        await tester.test_message_throughput(num_messages=1000, concurrent_connections=5)
        await tester.test_connection_stability(duration_seconds=15)
        await tester.test_burst_performance(burst_size=50, num_bursts=5)
        await tester.test_large_message_performance([1024, 10240, 102400])
        tester.print_results()
        tester.save_results('websocket_15x_performance_results.json')
    finally:
        server.close()
        await server.wait_closed()
        print('\n🛑 Echo server stopped')
if __name__ == '__main__':
    asyncio.run(main())
