"""
Comprehensive WebSocket 15.x Integration Test

This script tests real WebSocket connections with the new WebSocket 15.x features:
- Automatic reconnection with improved heuristics
- Proxy support (if available)
- Enhanced connection stability
- Real data streaming to analytics endpoints
- FastAPI WebSocket integration
- Coredis + WebSocket integration

Requirements:
- websockets>=15.0.0
- FastAPI server running on localhost:8000
- Redis server running on localhost:6379
"""
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
project_root = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(project_root))
try:
    import aiohttp
    import coredis
    import websockets
    from fastapi.testclient import TestClient
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed
except ImportError as e:
    print(f'Missing required dependencies: {e}')
    print('Install with: pip install websockets aiohttp coredis')
    sys.exit(1)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class WebSocket15xTester:
    """Comprehensive tester for WebSocket 15.x features"""

    def __init__(self):
        self.base_url = 'ws://localhost:8000'
        self.redis_url = 'redis://localhost:6379'
        self.results: dict[str, Any] = {}
        self.connected_websockets: list = []

    async def test_websocket_version(self):
        """Test that we're using WebSocket 15.x"""
        logger.info('Testing WebSocket version...')
        version = websockets.__version__
        logger.info('WebSocket version: %s', version)
        major_version = int(version.split('.')[0])
        if major_version >= 15:
            self.results['version_check'] = {'status': 'PASS', 'version': version, 'message': f'Using WebSocket {version}'}
        else:
            self.results['version_check'] = {'status': 'FAIL', 'version': version, 'message': f'Expected WebSocket 15.x, got {version}'}

    async def test_basic_connection(self):
        """Test basic WebSocket connection"""
        logger.info('Testing basic WebSocket connection...')
        try:
            websocket_url = f'{self.base_url}/api/v1/experiments/real-time/live/test-exp-1'
            async with connect(websocket_url) as websocket:
                logger.info('Connected to %s', websocket_url)
                await websocket.send(json.dumps({'type': 'ping'}))
                logger.info('Sent ping message')
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                logger.info('Received response: %s', response_data)
                self.results['basic_connection'] = {'status': 'PASS', 'message': 'Basic connection successful', 'response': response_data}
        except Exception as e:
            logger.error('Basic connection failed: %s', e)
            self.results['basic_connection'] = {'status': 'FAIL', 'message': f'Connection failed: {e!s}'}

    async def test_automatic_reconnection(self):
        """Test automatic reconnection feature from WebSocket 15.x"""
        logger.info('Testing automatic reconnection...')
        try:
            websocket_url = f'{self.base_url}/api/v1/experiments/real-time/live/reconnect-test'
            messages_received = []
            reconnect_count = 0
            async for websocket in connect(websocket_url):
                try:
                    logger.info('Connected (attempt #%s)', reconnect_count + 1)
                    reconnect_count += 1
                    await websocket.send(json.dumps({'type': 'reconnect_test', 'attempt': reconnect_count, 'timestamp': datetime.now().isoformat()}))
                    for i in range(3):
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                            messages_received.append(json.loads(message))
                            logger.info('Received message {i + 1}: %s', message)
                        except TimeoutError:
                            logger.info('Timeout waiting for message %s', i + 1)
                            break
                    if reconnect_count == 1:
                        logger.info('Simulating connection drop...')
                        await websocket.close()
                        continue
                    break
                except ConnectionClosed:
                    logger.info('Connection closed, will attempt reconnect...')
                    if reconnect_count >= 3:
                        break
                    continue
            self.results['automatic_reconnection'] = {'status': 'PASS' if reconnect_count > 1 else 'FAIL', 'reconnect_count': reconnect_count, 'messages_received': len(messages_received), 'message': f'Reconnected {reconnect_count} times, received {len(messages_received)} messages'}
        except Exception as e:
            logger.error('Reconnection test failed: %s', e)
            self.results['automatic_reconnection'] = {'status': 'FAIL', 'message': f'Reconnection test failed: {e!s}'}

    async def test_message_streaming(self):
        """Test real-time message streaming"""
        logger.info('Testing message streaming...')
        try:
            websocket_url = f'{self.base_url}/api/v1/experiments/real-time/live/streaming-test'
            messages_received = []
            async with connect(websocket_url) as websocket:
                await websocket.send(json.dumps({'type': 'request_metrics'}))
                start_time = time.time()
                while time.time() - start_time < 5.0:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        messages_received.append(json.loads(message))
                        logger.info('Streamed message: %s', len(messages_received))
                    except TimeoutError:
                        continue
                self.results['message_streaming'] = {'status': 'PASS' if len(messages_received) > 0 else 'FAIL', 'messages_count': len(messages_received), 'message': f'Received {len(messages_received)} streaming messages'}
        except Exception as e:
            logger.error('Message streaming test failed: %s', e)
            self.results['message_streaming'] = {'status': 'FAIL', 'message': f'Streaming test failed: {e!s}'}

    async def test_concurrent_connections(self):
        """Test multiple concurrent WebSocket connections"""
        logger.info('Testing concurrent connections...')

        async def single_connection(connection_id: int):
            try:
                websocket_url = f'{self.base_url}/api/v1/experiments/real-time/live/concurrent-test-{connection_id}'
                async with connect(websocket_url) as websocket:
                    await websocket.send(json.dumps({'type': 'identify', 'connection_id': connection_id, 'timestamp': datetime.now().isoformat()}))
                    response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    return {'connection_id': connection_id, 'status': 'success', 'response': response}
            except Exception as e:
                return {'connection_id': connection_id, 'status': 'error', 'error': str(e)}
        try:
            tasks = [single_connection(i) for i in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful_connections = sum((1 for r in results if isinstance(r, dict) and r.get('status') == 'success'))
            self.results['concurrent_connections'] = {'status': 'PASS' if successful_connections >= 8 else 'FAIL', 'total_attempted': 10, 'successful': successful_connections, 'message': f'{successful_connections}/10 concurrent connections successful'}
        except Exception as e:
            logger.error('Concurrent connections test failed: %s', e)
            self.results['concurrent_connections'] = {'status': 'FAIL', 'message': f'Concurrent test failed: {e!s}'}

    async def test_redis_integration(self):
        """Test Redis + WebSocket integration using coredis"""
        logger.info('Testing Redis + WebSocket integration...')
        try:
            redis_client = coredis.Redis.from_url(self.redis_url, decode_responses=True)
            await redis_client.ping()
            logger.info('Redis connection successful')
            websocket_url = f'{self.base_url}/api/v1/experiments/real-time/live/redis-test'
            async with connect(websocket_url) as websocket:
                await websocket.send(json.dumps({'type': 'subscribe_alerts'}))
                channel = 'experiment:redis-test:updates'
                test_message = {'type': 'redis_test_message', 'data': 'Published via Redis', 'timestamp': datetime.now().isoformat()}
                await redis_client.publish(channel, json.dumps(test_message))
                logger.info('Published message to Redis channel: %s', channel)
                received_messages = []
                for _ in range(5):
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        received_messages.append(json.loads(message))
                        logger.info('Received via WebSocket: %s', message)
                    except TimeoutError:
                        break
                self.results['redis_integration'] = {'status': 'PASS' if len(received_messages) > 0 else 'FAIL', 'messages_received': len(received_messages), 'message': f'Redis integration test - received {len(received_messages)} messages'}
            await redis_client.aclose()
        except Exception as e:
            logger.error('Redis integration test failed: %s', e)
            self.results['redis_integration'] = {'status': 'FAIL', 'message': f'Redis integration failed: {e!s}'}

    async def test_websocket_performance(self):
        """Test WebSocket performance with high message throughput"""
        logger.info('Testing WebSocket performance...')
        try:
            websocket_url = f'{self.base_url}/api/v1/experiments/real-time/live/performance-test'
            async with connect(websocket_url) as websocket:
                messages_sent = 0
                messages_received = 0
                start_time = time.time()
                send_task = asyncio.create_task(self._send_performance_messages(websocket))
                try:
                    while time.time() - start_time < 10.0:
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        messages_received += 1
                        if not send_task.done():
                            messages_sent = getattr(send_task, 'messages_sent', 0)
                except TimeoutError:
                    pass
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass
                duration = time.time() - start_time
                throughput = messages_received / duration if duration > 0 else 0
                self.results['websocket_performance'] = {'status': 'PASS' if throughput > 10 else 'FAIL', 'duration': duration, 'messages_sent': messages_sent, 'messages_received': messages_received, 'throughput_per_sec': throughput, 'message': f'Performance: {throughput:.1f} messages/sec'}
        except Exception as e:
            logger.error('Performance test failed: %s', e)
            self.results['websocket_performance'] = {'status': 'FAIL', 'message': f'Performance test failed: {e!s}'}

    async def _send_performance_messages(self, websocket):
        """Helper to send messages rapidly for performance testing"""
        messages_sent = 0
        try:
            while True:
                await websocket.send(json.dumps({'type': 'performance_test', 'sequence': messages_sent, 'timestamp': time.time()}))
                messages_sent += 1
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        finally:
            self.send_performance_messages.messages_sent = messages_sent

    async def test_health_endpoint(self):
        """Test the health endpoint for real-time services"""
        logger.info('Testing health endpoint...')
        try:
            async with aiohttp.ClientSession() as session:
                health_url = 'http://localhost:8000/api/v1/experiments/real-time/health'
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        logger.info('Health check response: %s', health_data)
                        self.results['health_endpoint'] = {'status': 'PASS', 'health_data': health_data, 'message': 'Health endpoint responding correctly'}
                    else:
                        self.results['health_endpoint'] = {'status': 'FAIL', 'message': f'Health endpoint returned status {response.status}'}
        except Exception as e:
            logger.error('Health endpoint test failed: %s', e)
            self.results['health_endpoint'] = {'status': 'FAIL', 'message': f'Health endpoint test failed: {e!s}'}

    async def test_fastapi_websocket_compatibility(self):
        """Test FastAPI WebSocket compatibility with websockets 15.x"""
        logger.info('Testing FastAPI WebSocket compatibility...')
        try:
            websocket_url = f'{self.base_url}/api/v1/experiments/real-time/live/fastapi-compat-test'
            async with connect(websocket_url) as websocket:
                ping_message = {'type': 'ping', 'timestamp': '2025-07-25T10:00:00Z', 'client_type': 'fastapi_compat'}
                await websocket.send(json.dumps(ping_message))
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                echo_message = {'type': 'echo', 'data': 'FastAPI WebSocket 15.x compatibility test'}
                await websocket.send(json.dumps(echo_message))
                echo_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                echo_data = json.loads(echo_response)
                self.results['fastapi_websocket_compatibility'] = {'status': 'PASS', 'ping_response': response_data, 'echo_response': echo_data, 'message': 'FastAPI WebSocket compatibility confirmed'}
        except Exception as e:
            logger.error('FastAPI WebSocket compatibility test failed: %s', e)
            self.results['fastapi_websocket_compatibility'] = {'status': 'FAIL', 'message': f'FastAPI compatibility test failed: {e!s}'}

    async def run_all_tests(self):
        """Run all WebSocket tests"""
        logger.info('Starting comprehensive WebSocket 15.x integration tests...')
        tests = [self.test_websocket_version, self.test_health_endpoint, self.test_basic_connection, self.test_message_streaming, self.test_concurrent_connections, self.test_automatic_reconnection, self.test_redis_integration, self.test_websocket_performance, self.test_fastapi_websocket_compatibility]
        for test in tests:
            try:
                await test()
            except Exception as e:
                test_name = test.__name__
                logger.error('Test {test_name} failed with exception: %s', e)
                self.results[test_name] = {'status': 'ERROR', 'message': f'Test failed with exception: {e!s}'}
        for ws in self.connected_websockets:
            try:
                await ws.close()
            except:
                pass
        return self.results

    def print_results(self):
        """Print test results in a formatted way"""
        print('\n' + '=' * 80)
        print('WEBSOCKET 15.X INTEGRATION TEST RESULTS')
        print('=' * 80)
        passed = 0
        failed = 0
        errors = 0
        for test_name, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            message = result.get('message', 'No message')
            status_symbol = {'PASS': '✅', 'FAIL': '❌', 'ERROR': '💥'}.get(status, '❓')
            print(f'{status_symbol} {test_name.upper()}: {message}')
            if status == 'PASS':
                passed += 1
            elif status == 'FAIL':
                failed += 1
            elif status == 'ERROR':
                errors += 1
        print('\n' + '-' * 80)
        print(f'SUMMARY: {passed} passed, {failed} failed, {errors} errors')
        print('-' * 80)
        return (passed, failed, errors)

async def main():
    """Main test runner"""
    print('WebSocket 15.x Integration Test Suite')
    print('=====================================')
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/api/v1/experiments/real-time/health') as response:
                if response.status != 200:
                    print('❌ FastAPI server not responding at localhost:8000')
                    print('Please start the server first: uvicorn prompt_improver.api.main:app --reload')
                    return 1
    except aiohttp.ClientConnectorError:
        print('❌ Cannot connect to FastAPI server at localhost:8000')
        print('Please start the server first: uvicorn prompt_improver.api.main:app --reload')
        return 1
    tester = WebSocket15xTester()
    results = await tester.run_all_tests()
    passed, failed, errors = tester.print_results()
    results_file = Path(__file__).parent / 'websocket_15x_test_results.json'
    with open(results_file, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'websocket_version': websockets.__version__, 'results': results, 'summary': {'passed': passed, 'failed': failed, 'errors': errors, 'total': len(results)}}, f, indent=2)
    print(f'\nDetailed results saved to: {results_file}')
    return 0 if failed == 0 and errors == 0 else 1
if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
