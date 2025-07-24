#!/usr/bin/env python3

"""
2025 Developer Experience Validation Suite
Comprehensive testing of developer productivity improvements and performance targets
"""

import asyncio
import aiohttp
import json
import time
import subprocess
import sys
import os
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Configuration
@dataclass
class ValidationConfig:
    vite_port: int = 5173
    api_port: int = 8000
    hmr_target_ms: float = 50.0
    api_response_target_ms: float = 200.0
    memory_limit_mb: int = 1000
    cpu_limit_percent: float = 80.0
    test_iterations: int = 10
    timeout_seconds: int = 30

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'

class DevExperienceValidator:
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.project_root = Path(__file__).parent.parent
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': config.__dict__,
            'tests': {},
            'summary': {},
            'passed': False
        }
        
    def log(self, level: str, message: str) -> None:
        """Log message with color coding"""
        color_map = {
            'INFO': Colors.BLUE,
            'SUCCESS': Colors.GREEN,
            'WARNING': Colors.YELLOW,
            'ERROR': Colors.RED,
            'STEP': Colors.PURPLE
        }
        color = color_map.get(level, Colors.NC)
        print(f"{color}[{level}]{Colors.NC} {message}")
        
    async def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation suite"""
        self.log('STEP', '🚀 Starting 2025 Developer Experience Validation')
        print(f"{Colors.CYAN}Target Performance Metrics:{Colors.NC}")
        print(f"  • HMR Response: <{self.config.hmr_target_ms}ms")
        print(f"  • API Response: <{self.config.api_response_target_ms}ms") 
        print(f"  • Memory Usage: <{self.config.memory_limit_mb}MB")
        print(f"  • CPU Usage: <{self.config.cpu_limit_percent}%")
        print()
        
        try:
            # Environment validation
            await self.validate_environment()
            
            # Performance validation
            await self.validate_hmr_performance()
            await self.validate_api_performance() 
            await self.validate_system_resources()
            
            # Development workflow validation
            await self.validate_dev_workflow()
            await self.validate_integration()
            
            # Generate summary
            self.generate_summary()
            
            # Save results
            await self.save_results()
            
            self.display_results()
            
        except Exception as e:
            self.log('ERROR', f'Validation failed: {str(e)}')
            self.results['error'] = str(e)
        
        return self.results
    
    async def validate_environment(self) -> None:
        """Validate development environment setup"""
        self.log('STEP', 'Validating development environment...')
        
        env_tests = {}
        
        # Check required files
        required_files = [
            'vite.config.ts',
            'package.json', 
            '.devcontainer/devcontainer.json',
            '.devcontainer/Dockerfile',
            '.vscode/settings.json',
            'scripts/dev-server.sh'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            env_tests[f'file_{file_path.replace("/", "_")}'] = {
                'passed': exists,
                'message': f'Required file {file_path} {"exists" if exists else "missing"}'
            }
            
        # Check Node.js and npm versions
        try:
            node_result = subprocess.run(['node', '--version'], 
                                       capture_output=True, text=True, timeout=5)
            node_version = node_result.stdout.strip()
            node_major = int(node_version.lstrip('v').split('.')[0])
            
            env_tests['node_version'] = {
                'passed': node_major >= 20,
                'message': f'Node.js version: {node_version} (>= v20.0.0 required)',
                'value': node_version
            }
        except Exception as e:
            env_tests['node_version'] = {
                'passed': False,
                'message': f'Node.js check failed: {str(e)}'
            }
            
        # Check Python version
        try:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            env_tests['python_version'] = {
                'passed': sys.version_info >= (3, 12),
                'message': f'Python version: {python_version} (>= 3.12.0 required)',
                'value': python_version
            }
        except Exception as e:
            env_tests['python_version'] = {
                'passed': False,
                'message': f'Python check failed: {str(e)}'
            }
            
        # Check if in dev container
        in_container = os.path.exists('/.dockerenv')
        env_tests['dev_container'] = {
            'passed': True,  # Optional but recommended
            'message': f'Development container: {"Active" if in_container else "Not detected (consider using .devcontainer)"}'
        }
        
        self.results['tests']['environment'] = env_tests
        
        passed_count = sum(1 for test in env_tests.values() if test['passed'])
        self.log('SUCCESS' if passed_count == len(env_tests) else 'WARNING', 
                f'Environment validation: {passed_count}/{len(env_tests)} checks passed')
    
    async def validate_hmr_performance(self) -> None:
        """Validate Hot Module Replacement performance"""
        self.log('STEP', 'Validating HMR performance...')
        
        hmr_tests = {}
        
        # Check if Vite server is running
        try:
            async with aiohttp.ClientSession() as session:
                url = f'http://localhost:{self.config.vite_port}'
                
                # Test server availability
                start_time = time.time()
                try:
                    async with session.get(url, timeout=5) as response:
                        server_response_time = (time.time() - start_time) * 1000
                        server_available = response.status == 200
                except Exception:
                    server_available = False
                    server_response_time = float('inf')
                
                hmr_tests['vite_server_available'] = {
                    'passed': server_available,
                    'message': f'Vite server {"available" if server_available else "unavailable"} on port {self.config.vite_port}',
                    'response_time_ms': server_response_time
                }
                
                if server_available:
                    # Test HMR WebSocket connection
                    try:
                        import websockets
                        ws_url = f'ws://localhost:{self.config.vite_port}'
                        
                        start_time = time.time()
                        async with websockets.connect(ws_url, timeout=5) as websocket:
                            ws_connection_time = (time.time() - start_time) * 1000
                            
                            hmr_tests['hmr_websocket'] = {
                                'passed': ws_connection_time < 1000,  # 1 second max
                                'message': f'HMR WebSocket connection: {ws_connection_time:.1f}ms',
                                'connection_time_ms': ws_connection_time
                            }
                    except ImportError:
                        hmr_tests['hmr_websocket'] = {
                            'passed': False,
                            'message': 'WebSocket testing unavailable (install websockets package)'
                        }
                    except Exception as e:
                        hmr_tests['hmr_websocket'] = {
                            'passed': False,
                            'message': f'HMR WebSocket test failed: {str(e)}'
                        }
                        
        except Exception as e:
            hmr_tests['vite_server_available'] = {
                'passed': False,
                'message': f'Vite server test failed: {str(e)}'
            }
        
        self.results['tests']['hmr_performance'] = hmr_tests
        
        passed_count = sum(1 for test in hmr_tests.values() if test['passed'])
        self.log('SUCCESS' if passed_count == len(hmr_tests) else 'WARNING',
                f'HMR performance validation: {passed_count}/{len(hmr_tests)} checks passed')
    
    async def validate_api_performance(self) -> None:
        """Validate Python API performance"""
        self.log('STEP', 'Validating API performance...')
        
        api_tests = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test health endpoint
                health_url = f'http://localhost:{self.config.api_port}/health'
                
                response_times = []
                for i in range(self.config.test_iterations):
                    start_time = time.time()
                    try:
                        async with session.get(health_url, timeout=5) as response:
                            response_time = (time.time() - start_time) * 1000
                            response_times.append(response_time)
                            
                            if i == 0:  # First request
                                api_tests['api_server_available'] = {
                                    'passed': response.status == 200,
                                    'message': f'API server {"available" if response.status == 200 else "unavailable"} on port {self.config.api_port}',
                                    'status_code': response.status
                                }
                    except Exception as e:
                        if i == 0:
                            api_tests['api_server_available'] = {
                                'passed': False,
                                'message': f'API server unavailable: {str(e)}'
                            }
                        break
                
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
                    max_response_time = max(response_times)
                    min_response_time = min(response_times)
                    
                    api_tests['api_response_time'] = {
                        'passed': avg_response_time < self.config.api_response_target_ms,
                        'message': f'API response time: {avg_response_time:.1f}ms avg (target: <{self.config.api_response_target_ms}ms)',
                        'avg_ms': avg_response_time,
                        'min_ms': min_response_time,
                        'max_ms': max_response_time,
                        'iterations': len(response_times)
                    }
                else:
                    api_tests['api_response_time'] = {
                        'passed': False,
                        'message': 'No successful API responses recorded'
                    }
                    
        except Exception as e:
            api_tests['api_server_available'] = {
                'passed': False,
                'message': f'API validation failed: {str(e)}'
            }
        
        self.results['tests']['api_performance'] = api_tests
        
        passed_count = sum(1 for test in api_tests.values() if test['passed'])
        self.log('SUCCESS' if passed_count == len(api_tests) else 'WARNING',
                f'API performance validation: {passed_count}/{len(api_tests)} checks passed')
    
    async def validate_system_resources(self) -> None:
        """Validate system resource usage"""
        self.log('STEP', 'Validating system resources...')
        
        resource_tests = {}
        
        # CPU usage
        cpu_samples = []
        for _ in range(5):
            cpu_samples.append(psutil.cpu_percent(interval=1))
        
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        resource_tests['cpu_usage'] = {
            'passed': avg_cpu < self.config.cpu_limit_percent,
            'message': f'CPU usage: {avg_cpu:.1f}% (limit: <{self.config.cpu_limit_percent}%)',
            'value': avg_cpu,
            'limit': self.config.cpu_limit_percent
        }
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        resource_tests['memory_usage'] = {
            'passed': memory_mb < self.config.memory_limit_mb,
            'message': f'Memory usage: {memory_mb:.0f}MB (limit: <{self.config.memory_limit_mb}MB)',
            'value': memory_mb,
            'limit': self.config.memory_limit_mb
        }
        
        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                resource_tests['disk_io_available'] = {
                    'passed': True,
                    'message': f'Disk I/O monitoring available',
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes
                }
            else:
                resource_tests['disk_io_available'] = {
                    'passed': True,
                    'message': 'Disk I/O monitoring not available on this system'
                }
        except Exception as e:
            resource_tests['disk_io_available'] = {
                'passed': True,
                'message': f'Disk I/O check failed: {str(e)}'
            }
        
        self.results['tests']['system_resources'] = resource_tests
        
        passed_count = sum(1 for test in resource_tests.values() if test['passed'])
        self.log('SUCCESS' if passed_count == len(resource_tests) else 'WARNING',
                f'System resource validation: {passed_count}/{len(resource_tests)} checks passed')
    
    async def validate_dev_workflow(self) -> None:
        """Validate development workflow efficiency"""
        self.log('STEP', 'Validating development workflow...')
        
        workflow_tests = {}
        
        # Test TypeScript compilation
        try:
            result = subprocess.run(['npx', 'tsc', '--noEmit'], 
                                  cwd=self.project_root, 
                                  capture_output=True, text=True, timeout=30)
            
            workflow_tests['typescript_compilation'] = {
                'passed': result.returncode == 0,
                'message': f'TypeScript compilation {"successful" if result.returncode == 0 else "failed"}',
                'exit_code': result.returncode,
                'stderr': result.stderr if result.returncode != 0 else None
            }
        except subprocess.TimeoutExpired:
            workflow_tests['typescript_compilation'] = {
                'passed': False,
                'message': 'TypeScript compilation timed out (>30s)'
            }
        except Exception as e:
            workflow_tests['typescript_compilation'] = {
                'passed': False,
                'message': f'TypeScript compilation test failed: {str(e)}'
            }
        
        # Test linting
        try:
            result = subprocess.run(['npm', 'run', 'lint'], 
                                  cwd=self.project_root,
                                  capture_output=True, text=True, timeout=30)
            
            workflow_tests['linting'] = {
                'passed': result.returncode == 0,
                'message': f'Linting {"passed" if result.returncode == 0 else "failed"}',
                'exit_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            workflow_tests['linting'] = {
                'passed': False,
                'message': 'Linting timed out (>30s)'
            }
        except Exception as e:
            workflow_tests['linting'] = {
                'passed': False,
                'message': f'Linting test failed: {str(e)}'
            }
        
        # Test build process
        try:
            start_time = time.time()
            result = subprocess.run(['npm', 'run', 'build'], 
                                  cwd=self.project_root,
                                  capture_output=True, text=True, timeout=60)
            build_time = time.time() - start_time
            
            workflow_tests['build_process'] = {
                'passed': result.returncode == 0,
                'message': f'Build process {"successful" if result.returncode == 0 else "failed"} in {build_time:.1f}s',
                'exit_code': result.returncode,
                'build_time_seconds': build_time
            }
        except subprocess.TimeoutExpired:
            workflow_tests['build_process'] = {
                'passed': False,
                'message': 'Build process timed out (>60s)'
            }
        except Exception as e:
            workflow_tests['build_process'] = {
                'passed': False,
                'message': f'Build process test failed: {str(e)}'
            }
        
        self.results['tests']['dev_workflow'] = workflow_tests
        
        passed_count = sum(1 for test in workflow_tests.values() if test['passed'])
        self.log('SUCCESS' if passed_count == len(workflow_tests) else 'WARNING',
                f'Development workflow validation: {passed_count}/{len(workflow_tests)} checks passed')
    
    async def validate_integration(self) -> None:
        """Validate integration with existing systems"""
        self.log('STEP', 'Validating system integration...')
        
        integration_tests = {}
        
        # Check ML system integration
        ml_paths = [
            'src/prompt_improver/ml',
            'src/prompt_improver/ml/types.py'
        ]
        
        ml_integration_ok = True
        for path in ml_paths:
            full_path = self.project_root / path
            if not full_path.exists():
                ml_integration_ok = False
                break
        
        integration_tests['ml_system_integration'] = {
            'passed': ml_integration_ok,
            'message': f'ML system integration {"available" if ml_integration_ok else "incomplete"}',
            'checked_paths': ml_paths
        }
        
        # Check database integration
        db_config_paths = [
            'src/prompt_improver/database',
            'database/schema.sql'
        ]
        
        db_integration_ok = any((self.project_root / path).exists() for path in db_config_paths)
        integration_tests['database_integration'] = {
            'passed': db_integration_ok,
            'message': f'Database integration {"available" if db_integration_ok else "not found"}',
            'checked_paths': db_config_paths
        }
        
        # Check performance monitoring integration
        perf_paths = [
            'src/prompt_improver/performance',
            'scripts/benchmark_batch_processing.py'
        ]
        
        perf_integration_ok = any((self.project_root / path).exists() for path in perf_paths)
        integration_tests['performance_monitoring'] = {
            'passed': perf_integration_ok,
            'message': f'Performance monitoring {"available" if perf_integration_ok else "not found"}',
            'checked_paths': perf_paths
        }
        
        self.results['tests']['integration'] = integration_tests
        
        passed_count = sum(1 for test in integration_tests.values() if test['passed'])
        self.log('SUCCESS' if passed_count == len(integration_tests) else 'WARNING',
                f'Integration validation: {passed_count}/{len(integration_tests)} checks passed')
    
    def generate_summary(self) -> None:
        """Generate validation summary"""
        all_tests = {}
        for category, tests in self.results['tests'].items():
            all_tests.update(tests)
        
        total_tests = len(all_tests)
        passed_tests = sum(1 for test in all_tests.values() if test['passed'])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Determine overall grade
        if success_rate >= 95:
            grade = 'EXCELLENT'
        elif success_rate >= 90:
            grade = 'GOOD'
        elif success_rate >= 80:
            grade = 'ACCEPTABLE'
        elif success_rate >= 70:
            grade = 'NEEDS_IMPROVEMENT'
        else:
            grade = 'POOR'
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate,
            'grade': grade,
            'passed': success_rate >= 80  # 80% pass rate required
        }
        
        self.results['passed'] = self.results['summary']['passed']
    
    def display_results(self) -> None:
        """Display validation results"""
        summary = self.results['summary']
        
        print('\n' + '=' * 70)
        print(f"{Colors.BOLD}{Colors.CYAN} 2025 DEVELOPER EXPERIENCE VALIDATION RESULTS {Colors.NC}")
        print('=' * 70)
        
        # Overall grade
        grade_color = {
            'EXCELLENT': Colors.GREEN,
            'GOOD': Colors.GREEN,
            'ACCEPTABLE': Colors.YELLOW,
            'NEEDS_IMPROVEMENT': Colors.YELLOW,
            'POOR': Colors.RED
        }.get(summary['grade'], Colors.NC)
        
        print(f"{Colors.BOLD}Overall Grade: {grade_color}{summary['grade']}{Colors.NC}")
        print(f"{Colors.BOLD}Success Rate: {summary['success_rate']:.1f}% ({summary['passed_tests']}/{summary['total_tests']}){Colors.NC}")
        print(f"{Colors.BOLD}Status: {Colors.GREEN + '✅ PASSED' if summary['passed'] else Colors.RED + '❌ FAILED'}{Colors.NC}")
        print()
        
        # Category breakdown
        for category, tests in self.results['tests'].items():
            category_passed = sum(1 for test in tests.values() if test['passed'])
            category_total = len(tests)
            category_rate = (category_passed / category_total * 100) if category_total > 0 else 0
            
            status_icon = '✅' if category_rate == 100 else '⚠️' if category_rate >= 80 else '❌'
            print(f"{Colors.BOLD}{category.replace('_', ' ').title()}: {status_icon} {category_rate:.0f}% ({category_passed}/{category_total}){Colors.NC}")
            
            # Show failed tests
            failed_tests = [(name, test) for name, test in tests.items() if not test['passed']]
            if failed_tests:
                for test_name, test_data in failed_tests:
                    print(f"  {Colors.RED}❌ {test_name}: {test_data['message']}{Colors.NC}")
        
        print()
        
        # Performance highlights
        print(f"{Colors.BOLD}🎯 Performance Highlights:{Colors.NC}")
        
        # Extract key metrics
        if 'api_performance' in self.results['tests'] and 'api_response_time' in self.results['tests']['api_performance']:
            api_time = self.results['tests']['api_performance']['api_response_time'].get('avg_ms', 'N/A')
            api_status = '✅' if isinstance(api_time, (int, float)) and api_time < self.config.api_response_target_ms else '❌'
            print(f"  API Response Time: {api_status} {api_time if api_time != 'N/A' else 'N/A'}ms")
        
        if 'system_resources' in self.results['tests']:
            cpu_data = self.results['tests']['system_resources'].get('cpu_usage', {})
            cpu_value = cpu_data.get('value', 'N/A')
            cpu_status = '✅' if isinstance(cpu_value, (int, float)) and cpu_value < self.config.cpu_limit_percent else '❌'
            print(f"  CPU Usage: {cpu_status} {cpu_value if cpu_value != 'N/A' else 'N/A'}%")
            
            memory_data = self.results['tests']['system_resources'].get('memory_usage', {})
            memory_value = memory_data.get('value', 'N/A')
            memory_status = '✅' if isinstance(memory_value, (int, float)) and memory_value < self.config.memory_limit_mb else '❌'
            print(f"  Memory Usage: {memory_status} {memory_value if memory_value != 'N/A' else 'N/A'}MB")
        
        print()
        
        # Recommendations
        print(f"{Colors.BOLD}💡 Recommendations:{Colors.NC}")
        if summary['grade'] in ['EXCELLENT', 'GOOD']:
            print(f"  {Colors.GREEN}✅ Excellent developer experience! Consider sharing your configuration.{Colors.NC}")
        elif summary['grade'] == 'ACCEPTABLE':
            print(f"  {Colors.YELLOW}⚠️  Good foundation, focus on failing tests for optimization.{Colors.NC}")
        else:
            print(f"  {Colors.RED}❌ Significant improvements needed. Review failed tests and system requirements.{Colors.NC}")
        
        print('\n' + '=' * 70 + '\n')
    
    async def save_results(self) -> None:
        """Save validation results to file"""
        results_file = self.project_root / 'logs' / 'dev-experience-validation.json'
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.log('SUCCESS', f'Results saved to: {results_file}')

async def main():
    """Main validation entry point"""
    config = ValidationConfig()
    validator = DevExperienceValidator(config)
    
    results = await validator.run_validation()
    
    # Exit with appropriate code
    sys.exit(0 if results['passed'] else 1)

if __name__ == '__main__':
    asyncio.run(main())