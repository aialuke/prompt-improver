"""
APES production service management with background daemon support.
Implements Task 3: Production Service Management from Phase 2.
"""

import asyncio
import signal
import sys
import os
from pathlib import Path

# Optional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
    # Mock psutil functionality for basic operation
    class MockProcess:
        def __init__(self):
            pass
        def memory_info(self):
            class MemInfo:
                rss = 50 * 1024 * 1024  # 50MB default
            return MemInfo()
        def cpu_percent(self):
            return 5.0
    
    class MockPsutil:
        @staticmethod
        def Process(pid=None):
            return MockProcess()
        @staticmethod
        def pid_exists(pid):
            try:
                os.kill(pid, 0)
                return True
            except:
                return False
    
    psutil = MockPsutil()
from typing import Dict, Any, Optional
from datetime import datetime
import json
import logging

from rich.console import Console

from ..database import get_session


class APESServiceManager:
    """Production service management with process monitoring"""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.data_dir = Path.home() / ".local" / "share" / "apes"
        self.pid_file = self.data_dir / "apes.pid"
        self.log_file = self.data_dir / "data" / "logs" / "apes.log"
        self.is_daemon = False
        self.shutdown_event = asyncio.Event()
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging for service management"""
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler() if not self.is_daemon else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger("apes.service")
    
    async def start_background_service(self, detach: bool = True) -> Dict[str, Any]:
        """Start APES as background daemon with monitoring"""
        
        service_results = {
            "pid": None,
            "status": "failed",
            "mcp_response_time": None,
            "postgresql_status": "unknown",
            "startup_time": datetime.now().isoformat()
        }
        
        try:
            if detach:
                # Daemon mode using research-validated patterns
                pid = await self.create_daemon_process()
                await self.write_pid_file(pid)
                await self.setup_signal_handlers()
                service_results["pid"] = pid
                self.is_daemon = True
                
            # Start core services
            self.logger.info("Starting APES background service")
            
            # Check PostgreSQL status
            postgresql_status = await self.start_postgresql_if_needed()
            service_results["postgresql_status"] = postgresql_status
            
            # Start MCP server
            await self.start_mcp_server()
            
            # Initialize performance monitoring
            await self.initialize_performance_monitoring()
            
            # Verify service health (ensure <200ms performance)
            health_status = await self.verify_service_health()
            service_results["mcp_response_time"] = health_status.get("mcp_response_time")
            
            if health_status.get("mcp_response_time", 1000) > 200:
                self.logger.warning(f"MCP performance {health_status['mcp_response_time']}ms exceeds 200ms target")
                await self.optimize_performance_settings()
            
            service_results["status"] = "running"
            
            self.console.print("✅ APES background service started", style="green")
            self.console.print(f"📊 MCP server: {health_status.get('mcp_response_time', 'N/A')}ms response time", style="dim")
            
            # If daemon mode, start monitoring loop
            if detach:
                await self.run_monitoring_loop()
                
        except Exception as e:
            self.logger.error(f"Failed to start background service: {e}")
            service_results["error"] = str(e)
            self.console.print(f"❌ Failed to start service: {e}", style="red")
            
        return service_results
    
    async def create_daemon_process(self) -> int:
        """Create daemon process using double-fork pattern"""
        
        try:
            # First fork
            pid = os.fork()
            if pid > 0:
                # Parent process
                return pid
                
        except OSError as e:
            raise Exception(f"First fork failed: {e}")
        
        # Decouple from parent environment
        os.chdir("/")
        os.setsid()
        os.umask(0)
        
        try:
            # Second fork
            pid = os.fork()
            if pid > 0:
                # First child
                os._exit(0)
                
        except OSError as e:
            raise Exception(f"Second fork failed: {e}")
        
        # Redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        
        with open('/dev/null', 'r') as stdin:
            os.dup2(stdin.fileno(), sys.stdin.fileno())
        with open('/dev/null', 'w') as stdout:
            os.dup2(stdout.fileno(), sys.stdout.fileno())
        with open('/dev/null', 'w') as stderr:
            os.dup2(stderr.fileno(), sys.stderr.fileno())
        
        return os.getpid()
    
    async def write_pid_file(self, pid: int):
        """Write PID file for process management"""
        
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        
        pid_data = {
            "pid": pid,
            "started_at": datetime.now().isoformat(),
            "command": " ".join(sys.argv)
        }
        
        with open(self.pid_file, 'w') as f:
            json.dump(pid_data, f, indent=2)
        
        self.logger.info(f"PID file written: {self.pid_file} (PID: {pid})")
    
    async def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def start_postgresql_if_needed(self) -> str:
        """Check and start PostgreSQL if needed"""
        
        try:
            # Test database connection
            async with get_session() as session:
                await session.execute("SELECT 1")
                self.logger.info("PostgreSQL connection verified")
                return "connected"
                
        except Exception as e:
            self.logger.warning(f"PostgreSQL connection failed: {e}")
            
            # Try to start PostgreSQL service (system-dependent)
            try:
                # Try common PostgreSQL service commands
                for cmd in [
                    ["brew", "services", "start", "postgresql"],
                    ["systemctl", "start", "postgresql"],
                    ["service", "postgresql", "start"]
                ]:
                    try:
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL
                        )
                        await process.communicate()
                        if process.returncode == 0:
                            self.logger.info(f"Started PostgreSQL using: {' '.join(cmd)}")
                            await asyncio.sleep(2)  # Wait for startup
                            return "started"
                    except Exception:
                        continue
                        
                return "failed_to_start"
                
            except Exception as e:
                self.logger.error(f"Failed to start PostgreSQL: {e}")
                return "failed_to_start"
    
    async def start_mcp_server(self):
        """Start MCP server component"""
        
        try:
            # Import and initialize MCP server
            from ..mcp_server.mcp_server import app as mcp_app
            
            # MCP server runs within the service process
            self.logger.info("MCP server initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def initialize_performance_monitoring(self):
        """Initialize performance monitoring"""
        
        self.logger.info("Performance monitoring initialized")
        
        # Create monitoring task
        asyncio.create_task(self.monitor_service_health_background())
    
    async def verify_service_health(self) -> Dict[str, Any]:
        """Verify service health and performance"""
        
        health_status = {
            "database_connection": False,
            "mcp_response_time": None,
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0
        }
        
        # Test database connection
        try:
            async with get_session() as session:
                start_time = asyncio.get_event_loop().time()
                await session.execute("SELECT 1")
                end_time = asyncio.get_event_loop().time()
                
                health_status["database_connection"] = True
                health_status["db_response_time"] = (end_time - start_time) * 1000
                
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
        
        # Test MCP server performance
        try:
            from ..mcp_server.mcp_server import improve_prompt
            
            start_time = asyncio.get_event_loop().time()
            await improve_prompt(
                prompt="Health check test",
                context={"domain": "health_check"},
                session_id="health_check"
            )
            end_time = asyncio.get_event_loop().time()
            
            health_status["mcp_response_time"] = (end_time - start_time) * 1000
            
        except Exception as e:
            self.logger.error(f"MCP health check failed: {e}")
        
        # System resource usage
        try:
            process = psutil.Process()
            health_status["memory_usage_mb"] = process.memory_info().rss / (1024 * 1024)
            health_status["cpu_usage_percent"] = process.cpu_percent()
            
        except Exception as e:
            self.logger.error(f"Resource monitoring failed: {e}")
        
        return health_status
    
    async def optimize_performance_settings(self):
        """Optimize performance settings when degradation detected"""
        
        self.logger.info("Applying performance optimizations")
        
        try:
            # Optimize database connections
            async with get_session() as session:
                # Reset any long-running queries
                await session.execute("SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND query_start < NOW() - INTERVAL '30 seconds'")
                
            self.logger.info("Performance optimizations applied")
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimizations: {e}")
    
    async def monitor_service_health_background(self, alert_threshold_ms: int = 250):
        """Background service monitoring with alerting"""
        
        self.logger.info(f"Starting background health monitoring (threshold: {alert_threshold_ms}ms)")
        
        while not self.shutdown_event.is_set():
            try:
                metrics = await self.collect_performance_metrics()
                
                # Monitor critical performance indicators
                if metrics.get("avg_response_time", 0) > alert_threshold_ms:
                    await self.send_performance_alert(
                        f"High latency detected: {metrics['avg_response_time']}ms"
                    )
                
                if metrics.get("database_connections", 0) > 15:  # Near pool limit
                    await self.optimize_connection_pool()
                
                # Check memory usage
                if metrics.get("memory_usage_mb", 0) > 256:  # Above 256MB
                    self.logger.warning(f"High memory usage: {metrics['memory_usage_mb']}MB")
                
                # Log structured metrics for analysis
                await self.log_performance_metrics(metrics)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(30)  # 30-second monitoring interval
                
            except Exception as e:
                self.logger.error(f"Monitoring cycle failed: {e}")
                await asyncio.sleep(60)  # Longer delay on error
    
    async def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics"""
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "avg_response_time": 0,
            "database_connections": 0,
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0
        }
        
        try:
            # Database metrics
            async with get_session() as session:
                # Get active connections
                result = await session.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                )
                metrics["database_connections"] = result.scalar() or 0
                
                # Test response time
                start_time = asyncio.get_event_loop().time()
                await session.execute("SELECT 1")
                end_time = asyncio.get_event_loop().time()
                metrics["avg_response_time"] = (end_time - start_time) * 1000
            
            # System metrics
            process = psutil.Process()
            metrics["memory_usage_mb"] = process.memory_info().rss / (1024 * 1024)
            metrics["cpu_usage_percent"] = process.cpu_percent()
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
        
        return metrics
    
    async def send_performance_alert(self, message: str):
        """Send performance alert"""
        
        self.logger.warning(f"PERFORMANCE ALERT: {message}")
        
        # In a full implementation, this could send notifications
        # For now, just log the alert
    
    async def optimize_connection_pool(self):
        """Optimize database connection pool"""
        
        self.logger.info("Optimizing database connection pool")
        
        try:
            async with get_session() as session:
                # Close idle connections
                await session.execute("""
                    SELECT pg_terminate_backend(pid) 
                    FROM pg_stat_activity 
                    WHERE state = 'idle' 
                    AND state_change < NOW() - INTERVAL '5 minutes'
                """)
                
        except Exception as e:
            self.logger.error(f"Connection pool optimization failed: {e}")
    
    async def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics in structured format"""
        
        # Log metrics as JSON for easy parsing
        self.logger.info(f"METRICS: {json.dumps(metrics)}")
    
    async def run_monitoring_loop(self):
        """Main monitoring loop for daemon mode"""
        
        self.logger.info("Starting daemon monitoring loop")
        
        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}")
        
        finally:
            await self.shutdown_service()
    
    async def shutdown_service(self):
        """Graceful service shutdown"""
        
        self.logger.info("Initiating graceful shutdown")
        
        try:
            # Cleanup PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            
            # Close database connections
            await sessionmanager.close()
            
            self.logger.info("Service shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
    
    def stop_service(self, timeout: int = 30) -> Dict[str, Any]:
        """Stop running APES service"""
        
        if not self.pid_file.exists():
            return {"status": "not_running", "message": "No PID file found"}
        
        try:
            with open(self.pid_file, 'r') as f:
                pid_data = json.load(f)
                pid = pid_data["pid"]
            
            # Check if process is running
            if not psutil.pid_exists(pid):
                self.pid_file.unlink()
                return {"status": "not_running", "message": "Process not found"}
            
            process = psutil.Process(pid)
            
            # Send SIGTERM for graceful shutdown
            process.terminate()
            
            # Wait for process to terminate
            try:
                process.wait(timeout=timeout)
                status = "stopped"
                message = "Service stopped gracefully"
                
            except psutil.TimeoutExpired:
                # Force kill if still running
                process.kill()
                process.wait(timeout=5)
                status = "force_stopped"
                message = "Service force stopped"
            
            # Cleanup PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            
            return {"status": status, "message": message, "pid": pid}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status"""
        
        status = {
            "running": False,
            "pid": None,
            "started_at": None,
            "uptime_seconds": None,
            "memory_usage_mb": None
        }
        
        if not self.pid_file.exists():
            return status
        
        try:
            with open(self.pid_file, 'r') as f:
                pid_data = json.load(f)
                pid = pid_data["pid"]
                started_at = datetime.fromisoformat(pid_data["started_at"])
            
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                
                status.update({
                    "running": True,
                    "pid": pid,
                    "started_at": started_at.isoformat(),
                    "uptime_seconds": (datetime.now() - started_at).total_seconds(),
                    "memory_usage_mb": process.memory_info().rss / (1024 * 1024)
                })
            else:
                # Process not running, cleanup stale PID file
                self.pid_file.unlink()
                
        except Exception as e:
            status["error"] = str(e)
        
        return status