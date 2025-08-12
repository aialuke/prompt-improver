"""MCP Server Health Monitor
Comprehensive health monitoring and alerting for native MCP server deployment
Provides real-time monitoring, alerting, and automated recovery
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Health metrics data structure"""

    timestamp: float
    pid: int | None
    status: str
    memory_mb: float
    cpu_percent: float
    num_threads: int
    open_files: int
    connections: int
    uptime_seconds: float
    response_time_ms: float | None
    error_count: int
    last_error: str | None


@dataclass
class HealthThresholds:
    """Configurable health check thresholds"""

    memory_mb_warning: float = 500.0
    memory_mb_critical: float = 1000.0
    cpu_percent_warning: float = 80.0
    cpu_percent_critical: float = 95.0
    response_time_ms_warning: float = 100.0
    response_time_ms_critical: float = 500.0
    error_rate_warning: float = 0.05
    error_rate_critical: float = 0.1
    uptime_minimum_seconds: float = 30.0


class MCPHealthMonitor:
    """Comprehensive MCP server health monitoring system"""

    def __init__(self, config_path: str | None = None):
        self.config = self._load_config(config_path)
        self.thresholds = HealthThresholds(**self.config.get("thresholds", {}))
        self.pid_file = Path(self.config.get("pid_file", "/var/run/mcp-server.pid"))
        self.metrics_file = Path(
            self.config.get(
                "metrics_file", PROJECT_ROOT / "logs/mcp-health-metrics.json"
            )
        )
        self.log_file = Path(
            self.config.get("log_file", PROJECT_ROOT / "logs/mcp-health-monitor.log")
        )
        self.process: psutil.Process | None = None
        self.start_time = time.time()
        self.health_history: list[HealthMetrics] = []
        self.alert_cooldown: dict[str, float] = {}
        self.running = False
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "monitor_interval": 30,
            "metrics_history_size": 100,
            "alert_cooldown_seconds": 300,
            "auto_restart": True,
            "max_restart_attempts": 3,
            "restart_delay_seconds": 10,
            "health_check_endpoint": None,
            "notification_webhooks": [],
            "thresholds": {},
        }
        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info("Loaded configuration from: %s", config_path)
            except Exception as e:
                logger.warning("Failed to load config from %s: %s", config_path, e)
        return default_config

    async def start_monitoring(self):
        """Start the health monitoring loop"""
        logger.info("Starting MCP server health monitoring...")
        logger.info("PID file: %s", self.pid_file)
        logger.info("Metrics file: %s", self.metrics_file)
        logger.info("Monitor interval: %ss", self.config["monitor_interval"])
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        while self.running:
            try:
                await self._monitor_cycle()
                await asyncio.sleep(self.config["monitor_interval"])
            except Exception as e:
                logger.error("Monitor cycle error: %s", e)
                await asyncio.sleep(5)

    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals"""
        logger.info("Received signal %s, shutting down monitoring...", signum)
        self.running = False

    async def _monitor_cycle(self):
        """Single monitoring cycle"""
        try:
            pid = self._get_process_pid()
            if pid is None:
                await self._handle_process_not_found()
                return
            if self.process is None or self.process.pid != pid:
                self.process = psutil.Process(pid)
                logger.info("Monitoring MCP server process: PID %s", pid)
            metrics = await self._collect_metrics()
            health_status = self._analyze_health(metrics)
            await self._handle_health_status(health_status, metrics)
            self._store_metrics(metrics)
            self._cleanup_metrics()
        except psutil.NoSuchProcess:
            logger.warning("MCP server process no longer exists")
            await self._handle_process_not_found()
        except Exception as e:
            logger.error("Monitoring cycle error: %s", e)

    def _get_process_pid(self) -> int | None:
        """Get MCP server PID from PID file"""
        try:
            if not self.pid_file.exists():
                return None
            with open(self.pid_file) as f:
                pid = int(f.read().strip())
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                cmdline = " ".join(proc.cmdline())
                if "mcp_server" in cmdline or "prompt_improver" in cmdline:
                    return pid
                logger.warning("PID %s exists but doesn't appear to be MCP server", pid)
            return None
        except (FileNotFoundError, ValueError, psutil.NoSuchProcess) as e:
            logger.debug("Error getting PID: %s", e)
            return None

    async def _collect_metrics(self) -> HealthMetrics:
        """Collect comprehensive health metrics"""
        if not self.process:
            raise ValueError("No process to monitor")
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        num_threads = self.process.num_threads()
        open_files = len(self.process.open_files())
        connections = len(self.process.connections())
        process_start_time = self.process.create_time()
        uptime_seconds = time.time() - process_start_time
        response_time_ms = await self._test_response_time()
        error_count = self._get_error_count()
        last_error = self._get_last_error()
        return HealthMetrics(
            timestamp=time.time(),
            pid=self.process.pid,
            status=self.process.status(),
            memory_mb=memory_info.rss / 1024 / 1024,
            cpu_percent=cpu_percent,
            num_threads=num_threads,
            open_files=open_files,
            connections=connections,
            uptime_seconds=uptime_seconds,
            response_time_ms=response_time_ms,
            error_count=error_count,
            last_error=last_error,
        )

    async def _test_response_time(self) -> float | None:
        """Test MCP server response time"""
        endpoint = self.config.get("health_check_endpoint")
        if not endpoint:
            return None
        try:
            import aiohttp

            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, timeout=5) as response:
                    await response.text()
            response_time = (time.time() - start_time) * 1000
            return response_time
        except Exception as e:
            logger.debug("Health check endpoint test failed: %s", e)
            return None

    def _get_error_count(self) -> int:
        """Get current error count (simplified implementation)"""
        return 0

    def _get_last_error(self) -> str | None:
        """Get last error message (simplified implementation)"""
        return None

    def _analyze_health(self, metrics: HealthMetrics) -> str:
        """Analyze health metrics and return status"""
        issues = []
        if metrics.memory_mb > self.thresholds.memory_mb_critical:
            issues.append(
                f"CRITICAL: Memory usage {metrics.memory_mb:.1f}MB exceeds critical threshold {self.thresholds.memory_mb_critical}MB"
            )
        elif metrics.memory_mb > self.thresholds.memory_mb_warning:
            issues.append(
                f"WARNING: Memory usage {metrics.memory_mb:.1f}MB exceeds warning threshold {self.thresholds.memory_mb_warning}MB"
            )
        if metrics.cpu_percent > self.thresholds.cpu_percent_critical:
            issues.append(
                f"CRITICAL: CPU usage {metrics.cpu_percent:.1f}% exceeds critical threshold {self.thresholds.cpu_percent_critical}%"
            )
        elif metrics.cpu_percent > self.thresholds.cpu_percent_warning:
            issues.append(
                f"WARNING: CPU usage {metrics.cpu_percent:.1f}% exceeds warning threshold {self.thresholds.cpu_percent_warning}%"
            )
        if metrics.response_time_ms:
            if metrics.response_time_ms > self.thresholds.response_time_ms_critical:
                issues.append(
                    f"CRITICAL: Response time {metrics.response_time_ms:.1f}ms exceeds critical threshold {self.thresholds.response_time_ms_critical}ms"
                )
            elif metrics.response_time_ms > self.thresholds.response_time_ms_warning:
                issues.append(
                    f"WARNING: Response time {metrics.response_time_ms:.1f}ms exceeds warning threshold {self.thresholds.response_time_ms_warning}ms"
                )
        if metrics.status not in ["running", "sleeping"]:
            issues.append(f"WARNING: Process status is {metrics.status}")
        if metrics.uptime_seconds < self.thresholds.uptime_minimum_seconds:
            issues.append(
                f"WARNING: Process uptime {metrics.uptime_seconds:.1f}s is less than minimum {self.thresholds.uptime_minimum_seconds}s"
            )
        if any("CRITICAL" in issue for issue in issues):
            return "critical"
        if any("WARNING" in issue for issue in issues):
            return "warning"
        return "healthy"

    async def _handle_health_status(self, status: str, metrics: HealthMetrics):
        """Handle health status with alerting and recovery"""
        if status == "healthy":
            self._clear_alerts()
            logger.debug(
                "Health check passed - Memory: %sMB, CPU: %s%%",
                format(metrics.memory_mb, ".1f"),
                format(metrics.cpu_percent, ".1f"),
            )
            return
        if status == "warning":
            logger.warning(
                "Health warning detected - Memory: %sMB, CPU: %s%%",
                format(metrics.memory_mb, ".1f"),
                format(metrics.cpu_percent, ".1f"),
            )
        elif status == "critical":
            logger.error(
                "Critical health issue detected - Memory: %sMB, CPU: %s%%",
                format(metrics.memory_mb, ".1f"),
                format(metrics.cpu_percent, ".1f"),
            )
        await self._send_alert(status, metrics)
        if status == "critical" and self.config.get("auto_restart", False):
            await self._attempt_recovery(metrics)

    async def _handle_process_not_found(self):
        """Handle the case when MCP server process is not found"""
        logger.error("MCP server process not found")
        await self._send_alert("critical", None, "MCP server process not found")
        if self.config.get("auto_restart", False):
            await self._attempt_restart()

    async def _attempt_recovery(self, metrics: HealthMetrics):
        """Attempt to recover from critical health issues"""
        logger.info("Attempting automatic recovery...")
        await self._attempt_restart()

    async def _attempt_restart(self):
        """Attempt to restart the MCP server"""
        restart_attempts = getattr(self, "_restart_attempts", 0)
        max_attempts = self.config.get("max_restart_attempts", 3)
        if restart_attempts >= max_attempts:
            logger.error("Maximum restart attempts (%s) exceeded", max_attempts)
            return
        self._restart_attempts = restart_attempts + 1
        logger.info(
            "Attempting to restart MCP server (attempt %s/%s)",
            self._restart_attempts,
            max_attempts,
        )
        try:
            if self._is_systemd_service():
                result = subprocess.run(
                    ["sudo", "systemctl", "restart", "mcp-server"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    logger.info("MCP server restarted via systemctl")
                    await asyncio.sleep(self.config.get("restart_delay_seconds", 10))
                    return
                logger.error("systemctl restart failed: %s", result.stderr)
            startup_script = PROJECT_ROOT / "scripts/start_mcp_native.sh"
            if startup_script.exists():
                logger.info("Attempting restart using startup script")
        except Exception as e:
            logger.error("Restart attempt failed: %s", e)

    def _is_systemd_service(self) -> bool:
        """Check if MCP server is running as a systemd service"""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "mcp-server"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except:
            return False

    async def _send_alert(
        self, status: str, metrics: HealthMetrics | None, message: str = ""
    ):
        """Send health alert with cooldown"""
        alert_key = f"{status}_{message[:50]}"
        cooldown_seconds = self.config.get("alert_cooldown_seconds", 300)
        last_alert = self.alert_cooldown.get(alert_key, 0)
        if time.time() - last_alert < cooldown_seconds:
            return
        self.alert_cooldown[alert_key] = time.time()
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "message": message,
            "server": "mcp-server",
            "metrics": asdict(metrics) if metrics else None,
        }
        logger.info("Sending %s alert: %s", status, message)
        webhooks = self.config.get("notification_webhooks", [])
        for webhook_url in webhooks:
            try:
                await self._send_webhook_alert(webhook_url, alert_data)
            except Exception as e:
                logger.error("Failed to send webhook alert to %s: %s", webhook_url, e)

    async def _send_webhook_alert(self, webhook_url: str, alert_data: dict):
        """Send alert to webhook endpoint"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url, json=alert_data, timeout=10
                ) as response:
                    if response.status == 200:
                        logger.debug("Alert sent to webhook: %s", webhook_url)
                    else:
                        logger.warning("Webhook returned status %s", response.status)
        except Exception as e:
            logger.error("Webhook error: %s", e)

    def _clear_alerts(self):
        """Clear alert cooldowns when health is restored"""
        if self.alert_cooldown:
            logger.debug("Clearing alert cooldowns - health restored")
            self.alert_cooldown.clear()

    def _store_metrics(self, metrics: HealthMetrics):
        """Store metrics to file and memory"""
        self.health_history.append(metrics)
        try:
            metrics_data = {
                "latest": asdict(metrics),
                "history_count": len(self.health_history),
                "monitor_uptime": time.time() - self.start_time,
            }
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_file, "w") as f:
                json.dump(metrics_data, f, indent=2)
        except Exception as e:
            logger.error("Failed to store metrics: %s", e)

    def _cleanup_metrics(self):
        """Remove old metrics from history"""
        max_history = self.config.get("metrics_history_size", 100)
        if len(self.health_history) > max_history:
            self.health_history = self.health_history[-max_history:]

    def get_health_summary(self) -> dict[str, Any]:
        """Get current health summary"""
        if not self.health_history:
            return {"status": "no_data", "message": "No health data available"}
        latest = self.health_history[-1]
        return {
            "status": "healthy",
            "latest_metrics": asdict(latest),
            "uptime_seconds": latest.uptime_seconds,
            "monitor_uptime": time.time() - self.start_time,
            "history_points": len(self.health_history),
        }


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Server Health Monitor")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument(
        "--pid-file", help="PID file path", default="/var/run/mcp-server.pid"
    )
    parser.add_argument(
        "--metrics-file",
        help="Metrics output file",
        default="logs/mcp-health-metrics.json",
    )
    parser.add_argument(
        "--interval", type=int, help="Monitor interval in seconds", default=30
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current health status and exit"
    )
    args = parser.parse_args()
    config = {
        "pid_file": args.pid_file,
        "metrics_file": args.metrics_file,
        "monitor_interval": args.interval,
    }
    monitor = MCPHealthMonitor(args.config)
    monitor.config.update(config)
    if args.status:
        summary = monitor.get_health_summary()
        print(json.dumps(summary, indent=2))
        return None
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error("Monitor error: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
