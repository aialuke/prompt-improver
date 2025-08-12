"""
Resource Monitor - 2025 Best Practices

Comprehensive system resource monitoring for ML workloads including
CPU, memory, GPU utilization, and system health metrics.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time
from typing import Any, Dict, List, Optional
import psutil
from ...utils.datetime_utils import aware_utc_now
logger = logging.getLogger(__name__)

@dataclass
class ResourceSnapshot:
    """System resource snapshot"""
    timestamp: datetime
    cpu_percent: float
    cpu_count: int
    load_avg_1m: float | None = None
    load_avg_5m: float | None = None
    load_avg_15m: float | None = None
    memory_total_gb: float
    memory_available_gb: float
    memory_used_gb: float
    memory_percent: float
    gpu_count: int = 0
    gpu_total_memory_gb: float = 0.0
    gpu_used_memory_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0

class ResourceMonitor:
    """
    System resource monitoring for ML workloads.
    
    Provides real-time monitoring of CPU, memory, GPU, and disk utilization
    with alerts for resource exhaustion.
    """

    def __init__(self, monitoring_interval_seconds: int=30):
        self.monitoring_interval_seconds = monitoring_interval_seconds
        self._gpu_available = False
        self._gpu_utils = self._initialize_gpu_monitoring()
        self._resource_history: list[ResourceSnapshot] = []
        self._max_history_size = 1000
        self.cpu_alert_threshold = 85.0
        self.memory_alert_threshold = 90.0
        self.gpu_memory_alert_threshold = 95.0
        self.disk_alert_threshold = 85.0
        logger.info('Resource Monitor initialized')

    def _initialize_gpu_monitoring(self) -> Any | None:
        """Initialize GPU monitoring with graceful fallback"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_available = True
            logger.info('GPU monitoring enabled')
            return pynvml
        except (ImportError, Exception) as e:
            logger.info('GPU monitoring not available: %s', e)
            self._gpu_available = False
            return None

    async def get_current_resources(self) -> ResourceSnapshot:
        """Get current system resource utilization"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = None
            try:
                if hasattr(psutil, 'getloadavg'):
                    load_avg_vals = psutil.getloadavg()
                    load_avg = {'1m': load_avg_vals[0], '5m': load_avg_vals[1], '15m': load_avg_vals[2]}
            except Exception:
                pass
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / 1024 ** 3
            memory_available_gb = memory.available / 1024 ** 3
            memory_used_gb = memory.used / 1024 ** 3
            memory_percent = memory.percent
            disk = psutil.disk_usage('/')
            disk_total_gb = disk.total / 1024 ** 3
            disk_used_gb = disk.used / 1024 ** 3
            disk_free_gb = disk.free / 1024 ** 3
            disk_percent = disk.percent
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            gpu_metrics = await self._get_gpu_metrics()
            snapshot = ResourceSnapshot(timestamp=aware_utc_now(), cpu_percent=cpu_percent, cpu_count=cpu_count, load_avg_1m=load_avg['1m'] if load_avg else None, load_avg_5m=load_avg['5m'] if load_avg else None, load_avg_15m=load_avg['15m'] if load_avg else None, memory_total_gb=memory_total_gb, memory_available_gb=memory_available_gb, memory_used_gb=memory_used_gb, memory_percent=memory_percent, gpu_count=gpu_metrics.get('gpu_count', 0), gpu_total_memory_gb=gpu_metrics.get('gpu_total_memory_gb', 0.0), gpu_used_memory_gb=gpu_metrics.get('gpu_used_memory_gb', 0.0), gpu_utilization_percent=gpu_metrics.get('gpu_utilization_percent', 0.0), disk_total_gb=disk_total_gb, disk_used_gb=disk_used_gb, disk_free_gb=disk_free_gb, disk_percent=disk_percent, network_bytes_sent=network_bytes_sent, network_bytes_recv=network_bytes_recv)
            self._add_to_history(snapshot)
            return snapshot
        except Exception as e:
            logger.error('Failed to get current resources: %s', e)
            return ResourceSnapshot(timestamp=aware_utc_now(), cpu_percent=0.0, cpu_count=0, memory_total_gb=0.0, memory_available_gb=0.0, memory_used_gb=0.0, memory_percent=0.0, disk_total_gb=0.0, disk_used_gb=0.0, disk_free_gb=0.0, disk_percent=0.0)

    async def _get_gpu_metrics(self) -> dict[str, Any]:
        """Get GPU utilization metrics"""
        if not self._gpu_available or not self._gpu_utils:
            return {}
        try:
            gpu_count = self._gpu_utils.nvmlDeviceGetCount()
            total_memory_gb = 0.0
            used_memory_gb = 0.0
            total_utilization = 0.0
            for i in range(gpu_count):
                handle = self._gpu_utils.nvmlDeviceGetHandleByIndex(i)
                mem_info = self._gpu_utils.nvmlDeviceGetMemoryInfo(handle)
                device_total_gb = mem_info.total / 1024 ** 3
                device_used_gb = mem_info.used / 1024 ** 3
                util_info = self._gpu_utils.nvmlDeviceGetUtilizationRates(handle)
                device_util = util_info.gpu
                total_memory_gb += device_total_gb
                used_memory_gb += device_used_gb
                total_utilization += device_util
            return {'gpu_count': gpu_count, 'gpu_total_memory_gb': total_memory_gb, 'gpu_used_memory_gb': used_memory_gb, 'gpu_utilization_percent': total_utilization / max(gpu_count, 1)}
        except Exception as e:
            logger.error('Failed to get GPU metrics: %s', e)
            return {}

    def _add_to_history(self, snapshot: ResourceSnapshot) -> None:
        """Add snapshot to history with size limit"""
        self._resource_history.append(snapshot)
        if len(self._resource_history) > self._max_history_size:
            self._resource_history = self._resource_history[-self._max_history_size:]

    async def get_resource_summary(self, hours: int=24) -> dict[str, Any]:
        """Get resource utilization summary for the specified period"""
        try:
            cutoff_time = aware_utc_now() - timedelta(hours=hours)
            recent_snapshots = [s for s in self._resource_history if s.timestamp > cutoff_time]
            if not recent_snapshots:
                current = await self.get_current_resources()
                recent_snapshots = [current]
            cpu_values = [s.cpu_percent for s in recent_snapshots]
            memory_values = [s.memory_percent for s in recent_snapshots]
            disk_values = [s.disk_percent for s in recent_snapshots]
            gpu_util_values = [s.gpu_utilization_percent for s in recent_snapshots if s.gpu_count > 0]
            gpu_memory_values = [s.gpu_used_memory_gb / max(s.gpu_total_memory_gb, 1) * 100 for s in recent_snapshots if s.gpu_total_memory_gb > 0]
            alerts = self._generate_resource_alerts(recent_snapshots[-1] if recent_snapshots else None)
            return {'period_hours': hours, 'timestamp': aware_utc_now().isoformat(), 'snapshots_analyzed': len(recent_snapshots), 'cpu_summary': {'avg_percent': float(sum(cpu_values) / len(cpu_values)) if cpu_values else 0.0, 'max_percent': float(max(cpu_values)) if cpu_values else 0.0, 'min_percent': float(min(cpu_values)) if cpu_values else 0.0, 'cpu_count': recent_snapshots[0].cpu_count if recent_snapshots else 0}, 'memory_summary': {'avg_percent': float(sum(memory_values) / len(memory_values)) if memory_values else 0.0, 'max_percent': float(max(memory_values)) if memory_values else 0.0, 'total_gb': recent_snapshots[0].memory_total_gb if recent_snapshots else 0.0, 'current_available_gb': recent_snapshots[-1].memory_available_gb if recent_snapshots else 0.0}, 'gpu_summary': {'available': self._gpu_available, 'gpu_count': recent_snapshots[0].gpu_count if recent_snapshots else 0, 'avg_utilization_percent': float(sum(gpu_util_values) / len(gpu_util_values)) if gpu_util_values else 0.0, 'max_utilization_percent': float(max(gpu_util_values)) if gpu_util_values else 0.0, 'avg_memory_percent': float(sum(gpu_memory_values) / len(gpu_memory_values)) if gpu_memory_values else 0.0, 'total_memory_gb': recent_snapshots[0].gpu_total_memory_gb if recent_snapshots else 0.0}, 'disk_summary': {'avg_percent': float(sum(disk_values) / len(disk_values)) if disk_values else 0.0, 'max_percent': float(max(disk_values)) if disk_values else 0.0, 'total_gb': recent_snapshots[0].disk_total_gb if recent_snapshots else 0.0, 'current_free_gb': recent_snapshots[-1].disk_free_gb if recent_snapshots else 0.0}, 'alerts': alerts, 'resource_health': self._assess_resource_health(recent_snapshots[-1] if recent_snapshots else None)}
        except Exception as e:
            logger.error('Failed to get resource summary: %s', e)
            return {'error': str(e), 'timestamp': aware_utc_now().isoformat()}

    def _generate_resource_alerts(self, current_snapshot: ResourceSnapshot | None) -> list[dict[str, Any]]:
        """Generate resource utilization alerts"""
        alerts = []
        if not current_snapshot:
            return alerts
        if current_snapshot.cpu_percent > self.cpu_alert_threshold:
            alerts.append({'type': 'cpu_high', 'severity': 'warning' if current_snapshot.cpu_percent < 95 else 'critical', 'message': f'High CPU utilization: {current_snapshot.cpu_percent:.1f}%', 'current_value': current_snapshot.cpu_percent, 'threshold': self.cpu_alert_threshold})
        if current_snapshot.memory_percent > self.memory_alert_threshold:
            alerts.append({'type': 'memory_high', 'severity': 'warning' if current_snapshot.memory_percent < 95 else 'critical', 'message': f'High memory utilization: {current_snapshot.memory_percent:.1f}%', 'current_value': current_snapshot.memory_percent, 'threshold': self.memory_alert_threshold})
        if current_snapshot.memory_available_gb < 1.0:
            alerts.append({'type': 'memory_low', 'severity': 'critical', 'message': f'Low available memory: {current_snapshot.memory_available_gb:.1f}GB', 'current_value': current_snapshot.memory_available_gb, 'threshold': 1.0})
        if current_snapshot.gpu_count > 0:
            gpu_memory_percent = current_snapshot.gpu_used_memory_gb / max(current_snapshot.gpu_total_memory_gb, 1) * 100
            if gpu_memory_percent > self.gpu_memory_alert_threshold:
                alerts.append({'type': 'gpu_memory_high', 'severity': 'warning', 'message': f'High GPU memory utilization: {gpu_memory_percent:.1f}%', 'current_value': gpu_memory_percent, 'threshold': self.gpu_memory_alert_threshold})
        if current_snapshot.disk_percent > self.disk_alert_threshold:
            alerts.append({'type': 'disk_high', 'severity': 'warning' if current_snapshot.disk_percent < 95 else 'critical', 'message': f'High disk utilization: {current_snapshot.disk_percent:.1f}%', 'current_value': current_snapshot.disk_percent, 'threshold': self.disk_alert_threshold})
        return alerts

    def _assess_resource_health(self, current_snapshot: ResourceSnapshot | None) -> dict[str, Any]:
        """Assess overall resource health"""
        if not current_snapshot:
            return {'status': 'unknown', 'score': 0.0}
        cpu_health = max(0.0, (100 - current_snapshot.cpu_percent) / 100)
        memory_health = max(0.0, (100 - current_snapshot.memory_percent) / 100)
        disk_health = max(0.0, (100 - current_snapshot.disk_percent) / 100)
        gpu_health = 1.0
        if current_snapshot.gpu_count > 0 and current_snapshot.gpu_total_memory_gb > 0:
            gpu_memory_percent = current_snapshot.gpu_used_memory_gb / current_snapshot.gpu_total_memory_gb * 100
            gpu_memory_health = max(0.0, (100 - gpu_memory_percent) / 100)
            gpu_util_health = max(0.0, (100 - current_snapshot.gpu_utilization_percent) / 100)
            gpu_health = (gpu_memory_health + gpu_util_health) / 2
        health_factors = [cpu_health, memory_health, disk_health, gpu_health]
        overall_score = sum(health_factors) / len(health_factors)
        if overall_score > 0.8:
            status = 'excellent'
        elif overall_score > 0.6:
            status = 'good'
        elif overall_score > 0.4:
            status = 'fair'
        elif overall_score > 0.2:
            status = 'poor'
        else:
            status = 'critical'
        return {'status': status, 'score': float(overall_score), 'component_scores': {'cpu': float(cpu_health), 'memory': float(memory_health), 'disk': float(disk_health), 'gpu': float(gpu_health)}, 'recommendations': self._generate_health_recommendations(current_snapshot, overall_score)}

    def _generate_health_recommendations(self, snapshot: ResourceSnapshot, health_score: float) -> list[str]:
        """Generate recommendations based on resource health"""
        recommendations = []
        if health_score < 0.3:
            recommendations.append('🚨 Critical resource pressure - immediate action required')
        if snapshot.cpu_percent > 90:
            recommendations.append('⚡ Reduce CPU load or scale horizontally')
        elif snapshot.cpu_percent > 70:
            recommendations.append('📊 Monitor CPU usage trends')
        if snapshot.memory_percent > 90:
            recommendations.append('💾 Free memory or increase RAM allocation')
        elif snapshot.memory_available_gb < 2:
            recommendations.append('💾 Low available memory - monitor for OOM risks')
        if snapshot.gpu_count > 0:
            gpu_memory_percent = snapshot.gpu_used_memory_gb / max(snapshot.gpu_total_memory_gb, 1) * 100
            if gpu_memory_percent > 90:
                recommendations.append('🎮 High GPU memory usage - optimize batch sizes')
            elif snapshot.gpu_utilization_percent > 90:
                recommendations.append('🎮 High GPU utilization - consider load balancing')
        if snapshot.disk_percent > 90:
            recommendations.append('💿 Clean up disk space or expand storage')
        elif snapshot.disk_percent > 80:
            recommendations.append('💿 Monitor disk usage growth')
        if not recommendations:
            recommendations.append('✅ Resource utilization within normal ranges')
        return recommendations

    async def cleanup_old_history(self, days: int=7) -> int:
        """Clean up old resource history"""
        try:
            cutoff_time = aware_utc_now() - timedelta(days=days)
            initial_count = len(self._resource_history)
            self._resource_history = [s for s in self._resource_history if s.timestamp > cutoff_time]
            cleaned_count = initial_count - len(self._resource_history)
            if cleaned_count > 0:
                logger.info('Cleaned up %s old resource snapshots', cleaned_count)
            return cleaned_count
        except Exception as e:
            logger.error('Failed to cleanup old history: %s', e)
            return 0
_resource_monitor: ResourceMonitor | None = None

async def get_resource_monitor() -> ResourceMonitor:
    """Get or create global resource monitor instance"""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor
