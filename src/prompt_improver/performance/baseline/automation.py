"""Baseline automation system for scheduled collection and analysis."""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import uuid

# Scheduling
try:
    import aiocron
    AIOCRON_AVAILABLE = True
except ImportError:
    AIOCRON_AVAILABLE = False

# Task queue
try:
    import celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

# Enhanced background task management
from ...performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority

from .baseline_collector import BaselineCollector
from .statistical_analyzer import StatisticalAnalyzer
from .regression_detector import RegressionDetector
from .profiler import ContinuousProfiler
from .models import BaselineMetrics, RegressionAlert, PerformanceTrend

logger = logging.getLogger(__name__)

class AutomationConfig:
    """Configuration for baseline automation."""
    
    def __init__(
        self,
        collection_schedule: str = "*/5 * * * *",  # Every 5 minutes
        analysis_schedule: str = "0 */1 * * *",   # Every hour
        reporting_schedule: str = "0 9 * * *",    # Daily at 9 AM
        regression_check_schedule: str = "*/10 * * * *",  # Every 10 minutes
        
        # Retention settings
        baseline_retention_days: int = 30,
        analysis_retention_days: int = 90,
        alert_retention_days: int = 365,
        
        # Analysis settings
        trend_analysis_hours: int = 24,
        regression_reference_hours: int = 168,  # 7 days
        min_baselines_for_analysis: int = 10,
        
        # Automation flags
        enable_auto_collection: bool = True,
        enable_auto_analysis: bool = True,
        enable_auto_regression_detection: bool = True,
        enable_auto_profiling: bool = False,
        enable_auto_reporting: bool = True,
        
        # Storage settings
        storage_path: Optional[Path] = None,
        backup_enabled: bool = True,
        backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    ):
        """Initialize automation configuration."""
        self.collection_schedule = collection_schedule
        self.analysis_schedule = analysis_schedule
        self.reporting_schedule = reporting_schedule
        self.regression_check_schedule = regression_check_schedule
        
        self.baseline_retention_days = baseline_retention_days
        self.analysis_retention_days = analysis_retention_days
        self.alert_retention_days = alert_retention_days
        
        self.trend_analysis_hours = trend_analysis_hours
        self.regression_reference_hours = regression_reference_hours
        self.min_baselines_for_analysis = min_baselines_for_analysis
        
        self.enable_auto_collection = enable_auto_collection
        self.enable_auto_analysis = enable_auto_analysis
        self.enable_auto_regression_detection = enable_auto_regression_detection
        self.enable_auto_profiling = enable_auto_profiling
        self.enable_auto_reporting = enable_auto_reporting
        
        self.storage_path = storage_path or Path("./automation")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.backup_enabled = backup_enabled
        self.backup_schedule = backup_schedule

class AutomationTaskResult:
    """Result of an automation task."""
    
    def __init__(
        self,
        task_id: str,
        task_type: str,
        status: str,
        start_time: datetime,
        end_time: datetime,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.status = status  # 'success', 'failed', 'partial'
        self.start_time = start_time
        self.end_time = end_time
        self.result_data = result_data or {}
        self.error_message = error_message
        
        self.duration_seconds = (end_time - start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'status': self.status,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'result_data': self.result_data,
            'error_message': self.error_message
        }

class BaselineAutomation:
    """Automated baseline collection, analysis, and monitoring system."""
    
    def __init__(
        self,
        config: Optional[AutomationConfig] = None,
        collector: Optional[BaselineCollector] = None,
        analyzer: Optional[StatisticalAnalyzer] = None,
        detector: Optional[RegressionDetector] = None,
        profiler: Optional[ContinuousProfiler] = None
    ):
        """Initialize baseline automation."""
        self.config = config or AutomationConfig()
        
        # Components
        self.collector = collector or BaselineCollector()
        self.analyzer = analyzer or StatisticalAnalyzer()
        self.detector = detector or RegressionDetector()
        self.profiler = profiler or ContinuousProfiler()
        
        # State management
        self._running = False
        self._scheduled_tasks: Dict[str, Any] = {}
        self._task_history: List[AutomationTaskResult] = []
        self._lock = asyncio.Lock()
        
        # Storage paths
        self.baselines_path = self.config.storage_path / "baselines"
        self.analysis_path = self.config.storage_path / "analysis"
        self.alerts_path = self.config.storage_path / "alerts"
        self.reports_path = self.config.storage_path / "reports"
        
        # Create directories
        for path in [self.baselines_path, self.analysis_path, self.alerts_path, self.reports_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"BaselineAutomation initialized with storage at {self.config.storage_path}")

    async def start_automation(self) -> None:
        """Start all automated processes."""
        if self._running:
            logger.warning("Automation already running")
            return
        
        self._running = True
        
        # Start baseline collection
        if self.config.enable_auto_collection:
            await self.collector.start_collection()
        
        # Start continuous profiling if enabled
        if self.config.enable_auto_profiling:
            await self.profiler.start_profiling()
        
        # Schedule automated tasks
        await self._schedule_automation_tasks()
        
        logger.info("Started baseline automation")

    async def stop_automation(self) -> None:
        """Stop all automated processes."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop collection and profiling
        await self.collector.stop_collection()
        await self.profiler.stop_profiling()
        
        # Cancel scheduled tasks
        await self._cancel_scheduled_tasks()
        
        logger.info("Stopped baseline automation")

    async def _schedule_automation_tasks(self) -> None:
        """Schedule all automation tasks."""
        if not AIOCRON_AVAILABLE:
            logger.warning("aiocron not available, using fallback scheduling")
            # Start fallback task loop using EnhancedBackgroundTaskManager
            task_manager = get_background_task_manager()
            await task_manager.submit_enhanced_task(
                task_id=f"baseline_automation_fallback_{str(uuid.uuid4())[:8]}",
                coroutine=self._fallback_task_loop(),
                priority=TaskPriority.NORMAL,
                tags={"service": "performance", "type": "automation", "component": "baseline", "mode": "fallback"}
            )
            return
        
        try:
            # Schedule trend analysis
            if self.config.enable_auto_analysis:
                analysis_task = aiocron.crontab(
                    self.config.analysis_schedule,
                    func=self._run_trend_analysis,
                    start=True
                )
                self._scheduled_tasks['analysis'] = analysis_task
                logger.info(f"Scheduled trend analysis: {self.config.analysis_schedule}")
            
            # Schedule regression detection
            if self.config.enable_auto_regression_detection:
                regression_task = aiocron.crontab(
                    self.config.regression_check_schedule,
                    func=self._run_regression_detection,
                    start=True
                )
                self._scheduled_tasks['regression'] = regression_task
                logger.info(f"Scheduled regression detection: {self.config.regression_check_schedule}")
            
            # Schedule reporting
            if self.config.enable_auto_reporting:
                reporting_task = aiocron.crontab(
                    self.config.reporting_schedule,
                    func=self._run_automated_reporting,
                    start=True
                )
                self._scheduled_tasks['reporting'] = reporting_task
                logger.info(f"Scheduled automated reporting: {self.config.reporting_schedule}")
            
            # Schedule cleanup
            cleanup_task = aiocron.crontab(
                "0 1 * * *",  # Daily at 1 AM
                func=self._run_cleanup,
                start=True
            )
            self._scheduled_tasks['cleanup'] = cleanup_task
            
            # Schedule backup if enabled
            if self.config.backup_enabled:
                backup_task = aiocron.crontab(
                    self.config.backup_schedule,
                    func=self._run_backup,
                    start=True
                )
                self._scheduled_tasks['backup'] = backup_task
                logger.info(f"Scheduled backup: {self.config.backup_schedule}")
        
        except Exception as e:
            logger.error(f"Failed to schedule automation tasks: {e}")

    async def _cancel_scheduled_tasks(self) -> None:
        """Cancel all scheduled tasks."""
        for task_name, task in self._scheduled_tasks.items():
            try:
                if hasattr(task, 'stop'):
                    task.stop()
                elif hasattr(task, 'cancel'):
                    task.cancel()
                logger.debug(f"Cancelled scheduled task: {task_name}")
            except Exception as e:
                logger.error(f"Error cancelling task {task_name}: {e}")
        
        self._scheduled_tasks.clear()

    async def _fallback_task_loop(self) -> None:
        """Fallback task loop when aiocron is not available."""
        logger.info("Starting fallback task scheduling")
        
        last_analysis = datetime.min.replace(tzinfo=timezone.utc)
        last_regression_check = datetime.min.replace(tzinfo=timezone.utc)
        last_reporting = datetime.min.replace(tzinfo=timezone.utc)
        last_cleanup = datetime.min.replace(tzinfo=timezone.utc)
        
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Trend analysis (every hour)
                if self.config.enable_auto_analysis and (current_time - last_analysis).total_seconds() >= 3600:
                    await self._run_trend_analysis()
                    last_analysis = current_time
                
                # Regression detection (every 10 minutes)
                if self.config.enable_auto_regression_detection and (current_time - last_regression_check).total_seconds() >= 600:
                    await self._run_regression_detection()
                    last_regression_check = current_time
                
                # Daily reporting (once per day)
                if self.config.enable_auto_reporting and (current_time - last_reporting).total_seconds() >= 86400:
                    if current_time.hour == 9:  # 9 AM
                        await self._run_automated_reporting()
                        last_reporting = current_time
                
                # Daily cleanup (once per day)
                if (current_time - last_cleanup).total_seconds() >= 86400:
                    if current_time.hour == 1:  # 1 AM
                        await self._run_cleanup()
                        last_cleanup = current_time
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in fallback task loop: {e}")
                await asyncio.sleep(60)

    async def _run_trend_analysis(self) -> None:
        """Run automated trend analysis."""
        task_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting automated trend analysis (task: {task_id})")
        
        try:
            # Load recent baselines
            baselines = await self.collector.load_recent_baselines(
                hours=self.config.trend_analysis_hours
            )
            
            if len(baselines) < self.config.min_baselines_for_analysis:
                logger.warning(f"Insufficient baselines for analysis ({len(baselines)} < {self.config.min_baselines_for_analysis})")
                return
            
            # Analyze trends for key metrics
            metrics_to_analyze = [
                'response_time', 'error_rate', 'throughput', 
                'cpu_utilization', 'memory_utilization'
            ]
            
            analysis_results = {}
            
            for metric_name in metrics_to_analyze:
                try:
                    trend = await self.analyzer.analyze_trend(
                        metric_name, baselines, self.config.trend_analysis_hours
                    )
                    analysis_results[metric_name] = {
                        'direction': trend.direction.value,
                        'magnitude': trend.magnitude,
                        'confidence_score': trend.confidence_score,
                        'sample_count': trend.sample_count,
                        'is_significant': trend.is_significant(),
                        'predicted_24h': trend.predicted_value_24h,
                        'predicted_7d': trend.predicted_value_7d
                    }
                except Exception as e:
                    logger.error(f"Failed to analyze trend for {metric_name}: {e}")
            
            # Save analysis results
            await self._save_analysis_results(task_id, analysis_results)
            
            # Record task result
            result = AutomationTaskResult(
                task_id=task_id,
                task_type='trend_analysis',
                status='success',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                result_data={
                    'metrics_analyzed': len(analysis_results),
                    'baselines_used': len(baselines),
                    'timeframe_hours': self.config.trend_analysis_hours
                }
            )
            
            await self._record_task_result(result)
            
            logger.info(f"Completed trend analysis (task: {task_id}): {len(analysis_results)} metrics")
        
        except Exception as e:
            logger.error(f"Trend analysis failed (task: {task_id}): {e}")
            
            result = AutomationTaskResult(
                task_id=task_id,
                task_type='trend_analysis',
                status='failed',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                error_message=str(e)
            )
            
            await self._record_task_result(result)

    async def _run_regression_detection(self) -> None:
        """Run automated regression detection."""
        task_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        logger.debug(f"Starting regression detection (task: {task_id})")
        
        try:
            # Get current baseline
            recent_baselines = await self.collector.load_recent_baselines(hours=1)
            if not recent_baselines:
                logger.debug("No recent baselines for regression detection")
                return
            
            current_baseline = recent_baselines[-1]  # Most recent
            
            # Get reference baselines
            reference_baselines = await self.collector.load_recent_baselines(
                hours=self.config.regression_reference_hours
            )
            
            if len(reference_baselines) < self.config.min_baselines_for_analysis:
                logger.debug(f"Insufficient reference baselines ({len(reference_baselines)})")
                return
            
            # Check for regressions
            alerts = await self.detector.check_for_regressions(
                current_baseline, reference_baselines[:-1]  # Exclude current
            )
            
            # Save alerts
            if alerts:
                await self._save_regression_alerts(task_id, alerts)
            
            # Record task result
            result = AutomationTaskResult(
                task_id=task_id,
                task_type='regression_detection',
                status='success',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                result_data={
                    'alerts_generated': len(alerts),
                    'reference_baselines': len(reference_baselines),
                    'current_baseline_id': current_baseline.baseline_id
                }
            )
            
            await self._record_task_result(result)
            
            if alerts:
                logger.info(f"Regression detection (task: {task_id}): {len(alerts)} alerts generated")
            else:
                logger.debug(f"Regression detection (task: {task_id}): No regressions detected")
        
        except Exception as e:
            logger.error(f"Regression detection failed (task: {task_id}): {e}")
            
            result = AutomationTaskResult(
                task_id=task_id,
                task_type='regression_detection',
                status='failed',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                error_message=str(e)
            )
            
            await self._record_task_result(result)

    async def _run_automated_reporting(self) -> None:
        """Run automated daily reporting."""
        task_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting automated reporting (task: {task_id})")
        
        try:
            # Generate daily performance report
            report_data = await self._generate_daily_report()
            
            # Save report
            await self._save_report(task_id, report_data)
            
            # Record task result
            result = AutomationTaskResult(
                task_id=task_id,
                task_type='automated_reporting',
                status='success',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                result_data={
                    'report_sections': len(report_data),
                    'report_date': start_time.strftime('%Y-%m-%d')
                }
            )
            
            await self._record_task_result(result)
            
            logger.info(f"Completed automated reporting (task: {task_id})")
        
        except Exception as e:
            logger.error(f"Automated reporting failed (task: {task_id}): {e}")
            
            result = AutomationTaskResult(
                task_id=task_id,
                task_type='automated_reporting',
                status='failed',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                error_message=str(e)
            )
            
            await self._record_task_result(result)

    async def _run_cleanup(self) -> None:
        """Run automated cleanup of old data."""
        task_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting automated cleanup (task: {task_id})")
        
        try:
            cleanup_stats = {
                'baselines_removed': 0,
                'analysis_removed': 0,
                'alerts_removed': 0
            }
            
            # Clean up old baselines
            baseline_cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.baseline_retention_days)
            cleanup_stats['baselines_removed'] = await self._cleanup_old_files(
                self.baselines_path, baseline_cutoff
            )
            
            # Clean up old analysis results
            analysis_cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.analysis_retention_days)
            cleanup_stats['analysis_removed'] = await self._cleanup_old_files(
                self.analysis_path, analysis_cutoff
            )
            
            # Clean up old alerts
            alert_cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.alert_retention_days)
            cleanup_stats['alerts_removed'] = await self._cleanup_old_files(
                self.alerts_path, alert_cutoff
            )
            
            # Clean up task history
            history_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            initial_count = len(self._task_history)
            self._task_history = [
                task for task in self._task_history
                if task.end_time > history_cutoff
            ]
            cleanup_stats['tasks_removed'] = initial_count - len(self._task_history)
            
            # Record task result
            result = AutomationTaskResult(
                task_id=task_id,
                task_type='cleanup',
                status='success',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                result_data=cleanup_stats
            )
            
            await self._record_task_result(result)
            
            logger.info(f"Completed cleanup (task: {task_id}): {cleanup_stats}")
        
        except Exception as e:
            logger.error(f"Cleanup failed (task: {task_id}): {e}")
            
            result = AutomationTaskResult(
                task_id=task_id,
                task_type='cleanup',
                status='failed',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                error_message=str(e)
            )
            
            await self._record_task_result(result)

    async def _run_backup(self) -> None:
        """Run automated backup."""
        task_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting automated backup (task: {task_id})")
        
        try:
            # Create backup directory
            backup_dir = self.config.storage_path / "backups" / start_time.strftime("%Y%m%d_%H%M%S")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy important files
            import shutil
            
            backup_stats = {
                'files_backed_up': 0,
                'backup_size_mb': 0
            }
            
            # Backup recent baselines (last 7 days)
            recent_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            for baseline_file in self.baselines_path.glob("baseline_*.json"):
                if baseline_file.stat().st_mtime > recent_cutoff.timestamp():
                    shutil.copy2(baseline_file, backup_dir)
                    backup_stats['files_backed_up'] += 1
            
            # Backup recent analysis results
            for analysis_file in self.analysis_path.glob("analysis_*.json"):
                if analysis_file.stat().st_mtime > recent_cutoff.timestamp():
                    shutil.copy2(analysis_file, backup_dir)
                    backup_stats['files_backed_up'] += 1
            
            # Calculate backup size
            backup_size = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
            backup_stats['backup_size_mb'] = backup_size / (1024 * 1024)
            
            # Record task result
            result = AutomationTaskResult(
                task_id=task_id,
                task_type='backup',
                status='success',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                result_data=backup_stats
            )
            
            await self._record_task_result(result)
            
            logger.info(f"Completed backup (task: {task_id}): {backup_stats}")
        
        except Exception as e:
            logger.error(f"Backup failed (task: {task_id}): {e}")
            
            result = AutomationTaskResult(
                task_id=task_id,
                task_type='backup',
                status='failed',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                error_message=str(e)
            )
            
            await self._record_task_result(result)

    async def _save_analysis_results(self, task_id: str, results: Dict[str, Any]) -> None:
        """Save trend analysis results."""
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp_str}_{task_id[:8]}.json"
        filepath = self.analysis_path / filename
        
        analysis_data = {
            'task_id': task_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis_type': 'trend_analysis',
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)

    async def _save_regression_alerts(self, task_id: str, alerts: List[RegressionAlert]) -> None:
        """Save regression detection alerts."""
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"alerts_{timestamp_str}_{task_id[:8]}.json"
        filepath = self.alerts_path / filename
        
        alerts_data = {
            'task_id': task_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'alert_count': len(alerts),
            'alerts': [alert.to_dict() for alert in alerts]
        }
        
        with open(filepath, 'w') as f:
            json.dump(alerts_data, f, indent=2, default=str)

    async def _save_report(self, task_id: str, report_data: Dict[str, Any]) -> None:
        """Save automated report."""
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp_str}_{task_id[:8]}.json"
        filepath = self.reports_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

    async def _generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily performance report."""
        report_date = datetime.now(timezone.utc).date()
        
        # Load recent baselines
        baselines_24h = await self.collector.load_recent_baselines(hours=24)
        baselines_7d = await self.collector.load_recent_baselines(hours=168)
        
        report = {
            'report_date': report_date.isoformat(),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'baselines_24h': len(baselines_24h),
                'baselines_7d': len(baselines_7d)
            }
        }
        
        if baselines_24h:
            # Calculate performance metrics
            latest_baseline = baselines_24h[-1]
            
            # Performance scores
            if len(baselines_24h) > 1:
                performance_score = await self.analyzer.calculate_performance_score(latest_baseline)
                report['performance_score'] = performance_score
            
            # Recent alerts
            report['recent_alerts'] = self.detector.get_active_alerts()
            
            # System statistics
            report['statistics'] = {
                'collector': self.collector.get_collection_status(),
                'detector': self.detector.get_alert_statistics(),
                'profiler': self.profiler.get_profiler_status()
            }
        
        return report

    async def _cleanup_old_files(self, directory: Path, cutoff_time: datetime) -> int:
        """Clean up files older than cutoff time."""
        removed_count = 0
        
        for file_path in directory.glob("*.json"):
            try:
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                if file_time < cutoff_time:
                    file_path.unlink()
                    removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
        
        return removed_count

    async def _record_task_result(self, result: AutomationTaskResult) -> None:
        """Record automation task result."""
        async with self._lock:
            self._task_history.append(result)
            
            # Keep only recent history
            if len(self._task_history) > 1000:
                self._task_history = self._task_history[-1000:]

    async def trigger_manual_analysis(self) -> AutomationTaskResult:
        """Manually trigger trend analysis."""
        logger.info("Manually triggering trend analysis")
        await self._run_trend_analysis()
        
        # Return the most recent analysis task result
        analysis_tasks = [t for t in self._task_history if t.task_type == 'trend_analysis']
        return analysis_tasks[-1] if analysis_tasks else None

    async def trigger_manual_regression_check(self) -> AutomationTaskResult:
        """Manually trigger regression detection."""
        logger.info("Manually triggering regression detection")
        await self._run_regression_detection()
        
        # Return the most recent regression task result
        regression_tasks = [t for t in self._task_history if t.task_type == 'regression_detection']
        return regression_tasks[-1] if regression_tasks else None

    def get_automation_status(self) -> Dict[str, Any]:
        """Get current automation status."""
        # Recent task statistics
        recent_tasks = [
            task for task in self._task_history
            if (datetime.now(timezone.utc) - task.end_time).total_seconds() < 86400  # Last 24h
        ]
        
        task_stats = {}
        for task in recent_tasks:
            task_type = task.task_type
            if task_type not in task_stats:
                task_stats[task_type] = {'success': 0, 'failed': 0, 'total': 0}
            
            task_stats[task_type]['total'] += 1
            task_stats[task_type][task.status] += 1
        
        return {
            'running': self._running,
            'config': {
                'auto_collection': self.config.enable_auto_collection,
                'auto_analysis': self.config.enable_auto_analysis,
                'auto_regression_detection': self.config.enable_auto_regression_detection,
                'auto_profiling': self.config.enable_auto_profiling,
                'auto_reporting': self.config.enable_auto_reporting
            },
            'scheduled_tasks': list(self._scheduled_tasks.keys()),
            'task_history_count': len(self._task_history),
            'recent_task_stats': task_stats,
            'storage_paths': {
                'baselines': str(self.baselines_path),
                'analysis': str(self.analysis_path),
                'alerts': str(self.alerts_path),
                'reports': str(self.reports_path)
            }
        }

    def get_recent_task_results(self, hours: int = 24, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent task results."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        filtered_tasks = [
            task for task in self._task_history
            if task.end_time > cutoff_time and (task_type is None or task.task_type == task_type)
        ]
        
        return [task.to_dict() for task in filtered_tasks]

# Global automation instance
_global_automation: Optional[BaselineAutomation] = None

def get_baseline_automation() -> BaselineAutomation:
    """Get the global baseline automation instance."""
    global _global_automation
    if _global_automation is None:
        _global_automation = BaselineAutomation()
    return _global_automation

def set_baseline_automation(automation: BaselineAutomation) -> None:
    """Set the global baseline automation instance."""
    global _global_automation
    _global_automation = automation

# Convenience functions
async def start_automated_baseline_system() -> None:
    """Start the complete automated baseline system."""
    automation = get_baseline_automation()
    await automation.start_automation()
    logger.info("Started automated baseline system")

async def stop_automated_baseline_system() -> None:
    """Stop the complete automated baseline system."""
    automation = get_baseline_automation()
    await automation.stop_automation()
    logger.info("Stopped automated baseline system")

async def get_automation_dashboard() -> Dict[str, Any]:
    """Get automation dashboard data."""
    automation = get_baseline_automation()
    return {
        'status': automation.get_automation_status(),
        'recent_tasks': automation.get_recent_task_results(hours=24),
        'component_status': {
            'collector': automation.collector.get_collection_status(),
            'detector': automation.detector.get_alert_statistics(),
            'profiler': automation.profiler.get_profiler_status()
        }
    }