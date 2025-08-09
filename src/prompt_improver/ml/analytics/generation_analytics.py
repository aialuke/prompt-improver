"""Generation History Tracking and Analytics System (Week 6)

Provides comprehensive tracking, analytics, and reporting for synthetic data
generation with trend analysis and effectiveness reporting.
"""
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from ...database.models import GenerationBatch, GenerationMethodPerformance, GenerationSession
from ...database.services.generation_service import GenerationDatabaseService
logger = logging.getLogger(__name__)

class GenerationHistoryTracker:
    """Tracks and analyzes generation history with comprehensive analytics"""

    def __init__(self, db_session: AsyncSession):
        self.db_service = GenerationDatabaseService(db_session)
        self.db_session = db_session

    async def start_tracking_session(self, generation_method: str, target_samples: int, session_type: str='synthetic_data', training_session_id: Optional[str]=None, configuration: Optional[Dict[str, Any]]=None, performance_gaps: Optional[Dict[str, float]]=None, focus_areas: Optional[List[str]]=None) -> str:
        """Start tracking a new generation session"""
        session = await self.db_service.create_generation_session(generation_method=generation_method, target_samples=target_samples, session_type=session_type, training_session_id=training_session_id, configuration=configuration, performance_gaps=performance_gaps, focus_areas=focus_areas)
        logger.info('Started tracking generation session %s', session.session_id)
        return session.session_id

    async def track_batch_completion(self, session_id: str, batch_number: int, batch_size: int, generation_method: str, processing_time: float, samples_generated: int, samples_filtered: int=0, error_count: int=0, quality_metrics: Optional[Dict[str, float]]=None, performance_metrics: Optional[Dict[str, float]]=None) -> str:
        """Track completion of a generation batch"""
        batch = await self.db_service.create_generation_batch(session_id=session_id, batch_number=batch_number, batch_size=batch_size, generation_method=generation_method, samples_requested=batch_size)
        await self.db_service.update_batch_performance(batch_id=batch.batch_id, processing_time_seconds=processing_time, samples_generated=samples_generated, samples_filtered=samples_filtered, error_count=error_count, memory_usage_mb=performance_metrics.get('memory_usage_mb') if performance_metrics else None, memory_peak_mb=performance_metrics.get('memory_peak_mb') if performance_metrics else None, efficiency_score=performance_metrics.get('efficiency_score') if performance_metrics else None, average_quality_score=quality_metrics.get('average_quality') if quality_metrics else None, diversity_score=quality_metrics.get('diversity_score') if quality_metrics else None)
        logger.info('Tracked batch {batch_number} completion for session %s', session_id)
        return batch.batch_id

    async def complete_session(self, session_id: str, final_sample_count: int, status: str='completed', error_message: Optional[str]=None) -> None:
        """Mark a generation session as complete"""
        await self.db_service.update_session_status(session_id=session_id, status=status, error_message=error_message, final_sample_count=final_sample_count)
        logger.info('Completed tracking for session {session_id} with %s samples', final_sample_count)

    async def record_method_performance(self, session_id: str, method_name: str, performance_metrics: Dict[str, Any]) -> None:
        """Record performance metrics for a generation method"""
        await self.db_service.record_method_performance(session_id=session_id, method_name=method_name, generation_time_seconds=performance_metrics.get('generation_time', 0.0), quality_score=performance_metrics.get('quality_score', 0.0), diversity_score=performance_metrics.get('diversity_score', 0.0), memory_usage_mb=performance_metrics.get('memory_usage_mb', 0.0), success_rate=performance_metrics.get('success_rate', 0.0), samples_generated=performance_metrics.get('samples_generated', 0), performance_gaps_addressed=performance_metrics.get('performance_gaps_addressed'), batch_size=performance_metrics.get('batch_size'), configuration=performance_metrics.get('configuration'))

class GenerationAnalytics:
    """Provides comprehensive analytics and reporting for generation history"""

    def __init__(self, db_session: AsyncSession):
        self.db_service = GenerationDatabaseService(db_session)
        self.db_session = db_session

    async def get_performance_trends(self, days_back: int=30, method_name: Optional[str]=None) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if method_name:
            performance_history = await self.db_service.get_method_performance_history(method_name=method_name, days_back=days_back)
        else:
            performance_history = []
            for method in ['statistical', 'neural', 'hybrid', 'diffusion']:
                method_history = await self.db_service.get_method_performance_history(method_name=method, days_back=days_back)
                performance_history.extend(method_history)
        if not performance_history:
            return {'status': 'no_data', 'message': 'No performance data available'}
        quality_scores = [p.quality_score for p in performance_history]
        diversity_scores = [p.diversity_score for p in performance_history]
        success_rates = [p.success_rate for p in performance_history]
        generation_times = [p.generation_time_seconds for p in performance_history]
        timestamps = [(p.recorded_at - performance_history[0].recorded_at).total_seconds() for p in performance_history]
        quality_trend = self._calculate_trend(timestamps, quality_scores)
        diversity_trend = self._calculate_trend(timestamps, diversity_scores)
        success_trend = self._calculate_trend(timestamps, success_rates)
        efficiency_trend = self._calculate_trend(timestamps, [-t for t in generation_times])
        return {'period_days': days_back, 'method_name': method_name, 'total_executions': len(performance_history), 'trends': {'quality_trend': quality_trend, 'diversity_trend': diversity_trend, 'success_trend': success_trend, 'efficiency_trend': efficiency_trend}, 'current_averages': {'quality_score': np.mean(quality_scores), 'diversity_score': np.mean(diversity_scores), 'success_rate': np.mean(success_rates), 'avg_generation_time': np.mean(generation_times)}, 'performance_ranges': {'quality_range': [min(quality_scores), max(quality_scores)], 'diversity_range': [min(diversity_scores), max(diversity_scores)], 'success_range': [min(success_rates), max(success_rates)]}}

    async def get_method_comparison(self, days_back: int=30) -> Dict[str, Any]:
        """Compare performance across different generation methods"""
        methods = ['statistical', 'neural', 'hybrid', 'diffusion']
        method_stats = {}
        for method in methods:
            performance_history = await self.db_service.get_method_performance_history(method_name=method, days_back=days_back)
            if performance_history:
                quality_scores = [p.quality_score for p in performance_history]
                diversity_scores = [p.diversity_score for p in performance_history]
                success_rates = [p.success_rate for p in performance_history]
                generation_times = [p.generation_time_seconds for p in performance_history]
                samples_generated = [p.samples_generated for p in performance_history]
                method_stats[method] = {'executions': len(performance_history), 'avg_quality': np.mean(quality_scores), 'avg_diversity': np.mean(diversity_scores), 'avg_success_rate': np.mean(success_rates), 'avg_generation_time': np.mean(generation_times), 'total_samples': sum(samples_generated), 'efficiency_score': sum(samples_generated) / sum(generation_times) if sum(generation_times) > 0 else 0}
            else:
                method_stats[method] = {'executions': 0, 'avg_quality': 0.0, 'avg_diversity': 0.0, 'avg_success_rate': 0.0, 'avg_generation_time': 0.0, 'total_samples': 0, 'efficiency_score': 0.0}
        method_rankings = {}
        for method, stats in method_stats.items():
            if stats['executions'] > 0:
                performance_score = 0.3 * stats['avg_quality'] + 0.2 * stats['avg_diversity'] + 0.2 * stats['avg_success_rate'] + 0.3 * min(1.0, stats['efficiency_score'] / 100.0)
                method_rankings[method] = performance_score
            else:
                method_rankings[method] = 0.0
        ranked_methods = sorted(method_rankings.items(), key=lambda x: x[1], reverse=True)
        return {'period_days': days_back, 'method_statistics': method_stats, 'method_rankings': dict(ranked_methods), 'best_method': ranked_methods[0][0] if ranked_methods else None, 'worst_method': ranked_methods[-1][0] if ranked_methods else None, 'analysis_timestamp': datetime.now().isoformat()}

    async def get_effectiveness_report(self, session_id: Optional[str]=None, days_back: int=7) -> Dict[str, Any]:
        """Generate comprehensive effectiveness report"""
        stats = await self.db_service.get_generation_statistics(days_back=days_back)
        method_comparison = await self.get_method_comparison(days_back=days_back)
        trends = await self.get_performance_trends(days_back=days_back)
        effectiveness_score = 0.0
        if stats['total_sessions'] > 0:
            effectiveness_score = 0.4 * stats['average_quality_score'] + 0.3 * stats['average_efficiency'] + 0.3 * (stats['total_samples_generated'] / (stats['total_sessions'] * 1000))
        return {'report_period': {'days_back': days_back, 'session_id': session_id, 'generated_at': datetime.now().isoformat()}, 'overall_effectiveness': {'effectiveness_score': min(1.0, effectiveness_score), 'total_sessions': stats['total_sessions'], 'total_samples_generated': stats['total_samples_generated'], 'average_quality': stats['average_quality_score'], 'average_efficiency': stats['average_efficiency']}, 'method_performance': method_comparison, 'performance_trends': trends, 'recommendations': self._generate_recommendations(stats, method_comparison, trends)}

    def _calculate_trend(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate trend correlation (-1 to 1)"""
        if len(x_values) < 2 or len(y_values) < 2:
            return 0.0
        try:
            correlation = np.corrcoef(x_values, y_values)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    def _generate_recommendations(self, stats: Dict[str, Any], method_comparison: Dict[str, Any], trends: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analytics"""
        recommendations = []
        if stats['average_quality_score'] < 0.7:
            recommendations.append('Consider increasing quality threshold or improving generation methods')
        if stats['average_efficiency'] < 0.6:
            recommendations.append('Optimize batch sizes and memory usage for better efficiency')
        best_method = method_comparison.get('best_method')
        if best_method:
            recommendations.append(f"Consider using '{best_method}' method more frequently for better performance")
        if trends.get('trends', {}).get('quality_trend', 0) < -0.3:
            recommendations.append('Quality is declining - review generation parameters and data sources')
        if trends.get('trends', {}).get('efficiency_trend', 0) < -0.3:
            recommendations.append('Efficiency is declining - consider system optimization or resource scaling')
        if not recommendations:
            recommendations.append('Generation performance is stable - continue current approach')
        return recommendations
