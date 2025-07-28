"""
Analytics Dashboard API Endpoints
Provides REST API endpoints for session analytics, performance trends, and dashboard data.

Key Features (2025 Standards):
- Real-time analytics data streaming
- Optimized query performance with caching
- Role-based access control
- WebSocket support for live updates
- Comprehensive error handling and monitoring
- OpenAPI documentation with examples
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..database.analytics_query_interface import (
    AnalyticsQueryInterface,
    TimeGranularity,
    MetricType,
    TrendAnalysisResult,
    AnalyticsQueryResult
)
from ..ml.analytics.session_summary_reporter import (
    SessionSummaryReporter,
    UserRole,
    ReportFormat
)
from ..ml.analytics.session_comparison_analyzer import (
    SessionComparisonAnalyzer,
    ComparisonDimension,
    ComparisonMethod
)
from ..utils.websocket_manager import connection_manager

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Create router for analytics endpoints
analytics_router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["analytics-dashboard"]
)

# Pydantic models for API requests/responses

class TimeRangeRequest(BaseModel):
    """Time range for analytics queries"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    hours: Optional[int] = Field(default=24, ge=1, le=8760)  # Max 1 year

class TrendAnalysisRequest(BaseModel):
    """Request for trend analysis"""
    time_range: TimeRangeRequest
    granularity: TimeGranularity = TimeGranularity.DAY
    metric_type: MetricType = MetricType.PERFORMANCE
    session_ids: Optional[List[str]] = None

class SessionComparisonRequest(BaseModel):
    """Request for session comparison"""
    session_a_id: str
    session_b_id: str
    dimension: ComparisonDimension = ComparisonDimension.PERFORMANCE
    method: ComparisonMethod = ComparisonMethod.T_TEST

class DashboardMetricsResponse(BaseModel):
    """Response for dashboard metrics"""
    current_period: Dict[str, Any]
    time_range: Dict[str, Any]
    last_updated: str
    previous_period: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None

class TrendAnalysisResponse(BaseModel):
    """Response for trend analysis"""
    time_series: List[Dict[str, Any]]
    trend_direction: str
    trend_strength: float
    correlation_coefficient: float
    seasonal_patterns: Dict[str, Any]
    metadata: Dict[str, Any]

class SessionComparisonResponse(BaseModel):
    """Response for session comparison"""
    session_a_id: str
    session_b_id: str
    comparison_dimension: str
    statistical_significance: bool
    p_value: float
    effect_size: float
    winner: str
    insights: List[str]
    recommendations: List[str]

# Dependency injection

async def get_analytics_interface(db_session: AsyncSession = Depends(get_session)) -> AnalyticsQueryInterface:
    """Get analytics query interface"""
    return AnalyticsQueryInterface(db_session)

async def get_session_reporter(db_session: AsyncSession = Depends(get_session)) -> SessionSummaryReporter:
    """Get session summary reporter"""
    return SessionSummaryReporter(db_session)

async def get_comparison_analyzer(db_session: AsyncSession = Depends(get_session)) -> SessionComparisonAnalyzer:
    """Get session comparison analyzer"""
    return SessionComparisonAnalyzer(db_session)

async def get_current_user_role(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> UserRole:
    """Get current user role (simplified for demo - implement proper auth)"""
    # In production, implement proper JWT token validation and role extraction
    if credentials and credentials.credentials:
        # Mock role extraction - replace with actual implementation
        return UserRole.ANALYST
    return UserRole.OPERATOR

# API Endpoints

@analytics_router.get("/dashboard/metrics", response_model=DashboardMetricsResponse)
async def get_dashboard_metrics(
    time_range_hours: int = Query(default=24, ge=1, le=168),  # Max 1 week
    include_comparisons: bool = Query(default=True),
    analytics: AnalyticsQueryInterface = Depends(get_analytics_interface)
):
    """
    Get comprehensive dashboard metrics for real-time display.

    Optimized for <1s response times with aggressive caching.
    """
    try:
        result = await analytics.get_dashboard_metrics(
            time_range_hours=time_range_hours,
            include_comparisons=include_comparisons
        )

        return DashboardMetricsResponse(**result)

    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve dashboard metrics"
        )

@analytics_router.post("/trends/analysis", response_model=TrendAnalysisResponse)
async def analyze_performance_trends(
    request: TrendAnalysisRequest,
    analytics: AnalyticsQueryInterface = Depends(get_analytics_interface)
):
    """
    Analyze performance trends with statistical analysis.

    Provides time-series data with trend direction and strength.
    """
    try:
        # Set default dates if not provided
        if not request.time_range.end_date:
            request.time_range.end_date = datetime.now(timezone.utc)
        if not request.time_range.start_date:
            if request.time_range.hours:
                request.time_range.start_date = request.time_range.end_date - timedelta(hours=request.time_range.hours)
            else:
                request.time_range.start_date = request.time_range.end_date - timedelta(days=30)

        result = await analytics.get_session_performance_trends(
            start_date=request.time_range.start_date,
            end_date=request.time_range.end_date,
            granularity=request.granularity,
            metric_type=request.metric_type,
            session_ids=request.session_ids
        )

        # Convert time series to serializable format
        time_series_data = []
        for point in result.time_series:
            time_series_data.append({
                "timestamp": point.timestamp.isoformat(),
                "value": point.value,
                "metadata": point.metadata or {}
            })

        return TrendAnalysisResponse(
            time_series=time_series_data,
            trend_direction=result.trend_direction,
            trend_strength=result.trend_strength,
            correlation_coefficient=result.correlation_coefficient,
            seasonal_patterns=result.seasonal_patterns,
            metadata={
                "granularity": request.granularity.value,
                "metric_type": request.metric_type.value,
                "session_count": len(request.session_ids) if request.session_ids else None
            }
        )

    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze performance trends"
        )

@analytics_router.post("/sessions/compare", response_model=SessionComparisonResponse)
async def compare_sessions(
    request: SessionComparisonRequest,
    analyzer: SessionComparisonAnalyzer = Depends(get_comparison_analyzer)
):
    """
    Compare two training sessions with statistical analysis.

    Provides detailed comparison with significance testing.
    """
    try:
        result = await analyzer.compare_sessions(
            session_a_id=request.session_a_id,
            session_b_id=request.session_b_id,
            dimension=request.dimension,
            method=request.method
        )

        return SessionComparisonResponse(
            session_a_id=result.session_a_id,
            session_b_id=result.session_b_id,
            comparison_dimension=result.comparison_dimension.value,
            statistical_significance=result.statistical_significance,
            p_value=result.p_value,
            effect_size=result.effect_size,
            winner=result.winner,
            insights=result.insights,
            recommendations=result.recommendations
        )

    except Exception as e:
        logger.error(f"Error comparing sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to compare sessions"
        )

@analytics_router.get("/sessions/{session_id}/summary")
async def get_session_summary(
    session_id: str,
    user_role: UserRole = Depends(get_current_user_role),
    reporter: SessionSummaryReporter = Depends(get_session_reporter)
):
    """
    Get comprehensive session summary with role-based customization.
    """
    try:
        summary = await reporter.generate_session_summary(session_id, user_role)

        # Convert to serializable format
        return {
            "session_id": summary.session_id,
            "status": summary.status,
            "executive_kpis": {
                "performance_score": summary.performance_score,
                "improvement_velocity": summary.improvement_velocity,
                "efficiency_rating": summary.efficiency_rating,
                "quality_index": summary.quality_index,
                "success_rate": summary.success_rate
            },
            "performance_metrics": {
                "initial_performance": summary.initial_performance,
                "final_performance": summary.final_performance,
                "best_performance": summary.best_performance,
                "total_improvement": summary.total_improvement,
                "improvement_rate": summary.improvement_rate,
                "performance_trend": summary.performance_trend
            },
            "training_statistics": {
                "total_iterations": summary.total_iterations,
                "successful_iterations": summary.successful_iterations,
                "failed_iterations": summary.failed_iterations,
                "average_iteration_duration": summary.average_iteration_duration
            },
            "insights": {
                "key_insights": summary.key_insights,
                "recommendations": summary.recommendations,
                "anomalies_detected": summary.anomalies_detected
            },
            "metadata": {
                "started_at": summary.started_at.isoformat(),
                "completed_at": summary.completed_at.isoformat() if summary.completed_at else None,
                "total_duration_hours": summary.total_duration_hours,
                "configuration": summary.configuration
            }
        }

    except Exception as e:
        logger.error(f"Error getting session summary: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve session summary"
        )

@analytics_router.get("/sessions/{session_id}/export")
async def export_session_report(
    session_id: str,
    format: ReportFormat = Query(default=ReportFormat.JSON),
    user_role: UserRole = Depends(get_current_user_role),
    reporter: SessionSummaryReporter = Depends(get_session_reporter)
):
    """
    Export session report in various formats.
    """
    try:
        output_path = await reporter.export_session_report(
            session_id=session_id,
            format=format,
            user_role=user_role
        )

        return JSONResponse({
            "status": "success",
            "export_path": output_path,
            "format": format.value,
            "session_id": session_id,
            "exported_at": datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        logger.error(f"Error exporting session report: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to export session report"
        )

@analytics_router.get("/distribution/performance")
async def get_performance_distribution(
    start_date: Optional[datetime] = Query(default=None),
    end_date: Optional[datetime] = Query(default=None),
    bucket_count: int = Query(default=20, ge=5, le=50),
    analytics: AnalyticsQueryInterface = Depends(get_analytics_interface)
):
    """
    Get performance distribution analysis with histogram data.
    """
    try:
        result = await analytics.get_performance_distribution_analysis(
            start_date=start_date,
            end_date=end_date,
            bucket_count=bucket_count
        )

        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Error getting performance distribution: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve performance distribution"
        )

@analytics_router.get("/correlation/analysis")
async def get_correlation_analysis(
    metrics: List[str] = Query(default=["performance", "efficiency", "duration"]),
    start_date: Optional[datetime] = Query(default=None),
    end_date: Optional[datetime] = Query(default=None),
    analytics: AnalyticsQueryInterface = Depends(get_analytics_interface)
):
    """
    Get correlation analysis between different session metrics.
    """
    try:
        result = await analytics.get_correlation_analysis(
            metrics=metrics,
            start_date=start_date,
            end_date=end_date
        )

        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Error getting correlation analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve correlation analysis"
        )

# WebSocket endpoints for real-time analytics

@analytics_router.websocket("/live/dashboard")
async def websocket_dashboard_endpoint(
    websocket: WebSocket,
    user_id: Optional[str] = Query(default=None)
):
    """
    WebSocket endpoint for real-time dashboard updates.

    Streams live analytics data to connected clients.
    """
    await connection_manager.connect(websocket, "dashboard", user_id)

    try:
        # Send initial dashboard data
        analytics = AnalyticsQueryInterface(websocket.state.db_session)
        initial_data = await analytics.get_dashboard_metrics(time_range_hours=24)

        await connection_manager.send_personal_message(
            websocket,
            {
                "type": "dashboard_data",
                "data": initial_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages
                data = await websocket.receive_json()

                if data.get("type") == "request_update":
                    # Client requesting fresh data
                    fresh_data = await analytics.get_dashboard_metrics(
                        time_range_hours=data.get("time_range_hours", 24)
                    )

                    await connection_manager.send_personal_message(
                        websocket,
                        {
                            "type": "dashboard_data",
                            "data": fresh_data,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )

                elif data.get("type") == "subscribe_session":
                    # Client subscribing to specific session updates
                    session_id = data.get("session_id")
                    if session_id:
                        await connection_manager.add_to_group(websocket, f"session_{session_id}")

                        # Send initial session data
                        reporter = SessionSummaryReporter(websocket.state.db_session)
                        session_data = await reporter.generate_session_summary(session_id)

                        await connection_manager.send_personal_message(
                            websocket,
                            {
                                "type": "session_data",
                                "session_id": session_id,
                                "data": session_data.__dict__,
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }
                        )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in dashboard WebSocket: {e}")
                await connection_manager.send_personal_message(
                    websocket,
                    {
                        "type": "error",
                        "message": "Internal server error",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Dashboard WebSocket connection error: {e}")
    finally:
        await connection_manager.disconnect(websocket, "dashboard")

@analytics_router.websocket("/live/session/{session_id}")
async def websocket_session_endpoint(
    websocket: WebSocket,
    session_id: str,
    user_id: Optional[str] = Query(default=None)
):
    """
    WebSocket endpoint for real-time session monitoring.

    Streams live session updates to connected clients.
    """
    await connection_manager.connect(websocket, f"session_{session_id}", user_id)

    try:
        # Send initial session data
        reporter = SessionSummaryReporter(websocket.state.db_session)
        initial_data = await reporter.generate_session_summary(session_id)

        await connection_manager.send_personal_message(
            websocket,
            {
                "type": "session_update",
                "session_id": session_id,
                "data": initial_data.__dict__,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_json()

                if data.get("type") == "request_update":
                    # Send fresh session data
                    fresh_data = await reporter.generate_session_summary(session_id)

                    await connection_manager.send_personal_message(
                        websocket,
                        {
                            "type": "session_update",
                            "session_id": session_id,
                            "data": fresh_data.__dict__,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in session WebSocket: {e}")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Session WebSocket connection error: {e}")
    finally:
        await connection_manager.disconnect(websocket, f"session_{session_id}")

# Background tasks for real-time updates

async def broadcast_dashboard_updates():
    """Background task to broadcast dashboard updates to all connected clients"""
    try:
        # This would be called periodically by a scheduler
        analytics = AnalyticsQueryInterface(None)  # Would need proper session injection
        dashboard_data = await analytics.get_dashboard_metrics(time_range_hours=24)

        await connection_manager.broadcast_to_group(
            "dashboard",
            {
                "type": "dashboard_update",
                "data": dashboard_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Error broadcasting dashboard updates: {e}")

async def broadcast_session_update(session_id: str, update_data: Dict[str, Any]):
    """Broadcast session update to all connected clients monitoring this session"""
    try:
        await connection_manager.broadcast_to_group(
            f"session_{session_id}",
            {
                "type": "session_update",
                "session_id": session_id,
                "data": update_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Error broadcasting session update: {e}")

# Health and status endpoints

@analytics_router.get("/health")
async def analytics_health_check():
    """Health check for analytics service"""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {
                "analytics_query_interface": "operational",
                "session_reporter": "operational",
                "comparison_analyzer": "operational",
                "websocket_connections": len(connection_manager.active_connections)
            }
        }

        return JSONResponse(health_status)

    except Exception as e:
        logger.error(f"Analytics health check failed: {e}")
        return JSONResponse(
            {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=503
        )

@analytics_router.get("/dashboard/config")
async def get_dashboard_config():
    """Get dashboard configuration for frontend initialization"""
    try:
        config = {
            "dashboard_settings": {
                "auto_refresh": True,
                "refresh_interval": 30,  # seconds
                "show_alerts": True,
                "chart_types": ["line", "bar", "histogram", "scatter"],
                "metrics_to_display": [
                    "performance_score",
                    "improvement_velocity",
                    "efficiency_rating",
                    "success_rate",
                    "error_rate"
                ]
            },
            "websocket_endpoints": {
                "dashboard": "/api/v1/analytics/live/dashboard",
                "session": "/api/v1/analytics/live/session/{session_id}"
            },
            "api_endpoints": {
                "dashboard_metrics": "/api/v1/analytics/dashboard/metrics",
                "trend_analysis": "/api/v1/analytics/trends/analysis",
                "session_comparison": "/api/v1/analytics/sessions/compare",
                "performance_distribution": "/api/v1/analytics/distribution/performance",
                "correlation_analysis": "/api/v1/analytics/correlation/analysis"
            },
            "user_roles": [role.value for role in UserRole],
            "comparison_dimensions": [dim.value for dim in ComparisonDimension],
            "metric_types": [metric.value for metric in MetricType],
            "time_granularities": [gran.value for gran in TimeGranularity]
        }

        return JSONResponse({"status": "success", "config": config})

    except Exception as e:
        logger.error(f"Error getting dashboard config: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve dashboard configuration"
        )
