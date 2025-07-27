"""
Comprehensive Business Metrics Package for Prompt Improver

Provides unified metrics collection across ML operations, API usage, system performance,
and business intelligence with real-time aggregation and dashboard exports.

Core Components:
- ML-specific metrics: Prompt improvement success rates, model inference performance
- API usage metrics: Endpoint analytics, user journey tracking, rate limiting effectiveness
- Performance metrics: Request pipeline stages, database operations, cache effectiveness
- Business intelligence: Feature adoption, cost tracking, resource utilization
- Real-time aggregation: Cross-metric correlation, intelligent alerting
- Dashboard exports: Multiple formats (JSON, CSV, Prometheus, Grafana, Excel)

All metrics integrate with existing monitoring infrastructure and provide
real-time business insights with sub-second latency.
"""

# ML-specific metrics
from .ml_metrics import (
    MLMetricsCollector,
    PromptCategory,
    ModelInferenceStage,
    get_ml_metrics_collector,
    record_prompt_improvement,
    record_model_inference
)

# API usage metrics
from .api_metrics import (
    APIMetricsCollector,
    EndpointCategory,
    HTTPMethod,
    UserJourneyStage,
    AuthenticationMethod,
    get_api_metrics_collector,
    record_api_request,
    record_user_journey_event
)

# Performance metrics
from .performance_metrics import (
    PerformanceMetricsCollector,
    PipelineStage,
    DatabaseOperation,
    CacheType,
    ExternalAPIType,
    get_performance_metrics_collector,
    record_pipeline_stage_timing
)

# Business intelligence metrics
from .business_intelligence_metrics import (
    BusinessIntelligenceMetricsCollector,
    FeatureCategory,
    UserTier,
    CostType,
    ResourceType,
    get_bi_metrics_collector,
    record_feature_usage,
    record_operational_cost
)

# Real-time aggregation engine
from .aggregation_engine import (
    RealTimeAggregationEngine,
    AggregationWindow,
    AlertSeverity,
    CorrelationType,
    get_aggregation_engine,
    start_aggregation_engine,
    stop_aggregation_engine,
    get_business_insights
)

# Dashboard exports
from .dashboard_exports import (
    DashboardExporter,
    ExportFormat,
    DashboardType,
    TimeRange,
    get_dashboard_exporter,
    export_executive_summary,
    export_ml_performance,
    export_real_time_monitoring
)

# Legacy system metrics (maintained for compatibility)
try:
    from .system_metrics import (
        SystemMetricsCollector,
        ConnectionAgeTracker, 
        RequestQueueMonitor,
        CacheEfficiencyMonitor,
        FeatureUsageAnalytics,
        MetricsConfig,
        get_system_metrics_collector
    )
    _legacy_available = True
except ImportError:
    _legacy_available = False

__all__ = [
    # ML metrics
    'MLMetricsCollector',
    'PromptCategory',
    'ModelInferenceStage', 
    'get_ml_metrics_collector',
    'record_prompt_improvement',
    'record_model_inference',
    
    # API metrics
    'APIMetricsCollector',
    'EndpointCategory',
    'HTTPMethod',
    'UserJourneyStage',
    'AuthenticationMethod',
    'get_api_metrics_collector',
    'record_api_request',
    'record_user_journey_event',
    
    # Performance metrics
    'PerformanceMetricsCollector',
    'PipelineStage',
    'DatabaseOperation',
    'CacheType',
    'ExternalAPIType',
    'get_performance_metrics_collector',
    'record_pipeline_stage_timing',
    
    # Business intelligence metrics
    'BusinessIntelligenceMetricsCollector',
    'FeatureCategory',
    'UserTier',
    'CostType',
    'ResourceType',
    'get_bi_metrics_collector',
    'record_feature_usage',
    'record_operational_cost',
    
    # Aggregation engine
    'RealTimeAggregationEngine',
    'AggregationWindow',
    'AlertSeverity',
    'CorrelationType',
    'get_aggregation_engine',
    'start_aggregation_engine',
    'stop_aggregation_engine',
    'get_business_insights',
    
    # Dashboard exports
    'DashboardExporter',
    'ExportFormat',
    'DashboardType',
    'TimeRange',
    'get_dashboard_exporter',
    'export_executive_summary',
    'export_ml_performance',
    'export_real_time_monitoring'
]

# Add legacy metrics if available
if _legacy_available:
    __all__.extend([
        'SystemMetricsCollector',
        'ConnectionAgeTracker',
        'RequestQueueMonitor', 
        'CacheEfficiencyMonitor',
        'FeatureUsageAnalytics',
        'MetricsConfig',
        'get_system_metrics_collector'
    ])


# Convenience functions for quick setup
async def initialize_all_metrics(config=None):
    """Initialize all metrics collectors and start the aggregation engine."""
    from .aggregation_engine import start_aggregation_engine
    await start_aggregation_engine(config)


async def shutdown_all_metrics():
    """Shutdown all metrics collectors and stop the aggregation engine."""
    from .aggregation_engine import stop_aggregation_engine
    await stop_aggregation_engine()


def get_metrics_summary():
    """Get a summary of all metrics collection statistics."""
    summary = {}
    
    try:
        ml_collector = get_ml_metrics_collector()
        summary["ml_metrics"] = ml_collector.get_collection_stats()
    except Exception:
        summary["ml_metrics"] = {"error": "Not available"}
    
    try:
        api_collector = get_api_metrics_collector()
        summary["api_metrics"] = api_collector.get_collection_stats()
    except Exception:
        summary["api_metrics"] = {"error": "Not available"}
    
    try:
        performance_collector = get_performance_metrics_collector()
        summary["performance_metrics"] = performance_collector.get_collection_stats()
    except Exception:
        summary["performance_metrics"] = {"error": "Not available"}
    
    try:
        bi_collector = get_bi_metrics_collector()
        summary["bi_metrics"] = bi_collector.get_collection_stats()
    except Exception:
        summary["bi_metrics"] = {"error": "Not available"}
    
    try:
        aggregation_engine = get_aggregation_engine()
        summary["aggregation_engine"] = aggregation_engine.get_processing_stats()
    except Exception:
        summary["aggregation_engine"] = {"error": "Not available"}
    
    try:
        dashboard_exporter = get_dashboard_exporter()
        summary["dashboard_exports"] = dashboard_exporter.get_export_stats()
    except Exception:
        summary["dashboard_exports"] = {"error": "Not available"}
    
    return summary