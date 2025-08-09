"""Example Usage of the SLO/SLA Monitoring System
=============================================

Demonstrates how to use the comprehensive SLO/SLA monitoring system
with real-world examples and best practices.
"""
import asyncio
import logging
from prompt_improver.monitoring.slo.feature_flag_integration import ErrorBudgetPolicyEnforcer, FeatureFlagManager, PolicyAction
from prompt_improver.monitoring.slo.framework import SLODefinition, SLOTarget, SLOTemplates, SLOTimeWindow, SLOType
from prompt_improver.monitoring.slo.integration import MetricsCollector, OpenTelemetryIntegration
from prompt_improver.monitoring.slo.monitor import SLOMonitor
from prompt_improver.monitoring.slo.reporting import DashboardGenerator, ExecutiveReporter, ReportPeriod, SLOReporter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_basic_slo_setup():
    """Example: Basic SLO setup and monitoring"""
    print('=== Basic SLO Setup Example ===')
    web_service_slo = SLOTemplates.web_service_availability('api-gateway')
    api_latency_slo = SLOTemplates.api_latency('api-gateway')
    error_rate_slo = SLOTemplates.error_rate('api-gateway')
    slo_definition = SLODefinition(name='api_gateway_slos', service_name='api-gateway', description='Comprehensive SLO definition for API Gateway service', owner_team='platform', targets=web_service_slo.targets + api_latency_slo.targets + error_rate_slo.targets)
    print(f'Created SLO definition with {len(slo_definition.targets)} targets')
    slo_monitor = SLOMonitor(slo_definition=slo_definition, redis_url='redis://localhost:6379/5')
    print('Adding sample measurements...')
    for i in range(100):
        success = i % 10 != 0
        slo_monitor.add_measurement(target_name='availability_24h', value=100.0 if success else 0.0)
    print(f'SLO setup complete for {slo_definition.name}')
