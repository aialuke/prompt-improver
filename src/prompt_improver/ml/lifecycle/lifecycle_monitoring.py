"""ML Lifecycle Monitoring

Enhanced monitoring for ML model lifecycle with comprehensive tracking.
# Legacy Prometheus reference removed - now using OpenTelemetry metrics
"""

def create_sla_target(name, metric_name, target_value, compliance_threshold=0.95):
    """Create SLA target for ML lifecycle monitoring."""
    import time
    return SLATarget(sla_id=f"sla_{name.lower().replace(' ', '_')}_{int(time.time())}", name=name, description=f'SLA target for {name}', metric_name=metric_name, target_value=target_value, target_operator='gte', compliance_threshold=compliance_threshold, scope=MonitoringScope.PLATFORM)