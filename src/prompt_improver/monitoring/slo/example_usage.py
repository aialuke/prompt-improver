"""
Example Usage of the SLO/SLA Monitoring System
=============================================

Demonstrates how to use the comprehensive SLO/SLA monitoring system
with real-world examples and best practices.
"""

import asyncio
import logging

from .framework import (
    SLODefinition, SLOTarget, SLOTimeWindow, SLOType, SLOTemplates
)
from .monitor import SLOMonitor
from .reporting import SLOReporter, DashboardGenerator, ExecutiveReporter, ReportPeriod
from .integration import OpenTelemetryIntegration, MetricsCollector, PrometheusRecordingRules
from .feature_flag_integration import FeatureFlagManager, ErrorBudgetPolicyEnforcer, PolicyAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_basic_slo_setup():
    """Example: Basic SLO setup and monitoring"""
    print("=== Basic SLO Setup Example ===")
    
    # 1. Create SLO definition using templates
    web_service_slo = SLOTemplates.web_service_availability("api-gateway")
    api_latency_slo = SLOTemplates.api_latency("api-gateway")
    error_rate_slo = SLOTemplates.error_rate("api-gateway")
    
    # Combine into comprehensive SLO definition
    slo_definition = SLODefinition(
        name="api_gateway_slos",
        service_name="api-gateway",
        description="Comprehensive SLO definition for API Gateway service",
        owner_team="platform",
        targets=web_service_slo.targets + api_latency_slo.targets + error_rate_slo.targets
    )
    
    print(f"Created SLO definition with {len(slo_definition.targets)} targets")
    
    # 2. Setup SLO monitoring
    slo_monitor = SLOMonitor(
        slo_definition=slo_definition,
        redis_url="redis://localhost:6379/5"
    )
    
    # 3. Add some sample measurements
    print("Adding sample measurements...")
    
    # Simulate API requests with varying performance
    for i in range(100):
        # Availability measurements
        success = i % 10 != 0  # 10% failure rate
        slo_monitor.add_measurement(
            target_name="availability_24h",
            value=100.0 if success else 0.0,
            success=success
        )
        
        # Latency measurements (p95)
        import random
        latency = random.gauss(150, 50) if success else random.gauss(800, 200)
        slo_monitor.add_measurement(
            target_name="latency_p95",
            value=max(0, latency),
            success=success
        )
        
        # Error rate measurements
        slo_monitor.add_measurement(
            target_name="error_rate_24h",
            value=0.0 if success else 1.0,
            success=success
        )
    
    # 4. Evaluate SLOs
    print("Evaluating SLOs...")
    results = await slo_monitor.evaluate_slos()
    
    print(f"Evaluation completed. Found {len(results['alerts'])} alerts")
    for target_name, result in results["slo_results"].items():
        if "window_results" in result:
            for window, metrics in result["window_results"].items():
                print(f"  {target_name} ({window}): {metrics['compliance_ratio']:.2f} compliance")
    
    return slo_monitor

async def example_opentelemetry_integration():
    """Example: OpenTelemetry integration for metrics collection"""
    print("\n=== OpenTelemetry Integration Example ===")
    
    # 1. Setup OpenTelemetry integration
    otel_integration = OpenTelemetryIntegration(
        meter_name="slo_monitoring_example",
        enable_prometheus=True,
        prometheus_port=8000
    )
    
    # 2. Create SLO definition
    slo_definition = SLODefinition(
        name="user_service_slos",
        service_name="user-service",
        description="User service SLO monitoring",
        owner_team="backend"
    )
    
    # Add custom SLO targets
    slo_definition.add_target(SLOTarget(
        name="api_availability",
        slo_type=SLOType.AVAILABILITY,
        target_value=99.9,  # 99.9% availability
        time_window=SLOTimeWindow.DAY_1,
        description="API availability over 24 hours"
    ))
    
    slo_definition.add_target(SLOTarget(
        name="response_time_p95",
        slo_type=SLOType.LATENCY,
        target_value=200.0,  # 200ms p95 latency
        time_window=SLOTimeWindow.DAY_1,
        unit="ms",
        description="95th percentile response time"
    ))
    
    # 3. Register SLO metrics with OpenTelemetry
    otel_integration.register_slo_metrics(slo_definition)
    
    print(f"Registered {len(otel_integration.get_registered_metrics())} metrics with OpenTelemetry")
    
    # 4. Setup metrics collector
    metrics_collector = MetricsCollector(
        redis_url="redis://localhost:6379/5",
        collection_interval=10  # Collect every 10 seconds
    )
    
    # Setup SLO monitor
    slo_monitor = SLOMonitor(slo_definition)
    metrics_collector.register_slo_monitor("user-service", slo_monitor)
    metrics_collector.set_opentelemetry_integration(otel_integration)
    
    # 5. Simulate some measurements
    print("Recording measurements to OpenTelemetry...")
    
    for i in range(50):
        import random
        
        # Record API request
        success = random.random() > 0.05  # 95% success rate
        latency = random.gauss(150, 30) if success else random.gauss(1000, 200)
        
        # Record to OpenTelemetry
        for target in slo_definition.targets:
            if target.slo_type == SLOType.AVAILABILITY:
                otel_integration.record_sli_measurement(target, 1.0 if success else 0.0, success)
            elif target.slo_type == SLOType.LATENCY:
                otel_integration.record_sli_measurement(target, max(0, latency), success)
        
        # Also add to SLO monitor for evaluation
        slo_monitor.add_measurement("api_availability", 1.0 if success else 0.0, success=success)
        slo_monitor.add_measurement("response_time_p95", max(0, latency), success=success)
    
    # 6. Generate Prometheus recording rules
    prometheus_rules = PrometheusRecordingRules()
    rules = prometheus_rules.generate_slo_recording_rules([slo_definition])
    
    print(f"Generated {len(rules['groups'])} Prometheus recording rule groups")
    
    # Export rules (optional)
    # prometheus_rules.export_rules_yaml(rules, "user_service_slo_rules.yml")
    
    if otel_integration.get_prometheus_metrics_url():
        print(f"Prometheus metrics available at: {otel_integration.get_prometheus_metrics_url()}")
    
    return slo_monitor, otel_integration

async def example_error_budget_and_alerting():
    """Example: Error budget monitoring and burn rate alerting"""
    print("\n=== Error Budget and Alerting Example ===")
    
    # 1. Create SLO definition with error budget policy
    slo_definition = SLODefinition(
        name="payment_service_slos",
        service_name="payment-service",
        description="Payment service SLO with strict error budget policy",
        owner_team="payments",
        error_budget_policy="rollback_features"  # Automatic feature rollback
    )
    
    # Add critical SLO targets
    slo_definition.add_target(SLOTarget(
        name="payment_availability",
        slo_type=SLOType.AVAILABILITY,
        target_value=99.95,  # 99.95% availability (very strict)
        time_window=SLOTimeWindow.DAY_1,
        description="Payment processing availability"
    ))
    
    slo_definition.add_target(SLOTarget(
        name="payment_latency_p99",
        slo_type=SLOType.LATENCY,
        target_value=500.0,  # 500ms p99 latency
        time_window=SLOTimeWindow.DAY_1,
        unit="ms",
        description="99th percentile payment processing time"
    ))
    
    # 2. Setup alert callbacks
    def alert_callback(alert):
        print(f"🚨 ALERT: {alert.severity.value.upper()} - {alert.message}")
        print(f"   Service: {alert.service_name}")
        print(f"   Burn Rate: {alert.burn_rate:.1f}x")
        print(f"   Current Value: {alert.current_value}")
    
    # 3. Setup SLO monitoring with alerting
    slo_monitor = SLOMonitor(
        slo_definition=slo_definition,
        alert_callbacks=[alert_callback]
    )
    
    # 4. Setup feature flag integration
    feature_flag_manager = FeatureFlagManager(redis_url="redis://localhost:6379/5")
    
    # Register feature flags for the payment service
    feature_flag_manager.register_feature_flag(
        name="new_payment_processor",
        description="New payment processing algorithm",
        service_name="payment-service",
        rollback_priority=1,  # High priority for rollback
        rollback_impact="high"
    )
    
    feature_flag_manager.register_feature_flag(
        name="advanced_fraud_detection",
        description="Advanced fraud detection system",
        service_name="payment-service",
        rollback_priority=2,  # Lower priority
        rollback_impact="medium"
    )
    
    # Set rollback policy
    feature_flag_manager.set_rollback_policy(
        service_name="payment-service",
        error_budget_threshold=85.0,  # Trigger at 85% budget consumption
        actions=[
            PolicyAction.ALERT_ONLY,
            PolicyAction.ROLLBACK_FEATURES,
            PolicyAction.BLOCK_DEPLOYS
        ]
    )
    
    # 5. Setup policy enforcer
    policy_enforcer = ErrorBudgetPolicyEnforcer(feature_flag_manager)
    await policy_enforcer.setup_error_budget_monitoring(
        slo_monitor.error_budget_monitor,
        slo_definition
    )
    
    # 6. Simulate a service degradation scenario
    print("Simulating service degradation...")
    
    # Normal operation
    for i in range(100):
        success = True
        latency = 150.0
        
        slo_monitor.add_measurement("payment_availability", 1.0, success=success)
        slo_monitor.add_measurement("payment_latency_p99", latency, success=success)
    
    # Service degradation - increased failures and latency
    print("Introducing service issues...")
    for i in range(200):
        import random
        success = random.random() > 0.15  # 15% failure rate (high!)
        latency = random.gauss(800, 200) if not success else random.gauss(200, 50)
        
        slo_monitor.add_measurement("payment_availability", 1.0 if success else 0.0, success=success)
        slo_monitor.add_measurement("payment_latency_p99", max(0, latency), success=success)
    
    # 7. Evaluate SLOs and trigger alerts
    print("Evaluating SLOs after degradation...")
    results = await slo_monitor.evaluate_slos()
    
    print(f"Generated {len(results['alerts'])} alerts")
    print(f"Error budget status: {results['error_budget_status']['overall_status']}")
    
    # 8. Check feature flag status
    flag_status = feature_flag_manager.get_feature_flag_status("payment-service")
    print(f"Feature flags rolled back: {flag_status['status_summary']['rollback']}")
    
    return slo_monitor, feature_flag_manager

async def example_reporting_and_dashboards():
    """Example: Generate reports and dashboards"""
    print("\n=== Reporting and Dashboards Example ===")
    
    # Setup SLO monitor with some historical data
    slo_monitor = await example_basic_slo_setup()
    
    # 1. Generate SLO compliance report
    slo_reporter = SLOReporter(output_dir="./reports")
    
    print("Generating SLO compliance report...")
    compliance_report = await slo_reporter.generate_compliance_report(
        slo_monitor=slo_monitor,
        period=ReportPeriod.DAILY
    )
    
    print(f"Report generated: {compliance_report.overall_compliance_percentage:.2f}% compliance")
    print(f"Status: {compliance_report.compliance_status}")
    print(f"Recommendations: {len(compliance_report.recommendations)}")
    
    # 2. Export report in different formats
    json_file = slo_reporter.export_report(compliance_report, format=slo_reporter.ReportFormat.JSON)
    html_file = slo_reporter.export_report(compliance_report, format=slo_reporter.ReportFormat.HTML)
    md_file = slo_reporter.export_report(compliance_report, format=slo_reporter.ReportFormat.MARKDOWN)
    
    print(f"Reports exported:")
    print(f"  JSON: {json_file}")
    print(f"  HTML: {html_file}")
    print(f"  Markdown: {md_file}")
    
    # 3. Generate dashboard
    dashboard_generator = DashboardGenerator(output_dir="./dashboards")
    
    dashboard_file = dashboard_generator.generate_slo_dashboard(
        slo_reports=[compliance_report],
        title="API Gateway SLO Dashboard"
    )
    
    print(f"Dashboard generated: {dashboard_file}")
    
    # 4. Generate executive summary
    executive_reporter = ExecutiveReporter()
    
    executive_summary = executive_reporter.generate_executive_summary(
        slo_reports=[compliance_report],
        sla_breaches=[],  # No breaches in this example
        period=ReportPeriod.DAILY
    )
    
    print(f"Executive Summary:")
    print(f"  Service Health: {executive_summary['executive_summary']['overall_service_health']:.1f}%")
    print(f"  Risk Level: {executive_summary['executive_summary']['risk_level']}")
    print(f"  Recommendations: {len(executive_summary['recommendations'])}")
    
    return compliance_report, dashboard_file

async def example_complete_monitoring_setup():
    """Example: Complete end-to-end monitoring setup"""
    print("\n=== Complete Monitoring Setup Example ===")
    
    # 1. Create comprehensive SLO definition
    slo_definition = SLODefinition(
        name="ecommerce_platform_slos",
        service_name="ecommerce-platform",
        description="Complete SLO monitoring for e-commerce platform",
        owner_team="platform",
        error_budget_policy="rollback_features"
    )
    
    # Add multiple SLO targets
    targets = [
        # Availability SLOs
        SLOTarget(
            name="web_availability",
            slo_type=SLOType.AVAILABILITY,
            target_value=99.9,
            time_window=SLOTimeWindow.DAY_1,
            description="Web frontend availability"
        ),
        SLOTarget(
            name="api_availability",
            slo_type=SLOType.AVAILABILITY,
            target_value=99.95,
            time_window=SLOTimeWindow.DAY_1,
            description="API availability"
        ),
        
        # Latency SLOs
        SLOTarget(
            name="page_load_p95",
            slo_type=SLOType.LATENCY,
            target_value=2000.0,
            time_window=SLOTimeWindow.DAY_1,
            unit="ms",
            description="95th percentile page load time"
        ),
        SLOTarget(
            name="api_latency_p99",
            slo_type=SLOType.LATENCY,
            target_value=500.0,
            time_window=SLOTimeWindow.DAY_1,
            unit="ms",
            description="99th percentile API response time"
        ),
        
        # Error rate SLOs
        SLOTarget(
            name="checkout_error_rate",
            slo_type=SLOType.ERROR_RATE,
            target_value=0.1,  # 0.1% error rate
            time_window=SLOTimeWindow.DAY_1,
            description="Checkout process error rate"
        )
    ]
    
    for target in targets:
        slo_definition.add_target(target)
    
    print(f"Created comprehensive SLO definition with {len(slo_definition.targets)} targets")
    
    # 2. Setup OpenTelemetry integration
    otel_integration = OpenTelemetryIntegration(
        meter_name="ecommerce_slo_monitoring",
        enable_prometheus=True
    )
    otel_integration.register_slo_metrics(slo_definition)
    
    # 3. Setup SLO monitoring
    slo_monitor = SLOMonitor(
        slo_definition=slo_definition,
        redis_url="redis://localhost:6379/5"
    )
    
    # 4. Setup feature flag management
    feature_flag_manager = FeatureFlagManager(redis_url="redis://localhost:6379/5")
    
    # Register feature flags
    flags = [
        ("new_checkout_flow", "New checkout flow with improved UX", 1, "high"),
        ("recommendation_engine", "ML-based product recommendations", 2, "medium"),
        ("advanced_search", "Enhanced search functionality", 3, "low"),
        ("premium_features", "Premium user features", 4, "medium")
    ]
    
    for name, description, priority, impact in flags:
        feature_flag_manager.register_feature_flag(
            name=name,
            description=description,
            service_name="ecommerce-platform",
            rollback_priority=priority,
            rollback_impact=impact
        )
    
    # Set aggressive rollback policy
    feature_flag_manager.set_rollback_policy(
        service_name="ecommerce-platform",
        error_budget_threshold=80.0,
        actions=[PolicyAction.ROLLBACK_FEATURES, PolicyAction.BLOCK_DEPLOYS]
    )
    
    # 5. Setup policy enforcement
    policy_enforcer = ErrorBudgetPolicyEnforcer(feature_flag_manager)
    await policy_enforcer.setup_error_budget_monitoring(
        slo_monitor.error_budget_monitor,
        slo_definition
    )
    
    # 6. Setup metrics collection
    metrics_collector = MetricsCollector(
        redis_url="redis://localhost:6379/5",
        collection_interval=30
    )
    metrics_collector.register_slo_monitor("ecommerce-platform", slo_monitor)
    metrics_collector.set_opentelemetry_integration(otel_integration)
    
    # 7. Start monitoring
    print("Starting comprehensive monitoring...")
    await slo_monitor.start_monitoring()
    await metrics_collector.start_collection()
    
    # 8. Simulate realistic traffic patterns
    print("Simulating realistic e-commerce traffic...")
    
    import random
    import asyncio
    
    # Simulate 24 hours of traffic (compressed to 10 seconds)
    for hour in range(24):
        # Traffic varies by hour (peak during business hours)
        traffic_multiplier = 0.3 + 0.7 * (0.5 + 0.5 * abs(12 - hour) / 12)
        requests_this_hour = int(100 * traffic_multiplier)
        
        for request in range(requests_this_hour):
            # Vary success rates and latencies by time of day
            base_success_rate = 0.995
            if 9 <= hour <= 17:  # Business hours - higher load, slightly more errors
                success_rate = base_success_rate - 0.002
                latency_multiplier = 1.2
            elif 22 <= hour or hour <= 6:  # Night hours - maintenance window
                success_rate = base_success_rate - 0.001
                latency_multiplier = 0.8
            else:
                success_rate = base_success_rate
                latency_multiplier = 1.0
            
            # Generate measurements for each target
            for target in slo_definition.targets:
                success = random.random() < success_rate
                
                if target.slo_type == SLOType.AVAILABILITY:
                    value = 1.0 if success else 0.0
                elif target.slo_type == SLOType.LATENCY:
                    if "page_load" in target.name:
                        base_latency = random.gauss(1200, 300) * latency_multiplier
                    else:  # API latency
                        base_latency = random.gauss(150, 50) * latency_multiplier
                    
                    value = max(0, base_latency * (5.0 if not success else 1.0))
                elif target.slo_type == SLOType.ERROR_RATE:
                    value = 0.0 if success else 1.0
                else:
                    value = 1.0
                
                # Record measurements
                slo_monitor.add_measurement(target.name, value, success=success)
                otel_integration.record_sli_measurement(target, value, success)
        
        # Small delay to simulate time progression
        await asyncio.sleep(0.01)
    
    # 9. Evaluate final results
    print("Evaluating comprehensive monitoring results...")
    results = await slo_monitor.evaluate_slos()
    
    print("\n=== Final Results ===")
    print(f"Alerts generated: {len(results['alerts'])}")
    print(f"Error budget status: {results['error_budget_status']['overall_status']}")
    
    # SLO compliance summary
    for target_name, result in results["slo_results"].items():
        if "window_results" in result:
            window_result = result["window_results"].get("24h", {})
            compliance = window_result.get("compliance_ratio", 0.0)
            is_compliant = window_result.get("is_compliant", False)
            status = "✅ PASS" if is_compliant else "❌ FAIL"
            print(f"  {target_name}: {compliance:.3f} compliance {status}")
    
    # Feature flag status
    flag_status = feature_flag_manager.get_feature_flag_status("ecommerce-platform")
    print(f"\nFeature Flags:")
    print(f"  Enabled: {flag_status['status_summary']['enabled']}")
    print(f"  Rolled back: {flag_status['status_summary']['rollback']}")
    
    # Generate final report
    slo_reporter = SLOReporter()
    final_report = await slo_reporter.generate_compliance_report(
        slo_monitor, ReportPeriod.DAILY
    )
    
    print(f"\nFinal Report:")
    print(f"  Overall Compliance: {final_report.overall_compliance_percentage:.2f}%")
    print(f"  Status: {final_report.compliance_status}")
    print(f"  Error Budget Consumed: {final_report.error_budget_consumed_percentage:.1f}%")
    print(f"  Recommendations: {len(final_report.recommendations)}")
    
    # Cleanup
    await slo_monitor.stop_monitoring()
    await metrics_collector.stop_collection()
    
    return {
        "slo_monitor": slo_monitor,
        "otel_integration": otel_integration,
        "feature_flag_manager": feature_flag_manager,
        "final_report": final_report,
        "results": results
    }

async def main():
    """Run all examples"""
    print("🚀 SLO/SLA Monitoring System Examples\n")
    
    try:
        # Run examples in sequence
        await example_basic_slo_setup()
        await example_opentelemetry_integration()
        await example_error_budget_and_alerting()
        await example_reporting_and_dashboards()
        
        # Run comprehensive example
        final_results = await example_complete_monitoring_setup()
        
        print("\n✅ All examples completed successfully!")
        print(f"Final system evaluation: {final_results['final_report'].compliance_status}")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        logger.exception("Example execution error")

if __name__ == "__main__":
    asyncio.run(main())