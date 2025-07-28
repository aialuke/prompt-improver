"""
OpenTelemetry Integration and Metrics Collection
===============================================

Implements integration with OpenTelemetry metrics, Prometheus recording rules,
and automated metric collection for SLO/SLA monitoring systems.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    from opentelemetry import metrics
    from opentelemetry.metrics import MeterProvider, Meter
    from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
    from opentelemetry.sdk.metrics.export import MetricExporter, MetricReader
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    metrics = MeterProvider = Meter = SDKMeterProvider = None
    MetricExporter = MetricReader = PrometheusMetricReader = None

try:
    import coredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    coredis = None

from .framework import SLODefinition, SLOTarget, SLOType
from .monitor import SLOMonitor

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """OpenTelemetry metric types for SLO monitoring"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    UP_DOWN_COUNTER = "up_down_counter"

@dataclass
class SLOMetricDefinition:
    """Definition of SLO-related metric"""
    name: str
    description: str
    unit: str
    metric_type: MetricType
    slo_target: SLOTarget
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_otel_name(self) -> str:
        """Convert to OpenTelemetry metric name format"""
        return f"slo_{self.slo_target.service_name}_{self.name}".replace("-", "_").replace(".", "_")

class OpenTelemetryIntegration:
    """Integration with OpenTelemetry for SLO metrics collection"""
    
    def __init__(
        self,
        meter_name: str = "slo_monitoring",
        version: str = "1.0.0",
        enable_prometheus: bool = True,
        prometheus_port: int = 8000
    ):
        self.meter_name = meter_name
        self.version = version
        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port
        
        # OpenTelemetry components
        self.meter_provider: Optional[MeterProvider] = None
        self.meter: Optional[Meter] = None
        self.prometheus_reader: Optional[PrometheusMetricReader] = None
        
        # Metric instruments
        self.instruments: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, SLOMetricDefinition] = {}
        
        # Initialize if OpenTelemetry is available
        if OTEL_AVAILABLE:
            self._initialize_opentelemetry()
        else:
            logger.warning("OpenTelemetry not available, metrics collection disabled")
    
    def _initialize_opentelemetry(self) -> None:
        """Initialize OpenTelemetry components"""
        try:
            # Create metric readers
            readers = []
            
            if self.enable_prometheus:
                self.prometheus_reader = PrometheusMetricReader()
                readers.append(self.prometheus_reader)
            
            # Create meter provider
            self.meter_provider = SDKMeterProvider(metric_readers=readers)
            metrics.set_meter_provider(self.meter_provider)
            
            # Get meter
            self.meter = self.meter_provider.get_meter(
                name=self.meter_name,
                version=self.version
            )
            
            logger.info("OpenTelemetry integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.meter = None
    
    def register_slo_metrics(self, slo_definition: SLODefinition) -> None:
        """Register metrics for SLO definition"""
        if not self.meter:
            return
        
        for target in slo_definition.targets:
            self._register_target_metrics(target)
    
    def _register_target_metrics(self, target: SLOTarget) -> None:
        """Register metrics for specific SLO target"""
        base_labels = {
            "service_name": target.service_name,
            "slo_target": target.name,
            "slo_type": target.slo_type.value
        }
        
        # SLI value metric
        sli_metric = SLOMetricDefinition(
            name="sli_value",
            description=f"Current SLI value for {target.name}",
            unit=target.unit,
            metric_type=MetricType.GAUGE,
            slo_target=target,
            labels=base_labels
        )
        self._create_metric_instrument(sli_metric)
        
        # SLO compliance ratio
        compliance_metric = SLOMetricDefinition(
            name="compliance_ratio",
            description=f"SLO compliance ratio for {target.name}",
            unit="ratio",
            metric_type=MetricType.GAUGE,
            slo_target=target,
            labels=base_labels
        )
        self._create_metric_instrument(compliance_metric)
        
        # Error budget consumption
        budget_metric = SLOMetricDefinition(
            name="error_budget_consumed",
            description=f"Error budget consumed for {target.name}",
            unit="percent",
            metric_type=MetricType.GAUGE,
            slo_target=target,
            labels=base_labels
        )
        self._create_metric_instrument(budget_metric)
        
        # Burn rate
        burn_rate_metric = SLOMetricDefinition(
            name="burn_rate",
            description=f"Error budget burn rate for {target.name}",
            unit="rate",
            metric_type=MetricType.GAUGE,
            slo_target=target,
            labels=base_labels
        )
        self._create_metric_instrument(burn_rate_metric)
        
        # Request counts for availability/error rate SLOs
        if target.slo_type in [SLOType.AVAILABILITY, SLOType.ERROR_RATE]:
            requests_metric = SLOMetricDefinition(
                name="requests_total",
                description=f"Total requests for {target.name}",
                unit="1",
                metric_type=MetricType.COUNTER,
                slo_target=target,
                labels=base_labels
            )
            self._create_metric_instrument(requests_metric)
            
            failed_requests_metric = SLOMetricDefinition(
                name="failed_requests_total",
                description=f"Failed requests for {target.name}",
                unit="1",
                metric_type=MetricType.COUNTER,
                slo_target=target,
                labels=base_labels
            )
            self._create_metric_instrument(failed_requests_metric)
        
        # Response time histogram for latency SLOs
        if target.slo_type == SLOType.LATENCY:
            latency_metric = SLOMetricDefinition(
                name="response_time",
                description=f"Response time distribution for {target.name}",
                unit="ms",
                metric_type=MetricType.HISTOGRAM,
                slo_target=target,
                labels=base_labels
            )
            self._create_metric_instrument(latency_metric)
    
    def _create_metric_instrument(self, metric_def: SLOMetricDefinition) -> None:
        """Create OpenTelemetry metric instrument"""
        if not self.meter:
            return
        
        metric_name = metric_def.to_otel_name()
        
        try:
            if metric_def.metric_type == MetricType.COUNTER:
                instrument = self.meter.create_counter(
                    name=metric_name,
                    description=metric_def.description,
                    unit=metric_def.unit
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                instrument = self.meter.create_histogram(
                    name=metric_name,
                    description=metric_def.description,
                    unit=metric_def.unit
                )
            elif metric_def.metric_type == MetricType.GAUGE:
                instrument = self.meter.create_gauge(
                    name=metric_name,
                    description=metric_def.description,
                    unit=metric_def.unit
                )
            elif metric_def.metric_type == MetricType.UP_DOWN_COUNTER:
                instrument = self.meter.create_up_down_counter(
                    name=metric_name,
                    description=metric_def.description,
                    unit=metric_def.unit
                )
            else:
                logger.warning(f"Unknown metric type: {metric_def.metric_type}")
                return
            
            self.instruments[metric_name] = instrument
            self.metric_definitions[metric_name] = metric_def
            
            logger.debug(f"Created metric instrument: {metric_name}")
            
        except Exception as e:
            logger.error(f"Failed to create metric instrument {metric_name}: {e}")
    
    def record_sli_measurement(
        self,
        target: SLOTarget,
        value: float,
        success: bool = True,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record SLI measurement to OpenTelemetry metrics"""
        if not self.meter:
            return
        
        combined_labels = {
            "service_name": target.service_name,
            "slo_target": target.name,
            "slo_type": target.slo_type.value
        }
        if labels:
            combined_labels.update(labels)
        
        # Record based on SLO type
        if target.slo_type == SLOType.AVAILABILITY:
            self._record_availability_metrics(target, success, combined_labels)
        elif target.slo_type == SLOType.LATENCY:
            self._record_latency_metrics(target, value, combined_labels)
        elif target.slo_type == SLOType.ERROR_RATE:
            self._record_error_rate_metrics(target, success, combined_labels)
        elif target.slo_type == SLOType.THROUGHPUT:
            self._record_throughput_metrics(target, value, combined_labels)
        
        # Always record the SLI value
        sli_gauge_name = f"slo_{target.service_name}_sli_value".replace("-", "_").replace(".", "_")
        if sli_gauge_name in self.instruments:
            self.instruments[sli_gauge_name].set(value, combined_labels)
    
    def _record_availability_metrics(
        self,
        target: SLOTarget,
        success: bool,
        labels: Dict[str, str]
    ) -> None:
        """Record availability-specific metrics"""
        # Total requests
        requests_counter_name = f"slo_{target.service_name}_requests_total".replace("-", "_").replace(".", "_")
        if requests_counter_name in self.instruments:
            self.instruments[requests_counter_name].add(1, labels)
        
        # Failed requests
        if not success:
            failed_counter_name = f"slo_{target.service_name}_failed_requests_total".replace("-", "_").replace(".", "_")
            if failed_counter_name in self.instruments:
                self.instruments[failed_counter_name].add(1, labels)
    
    def _record_latency_metrics(
        self,
        target: SLOTarget,
        latency_ms: float,
        labels: Dict[str, str]
    ) -> None:
        """Record latency-specific metrics"""
        histogram_name = f"slo_{target.service_name}_response_time".replace("-", "_").replace(".", "_")
        if histogram_name in self.instruments:
            self.instruments[histogram_name].record(latency_ms, labels)
    
    def _record_error_rate_metrics(
        self,
        target: SLOTarget,
        success: bool,
        labels: Dict[str, str]
    ) -> None:
        """Record error rate specific metrics"""
        # Total requests
        requests_counter_name = f"slo_{target.service_name}_requests_total".replace("-", "_").replace(".", "_")
        if requests_counter_name in self.instruments:
            self.instruments[requests_counter_name].add(1, labels)
        
        # Failed requests (errors)
        if not success:
            failed_counter_name = f"slo_{target.service_name}_failed_requests_total".replace("-", "_").replace(".", "_")
            if failed_counter_name in self.instruments:
                self.instruments[failed_counter_name].add(1, labels)
    
    def _record_throughput_metrics(
        self,
        target: SLOTarget,
        throughput: float,
        labels: Dict[str, str]
    ) -> None:
        """Record throughput-specific metrics"""
        # Throughput is typically recorded as a gauge showing current rate
        gauge_name = f"slo_{target.service_name}_sli_value".replace("-", "_").replace(".", "_")
        if gauge_name in self.instruments:
            self.instruments[gauge_name].set(throughput, labels)
    
    def update_slo_compliance_metrics(
        self,
        target: SLOTarget,
        compliance_ratio: float,
        error_budget_consumed: float,
        burn_rate: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Update SLO compliance and error budget metrics"""
        if not self.meter:
            return
        
        combined_labels = {
            "service_name": target.service_name,
            "slo_target": target.name,
            "slo_type": target.slo_type.value
        }
        if labels:
            combined_labels.update(labels)
        
        # Compliance ratio
        compliance_gauge_name = f"slo_{target.service_name}_compliance_ratio".replace("-", "_").replace(".", "_")
        if compliance_gauge_name in self.instruments:
            self.instruments[compliance_gauge_name].set(compliance_ratio, combined_labels)
        
        # Error budget consumed
        budget_gauge_name = f"slo_{target.service_name}_error_budget_consumed".replace("-", "_").replace(".", "_")
        if budget_gauge_name in self.instruments:
            self.instruments[budget_gauge_name].set(error_budget_consumed, combined_labels)
        
        # Burn rate
        burn_rate_gauge_name = f"slo_{target.service_name}_burn_rate".replace("-", "_").replace(".", "_")
        if burn_rate_gauge_name in self.instruments:
            self.instruments[burn_rate_gauge_name].set(burn_rate, combined_labels)
    
    def get_prometheus_metrics_url(self) -> Optional[str]:
        """Get Prometheus metrics endpoint URL"""
        if self.prometheus_reader and self.enable_prometheus:
            return f"http://localhost:{self.prometheus_port}/metrics"
        return None
    
    def get_registered_metrics(self) -> List[Dict[str, Any]]:
        """Get list of registered metrics"""
        return [
            {
                "name": name,
                "description": metric_def.description,
                "unit": metric_def.unit,
                "type": metric_def.metric_type.value,
                "slo_target": metric_def.slo_target.name,
                "service": metric_def.slo_target.service_name
            }
            for name, metric_def in self.metric_definitions.items()
        ]

class PrometheusRecordingRules:
    """Generate Prometheus recording rules for efficient SLO metric aggregation"""
    
    def __init__(self, rule_namespace: str = "slo"):
        self.rule_namespace = rule_namespace
        
    def generate_slo_recording_rules(
        self,
        slo_definitions: List[SLODefinition]
    ) -> Dict[str, Any]:
        """Generate Prometheus recording rules for SLO definitions"""
        rule_groups = []
        
        for slo_def in slo_definitions:
            group = self._generate_service_rule_group(slo_def)
            if group:
                rule_groups.append(group)
        
        return {
            "groups": rule_groups
        }
    
    def _generate_service_rule_group(self, slo_def: SLODefinition) -> Dict[str, Any]:
        """Generate recording rule group for a service"""
        rules = []
        
        for target in slo_def.targets:
            target_rules = self._generate_target_rules(target)
            rules.extend(target_rules)
        
        if not rules:
            return None
        
        return {
            "name": f"{self.rule_namespace}:{slo_def.service_name}",
            "interval": "30s",
            "rules": rules
        }
    
    def _generate_target_rules(self, target: SLOTarget) -> List[Dict[str, Any]]:
        """Generate recording rules for specific SLO target"""
        rules = []
        service = target.service_name
        target_name = target.name
        
        if target.slo_type == SLOType.AVAILABILITY:
            rules.extend(self._generate_availability_rules(service, target_name))
        elif target.slo_type == SLOType.LATENCY:
            rules.extend(self._generate_latency_rules(service, target_name))
        elif target.slo_type == SLOType.ERROR_RATE:
            rules.extend(self._generate_error_rate_rules(service, target_name))
        elif target.slo_type == SLOType.THROUGHPUT:
            rules.extend(self._generate_throughput_rules(service, target_name))
        
        # Add burn rate rules
        rules.extend(self._generate_burn_rate_rules(service, target_name, target))
        
        return rules
    
    def _generate_availability_rules(self, service: str, target: str) -> List[Dict[str, Any]]:
        """Generate availability recording rules"""
        return [
            {
                "record": f"{self.rule_namespace}:availability_rate_1h",
                "expr": f"""
                    sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[1h])) - 
                    sum(rate(slo_{service}_failed_requests_total{{slo_target="{target}"}}[1h])) 
                    /
                    sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[1h]))
                """.strip(),
                "labels": {
                    "service": service,
                    "slo_target": target,
                    "slo_type": "availability",
                    "window": "1h"
                }
            },
            {
                "record": f"{self.rule_namespace}:availability_rate_24h",
                "expr": f"""
                    sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[24h])) - 
                    sum(rate(slo_{service}_failed_requests_total{{slo_target="{target}"}}[24h])) 
                    /
                    sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[24h]))
                """.strip(),
                "labels": {
                    "service": service,
                    "slo_target": target,
                    "slo_type": "availability",
                    "window": "24h"
                }
            },
            {
                "record": f"{self.rule_namespace}:availability_rate_7d",
                "expr": f"""
                    sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[7d])) - 
                    sum(rate(slo_{service}_failed_requests_total{{slo_target="{target}"}}[7d])) 
                    /
                    sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[7d]))
                """.strip(),
                "labels": {
                    "service": service,
                    "slo_target": target,
                    "slo_type": "availability",
                    "window": "7d"
                }
            }
        ]
    
    def _generate_latency_rules(self, service: str, target: str) -> List[Dict[str, Any]]:
        """Generate latency recording rules"""
        return [
            {
                "record": f"{self.rule_namespace}:latency_p50_1h",
                "expr": f'histogram_quantile(0.50, rate(slo_{service}_response_time_bucket{{slo_target="{target}"}}[1h]))',
                "labels": {
                    "service": service,
                    "slo_target": target,
                    "slo_type": "latency",
                    "percentile": "50",
                    "window": "1h"
                }
            },
            {
                "record": f"{self.rule_namespace}:latency_p95_1h",
                "expr": f'histogram_quantile(0.95, rate(slo_{service}_response_time_bucket{{slo_target="{target}"}}[1h]))',
                "labels": {
                    "service": service,
                    "slo_target": target,
                    "slo_type": "latency",
                    "percentile": "95",
                    "window": "1h"
                }
            },
            {
                "record": f"{self.rule_namespace}:latency_p99_1h",
                "expr": f'histogram_quantile(0.99, rate(slo_{service}_response_time_bucket{{slo_target="{target}"}}[1h]))',
                "labels": {
                    "service": service,
                    "slo_target": target,
                    "slo_type": "latency",
                    "percentile": "99",
                    "window": "1h"
                }
            },
            {
                "record": f"{self.rule_namespace}:latency_p95_24h",
                "expr": f'histogram_quantile(0.95, rate(slo_{service}_response_time_bucket{{slo_target="{target}"}}[24h]))',
                "labels": {
                    "service": service,
                    "slo_target": target,
                    "slo_type": "latency",
                    "percentile": "95",
                    "window": "24h"
                }
            }
        ]
    
    def _generate_error_rate_rules(self, service: str, target: str) -> List[Dict[str, Any]]:
        """Generate error rate recording rules"""
        return [
            {
                "record": f"{self.rule_namespace}:error_rate_1h",
                "expr": f"""
                    sum(rate(slo_{service}_failed_requests_total{{slo_target="{target}"}}[1h])) 
                    /
                    sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[1h]))
                """.strip(),
                "labels": {
                    "service": service,
                    "slo_target": target,
                    "slo_type": "error_rate",
                    "window": "1h"
                }
            },
            {
                "record": f"{self.rule_namespace}:error_rate_24h",
                "expr": f"""
                    sum(rate(slo_{service}_failed_requests_total{{slo_target="{target}"}}[24h])) 
                    /
                    sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[24h]))
                """.strip(),
                "labels": {
                    "service": service,
                    "slo_target": target,
                    "slo_type": "error_rate",
                    "window": "24h"
                }
            }
        ]
    
    def _generate_throughput_rules(self, service: str, target: str) -> List[Dict[str, Any]]:
        """Generate throughput recording rules"""
        return [
            {
                "record": f"{self.rule_namespace}:throughput_1h",
                "expr": f'sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[1h]))',
                "labels": {
                    "service": service,
                    "slo_target": target,
                    "slo_type": "throughput",
                    "window": "1h"
                }
            },
            {
                "record": f"{self.rule_namespace}:throughput_24h",
                "expr": f'sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[24h]))',
                "labels": {
                    "service": service,
                    "slo_target": target,
                    "slo_type": "throughput",
                    "window": "24h"
                }
            }
        ]
    
    def _generate_burn_rate_rules(
        self,
        service: str,
        target: str,
        slo_target: SLOTarget
    ) -> List[Dict[str, Any]]:
        """Generate burn rate recording rules"""
        target_value = slo_target.target_value
        
        if slo_target.slo_type == SLOType.AVAILABILITY:
            # Burn rate = (1 - availability) / (1 - target_availability)
            allowed_error_rate = (100 - target_value) / 100
            
            return [
                {
                    "record": f"{self.rule_namespace}:burn_rate_1h",
                    "expr": f"""
                        (
                            sum(rate(slo_{service}_failed_requests_total{{slo_target="{target}"}}[1h])) 
                            /
                            sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[1h]))
                        ) / {allowed_error_rate}
                    """.strip(),
                    "labels": {
                        "service": service,
                        "slo_target": target,
                        "slo_type": "availability",
                        "window": "1h"
                    }
                },
                {
                    "record": f"{self.rule_namespace}:burn_rate_6h",
                    "expr": f"""
                        (
                            sum(rate(slo_{service}_failed_requests_total{{slo_target="{target}"}}[6h])) 
                            /
                            sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[6h]))
                        ) / {allowed_error_rate}
                    """.strip(),
                    "labels": {
                        "service": service,
                        "slo_target": target,
                        "slo_type": "availability",
                        "window": "6h"
                    }
                }
            ]
        
        elif slo_target.slo_type == SLOType.ERROR_RATE:
            # Burn rate = actual_error_rate / target_error_rate
            target_error_rate = target_value / 100
            
            return [
                {
                    "record": f"{self.rule_namespace}:burn_rate_1h",
                    "expr": f"""
                        (
                            sum(rate(slo_{service}_failed_requests_total{{slo_target="{target}"}}[1h])) 
                            /
                            sum(rate(slo_{service}_requests_total{{slo_target="{target}"}}[1h]))
                        ) / {target_error_rate}
                    """.strip(),
                    "labels": {
                        "service": service,
                        "slo_target": target,
                        "slo_type": "error_rate",
                        "window": "1h"
                    }
                }
            ]
        
        return []
    
    def export_rules_yaml(self, rules: Dict[str, Any], filename: str = "slo_recording_rules.yml") -> str:
        """Export recording rules as YAML file"""
        import yaml
        
        with open(filename, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False, sort_keys=False)
        
        return filename

class MetricsCollector:
    """Automated metrics collection from various sources"""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        collection_interval: int = 30
    ):
        self.redis_url = redis_url
        self.collection_interval = collection_interval
        self._redis_client = None
        
        # Data sources
        self.data_sources: List[Callable] = []
        self.slo_monitors: Dict[str, SLOMonitor] = {}
        
        # Collection state
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        
        # OpenTelemetry integration
        self.otel_integration: Optional[OpenTelemetryIntegration] = None
    
    async def get_redis_client(self) -> Optional[coredis.Redis]:
        """Get Redis client for metrics storage"""
        if not REDIS_AVAILABLE or not self.redis_url:
            return None
            
        if self._redis_client is None:
            try:
                self._redis_client = coredis.Redis.from_url(self.redis_url, decode_responses=True)
                await self._redis_client.ping()
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                return None
        
        return self._redis_client
    
    def set_opentelemetry_integration(self, integration: OpenTelemetryIntegration) -> None:
        """Set OpenTelemetry integration for metrics export"""
        self.otel_integration = integration
    
    def register_slo_monitor(self, name: str, monitor: SLOMonitor) -> None:
        """Register SLO monitor for metrics collection"""
        self.slo_monitors[name] = monitor
        
        # Register SLO metrics with OpenTelemetry
        if self.otel_integration:
            self.otel_integration.register_slo_metrics(monitor.slo_definition)
    
    def register_data_source(self, collector_func: Callable) -> None:
        """Register external data source for metrics collection"""
        self.data_sources.append(collector_func)
    
    async def start_collection(self) -> None:
        """Start automated metrics collection"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")
    
    async def stop_collection(self) -> None:
        """Stop metrics collection"""
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self) -> None:
        """Main collection loop"""
        while self.is_collecting:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_metrics(self) -> None:
        """Collect metrics from all sources"""
        # Collect from SLO monitors
        for name, monitor in self.slo_monitors.items():
            try:
                await self._collect_slo_metrics(name, monitor)
            except Exception as e:
                logger.error(f"Failed to collect metrics from SLO monitor {name}: {e}")
        
        # Collect from external data sources
        for data_source in self.data_sources:
            try:
                if asyncio.iscoroutinefunction(data_source):
                    await data_source()
                else:
                    data_source()
            except Exception as e:
                logger.error(f"Failed to collect metrics from data source: {e}")
    
    async def _collect_slo_metrics(self, monitor_name: str, monitor: SLOMonitor) -> None:
        """Collect metrics from SLO monitor"""
        try:
            # Get evaluation results
            results = await monitor.evaluate_slos()
            
            # Process each SLO target
            for target_name, calculator in monitor.calculators.items():
                target = next(
                    t for t in monitor.slo_definition.targets 
                    if t.name == target_name
                )
                
                # Get window results
                window_results = calculator.calculate_all_windows()
                
                # Update OpenTelemetry metrics
                if self.otel_integration:
                    for window, result in window_results.items():
                        # Calculate compliance and error budget metrics
                        compliance_ratio = result.compliance_ratio
                        
                        # Estimate error budget consumption (simplified)
                        error_budget_consumed = max(0, (1 - compliance_ratio) * 100)
                        
                        # Calculate burn rate (simplified)
                        burn_rate = error_budget_consumed / 100 if error_budget_consumed > 0 else 0
                        
                        # Update metrics
                        self.otel_integration.update_slo_compliance_metrics(
                            target=target,
                            compliance_ratio=compliance_ratio,
                            error_budget_consumed=error_budget_consumed,
                            burn_rate=burn_rate,
                            labels={"window": window.value}
                        )
                
                # Store in Redis
                await self._store_metrics_redis(monitor_name, target_name, window_results)
        
        except Exception as e:
            logger.error(f"Failed to collect SLO metrics for {monitor_name}: {e}")
    
    async def _store_metrics_redis(
        self,
        monitor_name: str,
        target_name: str,
        window_results: Dict[Any, Any]
    ) -> None:
        """Store metrics in Redis for historical analysis"""
        redis = await self.get_redis_client()
        if not redis:
            return
        
        try:
            timestamp = int(time.time())
            
            for window, result in window_results.items():
                key = f"slo_metrics:{monitor_name}:{target_name}:{window.value}"
                
                data = {
                    "timestamp": timestamp,
                    "current_value": result.current_value,
                    "target_value": result.target_value,
                    "compliance_ratio": result.compliance_ratio,
                    "measurement_count": result.measurement_count,
                    "is_compliant": result.is_compliant
                }
                
                # Store as sorted set for time-series data
                await redis.zadd(key, {json.dumps(data): timestamp})
                
                # Keep only last 1000 points
                await redis.zremrangebyrank(key, 0, -1001)
                
                # Set expiration
                await redis.expire(key, window.seconds * 2)
        
        except Exception as e:
            logger.error(f"Failed to store metrics in Redis: {e}")
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status"""
        return {
            "is_collecting": self.is_collecting,
            "collection_interval": self.collection_interval,
            "registered_monitors": list(self.slo_monitors.keys()),
            "data_sources_count": len(self.data_sources),
            "opentelemetry_enabled": self.otel_integration is not None,
            "redis_enabled": self.redis_url is not None
        }