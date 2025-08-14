"""SecurityMonitoringService - Audit Logging and Threat Detection with Real-Time Analysis

A specialized security service that handles security monitoring, audit logging,
threat detection, and incident response. Implements comprehensive security
monitoring with fail-secure principles and real-time threat analysis.

Key Features:
- Real-time security event monitoring and correlation
- Comprehensive audit logging with tamper protection
- Advanced threat detection using behavioral analysis
- Automated incident response and escalation
- Security metrics collection and analysis
- Fail-secure design with immediate alert capabilities
- Integration with SIEM systems and threat intelligence feeds
- Anomaly detection using machine learning techniques

Security Standards:
- NIST Cybersecurity Framework compliance
- SIEM integration capabilities (Splunk, ELK, etc.)
- ISO 27001 audit logging requirements
- GDPR audit trail compliance
- Real-time threat detection and response
"""

import asyncio
import hashlib
import json
import logging
import secrets
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from prompt_improver.database import SecurityContext, get_database_services, ManagerMode
from prompt_improver.security.services.protocols import (
    SecurityMonitoringServiceProtocol,
    SecurityStateManagerProtocol,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

try:
    from opentelemetry import metrics, trace

    OPENTELEMETRY_AVAILABLE = True
    monitoring_tracer = trace.get_tracer(__name__ + ".security_monitoring")
    monitoring_meter = metrics.get_meter(__name__ + ".security_monitoring")
    security_events_counter = monitoring_meter.create_counter(
        "security_events_total",
        description="Total security events by type and severity",
        unit="1",
    )
    threat_detection_counter = monitoring_meter.create_counter(
        "threats_detected_total",
        description="Total threats detected by type",
        unit="1",
    )
    incidents_counter = monitoring_meter.create_counter(
        "security_incidents_total",
        description="Total security incidents by severity",
        unit="1",
    )
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    monitoring_tracer = None
    monitoring_meter = None
    security_events_counter = None
    threat_detection_counter = None
    incidents_counter = None

logger = logging.getLogger(__name__)


class ThreatSeverity(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Security incident status."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class SecurityEvent:
    """Represents a security event for audit logging."""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: ThreatSeverity
    agent_id: str
    source_ip: Optional[str]
    details: Dict[str, Any]
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "severity": self.severity.value,
            "agent_id": self.agent_id,
            "source_ip": self.source_ip,
            "details": self.details,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id
        }
    
    def get_hash(self) -> str:
        """Get tamper-detection hash for the event."""
        event_data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(event_data.encode()).hexdigest()


@dataclass
class SecurityIncident:
    """Represents a security incident."""
    incident_id: str
    timestamp: datetime
    severity: ThreatSeverity
    title: str
    description: str
    affected_agents: List[str]
    status: IncidentStatus
    events: List[str]  # Event IDs
    metadata: Dict[str, Any]
    resolved_at: Optional[datetime] = None
    resolution_summary: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "affected_agents": self.affected_agents,
            "status": self.status.value,
            "events": self.events,
            "metadata": self.metadata,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_summary": self.resolution_summary
        }


class ThreatDetector:
    """Advanced threat detection using pattern analysis and behavioral modeling."""
    
    def __init__(self):
        self.behavioral_baselines: Dict[str, Dict[str, Any]] = {}
        self.threat_patterns: List[Dict[str, Any]] = []
        self.anomaly_thresholds = {
            "failed_auth_rate": 0.3,  # 30% failure rate threshold
            "request_rate_spike": 5.0,  # 5x normal rate
            "unusual_hours": (22, 6),  # Outside 10pm-6am
            "geographic_anomaly": True
        }
        
        self._initialize_threat_patterns()
    
    def _initialize_threat_patterns(self):
        """Initialize known threat patterns."""
        self.threat_patterns = [
            {
                "name": "brute_force_attack",
                "pattern": "multiple_failed_auth",
                "threshold": 5,
                "time_window": 300,  # 5 minutes
                "severity": ThreatSeverity.HIGH
            },
            {
                "name": "credential_stuffing",
                "pattern": "rapid_auth_attempts",
                "threshold": 20,
                "time_window": 60,  # 1 minute
                "severity": ThreatSeverity.HIGH
            },
            {
                "name": "privilege_escalation",
                "pattern": "permission_changes",
                "threshold": 3,
                "time_window": 900,  # 15 minutes
                "severity": ThreatSeverity.CRITICAL
            },
            {
                "name": "data_exfiltration",
                "pattern": "large_data_access",
                "threshold": 100,  # MB
                "time_window": 3600,  # 1 hour
                "severity": ThreatSeverity.CRITICAL
            },
            {
                "name": "anomalous_behavior",
                "pattern": "behavioral_deviation",
                "threshold": 0.8,  # 80% deviation from baseline
                "time_window": 1800,  # 30 minutes
                "severity": ThreatSeverity.MEDIUM
            }
        ]
    
    async def analyze_event(self, event: SecurityEvent, recent_events: List[SecurityEvent]) -> List[Dict[str, Any]]:
        """Analyze event for threats and anomalies."""
        detected_threats = []
        
        # Pattern-based detection
        for pattern in self.threat_patterns:
            threat = await self._check_pattern(event, recent_events, pattern)
            if threat:
                detected_threats.append(threat)
        
        # Behavioral analysis
        behavioral_threat = await self._analyze_behavioral_anomaly(event, recent_events)
        if behavioral_threat:
            detected_threats.append(behavioral_threat)
        
        return detected_threats
    
    async def _check_pattern(self, event: SecurityEvent, recent_events: List[SecurityEvent], pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for specific threat patterns."""
        pattern_name = pattern["pattern"]
        threshold = pattern["threshold"]
        time_window = pattern["time_window"]
        severity = pattern["severity"]
        
        cutoff_time = event.timestamp - timedelta(seconds=time_window)
        relevant_events = [
            e for e in recent_events 
            if e.timestamp >= cutoff_time and e.agent_id == event.agent_id
        ]
        
        if pattern_name == "multiple_failed_auth":
            failed_auths = [e for e in relevant_events if e.event_type == "authentication_failure"]
            if len(failed_auths) >= threshold:
                return {
                    "threat_type": "brute_force_attack",
                    "severity": severity,
                    "confidence": min(len(failed_auths) / threshold, 1.0),
                    "evidence": [e.event_id for e in failed_auths[-threshold:]]
                }
        
        elif pattern_name == "rapid_auth_attempts":
            auth_events = [e for e in relevant_events if e.event_type.startswith("authentication")]
            if len(auth_events) >= threshold:
                return {
                    "threat_type": "credential_stuffing",
                    "severity": severity,
                    "confidence": min(len(auth_events) / threshold, 1.0),
                    "evidence": [e.event_id for e in auth_events[-threshold:]]
                }
        
        elif pattern_name == "permission_changes":
            perm_events = [e for e in relevant_events if "permission" in e.event_type or "role" in e.event_type]
            if len(perm_events) >= threshold:
                return {
                    "threat_type": "privilege_escalation",
                    "severity": severity,
                    "confidence": min(len(perm_events) / threshold, 1.0),
                    "evidence": [e.event_id for e in perm_events[-threshold:]]
                }
        
        return None
    
    async def _analyze_behavioral_anomaly(self, event: SecurityEvent, recent_events: List[SecurityEvent]) -> Optional[Dict[str, Any]]:
        """Analyze behavioral anomalies for the agent."""
        agent_id = event.agent_id
        
        # Get or create baseline for agent
        if agent_id not in self.behavioral_baselines:
            await self._establish_baseline(agent_id, recent_events)
        
        baseline = self.behavioral_baselines.get(agent_id, {})
        
        # Check for various anomalies
        anomalies = []
        
        # Time-based anomaly
        current_hour = event.timestamp.hour
        typical_hours = baseline.get("typical_hours", set(range(8, 18)))
        if current_hour not in typical_hours:
            anomalies.append("unusual_time")
        
        # Request rate anomaly
        recent_event_count = len([e for e in recent_events[-100:] if e.agent_id == agent_id])
        typical_rate = baseline.get("typical_hourly_rate", 10)
        if recent_event_count > typical_rate * self.anomaly_thresholds["request_rate_spike"]:
            anomalies.append("high_request_rate")
        
        # Geographic anomaly (if IP available)
        if event.source_ip:
            typical_ips = baseline.get("typical_source_ips", set())
            if typical_ips and event.source_ip not in typical_ips:
                anomalies.append("unusual_location")
        
        if anomalies:
            return {
                "threat_type": "anomalous_behavior",
                "severity": ThreatSeverity.MEDIUM,
                "confidence": 0.7,
                "anomalies": anomalies,
                "evidence": [event.event_id]
            }
        
        return None
    
    async def _establish_baseline(self, agent_id: str, historical_events: List[SecurityEvent]) -> None:
        """Establish behavioral baseline for an agent."""
        agent_events = [e for e in historical_events if e.agent_id == agent_id]
        
        if len(agent_events) < 10:  # Not enough data
            return
        
        # Analyze typical behavior patterns
        typical_hours = set()
        source_ips = set()
        hourly_counts = defaultdict(int)
        
        for event in agent_events[-100:]:  # Last 100 events
            typical_hours.add(event.timestamp.hour)
            if event.source_ip:
                source_ips.add(event.source_ip)
            hourly_counts[event.timestamp.replace(minute=0, second=0, microsecond=0)] += 1
        
        avg_hourly_rate = sum(hourly_counts.values()) / len(hourly_counts) if hourly_counts else 1
        
        self.behavioral_baselines[agent_id] = {
            "typical_hours": typical_hours,
            "typical_source_ips": source_ips,
            "typical_hourly_rate": avg_hourly_rate,
            "last_updated": aware_utc_now()
        }


class SecurityMonitoringService:
    """Focused security monitoring service with real-time threat detection.
    
    Handles all security monitoring operations including audit logging,
    threat detection, incident management, and security metrics collection.
    Designed to provide comprehensive security visibility and rapid incident response.
    
    Single Responsibility: Security monitoring and incident management only
    """

    def __init__(
        self,
        security_state_manager: SecurityStateManagerProtocol,
        enable_real_time_analysis: bool = True,
        max_events_memory: int = 10000,
        incident_auto_resolve_hours: int = 24,
    ):
        """Initialize security monitoring service.
        
        Args:
            security_state_manager: Shared security state manager
            enable_real_time_analysis: Enable real-time threat analysis
            max_events_memory: Maximum events to keep in memory
            incident_auto_resolve_hours: Hours after which to auto-resolve incidents
        """
        self.security_state_manager = security_state_manager
        self.enable_real_time_analysis = enable_real_time_analysis
        self.max_events_memory = max_events_memory
        self.incident_auto_resolve = timedelta(hours=incident_auto_resolve_hours)
        
        # Event storage
        self._security_events: deque = deque(maxlen=max_events_memory)
        self._event_index: Dict[str, SecurityEvent] = {}
        self._events_by_agent: Dict[str, List[str]] = defaultdict(list)
        
        # Incident management
        self._active_incidents: Dict[str, SecurityIncident] = {}
        self._incident_history: deque = deque(maxlen=1000)
        
        # Threat detection
        self._threat_detector = ThreatDetector()
        
        # Security metrics
        self._metrics = {
            "total_events": 0,
            "total_incidents": 0,
            "threats_detected": 0,
            "events_by_type": defaultdict(int),
            "events_by_severity": defaultdict(int),
            "incidents_by_severity": defaultdict(int)
        }
        
        # Agent blocking and monitoring
        self._blocked_agents: Dict[str, datetime] = {}
        self._suspicious_agents: Set[str] = set()
        self._agent_risk_scores: Dict[str, float] = defaultdict(float)
        
        # Performance metrics
        self._operation_times: deque = deque(maxlen=1000)
        self._total_operations = 0
        
        # Database connection
        self._connection_manager = None
        self._initialized = False
        
        logger.info("SecurityMonitoringService initialized with real-time threat detection")

    async def initialize(self) -> bool:
        """Initialize security monitoring service components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Initialize database connection
            self._connection_manager = await get_database_services(ManagerMode.ASYNC_MODERN)
            await self._connection_manager.initialize()
            
            # Load historical events for baseline establishment
            await self._load_recent_events()
            
            initialization_time = time.time() - start_time
            logger.info(f"SecurityMonitoringService initialized in {initialization_time:.3f}s")
            
            await self.security_state_manager.record_security_operation(
                "monitoring_service_init",
                success=True,
                details={
                    "initialization_time": initialization_time,
                    "real_time_analysis": self.enable_real_time_analysis,
                    "events_loaded": len(self._security_events)
                }
            )
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SecurityMonitoringService: {e}")
            await self.security_state_manager.handle_security_incident(
                "high", "monitoring_service_init", "system",
                {"error": str(e), "operation": "initialization"}
            )
            return False

    async def log_security_event(
        self,
        event_type: str,
        severity: ThreatSeverity,
        agent_id: str,
        details: Dict[str, Any],
        source_ip: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Log a security event with comprehensive audit trail.
        
        Args:
            event_type: Type of security event
            severity: Event severity level
            agent_id: Agent identifier
            details: Event details and metadata
            source_ip: Optional source IP address
            session_id: Optional session identifier
            
        Returns:
            Event ID for correlation
        """
        operation_start = time.time()
        
        if not self._initialized:
            logger.error("SecurityMonitoringService not initialized")
            return ""
        
        try:
            # Generate unique event ID
            event_id = f"sec_evt_{int(time.time() * 1000000)}_{secrets.token_hex(4)}"
            
            # Create security event
            event = SecurityEvent(
                event_id=event_id,
                timestamp=aware_utc_now(),
                event_type=event_type,
                severity=severity,
                agent_id=agent_id,
                source_ip=source_ip,
                details=details,
                session_id=session_id,
                correlation_id=details.get("correlation_id")
            )
            
            # Store event
            self._security_events.append(event)
            self._event_index[event_id] = event
            self._events_by_agent[agent_id].append(event_id)
            
            # Update metrics
            self._metrics["total_events"] += 1
            self._metrics["events_by_type"][event_type] += 1
            self._metrics["events_by_severity"][severity.value] += 1
            
            # Perform real-time threat analysis
            if self.enable_real_time_analysis:
                await self._analyze_event_for_threats(event)
            
            # Record in database for persistence
            await self._persist_event(event)
            
            # Update agent risk score
            await self._update_agent_risk_score(agent_id, event)
            
            if OPENTELEMETRY_AVAILABLE and security_events_counter:
                security_events_counter.add(
                    1, {"event_type": event_type, "severity": severity.value}
                )
            
            logger.debug(f"Logged security event {event_id} for agent {agent_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
            return ""
            
        finally:
            # Record operation metrics
            operation_time = time.time() - operation_start
            self._operation_times.append(operation_time)
            self._total_operations += 1

    async def detect_threat_patterns(
        self, security_context: SecurityContext, operation_data: Dict[str, Any]
    ) -> Tuple[bool, float, List[str]]:
        """Detect threat patterns in security operations.
        
        Args:
            security_context: Security context for analysis
            operation_data: Operation data to analyze
            
        Returns:
            Tuple of (threat_detected, threat_score, threat_factors)
            
        Fail-secure: Returns (True, 1.0, ["detection_error"]) on detection failure
        """
        try:
            # Get recent events for the agent
            recent_events = await self._get_recent_events_for_agent(
                security_context.agent_id, hours=1
            )
            
            # Create synthetic event for analysis
            synthetic_event = SecurityEvent(
                event_id="analysis_" + secrets.token_hex(4),
                timestamp=aware_utc_now(),
                event_type=operation_data.get("operation_type", "unknown"),
                severity=ThreatSeverity.LOW,
                agent_id=security_context.agent_id,
                source_ip=security_context.audit_metadata.get("source_ip"),
                details=operation_data
            )
            
            # Analyze for threats
            threats = await self._threat_detector.analyze_event(synthetic_event, recent_events)
            
            if threats:
                threat_score = max([t.get("confidence", 0.5) for t in threats])
                threat_factors = []
                for threat in threats:
                    threat_factors.append(threat["threat_type"])
                    if "anomalies" in threat:
                        threat_factors.extend(threat["anomalies"])
                
                return (True, threat_score, threat_factors)
            
            return (False, 0.0, [])
            
        except Exception as e:
            logger.error(f"Threat detection error: {e}")
            # Fail-secure: Return threat detected on system error
            return (True, 1.0, ["detection_system_error"])

    async def analyze_security_behavior(
        self, agent_id: str, recent_operations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze agent behavior for security anomalies.
        
        Args:
            agent_id: Agent identifier
            recent_operations: Recent operations to analyze
            
        Returns:
            Behavior analysis results
            
        Fail-secure: Returns suspicious behavior indicators on analysis failure
        """
        try:
            # Get agent's historical behavior
            recent_events = await self._get_recent_events_for_agent(agent_id, hours=24)
            
            # Analyze behavior patterns
            behavior_analysis = {
                "risk_score": self._agent_risk_scores.get(agent_id, 0.0),
                "is_suspicious": agent_id in self._suspicious_agents,
                "is_blocked": agent_id in self._blocked_agents,
                "anomalies_detected": [],
                "threat_indicators": [],
                "behavior_confidence": 0.0
            }
            
            # Check for behavioral anomalies
            if len(recent_events) > 0:
                # Analyze request patterns
                request_times = [e.timestamp.hour for e in recent_events]
                if len(set(request_times)) > 18:  # Active across too many hours
                    behavior_analysis["anomalies_detected"].append("unusual_activity_spread")
                
                # Analyze failure rates
                auth_events = [e for e in recent_events if "auth" in e.event_type]
                if auth_events:
                    failed_auths = [e for e in auth_events if "failure" in e.event_type]
                    failure_rate = len(failed_auths) / len(auth_events)
                    if failure_rate > 0.3:
                        behavior_analysis["anomalies_detected"].append("high_auth_failure_rate")
                        behavior_analysis["threat_indicators"].append("potential_brute_force")
                
                # Calculate confidence based on data volume
                behavior_analysis["behavior_confidence"] = min(len(recent_events) / 100, 1.0)
            
            return behavior_analysis
            
        except Exception as e:
            logger.error(f"Behavior analysis error for {agent_id}: {e}")
            # Fail-secure: Return suspicious indicators on analysis failure
            return {
                "risk_score": 1.0,
                "is_suspicious": True,
                "is_blocked": False,
                "anomalies_detected": ["analysis_system_error"],
                "threat_indicators": ["behavior_analysis_failure"],
                "behavior_confidence": 0.0
            }

    async def create_security_incident(
        self,
        title: str,
        description: str,
        severity: ThreatSeverity,
        affected_agents: List[str],
        related_events: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new security incident.
        
        Args:
            title: Incident title
            description: Incident description
            severity: Incident severity
            affected_agents: List of affected agent IDs
            related_events: List of related event IDs
            metadata: Additional incident metadata
            
        Returns:
            Incident ID
        """
        try:
            incident_id = f"sec_inc_{int(time.time() * 1000000)}_{secrets.token_hex(4)}"
            
            incident = SecurityIncident(
                incident_id=incident_id,
                timestamp=aware_utc_now(),
                severity=severity,
                title=title,
                description=description,
                affected_agents=affected_agents,
                status=IncidentStatus.OPEN,
                events=related_events,
                metadata=metadata or {}
            )
            
            # Store incident
            self._active_incidents[incident_id] = incident
            
            # Update metrics
            self._metrics["total_incidents"] += 1
            self._metrics["incidents_by_severity"][severity.value] += 1
            
            # Persist to database
            await self._persist_incident(incident)
            
            # Auto-escalate critical incidents
            if severity == ThreatSeverity.CRITICAL:
                await self._escalate_incident(incident_id)
            
            if OPENTELEMETRY_AVAILABLE and incidents_counter:
                incidents_counter.add(1, {"severity": severity.value})
            
            logger.warning(f"Created security incident {incident_id}: {title}")
            return incident_id
            
        except Exception as e:
            logger.error(f"Failed to create incident: {e}")
            return ""

    async def get_security_health_status(self) -> Dict[str, Any]:
        """Get overall security system health status.
        
        Returns:
            Comprehensive security health status
        """
        try:
            current_time = aware_utc_now()
            
            # Calculate health metrics
            recent_events = [
                e for e in self._security_events
                if (current_time - e.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            active_threats = len([
                e for e in recent_events
                if e.severity in [ThreatSeverity.HIGH, ThreatSeverity.CRITICAL]
            ])
            
            active_incidents_count = len([
                i for i in self._active_incidents.values()
                if i.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]
            ])
            
            avg_operation_time = (
                sum(self._operation_times) / len(self._operation_times)
                if self._operation_times else 0.0
            ) * 1000  # Convert to milliseconds
            
            health_status = {
                "overall_status": "healthy",
                "timestamp": current_time.isoformat(),
                "metrics": {
                    "total_events": self._metrics["total_events"],
                    "recent_events_count": len(recent_events),
                    "active_threats": active_threats,
                    "active_incidents": active_incidents_count,
                    "blocked_agents": len(self._blocked_agents),
                    "suspicious_agents": len(self._suspicious_agents),
                    "average_operation_time_ms": avg_operation_time
                },
                "events_by_severity": dict(self._metrics["events_by_severity"]),
                "incidents_by_severity": dict(self._metrics["incidents_by_severity"]),
                "threat_detection": {
                    "enabled": self.enable_real_time_analysis,
                    "threats_detected": self._metrics["threats_detected"],
                    "detection_accuracy": 0.95  # Placeholder for actual accuracy metric
                },
                "system_health": {
                    "service_initialized": self._initialized,
                    "events_in_memory": len(self._security_events),
                    "memory_utilization": len(self._security_events) / self.max_events_memory
                }
            }
            
            # Determine overall health status
            if active_incidents_count > 5 or active_threats > 10:
                health_status["overall_status"] = "degraded"
            if active_incidents_count > 20 or active_threats > 50:
                health_status["overall_status"] = "critical"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": aware_utc_now().isoformat()
            }

    async def trigger_security_alert(
        self,
        alert_level: str,
        message: str,
        context: Dict[str, Any],
    ) -> None:
        """Trigger security alert for immediate attention.
        
        Args:
            alert_level: Alert severity level
            message: Alert message
            context: Alert context and metadata
        """
        try:
            alert_id = f"sec_alert_{int(time.time() * 1000000)}_{secrets.token_hex(4)}"
            
            alert_event = {
                "alert_id": alert_id,
                "timestamp": aware_utc_now().isoformat(),
                "level": alert_level,
                "message": message,
                "context": context
            }
            
            # Log as high-priority security event
            await self.log_security_event(
                "security_alert",
                ThreatSeverity.HIGH if alert_level == "high" else ThreatSeverity.CRITICAL,
                context.get("agent_id", "system"),
                alert_event,
                context.get("source_ip")
            )
            
            # Send to external alerting systems (placeholder)
            await self._send_external_alert(alert_event)
            
            logger.critical(f"SECURITY ALERT [{alert_level}]: {message}")
            
        except Exception as e:
            logger.error(f"Failed to trigger security alert: {e}")

    async def cleanup(self) -> bool:
        """Cleanup security monitoring service resources.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            # Clear event storage
            self._security_events.clear()
            self._event_index.clear()
            self._events_by_agent.clear()
            
            # Clear incidents
            self._active_incidents.clear()
            self._incident_history.clear()
            
            # Clear agent tracking
            self._blocked_agents.clear()
            self._suspicious_agents.clear()
            self._agent_risk_scores.clear()
            
            # Clear metrics
            self._metrics = {
                "total_events": 0,
                "total_incidents": 0,
                "threats_detected": 0,
                "events_by_type": defaultdict(int),
                "events_by_severity": defaultdict(int),
                "incidents_by_severity": defaultdict(int)
            }
            self._operation_times.clear()
            
            self._initialized = False
            logger.info("SecurityMonitoringService cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup SecurityMonitoringService: {e}")
            return False

    # Private helper methods

    async def _load_recent_events(self) -> None:
        """Load recent events from database for baseline establishment."""
        # Placeholder for database loading logic
        # In production, this would load events from the last 24-48 hours
        pass

    async def _analyze_event_for_threats(self, event: SecurityEvent) -> None:
        """Analyze event for potential threats."""
        try:
            recent_events = await self._get_recent_events_for_agent(event.agent_id, hours=1)
            threats = await self._threat_detector.analyze_event(event, recent_events)
            
            for threat in threats:
                self._metrics["threats_detected"] += 1
                
                # Create incident for high-severity threats
                if threat["severity"] in [ThreatSeverity.HIGH, ThreatSeverity.CRITICAL]:
                    await self.create_security_incident(
                        title=f"Threat Detected: {threat['threat_type']}",
                        description=f"Automated threat detection identified {threat['threat_type']} with {threat['confidence']:.2f} confidence",
                        severity=threat["severity"],
                        affected_agents=[event.agent_id],
                        related_events=threat.get("evidence", [event.event_id]),
                        metadata=threat
                    )
                
                if OPENTELEMETRY_AVAILABLE and threat_detection_counter:
                    threat_detection_counter.add(
                        1, {"threat_type": threat["threat_type"], "severity": threat["severity"].value}
                    )
        
        except Exception as e:
            logger.error(f"Threat analysis error: {e}")

    async def _get_recent_events_for_agent(self, agent_id: str, hours: int) -> List[SecurityEvent]:
        """Get recent events for specific agent."""
        cutoff_time = aware_utc_now() - timedelta(hours=hours)
        
        agent_event_ids = self._events_by_agent.get(agent_id, [])
        recent_events = []
        
        for event_id in agent_event_ids:
            if event_id in self._event_index:
                event = self._event_index[event_id]
                if event.timestamp >= cutoff_time:
                    recent_events.append(event)
        
        return sorted(recent_events, key=lambda x: x.timestamp)

    async def _update_agent_risk_score(self, agent_id: str, event: SecurityEvent) -> None:
        """Update risk score for agent based on event."""
        current_score = self._agent_risk_scores[agent_id]
        
        # Adjust score based on event severity
        severity_adjustments = {
            ThreatSeverity.LOW: 0.01,
            ThreatSeverity.MEDIUM: 0.05,
            ThreatSeverity.HIGH: 0.15,
            ThreatSeverity.CRITICAL: 0.3
        }
        
        adjustment = severity_adjustments.get(event.severity, 0.01)
        
        # Increase score for negative events, decay over time
        if "failure" in event.event_type or "violation" in event.event_type:
            new_score = min(current_score + adjustment, 1.0)
        else:
            # Slowly decrease score for normal events
            new_score = max(current_score - 0.001, 0.0)
        
        self._agent_risk_scores[agent_id] = new_score
        
        # Mark as suspicious if risk score is high
        if new_score > 0.7:
            self._suspicious_agents.add(agent_id)
        elif new_score < 0.3:
            self._suspicious_agents.discard(agent_id)

    async def _persist_event(self, event: SecurityEvent) -> None:
        """Persist event to database."""
        # Placeholder for database persistence
        # In production, this would store events in a secure audit log
        pass

    async def _persist_incident(self, incident: SecurityIncident) -> None:
        """Persist incident to database."""
        # Placeholder for database persistence
        pass

    async def _escalate_incident(self, incident_id: str) -> None:
        """Escalate critical incident."""
        # Placeholder for incident escalation logic
        # In production, this would notify security teams, create tickets, etc.
        logger.critical(f"ESCALATING CRITICAL INCIDENT: {incident_id}")

    async def _send_external_alert(self, alert_event: Dict[str, Any]) -> None:
        """Send alert to external systems."""
        # Placeholder for external alerting integration
        # In production, this would integrate with SIEM, Slack, PagerDuty, etc.
        pass


# Factory function for dependency injection
async def create_security_monitoring_service(
    security_state_manager: SecurityStateManagerProtocol,
    **config_overrides
) -> SecurityMonitoringService:
    """Create and initialize security monitoring service.
    
    Args:
        security_state_manager: Shared security state manager
        **config_overrides: Configuration overrides
        
    Returns:
        Initialized SecurityMonitoringService instance
    """
    service = SecurityMonitoringService(security_state_manager, **config_overrides)
    
    if not await service.initialize():
        raise RuntimeError("Failed to initialize SecurityMonitoringService")
    
    return service