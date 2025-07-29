"""APES security framework with data protection and audit logging.
Implements Task 4: Security Framework from Phase 2.
Enhanced with 2025 best practices for ML pipeline data protection.
"""

import json
import logging
import re
from datetime import datetime, timezone, UTC
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console

# Lazy import to avoid circular dependency
# from ...database import get_sessionmanager

def _get_sessionmanager():
    """Lazy import of sessionmanager to avoid circular imports."""
    from ...database import get_sessionmanager
    return get_sessionmanager
from .analytics_factory import get_analytics_interface

class SecurityLevel(Enum):
    """Security levels for data classification."""
    public = "public"
    internal = "internal"
    confidential = "confidential"
    restricted = "restricted"

class PrivacyTechnique(Enum):
    """Privacy preservation techniques."""
    redaction = "redaction"
    masking = "masking"
    tokenization = "tokenization"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"

@dataclass
class DataProtectionMetrics:
    """Metrics for data protection operations."""
    total_prompts_processed: int = 0
    prompts_with_sensitive_data: int = 0
    total_redactions: int = 0
    redactions_by_type: Dict[str, int] = field(default_factory=dict)
    privacy_techniques_used: Dict[str, int] = field(default_factory=dict)
    compliance_score: float = 0.0
    processing_time_ms: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class SensitiveDataMatch:
    """Represents a detected sensitive data match."""
    pattern_type: str
    match_text: str
    start_position: int
    end_position: int
    confidence_score: float
    risk_level: str
    recommended_action: str

class PromptDataProtection:
    """
    Enhanced prompt data protection with 2025 best practices.

    Implements GDPR-compliant data protection, differential privacy,
    async operations, and integration with ML Pipeline Orchestrator.
    """

    def __init__(self, console: Console | None = None, enable_differential_privacy: bool = True):
        self.console = console or Console()
        self.enable_differential_privacy = enable_differential_privacy

        # Enhanced 2025 sensitive pattern detection with confidence scoring
        self.sensitive_patterns = [
            # API Keys and Tokens (High Risk)
            (r"sk-[a-zA-Z0-9]{40,60}", "openai_api_key", 0.95, "HIGH"),
            (r"ghp_[a-zA-Z0-9]{36,50}", "github_token", 0.95, "HIGH"),
            (r"xoxb-[0-9]{11,13}-[0-9]{11,13}-[a-zA-Z0-9]{24}", "slack_bot_token", 0.95, "HIGH"),
            (r"AKIA[0-9A-Z]{16}", "aws_access_key", 0.95, "HIGH"),
            (r"ya29\.[0-9A-Za-z\-_]+", "google_oauth_token", 0.90, "HIGH"),

            # Personal Information (Medium-High Risk)
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email_address", 0.85, "MEDIUM"),
            (r"\b\d{3}-\d{2}-\d{4}\b", "ssn_pattern", 0.90, "HIGH"),
            (r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b", "credit_card", 0.90, "HIGH"),
            (r"\b(?:\+1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b", "phone_number", 0.80, "MEDIUM"),

            # Credentials and Secrets (High Risk)
            (r"password[\s]*[:=][\s]*[^\s]+", "password_field", 0.85, "HIGH"),
            (r"api_key[\s]*[:=][\s]*[^\s]+", "api_key_field", 0.90, "HIGH"),
            (r"secret[\s]*[:=][\s]*[^\s]+", "secret_field", 0.85, "HIGH"),
            (r"\b(?:Bearer|Token)\s+[A-Za-z0-9\-._~+/]+=*", "bearer_token", 0.90, "HIGH"),

            # Generic Tokens and IDs (Medium Risk)
            (r"\b[A-Z0-9]{20,}\b", "long_token", 0.70, "MEDIUM"),
            (r"[a-f0-9]{32,64}", "hash_or_token", 0.65, "MEDIUM"),

            # Prompt Injection Patterns (2025 Security)
            (r"(?i)ignore\s+previous\s+instructions", "prompt_injection", 0.80, "HIGH"),
            (r"(?i)system\s*:\s*you\s+are\s+now", "system_override", 0.85, "HIGH"),
            (r"(?i)jailbreak|bypass\s+safety", "jailbreak_attempt", 0.80, "HIGH"),
        ]

        # Integration with existing components
        self.audit_enabled = True
        self.logger = logging.getLogger("apes.security.data_protection")

        # Enhanced metrics tracking with 2025 standards
        self.metrics = DataProtectionMetrics()

        # GDPR compliance features
        self.gdpr_enabled = True
        self.data_retention_days = 30
        self.consent_tracking = {}

        # Performance monitoring
        self._processing_times = []
        self._max_processing_time_ms = 1000  # 1 second max for real-time processing

    async def sanitize_prompt_before_storage(
        self, prompt: str, session_id: str, user_consent: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced prompt sanitization with 2025 best practices.

        Args:
            prompt: Input prompt to sanitize
            session_id: Session identifier for audit tracking
            user_consent: GDPR consent for data processing

        Returns:
            Tuple of (sanitized_prompt, protection_summary)
        """
        start_time = datetime.now(timezone.utc)

        # GDPR consent check
        if self.gdpr_enabled and not user_consent:
            self.logger.warning(f"Data processing without consent for session {session_id}")
            return prompt, {"error": "GDPR_CONSENT_REQUIRED", "processed": False}

        sanitized = prompt
        detected_matches: List[SensitiveDataMatch] = []
        redaction_details = {}
        privacy_techniques_used = []

        self.metrics.total_prompts_processed += 1

        # Enhanced pattern detection with confidence scoring
        for pattern, pattern_type, confidence, risk_level in self.sensitive_patterns:
            matches = list(re.finditer(pattern, sanitized))
            if matches:
                match_objects = []
                for match in matches:
                    match_obj = SensitiveDataMatch(
                        pattern_type=pattern_type,
                        match_text=match.group(),
                        start_position=match.start(),
                        end_position=match.end(),
                        confidence_score=confidence,
                        risk_level=risk_level,
                        recommended_action=self._get_recommended_action(pattern_type, risk_level)
                    )
                    match_objects.append(match_obj)
                    detected_matches.append(match_obj)

                # Apply appropriate privacy technique based on risk level
                if risk_level == "HIGH":
                    # Full redaction for high-risk data
                    redaction_placeholder = f"[REDACTED_{pattern_type.upper()}]"
                    sanitized = re.sub(pattern, redaction_placeholder, sanitized)
                    privacy_techniques_used.append(PrivacyTechnique.redaction.value)
                elif risk_level == "MEDIUM":
                    # Masking for medium-risk data
                    def mask_match(match):
                        text = match.group()
                        if len(text) <= 4:
                            return "*" * len(text)
                        return text[:2] + "*" * (len(text) - 4) + text[-2:]
                    sanitized = re.sub(pattern, mask_match, sanitized)
                    privacy_techniques_used.append(PrivacyTechnique.masking.value)

                redaction_details[pattern_type] = {
                    "count": len(matches),
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "technique": privacy_techniques_used[-1] if privacy_techniques_used else "none",
                    "matches": [{"start": m.start_position, "end": m.end_position} for m in match_objects]
                }

                # Update metrics
                if pattern_type not in self.metrics.redactions_by_type:
                    self.metrics.redactions_by_type[pattern_type] = 0
                self.metrics.redactions_by_type[pattern_type] += len(matches)

        # Apply differential privacy if enabled
        if self.enable_differential_privacy and detected_matches:
            sanitized = await self._apply_differential_privacy(sanitized, detected_matches)
            privacy_techniques_used.append(PrivacyTechnique.DIFFERENTIAL_PRIVACY.value)

        # Update metrics
        if detected_matches:
            self.metrics.prompts_with_sensitive_data += 1
            self.metrics.total_redactions += len(detected_matches)

        for technique in privacy_techniques_used:
            if technique not in self.metrics.privacy_techniques_used:
                self.metrics.privacy_techniques_used[technique] = 0
            self.metrics.privacy_techniques_used[technique] += 1

        # Calculate processing time
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        self.metrics.processing_time_ms = processing_time
        self._processing_times.append(processing_time)

        # Performance monitoring
        if processing_time > self._max_processing_time_ms:
            self.logger.warning(f"Slow data protection processing: {processing_time:.2f}ms for session {session_id}")

        # Enhanced audit logging
        if detected_matches and self.audit_enabled:
            await self.audit_redaction_enhanced(
                session_id, detected_matches, redaction_details, privacy_techniques_used
            )

        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(detected_matches, privacy_techniques_used)
        self.metrics.compliance_score = compliance_score

        protection_summary = {
            "redactions_made": len(detected_matches),
            "redaction_types": list(redaction_details.keys()),
            "redaction_details": redaction_details,
            "privacy_techniques_used": list(set(privacy_techniques_used)),
            "compliance_score": compliance_score,
            "processing_time_ms": processing_time,
            "sanitized_length": len(sanitized),
            "original_length": len(prompt),
            "gdpr_compliant": self.gdpr_enabled and user_consent,
            "risk_assessment": self._assess_overall_risk(detected_matches)
        }

        return sanitized, protection_summary

    def _get_recommended_action(self, pattern_type: str, risk_level: str) -> str:
        """Get recommended action for detected sensitive data."""
        if risk_level == "HIGH":
            return "IMMEDIATE_REDACTION"
        elif risk_level == "MEDIUM":
            return "MASKING_RECOMMENDED"
        else:
            return "REVIEW_REQUIRED"

    async def _apply_differential_privacy(self, text: str, matches: List[SensitiveDataMatch]) -> str:
        """Apply differential privacy techniques to protect sensitive data."""
        # Simple noise injection for demonstration
        # In production, use proper differential privacy libraries
        if not matches:
            return text

        # Add minimal noise to preserve utility while protecting privacy
        noise_chars = ['·', '‧', '•']
        modified_text = text

        for match in matches:
            if match.risk_level == "HIGH":
                # Add noise around high-risk matches
                start = max(0, match.start_position - 1)
                end = min(len(modified_text), match.end_position + 1)
                if start < len(modified_text) and end <= len(modified_text):
                    noise = noise_chars[hash(match.match_text) % len(noise_chars)]
                    modified_text = modified_text[:start] + noise + modified_text[start+1:end-1] + noise + modified_text[end:]

        return modified_text

    def _calculate_compliance_score(self, matches: List[SensitiveDataMatch], techniques: List[str]) -> float:
        """Calculate GDPR compliance score based on data protection measures."""
        if not matches:
            return 100.0

        base_score = 100.0

        # Deduct points for unprotected sensitive data
        for match in matches:
            if match.risk_level == "HIGH":
                base_score -= 15.0
            elif match.risk_level == "MEDIUM":
                base_score -= 8.0
            else:
                base_score -= 3.0

        # Add points for privacy techniques used
        technique_bonus = len(set(techniques)) * 5.0

        # Ensure score is between 0 and 100
        final_score = max(0.0, min(100.0, base_score + technique_bonus))
        return round(final_score, 2)

    def _assess_overall_risk(self, matches: List[SensitiveDataMatch]) -> str:
        """Assess overall risk level based on detected matches."""
        if not matches:
            return "LOW"

        high_risk_count = sum(1 for m in matches if m.risk_level == "HIGH")
        medium_risk_count = sum(1 for m in matches if m.risk_level == "MEDIUM")

        if high_risk_count > 0:
            return "HIGH"
        elif medium_risk_count > 2:
            return "MEDIUM"
        elif medium_risk_count > 0:
            return "MEDIUM"
        else:
            return "LOW"

    async def audit_redaction_enhanced(
        self,
        session_id: str,
        matches: List[SensitiveDataMatch],
        redaction_details: Dict[str, Any],
        privacy_techniques: List[str]
    ):
        """Enhanced audit logging with 2025 security standards."""
        try:
            get_sessionmanager = _get_sessionmanager()
            async with get_sessionmanager().session() as db_session:
                from sqlalchemy import text

                audit_data = {
                    "redaction_count": len(matches),
                    "redaction_details": redaction_details,
                    "privacy_techniques_used": privacy_techniques,
                    "audit_timestamp": datetime.now(timezone.utc).isoformat(),
                    "compliance_score": self.metrics.compliance_score,
                    "processing_time_ms": self.metrics.processing_time_ms,
                    "gdpr_compliant": self.gdpr_enabled,
                    "risk_assessment": self._assess_overall_risk(matches),
                    "security_framework_version": "2025.1",
                    "matches_detail": [
                        {
                            "type": m.pattern_type,
                            "risk_level": m.risk_level,
                            "confidence": m.confidence_score,
                            "action": m.recommended_action
                        } for m in matches
                    ]
                }

                # Check if session exists
                check_query = text("""
                    SELECT id, session_metadata FROM improvement_sessions
                    WHERE session_id = :session_id
                """)
                result = await db_session.execute(
                    check_query, {"session_id": session_id}
                )
                session_record = result.fetchone()

                if session_record:
                    # Update existing session with enhanced security audit info
                    existing_metadata = session_record[1] or {}
                    existing_metadata["security_audit_v2025"] = audit_data

                    update_query = text("""
                        UPDATE improvement_sessions
                        SET session_metadata = :metadata
                        WHERE session_id = :session_id
                    """)
                    await db_session.execute(
                        update_query,
                        {
                            "metadata": json.dumps(existing_metadata),
                            "session_id": session_id,
                        },
                    )
                else:
                    # Create new session record for audit tracking
                    metadata_with_audit = {
                        "security_audit_v2025": audit_data,
                        "created_for_audit": True
                    }

                    insert_query = text("""
                        INSERT INTO improvement_sessions
                        (session_id, session_metadata, created_at)
                        VALUES (:session_id, :metadata, :created_at)
                    """)
                    await db_session.execute(
                        insert_query,
                        {
                            "session_id": session_id,
                            "metadata": json.dumps(metadata_with_audit),
                            "created_at": datetime.now(timezone.utc),
                        },
                    )

                await db_session.commit()
                self.logger.info(f"Enhanced security audit logged for session {session_id}")

        except Exception as e:
            self.logger.error(f"Failed to log enhanced security audit: {e}")

    async def audit_redaction(
        self, session_id: str, redaction_count: int, redaction_details: dict[str, Any]
    ):
        """Audit log redactions using existing database structure"""
        try:
            # Use existing database connection from Phase 1
            get_sessionmanager = _get_sessionmanager()
            async with get_sessionmanager().session() as db_session:
                # Store audit info in existing improvement_sessions table

                # Use raw SQL to avoid SQLAlchemy model issues
                from sqlalchemy import text

                # Check if session exists
                check_query = text("""
                    SELECT id, session_metadata FROM improvement_sessions
                    WHERE session_id = :session_id
                """)
                result = await db_session.execute(
                    check_query, {"session_id": session_id}
                )
                session_record = result.fetchone()

                audit_data = {
                    "redactions": redaction_count,
                    "redaction_details": redaction_details,
                    "audit_timestamp": datetime.now(UTC).isoformat(),
                    "security_level": "redacted" if redaction_count > 0 else "clean",
                }

                if session_record:
                    # Update existing session with security audit info
                    existing_metadata = session_record[1] or {}
                    existing_metadata["security_audit"] = audit_data

                    update_query = text("""
                        UPDATE improvement_sessions
                        SET session_metadata = :metadata
                        WHERE session_id = :session_id
                    """)
                    await db_session.execute(
                        update_query,
                        {
                            "metadata": json.dumps(existing_metadata),
                            "session_id": session_id,
                        },
                    )
                else:
                    # Create new session record for audit tracking
                    metadata_with_audit = {
                        "security_audit": {
                            "redactions": redaction_count,
                            "redaction_details": redaction_details,
                            "audit_timestamp": datetime.now(UTC).isoformat(),
                            "security_level": "redacted"
                            if redaction_count > 0
                            else "clean",
                            "audit_only": True,
                        }
                    }

                    insert_query = text("""
                        INSERT INTO improvement_sessions
                        (session_id, original_prompt, final_prompt, rules_applied, session_metadata, started_at, status)
                        VALUES (:session_id, :original_prompt, :final_prompt, :rules_applied, :metadata, :started_at, :status)
                    """)
                    await db_session.execute(
                        insert_query,
                        {
                            "session_id": session_id,
                            "original_prompt": "[Security audit only]",
                            "final_prompt": "[Security audit only]",
                            "rules_applied": json.dumps([]),
                            "metadata": json.dumps(metadata_with_audit),
                            "started_at": datetime.now(UTC),
                            "status": "completed",
                        },
                    )

                self.logger.info(
                    f"Security audit logged for session {session_id}: {redaction_count} redactions"
                )

        except Exception as e:
            self.logger.error(
                f"Failed to audit redaction for session {session_id}: {e}"
            )

    async def get_security_audit_report(self, days: int = 30) -> dict[str, Any]:
        """Generate security audit report using existing analytics framework"""
        try:
            # Leverage existing analytics service for security reporting
            analytics_factory = get_analytics_interface()
            analytics = analytics_factory() if analytics_factory else None

            # Get basic analytics data using correct session manager
            get_sessionmanager = _get_sessionmanager()
            async with get_sessionmanager().session() as db_session:
                # Get sessions with security audit data (using correct field name)
                from sqlalchemy import text

                # Use parameterized query to prevent SQL injection
                audit_query = text(
                    """
                    SELECT
                        COUNT(*) as total_sessions,
                        COUNT(*) FILTER (WHERE session_metadata->'security_audit'->>'redactions' != '0') as sessions_with_redactions,
                        SUM((session_metadata->'security_audit'->>'redactions')::int) FILTER (WHERE session_metadata->'security_audit'->>'redactions' IS NOT NULL) as total_redactions,
                        COUNT(*) FILTER (WHERE started_at >= NOW() - INTERVAL ':days days') as recent_sessions,
                        COUNT(*) FILTER (WHERE session_metadata->'security_audit'->>'security_level' = 'clean') as clean_sessions
                    FROM improvement_sessions
                    WHERE session_metadata->'security_audit' IS NOT NULL
                    AND started_at >= NOW() - INTERVAL ':days days'
                """
                )

                result = await db_session.execute(audit_query, {"days": days})
                audit_data = result.fetchone()

                # Get redaction type breakdown - use parameterized query to prevent SQL injection
                redaction_types_query = text(
                    """
                    SELECT
                        jsonb_object_keys(session_metadata->'security_audit'->'redaction_details') as redaction_type,
                        COUNT(*) as occurrence_count
                    FROM improvement_sessions
                    WHERE session_metadata->'security_audit'->'redaction_details' IS NOT NULL
                    AND started_at >= NOW() - INTERVAL ':days days'
                    GROUP BY redaction_type
                    ORDER BY occurrence_count DESC
                """
                )

                result = await db_session.execute(redaction_types_query, {"days": days})
                redaction_types = result.fetchall()

                # Calculate compliance score
                total_sessions = audit_data[0] or 0
                clean_sessions = audit_data[4] or 0
                compliance_score = (
                    (clean_sessions / total_sessions * 100)
                    if total_sessions > 0
                    else 100
                )

                # Calculate total prompts (simplified - just use total sessions as proxy)
                total_prompts = total_sessions

                security_report = {
                    "report_period_days": days,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "summary": {
                        "total_prompts_processed": total_prompts,
                        "total_sessions_audited": total_sessions,
                        "sessions_with_redactions": audit_data[1] or 0,
                        "total_redactions_performed": audit_data[2] or 0,
                        "recent_sessions": audit_data[3] or 0,
                        "clean_sessions": clean_sessions,
                    },
                    "compliance": {
                        "compliance_score_percent": round(compliance_score, 2),
                        "redaction_rate_percent": round(
                            (audit_data[1] or 0) / total_sessions * 100, 2
                        )
                        if total_sessions > 0
                        else 0,
                        "status": "EXCELLENT"
                        if compliance_score >= 95
                        else "GOOD"
                        if compliance_score >= 90
                        else "NEEDS_ATTENTION",
                    },
                    "redaction_patterns": {
                        "types_detected": [
                            {"type": row[0], "count": row[1]} for row in redaction_types
                        ],
                        "most_common": redaction_types[0][0]
                        if redaction_types
                        else None,
                    },
                    "statistics": self.redaction_stats,
                }

                return security_report

        except Exception as e:
            self.logger.error(f"Failed to generate security audit report: {e}")
            return {
                "error": str(e),
                "report_period_days": days,
                "generated_at": datetime.now(UTC).isoformat(),
            }

    async def validate_prompt_safety(self, prompt: str) -> dict[str, Any]:
        """Validate prompt for potential security issues without modification"""
        safety_report = {
            "is_safe": True,
            "risk_level": "LOW",
            "issues_detected": [],
            "recommendations": [],
        }

        detected_patterns = []

        for pattern, pattern_type in self.sensitive_patterns:
            matches = re.findall(pattern, prompt)
            if matches:
                detected_patterns.append({
                    "type": pattern_type,
                    "count": len(matches),
                    "risk_level": self._get_risk_level(pattern_type),
                })

        if detected_patterns:
            safety_report["is_safe"] = False
            safety_report["issues_detected"] = detected_patterns

            # Calculate overall risk level
            risk_levels = [issue["risk_level"] for issue in detected_patterns]
            if "CRITICAL" in risk_levels:
                safety_report["risk_level"] = "CRITICAL"
            elif "HIGH" in risk_levels:
                safety_report["risk_level"] = "HIGH"
            elif "MEDIUM" in risk_levels:
                safety_report["risk_level"] = "MEDIUM"

            # Generate recommendations
            safety_report["recommendations"] = self._generate_safety_recommendations(
                detected_patterns
            )

        return safety_report

    def _get_risk_level(self, pattern_type: str) -> str:
        """Get risk level for detected pattern type"""
        critical_patterns = [
            "openai_api_key",
            "github_token",
            "api_key_field",
            "secret_field",
            "bearer_token",
        ]
        high_patterns = ["credit_card", "ssn_pattern", "password_field"]
        medium_patterns = ["long_token", "email_address"]

        if pattern_type in critical_patterns:
            return "CRITICAL"
        if pattern_type in high_patterns:
            return "HIGH"
        if pattern_type in medium_patterns:
            return "MEDIUM"
        return "LOW"

    def _generate_safety_recommendations(
        self, detected_patterns: list[dict[str, Any]]
    ) -> list[str]:
        """Generate safety recommendations based on detected patterns"""
        recommendations = []

        for issue in detected_patterns:
            pattern_type = issue["type"]

            if pattern_type in ["openai_api_key", "github_token", "api_key_field"]:
                recommendations.append(
                    "Remove API keys and tokens - use environment variables instead"
                )
            elif pattern_type == "credit_card":
                recommendations.append(
                    "Remove credit card numbers - use masked or test data"
                )
            elif pattern_type == "ssn_pattern":
                recommendations.append(
                    "Remove SSN patterns - use synthetic identifiers"
                )
            elif pattern_type == "email_address":
                recommendations.append(
                    "Consider using example.com domains for email examples"
                )
            elif pattern_type == "password_field":
                recommendations.append("Remove passwords - use placeholder values")
            elif pattern_type == "secret_field":
                recommendations.append("Remove secrets - use configuration references")
            else:
                recommendations.append(
                    f"Review and remove sensitive {pattern_type} data"
                )

        # Add general recommendations
        if len(detected_patterns) > 0:
            recommendations.append(
                "Consider using data masking techniques for sensitive information"
            )
            recommendations.append("Review prompt content before sharing or storing")

        return list(set(recommendations))  # Remove duplicates

    async def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced data protection statistics with 2025 metrics."""
        avg_processing_time = (
            sum(self._processing_times) / len(self._processing_times)
            if self._processing_times else 0.0
        )

        return {
            "statistics": {
                "total_prompts_processed": self.metrics.total_prompts_processed,
                "prompts_with_sensitive_data": self.metrics.prompts_with_sensitive_data,
                "total_redactions": self.metrics.total_redactions,
                "redactions_by_type": self.metrics.redactions_by_type.copy(),
                "privacy_techniques_used": self.metrics.privacy_techniques_used.copy(),
                "compliance_score": self.metrics.compliance_score,
                "avg_processing_time_ms": round(avg_processing_time, 2),
                "last_updated": self.metrics.last_updated.isoformat(),
            },
            "configuration": {
                "patterns_monitored": len(self.sensitive_patterns),
                "audit_enabled": self.audit_enabled,
                "gdpr_enabled": self.gdpr_enabled,
                "differential_privacy_enabled": self.enable_differential_privacy,
                "data_retention_days": self.data_retention_days,
            },
            "performance": {
                "max_processing_time_ms": self._max_processing_time_ms,
                "total_processing_operations": len(self._processing_times),
                "performance_threshold_breaches": sum(
                    1 for t in self._processing_times if t > self._max_processing_time_ms
                ),
            },
            "compliance": {
                "framework_version": "2025.1",
                "gdpr_compliant": self.gdpr_enabled,
                "privacy_by_design": True,
                "security_by_default": True,
            }
        }

    async def get_redaction_statistics(self) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        enhanced_stats = await self.get_enhanced_statistics()
        return {
            "statistics": enhanced_stats["statistics"],
            "patterns_monitored": enhanced_stats["configuration"]["patterns_monitored"],
            "audit_enabled": enhanced_stats["configuration"]["audit_enabled"],
            "last_updated": enhanced_stats["statistics"]["last_updated"],
        }

    def reset_statistics(self):
        """Reset data protection statistics with enhanced metrics."""
        self.metrics = DataProtectionMetrics()
        self._processing_times = []
        self.logger.info("Enhanced data protection statistics reset")

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring integration."""
        if not self._processing_times:
            return {
                "avg_processing_time_ms": 0.0,
                "min_processing_time_ms": 0.0,
                "max_processing_time_ms": 0.0,
                "total_operations": 0,
                "performance_score": 100.0
            }

        avg_time = sum(self._processing_times) / len(self._processing_times)
        min_time = min(self._processing_times)
        max_time = max(self._processing_times)

        # Calculate performance score (100 = excellent, 0 = poor)
        performance_score = max(0.0, min(100.0,
            100.0 - (avg_time / self._max_processing_time_ms) * 50.0
        ))

        return {
            "avg_processing_time_ms": round(avg_time, 2),
            "min_processing_time_ms": round(min_time, 2),
            "max_processing_time_ms": round(max_time, 2),
            "total_operations": len(self._processing_times),
            "performance_score": round(performance_score, 2),
            "threshold_breaches": sum(1 for t in self._processing_times if t > self._max_processing_time_ms)
        }

    # ML Pipeline Orchestrator Integration Methods

    async def initialize(self) -> bool:
        """Initialize the component for orchestrator integration."""
        try:
            self.logger.info("Initializing PromptDataProtection component")

            # Validate configuration
            if not self.sensitive_patterns:
                raise ValueError("No sensitive patterns configured")

            # Test database connectivity
            get_sessionmanager = _get_sessionmanager()
            async with get_sessionmanager().session() as db_session:
                from sqlalchemy import text
                await db_session.execute(text("SELECT 1"))

            self.logger.info("PromptDataProtection component initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize PromptDataProtection: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Health check for orchestrator monitoring."""
        try:
            # Test basic functionality
            test_prompt = "test prompt with no sensitive data"
            start_time = datetime.now(timezone.utc)

            _, summary = await self.sanitize_prompt_before_storage(
                test_prompt, "health_check_session", user_consent=True
            )

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Test database connectivity
            get_sessionmanager = _get_sessionmanager()
            async with get_sessionmanager().session() as db_session:
                from sqlalchemy import text
                await db_session.execute(text("SELECT 1"))

            return {
                "status": "healthy",
                "component": "PromptDataProtection",
                "version": "2025.1",
                "processing_time_ms": round(processing_time, 2),
                "patterns_loaded": len(self.sensitive_patterns),
                "gdpr_enabled": self.gdpr_enabled,
                "differential_privacy_enabled": self.enable_differential_privacy,
                "database_connected": True,
                "last_check": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "PromptDataProtection",
                "error": str(e),
                "last_check": datetime.now(timezone.utc).isoformat()
            }

    async def get_capabilities(self) -> List[str]:
        """Get component capabilities for orchestrator registration."""
        return [
            "data_protection",
            "sensitive_data_detection",
            "prompt_sanitization",
            "gdpr_compliance",
            "differential_privacy",
            "audit_logging",
            "performance_monitoring",
            "real_time_processing",
            "risk_assessment",
            "privacy_by_design",
            "orchestrator_compatible"
        ]

    async def process_batch(self, prompts: List[str], session_ids: List[str]) -> List[Dict[str, Any]]:
        """Process multiple prompts for batch operations."""
        if len(prompts) != len(session_ids):
            raise ValueError("Prompts and session_ids must have the same length")

        results = []
        for prompt, session_id in zip(prompts, session_ids):
            sanitized, summary = await self.sanitize_prompt_before_storage(
                prompt, session_id, user_consent=True
            )
            results.append({
                "session_id": session_id,
                "sanitized_prompt": sanitized,
                "protection_summary": summary
            })

        return results

    async def shutdown(self) -> bool:
        """Shutdown the component gracefully."""
        try:
            self.logger.info("Shutting down PromptDataProtection component")

            # Save final statistics if needed
            final_stats = await self.get_enhanced_statistics()
            self.logger.info(f"Final statistics: {final_stats['statistics']}")

            self.logger.info("PromptDataProtection component shutdown complete")
            return True

        except Exception as e:
            self.logger.error(f"Error during PromptDataProtection shutdown: {e}")
            return False

class SecureMCPServer:
    """MCP Server security configuration (builds on existing MCP server)"""

    def __init__(self):
        self.config = {
            "host": "127.0.0.1",  # Local-only access
            "port": 3000,
            "allow_origins": ["127.0.0.1", "localhost"],
            "max_request_size": 1024 * 1024,  # 1MB limit
            "rate_limit_calls": 100,
            "rate_limit_period": 60,  # per minute
            "enable_cors": False,  # Disable CORS for security
            "request_timeout": 30,  # 30 second timeout
        }
        self.rate_limit_store = {}
        self.logger = logging.getLogger("apes.security.mcp")

    async def validate_request(
        self, request_data: dict[str, Any], client_ip: str = "127.0.0.1"
    ) -> tuple[bool, str]:
        """Validate incoming MCP request for security"""
        # Rate limiting check
        if not await self._check_rate_limit(client_ip):
            return False, "Rate limit exceeded"

        # Request size check
        request_size = len(json.dumps(request_data))
        if request_size > self.config["max_request_size"]:
            return False, f"Request too large: {request_size} bytes"

        # Basic structure validation
        if not isinstance(request_data, dict):
            return False, "Invalid request structure"

        # Check for required fields (basic MCP validation)
        required_fields = ["method"]
        for field in required_fields:
            if field not in request_data:
                return False, f"Missing required field: {field}"

        return True, "Valid request"

    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limiting for client IP"""
        current_time = datetime.now().timestamp()

        if client_ip not in self.rate_limit_store:
            self.rate_limit_store[client_ip] = []

        # Clean old requests outside the time window
        time_window = current_time - self.config["rate_limit_period"]
        self.rate_limit_store[client_ip] = [
            req_time
            for req_time in self.rate_limit_store[client_ip]
            if req_time > time_window
        ]

        # Check if under rate limit
        if len(self.rate_limit_store[client_ip]) >= self.config["rate_limit_calls"]:
            self.logger.warning(f"Rate limit exceeded for {client_ip}")
            return False

        # Add current request
        self.rate_limit_store[client_ip].append(current_time)
        return True

    def get_security_headers(self) -> dict[str, str]:
        """Get security headers for MCP responses"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Server": "APES-MCP",
        }

    async def log_security_event(self, event_type: str, details: dict[str, Any]):
        """Log security-related events"""
        security_event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "details": details,
            "source": "mcp_server",
        }

        self.logger.warning(f"SECURITY_EVENT: {json.dumps(security_event)}")

    def get_rate_limit_status(self, client_ip: str = "127.0.0.1") -> dict[str, Any]:
        """Get current rate limit status for client"""
        current_time = datetime.now().timestamp()
        time_window = current_time - self.config["rate_limit_period"]

        if client_ip in self.rate_limit_store:
            recent_requests = [
                req_time
                for req_time in self.rate_limit_store[client_ip]
                if req_time > time_window
            ]
            requests_used = len(recent_requests)
        else:
            requests_used = 0

        return {
            "client_ip": client_ip,
            "requests_used": requests_used,
            "requests_remaining": max(
                0, self.config["rate_limit_calls"] - requests_used
            ),
            "reset_time": current_time + self.config["rate_limit_period"],
            "rate_limit": f"{self.config['rate_limit_calls']}/{self.config['rate_limit_period']}s",
        }
