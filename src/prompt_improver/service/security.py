"""APES security framework with data protection and audit logging.
Implements Task 4: Security Framework from Phase 2.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any

from rich.console import Console

from ..database import get_sessionmanager
from ..services.analytics import AnalyticsService


class PromptDataProtection:
    """Protect sensitive data in prompts with audit logging"""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

        # Research-validated sensitive pattern detection (flexible and realistic)
        self.sensitive_patterns = [
            (
                r"sk-[a-zA-Z0-9]{40,60}",
                "openai_api_key",
            ),  # Flexible length for real OpenAI keys
            (r"ghp_[a-zA-Z0-9]{36,50}", "github_token"),  # Flexible GitHub token length
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email_address"),
            (r"\b\d{3}-\d{2}-\d{4}\b", "ssn_pattern"),
            (r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b", "credit_card"),
            (r"\b[A-Z0-9]{20,}\b", "long_token"),
            (r"password[\s]*[:=][\s]*[^\s]+", "password_field"),
            (r"api_key[\s]*[:=][\s]*[^\s]+", "api_key_field"),
            (r"secret[\s]*[:=][\s]*[^\s]+", "secret_field"),
            (r"\b(?:Bearer|Token)\s+[A-Za-z0-9\-._~+/]+=*", "bearer_token"),
        ]

        # Integration with existing database models
        self.audit_enabled = True
        self.logger = logging.getLogger("apes.security")

        # Statistics tracking
        self.redaction_stats = {
            "total_prompts_processed": 0,
            "prompts_with_sensitive_data": 0,
            "total_redactions": 0,
            "redactions_by_type": {},
        }

    async def sanitize_prompt_before_storage(
        self, prompt: str, session_id: str
    ) -> tuple[str, dict[str, Any]]:
        """Remove sensitive information before storage in existing database"""
        sanitized = prompt
        redactions_made = []
        redaction_details = {}

        self.redaction_stats["total_prompts_processed"] += 1

        for pattern, pattern_type in self.sensitive_patterns:
            matches = re.findall(pattern, sanitized)
            if matches:
                # Replace with pattern-specific redaction
                redaction_placeholder = f"[REDACTED_{pattern_type.upper()}]"
                sanitized = re.sub(pattern, redaction_placeholder, sanitized)

                redactions_made.extend(matches)
                redaction_details[pattern_type] = {
                    "count": len(matches),
                    "placeholder": redaction_placeholder,
                }

                # Update statistics
                if pattern_type not in self.redaction_stats["redactions_by_type"]:
                    self.redaction_stats["redactions_by_type"][pattern_type] = 0
                self.redaction_stats["redactions_by_type"][pattern_type] += len(matches)

        # Update statistics if redactions were made
        if redactions_made:
            self.redaction_stats["prompts_with_sensitive_data"] += 1
            self.redaction_stats["total_redactions"] += len(redactions_made)

        # Log redactions for audit (integrate with existing audit system)
        if redactions_made and self.audit_enabled:
            await self.audit_redaction(
                session_id, len(redactions_made), redaction_details
            )

        redaction_summary = {
            "redactions_made": len(redactions_made),
            "redaction_types": list(redaction_details.keys()),
            "redaction_details": redaction_details,
            "sanitized_length": len(sanitized),
            "original_length": len(prompt),
        }

        return sanitized, redaction_summary

    async def audit_redaction(
        self, session_id: str, redaction_count: int, redaction_details: dict[str, Any]
    ):
        """Audit log redactions using existing database structure"""
        try:
            # Use existing database connection from Phase 1
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
                    "audit_timestamp": datetime.utcnow().isoformat(),
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
                            "audit_timestamp": datetime.utcnow().isoformat(),
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
                            "started_at": datetime.utcnow(),
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
            analytics = AnalyticsService()

            # Get basic analytics data using correct session manager
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
                    "generated_at": datetime.utcnow().isoformat(),
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
                "generated_at": datetime.utcnow().isoformat(),
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

    async def get_redaction_statistics(self) -> dict[str, Any]:
        """Get current redaction statistics"""
        return {
            "statistics": self.redaction_stats.copy(),
            "patterns_monitored": len(self.sensitive_patterns),
            "audit_enabled": self.audit_enabled,
            "last_updated": datetime.utcnow().isoformat(),
        }

    def reset_statistics(self):
        """Reset redaction statistics (for testing or periodic reset)"""
        self.redaction_stats = {
            "total_prompts_processed": 0,
            "prompts_with_sensitive_data": 0,
            "total_redactions": 0,
            "redactions_by_type": {},
        }

        self.logger.info("Redaction statistics reset")


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
            "timestamp": datetime.utcnow().isoformat(),
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
