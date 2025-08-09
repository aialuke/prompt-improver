"""Unified Authentication Manager for Local MCP Server Tools

Consolidates authentication patterns from:
- UnifiedKeyManager (key rotation and encryption)
- SessionStore (session management)
- Rate limiting middleware
- Authentication testing patterns

Provides secure, simplified authentication optimized for local tools without
JWT/OAuth complexity while maintaining enterprise-grade security standards.

Following 2025 Security Best Practices:
- Fail-secure by default (no fail-open vulnerabilities)
- Zero-trust architecture
- 256-bit cryptographically secure API keys
- Comprehensive audit logging
- Integration with unified infrastructure
"""
import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple
from sqlmodel import Field, SQLModel
from prompt_improver.database.unified_connection_manager import ManagerMode, SecurityContext, SecurityPerformanceMetrics, SecurityThreatScore, SecurityValidationResult, create_security_context, create_security_context_from_auth_result, get_unified_manager
from prompt_improver.security.key_manager import AuditEvent, SecurityAuditLogger, SecurityLevel, UnifiedKeyManager
from prompt_improver.security.unified_rate_limiter import RateLimitTier, get_unified_rate_limiter
from prompt_improver.utils.session_store import SessionStore
logger = logging.getLogger(__name__)

class AuthenticationMethod(str, Enum):
    """Authentication methods supported by unified manager."""
    API_KEY = 'api_key'
    SESSION_TOKEN = 'session_token'
    SYSTEM_TOKEN = 'system_token'

class AuthenticationStatus(str, Enum):
    """Authentication status results."""
    SUCCESS = 'success'
    FAILED = 'failed'
    EXPIRED = 'expired'
    REVOKED = 'revoked'
    RATE_LIMITED = 'rate_limited'
    SYSTEM_ERROR = 'system_error'

class AuthenticationResult(SQLModel):
    """Complete authentication result with security context and audit data."""
    success: bool = Field(description='Whether authentication succeeded')
    status: AuthenticationStatus = Field(description='Authentication status result')
    agent_id: str | None = Field(default=None, min_length=1, max_length=100, description='Authenticated agent ID')
    authentication_method: AuthenticationMethod = Field(description='Method used for authentication')
    security_context: SecurityContext | None = Field(default=None, description='Established security context')
    session_id: str | None = Field(default=None, min_length=8, max_length=128, description='Session identifier if applicable')
    rate_limit_tier: str = Field(min_length=1, max_length=50, description='Assigned rate limiting tier')
    error_message: str | None = Field(default=None, max_length=500, description='Error message if authentication failed')
    audit_metadata: dict[str, Any] = Field(default_factory=dict, description='Metadata for security audit logging')
    performance_metrics: dict[str, float] = Field(default_factory=dict, description='Authentication performance metrics')

class APIKeyData(SQLModel):
    """API key metadata and management data."""
    key_id: str = Field(min_length=8, max_length=64, description='Unique API key identifier')
    api_key_hash: str = Field(min_length=64, max_length=64, description='SHA-256 hash of API key')
    agent_id: str = Field(min_length=1, max_length=100, description='Agent identifier owning this key')
    permissions: list[str] = Field(description='List of permissions granted to this API key')
    tier: str = Field(min_length=1, max_length=50, description='Rate limiting tier for this key')
    created_at: datetime = Field(description='Timestamp when API key was created')
    expires_at: datetime | None = Field(default=None, description='Timestamp when API key expires')
    last_used: datetime | None = Field(default=None, description='Timestamp of last API key usage')
    usage_count: int = Field(ge=0, description='Number of times this API key has been used')
    active: bool = Field(default=True, description='Whether this API key is currently active')

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {'key_id': self.key_id, 'api_key_hash': self.api_key_hash, 'agent_id': self.agent_id, 'permissions': self.permissions, 'tier': self.tier, 'created_at': self.created_at.isoformat(), 'expires_at': self.expires_at.isoformat() if self.expires_at else None, 'last_used': self.last_used.isoformat() if self.last_used else None, 'usage_count': self.usage_count, 'active': self.active}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'APIKeyData':
        """Create from dictionary."""
        return cls(key_id=data['key_id'], api_key_hash=data['api_key_hash'], agent_id=data['agent_id'], permissions=data['permissions'], tier=data['tier'], created_at=datetime.fromisoformat(data['created_at']), expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None, last_used=datetime.fromisoformat(data['last_used']) if data['last_used'] else None, usage_count=data['usage_count'], active=data['active'])

class AuthenticationSession(SQLModel):
    """Active authentication session data."""
    session_id: str = Field(min_length=8, max_length=128, description='Unique session identifier')
    agent_id: str = Field(min_length=1, max_length=100, description='Agent identifier for this session')
    created_at: datetime = Field(description='Timestamp when session was created')
    expires_at: datetime = Field(description='Timestamp when session expires')
    session_data: dict[str, Any] = Field(description='Session-specific data and context')

class APIKeyManager:
    """Secure API key management with 256-bit keys and comprehensive validation.

    Features:
    - 256-bit cryptographically secure API key generation
    - Key metadata and permissions management
    - Integration with UnifiedKeyManager for encryption
    - Expiration and revocation support
    - Comprehensive audit logging
    """

    def __init__(self, key_manager: UnifiedKeyManager):
        self.key_manager = key_manager
        self.connection_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        self.api_keys: dict[str, APIKeyData] = {}
        self.security_auditor = SecurityAuditLogger('APIKeyManager')

    async def generate_api_key(self, agent_id: str, permissions: list[str], tier: str='basic', expires_hours: int | None=None) -> tuple[str, dict[str, Any]]:
        """Generate 256-bit secure API key with comprehensive metadata."""
        start_time = time.perf_counter()
        try:
            key_bytes = secrets.token_bytes(32)
            api_key = f"pi_key_{base64.urlsafe_b64encode(key_bytes).decode().rstrip('=')}"
            key_data = APIKeyData(key_id=f'key_{int(time.time() * 1000000)}', api_key_hash=self._hash_api_key(api_key), agent_id=agent_id, permissions=permissions, tier=tier, created_at=datetime.utcnow(), expires_at=datetime.utcnow() + timedelta(hours=expires_hours) if expires_hours else None, last_used=None, usage_count=0, active=True)
            encrypted_data, encryption_key_id = self.key_manager.encrypt_string(json.dumps(key_data.to_dict()), None)
            security_context = await create_security_context(agent_id='api_key_manager', tier='system', authenticated=True)
            await self.connection_manager.set_cached(key=f'api_key:{self._hash_api_key(api_key)}', value={'encrypted_data': encrypted_data, 'encryption_key_id': encryption_key_id, 'agent_id': agent_id, 'tier': tier, 'active': True}, ttl_seconds=expires_hours * 3600 if expires_hours else None, security_context=security_context)
            self.api_keys[api_key] = key_data
            self.security_auditor.log_event(AuditEvent.KEY_GENERATED, {'key_id': key_data.key_id, 'agent_id': agent_id, 'tier': tier, 'permissions': permissions, 'expires_hours': expires_hours, 'generation_time_ms': (time.perf_counter() - start_time) * 1000}, SecurityLevel.HIGH)
            metadata = {'key_id': key_data.key_id, 'agent_id': agent_id, 'tier': tier, 'permissions': permissions, 'expires_at': key_data.expires_at.isoformat() if key_data.expires_at else None, 'generation_time_ms': (time.perf_counter() - start_time) * 1000}
            logger.info('Generated API key for agent {agent_id}: %s', key_data.key_id)
            return (api_key, metadata)
        except Exception as e:
            logger.error('API key generation failed for agent {agent_id}: %s', e)
            self.security_auditor.log_event(AuditEvent.SECURITY_VIOLATION, {'agent_id': agent_id, 'error': str(e), 'operation': 'api_key_generation'}, SecurityLevel.CRITICAL)
            raise RuntimeError(f'API key generation failed: {e!s}')

    async def validate_api_key(self, api_key: str) -> dict[str, Any]:
        """Validate API key and return comprehensive validation result."""
        start_time = time.perf_counter()
        try:
            if api_key in self.api_keys:
                key_data = self.api_keys[api_key]
                if key_data.active and (not key_data.expires_at or key_data.expires_at > datetime.utcnow()):
                    return self._create_validation_success(key_data, start_time)
            security_context = await create_security_context(agent_id='api_key_manager', tier='system', authenticated=True)
            cached_data = await self.connection_manager.get_cached(key=f'api_key:{self._hash_api_key(api_key)}', security_context=security_context)
            if not cached_data or not cached_data.get('active'):
                return self._create_validation_failure('API key not found or inactive', start_time)
            decrypted_json = self.key_manager.decrypt_string(cached_data['encrypted_data'], cached_data['encryption_key_id'])
            key_data = APIKeyData.from_dict(json.loads(decrypted_json))
            if key_data.expires_at and key_data.expires_at <= datetime.utcnow():
                return self._create_validation_failure('API key expired', start_time)
            key_data.last_used = datetime.utcnow()
            key_data.usage_count += 1
            updated_encrypted_data, _ = self.key_manager.encrypt_string(json.dumps(key_data.to_dict()), cached_data['encryption_key_id'])
            await self.connection_manager.set_cached(key=f'api_key:{self._hash_api_key(api_key)}', value={**cached_data, 'encrypted_data': updated_encrypted_data}, ttl_seconds=None, security_context=security_context)
            self.api_keys[api_key] = key_data
            return self._create_validation_success(key_data, start_time)
        except Exception as e:
            logger.error('API key validation error: %s', e)
            self.security_auditor.log_event(AuditEvent.SECURITY_VIOLATION, {'error': str(e), 'operation': 'api_key_validation', 'api_key_hash': self._hash_api_key(api_key)[:8] + '...'}, SecurityLevel.CRITICAL)
            return self._create_validation_failure(f'Validation system error: {e!s}', start_time)

    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key and remove from all caches."""
        try:
            api_key_hash = self._hash_api_key(api_key)
            if api_key in self.api_keys:
                key_data = self.api_keys[api_key]
                del self.api_keys[api_key]
                self.security_auditor.log_event(AuditEvent.KEY_DELETED, {'key_id': key_data.key_id, 'agent_id': key_data.agent_id, 'usage_count': key_data.usage_count}, SecurityLevel.HIGH)
            security_context = await create_security_context(agent_id='api_key_manager', tier='system', authenticated=True)
            await self.connection_manager.delete_cached(key=f'api_key:{api_key_hash}', security_context=security_context)
            logger.info('API key revoked: %s...', api_key_hash[:8])
            return True
        except Exception as e:
            logger.error('API key revocation failed: %s', e)
            return False

    async def get_api_key_status(self, api_key: str) -> dict[str, Any]:
        """Get comprehensive API key status information."""
        try:
            validation_result = await self.validate_api_key(api_key)
            if validation_result['valid']:
                return {'exists': True, 'type': 'api_key', 'agent_id': validation_result['agent_id'], 'tier': validation_result['tier'], 'permissions': validation_result['permissions'], 'active': True, 'usage_count': validation_result['usage_count'], 'age_hours': validation_result['age_hours']}
            return {'exists': False, 'active': False, 'error': validation_result['error']}
        except Exception as e:
            return {'exists': False, 'active': False, 'error': f'Status check failed: {e!s}'}

    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage using SHA-256."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _create_validation_success(self, key_data: APIKeyData, start_time: float) -> dict[str, Any]:
        """Create successful validation result."""
        return {'valid': True, 'agent_id': key_data.agent_id, 'key_id': key_data.key_id, 'permissions': key_data.permissions, 'tier': key_data.tier, 'age_hours': (datetime.utcnow() - key_data.created_at).total_seconds() / 3600, 'usage_count': key_data.usage_count, 'validation_time_ms': (time.perf_counter() - start_time) * 1000}

    def _create_validation_failure(self, error: str, start_time: float) -> dict[str, Any]:
        """Create failed validation result."""
        return {'valid': False, 'error': error, 'validation_time_ms': (time.perf_counter() - start_time) * 1000}

class UnifiedAuthenticationManager:
    """Unified authentication manager for local MCP server tools.

    Consolidates authentication patterns from multiple components:
    - UnifiedKeyManager integration for secure key operations
    - SessionStore integration for session management
    - UnifiedRateLimiter integration for abuse prevention
    - Comprehensive audit logging and security monitoring

    Features:
    - 256-bit secure API key generation and validation
    - Session token management with TTL and cleanup
    - Integration with unified infrastructure components
    - Fail-secure authentication policies (deny by default)
    - Zero-trust security architecture
    - Comprehensive audit logging with OpenTelemetry integration
    """

    def __init__(self, key_manager: UnifiedKeyManager | None=None, session_store: SessionStore | None=None, rate_limiter=None, enable_audit_logging: bool=True, fail_secure: bool=True):
        """Initialize unified authentication manager with comprehensive security."""
        self.key_manager = key_manager or UnifiedKeyManager()
        self.session_store = session_store or SessionStore()
        self.rate_limiter = rate_limiter
        self.api_key_manager = APIKeyManager(self.key_manager)
        self.fail_secure = fail_secure
        self.zero_trust_mode = True
        self.security_auditor = SecurityAuditLogger('UnifiedAuthManager') if enable_audit_logging else None
        self.active_sessions: dict[str, AuthenticationSession] = {}
        self.failed_attempts: dict[str, list[datetime]] = {}
        self.performance_metrics = {'authentication_attempts': 0, 'successful_authentications': 0, 'failed_authentications': 0, 'average_auth_time_ms': 0.0, 'api_key_validations': 0, 'session_token_validations': 0}
        logger.info('UnifiedAuthenticationManager initialized with fail-secure and zero-trust policies')

    async def initialize(self) -> None:
        """Initialize async components."""
        if not self.rate_limiter:
            self.rate_limiter = await get_unified_rate_limiter()
        asyncio.create_task(self._session_cleanup_task())
        logger.info('UnifiedAuthenticationManager async initialization complete')

    async def authenticate_request(self, request_context: dict[str, Any]) -> AuthenticationResult:
        """Authenticate incoming request using multiple authentication methods.

        Implements fail-secure policies and comprehensive audit logging.
        Supports API key and session token authentication optimized for local tools.

        Args:
            request_context: Request context containing headers and agent information

        Returns:
            AuthenticationResult with security context and performance metrics
        """
        start_time = time.perf_counter()
        self.performance_metrics['authentication_attempts'] += 1
        agent_id = request_context.get('agent_id', 'unknown')
        try:
            api_key = self._extract_api_key(request_context)
            session_token = self._extract_session_token(request_context)
            auth_method = self._determine_auth_method(api_key, session_token)
            if self._is_agent_locked_out(agent_id):
                return self._create_failed_result(agent_id, 'Too many failed attempts - agent locked out', auth_method, start_time, AuthenticationStatus.RATE_LIMITED)
            if auth_method == AuthenticationMethod.API_KEY and api_key:
                result = await self._authenticate_api_key(api_key, agent_id, request_context, start_time)
            elif auth_method == AuthenticationMethod.SESSION_TOKEN and session_token:
                result = await self._authenticate_session_token(session_token, agent_id, request_context, start_time)
            else:
                result = self._create_failed_result(agent_id, 'No valid authentication method provided', auth_method, start_time, AuthenticationStatus.FAILED)
            if result.success and result.security_context:
                await self._apply_rate_limiting(result, request_context)
            else:
                self._track_failed_attempt(agent_id)
            if result.success:
                self.performance_metrics['successful_authentications'] += 1
            else:
                self.performance_metrics['failed_authentications'] += 1
            auth_time = (time.perf_counter() - start_time) * 1000
            result.performance_metrics['total_auth_time_ms'] = auth_time
            total_auths = self.performance_metrics['authentication_attempts']
            current_avg = self.performance_metrics['average_auth_time_ms']
            self.performance_metrics['average_auth_time_ms'] = (current_avg * (total_auths - 1) + auth_time) / total_auths
            if self.security_auditor:
                await self._log_authentication_attempt(result, request_context, auth_time)
            return result
        except Exception as e:
            error_message = f'Authentication system error: {e!s}' if self.fail_secure else str(e)
            logger.error('Authentication error for agent {agent_id}: %s', e)
            if self.security_auditor:
                self.security_auditor.log_event(AuditEvent.SECURITY_VIOLATION, {'agent_id': agent_id, 'error': str(e), 'operation': 'authentication_request', 'fail_secure_applied': self.fail_secure}, SecurityLevel.CRITICAL)
            self.performance_metrics['failed_authentications'] += 1
            return self._create_failed_result(agent_id, error_message, AuthenticationMethod.API_KEY, start_time, AuthenticationStatus.SYSTEM_ERROR)

    async def generate_api_key(self, agent_id: str, permissions: list[str], tier: str='basic', expires_hours: int | None=None) -> tuple[str, dict[str, Any]]:
        """Generate secure 256-bit API key for agent with comprehensive metadata."""
        return await self.api_key_manager.generate_api_key(agent_id, permissions, tier, expires_hours)

    async def create_session_token(self, agent_id: str, authentication_data: dict[str, Any], expires_minutes: int=60) -> tuple[str, dict[str, Any]]:
        """Create session token for repeated operations with integrated session management."""
        start_time = time.perf_counter()
        try:
            session_token = f'session_{agent_id}_{secrets.token_urlsafe(32)}'
            session_data = {'agent_id': agent_id, 'created_at': datetime.utcnow().isoformat(), 'expires_at': (datetime.utcnow() + timedelta(minutes=expires_minutes)).isoformat(), 'authentication_method': AuthenticationMethod.SESSION_TOKEN.value, **authentication_data}
            success = await self.session_store.set(session_token, session_data)
            if not success:
                raise RuntimeError('Failed to create session token in session store')
            auth_session = AuthenticationSession(session_id=session_token, agent_id=agent_id, created_at=datetime.utcnow(), expires_at=datetime.utcnow() + timedelta(minutes=expires_minutes), session_data=session_data)
            self.active_sessions[session_token] = auth_session
            metadata = {'session_id': session_token, 'expires_at': auth_session.expires_at.isoformat(), 'agent_id': agent_id, 'creation_time_ms': (time.perf_counter() - start_time) * 1000}
            if self.security_auditor:
                self.security_auditor.log_event(AuditEvent.KEY_GENERATED, {'session_id': session_token, 'agent_id': agent_id, 'expires_minutes': expires_minutes, 'creation_time_ms': metadata['creation_time_ms']}, SecurityLevel.enhanced)
            logger.info('Created session token for agent %s: %s...', agent_id, session_token[:8])
            return (session_token, metadata)
        except Exception as e:
            logger.error('Session token creation failed for agent {agent_id}: %s', e)
            raise RuntimeError(f'Session token creation failed: {e!s}')

    async def revoke_authentication(self, identifier: str, revocation_type: str='api_key') -> bool:
        """Revoke API key or session token with comprehensive cleanup."""
        try:
            if revocation_type == 'api_key':
                return await self.api_key_manager.revoke_api_key(identifier)
            if revocation_type == 'session_token':
                return await self._revoke_session_token(identifier)
            logger.warning('Unknown revocation type: %s', revocation_type)
            return False
        except Exception as e:
            logger.error('Authentication revocation failed: %s', e)
            return False

    async def get_authentication_status(self, identifier: str) -> dict[str, Any]:
        """Get comprehensive status of API key or session token."""
        api_key_status = await self.api_key_manager.get_api_key_status(identifier)
        if api_key_status['exists']:
            return api_key_status
        session_data = await self.session_store.get(identifier)
        if session_data:
            return {'exists': True, 'type': 'session_token', 'agent_id': session_data.get('agent_id'), 'expires_at': session_data.get('expires_at'), 'active': datetime.fromisoformat(session_data['expires_at']) > datetime.utcnow(), 'age_minutes': (datetime.utcnow() - datetime.fromisoformat(session_data['created_at'])).total_seconds() / 60}
        return {'exists': False, 'active': False}

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive authentication performance metrics."""
        return {**self.performance_metrics, 'active_sessions_count': len(self.active_sessions), 'failed_attempt_agents_count': len(self.failed_attempts), 'api_key_cache_size': len(self.api_key_manager.api_keys), 'authentication_success_rate': self.performance_metrics['successful_authentications'] / max(1, self.performance_metrics['authentication_attempts']) * 100}

    async def _authenticate_api_key(self, api_key: str, agent_id: str, request_context: dict[str, Any], start_time: float) -> AuthenticationResult:
        """Authenticate using API key with comprehensive validation."""
        self.performance_metrics['api_key_validations'] += 1
        validation_result = await self.api_key_manager.validate_api_key(api_key)
        if not validation_result['valid']:
            return self._create_failed_result(agent_id, validation_result['error'], AuthenticationMethod.API_KEY, start_time, AuthenticationStatus.FAILED)
        security_context = await create_security_context_from_auth_result(agent_id=validation_result['agent_id'], tier=validation_result['tier'], authenticated=True, authentication_method='api_key', permissions=validation_result['permissions'], security_level='enhanced' if validation_result['tier'] in ['professional', 'enterprise'] else 'basic', expires_minutes=240, auth_result_metadata={'validation_result': validation_result, 'api_key_id': validation_result['key_id'], 'authentication_manager': 'unified_authentication_manager', 'authentication_timestamp': time.time()})
        validation_result_obj = SecurityValidationResult(validated=True, validation_method='api_key_validation', validation_timestamp=time.time(), validation_duration_ms=validation_result['validation_time_ms'], security_incidents=[], rate_limit_status='validated', encryption_required=False, audit_trail_id=f"api_key_{validation_result['key_id']}")
        security_context.validation_result = validation_result_obj
        threat_score = SecurityThreatScore(level='low', score=0.05, factors=['successful_api_key_auth'], last_updated=time.time())
        security_context.threat_score = threat_score
        performance_metrics = SecurityPerformanceMetrics(authentication_time_ms=validation_result['validation_time_ms'], total_security_overhead_ms=validation_result['validation_time_ms'], operations_count=1, last_performance_check=time.time())
        security_context.performance_metrics = performance_metrics
        return AuthenticationResult(success=True, status=AuthenticationStatus.SUCCESS, agent_id=validation_result['agent_id'], authentication_method=AuthenticationMethod.API_KEY, security_context=security_context, session_id=None, rate_limit_tier=validation_result['tier'], error_message=None, audit_metadata={'api_key_id': validation_result['key_id'], 'permissions': validation_result['permissions'], 'key_age_hours': validation_result['age_hours'], 'usage_count': validation_result['usage_count']}, performance_metrics={'validation_time_ms': validation_result['validation_time_ms']})

    async def _authenticate_session_token(self, session_token: str, agent_id: str, request_context: dict[str, Any], start_time: float) -> AuthenticationResult:
        """Authenticate using session token with expiration and lifecycle management."""
        self.performance_metrics['session_token_validations'] += 1
        session_data = await self.session_store.get(session_token)
        if not session_data:
            return self._create_failed_result(agent_id, 'Invalid session token', AuthenticationMethod.SESSION_TOKEN, start_time, AuthenticationStatus.FAILED)
        expires_at = datetime.fromisoformat(session_data['expires_at'])
        if expires_at <= datetime.utcnow():
            await self._revoke_session_token(session_token)
            return self._create_failed_result(agent_id, 'Session token expired', AuthenticationMethod.SESSION_TOKEN, start_time, AuthenticationStatus.EXPIRED)
        security_context = await create_security_context_from_auth_result(agent_id=session_data['agent_id'], tier=session_data.get('tier', 'basic'), authenticated=True, authentication_method='session_token', auth_result_metadata={'session_data': session_data, 'session_token': session_token, 'authentication_manager': 'unified_authentication_manager', 'session_validation_timestamp': time.time()})
        await self.session_store.touch(session_token)
        return AuthenticationResult(success=True, status=AuthenticationStatus.SUCCESS, agent_id=session_data['agent_id'], authentication_method=AuthenticationMethod.SESSION_TOKEN, security_context=security_context, session_id=session_token, rate_limit_tier=session_data.get('tier', 'basic'), error_message=None, audit_metadata={'session_age_minutes': (datetime.utcnow() - datetime.fromisoformat(session_data['created_at'])).total_seconds() / 60, 'session_data_keys': list(session_data.keys())})

    def _extract_api_key(self, request_context: dict[str, Any]) -> str | None:
        """Extract API key from request context headers."""
        headers = request_context.get('headers', {})
        return headers.get('x-api-key') or headers.get('authorization', '').replace('Bearer ', '') or headers.get('api-key')

    def _extract_session_token(self, request_context: dict[str, Any]) -> str | None:
        """Extract session token from request context headers."""
        headers = request_context.get('headers', {})
        return headers.get('x-session-token') or headers.get('session-token')

    def _determine_auth_method(self, api_key: str | None, session_token: str | None) -> AuthenticationMethod:
        """Determine authentication method based on available credentials."""
        if api_key:
            return AuthenticationMethod.API_KEY
        if session_token:
            return AuthenticationMethod.SESSION_TOKEN
        return AuthenticationMethod.API_KEY

    def _is_agent_locked_out(self, agent_id: str) -> bool:
        """Check if agent is locked out due to failed attempts."""
        if agent_id not in self.failed_attempts:
            return False
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_attempts = [attempt for attempt in self.failed_attempts[agent_id] if attempt > one_hour_ago]
        return len(recent_attempts) >= 10

    def _track_failed_attempt(self, agent_id: str) -> None:
        """Track failed authentication attempt for rate limiting."""
        if agent_id not in self.failed_attempts:
            self.failed_attempts[agent_id] = []
        self.failed_attempts[agent_id].append(datetime.utcnow())
        one_day_ago = datetime.utcnow() - timedelta(hours=24)
        self.failed_attempts[agent_id] = [attempt for attempt in self.failed_attempts[agent_id] if attempt > one_day_ago]

    async def _apply_rate_limiting(self, auth_result: AuthenticationResult, request_context: dict[str, Any]) -> None:
        """Apply rate limiting using unified rate limiter."""
        if not self.rate_limiter or not auth_result.security_context:
            return
        try:
            await self.rate_limiter.check_rate_limit(agent_id=auth_result.agent_id, tier=auth_result.rate_limit_tier, authenticated=True)
        except Exception as e:
            logger.warning('Rate limiting error for {auth_result.agent_id}: %s', e)

    async def _revoke_session_token(self, session_token: str) -> bool:
        """Revoke session token and clean up all references."""
        try:
            await self.session_store.delete(session_token)
            if session_token in self.active_sessions:
                session = self.active_sessions[session_token]
                del self.active_sessions[session_token]
                if self.security_auditor:
                    self.security_auditor.log_event(AuditEvent.KEY_DELETED, {'session_id': session_token, 'agent_id': session.agent_id, 'session_duration_minutes': (datetime.utcnow() - session.created_at).total_seconds() / 60}, SecurityLevel.enhanced)
            logger.info('Session token revoked: %s...', session_token[:8])
            return True
        except Exception as e:
            logger.error('Session token revocation failed: %s', e)
            return False

    def _create_failed_result(self, agent_id: str, error_message: str, auth_method: AuthenticationMethod, start_time: float, status: AuthenticationStatus) -> AuthenticationResult:
        """Create standardized failed authentication result."""
        return AuthenticationResult(success=False, status=status, agent_id=agent_id, authentication_method=auth_method, security_context=None, session_id=None, rate_limit_tier='basic', error_message=error_message, audit_metadata={'failure_reason': error_message, 'authentication_method': auth_method.value}, performance_metrics={'auth_time_ms': (time.perf_counter() - start_time) * 1000})

    async def _log_authentication_attempt(self, result: AuthenticationResult, request_context: dict[str, Any], auth_time_ms: float) -> None:
        """Log authentication attempt with comprehensive audit data."""
        self.security_auditor.log_event(AuditEvent.KEY_ACCESSED if result.success else AuditEvent.SECURITY_VIOLATION, {'agent_id': result.agent_id, 'authentication_method': result.authentication_method.value, 'success': result.success, 'status': result.status.value, 'auth_time_ms': auth_time_ms, 'rate_limit_tier': result.rate_limit_tier, 'error_message': result.error_message, 'request_context_keys': list(request_context.keys()), **result.audit_metadata}, SecurityLevel.HIGH if result.success else SecurityLevel.CRITICAL)

    async def _session_cleanup_task(self) -> None:
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(1800)
                expired_sessions = []
                current_time = datetime.utcnow()
                for session_id, session in self.active_sessions.items():
                    if session.expires_at <= current_time:
                        expired_sessions.append(session_id)
                for session_id in expired_sessions:
                    await self._revoke_session_token(session_id)
                one_day_ago = current_time - timedelta(hours=24)
                for agent_id in list(self.failed_attempts.keys()):
                    self.failed_attempts[agent_id] = [attempt for attempt in self.failed_attempts[agent_id] if attempt > one_day_ago]
                    if not self.failed_attempts[agent_id]:
                        del self.failed_attempts[agent_id]
                if expired_sessions:
                    logger.info('Cleaned up %s expired sessions', len(expired_sessions))
            except Exception as e:
                logger.error('Session cleanup task error: %s', e)
_unified_auth_manager: UnifiedAuthenticationManager | None = None

async def get_unified_authentication_manager() -> UnifiedAuthenticationManager:
    """Get global unified authentication manager instance with proper initialization."""
    global _unified_auth_manager
    if _unified_auth_manager is None:
        _unified_auth_manager = UnifiedAuthenticationManager(enable_audit_logging=True, fail_secure=True)
        await _unified_auth_manager.initialize()
    return _unified_auth_manager

def require_local_authentication(permissions: list[str]=None, tier: str='basic'):
    """Decorator to require authentication for MCP tool functions.

    Provides seamless authentication integration for local MCP server tools
    without complex web authentication flows.

    Args:
        permissions: Required permissions for the operation
        tier: Minimum rate limiting tier required

    Returns:
        Decorated function with authentication enforcement
    """

    def decorator(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            auth_context = kwargs.get('auth_context', {})
            auth_manager = await get_unified_authentication_manager()
            auth_result = await auth_manager.authenticate_request(auth_context)
            if not auth_result.success:
                return {'error': 'Authentication required', 'message': auth_result.error_message, 'status': auth_result.status.value, 'authentication_required': True, 'supported_methods': ['api_key', 'session_token']}
            if permissions:
                user_permissions = auth_result.audit_metadata.get('permissions', [])
                if not all((perm in user_permissions for perm in permissions)):
                    return {'error': 'Insufficient permissions', 'required_permissions': permissions, 'user_permissions': user_permissions, 'status': 'permission_denied'}
            kwargs['auth_result'] = auth_result
            kwargs['security_context'] = auth_result.security_context
            kwargs['authenticated_agent_id'] = auth_result.agent_id
            result = await func(*args, **kwargs)
            if isinstance(result, dict):
                result['authentication'] = {'agent_id': auth_result.agent_id, 'method': auth_result.authentication_method.value, 'tier': auth_result.rate_limit_tier, 'session_id': auth_result.session_id, 'auth_time_ms': auth_result.performance_metrics.get('total_auth_time_ms', 0)}
            return result
        return wrapper
    return decorator

class AuthenticationFlowOrchestrator:
    """Orchestrates authentication flows specifically designed for local MCP server tools.

    Provides simplified setup and management of authentication for local tools
    without the complexity of web-based authentication flows.
    """

    def __init__(self, auth_manager: UnifiedAuthenticationManager | None=None):
        self.auth_manager = auth_manager

    async def _get_auth_manager(self) -> UnifiedAuthenticationManager:
        """Get authentication manager instance."""
        if not self.auth_manager:
            self.auth_manager = await get_unified_authentication_manager()
        return self.auth_manager

    async def authenticate_local_tool(self, tool_config: dict[str, Any]) -> AuthenticationResult:
        """Authenticate local MCP tool using configuration.

        Args:
            tool_config: Tool configuration containing:
                - tool_name: Name of the tool
                - api_key: Optional API key
                - session_file: Optional session file path
                - permissions: Required permissions

        Returns:
            AuthenticationResult for the tool
        """
        auth_manager = await self._get_auth_manager()
        tool_name = tool_config.get('tool_name', 'unknown_tool')
        api_key = tool_config.get('api_key')
        session_file = tool_config.get('session_file')
        request_context = {'agent_id': f'local_tool_{tool_name}', 'tool_name': tool_name, 'local_authentication': True, 'headers': {}}
        if api_key:
            request_context['headers']['x-api-key'] = api_key
        if session_file and os.path.exists(session_file):
            try:
                with open(session_file) as f:
                    session_data = json.load(f)
                    if session_data.get('session_token'):
                        request_context['headers']['x-session-token'] = session_data['session_token']
            except Exception as e:
                logger.warning('Failed to load session file {session_file}: %s', e)
        return await auth_manager.authenticate_request(request_context)

    async def setup_local_tool_authentication(self, tool_name: str, permissions: list[str], tier: str='basic', expires_hours: int | None=None) -> dict[str, Any]:
        """Set up authentication for a new local tool.

        Args:
            tool_name: Name of the tool
            permissions: Required permissions
            tier: Rate limiting tier
            expires_hours: Optional API key expiration

        Returns:
            Complete setup information including API key and session options
        """
        auth_manager = await self._get_auth_manager()
        agent_id = f'local_tool_{tool_name}'
        api_key, key_metadata = await auth_manager.generate_api_key(agent_id=agent_id, permissions=permissions, tier=tier, expires_hours=expires_hours)
        session_token, session_metadata = await auth_manager.create_session_token(agent_id=agent_id, authentication_data={'tool_name': tool_name, 'permissions': permissions, 'tier': tier, 'setup_timestamp': datetime.utcnow().isoformat()}, expires_minutes=480)
        return {'agent_id': agent_id, 'tool_name': tool_name, 'api_key': api_key, 'session_token': session_token, 'setup_complete': True, 'authentication_methods': ['api_key', 'session_token'], 'recommended_usage': 'api_key', 'key_metadata': key_metadata, 'session_metadata': session_metadata, 'setup_instructions': {'api_key_usage': "Add 'x-api-key' header to requests", 'session_token_usage': "Add 'x-session-token' header to requests", 'local_config_example': {'tool_name': tool_name, 'api_key': api_key, 'session_file': f'/tmp/{tool_name}_session.json'}}}

    async def save_session_file(self, session_token: str, session_metadata: dict[str, Any], file_path: str) -> bool:
        """Save session token to file for local tool usage."""
        try:
            session_file_data = {'session_token': session_token, 'created_at': datetime.utcnow().isoformat(), **session_metadata}
            with open(file_path, 'w') as f:
                json.dump(session_file_data, f, indent=2)
            logger.info('Session file saved: %s', file_path)
            return True
        except Exception as e:
            logger.error('Failed to save session file {file_path}: %s', e)
            return False
