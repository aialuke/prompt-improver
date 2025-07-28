"""MCP-specific JWT authentication service for 2025 security standards.

Implements agent-specific authentication with rate limiting tiers and permissions
according to Model Context Protocol security guidelines.
"""

import os
import secrets
import time
from typing import Any, Dict, List, Optional
from enum import Enum

import jwt

class AgentType(str, Enum):
    """Supported MCP agent types with specific capabilities."""
    CLAUDE_CODE = "claude-code"
    AUGMENT_CODE = "augment-code"
    EXTERNAL_AGENT = "external-agent"

class RateLimitTier(str, Enum):
    """Rate limiting tiers with different request allowances."""
    BASIC = "basic"          # 60 req/min, burst 90
    PROFESSIONAL = "professional"  # 300 req/min, burst 450
    ENTERPRISE = "enterprise"      # 1000+ req/min, burst 1500

class MCPPermission(str, Enum):
    """MCP-specific permissions for rule application operations."""
    RULE_READ = "rule:read"
    RULE_APPLY = "rule:apply"
    RULE_DISCOVER = "rule:discover"
    FEEDBACK_WRITE = "feedback:write"
    PERFORMANCE_READ = "performance:read"

class MCPAuthenticationService:
    """Enhanced authentication service for MCP agents with 2025 security standards."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize MCP authentication service.
        
        Args:
            secret_key: JWT secret key. If None, uses MCP_JWT_SECRET_KEY environment variable.
        """
        self.secret_key = secret_key or os.getenv("MCP_JWT_SECRET_KEY")
        if not self.secret_key:
            raise ValueError(
                "MCP_JWT_SECRET_KEY environment variable must be set for MCP authentication"
            )
        
        # Validate secret key strength (minimum 32 characters for 2025 standards)
        if len(self.secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters for security compliance")
        
        # Token expiry: 1 hour for security (as per MCP 2025 guidelines)
        self.token_expiry_seconds = 3600
        
        # Agent-specific permissions mapping
        self.agent_permissions = {
            AgentType.CLAUDE_CODE: [
                MCPPermission.RULE_READ,
                MCPPermission.RULE_APPLY,
                MCPPermission.RULE_DISCOVER,
                MCPPermission.FEEDBACK_WRITE,
                MCPPermission.PERFORMANCE_READ
            ],
            AgentType.AUGMENT_CODE: [
                MCPPermission.RULE_READ,
                MCPPermission.RULE_APPLY,
                MCPPermission.RULE_DISCOVER,
                MCPPermission.FEEDBACK_WRITE,
                MCPPermission.PERFORMANCE_READ
            ],
            AgentType.EXTERNAL_AGENT: [
                MCPPermission.RULE_READ,
                MCPPermission.RULE_APPLY,
                MCPPermission.FEEDBACK_WRITE
            ]
        }
        
        # Default rate limit tiers by agent type
        self.default_rate_tiers = {
            AgentType.CLAUDE_CODE: RateLimitTier.PROFESSIONAL,
            AgentType.AUGMENT_CODE: RateLimitTier.PROFESSIONAL,
            AgentType.EXTERNAL_AGENT: RateLimitTier.BASIC
        }

    def create_agent_token(
        self,
        agent_id: str,
        agent_type: AgentType,
        rate_limit_tier: Optional[RateLimitTier] = None,
        custom_permissions: Optional[List[MCPPermission]] = None
    ) -> str:
        """Create JWT token for MCP agent with 2025 security standards.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (claude-code, augment-code, external-agent)
            rate_limit_tier: Rate limiting tier (defaults to agent type default)
            custom_permissions: Custom permissions (defaults to agent type permissions)
            
        Returns:
            JWT token string
            
        Raises:
            ValueError: If agent_id is invalid or agent_type is unsupported
        """
        if not agent_id or len(agent_id.strip()) == 0:
            raise ValueError("agent_id cannot be empty")
        
        if agent_type not in AgentType:
            raise ValueError(f"Unsupported agent_type: {agent_type}")
        
        # Use defaults if not specified
        tier = rate_limit_tier or self.default_rate_tiers[agent_type]
        permissions = custom_permissions or self.agent_permissions[agent_type]
        
        # Create JWT payload according to MCP 2025 specification
        current_time = int(time.time())
        payload = {
            # Standard JWT claims
            "iss": "apes-mcp-server",           # Issuer
            "sub": agent_id,                    # Subject (agent ID)
            "aud": "rule-application",          # Audience
            "iat": current_time,                # Issued at
            "exp": current_time + self.token_expiry_seconds,  # Expires (1 hour)
            "jti": secrets.token_urlsafe(16),   # JWT ID for uniqueness
            
            # MCP-specific claims
            "agent_type": agent_type.value,
            "permissions": [p.value for p in permissions],
            "rate_limit_tier": tier.value,
            
            # Security metadata
            "token_version": "2025.1",
            "security_level": "production"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def validate_agent_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate MCP agent JWT token and return payload if valid.
        
        Args:
            token: JWT token string
            
        Returns:
            Token payload dict if valid, None if invalid/expired
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"],
                options={
                    "verify_exp": True,
                    "verify_iss": True,
                    "verify_aud": True,
                    "verify_iat": True
                },
                issuer="apes-mcp-server",
                audience="rule-application"
            )
            
            # Validate MCP-specific claims
            required_claims = ["agent_type", "permissions", "rate_limit_tier"]
            for claim in required_claims:
                if claim not in payload:
                    return None
            
            # Validate agent_type is supported
            if payload["agent_type"] not in [t.value for t in AgentType]:
                return None
            
            # Validate rate_limit_tier is supported
            if payload["rate_limit_tier"] not in [t.value for t in RateLimitTier]:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None

    def check_permission(self, token_payload: Dict[str, Any], required_permission: MCPPermission) -> bool:
        """Check if token has required permission.
        
        Args:
            token_payload: Validated JWT payload
            required_permission: Permission to check
            
        Returns:
            True if permission granted, False otherwise
        """
        permissions = token_payload.get("permissions", [])
        return required_permission.value in permissions

    def get_rate_limit_config(self, token_payload: Dict[str, Any]) -> Dict[str, int]:
        """Get rate limiting configuration for token.
        
        Args:
            token_payload: Validated JWT payload
            
        Returns:
            Dict with rate_limit_per_minute and burst_capacity
        """
        tier = token_payload.get("rate_limit_tier", RateLimitTier.BASIC.value)
        
        configs = {
            RateLimitTier.BASIC.value: {
                "rate_limit_per_minute": 60,
                "burst_capacity": 90
            },
            RateLimitTier.PROFESSIONAL.value: {
                "rate_limit_per_minute": 300,
                "burst_capacity": 450
            },
            RateLimitTier.ENTERPRISE.value: {
                "rate_limit_per_minute": 1000,
                "burst_capacity": 1500
            }
        }
        
        return configs.get(tier, configs[RateLimitTier.BASIC.value])

    def create_test_tokens(self) -> Dict[str, str]:
        """Create test tokens for all agent types (for testing only).
        
        Returns:
            Dict mapping agent types to test tokens
        """
        tokens = {}
        for agent_type in AgentType:
            tokens[agent_type.value] = self.create_agent_token(
                agent_id=f"test-{agent_type.value}-agent",
                agent_type=agent_type
            )
        return tokens
