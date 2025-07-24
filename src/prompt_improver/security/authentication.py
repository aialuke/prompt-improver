"""Real authentication service implementation for production use and integration testing."""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt

from prompt_improver.utils.datetime_utils import aware_utc_now

class AuthenticationService:
    """Real authentication service that implements secure authentication flows."""

    def __init__(self, secret_key: str | None = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.users: dict[str, dict[str, Any]] = {}
        self.token_expiry_hours = 24

    def hash_password(self, password: str, salt: str | None = None) -> str:
        """Hash password using PBKDF2 with SHA-256."""
        if salt is None:
            salt = secrets.token_hex(16)

        # Use PBKDF2 with 100,000 iterations (2025 security standard)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}:{password_hash.hex()}"

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            salt, hash_value = password_hash.split(':')
            return hmac.compare_digest(
                self.hash_password(password, salt),
                password_hash
            )
        except ValueError:
            return False

    def register_user(self, username: str, password: str, email: str, roles: list = None) -> dict[str, Any]:
        """Register a new user with secure password hashing."""
        if username in self.users:
            raise ValueError(f"User {username} already exists")

        user_data = {
            "user_id": f"user_{secrets.token_hex(8)}",
            "username": username,
            "email": email,
            "password_hash": self.hash_password(password),
            "roles": roles or ["user"],
            "created_at": aware_utc_now(),
            "last_login": None,
            "failed_login_attempts": 0,
            "account_locked": False,
        }

        self.users[username] = user_data
        return {k: v for k, v in user_data.items() if k != "password_hash"}

    def authenticate_user(self, username: str, password: str) -> dict[str, Any] | None:
        """Authenticate user with password verification and account lockout protection."""
        if username not in self.users:
            return None

        user = self.users[username]

        # Check account lockout (after 5 failed attempts)
        if user.get("account_locked", False):
            return None

        # Verify password
        if not self.verify_password(password, user["password_hash"]):
            # Increment failed login attempts
            user["failed_login_attempts"] = user.get("failed_login_attempts", 0) + 1
            if user["failed_login_attempts"] >= 5:
                user["account_locked"] = True
            return None

        # Reset failed attempts on successful login
        user["failed_login_attempts"] = 0
        user["last_login"] = aware_utc_now()

        return {k: v for k, v in user.items() if k != "password_hash"}

    def create_jwt_token(self, user_data: dict[str, Any]) -> str:
        """Create JWT token with secure claims and expiration."""
        payload = {
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "email": user_data["email"],
            "roles": user_data["roles"],
            "iat": int(time.time()),
            "exp": int(time.time() + (self.token_expiry_hours * 3600)),
            "iss": "prompt_improver_auth",
        }

        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def validate_jwt_token(self, token: str) -> dict[str, Any] | None:
        """Validate JWT token and return payload if valid."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"],
                options={"verify_exp": True, "verify_iss": True},
                issuer="prompt_improver_auth"
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def revoke_token(self, token: str) -> bool:
        """Revoke a specific token (simplified implementation using blacklist)."""
        # In production, this would use a token blacklist stored in Redis/database
        # For testing, we'll use a simple in-memory set
        if not hasattr(self, '_revoked_tokens'):
            self._revoked_tokens = set()

        self._revoked_tokens.add(token)
        return True

    def is_token_revoked(self, token: str) -> bool:
        """Check if token has been revoked."""
        if not hasattr(self, '_revoked_tokens'):
            return False
        return token in self._revoked_tokens

    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Get user by user ID."""
        for user in self.users.values():
            if user["user_id"] == user_id:
                return {k: v for k, v in user.items() if k != "password_hash"}
        return None