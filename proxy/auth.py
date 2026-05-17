"""
T-08: JWT identity layer.

Provides JWT signing and verification utilities.
Agent identity is carried via Bearer token in Authorization header.

Token claims:
  sub   — agent identifier (e.g. "agent_123")
  role  — one of: dev_agent, staging_agent, prod_agent, admin_agent, hr_agent
  exp   — expiry timestamp
  iat   — issued-at timestamp
"""

import time
from typing import Optional
import jwt
from fastapi import HTTPException, status

# In production this would be loaded from a secrets manager.
# For the paper's local stack we use a fixed symmetric secret.
JWT_SECRET = "governed-mcp-paper-secret-do-not-use-in-prod"
JWT_ALGORITHM = "HS256"
JWT_TTL_SECONDS = 3600

VALID_ROLES = {
    "payments_agent", "dev_agent", "support_agent", "data_agent", "content_agent",
    "identity_agent", "hr_agent", "ops_agent", "media_agent", "wellness_agent",
    "admin_agent",
}


def sign_token(agent_id: str, role: str, ttl: int = JWT_TTL_SECONDS) -> str:
    """Sign a JWT for an agent with a given role."""
    if role not in VALID_ROLES:
        raise ValueError(f"Unknown role: {role}. Must be one of {VALID_ROLES}")
    now = int(time.time())
    payload = {
        "sub": agent_id,
        "role": role,
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(authorization: Optional[str]) -> dict:
    """
    Decode and verify a Bearer JWT from the Authorization header.

    Returns the decoded claims dict on success.
    Raises HTTPException 401 on missing/invalid/expired token.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header (expected: Bearer <token>)",
        )

    token = authorization.removeprefix("Bearer ").strip()

    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )

    role = claims.get("role")
    if role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token contains unknown role: {role}",
        )

    return claims
