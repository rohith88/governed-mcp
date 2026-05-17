"""
ABAC policy evaluator for semantic business attributes.

Roles map to allowed_attributes lists in policy.yaml.
An agent may only request tools tagged with attributes its role permits.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

POLICY_PATH = Path(__file__).parent / "policy.yaml"


@lru_cache(maxsize=1)
def _load_policy() -> dict:
    with POLICY_PATH.open() as f:
        return yaml.safe_load(f)


def allowed_attributes_for_role(role: str) -> list[str]:
    """Return the list of attributes a role is permitted to access."""
    policy = _load_policy()
    role_cfg = policy.get("roles", {}).get(role)
    if role_cfg is None:
        return []
    return role_cfg.get("allowed_attributes", [])


def is_attribute_allowed(role: str, attribute: str) -> bool:
    """Return True if the role may access tools tagged with `attribute`."""
    return attribute in allowed_attributes_for_role(role)


def build_mongo_query(role: str, requested_attributes: Optional[list[str]] = None) -> dict:
    """
    Build a MongoDB query that returns only tools the role may access.

    If `requested_attributes` is supplied (from ?attribute= params), the query
    is further restricted to those attributes — but they must still be within
    the role's allowed set (enforced upstream by is_attribute_allowed).
    """
    allowed = allowed_attributes_for_role(role)
    if not allowed:
        return {"attributes": {"$in": []}}

    if requested_attributes:
        effective = [a for a in requested_attributes if a in allowed]
    else:
        effective = allowed

    return {"attributes": {"$in": effective}}


def is_tool_authorized(role: str, tool_doc: dict) -> bool:
    """
    Defense-in-depth check: does the tool's attribute set overlap with
    the role's allowed attributes?
    """
    allowed = set(allowed_attributes_for_role(role))
    tool_attrs = set(tool_doc.get("attributes", []))
    return bool(allowed & tool_attrs)
