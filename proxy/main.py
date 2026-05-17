"""
Governed MCP Proxy — FastAPI application.

Endpoints:
  GET /mcp/tools/list?attribute=payments[&attribute=developer]
      → attribute-filtered tool discovery (returns MCP tools/list payload)
  POST /mcp/tools/call
      → authorized tool invocation with RPC enforcement

Identity: JWT Bearer token in Authorization header.
Attribute filtering: ABAC policy from policy.yaml.
Storage: MongoDB (governed_mcp.tools collection).
Timing: per-stage perf_counter timestamps in X-Timing-* response headers.

Run:
    uvicorn proxy.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

from .abac import allowed_attributes_for_role, build_mongo_query, is_attribute_allowed, is_tool_authorized
from .auth import verify_token

# ── Configuration ─────────────────────────────────────────────────────────────

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "governed_mcp"
MONGO_COLLECTION = "tools"
TOOL_SERVER_URL = "http://localhost:8001"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("governed_mcp.proxy")

audit_log = logging.getLogger("governed_mcp.audit")
_audit_handler = logging.FileHandler("proxy_audit.jsonl")
_audit_handler.setFormatter(logging.Formatter("%(message)s"))
audit_log.addHandler(_audit_handler)
audit_log.propagate = False


# ── Application lifecycle ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mongo = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    app.state.col = app.state.mongo[MONGO_DB][MONGO_COLLECTION]
    log.info("Connected to MongoDB at %s", MONGO_URI)
    yield
    app.state.mongo.close()


app = FastAPI(
    title="Governed MCP Proxy",
    version="2.0.0",
    description="Semantic-attribute-filtered, ABAC-enforced MCP tool discovery proxy.",
    lifespan=lifespan,
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tool_to_mcp(doc: dict) -> dict:
    return {
        "name": doc["name"],
        "description": doc["description"],
        "inputSchema": doc.get("inputSchema", {}),
        "attributes": doc.get("attributes", []),
        "transport": doc.get("transport", "http"),
        "service": doc.get("service", ""),
    }


def _audit(agent_id: str, role: str, tool_name: str, outcome: str, latency_ms: float) -> None:
    record = {
        "ts": time.time(),
        "agent_id": agent_id,
        "role": role,
        "tool_name": tool_name,
        "outcome": outcome,  # ALLOWED | BLOCKED | NOT_FOUND | ATTR_DENIED
        "latency_ms": round(latency_ms, 3),
    }
    audit_log.info(json.dumps(record))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/mcp/tools/list")
async def list_tools(
    request: Request,
    attribute: List[str] = Query(default=[]),
    authorization: str = Header(...),
) -> JSONResponse:
    """
    Attribute-filtered tool discovery.

    Clients pass one or more ?attribute= query params:
        GET /mcp/tools/list?attribute=payments
        GET /mcp/tools/list?attribute=payments&attribute=developer

    If ANY requested attribute is outside the agent's allowed set → 403.
    If no attribute is specified → return all tools the role may access.

    Stage timings (X-Timing-* response headers, all in milliseconds):
      t0→t1  JWT verification
      t1→t2  Attribute authorization check
      t2→t3  MongoDB query
      t3→t4  ABAC defense-in-depth filter
      t4→t5  Response serialization
    """
    t0 = time.perf_counter()

    # Stage 1: JWT
    claims = verify_token(authorization)
    role = claims["role"]
    agent_id = claims["sub"]
    t1 = time.perf_counter()

    # Stage 2: Attribute authorization — hard 403 if any attribute is denied
    if attribute:
        denied = [a for a in attribute if not is_attribute_allowed(role, a)]
        if denied:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "attribute_not_authorized",
                    "message": (
                        f"Role '{role}' is not permitted to access attribute(s): {denied}. "
                        f"Allowed: {allowed_attributes_for_role(role)}"
                    ),
                    "denied_attributes": denied,
                    "role": role,
                },
            )
    t2 = time.perf_counter()

    # Stage 3: MongoDB query
    query = build_mongo_query(role, attribute if attribute else None)
    col = request.app.state.col
    raw_tools = await col.find(query, {"_id": 0}).to_list(length=None)
    t3 = time.perf_counter()

    # Stage 4: ABAC defense-in-depth
    authorized = [t for t in raw_tools if is_tool_authorized(role, t)]
    t4 = time.perf_counter()

    # Stage 5: Serialize
    payload = [_tool_to_mcp(t) for t in authorized]
    t5 = time.perf_counter()

    log.info(
        "tools/list agent=%s role=%s attributes=%s returned=%d",
        agent_id, role, attribute or "all", len(payload),
    )

    return JSONResponse(
        content={"tools": payload, "count": len(payload)},
        headers={
            "X-Timing-JWT-ms":    f"{(t1 - t0) * 1000:.3f}",
            "X-Timing-AttrAuth-ms": f"{(t2 - t1) * 1000:.3f}",
            "X-Timing-Query-ms":  f"{(t3 - t2) * 1000:.3f}",
            "X-Timing-ABAC-ms":   f"{(t4 - t3) * 1000:.3f}",
            "X-Timing-Serial-ms": f"{(t5 - t4) * 1000:.3f}",
            "X-Timing-Total-ms":  f"{(t5 - t0) * 1000:.3f}",
        },
    )


@app.post("/mcp/tools/call")
async def call_tool(
    request_body: ToolCallRequest,
    request: Request,
    authorization: str = Header(...),
) -> JSONResponse:
    """
    Authorized tool invocation at the RPC layer.

    Flow:
      1. Verify JWT
      2. Look up tool in MongoDB
      3. ABAC check: does the tool's attribute overlap with the role's allowed set?
      4. On deny → 403 + audit record (BLOCKED)
      5. On allow → route to backend by transport type + audit record (ALLOWED)

    PBR (Proxy Block Rate) must be 100%: unauthorized tools never reach
    the backend. This is a correctness property.
    """
    t_start = time.perf_counter()

    # Stage 1: JWT
    claims = verify_token(authorization)
    role = claims["role"]
    agent_id = claims["sub"]

    # Stage 2: Tool lookup
    col = request.app.state.col
    tool_doc = await col.find_one({"name": request_body.tool_name}, {"_id": 0})

    if tool_doc is None:
        latency = (time.perf_counter() - t_start) * 1000
        _audit(agent_id, role, request_body.tool_name, "NOT_FOUND", latency)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{request_body.tool_name}' not found in registry",
        )

    # Stage 3: ABAC authorization
    authorized = is_tool_authorized(role, tool_doc)
    latency = (time.perf_counter() - t_start) * 1000

    if not authorized:
        _audit(agent_id, role, request_body.tool_name, "BLOCKED", latency)
        log.warning(
            "BLOCKED agent=%s role=%s tool=%s attributes=%s",
            agent_id, role, request_body.tool_name, tool_doc.get("attributes"),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "tool_not_authorized",
                "message": (
                    f"Role '{role}' is not permitted to invoke '{request_body.tool_name}' "
                    f"(attributes={tool_doc.get('attributes')})"
                ),
                "role": role,
                "tool_name": request_body.tool_name,
                "tool_attributes": tool_doc.get("attributes"),
            },
        )

    # Stage 4: Route by transport
    _audit(agent_id, role, request_body.tool_name, "ALLOWED", latency)
    log.info("ALLOWED agent=%s role=%s tool=%s", agent_id, role, request_body.tool_name)

    transport = tool_doc.get("transport", "http")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{TOOL_SERVER_URL}/tools/execute",
                json={
                    "tool_name": request_body.tool_name,
                    "arguments": request_body.arguments,
                    "transport": transport,
                },
                headers={"X-Agent-Id": agent_id, "X-Agent-Role": role},
            )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tool backend unavailable",
        )


# ── Health & metadata ─────────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request) -> dict:
    try:
        await request.app.state.mongo.admin.command("ping")
        mongo_ok = True
    except Exception:
        mongo_ok = False
    return {"status": "ok" if mongo_ok else "degraded", "mongo": mongo_ok}


@app.get("/mcp/attributes")
async def list_attributes(
    request: Request,
    authorization: str = Header(...),
) -> JSONResponse:
    """Returns the full attribute catalogue and which ones this agent may access."""
    claims = verify_token(authorization)
    role = claims["role"]
    allowed = allowed_attributes_for_role(role)

    col = request.app.state.mongo[MONGO_DB]["attributes"]
    attrs = await col.find({}, {"_id": 0}).to_list(length=None)

    for a in attrs:
        a["allowed"] = a["name"] in allowed

    return JSONResponse(content={"attributes": attrs, "role": role, "allowed": allowed})
