"""
Mock Tool Execution Server (localhost:8001).

Simulates the downstream tool execution layer.
Every invocation is logged so Experiment 4 can verify
that unauthorized tools never receive a 200 response.

Run:
  uvicorn mock_server.main:app --host 0.0.0.0 --port 8001 --reload
"""

import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mock_server")

# Structured invocation log — parsed by benchmark/log_parser.py (T-29)
invocation_log = logging.getLogger("mock_server.invocations")
_handler = logging.FileHandler("mock_server_invocations.jsonl")
_handler.setFormatter(logging.Formatter("%(message)s"))
invocation_log.addHandler(_handler)
invocation_log.propagate = False

app = FastAPI(title="Mock Tool Server", version="1.0.0")


class ExecuteRequest(BaseModel):
    tool_name: str
    arguments: dict = {}


@app.post("/tools/execute")
async def execute_tool(body: ExecuteRequest, request: Request) -> JSONResponse:
    """
    Simulate tool execution and log every invocation.
    Returns a deterministic mock result based on tool name.
    """
    agent_id = request.headers.get("X-Agent-Id", "unknown")
    agent_role = request.headers.get("X-Agent-Role", "unknown")
    invocation_id = str(uuid.uuid4())

    record = {
        "ts": time.time(),
        "invocation_id": invocation_id,
        "agent_id": agent_id,
        "agent_role": agent_role,
        "tool_name": body.tool_name,
        "status": "executed",
    }
    invocation_log.info(json.dumps(record))
    log.info("EXECUTE agent=%s role=%s tool=%s", agent_id, agent_role, body.tool_name)

    # Return a mock result — content is irrelevant for TSA/compliance experiments
    return JSONResponse({
        "invocation_id": invocation_id,
        "tool_name": body.tool_name,
        "success": True,
        "result": {
            "mock": True,
            "message": f"Mock execution of '{body.tool_name}' succeeded.",
            "arguments_received": body.arguments,
        },
    })


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
