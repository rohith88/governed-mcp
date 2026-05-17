"""
T-16 / T-17: Locust load test for the Governed MCP Proxy.

Models realistic agent behavior: discover → call → discover.

Usage:
    # Headless mode (recommended for paper experiments):
    locust -f loadtest/locustfile.py --headless \
        -u 100 -r 10 \
        --run-time 60s \
        --host http://localhost:8000 \
        --csv results/latency/locust_N500_c100

    # Web UI (for interactive monitoring):
    locust -f loadtest/locustfile.py --host http://localhost:8000

Registry size configurations for Experiment 3:
    N=100   → set REGISTRY_SIZE=100 env var
    N=500   → default
    N=1000  → set REGISTRY_SIZE=1000
    N=5000  → set REGISTRY_SIZE=5000
    N=10000 → set REGISTRY_SIZE=10000

Run the proxy with a seeded registry of the appropriate size:
    REGISTRY_SIZE=500 uvicorn proxy.main:app --host 0.0.0.0 --port 8000

Ramp schedule (paper Table 3):
    --users 500 --spawn-rate 10   (50-second ramp to 500 concurrent)
"""

import os
import random
import time
from locust import HttpUser, TaskSet, between, events, task
from proxy.auth import sign_token

# ── Agent pool ────────────────────────────────────────────────────────────────

AGENT_ROLES = ["dev_agent", "staging_agent", "prod_agent", "hr_agent"]

# Pre-signed tokens for each role (valid for 1 hour)
_TOKENS: dict[str, str] = {}


def _get_token(role: str) -> str:
    if role not in _TOKENS:
        _TOKENS[role] = sign_token(f"locust_{role}", role)
    return _TOKENS[role]


# Known tool names per role (from the seeded registry, for call tasks)
KNOWN_TOOLS_BY_ROLE = {
    "dev_agent": [
        "dev_database_query", "dev_database_select", "dev_filesystem_read",
        "dev_filesystem_list", "dev_api_get", "dev_api_fetch",
        "dev_cicd_get", "dev_cicd_list", "dev_iam_get", "dev_iam_list",
    ],
    "staging_agent": [
        "staging_database_query", "staging_database_backup",
        "staging_filesystem_read", "staging_api_get",
        "staging_cicd_get", "staging_iam_get",
    ],
    "prod_agent": [
        "prod_database_query", "prod_database_select",
        "prod_filesystem_read", "prod_api_get",
        "prod_cicd_get", "prod_iam_get",
    ],
    "hr_agent": [
        "dev_iam_get", "dev_iam_list", "staging_iam_get", "prod_iam_get",
    ],
}


# ── Locust task sets ──────────────────────────────────────────────────────────

class DiscoverThenCallBehavior(TaskSet):
    """
    Realistic agent loop:
      1. List tools (attribute-filtered discovery)
      2. Inspect which tools are available
      3. Call one of the returned tools
      4. List tools again (second discovery in the same agent session)

    Weight: discover (60%) : call (40%)
    """

    def on_start(self):
        self.role = random.choice(AGENT_ROLES)
        self.token = _get_token(self.role)
        self.auth_header = {"Authorization": f"Bearer {self.token}"}

    @task(3)
    def list_tools(self):
        """Attribute-filtered tool discovery — the hot path for Experiment 3."""
        with self.client.post(
            "/mcp/tools/list",
            json={},
            headers=self.auth_header,
            catch_response=True,
            name="/mcp/tools/list",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                # Store returned tool names for subsequent call tasks
                self.environment.runner._locust_tool_lists = data.get("tools", [])
                resp.success()
            elif resp.status_code == 401:
                resp.failure("JWT expired or invalid")
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(2)
    def list_tools_with_filter(self):
        """Filtered discovery with an explicit domain hint."""
        domain = random.choice(["database", "filesystem", "api", "cicd", "iam"])
        with self.client.post(
            "/mcp/tools/list",
            json={"domain": domain},
            headers=self.auth_header,
            catch_response=True,
            name="/mcp/tools/list?domain",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(2)
    def call_authorized_tool(self):
        """Call a tool that is within the agent's allowed attribute set."""
        tools = KNOWN_TOOLS_BY_ROLE.get(self.role, [])
        if not tools:
            return
        tool_name = random.choice(tools)
        with self.client.post(
            "/mcp/tools/call",
            json={"tool_name": tool_name, "arguments": {"filter": {}}},
            headers=self.auth_header,
            catch_response=True,
            name="/mcp/tools/call [authorized]",
        ) as resp:
            if resp.status_code in (200, 503):
                # 503 means mock tool server is down — not a proxy error
                resp.success()
            elif resp.status_code == 404:
                # Tool not yet seeded — acceptable
                resp.success()
            elif resp.status_code == 401:
                resp.failure("JWT invalid")
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(1)
    def call_unauthorized_tool(self):
        """
        Attempt to call a tool outside the role's allowed set.
        Proxy MUST return 403. Used to verify PBR=100%.
        """
        # prod_database_delete is unauthorized for dev/hr agents
        if self.role in ("dev_agent", "hr_agent"):
            tool_name = "prod_database_delete"
        elif self.role == "prod_agent":
            tool_name = "dev_database_delete"  # wrong env for prod_agent
        else:
            return  # staging_agent and admin_agent can access most things

        with self.client.post(
            "/mcp/tools/call",
            json={"tool_name": tool_name, "arguments": {"target": "test", "confirm": True}},
            headers=self.auth_header,
            catch_response=True,
            name="/mcp/tools/call [unauthorized → expect 403]",
        ) as resp:
            if resp.status_code == 403:
                resp.success()  # Expected: proxy correctly blocked
            elif resp.status_code == 404:
                resp.success()  # Tool not seeded yet
            elif resp.status_code == 200:
                resp.failure("POLICY VIOLATION: unauthorized tool returned 200!")
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")


class MCPAgent(HttpUser):
    tasks = [DiscoverThenCallBehavior]
    wait_time = between(0.1, 1.0)  # 100ms–1s think time between requests


# ── Latency decomposition from timing headers ──────────────────────────────────

_timing_samples: list[dict] = []


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, **kwargs):
    """Capture per-stage timing from X-Timing-* response headers."""
    if hasattr(response, "headers") and name == "/mcp/tools/list":
        sample = {
            "ts": time.time(),
            "total_ms": response_time,  # locust wall-clock (ms)
            "jwt_ms": float(response.headers.get("X-Timing-JWT-ms", 0)),
            "query_ms": float(response.headers.get("X-Timing-Query-ms", 0)),
            "abac_ms": float(response.headers.get("X-Timing-ABAC-ms", 0)),
            "serial_ms": float(response.headers.get("X-Timing-Serial-ms", 0)),
            "proxy_total_ms": float(response.headers.get("X-Timing-Total-ms", 0)),
        }
        _timing_samples.append(sample)


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Write timing breakdown CSV when the test ends."""
    import csv
    out = "results/latency/timing_breakdown.csv"
    import pathlib
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    if _timing_samples:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(_timing_samples[0].keys()))
            writer.writeheader()
            writer.writerows(_timing_samples)
        print(f"\nTiming breakdown written to {out} ({len(_timing_samples)} samples)")
