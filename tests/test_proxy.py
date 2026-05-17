"""
Proxy unit tests.

Tests cover:
  - tools/list: correct attribute-filtered tool set per agent role
  - tools/list: unauthorized attribute returns 403
  - tools/call: unauthorized tool blocked with 403
  - tools/call: missing/invalid JWT returns 401
  - tools/call: unknown tool returns 404
  - tools/call: authorized call forwarded to backend
  - ABAC: defense-in-depth layer strips tools even if MongoDB leaks them
  - ABAC: admin_agent sees all attributes
  - Timing headers present on tools/list response

Run:
  pytest tests/test_proxy.py -v

Requires:
  - MongoDB running at localhost:27017 with governed_mcp.tools seeded
    (or use the mock_col fixture which patches Motor)
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

from proxy.main import app
from proxy.auth import sign_token

# ── Representative tool documents ────────────────────────────────────────────

_DEV_TOOL = {
    "name": "github_create_pull_request",
    "description": "Opens a pull request on GitHub.",
    "attributes": ["developer"],
    "inputSchema": {"type": "object", "properties": {}, "required": []},
}

_PAYMENTS_TOOL = {
    "name": "stripe_create_charge",
    "description": "Creates a Stripe charge.",
    "attributes": ["payments"],
    "inputSchema": {"type": "object", "properties": {}, "required": []},
}

_IDENTITY_TOOL = {
    "name": "auth0_revoke_token",
    "description": "Revokes an Auth0 access token.",
    "attributes": ["identity"],
    "inputSchema": {"type": "object", "properties": {}, "required": []},
}

_CRM_TOOL = {
    "name": "hubspot_create_contact",
    "description": "Creates a HubSpot contact.",
    "attributes": ["crm"],
    "inputSchema": {"type": "object", "properties": {}, "required": []},
}


def _make_mock_col(find_result=None, find_one_result=None):
    """Build a mock Motor collection.

    Motor's find() is synchronous (returns a cursor); to_list() on the cursor is async.
    find_one() is async.
    """
    col = MagicMock()
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=find_result or [])
    col.find = MagicMock(return_value=cursor)
    col.find_one = AsyncMock(return_value=find_one_result)
    return col


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── Auth tests ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_authorization_header(client):
    resp = await client.get("/mcp/tools/list")
    assert resp.status_code == 422  # FastAPI: required Header missing


@pytest.mark.asyncio
async def test_malformed_token(client):
    resp = await client.get(
        "/mcp/tools/list",
        headers={"Authorization": "Bearer not-a-valid-jwt"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_bearer_prefix_required(client):
    token = sign_token("agent_1", "dev_agent")
    resp = await client.get(
        "/mcp/tools/list",
        headers={"Authorization": token},  # missing "Bearer " prefix
    )
    assert resp.status_code == 401


# ── tools/list: happy path ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dev_agent_sees_developer_tools(client):
    """dev_agent receives only developer-attribute tools."""
    token = sign_token("agent_dev", "dev_agent")
    mock_col = _make_mock_col(find_result=[_DEV_TOOL])

    with patch.object(app.state, "col", mock_col, create=True):
        resp = await client.get(
            "/mcp/tools/list",
            params={"attribute": "developer"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["tools"][0]["name"] == "github_create_pull_request"


@pytest.mark.asyncio
async def test_payments_agent_sees_payments_tools(client):
    token = sign_token("agent_pay", "payments_agent")
    mock_col = _make_mock_col(find_result=[_PAYMENTS_TOOL])

    with patch.object(app.state, "col", mock_col, create=True):
        resp = await client.get(
            "/mcp/tools/list",
            params={"attribute": "payments"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    assert resp.json()["count"] == 1


# ── tools/list: 403 on unauthorized attribute ─────────────────────────────────

@pytest.mark.asyncio
async def test_dev_agent_denied_payments_attribute(client):
    """dev_agent requesting ?attribute=payments must receive 403."""
    token = sign_token("agent_dev", "dev_agent")

    with patch.object(app.state, "col", _make_mock_col(), create=True):
        resp = await client.get(
            "/mcp/tools/list",
            params={"attribute": "payments"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["error"] == "attribute_not_authorized"
    assert "payments" in detail["denied_attributes"]


@pytest.mark.asyncio
async def test_payments_agent_denied_developer_attribute(client):
    token = sign_token("agent_pay", "payments_agent")

    with patch.object(app.state, "col", _make_mock_col(), create=True):
        resp = await client.get(
            "/mcp/tools/list",
            params={"attribute": "developer"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 403


# ── ABAC defense-in-depth (unit tests, no HTTP) ───────────────────────────────

def test_dev_agent_cannot_access_payments_tool():
    """Even if MongoDB leaks a payments tool, ABAC strips it."""
    from proxy.abac import is_tool_authorized
    assert not is_tool_authorized("dev_agent", _PAYMENTS_TOOL)


def test_payments_agent_cannot_access_developer_tool():
    from proxy.abac import is_tool_authorized
    assert not is_tool_authorized("payments_agent", _DEV_TOOL)


def test_hr_agent_allowed_identity_and_crm():
    """hr_agent may access identity and crm, but not developer or payments."""
    from proxy.abac import is_tool_authorized
    assert is_tool_authorized("hr_agent", _IDENTITY_TOOL)
    assert is_tool_authorized("hr_agent", _CRM_TOOL)
    assert not is_tool_authorized("hr_agent", _DEV_TOOL)
    assert not is_tool_authorized("hr_agent", _PAYMENTS_TOOL)


def test_admin_agent_sees_all():
    """admin_agent is authorized for every attribute."""
    from proxy.abac import is_tool_authorized
    for tool in [_DEV_TOOL, _PAYMENTS_TOOL, _IDENTITY_TOOL, _CRM_TOOL]:
        assert is_tool_authorized("admin_agent", tool)


def test_unknown_role_returns_empty_attributes():
    from proxy.abac import allowed_attributes_for_role
    assert allowed_attributes_for_role("ghost_agent") == []


# ── tools/call: authorization ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_blocked_returns_403(client):
    """dev_agent calling a payments tool must receive 403."""
    token = sign_token("agent_dev", "dev_agent")
    mock_col = _make_mock_col(find_one_result=_PAYMENTS_TOOL)

    with patch.object(app.state, "col", mock_col, create=True):
        resp = await client.post(
            "/mcp/tools/call",
            json={"tool_name": "stripe_create_charge", "arguments": {}},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["error"] == "tool_not_authorized"
    assert detail["tool_name"] == "stripe_create_charge"
    assert detail["role"] == "dev_agent"


@pytest.mark.asyncio
async def test_call_blocked_403_includes_role(client):
    token = sign_token("agent_pay", "payments_agent")
    mock_col = _make_mock_col(find_one_result=_DEV_TOOL)

    with patch.object(app.state, "col", mock_col, create=True):
        resp = await client.post(
            "/mcp/tools/call",
            json={"tool_name": "github_create_pull_request", "arguments": {}},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 403
    assert resp.json()["detail"]["role"] == "payments_agent"


@pytest.mark.asyncio
async def test_call_unknown_tool_returns_404(client):
    token = sign_token("agent_dev", "dev_agent")
    mock_col = _make_mock_col(find_one_result=None)

    with patch.object(app.state, "col", mock_col, create=True):
        resp = await client.post(
            "/mcp/tools/call",
            json={"tool_name": "nonexistent_tool_xyz", "arguments": {}},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_call_authorized_forwards_to_backend(client):
    """Authorized call is forwarded to the mock tool server."""
    token = sign_token("agent_dev", "dev_agent")
    mock_col = _make_mock_col(find_one_result=_DEV_TOOL)

    mock_response = AsyncMock()
    mock_response.json = lambda: {"success": True, "tool_name": "github_create_pull_request"}
    mock_response.status_code = 200

    with patch.object(app.state, "col", mock_col, create=True):
        with patch("proxy.main.httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=None)
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_http

            resp = await client.post(
                "/mcp/tools/call",
                json={"tool_name": "github_create_pull_request", "arguments": {}},
                headers={"Authorization": f"Bearer {token}"},
            )

    assert resp.status_code == 200
    assert resp.json()["success"] is True


# ── Timing headers ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timing_headers_present(client):
    """tools/list response must include all X-Timing-* headers."""
    token = sign_token("agent_dev", "dev_agent")
    mock_col = _make_mock_col(find_result=[])

    with patch.object(app.state, "col", mock_col, create=True):
        resp = await client.get(
            "/mcp/tools/list",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    for header in [
        "X-Timing-JWT-ms", "X-Timing-AttrAuth-ms", "X-Timing-Query-ms",
        "X-Timing-ABAC-ms", "X-Timing-Total-ms",
    ]:
        assert header in resp.headers, f"Missing timing header: {header}"
        assert float(resp.headers[header]) >= 0
