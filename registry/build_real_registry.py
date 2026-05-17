"""
Build real API tool registry from compact definitions in tools_data.py.

Run:
    python registry/build_real_registry.py
    python registry/build_real_registry.py --seed-mongo
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .tools_data import ATTRIBUTES, PARAMS_OVERRIDE, TOOLS_BY_SERVICE

# ── Display names ──────────────────────────────────────────────────────────────

SERVICE_NAMES = {
    "stripe": "Stripe", "paypal": "PayPal", "github": "GitHub",
    "jira": "Jira", "slack": "Slack", "zendesk": "Zendesk",
    "auth0": "Auth0", "twilio": "Twilio", "gdrive": "Google Drive",
    "sendgrid": "SendGrid", "notion": "Notion", "hubspot": "HubSpot",
    "mixpanel": "Mixpanel", "airtable": "Airtable", "s3": "AWS S3",
    "amplitude": "Amplitude", "docusign": "DocuSign",
}

# ── Default params by leading verb ────────────────────────────────────────────

_ID = [("id", "string", True)]
_LIST = [("limit", "integer", False), ("offset", "integer", False), ("filter", "string", False)]

VERB_PARAMS: dict[str, list] = {
    "create":   [("name", "string", True), ("description", "string", False)],
    "get":      _ID,
    "list":     _LIST,
    "update":   [("id", "string", True), ("name", "string", False), ("description", "string", False)],
    "delete":   _ID,
    "search":   [("query", "string", True), ("limit", "integer", False)],
    "send":     _ID,
    "cancel":   _ID,
    "capture":  _ID,
    "confirm":  _ID,
    "archive":  _ID,
    "assign":   [("id", "string", True), ("assignee", "string", True)],
    "merge":    [("id", "string", True), ("source_id", "string", True)],
    "trigger":  [("id", "string", True), ("ref", "string", True)],
    "post":     [("channel", "string", True), ("text", "string", True)],
    "upload":   [("filename", "string", True), ("content", "string", False)],
    "download": _ID,
    "export":   [("start_date", "string", False), ("end_date", "string", False)],
    "import":   [("data", "array", True)],
    "track":    [("event", "string", True), ("properties", "object", False)],
    "finalize": _ID,
    "void":     _ID,
    "pay":      _ID,
    "activate": _ID,
    "deactivate": _ID,
    "start":    _ID,
    "complete": _ID,
    "close":    _ID,
    "open":     _ID,
    "transition": [("id", "string", True), ("status", "string", True)],
    "enable":   _ID,
    "disable":  _ID,
    "block":    _ID,
    "unblock":  _ID,
    "suspend":  _ID,
    "resume":   _ID,
    "release":  _ID,
    "expire":   _ID,
    "protect":  _ID,
    "add":      [("id", "string", True), ("member", "string", True)],
    "remove":   [("id", "string", True), ("member", "string", True)],
    "invite":   [("id", "string", True), ("user", "string", True)],
    "share":    [("id", "string", True), ("email", "string", True)],
    "copy":     [("id", "string", True), ("destination", "string", False)],
    "move":     [("id", "string", True), ("destination", "string", True)],
    "rename":   [("id", "string", True), ("new_name", "string", True)],
    "fork":     [("owner", "string", True), ("repo", "string", True)],
    "transfer":  [("id", "string", True), ("new_owner", "string", True)],
    "pin":      _ID,
    "unpin":    _ID,
    "star":     _ID,
    "apply":    [("id", "string", True), ("target_id", "string", True)],
    "query":    [("filter", "object", False), ("sort", "string", False)],
    "append":   [("id", "string", True), ("content", "array", True)],
    "reply":    [("id", "string", True), ("text", "string", True)],
    "set":      [("id", "string", True), ("value", "string", True)],
    "put":      [("id", "string", True), ("body", "string", True)],
    "log":      [("message", "string", True), ("metadata", "object", False)],
    "empty":    [],
    "buy":      [("item", "string", True)],
    "make":     [("to", "string", True), ("from_", "string", True)],
    "show":     _ID,
    "identify": [("user_id", "string", True), ("properties", "object", True)],
    "group":    [("group_id", "string", True), ("properties", "object", False)],
    "submit":   [("id", "string", True), ("event", "string", True)],
    "solve":    _ID,
    "pause":    _ID,
    "modify":   _ID,
    "announce": [("text", "string", True)],
}


# ── Description generator ─────────────────────────────────────────────────────

_VERB_PHRASES = {
    "create": "Creates", "get": "Retrieves", "list": "Lists",
    "update": "Updates", "delete": "Permanently deletes", "search": "Searches",
    "send": "Sends", "cancel": "Cancels", "capture": "Captures",
    "confirm": "Confirms", "archive": "Archives", "assign": "Assigns",
    "merge": "Merges", "trigger": "Triggers", "post": "Posts",
    "upload": "Uploads", "download": "Downloads", "export": "Exports",
    "import": "Imports", "track": "Tracks", "finalize": "Finalizes",
    "void": "Voids", "pay": "Pays", "activate": "Activates",
    "deactivate": "Deactivates", "start": "Starts", "complete": "Completes",
    "close": "Closes", "open": "Opens", "transition": "Transitions",
    "enable": "Enables", "disable": "Disables", "block": "Blocks",
    "unblock": "Unblocks", "suspend": "Suspends", "resume": "Resumes",
    "release": "Releases", "expire": "Expires", "protect": "Protects",
    "add": "Adds", "remove": "Removes", "invite": "Invites",
    "share": "Shares", "copy": "Copies", "move": "Moves", "rename": "Renames",
    "fork": "Forks", "transfer": "Transfers", "pin": "Pins", "unpin": "Unpins",
    "star": "Stars", "apply": "Applies", "query": "Queries", "append": "Appends",
    "reply": "Replies to", "set": "Sets", "put": "Puts", "log": "Logs",
    "empty": "Empties", "buy": "Purchases", "make": "Makes", "show": "Shows",
    "identify": "Identifies", "group": "Groups", "submit": "Submits",
    "solve": "Marks as solved", "pause": "Pauses", "modify": "Modifies",
    "compare": "Compares",
}


def _make_description(action: str, service: str) -> str:
    svc = SERVICE_NAMES.get(service, service.title())
    parts = action.split("_")
    verb = parts[0]
    noun = " ".join(parts[1:]).replace("_", " ") if len(parts) > 1 else "resource"
    phrase = _VERB_PHRASES.get(verb, verb.title() + "s")
    return f"{phrase} a {noun} via the {svc} API."


# ── Schema builder ────────────────────────────────────────────────────────────

def _build_input_schema(tool_name: str, action: str) -> dict:
    raw_params = PARAMS_OVERRIDE.get(tool_name) or VERB_PARAMS.get(action.split("_")[0], _ID)
    properties: dict = {}
    required: list = []
    for (pname, ptype, preq) in raw_params:
        properties[pname] = {"type": ptype, "description": pname.replace("_", " ").capitalize()}
        if preq:
            required.append(pname)
    return {"type": "object", "properties": properties, "required": required}


# ── Main generator ────────────────────────────────────────────────────────────

def generate() -> list[dict]:
    tools: list[dict] = []
    for attribute, services in TOOLS_BY_SERVICE.items():
        for service, actions in services.items():
            svc_display = SERVICE_NAMES.get(service, service)
            for action in actions:
                name = f"{service}_{action}"
                tools.append({
                    "name": name,
                    "description": _make_description(action, service),
                    "transport": "http",
                    "service": service,
                    "service_display": svc_display,
                    "attributes": [attribute],
                    "inputSchema": _build_input_schema(name, action),
                })
    return tools


# ── Output ────────────────────────────────────────────────────────────────────

def build(output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tools = generate()

    (output_dir / "real_tools.json").write_text(json.dumps(tools, indent=2))
    (output_dir / "attributes.json").write_text(json.dumps(ATTRIBUTES, indent=2))

    by_service = Counter(t["service"] for t in tools)
    by_attr = Counter(t["attributes"][0] for t in tools)

    print(f"Generated {len(tools)} tools")
    print("\nBy service:")
    for svc, n in sorted(by_service.items(), key=lambda x: -x[1]):
        print(f"  {svc:20s} {n:3d}")
    print("\nBy attribute:")
    for attr, n in sorted(by_attr.items(), key=lambda x: -x[1]):
        print(f"  {attr:20s} {n:3d}")
    return tools


def seed_mongo(tools: list[dict]) -> None:
    from pymongo import MongoClient, ASCENDING
    client = MongoClient("mongodb://localhost:27017")
    db = client["governed_mcp"]

    col = db["tools"]
    col.drop()
    col.insert_many(tools)
    col.create_index([("attributes", ASCENDING), ("service", ASCENDING)], name="attributes_service_idx")
    print(f"\nSeeded {col.count_documents({})} tools → governed_mcp.tools")

    acol = db["attributes"]
    acol.drop()
    acol.insert_many([{**a} for a in ATTRIBUTES])
    print(f"Seeded {acol.count_documents({})} attributes → governed_mcp.attributes")
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="registry")
    parser.add_argument("--seed-mongo", action="store_true")
    args = parser.parse_args()

    tools = build(Path(args.output_dir))
    if args.seed_mongo:
        seed_mongo(tools)
