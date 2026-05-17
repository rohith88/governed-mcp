"""
T-02 / T-04: Generate 500+ synthetic tool schemas and seed MongoDB.

Usage:
    python seed_tools.py --generate      # Generate tools.json only
    python seed_tools.py --seed          # Seed MongoDB (requires running instance)
    python seed_tools.py --verify        # Spot-check queries
    python seed_tools.py --all           # Generate + seed + verify
"""

import json
import random
import argparse
import sys
from pathlib import Path
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ServerSelectionTimeoutError

# ── Taxonomy ──────────────────────────────────────────────────────────────────

ENVS = ["dev", "staging", "prod"]          # 'all-envs' reserved for cross-env tools
DOMAINS = ["database", "filesystem", "api", "cicd", "iam"]
OP_CLASSES = ["read", "write", "delete", "admin", "audit"]

RISK_BY_OP = {
    "read":   ["low-risk", "medium-risk"],
    "write":  ["low-risk", "medium-risk", "high-risk"],
    "delete": ["medium-risk", "high-risk", "destructive"],
    "admin":  ["high-risk", "destructive"],
    "audit":  ["low-risk"],
}

# Domain-specific operation verbs
DOMAIN_OPS = {
    "database": {
        "read":   ["query", "select", "fetch", "list", "inspect", "count", "describe"],
        "write":  ["insert", "update", "upsert", "backup", "snapshot", "import", "restore"],
        "delete": ["delete", "drop", "truncate", "purge", "archive"],
        "admin":  ["migrate", "reindex", "vacuum", "configure", "tune", "failover"],
        "audit":  ["audit", "log", "monitor", "trace"],
    },
    "filesystem": {
        "read":   ["read", "list", "stat", "checksum", "search", "diff"],
        "write":  ["write", "copy", "move", "upload", "compress", "sync"],
        "delete": ["delete", "remove", "clean", "purge", "wipe"],
        "admin":  ["mount", "chmod", "chown", "quota", "defrag"],
        "audit":  ["audit", "scan", "monitor", "log"],
    },
    "api": {
        "read":   ["get", "fetch", "list", "ping", "health", "inspect", "describe"],
        "write":  ["post", "put", "patch", "register", "configure", "update", "create"],
        "delete": ["delete", "revoke", "deregister", "remove"],
        "admin":  ["rotate", "throttle", "ratelimit", "gateway"],
        "audit":  ["audit", "log", "trace", "monitor"],
    },
    "cicd": {
        "read":   ["get", "list", "status", "inspect", "fetch", "describe"],
        "write":  ["trigger", "build", "deploy", "publish", "push", "run"],
        "delete": ["cancel", "rollback", "delete", "cleanup"],
        "admin":  ["configure", "schedule", "promote", "gate"],
        "audit":  ["audit", "log", "report", "monitor"],
    },
    "iam": {
        "read":   ["get", "list", "describe", "check", "validate", "inspect"],
        "write":  ["create", "update", "assign", "grant", "bind", "enable"],
        "delete": ["delete", "revoke", "disable", "remove", "unbind"],
        "admin":  ["rotate", "reset", "configure", "enforce", "audit-policy"],
        "audit":  ["audit", "log", "report", "monitor"],
    },
}

# Qualifiers add specificity to tool names (optional, ~40% of tools get one)
QUALIFIERS = {
    "database": ["full", "incremental", "schema", "data", "bulk", "table", "index", "schema-only"],
    "filesystem": ["recursive", "metadata", "large", "encrypted", "compressed", "logs", "temp"],
    "api": ["webhook", "rest", "graphql", "config", "key", "endpoint", "token", "batch"],
    "cicd": ["pipeline", "stage", "artifact", "container", "image", "branch", "tag", "release"],
    "iam": ["user", "role", "group", "policy", "permission", "credential", "cert", "token"],
}

# Description templates per (op_class, domain)
DESC_TEMPLATES = {
    ("read", "database"): [
        "Executes a read query against the {env} {domain} cluster and returns structured results. Supports parameterized queries and pagination. Suitable for inspection tasks requiring non-mutating access.",
        "Fetches records from the {env} {domain} without modifying underlying data. Returns paginated JSON results with field selection support. Safe for analytics and reporting workloads.",
        "Queries the {env} {domain} instance using specified filters. Returns matching rows with configurable projections. Operates in read-committed isolation mode.",
    ],
    ("write", "database"): [
        "Writes new records to the {env} {domain} with transaction support. Performs input validation and constraint checking before commit. Rolls back on partial failures.",
        "Inserts or updates records in the {env} {domain} using upsert semantics. Supports batch operations and returns affected row counts. Acquires row-level locks during execution.",
        "Creates a snapshot of the {env} {domain} and persists it to durable storage. Captures all tables and indexes by default. Reports completion status with checksum verification.",
    ],
    ("delete", "database"): [
        "Permanently removes records from the {env} {domain} matching the specified criteria. Operation is irreversible; requires explicit confirmation parameter. Acquires exclusive lock during execution.",
        "Drops the specified tables from the {env} {domain} including all dependent objects. Cascades to foreign key constraints. Cannot be rolled back once committed.",
        "Purges expired or archived records from the {env} {domain} according to retention policy. Runs in batches to minimize lock contention. Logs all deleted row IDs before removal.",
    ],
    ("admin", "database"): [
        "Executes a schema migration on the {env} {domain} using the provided migration script. Acquires exclusive access during migration. Validates pre- and post-migration schema consistency.",
        "Performs administrative maintenance on the {env} {domain} including index rebuilds and statistics updates. Temporarily increases I/O load. Should be run during low-traffic windows.",
        "Configures replication and failover parameters on the {env} {domain} node. Modifies cluster topology settings. Requires cluster admin privileges.",
    ],
    ("audit", "database"): [
        "Generates a compliance audit report for the {env} {domain} covering read/write access patterns over the specified time window. Output is append-only log format.",
        "Monitors and logs all query executions on the {env} {domain} for security analysis. Captures user, timestamp, query text, and rows affected. Zero impact on query performance.",
        "Traces data lineage for specified tables in the {env} {domain}. Reports all operations that modified the selected records in the audit window.",
    ],
    ("read", "filesystem"): [
        "Reads file contents or directory listings from the {env} {domain}. Supports recursive traversal and glob patterns. Returns metadata including size, permissions, and modification timestamps.",
        "Retrieves file metadata and checksums from the {env} {domain} without transferring file contents. Suitable for integrity verification workflows.",
        "Lists directory contents on the {env} {domain} with configurable depth and filter options. Returns structured JSON with file attributes. Read-only operation.",
    ],
    ("write", "filesystem"): [
        "Writes or updates files on the {env} {domain} storage. Supports atomic writes using temp-file-and-rename semantics. Verifies checksum on write completion.",
        "Copies files from source to the {env} {domain} with progress tracking and resume support. Preserves file metadata and permissions. Fails-safe on partial writes.",
        "Compresses and archives a directory on the {env} {domain} using configurable compression algorithms. Reports archive size and integrity hash on completion.",
    ],
    ("delete", "filesystem"): [
        "Permanently deletes files or directories from the {env} {domain}. Supports recursive deletion with dry-run mode. Unrecoverable once confirmed.",
        "Removes temporary and cached files from the {env} {domain} based on age and size thresholds. Logs each deletion before execution. Reports total freed space.",
        "Wipes a storage volume on the {env} {domain} by overwriting with zeroes. Multi-pass secure wipe mode available. Irreversible; requires explicit confirmation.",
    ],
    ("admin", "filesystem"): [
        "Configures storage quotas and access permissions on the {env} {domain}. Modifies ACLs and ownership attributes. Changes take effect immediately.",
        "Mounts or configures network storage volumes on the {env} {domain}. Modifies filesystem topology and mount options. Requires elevated privileges.",
        "Runs defragmentation and space reclamation on the {env} {domain} storage. Increases I/O utilization during operation. Recommended during maintenance windows.",
    ],
    ("audit", "filesystem"): [
        "Scans the {env} {domain} for unauthorized file modifications using hash comparison against baseline. Reports all changed, added, and deleted paths.",
        "Monitors file access patterns on the {env} {domain} for compliance reporting. Captures read, write, and delete events with actor attribution.",
        "Generates a storage inventory report for the {env} {domain} including file counts, sizes by type, and access frequency metrics.",
    ],
    ("read", "api"): [
        "Sends a GET request to the {env} {domain} endpoint and returns the parsed response. Supports authentication headers and query parameter injection. Non-mutating operation.",
        "Fetches the current configuration or status from the {env} {domain} API. Returns structured JSON with full response headers. Uses read-only API credentials.",
        "Queries the {env} {domain} service health endpoint and reports availability, latency, and version information. Safe for monitoring use.",
    ],
    ("write", "api"): [
        "Registers a new webhook or API configuration on the {env} {domain}. Validates payload schema before submission. Returns the created resource identifier.",
        "Updates an existing API resource on the {env} {domain} with the provided payload. Performs patch semantics for partial updates. Returns updated resource state.",
        "Creates or modifies API endpoint configuration on the {env} {domain}. Validates endpoint accessibility before activation. Logs configuration diff.",
    ],
    ("delete", "api"): [
        "Revokes API credentials or deregisters a webhook from the {env} {domain}. Immediately invalidates the credential. Cannot be undone without re-registration.",
        "Removes an API resource or configuration from the {env} {domain}. Verifies no active dependencies before deletion. Logs the removed resource before execution.",
        "Deletes a registered API endpoint from the {env} {domain} gateway. Existing in-flight requests complete before deletion. Idempotent if resource already absent.",
    ],
    ("admin", "api"): [
        "Rotates API keys and secrets on the {env} {domain} gateway. Issues new credentials and invalidates old ones with configurable grace period. Logs rotation event.",
        "Configures rate limiting and throttle policies on the {env} {domain} API gateway. Changes apply within one request cycle. Requires gateway admin role.",
        "Manages API versioning and deprecation settings on the {env} {domain}. Activates or retires API versions. Affects all consumers of the versioned endpoint.",
    ],
    ("audit", "api"): [
        "Generates an API access audit log for the {env} {domain} covering all authenticated requests in the specified window. Exports to structured JSONL format.",
        "Monitors and reports on API usage patterns for the {env} {domain}. Identifies anomalous access rates, unusual callers, and error spikes.",
        "Traces individual API request chains through the {env} {domain} for forensic analysis. Reconstructs full request-response lifecycle with timing.",
    ],
    ("read", "cicd"): [
        "Retrieves the status and logs for CI/CD pipeline runs on the {env} {domain}. Returns pipeline configuration, stage results, and artifact references.",
        "Fetches build artifacts and test reports from the {env} {domain} CI system. Read-only access to pipeline state. Supports filtering by branch and build ID.",
        "Inspects pipeline configuration and stage definitions on the {env} {domain}. Returns YAML/JSON pipeline spec with current status.",
    ],
    ("write", "cicd"): [
        "Triggers a new pipeline run on the {env} {domain} CI/CD system with specified parameters. Queues the job and returns the run identifier. Non-destructive to existing runs.",
        "Deploys a build artifact to the {env} {domain} environment. Performs pre-deployment health checks and rolls back on failure.",
        "Publishes a new container image to the {env} {domain} registry. Tags the image and updates the latest pointer. Fails on image scan policy violations.",
    ],
    ("delete", "cicd"): [
        "Cancels an in-progress pipeline run on the {env} {domain}. Gracefully terminates running stages and cleans up temporary resources.",
        "Rolls back the most recent deployment on the {env} {domain} to the previous stable version. Restores prior container image and configuration.",
        "Deletes pipeline artifacts and build cache from the {env} {domain} CI storage. Frees allocated storage. Does not affect deployed artifacts.",
    ],
    ("admin", "cicd"): [
        "Configures pipeline triggers, schedules, and branch policies on the {env} {domain}. Changes take effect on the next pipeline run.",
        "Promotes a build artifact through the {env} {domain} deployment gates. Bypasses non-mandatory quality checks. Requires pipeline admin role.",
        "Gates or unblocks a deployment stage on the {env} {domain} pipeline. Allows manual approval workflows. Logged with approver identity.",
    ],
    ("audit", "cicd"): [
        "Generates a deployment audit report for the {env} {domain} covering all deployments, approvals, and rollbacks in the audit window.",
        "Monitors pipeline execution patterns on the {env} {domain} for compliance reporting. Reports on SLA adherence, failure rates, and approval workflows.",
        "Traces artifact provenance through the {env} {domain} build pipeline. Produces a full chain-of-custody report for compliance.",
    ],
    ("read", "iam"): [
        "Retrieves user, role, and permission assignments from the {env} {domain} directory. Returns full entitlement inventory for specified principals.",
        "Describes IAM policies and role bindings on the {env} {domain}. Lists all permissions granted to specified identities. Read-only operation.",
        "Validates whether a specified principal has the required permission on the {env} {domain}. Returns allow/deny with policy attribution.",
    ],
    ("write", "iam"): [
        "Creates a new user account or role in the {env} {domain} identity store. Sets initial permissions and group memberships. Triggers provisioning workflows.",
        "Updates role bindings and permission assignments on the {env} {domain}. Modifies effective permissions immediately. Logs all changes with actor attribution.",
        "Grants a specified permission or role to a principal on the {env} {domain}. Validates against policy before applying. Idempotent if already granted.",
    ],
    ("delete", "iam"): [
        "Permanently deletes a user account or role from the {env} {domain}. Revokes all sessions and access tokens. Irreversible; requires explicit confirmation.",
        "Revokes a permission or role binding from a principal on the {env} {domain}. Takes effect immediately. Terminates any active sessions using the revoked permission.",
        "Disables a user account on the {env} {domain} without deleting underlying data. Blocks all authentication attempts. Can be re-enabled by an admin.",
    ],
    ("admin", "iam"): [
        "Rotates service account credentials and API keys on the {env} {domain}. Issues new credentials with configurable TTL. Invalidates old credentials after grace period.",
        "Configures authentication policy and MFA requirements on the {env} {domain}. Changes apply to all users in scope on next authentication.",
        "Enforces password and credential rotation policies across the {env} {domain}. Sends expiration notifications to affected users.",
    ],
    ("audit", "iam"): [
        "Generates a privilege access audit report for the {env} {domain} covering all privileged operations in the specified time window. Output in compliance-ready format.",
        "Monitors authentication and authorization events on the {env} {domain}. Detects anomalous login patterns and unusual permission usage.",
        "Reports on dormant accounts and unused permissions on the {env} {domain}. Identifies stale entitlements for cleanup review.",
    ],
}

# ── Input schema templates ────────────────────────────────────────────────────

INPUT_SCHEMAS = {
    "read": {
        "type": "object",
        "properties": {
            "filter": {"type": "object", "description": "Query filter criteria"},
            "limit": {"type": "integer", "description": "Maximum records to return", "default": 100},
            "fields": {"type": "array", "items": {"type": "string"}, "description": "Fields to include in response"},
        },
        "required": ["filter"],
    },
    "write": {
        "type": "object",
        "properties": {
            "payload": {"type": "object", "description": "Data to write or update"},
            "dry_run": {"type": "boolean", "description": "If true, validate without persisting", "default": False},
        },
        "required": ["payload"],
    },
    "delete": {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Identifier of the resource to delete"},
            "confirm": {"type": "boolean", "description": "Must be true to execute deletion"},
            "dry_run": {"type": "boolean", "description": "If true, report scope without deleting", "default": False},
        },
        "required": ["target", "confirm"],
    },
    "admin": {
        "type": "object",
        "properties": {
            "config": {"type": "object", "description": "Administrative configuration parameters"},
            "force": {"type": "boolean", "description": "Override soft safety checks", "default": False},
        },
        "required": ["config"],
    },
    "audit": {
        "type": "object",
        "properties": {
            "start_time": {"type": "string", "format": "date-time", "description": "Audit window start (ISO 8601)"},
            "end_time": {"type": "string", "format": "date-time", "description": "Audit window end (ISO 8601)"},
            "format": {"type": "string", "enum": ["json", "csv", "jsonl"], "default": "json"},
        },
        "required": ["start_time", "end_time"],
    },
}


def generate_description(env: str, domain: str, op_class: str) -> str:
    """Generate a standardized 1-3 sentence, 30-60 word description."""
    key = (op_class, domain)
    templates = DESC_TEMPLATES.get(key, [
        f"Performs a {op_class} operation on the {env} {domain} system. Executes the requested action with appropriate access controls and returns a structured result.",
    ])
    template = random.choice(templates)
    return template.format(env=env, domain=domain)


def generate_tool(env: str, domain: str, op_class: str, operation: str, qualifier: str = None) -> dict:
    """Generate a single tool schema."""
    name_parts = [env, domain, operation]
    if qualifier:
        name_parts.append(qualifier)
    tool_name = "_".join(name_parts)

    risk_options = RISK_BY_OP[op_class]
    # Weight toward lower risk for read/audit
    if op_class in ("read", "audit"):
        risk = risk_options[0]
    elif op_class == "write":
        risk = random.choice(risk_options)
    else:
        risk = random.choice(risk_options)

    description = generate_description(env, domain, op_class)
    attributes = [env, op_class, risk, domain]

    return {
        "name": tool_name,
        "description": description,
        "domain": domain,
        "environment": env,
        "op_class": op_class,
        "risk": risk,
        "attributes": attributes,
        "inputSchema": INPUT_SCHEMAS[op_class],
        "outputSchema": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "result": {"type": "object"},
                "error": {"type": "string"},
            },
        },
    }


def generate_registry(target_count: int = 520) -> list[dict]:
    """
    Generate a registry of synthetic tools.

    Strategy: systematically cover all env × domain × op_class combinations,
    then add qualifiers to reach target_count.
    """
    tools = []
    seen_names = set()

    # Phase 1: one tool per (env, domain, op_class, first_operation)
    for env in ENVS:
        for domain in DOMAINS:
            for op_class in OP_CLASSES:
                ops = DOMAIN_OPS[domain][op_class]
                for op in ops[:2]:  # two base tools per combination
                    tool = generate_tool(env, domain, op_class, op)
                    if tool["name"] not in seen_names:
                        tools.append(tool)
                        seen_names.add(tool["name"])

    # Phase 2: add qualifiers to reach target
    random.seed(42)
    all_combinations = [
        (env, domain, op_class, op)
        for env in ENVS
        for domain in DOMAINS
        for op_class in OP_CLASSES
        for op in DOMAIN_OPS[domain][op_class]
    ]
    random.shuffle(all_combinations)

    while len(tools) < target_count:
        for env, domain, op_class, op in all_combinations:
            if len(tools) >= target_count:
                break
            qualifier = random.choice(QUALIFIERS[domain])
            tool = generate_tool(env, domain, op_class, op, qualifier)
            if tool["name"] not in seen_names:
                tools.append(tool)
                seen_names.add(tool["name"])

    # Add a cross-env 'all-envs' set for global tools
    for domain in DOMAINS:
        tool = generate_tool("all-envs", domain, "audit", "audit")
        if tool["name"] not in seen_names:
            tools.append(tool)
            seen_names.add(tool["name"])

    return tools


def seed_mongodb(tools: list[dict], uri: str = "mongodb://localhost:27017") -> None:
    """T-03/T-04: Seed MongoDB with tools and create compound index."""
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        print("ERROR: Cannot connect to MongoDB at", uri)
        print("Start MongoDB with: brew services start mongodb-community")
        sys.exit(1)

    db = client["governed_mcp"]
    col = db["tools"]

    # Drop and recreate for clean seeding
    col.drop()

    # Insert tools
    col.insert_many(tools)
    print(f"Inserted {len(tools)} tools into governed_mcp.tools")

    # T-03: Create compound index on {attributes, domain, environment}
    col.create_index(
        [("attributes", ASCENDING), ("domain", ASCENDING), ("environment", ASCENDING)],
        name="attributes_domain_env_idx",
    )
    col.create_index([("name", ASCENDING)], unique=True, name="name_unique_idx")
    col.create_index([("domain", ASCENDING)], name="domain_idx")
    col.create_index([("op_class", ASCENDING)], name="op_class_idx")
    print("Created indexes: attributes_domain_env_idx, name_unique_idx, domain_idx, op_class_idx")

    client.close()


def verify_queries(uri: str = "mongodb://localhost:27017") -> None:
    """T-04: Spot-check queries to verify correct subsets are returned."""
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client["governed_mcp"]
    col = db["tools"]

    checks = [
        {"attributes": {"$all": ["staging", "database"]}},
        {"attributes": {"$all": ["prod", "read", "low-risk"]}},
        {"attributes": {"$all": ["dev", "delete"]}},
        {"domain": "iam", "attributes": {"$in": ["admin"]}},
        {"attributes": {"$all": ["staging", "cicd", "write"]}},
    ]

    print("\n── Spot-check queries ──────────────────────────────────────")
    for q in checks:
        count = col.count_documents(q)
        sample = list(col.find(q, {"name": 1, "_id": 0}).limit(3))
        names = [t["name"] for t in sample]
        print(f"  Query: {q}")
        print(f"  Count: {count}  Sample: {names}\n")

    total = col.count_documents({})
    print(f"Total tools in registry: {total}")
    client.close()


def main():
    parser = argparse.ArgumentParser(description="Generate and seed tool registry")
    parser.add_argument("--generate", action="store_true", help="Generate tools.json")
    parser.add_argument("--seed", action="store_true", help="Seed MongoDB")
    parser.add_argument("--verify", action="store_true", help="Run spot-check queries")
    parser.add_argument("--all", action="store_true", help="Generate + seed + verify")
    parser.add_argument("--count", type=int, default=520, help="Target tool count")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--output", default="registry/tools.json")
    args = parser.parse_args()

    if not any([args.generate, args.seed, args.verify, args.all]):
        parser.print_help()
        sys.exit(0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.generate or args.all:
        print(f"Generating {args.count} tools...")
        tools = generate_registry(args.count)
        with open(output_path, "w") as f:
            json.dump(tools, f, indent=2)
        print(f"Written {len(tools)} tools to {output_path}")

    if args.seed or args.all:
        if not output_path.exists():
            print("tools.json not found — run --generate first")
            sys.exit(1)
        with open(output_path) as f:
            tools = json.load(f)
        seed_mongodb(tools, args.mongo_uri)

    if args.verify or args.all:
        verify_queries(args.mongo_uri)


if __name__ == "__main__":
    main()
