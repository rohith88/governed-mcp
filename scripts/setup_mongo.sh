#!/usr/bin/env bash
# T-03: Install MongoDB Community Edition and create governed_mcp database with indexes.
# macOS (Apple Silicon / Intel)

set -euo pipefail

echo "── Governed MCP: MongoDB Setup ─────────────────────────────────────"

# ── 1. Install MongoDB Community Edition ─────────────────────────────────────

if command -v mongod &>/dev/null; then
    echo "✓ mongod already installed: $(mongod --version | head -1)"
else
    echo "Installing MongoDB Community Edition via Homebrew..."
    brew tap mongodb/brew
    brew install mongodb-community
fi

# ── 2. Start MongoDB service ──────────────────────────────────────────────────

if brew services list | grep mongodb-community | grep -q started; then
    echo "✓ MongoDB service already running"
else
    echo "Starting MongoDB service..."
    brew services start mongodb-community
    sleep 2
fi

# ── 3. Verify connectivity ────────────────────────────────────────────────────

echo "Testing MongoDB connection..."
mongosh --quiet --eval "db.runCommand({ ping: 1 })" || {
    echo "ERROR: Cannot connect to MongoDB. Check that the service started correctly."
    exit 1
}
echo "✓ MongoDB connection OK"

# ── 4. Create database and indexes via Python ─────────────────────────────────

python - <<'PYEOF'
from pymongo import MongoClient, ASCENDING

client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
client.admin.command("ping")

db = client["governed_mcp"]
col = db["tools"]

# Compound index on attributes array (primary filter for governed queries)
col.create_index(
    [("attributes", ASCENDING), ("domain", ASCENDING), ("environment", ASCENDING)],
    name="attributes_domain_env_idx",
    background=True,
)
# Supporting indexes for ABAC field-based queries
col.create_index([("name", ASCENDING)], unique=True, name="name_unique_idx", background=True)
col.create_index([("domain", ASCENDING)], name="domain_idx", background=True)
col.create_index([("op_class", ASCENDING)], name="op_class_idx", background=True)
col.create_index([("environment", ASCENDING)], name="environment_idx", background=True)
col.create_index([("risk", ASCENDING)], name="risk_idx", background=True)

# Compound index matching build_mongo_query() filter pattern
col.create_index(
    [("environment", ASCENDING), ("op_class", ASCENDING),
     ("risk", ASCENDING), ("domain", ASCENDING)],
    name="abac_compound_idx",
    background=True,
)

print("✓ Database 'governed_mcp' created")
print("✓ Indexes created:", [ix["name"] for ix in col.list_indexes()])
client.close()
PYEOF

echo ""
echo "── Setup complete ───────────────────────────────────────────────────"
echo ""
echo "Next steps:"
echo "  1. Generate tool registry:  python registry/seed_tools.py --all"
echo "  2. Verify seeding:          python registry/seed_tools.py --verify"
echo "  3. Run proxy:               uvicorn proxy.main:app --port 8000 --reload"
echo "  4. Run mock server:         uvicorn mock_server.main:app --port 8001 --reload"
echo ""
