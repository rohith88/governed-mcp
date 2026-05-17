# Governed MCP Proxy

Artifact for the paper **"Governed MCP: Attribute-Based Access Control for LLM Tool Use"**.

The proxy sits between an LLM agent and a tool registry. It enforces ABAC policy at two points: (1) tool discovery (`tools/list`) — unauthorized tools are never returned to the model context; (2) tool invocation (`tools/call`) — a second check blocks any tool the model attempts to call that it was not shown.

---

## Repository structure

```
proxy/          FastAPI proxy — JWT verification, ABAC policy, MongoDB query
benchmark/      Adversarial task generator and experiment runner
registry/       507 MCP-format tools across 7 semantic attribute domains
mock_server/    Stub backend that logs every tool invocation
tests/          Proxy unit tests (pytest)
loadtest/       Locust load test for concurrency experiments
scripts/        Latency benchmark and MongoDB setup
results/        SQLite databases from adversarial experiments + latency JSON
analysis/       Metric computation notebooks and scripts
```

---

## Setup

**Requirements:** Python 3.10+, MongoDB 7.x

```bash
pip install -r requirements.txt
```

Start MongoDB and seed the tool registry:

```bash
mongod --dbpath /tmp/governed-mcp-data &
python registry/seed_tools.py
```

---

## Running the proxy

```bash
# Terminal 1 — mock tool backend
uvicorn mock_server.main:app --host 0.0.0.0 --port 8001

# Terminal 2 — governed proxy
uvicorn proxy.main:app --host 0.0.0.0 --port 8000 --reload
```

Example discovery request:

```bash
TOKEN=$(python -c "from proxy.auth import sign_token; print(sign_token('agent1', 'dev_agent'))")
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:8000/mcp/tools/list?attribute=developer"
```

---

## Running the benchmark

### Adversarial experiment (Table 2 in paper)

```bash
# LLM-only baseline (no proxy)
python benchmark/runner.py --mode adversarial \
    --model claude-haiku-4-5-20251001 --provider anthropic \
    --no-proxy --db results/adv_haiku_llmonly.db

# Prompted baseline (explicit allowlist in system prompt)
python benchmark/runner.py --mode adversarial \
    --model claude-haiku-4-5-20251001 --provider anthropic \
    --no-proxy --prompted --db results/adv_haiku_prompted.db

# Governed (proxy filters unauthorized tools)
python benchmark/runner.py --mode adversarial \
    --model claude-haiku-4-5-20251001 --provider anthropic \
    --db results/adv_haiku_governed.db
```

Supported providers: `openai`, `anthropic`, `openrouter`, `gemini`, `groq`, `together`, `ollama`

API keys are read from environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`.

### Tool Selection Accuracy experiment (Table 1 in paper)

```bash
python benchmark/runner.py --mode tsa \
    --model claude-haiku-4-5-20251001 --provider anthropic \
    --n-levels 100 500 1000 2000
```

---

## Running tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/test_proxy.py -v
```

---

## Latency benchmark

Requires the proxy running on port 8000 with MongoDB seeded:

```bash
python scripts/latency_benchmark.py
# Results saved to results/proxy_latency.json
```

---

## Results

Pre-computed results from the paper are in `results/`:

| File | Description |
|------|-------------|
| `adv_haiku_{governed,prompted,llmonly}.db` | Claude Haiku adversarial runs |
| `adv_qwen7b_{governed,prompted,llmonly}.db` | Qwen2.5-7B adversarial runs |
| `adv_llama8b_{governed,prompted,llmonly}.db` | Llama-3.1-8B adversarial runs |
| `proxy_latency.json` | Per-stage proxy latency (1,000 requests) |

SQLite schema is defined in `benchmark/runner.py` (`SCHEMA`). The key table is `adversarial_results` with columns `condition`, `llm_attempted_unauthorized`, and `category`.

---

## ABAC policy

Roles and their permitted attributes are defined in `proxy/policy.yaml`. The proxy reads this at startup and caches it. To add a new role, add an entry to `policy.yaml` and restart the proxy — no code changes required.
