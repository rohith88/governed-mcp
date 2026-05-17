"""
Measure proxy per-stage latency over 1,000 requests.
Reads X-Timing-* response headers and reports median per stage.
Run from repo root: python3 scripts/latency_benchmark.py
"""
import httpx, statistics, sys
sys.path.insert(0, ".")
from proxy.auth import sign_token

N = 1000
BASE = "http://localhost:8001"
TOKEN = sign_token("bench_agent", "dev_agent")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

stages = {
    "JWT":    "x-timing-jwt-ms",
    "AttrAuth": "x-timing-attrauth-ms",
    "MongoDB":  "x-timing-query-ms",
    "ABAC filter": "x-timing-abac-ms",
    "Total":    "x-timing-total-ms",
}

buckets = {k: [] for k in stages}

print(f"Running {N} requests against {BASE}/mcp/tools/list?attribute=developer ...")
with httpx.Client(timeout=10) as client:
    for i in range(N):
        r = client.get(f"{BASE}/mcp/tools/list", params={"attribute": "developer"}, headers=HEADERS)
        if r.status_code != 200:
            print(f"  [WARN] request {i} returned {r.status_code}")
            continue
        for label, header in stages.items():
            val = r.headers.get(header)
            if val:
                buckets[label].append(float(val))

print(f"\n{'Stage':<20} {'Median (ms)':>12} {'P95 (ms)':>10} {'N':>6}")
print("-" * 52)
for label, vals in buckets.items():
    if vals:
        med = statistics.median(vals)
        p95 = sorted(vals)[int(0.95 * len(vals))]
        print(f"{label:<20} {med:>12.3f} {p95:>10.3f} {len(vals):>6}")
    else:
        print(f"{label:<20} {'N/A':>12}")

# Save results
import json, datetime
out = {
    "timestamp": datetime.datetime.now().isoformat(),
    "n_requests": N,
    "results": {
        label: {
            "median_ms": round(statistics.median(vals), 3),
            "p95_ms": round(sorted(vals)[int(0.95*len(vals))], 3),
            "n": len(vals)
        }
        for label, vals in buckets.items() if vals
    }
}
with open("results/proxy_latency.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to results/proxy_latency.json")
