"""
T-14: Token counting for all registry tools using tiktoken cl100k_base.

Counts: tool name + description + property names + property descriptions.
Does NOT count JSON structural tokens (braces, colons, quotes).

Output: analysis/tool_token_counts.csv

Usage:
    python analysis/token_count.py
    python analysis/token_count.py --registry registry/tools.json --output analysis/tool_token_counts.csv
"""

import argparse
import csv
import json
from pathlib import Path

import tiktoken


ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def count_tool_tokens(tool: dict) -> dict:
    """
    Count tokens for the semantic content of a tool definition.
    Counts: name + description + property names + property descriptions.
    Excludes JSON structural tokens per paper methodology (Section 4.2).
    """
    name_tokens = count_tokens(tool.get("name", ""))
    desc_tokens = count_tokens(tool.get("description", ""))

    # Count property names and descriptions from inputSchema
    prop_name_tokens = 0
    prop_desc_tokens = 0
    schema = tool.get("inputSchema", {})
    props = schema.get("properties", {})
    for prop_name, prop_schema in props.items():
        prop_name_tokens += count_tokens(prop_name)
        if "description" in prop_schema:
            prop_desc_tokens += count_tokens(prop_schema["description"])

    total = name_tokens + desc_tokens + prop_name_tokens + prop_desc_tokens

    return {
        "name": tool["name"],
        "domain": tool.get("domain", ""),
        "environment": tool.get("environment", ""),
        "op_class": tool.get("op_class", ""),
        "risk": tool.get("risk", ""),
        "name_tokens": name_tokens,
        "desc_tokens": desc_tokens,
        "prop_name_tokens": prop_name_tokens,
        "prop_desc_tokens": prop_desc_tokens,
        "total_tokens": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Count tokens for registry tools")
    parser.add_argument("--registry", default="registry/tools.json")
    parser.add_argument("--output", default="analysis/tool_token_counts.csv")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"ERROR: Registry not found at {args.registry}")
        print("Run: python registry/seed_tools.py --generate")
        return

    with open(registry_path) as f:
        tools = json.load(f)

    rows = [count_tool_tokens(t) for t in tools]

    # Summary statistics
    totals = [r["total_tokens"] for r in rows]
    print(f"\n── Token Count Summary ────────────────────────────────────────")
    print(f"  Tools:            {len(rows)}")
    print(f"  Total tokens:     {sum(totals):,}")
    print(f"  Mean per tool:    {sum(totals)/len(totals):.1f}")
    print(f"  Median per tool:  {sorted(totals)[len(totals)//2]}")
    print(f"  Min/Max:          {min(totals)} / {max(totals)}")

    # By domain
    domain_stats: dict[str, list[int]] = {}
    for r in rows:
        domain_stats.setdefault(r["domain"], []).append(r["total_tokens"])
    print(f"\n  By domain:")
    for domain, toks in sorted(domain_stats.items()):
        print(f"    {domain:<15} mean={sum(toks)/len(toks):.1f}  n={len(toks)}")

    # Write CSV
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Written to: {out}")


if __name__ == "__main__":
    main()
