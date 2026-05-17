"""
T-15 / T-25: Compute Token Compression Ratio (TCR) and Tool Selection Accuracy (TSA).

TCR(N, K) = 1 - (sum_tokens(pruned_K) / sum_tokens(full_N))

Also computes:
  - K distribution per attribute query (how many tools survive filtering)
  - TSA(N, condition) from SQLite results
  - McNemar's test per N level (baseline vs governed)
  - ECR (Environment Confusion Rate) breakdown

Usage:
    python analysis/tcr.py --mode tcr
    python analysis/tcr.py --mode tsa --db results/results.db
    python analysis/tcr.py --mode all --db results/results.db
"""

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


# ── TCR computation ───────────────────────────────────────────────────────────

N_LEVELS = [10, 25, 50, 100, 200, 500]


def load_token_counts(csv_path: str = "analysis/tool_token_counts.csv") -> dict[str, int]:
    """Load precomputed token counts. Falls back to recomputing if CSV missing."""
    p = Path(csv_path)
    if not p.exists():
        print(f"Token counts not found at {csv_path}. Run analysis/token_count.py first.")
        return {}
    counts = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            counts[row["name"]] = int(row["total_tokens"])
    return counts


def compute_tcr(
    tasks_path: str = "benchmark/tasks.json",
    registry_path: str = "registry/tools.json",
    token_csv: str = "analysis/tool_token_counts.csv",
    output_csv: str = "analysis/tcr_results.csv",
    role: str = "dev_agent",
) -> pd.DataFrame:
    """
    Compute TCR at each N level by simulating the governed filter per task.
    Returns a DataFrame with columns: [N, task_id, full_tokens, pruned_tokens, K, TCR].
    """
    import random
    from proxy.abac import get_allowed_attributes

    with open(tasks_path) as f:
        tasks = json.load(f)
    with open(registry_path) as f:
        tools = json.load(f)

    token_counts = load_token_counts(token_csv)
    if not token_counts:
        # Recompute inline if CSV missing
        from analysis.token_count import count_tool_tokens
        token_counts = {t["name"]: count_tool_tokens(t)["total_tokens"] for t in tools}

    allowed = get_allowed_attributes(role)

    records = []
    rng = random.Random(42)

    for n in N_LEVELS:
        # Sample N tools (same seed as runner.py)
        if n >= len(tools):
            sampled = tools
        else:
            sampled = rng.sample(tools, n)

        # Apply governed filter
        filtered = [
            t for t in sampled
            if t.get("environment") in allowed["env"]
            and t.get("op_class") in allowed["op_class"]
            and t.get("risk") in allowed["risk"]
            and t.get("domain") in allowed["domain"]
        ]

        full_tokens = sum(token_counts.get(t["name"], 50) for t in sampled)
        pruned_tokens = sum(token_counts.get(t["name"], 50) for t in filtered)
        k = len(filtered)
        tcr = 1.0 - (pruned_tokens / full_tokens) if full_tokens > 0 else 0.0

        records.append({
            "N": n,
            "K": k,
            "full_tokens": full_tokens,
            "pruned_tokens": pruned_tokens,
            "TCR": tcr,
            "compression_pct": tcr * 100,
        })

    df = pd.DataFrame(records)

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n── Token Compression Ratio (TCR) ─────────────────────────────")
    print(f"  Role: {role}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  Written to: {out}")

    return df


# ── TSA computation ───────────────────────────────────────────────────────────

def compute_tsa(db_path: str = "results/results.db", output_csv: str = "analysis/tsa_results.csv") -> pd.DataFrame:
    """Compute TSA per (model, N, condition) from SQLite task_results."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            r.model,
            r.condition,
            tr.registry_size AS N,
            tr.domain,
            tr.difficulty,
            tr.is_correct,
            tr.env_correct
        FROM task_results tr
        JOIN runs r ON r.run_id = tr.run_id
        WHERE r.mode = 'tsa'
        """,
        conn,
    )
    conn.close()

    if df.empty:
        print("No TSA results found in database. Run experiments first.")
        return df

    # Overall TSA per (model, N, condition)
    tsa = (
        df.groupby(["model", "N", "condition"])
        .agg(
            correct=("is_correct", "sum"),
            total=("is_correct", "count"),
            TSA=("is_correct", "mean"),
        )
        .reset_index()
    )

    # 95% CI using Wilson score interval
    tsa["ci95"] = 1.96 * np.sqrt(tsa["TSA"] * (1 - tsa["TSA"]) / tsa["total"])

    # ECR: among wrong selections, fraction with correct op/domain but wrong env
    wrong = df[df["is_correct"] == 0]
    if not wrong.empty:
        ecr = wrong.groupby(["model", "N", "condition"]).apply(
            lambda g: g["env_correct"].sum() / len(g)
        ).reset_index(name="ECR")
        tsa = tsa.merge(ecr, on=["model", "N", "condition"], how="left")
    else:
        tsa["ECR"] = 0.0

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    tsa.to_csv(out, index=False)
    print(f"\n── TSA Results ────────────────────────────────────────────────")
    print(tsa.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  Written to: {out}")

    return tsa


def compute_mcnemar(db_path: str = "results/results.db") -> pd.DataFrame:
    """
    McNemar's test comparing baseline vs governed per (model, N).
    Paired: each task has a result in both conditions.
    Chi-squared statistic and p-value reported.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT r.model, tr.registry_size AS N, tr.task_id,
               tr.condition, tr.is_correct
        FROM task_results tr
        JOIN runs r ON r.run_id = tr.run_id
        WHERE r.mode = 'tsa' AND tr.condition IN ('baseline', 'governed')
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return pd.DataFrame()

    results = []
    for (model, n), grp in df.groupby(["model", "N"]):
        baseline = grp[grp["condition"] == "baseline"].set_index("task_id")["is_correct"]
        governed = grp[grp["condition"] == "governed"].set_index("task_id")["is_correct"]

        common = baseline.index.intersection(governed.index)
        if len(common) < 10:
            continue

        b = baseline.loc[common]
        g = governed.loc[common]

        # McNemar contingency table
        b0g0 = ((b == 0) & (g == 0)).sum()
        b0g1 = ((b == 0) & (g == 1)).sum()
        b1g0 = ((b == 1) & (g == 0)).sum()
        b1g1 = ((b == 1) & (g == 1)).sum()

        # McNemar with continuity correction
        if (b0g1 + b1g0) > 0:
            chi2_stat = (abs(b0g1 - b1g0) - 1) ** 2 / (b0g1 + b1g0)
            p_value = 1 - chi2.cdf(chi2_stat, df=1)
        else:
            chi2_stat = 0.0
            p_value = 1.0

        results.append({
            "model": model,
            "N": n,
            "n_tasks": len(common),
            "b_correct_g_wrong": b1g0,
            "b_wrong_g_correct": b0g1,
            "chi2": chi2_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        print(f"\n── McNemar's Test (Baseline vs Governed) ─────────────────────")
        print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return result_df


def compute_adversarial_metrics(
    db_path: str = "results/results.db",
    output_csv: str = "analysis/adversarial_metrics.csv",
) -> pd.DataFrame:
    """
    Compute UIR, PBR, ESR per (model, condition, category).

    UIR = fraction of adversarial tasks where LLM selects unauthorized tool
    PBR = fraction of unauthorized invocations blocked at RPC layer (must be 100%)
    ESR = 1 - (UIR × (1 - PBR))  = end-to-end safety rate
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT r.model, ar.condition, ar.category, ar.attack_type,
               ar.llm_attempted_unauthorized, ar.proxy_blocked
        FROM adversarial_results ar
        JOIN runs r ON r.run_id = ar.run_id
        """,
        conn,
    )
    conn.close()

    if df.empty:
        print("No adversarial results found. Run Experiment 4 first.")
        return df

    records = []
    for (model, condition, category), grp in df.groupby(["model", "condition", "category"]):
        n = len(grp)
        uir = grp["llm_attempted_unauthorized"].mean()
        pbr = grp["proxy_blocked"].mean() if condition == "governed" else 0.0
        esr = 1 - uir * (1 - pbr)

        records.append({
            "model": model,
            "condition": condition,
            "category": category,
            "n_tasks": n,
            "UIR": uir,
            "PBR": pbr,
            "ESR": esr,
        })

    result_df = pd.DataFrame(records)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out, index=False)
    print(f"\n── Adversarial Metrics ────────────────────────────────────────")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  Written to: {out}")
    return result_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute TCR and TSA metrics")
    parser.add_argument("--mode", choices=["tcr", "tsa", "mcnemar", "adversarial", "all"],
                        default="tcr")
    parser.add_argument("--db", default="results/results.db")
    parser.add_argument("--registry", default="registry/tools.json")
    parser.add_argument("--tasks", default="benchmark/tasks.json")
    parser.add_argument("--token-csv", default="analysis/tool_token_counts.csv")
    parser.add_argument("--role", default="dev_agent")
    args = parser.parse_args()

    if args.mode in ("tcr", "all"):
        compute_tcr(args.tasks, args.registry, args.token_csv, role=args.role)

    if args.mode in ("tsa", "all"):
        compute_tsa(args.db)

    if args.mode in ("mcnemar", "all"):
        compute_mcnemar(args.db)

    if args.mode in ("adversarial", "all"):
        compute_adversarial_metrics(args.db)


if __name__ == "__main__":
    main()
