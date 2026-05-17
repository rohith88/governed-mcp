"""
Generate Figure 2: UIR bar chart across models and conditions.

Reads from the adversarial result SQLite databases in results/ and writes
paper/uir_results.pdf (and .png).

Usage:
    python analysis/plot_uir.py
"""

import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Data sources ──────────────────────────────────────────────────────────────

RESULTS = [
    # (model_label, condition, db_path)
    ("Qwen 2.5 7B",      "llm_only",  "results/adv_qwen7b_llmonly.db"),
    ("Qwen 2.5 7B",      "prompted",  "results/adv_qwen7b_prompted.db"),
    ("Qwen 2.5 7B",      "governed",  "results/adv_qwen7b_governed.db"),
    ("Llama 3.1 8B",     "llm_only",  "results/adv_llama8b_llmonly.db"),
    ("Llama 3.1 8B",     "prompted",  "results/adv_llama8b_prompted.db"),
    ("Llama 3.1 8B",     "governed",  "results/adv_llama8b_governed.db"),
    ("Claude Haiku 3.5", "llm_only",  "results/adv_haiku_llmonly.db"),
    ("Claude Haiku 3.5", "prompted",  "results/adv_haiku_prompted.db"),
    ("Claude Haiku 3.5", "governed",  "results/adv_haiku_governed.db"),
]

CONDITION_LABELS = {
    "llm_only": "Unfiltered",
    "prompted": "Prompted",
    "governed": "Governed",
}

CONDITION_COLORS = {
    "llm_only": "#d62728",
    "prompted": "#ff7f0e",
    "governed": "#1f77b4",
}


def get_uir(db_path: str) -> float:
    """Return UIR (fraction of tasks with llm_attempted_unauthorized=1)."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT AVG(llm_attempted_unauthorized) FROM adversarial_results"
    ).fetchone()
    conn.close()
    if row and row[0] is not None:
        return float(row[0]) * 100
    return 0.0


# ── Compute UIR for each entry ────────────────────────────────────────────────

data: dict[str, dict[str, float]] = {}
for model_label, condition, db_path in RESULTS:
    if not Path(db_path).exists():
        print(f"  [WARN] missing: {db_path}")
        continue
    uir = get_uir(db_path)
    data.setdefault(model_label, {})[condition] = uir
    print(f"  {model_label:20s} {CONDITION_LABELS[condition]:12s} UIR={uir:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────

models = ["Qwen 2.5 7B", "Llama 3.1 8B", "Claude Haiku 3.5"]
conditions = ["llm_only", "prompted", "governed"]

x = np.arange(len(models))
width = 0.22
offsets = [-width, 0, width]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

fig, ax = plt.subplots(figsize=(7, 4))

for offset, condition in zip(offsets, conditions):
    uirs = [data.get(m, {}).get(condition, 0.0) for m in models]
    bars = ax.bar(x + offset, uirs, width,
                  color=CONDITION_COLORS[condition], alpha=0.88,
                  label=CONDITION_LABELS[condition])
    for bar, val in zip(bars, uirs):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=9,
            )

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Unauthorized Invocation Rate (%)")
ax.set_ylim(0, 85)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out_pdf = Path("paper/uir_results.pdf")
out_png = Path("paper/uir_results.png")
out_pdf.parent.mkdir(exist_ok=True)
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"\nSaved: {out_pdf}  {out_png}")
