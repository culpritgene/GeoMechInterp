"""Summarise group-tracking probes: best held-out R^2 per probe family at
parameter budgets, per (cell, layer, target), next to the linear null control.

    python projects/group_tracking/summarize_group_probes.py

Reads results/runs/<run>_probes.json (rows with probe, size, n_params, r2 or
acc, cell, layer, target) and results/runs/null_<group>_d<d>*.json, writes
results/group_probes.md and results/group_probes.png.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "results"
RUNS = RES / "runs"
FAMS = ["linear", "sae+linear", "relu+linear", "spline1L"]
LABEL = {"linear": "linear (ridge)", "sae+linear": "top-k SAE + linear", "relu+linear": "ReLU hinges + linear", "spline1L": "cubic splines + linear"}
COLOR = {"linear": "#898781", "sae+linear": "#2a78d6", "relu+linear": "#eb6834", "spline1L": "#e87ba4"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
BUDGETS = [500, 1000, 2500, 5000, 10000, 50000, 10 ** 7]


def load_rows(pattern: str, kind: str) -> pd.DataFrame:
    out = []
    for f in sorted(glob.glob(str(RUNS / pattern))):
        j = json.load(open(f)); name = Path(f).name
        for r in j.get("rows", []):
            r = dict(r); r["file"] = name; r["kind"] = kind
            m = re.match(r"(?:group_|null_)(\w+?)_(?:L4_)?d(\d+)", name)
            if m:
                r["group"], r["d"] = m.group(1), int(m.group(2))
            r["metric"] = r.get("r2", r.get("acc"))
            out.append(r)
    return pd.DataFrame(out)


def best_under(df, fam, budget):
    s = df[(df.probe == fam) & (df.n_params <= budget)]
    return float(s.metric.max()) if len(s) else np.nan


def main():
    probes = load_rows("group_*_probes.json", "model"); nulls = load_rows("null_*.json", "null")
    if probes.empty:
        print("no probe results yet"); return
    RES.mkdir(exist_ok=True)
    lines = ["# Group-tracking probe summary", "", "Best held-out R² (or accuracy for element/eps) per family at or below a parameter budget.", ""]
    runs = sorted(probes.file.unique())
    for run in runs:
        P = probes[probes.file == run]
        lines += [f"## {run}", ""]
        for (cell, layer, target), sub in P.groupby(["cell", "layer", "target"]):
            hdr = f"| {cell} L{layer} {target} | " + " | ".join(f"<= {b}" if b < 10 ** 6 else "any" for b in BUDGETS) + " |"
            lines += [hdr, "|---|" + "---|" * len(BUDGETS)]
            fams = [f for f in FAMS if f in set(sub.probe)] + [f for f in ("linear_softmax", "logistic") if f in set(sub.probe)]
            for fam in fams:
                cells = [best_under(sub, fam, b) for b in BUDGETS]
                lines.append(f"| {LABEL.get(fam, fam)} | " + " | ".join("-" if np.isnan(c) else f"{c:.3f}" for c in cells) + " |")
            lines.append("")
    if not nulls.empty:
        lines += ["## Linear null controls (probe-capacity ceiling): spline - hinge gap", ""]
        for f, sub in nulls.groupby("file"):
            lines.append(f"### {f}")
            for target, s in sub.groupby("target"):
                gaps = [best_under(s, "spline1L", b) - best_under(s, "relu+linear", b) for b in BUDGETS]
                lines.append(f"- {target}: " + ", ".join(f"<= {b}: {g:+.3f}" for b, g in zip(BUDGETS, gaps) if not np.isnan(g)))
            lines.append("")
    (RES / "group_probes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:80]))

    # figure: irrep target, gen cell, best layer per run, R^2 vs params
    P = probes[(probes.target == "irrep") & (probes.cell == "gen")]
    if P.empty:
        return
    runs = sorted(P.file.unique())
    fig, axes = plt.subplots(1, len(runs), figsize=(3.6 * len(runs), 3.4), sharey=True); fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, run in zip(axes, runs):
        sub = P[P.file == run]
        layer = sub.groupby("layer").metric.max().idxmax(); sub = sub[sub.layer == layer]
        ax.set_facecolor(SURFACE)
        for fam in FAMS:
            s = sub[sub.probe == fam].sort_values("n_params")
            if s.empty:
                continue
            if fam == "linear":
                ax.axhline(s.metric.iloc[0], color=COLOR[fam], linewidth=2, label=LABEL[fam])
            elif fam == "sae+linear":
                ax.scatter(s.n_params, s.metric, color=COLOR[fam], s=28, linewidths=0, label=LABEL[fam], zorder=3)
            else:
                ax.plot(s.n_params, s.metric, color=COLOR[fam], linewidth=2, marker="o", markersize=4, label=LABEL[fam])
        ax.set_xscale("log"); ax.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(run.replace("_probes.json", "") + f" (layer {layer})", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("probe parameters", fontsize=8, color=MUTED)
    axes[0].set_ylabel("R² for irrep coordinates of the prefix product", fontsize=8, color=MUTED)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(RES / "group_probes.png", dpi=140, facecolor=SURFACE)
    print("saved", RES / "group_probes.png")


if __name__ == "__main__":
    main()
