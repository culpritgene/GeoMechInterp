"""Summarise probe results: R^2 vs parameter count per probe family.

    python projects/path_transformer/summarize_probes.py --file probes_v2.json --layer 4

Writes results/probes_<file>.png (small multiples: rows = targets, cols =
shapes) and results/probes_<file>.md (best R^2 per family under budgets).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CKPT = Path("/var/tmp/geomech_ckpt")
RES = Path(__file__).resolve().parent / "results"
FAMS = ["linear", "sae+linear", "relu+linear", "spline1L", "spline2L", "spline"]
LABEL = {"linear": "linear (ridge)", "sae+linear": "top-k SAE + linear", "relu+linear": "ReLU hinges + linear",
         "spline1L": "cubic splines + linear", "spline2L": "2-layer cubic KAN", "spline": "2-layer cubic KAN"}
COLOR = {"linear": "#898781", "sae+linear": "#2a78d6", "relu+linear": "#eb6834", "spline1L": "#e87ba4", "spline2L": "#1baf7a", "spline": "#1baf7a"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
TARGETS = ["pos", "goal", "dir", "remain"]
BUDGETS = [1000, 2500, 5000, 10000, 40000, 300000]


def load(file: str) -> pd.DataFrame:
    rows = []
    for d in sorted(CKPT.glob("*_flat_*")):
        f = d / file
        if f.exists():
            for r in json.load(open(f)):
                r["run"] = d.name; rows.append(r)
    return pd.DataFrame(rows)


def main(a):
    df = load(a.file)
    df = df[df.layer == a.layer]
    runs = sorted(df.run.unique()); fams = [f for f in FAMS if f in set(df.probe)]
    fig, axes = plt.subplots(len(TARGETS), len(runs), figsize=(4.2 * len(runs), 3.0 * len(TARGETS)), sharex=True, sharey="row")
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes)
    for j, run in enumerate(runs):
        for i, tgt in enumerate(TARGETS):
            ax = axes[i, j]; ax.set_facecolor(SURFACE)
            sub = df[(df.run == run) & (df.target == tgt)]
            for fam in fams:
                s = sub[sub.probe == fam].sort_values("n_params")
                if s.empty:
                    continue
                if fam == "linear":
                    ax.axhline(s.r2.iloc[0], color=COLOR[fam], linewidth=2, label=LABEL[fam])
                elif fam == "sae+linear":  # several k per width share a parameter count: markers only
                    ax.scatter(s.n_params, s.r2, color=COLOR[fam], s=28, linewidths=0, label=LABEL[fam], zorder=3)
                else:
                    ax.plot(s.n_params, s.r2, color=COLOR[fam], linewidth=2, marker="o", markersize=4, label=LABEL[fam])
            ax.set_xscale("log"); ax.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            if i == 0:
                ax.set_title(run.replace("_flat", ""), fontsize=10, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(f"R² for {tgt}", fontsize=9, color=MUTED)
            if i == len(TARGETS) - 1:
                ax.set_xlabel("probe parameters", fontsize=9, color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(fams), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Decoding geometry from the layer-{a.layer} residual stream: held-out R² vs probe size", x=0.01, y=0.995, ha="left", va="top", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    RES.mkdir(exist_ok=True)
    out = RES / f"probes_{a.file.replace('.json', '')}_L{a.layer}.png"
    fig.savefig(out, dpi=140, facecolor=SURFACE); print("saved", out)

    lines = [f"# Probe summary ({a.file}, layer {a.layer})", "", "Best held-out R² per probe family at or below a parameter budget, mean over shapes.", ""]
    for tgt in TARGETS:
        lines += [f"## target: {tgt}", "", "| probe | " + " | ".join(f"<= {b}" for b in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
        for fam in fams:
            cells = []
            for b in BUDGETS:
                vals = [df[(df.run == r) & (df.target == tgt) & (df.probe == fam) & (df.n_params <= b)].r2.max() for r in runs]
                vals = [v for v in vals if not np.isnan(v)]
                cells.append(f"{np.mean(vals):.3f}" if len(vals) == len(runs) else "-")
            lines.append(f"| {LABEL[fam]} | " + " | ".join(cells) + " |")
        lines.append("")
    (RES / f"probes_{a.file.replace('.json', '')}_L{a.layer}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="probes_v2.json")
    ap.add_argument("--layer", type=int, default=4)
    main(ap.parse_args())
