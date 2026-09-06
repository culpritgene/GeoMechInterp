"""Summarise the parameter-efficiency sweep.

    python projects/manifold_features/plot_results.py

Reads results/sweep_*.csv, averages seeds, and writes
    results/acc_vs_params.png        test accuracy vs #params, one panel per shape
    results/near_acc_vs_params.png   accuracy on points within 0.05 of the surface
    results/params_to_threshold.csv  min #params per family reaching each accuracy
    results/summary.md               the same table as markdown
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SHAPES  # noqa: E402

RES = Path(__file__).resolve().parent / "results"

# categorical slots from the dataviz reference palette, fixed order per family
FAMILIES = ["sae", "sae_weak", "relu1", "mlp2", "spline1", "spline3"]
LABELS = {
    "sae": "SAE (sparse ReLU, L1=1e-3)",
    "sae_weak": "SAE (sparse ReLU, L1=1e-4)",
    "relu1": "dense ReLU, 1 hidden",
    "mlp2": "ReLU MLP, 2 hidden",
    "spline1": "linear-spline net (k=1)",
    "spline3": "cubic-spline net (k=3)",
}
COLORS = {"sae": "#2a78d6", "sae_weak": "#008300", "relu1": "#eb6834", "mlp2": "#1baf7a", "spline1": "#eda100", "spline3": "#e87ba4"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
THRESHOLDS = [0.95, 0.98, 0.99, 0.995]


def load() -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(glob.glob(str(RES / "sweep_*.csv")))]
    df = pd.concat(frames, ignore_index=True)
    num = [c for c in df.columns if c.startswith(("test_", "val_", "train_")) or c in ("sparsity", "seconds")]
    agg = df.groupby(["shape", "family", "size", "n_params"])[num].mean().reset_index()
    agg["n_seeds"] = df.groupby(["shape", "family", "size", "n_params"]).size().values
    return agg


def pareto(sub: pd.DataFrame, metric: str):
    """Running max of metric over increasing parameter count."""
    s = sub.sort_values("n_params")
    return s.n_params.values, np.maximum.accumulate(s[metric].values)


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7"); ax.spines[sp].set_linewidth(1)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def small_multiples(agg: pd.DataFrame, metric: str, title: str, out: Path, ylim=(0.5, 1.0)):
    shapes = [s for s in SHAPES if s in set(agg["shape"])]
    ncol = 4; nrow = int(np.ceil(len(shapes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, shape in zip(axes.flat, shapes):
        style(ax)
        sub = agg[agg["shape"] == shape]
        for fam in FAMILIES:
            f = sub[sub.family == fam]
            if f.empty:
                continue
            ax.scatter(f.n_params, f[metric], s=12, color=COLORS[fam], alpha=0.35, linewidths=0)
            x, y = pareto(f, metric)
            ax.plot(x, y, color=COLORS[fam], linewidth=2, solid_joinstyle="round", label=LABELS[fam])
        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.set_title(shape, fontsize=10, color=INK, loc="left")
    for ax in axes.flat[len(shapes):]:
        ax.axis("off")
    for ax in axes[-1] if nrow > 1 else [axes]:
        ax.set_xlabel("parameters", color=MUTED, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(FAMILIES), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=12, color=INK)
    fig.text(0.01, 0.962, "lines: best accuracy achievable at or below a parameter count (mean over seeds); dots: individual configs",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.945))
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def params_to_threshold(agg: pd.DataFrame, metric: str = "test_acc") -> pd.DataFrame:
    rows = []
    for (shape, fam), sub in agg.groupby(["shape", "family"]):
        x, y = pareto(sub, metric)
        row = {"shape": shape, "family": fam, "max_acc": float(y.max()), "max_params": int(x[-1])}
        for t in THRESHOLDS:
            hit = np.where(y >= t)[0]
            row[f"params@{t}"] = int(x[hit[0]]) if len(hit) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    agg = load()
    RES.mkdir(exist_ok=True)
    small_multiples(agg, "test_acc", "Test accuracy vs parameter count (exterior / surface / interior)", RES / "acc_vs_params.png")
    small_multiples(agg, "test_acc_near", "Accuracy on off-surface points within 0.05 of the surface", RES / "near_acc_vs_params.png", ylim=(0.0, 1.0))
    tab = params_to_threshold(agg)
    tab.to_csv(RES / "params_to_threshold.csv", index=False)
    # aggregate across shapes: geometric mean of params needed, and how many shapes reach the threshold
    lines = ["# Parameter-efficiency summary", "", f"Shapes: {agg["shape"].nunique()}, seeds per config: {int(agg.n_seeds.min())}-{int(agg.n_seeds.max())}", ""]
    lines += ["## Minimum parameters to reach a test accuracy (geometric mean over shapes that reach it; n = shapes reaching it)", ""]
    lines.append("| family | " + " | ".join(f"{t:.1%}" for t in THRESHOLDS) + " | max acc (mean) |")
    lines.append("|---|" + "---|" * (len(THRESHOLDS) + 1))
    for fam in FAMILIES:
        f = tab[tab.family == fam]
        if f.empty:
            continue
        cells = []
        for t in THRESHOLDS:
            v = f[f"params@{t}"].dropna()
            cells.append(f"{np.exp(np.log(v).mean()):.0f} (n={len(v)})" if len(v) else "not reached")
        lines.append(f"| {LABELS[fam]} | " + " | ".join(cells) + f" | {f.max_acc.mean():.4f} |")
    lines += ["", "## Per shape: parameters to reach 99% test accuracy", ""]
    piv = tab.pivot(index="shape", columns="family", values="params@0.99").reindex([s for s in SHAPES if s in set(tab["shape"])])
    piv = piv[[f for f in FAMILIES if f in piv.columns]]
    lines.append("| shape | " + " | ".join(piv.columns) + " |")
    lines.append("|---|" + "---|" * len(piv.columns))
    for shape, r in piv.iterrows():
        lines.append(f"| {shape} | " + " | ".join("-" if np.isnan(v) else f"{int(v)}" for v in r.values) + " |")
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
