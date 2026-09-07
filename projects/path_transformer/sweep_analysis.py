"""Summarise the manifold sweep: task quality and probe gaps as a function of
curvature / hole size, sampling density, oct-tree depth, surface-shell noise,
path temperature and observation noise, for two model sizes.

Reads, per dataset name D and size tag S in {L3_d64, L6_d256}:
    /var/tmp/geomech_ckpt/<D>_flat_<S>/summary.json      (training + generation quality)
    /var/tmp/geomech_ckpt/<D>_flat_<S>/probes_v3.json    (matched-parameter probes)
    data/geodesic_datasets/generated/<D>.npz             (meta: n_cells, temperature, p_obs)
    /var/tmp/geomech_data/manifolds/<M>.npz              (meta: r, n_points, depth, shell)

Writes results/sweep_summary.md and results/sweep_<metric>.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CKPT = Path("/var/tmp/geomech_ckpt")
GEN = Path("/var/tmp/geomech_data/generated")
MAN = Path("/var/tmp/geomech_data/manifolds")
RES = Path(__file__).resolve().parent / "results"
SIZES = ["L3_d64", "L6_d256"]
AXES = {  # axis name -> (datasets in order, x values, x label)
    "hole": (["ring_r012", "ring_r025", "ring_r040", "ring_r050"], [0.12, 0.25, 0.40, 0.50], "minor radius r (hole radius = 1 - r)"),
    "density": (["ring_n3k", "ring_r025", "ring_n40k"], [3000, 11000, 40000], "surface samples"),
    "depth": (["ring_d4", "ring_r025", "ring_d6"], [4, 5, 6], "oct-tree depth"),
    "shell": (["ring_r025", "ring_shell015", "ring_shell030", "ring_shell060"], [0, 0.15, 0.30, 0.60], "shell half-thickness / r"),
    "temperature": (["ring_r025", "ring_T025", "ring_T05"], [0, 0.25, 0.5], "path temperature"),
    "p_obs": (["ring_r025", "ring_obs005", "ring_obs015"], [0, 0.05, 0.15], "observation noise p"),
    "chain": (["chain_r025", "chain_r045"], [0.25, 0.45], "chain minor radius r"),
}
COLORS = {"L3_d64": "#2a78d6", "L6_d256": "#eb6834"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"


def best_under(rows, probe, target, layer, budget):
    v = [r["r2"] for r in rows if r["probe"] == probe and r["target"] == target and r["layer"] == layer and r["n_params"] <= budget]
    return max(v) if v else np.nan


def collect() -> pd.DataFrame:
    recs = []
    for axis, (names, xs, _) in AXES.items():
        for name, x in zip(names, xs):
            gmeta = {}
            f = GEN / f"{name}.npz"
            if f.exists():
                z = np.load(f, allow_pickle=True); gmeta = json.loads(str(z["meta"])); gmeta["n_cells"] = int(z["cell_table"].shape[0])
                # training paths may be stochastic (temperature > 0): their length relative to the exact geodesic
                gmeta["train_path_over_geo"] = float(np.mean(z["path_len"] / z["geo_dist"])) if "path_len" in z.files else 1.0
                mname = gmeta.get("shape")
                mf = MAN / f"{mname}.npz"
                if mf.exists():
                    gmeta.update({f"m_{k}": v for k, v in json.loads(str(np.load(mf, allow_pickle=True)["meta"])).items()})
            for size in SIZES:
                d = CKPT / f"{name}_flat_{size}"
                rec = dict(axis=axis, dataset=name, x=x, size=size, n_cells=gmeta.get("n_cells"), r=gmeta.get("m_r"), train_path_over_geo=gmeta.get("train_path_over_geo"),
                           shell=gmeta.get("m_shell"), temperature=gmeta.get("temperature"), p_obs=gmeta.get("p_obs"))
                s = d / "summary.json"
                if s.exists():
                    sj = json.load(open(s))
                    rec.update(success=sj["test_success"], valid=sj["test_valid"], reached=sj["test_reached"], len_ratio=sj["test_len_ratio"],
                               len_over_geo=sj["test_len_ratio"] * gmeta.get("train_path_over_geo", 1.0),
                               val_success=sj["best_val_success"], n_params=sj["n_params"])
                p = d / "probes_v3.json"
                if p.exists():
                    rows = json.load(open(p)); mid = 2 if size == "L3_d64" else 4
                    for tgt in ("pos", "dir"):
                        lin = [r["r2"] for r in rows if r["probe"] == "linear" and r["target"] == tgt and r["layer"] == mid]
                        rec[f"{tgt}_linear"] = lin[0] if lin else np.nan
                        for b in (2500, 10000):
                            rec[f"{tgt}_relu_{b}"] = best_under(rows, "relu+linear", tgt, mid, b)
                            rec[f"{tgt}_spline_{b}"] = best_under(rows, "spline1L", tgt, mid, b)
                            rec[f"{tgt}_gap_{b}"] = rec[f"{tgt}_spline_{b}"] - rec[f"{tgt}_relu_{b}"]
                        rec[f"{tgt}_sae"] = best_under(rows, "sae+linear", tgt, mid, 10 ** 7)
                recs.append(rec)
    return pd.DataFrame(recs)


def plot(df: pd.DataFrame, metric: str, ylabel: str, out: Path, ylim=None):
    axes_names = list(AXES)
    fig, axs = plt.subplots(1, len(axes_names), figsize=(3.3 * len(axes_names), 3.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, axis in zip(axs, axes_names):
        ax.set_facecolor(SURFACE)
        for size in SIZES:
            s = df[(df.axis == axis) & (df["size"] == size)].sort_values("x")
            if s[metric].notna().any():
                ax.plot(s.x, s[metric], color=COLORS[size], linewidth=2, marker="o", markersize=5, label=size)
        ax.set_title(axis, fontsize=10, color=INK, loc="left"); ax.set_xlabel(AXES[axis][2], fontsize=8, color=MUTED)
        ax.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if axis in ("density",):
            ax.set_xscale("log")
        if ylim:
            ax.set_ylim(*ylim)
    axs[0].set_ylabel(ylabel, fontsize=9, color=MUTED)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out, dpi=140, facecolor=SURFACE); plt.close(fig)


def main():
    df = collect(); RES.mkdir(exist_ok=True)
    df.to_csv(RES / "sweep_summary.csv", index=False)
    plot(df, "success", "test success (valid & reached)", RES / "sweep_success.png", ylim=(0.5, 1.0))
    plot(df, "len_over_geo", "generated path length / exact geodesic", RES / "sweep_len_ratio.png")
    if "pos_gap_2500" in df:
        plot(df, "pos_linear", "linear R² for position (mid layer)", RES / "sweep_pos_linear.png", ylim=(0, 1))
        plot(df, "pos_gap_2500", "spline - ReLU R² for position (<= 2.5k params)", RES / "sweep_pos_gap2500.png")
        plot(df, "dir_gap_2500", "spline - ReLU R² for step direction (<= 2.5k params)", RES / "sweep_dir_gap2500.png")
    cols = ["axis", "dataset", "x", "size", "n_cells", "success", "len_over_geo", "pos_linear", "pos_relu_2500", "pos_spline_2500", "pos_gap_2500", "pos_gap_10000", "pos_sae", "dir_gap_2500"]
    cols = [c for c in cols if c in df]
    lines = ["# Manifold sweep summary", "", "success = valid and goal-reaching on held-out pairs; probe columns are held-out R² at the middle layer (2 for L3_d64, 4 for L6_d256).", ""]
    lines.append("| " + " | ".join(cols) + " |"); lines.append("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(("" if pd.isna(r[c]) else (f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]))) for c in cols) + " |")
    (RES / "sweep_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
