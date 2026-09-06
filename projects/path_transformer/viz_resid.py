"""Visualise the residual-stream geometry of a trained path model.

For single_ring the ground-truth position of a cell is parametrised by the
toroidal angle theta (around the ring axis, dataset axis 1) and the poloidal
angle phi (around the tube).  Each panel is a 2-D PCA of the layer-l residual
at path-token positions, coloured by theta (cyclic colormap), plus a panel
coloured by phi and one by remaining distance to the goal.

    python projects/path_transformer/viz_resid.py --run single_ring_flat_L6_d256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe import CKPT, collect  # noqa: E402

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"


def pca2(X):
    X = X - X.mean(0)
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    ev = (S ** 2 / (S ** 2).sum())[:2].sum().item()
    return (X @ Vt[:2].T).numpy(), ev


def main(a):
    device = torch.device("cuda")
    feats, tg, pd = collect(a.run, a.n_seq, "test", device)
    P = tg["pos"] * 1.0  # normalised cell centres
    theta = np.arctan2(P[:, 2], P[:, 0])
    rho = np.hypot(P[:, 0], P[:, 2])
    phi = np.arctan2(P[:, 1], rho - np.median(rho))
    colour = {"theta (around ring)": (theta, "twilight"), "phi (around tube)": (phi, "twilight"),
              "remaining distance": (tg["remain"][:, 0], "Blues")}
    layers = sorted(feats)
    sub = np.random.default_rng(0).choice(len(P), size=min(a.n_points, len(P)), replace=False)
    fig, axes = plt.subplots(len(colour), len(layers), figsize=(2.6 * len(layers), 2.6 * len(colour)))
    fig.patch.set_facecolor(SURFACE)
    for j, l in enumerate(layers):
        Z, ev = pca2(feats[l])
        for i, (name, (c, cmap)) in enumerate(colour.items()):
            ax = axes[i, j]; ax.set_facecolor(SURFACE)
            ax.scatter(Z[sub, 0], Z[sub, 1], c=c[sub], cmap=cmap, s=3, linewidths=0, alpha=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#e1e0d9")
            if i == 0:
                ax.set_title(f"layer {l}  (PC1+2: {ev:.0%} var)", fontsize=9, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(name, fontsize=9, color=MUTED)
    fig.suptitle(f"{a.run}: residual stream at path tokens, 2-D PCA per layer", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = CKPT / a.run / "resid_pca.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print("saved", out)

    # supervised linear view: ridge-fit the 2-D projection that best predicts
    # (cos theta, sin theta); if the ring is linearly embedded this is a circle
    from probe import ridge
    rng = np.random.default_rng(1); perm = rng.permutation(len(P)); tr, te = perm[: len(P) // 2], perm[len(P) // 2:]
    Y = torch.from_numpy(np.stack([np.cos(theta), np.sin(theta)], 1).astype(np.float32)).to(device)
    fig, axes = plt.subplots(2, len(layers), figsize=(2.6 * len(layers), 5.4)); fig.patch.set_facecolor(SURFACE)
    for j, l in enumerate(layers):
        X = feats[l].to(device); X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
        pred, _ = ridge(X[tr], Y[tr], X[te], lam=1.0)
        pred = pred.cpu().numpy(); Yte = Y[te].cpu().numpy()
        r2c = 1 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()
        for i, (c, cmap, nm) in enumerate([(theta[te], "twilight", "theta"), (tg["remain"][te, 0], "Blues", "remaining")]):
            ax = axes[i, j]; ax.set_facecolor(SURFACE)
            ax.scatter(pred[:, 0], pred[:, 1], c=c, cmap=cmap, s=3, linewidths=0, alpha=0.8)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_color("#e1e0d9")
            if i == 0:
                ax.set_title(f"layer {l}  (R2 {r2c:.2f})", fontsize=9, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(f"coloured by {nm}", fontsize=9, color=MUTED)
        print(f"layer {l}: linear R2 for (cos,sin) of ring angle = {r2c:.3f}")
    fig.suptitle(f"{a.run}: ridge projection of the residual onto (cos, sin) of the ring angle, held-out tokens", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = CKPT / a.run / "resid_ring_projection.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print("saved", out)
    # how much of position is in a low-dimensional linear subspace per layer
    for l in layers:
        X = feats[l] - feats[l].mean(0)
        _, S, _ = torch.linalg.svd(X, full_matrices=False)
        cum = (S ** 2).cumsum(0) / (S ** 2).sum()
        print(f"layer {l}: dims for 90% var = {int((cum < 0.9).sum()) + 1}, 99% = {int((cum < 0.99).sum()) + 1}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="single_ring_flat_L6_d256")
    ap.add_argument("--n_seq", type=int, default=1500)
    ap.add_argument("--n_points", type=int, default=6000)
    main(ap.parse_args())
