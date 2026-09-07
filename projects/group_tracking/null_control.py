"""Probe-capacity ceiling for the group targets (synthetic-linear null control).

The exact ground-truth latent (irrep coordinates of a uniform random element
at the chosen frequencies; for D_n the entries (c, s, eps c, eps s) of the 2-D
irreps, WITHOUT the sign itself) is embedded by a random orthogonal map into
R^d together with `--n_nuisance` (default 40) standard Gaussian nuisance
dimensions and isotropic Gaussian noise of std `--noise`.  The identical probe
grid of probe_group.py is then run on targets that are exact functions of the
latent:
    irrep  : the embedded coordinates themselves (linear; sanity ceiling)
    eps    : sign character, a quadratic form of two embedded circles (D_n)
    harm<f>: cos/sin of the lowest harmonics NOT embedded (Chebyshev
             polynomials of the stored circle), f = max(freqs)+1, +2
Whatever spline-vs-hinge gap appears here is what the probes alone can
produce; a transformer result is meaningful only in excess of it.

    python projects/group_tracking/null_control.py --group D36 --d 64 --freqs 1
    python projects/group_tracking/null_control.py --group Z36 --d 64 --freqs 1 --tiny
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from groups import make_group  # noqa: E402
from probe_group import BUDGETS, FULL_SIZES, TINY_SIZES, run_grid, summarize  # noqa: E402

OUT = Path("/var/tmp/geomech_ckpt/null_control")


def make_null(G, freqs: list[int], d: int, n: int, n_nuisance: int, noise: float, seed: int):
    """Returns X (n, d) float32 and a dict of targets, plus the latent."""
    rng = np.random.default_rng(seed)
    el = rng.integers(0, G.order, size=n)
    cols = [c for f in freqs for name in G.coord_groups for c in G.coord_groups[name]
            if name in (f"f{f}",) or name.endswith(f"_f{f}")]
    cols = sorted(set(cols))
    z = G.coords[el][:, cols]
    D = z.shape[1]
    assert d >= D + n_nuisance, f"d={d} too small for {D} latent + {n_nuisance} nuisance dims"
    latent = np.concatenate([z, rng.standard_normal((n, n_nuisance))], 1)
    Q, _ = np.linalg.qr(rng.standard_normal((d, D + n_nuisance)))         # orthonormal columns
    X = latent @ Q.T + noise * rng.standard_normal((n, d))
    tg = {"irrep": z.astype(np.float32)}
    if G.sign is not None:
        tg["eps"] = G.sign[el][:, None].astype(np.float32)
    ax = G.axes[0]
    for f in (max(freqs) + 1, max(freqs) + 2):
        if f <= ax.period // 2:
            th = 2 * np.pi * f * ax.index[el] / ax.period
            tg[f"harm{f}"] = np.stack([np.cos(th), np.sin(th)], 1).astype(np.float32)
    return X.astype(np.float32), tg, dict(latent_dim=D, coord_names=[G.coord_names[c] for c in cols])


def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = make_group(a.group, n_freq=max(max(a.freqs) + 2, a.n_freq) if a.group.startswith(("Z", "D")) else a.n_freq, p_reflect=a.p_reflect)
    X, tg, info = make_null(G, a.freqs, a.d, a.n, a.n_nuisance, a.noise, a.seed)
    print(f"null control {G.name}: latent {info['coord_names']} + {a.n_nuisance} nuisance dims -> R^{a.d}, noise {a.noise}, n={a.n}")
    sizes = TINY_SIZES if a.tiny else FULL_SIZES
    steps = 300 if a.tiny and a.steps == 3000 else a.steps
    seqs = np.arange(len(X))
    Xt = torch.from_numpy(X).to(device)
    rows = []
    for target, y in tg.items():
        rs = run_grid(Xt, torch.from_numpy(y).to(device), seqs, device, a.n_train, sizes, steps, seed=a.seed, binary=(target == "eps"))
        for r in rs:
            r.update(cell="null", layer=0, target=target); rows.append(r); print(json.dumps(r), flush=True)
    print(summarize(rows))
    # spline-vs-hinge at each budget
    print("\nspline1L minus relu+linear (best R^2 under budget):")
    print("| target | " + " | ".join(f"<= {b}" for b in BUDGETS) + " |", "\n|---|" + "---|" * len(BUDGETS))
    for target in tg:
        cells = []
        for b in BUDGETS:
            s = [r["r2"] for r in rows if r["target"] == target and r["probe"] == "spline1L" and r["n_params"] <= b]
            h = [r["r2"] for r in rows if r["target"] == target and r["probe"] == "relu+linear" and r["n_params"] <= b]
            cells.append(f"{max(s) - max(h):+.3f}" if s and h else "-")
        print(f"| {target} | " + " | ".join(cells) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(a.out or OUT / f"null_{G.name}_d{a.d}_f{'-'.join(map(str, a.freqs))}{'_tiny' if a.tiny else ''}.json")
    out.write_text(json.dumps(dict(group=G.name, d=a.d, freqs=a.freqs, n_nuisance=a.n_nuisance, noise=a.noise, **info, rows=rows), indent=1))
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="D36")
    ap.add_argument("--freqs", nargs="+", type=int, default=[1], help="frequencies whose irrep coordinates are embedded")
    ap.add_argument("--n_freq", type=int, default=3)
    ap.add_argument("--p_reflect", type=float, default=0.2)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n_nuisance", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
