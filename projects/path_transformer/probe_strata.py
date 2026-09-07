"""Stratified probing: does the spline-vs-hinge gap depend on local geometry?

For each path token we attach, from the tori fitted to the OBJ mesh:
    absK     |Gaussian curvature| of the nearest torus at the current cell
    contact  distance from the current cell to the nearest *other* torus
             surface (small near link crossings / touching regions; inf for
             single_ring)
Targets:
    dir  : unit step direction (next cell centre - current cell centre)
    pos  : current cell centre
Probes are trained on all tokens (split by sequence) and scored on held-out
tokens per stratum (|K| terciles, contact < 0.35, far from everything).

    python projects/path_transformer/probe_strata.py --run single_ring_flat_L6_d256 --layers 2 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "manifold_features"))
from data import fit_tori_from_obj, torus_frames  # noqa: E402
from probe import CKPT, ReLUProbe, SAE, Spline1LProbe, collect, r2, ridge, train_sae, train_supervised, n_params  # noqa: E402


def local_geometry(P_raw: np.ndarray, shape: str, tori_arr=None):
    """|K| of the nearest torus and distance to the nearest other torus.
    `tori_arr` (T, 8): centre, axis, R, r rows from an analytic manifold; if
    None the tori are fitted from the shipped OBJ mesh of `shape`."""
    if tori_arr is not None:
        C, A = np.asarray(tori_arr)[:, :3], np.asarray(tori_arr)[:, 3:6]; R, r = np.asarray(tori_arr)[:, 6], np.asarray(tori_arr)[:, 7]
    else:
        C, A, R, r = torus_frames(fit_tori_from_obj(shape))
    d_all, K_all, axdist = [], [], []
    for c, a, Rj, rj in zip(C, A, R, r):
        v = P_raw - c; h = v @ a; rho = np.linalg.norm(v - np.outer(h, a), axis=1)
        d = np.sqrt((rho - Rj) ** 2 + h ** 2) - rj
        cosv = np.clip((rho - Rj) / np.maximum(np.sqrt((rho - Rj) ** 2 + h ** 2), 1e-9), -1, 1)
        d_all.append(d); K_all.append(cosv / (rj * (Rj + rj * cosv)))
        axdist.append(np.sqrt((rho - Rj) ** 2 + h ** 2))   # distance to the torus' core circle
    d_all = np.stack(d_all, 1); K_all = np.stack(K_all, 1)
    nearest = np.abs(d_all).argmin(1)
    local_geometry.nearest = nearest
    absK = np.abs(K_all[np.arange(len(P_raw)), nearest])
    if d_all.shape[1] > 1:
        others = np.abs(d_all).copy(); others[np.arange(len(P_raw)), nearest] = np.inf
        contact = others.min(1)
    else:
        contact = np.full(len(P_raw), np.inf)
    return absK, contact


def main(a):
    device = torch.device("cuda")
    feats, tg, pd = collect(a.run, a.n_seq, "test", device)
    shape = a.run.split("_flat")[0]
    lo, hi = pd.cell_center.min(0), pd.cell_center.max(0); ctr = (lo + hi) / 2; half = (hi - lo).max() / 2
    P_raw = tg["pos"] * half + ctr
    absK, contact = local_geometry(P_raw, shape, pd.extras.get("tori"))
    step = tg["next"] - tg["pos"]; dirn = step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-9)
    tg["dir"] = dirn.astype(np.float32)
    tg["sheet"] = (2.0 * (local_geometry.nearest == 0) - 1.0)[:, None].astype(np.float32)   # which torus the current cell lies on (+-1)
    q = np.quantile(absK, [1 / 3, 2 / 3])
    strata = {"all": np.ones(len(absK), bool), "lowK": absK < q[0], "midK": (absK >= q[0]) & (absK < q[1]), "highK": absK >= q[1]}
    if np.isfinite(contact).any():
        strata["near_contact"] = contact < a.contact_r
        strata["far_contact"] = contact >= a.contact_r
    print(json.dumps({k: int(v.sum()) for k, v in strata.items()}), "|K| terciles", q.round(2).tolist(), flush=True)

    seqs = tg["seq"]; useq = np.unique(seqs); rng = np.random.default_rng(0); rng.shuffle(useq)
    tr_seq = set(useq[: int(0.8 * len(useq))]); va_seq = set(useq[int(0.8 * len(useq)): int(0.9 * len(useq))])
    tr = np.array([s in tr_seq for s in seqs]); va = np.array([s in va_seq for s in seqs]); te = ~(tr | va)
    if a.train_stratum != "all":       # local probes: fit on tokens of one stratum only (scored on every stratum)
        keep = strata[a.train_stratum]; tr &= keep; va &= keep
        print(f"training probes on stratum {a.train_stratum}: {int(tr.sum())} train / {int(va.sum())} val tokens", flush=True)
    rows = []
    for layer in a.layers:
        X = feats[layer].to(device); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu) / sd
        Xtr, Xva, Xte = Xn[tr][: a.n_train], Xn[va], Xn[te]
        for target in a.targets:
            y = torch.from_numpy(tg[target]).to(device); ytr, yva, yte = y[tr][: a.n_train], y[va], y[te]
            out = y.shape[1]
            preds = {}
            with torch.no_grad():
                preds["linear"], _ = ridge(Xtr, ytr, Xte)
            for m in a.widths:
                net = train_supervised(ReLUProbe(X.shape[1], m, out).to(device), Xtr, ytr, Xva, yva, steps=a.steps)
                with torch.no_grad():
                    preds[f"relu m={m}"] = net(Xte)
                net = train_supervised(Spline1LProbe(X.shape[1], m, out).to(device), Xtr, ytr, Xva, yva, steps=a.steps, lr=a.lr_spline)
                with torch.no_grad():
                    preds[f"spline m={m}"] = net(Xte)
            sae, ev, l0 = train_sae(Xtr, 1024, 32, steps=a.steps)
            with torch.no_grad():
                preds["sae m=1024 k=32"], _ = ridge(sae.encode(Xtr), ytr, sae.encode(Xte))
            yte_np = yte.cpu().numpy(); te_idx = np.where(te)[0]
            for name, p in preds.items():
                p = p.cpu().numpy()
                for sname, smask in strata.items():
                    m = smask[te_idx]
                    if m.sum() < 200:
                        continue
                    rows.append(dict(run=a.run, layer=layer, target=target, probe=name, stratum=sname, n=int(m.sum()), r2=r2(p[m], yte_np[m])))
            print(f"layer {layer} {target}:  " + "".join(f"{s:>14s}" for s in strata), flush=True)
            for name in preds:
                vals = {r["stratum"]: r["r2"] for r in rows if r["layer"] == layer and r["target"] == target and r["probe"] == name}
                print(f"  {name:18s}" + "".join(f"{vals[s]:14.3f}" if s in vals else " " * 14 for s in strata), flush=True)
    out = CKPT / a.run / ("probes_strata.json" if a.train_stratum == "all" else f"probes_strata_{a.train_stratum}.json")
    json.dump(rows, open(out, "w"), indent=1); print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="single_ring_flat_L6_d256")
    ap.add_argument("--layers", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--targets", nargs="+", default=["dir", "pos", "sheet"])
    ap.add_argument("--widths", nargs="+", type=int, default=[8, 32, 128])
    ap.add_argument("--n_seq", type=int, default=3000)
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr_spline", type=float, default=3e-3)
    ap.add_argument("--contact_r", type=float, default=0.35)
    ap.add_argument("--train_stratum", default="all", help="all, or a stratum name (e.g. near_contact) to train the probes on that stratum only")
    main(ap.parse_args())
