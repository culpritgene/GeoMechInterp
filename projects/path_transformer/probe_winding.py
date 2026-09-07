"""Probe the residual stream of a winding-class path transformer (Rung 1) for
the circle x counter features, with the probe families and matched sizes of
probe.py, scored per stratum.

For every path-token position of held-out sequences (flat format: one token
per cell; the residual at the position of cell t predicts cell t+1) we record
the residual after each block and the ground truth, with theta the ring angle
of the current cell (cell_theta from gen_winding.py, cut at theta = 0), k_t
the winding count so far (cut-edge signs summed along the path prefix, k_0 =
0), Theta_t = theta + 2 pi k_t the unrolled angle and k the class of the path:
    T1    : (cos theta, sin theta) ++ one-hot(k_t)        dim 2 + (2K+1)
    T2    : remaining cover displacement to the goal on the class-k path,
            (Theta_goal - Theta_t) / 2 pi in laps          dim 1 (continuous)
    T3    : slow circle (cos(Theta_t/3), sin(Theta_t/3)), embeds the cover
            without a branch cut for K = 1                dim 2
    circ  : (cos theta, sin theta) alone;  kt : one-hot(k_t) alone
    Theta : Theta_t / 2 pi (scalar helix coordinate, chart-dependent)
    pos, goal, next, dir, remain : as in probe.py
Strata (masks over tokens, evaluated on the held-out tokens of probes trained
on all tokens):
    all; pre_cross / mid_cross / post_cross : before the first / between /
    after the last cut crossing of the path (paths with >= 1 crossing);
    no_cross : paths that never cross; near_cut / far_cut : angular distance
    of the current cell to the cut below / above --near_rad; class_-1 / 0 / 1.
Probe families (probe.py): linear ridge, top-k SAE + ridge, ReLU hinges +
linear, cubic splines + linear (Spline1LProbe, no tanh, grid [-3, 3]) and the
2-layer cubic KAN on a learned projection.  `r2` is the within-stratum R^2
averaged over the target dimensions that have non-zero variance inside the
stratum (it can be very negative where a dimension is nearly constant, e.g.
cos theta near the cut); `r2_glob` = 1 - MSE_stratum / Var_global is
comparable across strata; per-dimension R^2 is stored for the 'all' stratum.

    python projects/path_transformer/probe_winding.py --run single_ring_winding_flat_L3_d64 --layers 0 2 3
Writes <run>/probes_winding.json.
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
from model import GPT, GPTConfig  # noqa: E402
from probe import CKPT, ReLUProbe, Spline1LProbe, SplineProbe, n_params, ridge, train_sae, train_supervised  # noqa: E402
from winding_data import TWO_PI, WindingData  # noqa: E402

# matched sizes of probe.py (__main__) and a tiny grid for smoke runs
SIZES = {
    "full": {"sae": [(256, 8), (256, 32), (1024, 8), (1024, 32), (1024, 128), (4096, 32)],
             "relu": [2, 4, 8, 16, 32, 64, 128, 512],
             "spline1L": [2, 4, 8, 16, 32, 64, 128],
             "spline": [(2, 4, 8), (4, 8, 8), (8, 16, 16), (16, 32, 16)]},
    "tiny": {"sae": [(256, 32)], "relu": [4, 32], "spline1L": [4, 32], "spline": [(4, 8, 8)]},
}
STRATA = ["all", "pre_cross", "mid_cross", "post_cross", "no_cross", "near_cut", "far_cut"]


# ---------------------------------------------------------------- scoring
def r2_dims(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-dimension R^2; NaN for dimensions with (near) zero variance."""
    ss = ((y - pred) ** 2).sum(0); st = ((y - y.mean(0)) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1 - ss / st
    out[st <= 1e-8 * max(len(y), 1)] = np.nan
    return out


def r2_mean(pred: np.ndarray, y: np.ndarray) -> float:
    d = r2_dims(pred, y)
    return float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")


# ---------------------------------------------------------------- activations
@torch.no_grad()
def collect(run: str, n_seq: int, split: str, device, near_rad: float = 0.5):
    """Residuals, targets and strata for the path tokens of `n_seq` sequences
    of `split`.  Returns (feats {layer: (T, d)}, targets {name: (T, dim)},
    strata {name: bool (T,)}, WindingData)."""
    ck = torch.load(CKPT / run / "best.pt", map_location=device)
    a = ck["args"]
    assert a.get("data") == "winding", f"{run} was not trained with --data winding"
    wd = WindingData(a["shape"], a["fmt"])
    assert wd.fmt == "flat", "probe assumes one token per cell"
    model = GPT(GPTConfig(**ck["cfg"])).to(device); model.load_state_dict(ck["model"]); model.eval()
    X, idx = wd.tensors(split, n_seq)
    lo, hi = wd.cell_center.min(0), wd.cell_center.max(0)
    ctr = (lo + hi) / 2; half = (hi - lo).max() / 2
    C = (wd.cell_center - ctr) / half
    K = wd.K; nK = 2 * K + 1; theta = wd.cell_theta
    feats = {l: [] for l in range(len(model.blocks) + 1)}
    tg = {k: [] for k in ("pos", "goal", "next", "remain", "T1", "T2", "T3", "circ", "kt", "Theta", "seq", "t")}
    st = {k: [] for k in STRATA[1:] + ["cls"]}
    for b in range(0, len(X), 256):
        xb = X[b:b + 256].to(device)
        _, resid = model(xb, return_resid=True)
        for r, i in enumerate(idx[b:b + 256]):
            L = int(wd.lengths[i]); cells = wd.cell_ids[i, :L]
            p0 = wd.prompt_len; pos = slice(p0, p0 + L - 1)          # residual at cell t predicts cell t+1
            for l in feats:
                feats[l].append(resid[l][r, pos].float().cpu())
            signs = wd.step_signs(cells)                                # step t: cells[t] -> cells[t+1]
            kt = np.r_[0, np.cumsum(signs)[:-1]]                        # winding on arrival at cells[t], t < L-1
            k_final = int(signs.sum())
            th = theta[cells[:-1]]; Th = th + TWO_PI * kt
            Th_goal = theta[cells[-1]] + TWO_PI * k_final
            onehot = np.eye(nK)[np.clip(kt + K, 0, nK - 1)]
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(wd.cell_center[cells], axis=0), axis=1))]
            tg["pos"].append(C[cells[:-1]]); tg["next"].append(C[cells[1:]])
            tg["goal"].append(np.repeat(C[cells[-1]][None], L - 1, 0))
            tg["remain"].append(((cum[-1] - cum[:-1]) / cum[-1])[:, None])
            tg["circ"].append(np.stack([np.cos(th), np.sin(th)], 1)); tg["kt"].append(onehot)
            tg["T1"].append(np.concatenate([np.cos(th)[:, None], np.sin(th)[:, None], onehot], 1))
            tg["T2"].append(((Th_goal - Th) / TWO_PI)[:, None])
            tg["T3"].append(np.stack([np.cos(Th / 3), np.sin(Th / 3)], 1))
            tg["Theta"].append((Th / TWO_PI)[:, None])
            tg["seq"].append(np.full(L - 1, i)); tg["t"].append(np.arange(L - 1))
            t = np.arange(L - 1); cross = np.where(signs != 0)[0]
            if len(cross):
                pre = t <= cross[0]; post = t > cross[-1]; mid = ~(pre | post); noc = np.zeros(L - 1, bool)
            else:
                pre = post = mid = np.zeros(L - 1, bool); noc = np.ones(L - 1, bool)
            near = np.minimum(th, TWO_PI - th) < near_rad
            st["pre_cross"].append(pre); st["mid_cross"].append(mid); st["post_cross"].append(post); st["no_cross"].append(noc)
            st["near_cut"].append(near); st["far_cut"].append(~near); st["cls"].append(np.full(L - 1, k_final))
    feats = {l: torch.cat(v) for l, v in feats.items()}
    tg = {k: np.concatenate(v).astype(np.float32) for k, v in tg.items()}
    step = tg["next"] - tg["pos"]
    tg["dir"] = (step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    st = {k: np.concatenate(v) for k, v in st.items()}
    cls = st.pop("cls")
    strata = {"all": np.ones(len(cls), bool), **st, **{f"class_{k}": cls == k for k in range(-K, K + 1)}}
    return feats, tg, strata, wd


# ---------------------------------------------------------------- probes
def run_probes_strata(feats, tg, strata, layer, target, device, n_train, sizes, steps, seed=0, min_n=200):
    """probe.run_probes with per-stratum scoring: probes are trained on all
    tokens (split by sequence 0.8/0.1/0.1) and R^2 is reported on the held-out
    tokens of every stratum with at least `min_n` of them."""
    torch.manual_seed(seed)
    X = feats[layer].to(device); y = torch.from_numpy(tg[target]).to(device)
    seqs = tg["seq"]; useq = np.unique(seqs); rng = np.random.default_rng(seed); rng.shuffle(useq)
    tr_seq = set(useq[: int(0.8 * len(useq))]); va_seq = set(useq[int(0.8 * len(useq)): int(0.9 * len(useq))])
    tr = torch.from_numpy(np.array([s in tr_seq for s in seqs])).to(device)
    va = torch.from_numpy(np.array([s in va_seq for s in seqs])).to(device)
    te = ~(tr | va)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    Xtr, ytr, Xva, yva, Xte, yte = Xn[tr][:n_train], y[tr][:n_train], Xn[va], y[va], Xn[te], y[te]
    out = y.shape[1]; d = X.shape[1]
    yte_np = yte.cpu().numpy(); te_idx = te.cpu().numpy()
    preds: list[tuple[dict, np.ndarray]] = []
    with torch.no_grad():
        pred, npar = ridge(Xtr, ytr, Xte)
    preds.append((dict(probe="linear", size="-", n_params=npar), pred.cpu().numpy()))
    for m, k in sizes["sae"]:
        sae, ev, l0 = train_sae(Xtr, m, k, steps=steps)
        with torch.no_grad():
            pred, npar = ridge(sae.encode(Xtr), ytr, sae.encode(Xte))
        preds.append((dict(probe="sae+linear", size=f"m{m}k{k}", n_params=n_params(sae.enc) + npar, sae_ev=ev, sae_l0=l0), pred.cpu().numpy()))
    for m in sizes["relu"]:
        net = train_supervised(ReLUProbe(d, m, out).to(device), Xtr, ytr, Xva, yva, steps=steps)
        with torch.no_grad():
            preds.append((dict(probe="relu+linear", size=m, n_params=n_params(net)), net(Xte).cpu().numpy()))
    for m in sizes["spline1L"]:
        net = train_supervised(Spline1LProbe(d, m, out).to(device), Xtr, ytr, Xva, yva, steps=steps, lr=3e-3)
        with torch.no_grad():
            preds.append((dict(probe="spline1L", size=m, n_params=n_params(net)), net(Xte).cpu().numpy()))
    for k, h, G in sizes["spline"]:
        net = train_supervised(SplineProbe(d, k, out, h=h, G=G).to(device), Xtr, ytr, Xva, yva, steps=steps, lr=1e-2)
        with torch.no_grad():
            preds.append((dict(probe="spline2L", size=f"k{k}h{h}G{G}", n_params=n_params(net)), net(Xte).cpu().numpy()))
    rows = []
    var_all = yte_np.var(0)                                   # global test variance per dimension
    for info, p in preds:
        for sname, smask in strata.items():
            m = smask[te_idx]
            if m.sum() < min_n:
                continue
            # r2: within-stratum (SS_tot from the stratum's own variance; very negative when a target
            # dimension is nearly constant inside the stratum, e.g. cos theta near the cut)
            # r2_glob: 1 - MSE_stratum / Var_global, comparable across strata
            mse = ((p[m] - yte_np[m]) ** 2).mean(0)
            row = dict(**info, layer=layer, target=target, stratum=sname, n=int(m.sum()), r2=r2_mean(p[m], yte_np[m]),
                       r2_glob=float(1 - (mse / np.maximum(var_all, 1e-12)).mean()))
            if sname == "all":
                row["r2_dims"] = [None if not np.isfinite(v) else float(v) for v in r2_dims(p[m], yte_np[m])]
            rows.append(row)
    return rows


def print_table(rows, layer, target, strata_names):
    print(f"layer {layer} {target}:" + "".join(f"{s:>12s}" for s in strata_names), flush=True)
    seen = []
    for r in rows:
        key = (r["probe"], str(r["size"]))
        if r["layer"] == layer and r["target"] == target and key not in seen:
            seen.append(key)
    for probe, size in seen:
        vals = {r["stratum"]: r["r2"] for r in rows if r["layer"] == layer and r["target"] == target and r["probe"] == probe and str(r["size"]) == size}
        print(f"  {probe + ' ' + size:22s}" + "".join(f"{vals[s]:12.3f}" if s in vals and np.isfinite(vals[s]) else " " * 12 for s in strata_names), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stratified circle x counter probes of a winding-class path transformer.")
    ap.add_argument("--run", default="single_ring_winding_flat_L3_d64")
    ap.add_argument("--layers", nargs="+", default=["all"], help="residual layers (0 = embeddings) or 'all'")
    ap.add_argument("--targets", nargs="+", default=["T1", "T2", "T3", "pos", "dir"])
    ap.add_argument("--split", default="test", help="sequence split to probe (test or test_goalheld)")
    ap.add_argument("--n_seq", type=int, default=3000)
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--grid", default="full", choices=sorted(SIZES))
    ap.add_argument("--near_rad", type=float, default=0.5, help="near-cut stratum: angular distance to the cut (rad)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = torch.device("cuda")
    feats, tg, strata, wd = collect(a.run, a.n_seq, a.split, device, a.near_rad)
    layers = list(range(len(feats))) if a.layers == ["all"] else [int(l) for l in a.layers]
    print(f"{a.run}: {len(tg['pos'])} path tokens from {a.n_seq} {a.split} sequences, d={feats[0].shape[1]}, K={wd.K}", flush=True)
    print("strata sizes " + json.dumps({k: int(v.sum()) for k, v in strata.items()}), flush=True)
    rows = []
    strata_names = list(strata)
    for layer in layers:
        for target in a.targets:
            rs = run_probes_strata(feats, tg, strata, layer, target, device, a.n_train, SIZES[a.grid], a.steps, a.seed)
            rows += rs
            print_table(rs, layer, target, strata_names)
    out = Path(a.out or CKPT / a.run / "probes_winding.json")
    out.write_text(json.dumps(dict(run=a.run, split=a.split, n_seq=a.n_seq, n_train=a.n_train, steps=a.steps, grid=a.grid, near_rad=a.near_rad,
                                   strata_sizes={k: int(v.sum()) for k, v in strata.items()}, rows=rows), indent=1))
    print("saved", out)
