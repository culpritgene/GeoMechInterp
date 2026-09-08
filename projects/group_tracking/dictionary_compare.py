"""Unsupervised dictionaries compared at matched parameters: top-k SAE (hinge
codes) versus a top-k spline autoencoder (SpAE) whose codes are univariate
cubic-spline responses on learned directions, both read out by the same ridge
regression. The question: is the SAE's failure on polynomial features due to
the linear readout over first-order codes (then the SpAE should read them)?

    SAE  : h_j = ReLU(w_j . x + b_j), top-k, linear decoder, MSE
    SpAE : h_j = phi_j(w_j . x + b_j), phi_j a cubic B-spline on [-3, 3] with
           G intervals, top-k on |h|, linear decoder, MSE
Encoder parameters: SAE d*m + m; SpAE d*m + m + m*(G+3).

Kinds:
    --kind null   synthetic linear circle + nuisance (null_control.make_null)
    --kind group  residuals of a group-tracking model (probe_group.collect)
    --kind path   residuals of a path model (probe.collect)

    python projects/group_tracking/dictionary_compare.py --kind group --run group_D36_L4_d64_long --layers 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "path_transformer")); sys.path.insert(0, str(HERE.parent / "manifold_features"))
from models import bspline_basis  # noqa: E402
from probe import CKPT, Spline1LProbe, r2, ridge, train_sae, train_supervised  # noqa: E402


class SplineAE(nn.Module):
    def __init__(self, d, m, k, G=8, degree=3, lo=-3.0, hi=3.0):
        super().__init__()
        self.enc = nn.Linear(d, m); self.dec = nn.Linear(m, d); self.k, self.deg, self.lo, self.hi = k, degree, lo, hi
        step = (hi - lo) / G
        self.register_buffer("grid", torch.linspace(lo - degree * step, hi + degree * step, G + 2 * degree + 1))
        self.coef = nn.Parameter(torch.randn(m, G + degree) * 0.3)

    def encode(self, x):
        u = self.enc(x).clamp(self.lo, self.hi - 1e-6)
        h = (bspline_basis(u, self.grid, self.deg) * self.coef).sum(-1)
        if self.k and self.k < h.shape[1]:
            idx = h.abs().topk(self.k, dim=1).indices
            h = torch.zeros_like(h).scatter(1, idx, h.gather(1, idx))
        return h

    def forward(self, x):
        h = self.encode(x); return self.dec(h), h


class TensorSplineAE(nn.Module):
    """Second-order dictionary: code j is a tensor-product cubic B-spline surface
    phi_j(a_j . x, b_j . x) on a pair of learned directions of the same vector;
    top-k on |h|, linear decoder.  Encoder params: 2*d*m + 2m + m*(G+3)^2."""
    def __init__(self, d, m, k, G=6, degree=3, lo=-3.0, hi=3.0, d_out=None):
        super().__init__()
        self.pa = nn.Linear(d, m); self.pb = nn.Linear(d, m); self.dec = nn.Linear(m, d_out or d)
        self.k, self.deg, self.lo, self.hi = k, degree, lo, hi
        step = (hi - lo) / G
        self.register_buffer("grid", torch.linspace(lo - degree * step, hi + degree * step, G + 2 * degree + 1))
        self.coef = nn.Parameter(torch.randn(m, G + degree, G + degree) * 0.2)

    def encode(self, x):
        u = self.pa(x).clamp(self.lo, self.hi - 1e-6); v = self.pb(x).clamp(self.lo, self.hi - 1e-6)
        bu = bspline_basis(u, self.grid, self.deg); bv = bspline_basis(v, self.grid, self.deg)          # (N, m, nb)
        h = torch.einsum("nmi,nmj,mij->nm", bu, bv, self.coef)
        if self.k and self.k < h.shape[1]:
            idx = h.abs().topk(self.k, dim=1).indices; h = torch.zeros_like(h).scatter(1, idx, h.gather(1, idx))
        return h

    def forward(self, x):
        h = self.encode(x); return self.dec(h), h

    def enc_params(self):
        return self.pa.weight.numel() + self.pa.bias.numel() + self.pb.weight.numel() + self.pb.bias.numel() + self.coef.numel()


class HingeAE(nn.Module):
    """Top-k SAE with a decoder of arbitrary output width (transcoder-capable)."""
    def __init__(self, d, m, k, d_out=None):
        super().__init__(); self.enc = nn.Linear(d, m); self.dec = nn.Linear(m, d_out or d); self.k = k

    def encode(self, x):
        h = F.relu(self.enc(x))
        if self.k and self.k < h.shape[1]:
            idx = h.topk(self.k, dim=1).indices; h = torch.zeros_like(h).scatter(1, idx, h.gather(1, idx))
        return h

    def forward(self, x):
        h = self.encode(x); return self.dec(h), h


@torch.no_grad()
def resample_dead(net, dead: torch.Tensor, X: torch.Tensor):
    """Re-initialise codes that never fired: encoder rows pointed at random
    inputs (unit norm), biases zero, spline coefficients fresh. Applied
    identically to every family."""
    idx = torch.where(dead)[0]
    if len(idx) == 0:
        return
    ex = X[torch.randint(0, len(X), (len(idx),), device=X.device)]; ex = ex / (ex.norm(dim=1, keepdim=True) + 1e-6)
    for lin in [m for n, m in net.named_children() if n in ("enc", "pa", "pb")]:
        lin.weight[idx] = ex * 0.5 if lin is getattr(net, "enc", None) or lin is getattr(net, "pa", None) else torch.randn_like(ex) * (0.5 / ex.shape[1] ** 0.5)
        lin.bias[idx] = 0.0
    if hasattr(net, "coef"):
        net.coef[idx] = torch.randn_like(net.coef[idx]) * (0.3 if net.coef.dim() == 2 else 0.2)
    net.dec.weight[:, idx] = torch.randn_like(net.dec.weight[:, idx]) * 0.01


def train_dict(net, X, Y, steps=3000, lr=1e-3, batch=4096, resample_every=1000):
    """Train a dictionary to predict Y from X (Y = X: autoencoder; Y = later residual: transcoder),
    resampling codes that have not fired in the last `resample_every` steps (not in the final quarter)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    fired = None
    for step in range(1, steps + 1):
        bi = torch.randint(0, len(X), (batch,), device=X.device)
        rec, h = net(X[bi]); loss = F.mse_loss(rec, Y[bi]); opt.zero_grad(); loss.backward(); opt.step()
        f = (h != 0).any(0); fired = f if fired is None else (fired | f)
        if resample_every and step % resample_every == 0 and step <= 0.75 * steps:
            resample_dead(net, ~fired, X); fired = None
    with torch.no_grad():
        rec, h = net(X[:8192]); ev = 1 - ((rec - Y[:8192]) ** 2).sum() / ((Y[:8192] - Y[:8192].mean(0)) ** 2).sum(); l0 = (h != 0).float().sum(1).mean()
    return net, float(ev), float(l0)


def train_spae(X, m, k, G=8, steps=3000, lr=1e-3, batch=4096, Y=None):
    net = SplineAE(X.shape[1], m, k, G=G).to(X.device)
    if Y is not None and Y.shape[1] != X.shape[1]:
        net.dec = nn.Linear(m, Y.shape[1]).to(X.device)
    return train_dict(net, X, X if Y is None else Y, steps, lr, batch)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def load(a, device):
    if a.kind == "null":
        from groups import make_group
        from null_control import make_null
        G = make_group(a.group); X, tg, info = make_null(G, a.freqs, a.d, a.n, 40, 0.1, a.seed)
        return {0: torch.from_numpy(X)}, tg, np.arange(len(X)), f"null_{a.group}_d{a.d}"
    if a.kind == "group":
        from probe_group import collect
        feats, tg, gd, freqs, ck = collect(a.run, a.n_seq, ["test"], device)
        return feats, tg, tg["seq"], a.run
    from probe import collect
    feats, tg, pd = collect(a.run, a.n_seq, "test", device)
    return feats, tg, tg["seq"], a.run


def main(a):
    device = torch.device("cuda")
    feats, tg, seqs, name = load(a, device)
    useq = np.unique(seqs); rng = np.random.default_rng(a.seed); rng.shuffle(useq)
    tr_s = set(useq[: int(0.8 * len(useq))]); va_s = set(useq[int(0.8 * len(useq)): int(0.9 * len(useq))])
    tr = np.array([s in tr_s for s in seqs]); va = np.array([s in va_s for s in seqs]); te = ~(tr | va)
    targets = [t for t in a.targets if t in tg] or [t for t in tg if t not in ("seq", "t", "final", "element")]
    rows = []
    for layer in (a.layers if a.kind != "null" else [0]):
        X = feats[layer].to(device); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
        Xtr, Xva, Xte = X[tr][: a.n_train], X[va], X[te]; d = X.shape[1]
        dicts = []
        Ytr = None
        if a.objective in ("next_layer", "delta"):
            # transcoder: predict the residual at --out_layer (next_layer) or what the layers in between
            # WRITE, i.e. resid[out] - resid[layer] (delta), from the layer-l residual
            Yall = feats[a.out_layer].to(device)
            if a.objective == "delta":
                Yall = Yall - feats[layer].to(device)
            Yall = (Yall - Yall[tr].mean(0)) / (Yall[tr].std(0) + 1e-6); Ytr = Yall[tr][: a.n_train]
        tag = "" if Ytr is None else (f"->L{a.out_layer}" if a.objective == "next_layer" else f"->dL{a.out_layer}")
        for m, k in a.sae:
            sae, ev, l0 = (train_sae(Xtr, m, k, steps=a.steps) if Ytr is None else train_dict(HingeAE(d, m, k, Ytr.shape[1]).to(device), Xtr, Ytr, a.steps))
            dicts.append((f"sae{tag}+linear", f"m{m}k{k}", sae, ev, l0, n_params(sae.enc)))
        for m, k in a.spae:
            spae, ev, l0 = train_spae(Xtr, m, k, G=a.G, steps=a.steps, Y=Ytr); dicts.append((f"spae{tag}+linear", f"m{m}k{k}G{a.G}", spae, ev, l0, n_params(spae.enc) + spae.coef.numel()))
        for m, k in a.tsae:
            net = TensorSplineAE(d, m, k, G=a.G2, d_out=(Ytr.shape[1] if Ytr is not None else None)).to(device)
            tsae, ev, l0 = train_dict(net, Xtr, Xtr if Ytr is None else Ytr, a.steps); dicts.append((f"tsae{tag}+linear", f"m{m}k{k}G{a.G2}", tsae, ev, l0, tsae.enc_params()))
        print(f"{name} layer {layer}: d={d}, dictionaries: " + ", ".join(f"{f} {s} EV={ev:.3f} L0={l0:.0f} enc_params={npar}" for f, s, _, ev, l0, npar in dicts), flush=True)
        for target in targets:
            y = torch.from_numpy(np.asarray(tg[target], dtype=np.float32)).to(device); ytr, yva, yte = y[tr][: a.n_train], y[va], y[te]; yte_np = yte.cpu().numpy()
            with torch.no_grad():
                p, npar = ridge(Xtr, ytr, Xte)
            rows.append(dict(name=name, layer=layer, target=target, probe="linear", size="-", n_params=npar, r2=r2(p.cpu().numpy(), yte_np)))
            for fam, size, net, ev, l0, npar in dicts:
                with torch.no_grad():
                    Htr, Hte = net.encode(Xtr), net.encode(Xte); p, rp = ridge(Htr, ytr, Hte)
                rows.append(dict(name=name, layer=layer, target=target, probe=fam, size=size, n_params=npar + rp, ev=ev, l0=l0, r2=r2(p.cpu().numpy(), yte_np)))
            for m in a.sup_widths:
                net = train_supervised(Spline1LProbe(d, m, y.shape[1]).to(device), Xtr, ytr, Xva, yva, steps=a.steps, lr=3e-3)
                with torch.no_grad():
                    rows.append(dict(name=name, layer=layer, target=target, probe="spline1L (supervised)", size=m, n_params=n_params(net), r2=r2(net(Xte).cpu().numpy(), yte_np)))
            print(f"  target {target:8s}: " + " | ".join(f"{r['probe']} {r['size']}: {r['r2']:.3f}" for r in rows if r["layer"] == layer and r["target"] == target), flush=True)
    out = Path(a.out or (CKPT / f"dict_{name}.json")); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(name=name, kind=a.kind, rows=rows, args=vars(a)), indent=1)); print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="group", choices=["null", "group", "path"])
    ap.add_argument("--run", default="group_D36_L4_d64_long")
    ap.add_argument("--group", default="D36"); ap.add_argument("--freqs", nargs="+", type=int, default=[1]); ap.add_argument("--d", type=int, default=64); ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--layers", nargs="+", type=int, default=[2])
    ap.add_argument("--targets", nargs="+", default=[])
    ap.add_argument("--sae", nargs="+", type=lambda s: tuple(int(x) for x in s.split(",")), default=[(256, 8), (256, 32), (1024, 32)])
    ap.add_argument("--spae", nargs="+", type=lambda s: tuple(int(x) for x in s.split(",")), default=[(256, 8), (256, 32), (1024, 32)])
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--tsae", nargs="+", type=lambda s: tuple(int(x) for x in s.split(",")), default=[(64, 8), (128, 16), (256, 32)], help="tensor-spline dictionary sizes (m, k); [] to skip")
    ap.add_argument("--G2", type=int, default=6, help="grid intervals per axis for tensor-spline codes")
    ap.add_argument("--sup_widths", nargs="+", type=int, default=[8])
    ap.add_argument("--n_seq", type=int, default=6000); ap.add_argument("--n_train", type=int, default=40000); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", default="recon", choices=["recon", "next_layer", "delta"], help="recon: autoencoder; next_layer: transcoder predicting the residual at --out_layer; delta: transcoder predicting resid[out_layer] - resid[layer]")
    ap.add_argument("--out_layer", type=int, default=4)
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
