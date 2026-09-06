"""Probe the residual stream of a trained path transformer for geometric
features, comparing feature extractors at matched parameter counts.

For every path-token position of held-out sequences we record the residual
stream after each block and the ground truth
    pos   : 3-D centre of the current cell (normalised to [-1,1])
    goal  : 3-D centre of the goal cell
    next  : 3-D centre of the next cell on the path (the model's decision)
    remain: normalised remaining polyline length to the goal

Probes (all read out `target` from the residual vector r in R^d):
    linear   : W r + b                                       (ridge regression)
    sae+lin  : top-k SAE trained unsupervised on r, then ridge on its codes
    relu+lin : same encoder trained supervised for the target (dense hinges)
    spline1L : relu+lin with every hinge replaced by a cubic spline: m learned
               directions, one univariate cubic B-spline each, summed
    spline2L : learned projection d -> k, then a two-layer cubic-spline (KAN)
               network k -> h -> out (supervised)
Reported: R^2 on held-out sequences and parameter count.

    python projects/path_transformer/probe.py --run single_ring_flat_L6_d256 --layers 0 2 4 6
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "manifold_features"))
from model import GPT, GPTConfig  # noqa: E402
from models import KANLayer  # noqa: E402
from paths_data import PAD, PathData  # noqa: E402

CKPT = Path("/var/tmp/geomech_ckpt")


# ---------------------------------------------------------------- activations
@torch.no_grad()
def collect(run: str, n_seq: int, split: str, device):
    ck = torch.load(CKPT / run / "best.pt", map_location=device)
    a = ck["args"]
    pd = PathData(a["shape"], a["fmt"])
    model = GPT(GPTConfig(**ck["cfg"])).to(device); model.load_state_dict(ck["model"]); model.eval()
    X, idx = pd.tensors(split, n_seq)
    lo, hi = pd.cell_center.min(0), pd.cell_center.max(0)
    ctr = (lo + hi) / 2; half = (hi - lo).max() / 2
    C = (pd.cell_center - ctr) / half
    feats, tg = {l: [] for l in range(len(model.blocks) + 1)}, {"pos": [], "goal": [], "next": [], "remain": [], "seq": [], "t": []}
    assert pd.fmt == "flat", "probe currently assumes one token per cell"
    for b in range(0, len(X), 256):
        xb = X[b:b + 256].to(device)
        _, resid = model(xb, return_resid=True)
        for r, i in enumerate(idx[b:b + 256]):
            L = int(pd.lengths[i]); cells = pd.cell_ids[i, :L]
            # path tokens sit at positions prompt_len .. prompt_len+L-1; the
            # residual at position p predicts the token at p+1, so use the
            # positions of cells[0..L-2] whose "next" is cells[1..L-1]
            p0 = pd.prompt_len
            pos = slice(p0, p0 + L - 1)
            for l in feats:
                feats[l].append(resid[l][r, pos].float().cpu())
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pd.cell_center[cells], axis=0), axis=1))]
            tg["pos"].append(C[cells[:-1]]); tg["next"].append(C[cells[1:]])
            tg["goal"].append(np.repeat(C[cells[-1]][None], L - 1, 0))
            tg["remain"].append(((cum[-1] - cum[:-1]) / cum[-1])[:, None])
            tg["seq"].append(np.full(L - 1, i)); tg["t"].append(np.arange(L - 1))
    feats = {l: torch.cat(v) for l, v in feats.items()}
    tg = {k: np.concatenate(v).astype(np.float32) for k, v in tg.items()}
    return feats, tg, pd


# ---------------------------------------------------------------- probes
def r2(pred, y):
    ss = ((y - pred) ** 2).sum(0); st = ((y - y.mean(0)) ** 2).sum(0)
    return float((1 - ss / st).mean())


def ridge(Xtr, ytr, Xte, lam=1e-2):
    Xa = torch.cat([Xtr, torch.ones(len(Xtr), 1, device=Xtr.device)], 1)
    Xb = torch.cat([Xte, torch.ones(len(Xte), 1, device=Xte.device)], 1)
    A = Xa.T @ Xa + lam * torch.eye(Xa.shape[1], device=Xa.device)
    W = torch.linalg.solve(A, Xa.T @ ytr)
    return Xb @ W, Xa.shape[1] * ytr.shape[1]


class SAE(nn.Module):
    """Top-k sparse autoencoder: exactly k active ReLU features per token."""
    def __init__(self, d, m, k):
        super().__init__()
        self.enc = nn.Linear(d, m); self.dec = nn.Linear(m, d, bias=True); self.k = k

    def encode(self, x):
        h = F.relu(self.enc(x))
        idx = h.topk(self.k, dim=1).indices
        return torch.zeros_like(h).scatter(1, idx, h.gather(1, idx))

    def forward(self, x):
        h = self.encode(x); return self.dec(h), h


def train_sae(X, m, k, steps=3000, lr=1e-3, batch=4096):
    d = X.shape[1]; sae = SAE(d, m, k).to(X.device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for s in range(steps):
        xb = X[torch.randint(0, len(X), (batch,), device=X.device)]
        rec, h = sae(xb)
        loss = F.mse_loss(rec, xb)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        rec, h = sae(X[:8192])
        ev = 1 - ((rec - X[:8192]) ** 2).sum() / ((X[:8192] - X[:8192].mean(0)) ** 2).sum()
        l0 = (h > 0).float().sum(1).mean()
    return sae, float(ev), float(l0)


def train_supervised(net, Xtr, ytr, Xva, yva, steps=3000, lr=3e-3, batch=4096, reg=None):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.01)
    best, best_state = -1e9, None
    for s in range(1, steps + 1):
        bi = torch.randint(0, len(Xtr), (batch,), device=Xtr.device)
        loss = F.mse_loss(net(Xtr[bi]), ytr[bi]) + (reg(net) if reg else 0.0)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 250 == 0:
            with torch.no_grad():
                v = r2(net(Xva).cpu().numpy(), yva.cpu().numpy())
            if v > best:
                best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net


class ReLUProbe(nn.Module):
    def __init__(self, d, m, out):
        super().__init__(); self.enc = nn.Linear(d, m); self.dec = nn.Linear(m, out)

    def forward(self, x):
        return self.dec(F.relu(self.enc(x)))


class SplineProbe(nn.Module):
    """r -> P r in R^k (learned projection, tanh-squashed onto the grid)
    -> cubic KAN k -> h -> tanh -> cubic KAN h -> out (two spline layers so
    that interactions between projected directions can be represented)."""
    def __init__(self, d, k, out, h=None, G=8, degree=3):
        super().__init__(); h = h or k
        self.proj = nn.Linear(d, k); self.kan1 = KANLayer(k, h, G=G, k=degree); self.kan2 = KANLayer(h, out, G=G, k=degree)

    def forward(self, x):
        return self.kan2(torch.tanh(self.kan1(torch.tanh(self.proj(x)))))


class Spline1LProbe(nn.Module):
    """Exact analogue of ReLUProbe with hinges replaced by cubic splines:
    out = sum_j f_j(a_j . r + b_j), f_j a cubic B-spline on G intervals.
    Same number of projection directions m as the ReLU probe's hidden units."""
    def __init__(self, d, m, out, G=8, degree=3):
        super().__init__(); self.proj = nn.Linear(d, m); self.kan = KANLayer(m, out, G=G, k=degree)

    def forward(self, x):
        return self.kan(torch.tanh(self.proj(x)))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def run_probes(feats, tg, layer, target, device, n_train, sizes, steps, seed=0):
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
    res = []
    with torch.no_grad():
        pred, npar = ridge(Xtr, ytr, Xte)
    res.append(dict(probe="linear", size="-", n_params=npar, r2=r2(pred.cpu().numpy(), yte.cpu().numpy())))
    for m, k in sizes["sae"]:
        sae, ev, l0 = train_sae(Xtr, m, k, steps=steps)
        with torch.no_grad():
            Htr, Hte = sae.encode(Xtr), sae.encode(Xte)
            pred, npar = ridge(Htr, ytr, Hte)
        res.append(dict(probe="sae+linear", size=f"m{m}k{k}", n_params=n_params(sae.enc) + npar, r2=r2(pred.cpu().numpy(), yte.cpu().numpy()), sae_ev=ev, sae_l0=l0))
    for m in sizes["relu"]:
        net = train_supervised(ReLUProbe(d, m, out).to(device), Xtr, ytr, Xva, yva, steps=steps)
        with torch.no_grad():
            res.append(dict(probe="relu+linear", size=m, n_params=n_params(net), r2=r2(net(Xte).cpu().numpy(), yte.cpu().numpy())))
    for m in sizes["spline1L"]:
        net = train_supervised(Spline1LProbe(d, m, out).to(device), Xtr, ytr, Xva, yva, steps=steps, lr=1e-2)
        with torch.no_grad():
            res.append(dict(probe="spline1L", size=m, n_params=n_params(net), r2=r2(net(Xte).cpu().numpy(), yte.cpu().numpy())))
    for k, h, G in sizes["spline"]:
        net = train_supervised(SplineProbe(d, k, out, h=h, G=G).to(device), Xtr, ytr, Xva, yva, steps=steps, lr=1e-2)
        with torch.no_grad():
            res.append(dict(probe="spline2L", size=f"k{k}h{h}G{G}", n_params=n_params(net), r2=r2(net(Xte).cpu().numpy(), yte.cpu().numpy())))
    for r in res:
        r.update(layer=layer, target=target)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="single_ring_flat_L6_d256")
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 2, 4, 6])
    ap.add_argument("--targets", nargs="+", default=["pos", "goal", "next", "remain"])
    ap.add_argument("--n_seq", type=int, default=3000)
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = torch.device("cuda")
    feats, tg, pd = collect(a.run, a.n_seq, "test", device)
    print(f"{a.run}: {len(tg['pos'])} path tokens from {a.n_seq} sequences, d={feats[0].shape[1]}", flush=True)
    sizes = {"sae": [(256, 8), (256, 32), (1024, 8), (1024, 32), (1024, 128), (4096, 32)],
             "relu": [2, 4, 8, 16, 32, 64, 128, 512],
             "spline1L": [2, 4, 8, 16, 32, 64, 128],
             "spline": [(2, 4, 8), (4, 8, 8), (8, 16, 16), (16, 32, 16)]}
    rows = []
    for layer in a.layers:
        for target in a.targets:
            rs = run_probes(feats, tg, layer, target, device, a.n_train, sizes, a.steps)
            rows += rs
            for r in rs:
                print(json.dumps(r), flush=True)
    out = Path(a.out or CKPT / a.run / "probes.json")
    out.write_text(json.dumps(rows, indent=1))
    print("saved", out)
