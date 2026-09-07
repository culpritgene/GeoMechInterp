"""Pairwise (cross-token) probes: read the composition law from two tokens.

For every generator step t >= 2 of a Cayley walk the pair is
    A = residual at layer l at the previous generator position (state after
        t-1 generators, as the model encodes it)
    B = residual at layer 0 at the current generator position (the token
        embedding of generator g_t)
and the target is the exact irrep coordinates (and, for D_n, the sign) of the
prefix product AFTER applying g_t, i.e. the value the layer has to compute by
combining the two tokens. Composition of rotations is bilinear in the two
circles, and the sign of a composed reflection is a product of signs, so the
question is which readout family expresses that combination cheaply.

Families on the concatenation [A, B], at matched parameter counts:
    linear        ridge on [A, B]
    bilinear      low-rank bilinear form  y = sum_r (a_r . A)(b_r . B)  (exact second order)
    relu+linear   m hinges on [A, B]
    spline1L      m additive cubic splines on [A, B]   (first order across tokens)
    tspline       k learned projection pairs (u = a.A, v = b.B) each with a
                  tensor-product cubic B-spline surface phi(u, v)             (second order)
    sae+linear    top-k SAEs on A and on B, ridge on the concatenated codes

    python projects/group_tracking/probe_pair.py --run group_D36_L4_d64_long --layers 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "path_transformer")); sys.path.insert(0, str(HERE.parent / "manifold_features"))
from models import bspline_basis  # noqa: E402
from probe import CKPT, ReLUProbe, Spline1LProbe, r2, ridge, train_sae, train_supervised  # noqa: E402
from probe_group import GroupData, GPT, GPTConfig, n_params  # noqa: E402


# ---------------------------------------------------------------- collection
@torch.no_grad()
def collect_pairs(run: str, n_seq: int, layer: int, device):
    ck = torch.load(CKPT / run / "best.pt", map_location=device)
    gd = GroupData(ck["args"]["data"]); G = gd.group
    model = GPT(GPTConfig(**ck["cfg"])).to(device); model.load_state_dict(ck["model"]); model.eval()
    A, B, el, gen_tok, seqs = [], [], [], [], []
    X, idx = gd.tensors("test", n_seq)
    for b in range(0, len(X), 512):
        xb = X[b:b + 512].to(device)
        _, resid = model(xb, return_resid=True)
        for r, i in enumerate(idx[b:b + 512]):
            k = int(gd.k[i]); pos = gd.generator_positions(i)             # positions of g_1..g_k
            if k < 2:
                continue
            A.append(resid[layer][r, pos[:-1]].float().cpu())            # state after g_1..g_{t-1}
            B.append(resid[0][r, pos[1:]].float().cpu())                 # embedding of g_t
            el.append(gd.prefix[i, 2:k + 1]); gen_tok.append(gd.tokens[i, pos[1:]]); seqs.append(np.full(k - 1, i))
    A = torch.cat(A); B = torch.cat(B); el = np.concatenate(el); gen_tok = np.concatenate(gen_tok); seqs = np.concatenate(seqs)
    tg = {name: G.coords[el][:, cols].astype(np.float32) for name, cols in G.coord_groups.items()}
    return A, B, tg, gen_tok, seqs, G


# ---------------------------------------------------------------- families
class BilinearProbe(nn.Module):
    def __init__(self, dA, dB, rank, out):
        super().__init__(); self.dA = dA
        self.a = nn.Linear(dA, rank * out, bias=False); self.b = nn.Linear(dB, rank * out, bias=False)
        self.rank, self.out = rank, out; self.bias = nn.Parameter(torch.zeros(out))
        self.lin = nn.Linear(dA + dB, out)   # plus a linear term so the family contains the ridge probe

    def forward(self, x):
        A, B = x[:, :self.dA], x[:, self.dA:]
        u = self.a(A).view(-1, self.out, self.rank); v = self.b(B).view(-1, self.out, self.rank)
        return (u * v).sum(-1) + self.bias + self.lin(x)


class TensorSplineProbe(nn.Module):
    """k projection pairs, each with a tensor-product cubic B-spline surface."""
    def __init__(self, dA, dB, k, out, G=6, degree=3, lo=-3.0, hi=3.0):
        super().__init__(); self.dA, self.k, self.deg = dA, k, degree
        self.pa = nn.Linear(dA, k); self.pb = nn.Linear(dB, k)
        step = (hi - lo) / G
        self.register_buffer("grid", torch.linspace(lo - degree * step, hi + degree * step, G + 2 * degree + 1))
        self.lo, self.hi = lo, hi; nb = G + degree
        self.coef = nn.Parameter(torch.randn(k, out, nb, nb) * 0.1); self.bias = nn.Parameter(torch.zeros(out))

    def surfaces(self, x):
        A, B = x[:, :self.dA], x[:, self.dA:]
        u = self.pa(A).clamp(self.lo, self.hi - 1e-6); v = self.pb(B).clamp(self.lo, self.hi - 1e-6)
        return u, v

    def forward(self, x):
        u, v = self.surfaces(x)
        bu = bspline_basis(u, self.grid, self.deg); bv = bspline_basis(v, self.grid, self.deg)   # (N, k, nb)
        return torch.einsum("nki,nkj,koij->no", bu, bv, self.coef) + self.bias

    @torch.no_grad()
    def surface_grid(self, n=61, device="cpu"):
        g = torch.linspace(self.lo, self.hi - 1e-6, n, device=device)
        bu = bspline_basis(g[:, None].expand(n, self.k), self.grid, self.deg)               # (n, k, nb)
        phi = torch.einsum("aki,bkj,koij->koab", bu, bu, self.coef)                           # (k, out, n, n)
        return g.cpu().numpy(), phi.cpu().numpy()


# ---------------------------------------------------------------- main
def run_family(name, net, Xtr, ytr, Xva, yva, Xte, yte, steps, lr):
    net = train_supervised(net, Xtr, ytr, Xva, yva, steps=steps, lr=lr)
    with torch.no_grad():
        pred = net(Xte).cpu().numpy()
    return net, pred


def main(a):
    device = torch.device("cuda")
    out_rows, surfaces = [], {}
    for layer in a.layers:
        A, B, tg, gen_tok, seqs = collect_pairs(a.run, a.n_seq, layer, device)[:5]
        useq = np.unique(seqs); rng = np.random.default_rng(a.seed); rng.shuffle(useq)
        tr_s = set(useq[: int(0.8 * len(useq))]); va_s = set(useq[int(0.8 * len(useq)): int(0.9 * len(useq))])
        tr = np.array([s in tr_s for s in seqs]); va = np.array([s in va_s for s in seqs]); te = ~(tr | va)
        X = torch.cat([A, B], 1).to(device)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
        dA = A.shape[1]; dB = B.shape[1]
        Xtr, Xva, Xte = X[tr][: a.n_train], X[va], X[te]
        strata = {"all": np.ones(te.sum(), bool)}
        if a.run.split("_")[1].startswith("D"):
            refl = np.array([t >= 4 + 72 + 6 for t in gen_tok])   # D36: generator tokens r^{+-1..3} come first, then s, sr
            strata["rotation_step"] = ~refl[te]; strata["reflection_step"] = refl[te]
        print(f"layer {layer}: {int(tr.sum())} train / {int(te.sum())} test pairs, dA={dA} dB={dB}", flush=True)
        for target in a.targets:
            if target not in tg:
                continue
            y = torch.from_numpy(tg[target]).to(device); ytr, yva, yte = y[tr][: a.n_train], y[va], y[te]; out = y.shape[1]
            preds = {}
            with torch.no_grad():
                p, npar = ridge(Xtr, ytr, Xte); preds[("linear", "-", npar)] = p.cpu().numpy()
            for rk in a.bilinear_ranks:
                net = BilinearProbe(dA, dB, rk, out).to(device); net, p = run_family("bilinear", net, Xtr, ytr, Xva, yva, Xte, yte, a.steps, 3e-3)
                preds[("bilinear", rk, n_params(net))] = p
            for m in a.widths:
                net = ReLUProbe(dA + dB, m, out).to(device); net, p = run_family("relu", net, Xtr, ytr, Xva, yva, Xte, yte, a.steps, 3e-3)
                preds[("relu+linear", m, n_params(net))] = p
                net = Spline1LProbe(dA + dB, m, out).to(device); net, p = run_family("spline1L", net, Xtr, ytr, Xva, yva, Xte, yte, a.steps, 3e-3)
                preds[("spline1L", m, n_params(net))] = p
            for k in a.pairs:
                net = TensorSplineProbe(dA, dB, k, out, G=a.G).to(device); net, p = run_family("tspline", net, Xtr, ytr, Xva, yva, Xte, yte, a.steps, 3e-3)
                preds[("tspline", k, n_params(net))] = p
                if k == 1:
                    g, phi = net.surface_grid(device=device)
                    with torch.no_grad():
                        u, v = net.surfaces(Xte[:4000])
                    surfaces[f"L{layer}_{target}"] = dict(grid=g.tolist(), phi=phi[0].tolist(), u=u[:, 0].cpu().numpy().tolist(), v=v[:, 0].cpu().numpy().tolist(),
                                                          y=yte[:4000].cpu().numpy().tolist())
            for (m, k) in a.sae:
                saeA, _, _ = train_sae(Xtr[:, :dA], m, k, steps=a.steps); saeB, _, _ = train_sae(Xtr[:, dA:], m, k, steps=a.steps)
                with torch.no_grad():
                    Htr = torch.cat([saeA.encode(Xtr[:, :dA]), saeB.encode(Xtr[:, dA:])], 1); Hte = torch.cat([saeA.encode(Xte[:, :dA]), saeB.encode(Xte[:, dA:])], 1)
                    p, npar = ridge(Htr, ytr, Hte)
                preds[("sae+linear", f"m{m}k{k}", n_params(saeA.enc) + n_params(saeB.enc) + npar)] = p.cpu().numpy()
            yte_np = yte.cpu().numpy()
            print(f"  target {target}:")
            for (fam, size, npar), p in preds.items():
                row = dict(run=a.run, layer=layer, target=target, probe=fam, size=str(size), n_params=int(npar))
                for sname, smask in strata.items():
                    row[f"r2_{sname}"] = r2(p[smask], yte_np[smask]) if smask.sum() > 100 else float("nan")
                row["r2"] = row["r2_all"]; out_rows.append(row)
                print(f"    {fam:12s} {str(size):8s} params={npar:7d}  " + "  ".join(f"{s}={row[f'r2_{s}']:.3f}" for s in strata), flush=True)
    res = dict(run=a.run, rows=out_rows, surfaces=surfaces, args=vars(a))
    out = Path(a.out or CKPT / a.run / "probes_pair.json"); out.write_text(json.dumps(res)); print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="group_D36_L4_d64_long")
    ap.add_argument("--layers", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--targets", nargs="+", default=["irrep", "f1", "eps"])
    ap.add_argument("--widths", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--pairs", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--bilinear_ranks", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--sae", nargs="+", type=lambda s: tuple(int(x) for x in s.split(",")), default=[(256, 8), (1024, 32)])
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--n_seq", type=int, default=10000)
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
