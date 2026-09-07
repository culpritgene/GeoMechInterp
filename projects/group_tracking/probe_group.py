"""Probe the residual stream of a trained group-tracking GPT for the exact
group state, comparing probe families at matched parameter counts.

Residuals are read at every generator position (the token g_t, whose prefix
product e_0 g_1 .. g_t is the target) and at the final SEP position (the
answer position; with tied embeddings the element is linear there by
construction, so it is reported separately) of held-out sequences.

Targets (from the exact prefix product):
    irrep    : all irrep coordinates (D_n: (c, s, eps c, eps s) per frequency;
               Z_n: (c, s) per frequency; torus: both circles)      R^2 grid
    f1, f2.. : one frequency at a time                                R^2 grid
    eps      : sign character (D_n only, +-1)                         R^2 grid (+ sign accuracy) and a logistic probe
    harm     : lowest Fourier harmonic on the rotation axis that the model's
               element-token embedding does NOT contain (frequencies detected
               by FFT of the embedding rows; reported)                R^2 grid
    element  : the element id, linear softmax probe (cross-entropy)   accuracy

Families (imported from projects/path_transformer/probe.py): ridge, top-k SAE
+ ridge, ReLU hinges + linear, cubic splines (Spline1LProbe) + linear, at the
sizes of probe.py.  Output: probes.json in the run directory and a table of
the best R^2 per family under parameter budgets.

    python projects/group_tracking/probe_group.py --run group_Z36_L2_d64 --layers 0 1 2
    python projects/group_tracking/probe_group.py --run group_Z36_L2_d64 --tiny   # smoke grid
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
sys.path.insert(0, str(HERE.parent / "path_transformer"))
from model import GPT, GPTConfig  # noqa: E402
from probe import ReLUProbe, Spline1LProbe, r2, ridge, train_sae, train_supervised  # noqa: E402
from group_data import GroupData  # noqa: E402
from groups import Group  # noqa: E402

CKPT = Path("/var/tmp/geomech_ckpt")
BUDGETS = [1000, 2500, 5000, 10000, 40000, 300000]
FULL_SIZES = {"sae": [(256, 8), (1024, 8), (1024, 32), (4096, 32)],
              "relu": [2, 4, 8, 16, 32, 64, 128], "spline1L": [2, 4, 8, 16, 32, 64, 128]}
TINY_SIZES = {"sae": [(256, 8)], "relu": [2, 8], "spline1L": [2, 8]}


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- model frequencies
def detect_frequencies(emb: np.ndarray, G: Group, cover: float = 0.9) -> dict:
    """FFT of the element-token embedding rows along each cyclic axis of the
    group.  For every axis: power fraction per frequency 1..period/2, the
    smallest set of frequencies holding `cover` of the AC power (`used`) and
    the lowest harmonic outside it (`lowest_unused`, None if all are used)."""
    out = {}
    emb = emb - emb.mean(0, keepdims=True)
    for ax in G.axes:
        n_other = int(ax.other.max()) + 1
        E = np.zeros((n_other, ax.period, emb.shape[1]))
        E[ax.other, ax.index] = emb
        P = (np.abs(np.fft.rfft(E, axis=1)) ** 2).sum((0, 2))[1:]        # f = 1 .. period//2
        frac = P / max(P.sum(), 1e-12)
        order = np.argsort(-frac)
        used, acc = [], 0.0
        for j in order:
            used.append(int(j) + 1); acc += frac[j]
            if acc >= cover:
                break
        unused = [f for f in range(1, ax.period // 2 + 1) if f not in used]
        out[ax.name] = dict(period=ax.period, power_frac={int(j) + 1: round(float(frac[j]), 4) for j in order[:12]},
                            used=sorted(used), lowest_unused=min(unused) if unused else None)
    return out


# ---------------------------------------------------------------- activations
@torch.no_grad()
def collect(run: str, n_seq: int, splits: list[str], device, min_step: int = 1):
    ck = torch.load(CKPT / run / "best.pt", map_location=device)
    gd = GroupData(ck["args"]["data"]); G = gd.group
    model = GPT(GPTConfig(**ck["cfg"])).to(device); model.load_state_dict(ck["model"]); model.eval()
    freqs = detect_frequencies(model.tok.weight[4:4 + G.order].float().cpu().numpy(), G)
    feats = {l: [] for l in range(len(model.blocks) + 1)}
    tg = {"element": [], "seq": [], "t": [], "final": []}
    for split in splits:
        X, idx = gd.tensors(split, n_seq)
        for b in range(0, len(X), 512):
            xb = X[b:b + 512].to(device)
            _, resid = model(xb, return_resid=True)
            for r, i in enumerate(idx[b:b + 512]):
                k = int(gd.k[i])
                pos = np.r_[gd.generator_positions(i)[min_step - 1:], gd.final_sep_position(i)]
                steps = np.r_[np.arange(min_step, k + 1), k]
                for l in feats:
                    feats[l].append(resid[l][r, pos].float().cpu())
                tg["element"].append(gd.prefix[i, steps]); tg["seq"].append(np.full(len(pos), i)); tg["t"].append(steps)
                tg["final"].append(np.r_[np.zeros(len(pos) - 1, dtype=np.int64), 1])
    feats = {l: torch.cat(v) for l, v in feats.items()}
    tg = {k: np.concatenate(v) for k, v in tg.items()}
    el = tg["element"]
    for name, cols in G.coord_groups.items():
        tg[name] = G.coords[el][:, cols].astype(np.float32)
    ax = G.axes[0]
    f = freqs[ax.name]["lowest_unused"]
    if f is not None:
        th = 2 * np.pi * f * ax.index[el] / ax.period
        tg["harm"] = np.stack([np.cos(th), np.sin(th)], 1).astype(np.float32)
    return feats, tg, gd, freqs, ck


# ---------------------------------------------------------------- probes
def split_by_seq(seqs: np.ndarray, seed: int, device):
    useq = np.unique(seqs); rng = np.random.default_rng(seed); rng.shuffle(useq)
    tr_seq = set(useq[: int(0.8 * len(useq))].tolist()); va_seq = set(useq[int(0.8 * len(useq)): int(0.9 * len(useq))].tolist())
    tr = torch.from_numpy(np.array([s in tr_seq for s in seqs])).to(device)
    va = torch.from_numpy(np.array([s in va_seq for s in seqs])).to(device)
    return tr, va, ~(tr | va)


def standardise(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    return (X - mu) / sd


def train_classifier(Xtr, ytr, Xva, yva, Xte, yte, n_classes: int, steps: int = 3000, lr: float = 3e-3, batch: int = 4096, wd: float = 0.0):
    """Linear softmax probe trained with cross-entropy (logistic when n_classes == 2)."""
    net = nn.Linear(Xtr.shape[1], n_classes).to(Xtr.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.01)
    best, best_state = -1.0, None
    for s in range(1, steps + 1):
        bi = torch.randint(0, len(Xtr), (batch,), device=Xtr.device)
        loss = F.cross_entropy(net(Xtr[bi]), ytr[bi])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 250 == 0:
            with torch.no_grad():
                v = float((net(Xva).argmax(1) == yva).float().mean())
            if v > best:
                best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        acc = float((net(Xte).argmax(1) == yte).float().mean())
    return acc, n_params(net)


def run_grid(X: torch.Tensor, y: torch.Tensor, seqs: np.ndarray, device, n_train: int, sizes: dict, steps: int, seed: int = 0,
             binary: bool = False) -> list[dict]:
    """Ridge, top-k SAE + ridge, ReLU + linear and cubic-spline + linear probes
    for a regression target y; held-out R^2 (and sign accuracy if binary)."""
    torch.manual_seed(seed)
    tr, va, te = split_by_seq(seqs, seed, device)
    Xn = standardise(X, tr)
    Xtr, ytr, Xva, yva, Xte, yte = Xn[tr][:n_train], y[tr][:n_train], Xn[va], y[va], Xn[te], y[te]
    d, out = X.shape[1], y.shape[1]
    yte_np = yte.cpu().numpy()

    def score(pred):
        pred = pred.cpu().numpy()
        r = dict(r2=r2(pred, yte_np))
        if binary:
            r["acc"] = float((np.sign(pred) == np.sign(yte_np)).mean())
        return r

    res = []
    with torch.no_grad():
        pred, npar = ridge(Xtr, ytr, Xte)
    res.append(dict(probe="linear", size="-", n_params=npar, **score(pred)))
    for m, k in sizes.get("sae", []):
        sae, ev, l0 = train_sae(Xtr, m, k, steps=steps)
        with torch.no_grad():
            Htr, Hte = sae.encode(Xtr), sae.encode(Xte)
            pred, npar = ridge(Htr, ytr, Hte)
        res.append(dict(probe="sae+linear", size=f"m{m}k{k}", n_params=n_params(sae.enc) + npar, sae_ev=ev, sae_l0=l0, **score(pred)))
    for m in sizes.get("relu", []):
        net = train_supervised(ReLUProbe(d, m, out).to(device), Xtr, ytr, Xva, yva, steps=steps, lr=3e-3)
        with torch.no_grad():
            res.append(dict(probe="relu+linear", size=m, n_params=n_params(net), **score(net(Xte))))
    for m in sizes.get("spline1L", []):
        net = train_supervised(Spline1LProbe(d, m, out).to(device), Xtr, ytr, Xva, yva, steps=steps, lr=3e-3)
        with torch.no_grad():
            res.append(dict(probe="spline1L", size=m, n_params=n_params(net), **score(net(Xte))))
    return res


def classifier_rows(X, labels: np.ndarray, seqs, device, n_train, steps, seed=0) -> dict:
    tr, va, te = split_by_seq(seqs, seed, device)
    Xn = standardise(X, tr)
    y = torch.from_numpy(labels.astype(np.int64)).to(device)
    n_classes = int(labels.max()) + 1
    acc, npar = train_classifier(Xn[tr][:n_train], y[tr][:n_train], Xn[va], y[va], Xn[te], y[te], n_classes, steps=steps)
    return dict(probe="linear_softmax", size="-", n_params=npar, acc=acc, chance=float(np.bincount(labels).max() / len(labels)))


# ---------------------------------------------------------------- reporting
def summarize(rows: list[dict], budgets=BUDGETS) -> str:
    """Best R^2 per family at or below each parameter budget, per (cell, layer, target)."""
    fams = ["linear", "sae+linear", "relu+linear", "spline1L"]
    keys = sorted({(r["cell"], r["layer"], r["target"]) for r in rows if "r2" in r}, key=lambda t: (t[0], t[1], t[2]))
    lines = ["| cell | layer | target | probe | " + " | ".join(f"<= {b}" for b in budgets) + " |", "|---|---|---|---|" + "---|" * len(budgets)]
    for cell, layer, target in keys:
        sub = [r for r in rows if r.get("cell") == cell and r["layer"] == layer and r["target"] == target and "r2" in r]
        for fam in fams:
            fr = [r for r in sub if r["probe"] == fam]
            if not fr:
                continue
            cells = []
            for b in budgets:
                v = [r["r2"] for r in fr if r["n_params"] <= b]
                cells.append(f"{max(v):.3f}" if v else "-")
            lines.append(f"| {cell} | {layer} | {target} | {fam} | " + " | ".join(cells) + " |")
    cls = [r for r in rows if r.get("probe") in ("linear_softmax", "logistic")]
    if cls:
        lines += ["", "| cell | layer | target | probe | accuracy | chance |", "|---|---|---|---|---|---|"]
        for r in cls:
            lines.append(f"| {r['cell']} | {r['layer']} | {r['target']} | {r['probe']} | {r['acc']:.4f} | {r['chance']:.3f} |")
    return "\n".join(lines)


def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes = TINY_SIZES if a.tiny else FULL_SIZES
    steps = 300 if a.tiny and a.steps == 3000 else a.steps
    feats, tg, gd, freqs, ck = collect(a.run, a.n_seq, a.splits, device, a.min_step)
    G = gd.group
    print(f"{a.run}: {len(tg['seq'])} positions from {len(np.unique(tg['seq']))} sequences, d={feats[0].shape[1]}, layers 0..{len(feats) - 1}")
    print("model frequencies (FFT of element embeddings):", json.dumps(freqs))
    targets = [t for t in a.targets if t in tg] if a.targets else [t for t in ["irrep"] + [f"f{f}" for f in range(1, G.n_freq + 1)] + ["eps", "harm"] if t in tg]
    if "harm" in tg:
        print(f"harmonic target: frequency {freqs[G.axes[0].name]['lowest_unused']} on axis {G.axes[0].name}")
    layers = a.layers if a.layers is not None else list(range(len(feats)))
    rows = []
    for cell in a.cells:
        sel = tg["final"] == (1 if cell == "final" else 0)
        seqs = tg["seq"][sel]
        for layer in layers:
            X = feats[layer][torch.from_numpy(sel)].to(device)
            r = classifier_rows(X, tg["element"][sel], seqs, device, a.n_train, steps)
            r.update(cell=cell, layer=layer, target="element"); rows.append(r); print(json.dumps(r), flush=True)
            if "eps" in tg:
                r = classifier_rows(X, (tg["eps"][sel, 0] > 0).astype(np.int64), seqs, device, a.n_train, steps)
                r.update(cell=cell, layer=layer, target="eps", probe="logistic"); rows.append(r); print(json.dumps(r), flush=True)
            for target in targets:
                y = torch.from_numpy(tg[target][sel]).to(device)
                rs = run_grid(X, y, seqs, device, a.n_train, sizes, steps, binary=(target == "eps"))
                for r in rs:
                    r.update(cell=cell, layer=layer, target=target); rows.append(r); print(json.dumps(r), flush=True)
    out = Path(a.out or CKPT / a.run / "probes.json")
    out.write_text(json.dumps(dict(run=a.run, group=G.name, model_frequencies=freqs, n_positions=int(len(tg["seq"])), rows=rows), indent=1))
    print(summarize(rows)); print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="group_Z36_L2_d64")
    ap.add_argument("--layers", nargs="+", type=int, default=None, help="residual layers (default: all)")
    ap.add_argument("--cells", nargs="+", default=["gen", "final"], choices=["gen", "final"])
    ap.add_argument("--targets", nargs="+", default=None, help="subset of irrep f1.. eps harm (default: all available)")
    ap.add_argument("--splits", nargs="+", default=["test"], help="held-out splits to probe")
    ap.add_argument("--n_seq", type=int, default=10000, help="sequences per split")
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--min_step", type=int, default=1, help="first generator step t to include in the gen cell")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--tiny", action="store_true", help="smoke grid: 2 sizes per family, 300 steps")
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
