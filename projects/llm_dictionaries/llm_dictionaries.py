"""Dictionaries on a text LLM (GPT-2 small): top-k SAE (hinge codes) vs
spline autoencoder (univariate cubic codes) vs tensor-spline autoencoder
(cubic surfaces on pairs of directions), at matched encoder parameters, under
two objectives:
    recon : reconstruct the residual stream after block L
    delta : predict what block L+1 writes, resid[L+1] - resid[L], from resid[L]
Metrics on held-out text: explained variance of the objective, L0, dead
features, and LOSS RECOVERED when the dictionary's output is spliced back
into the model:
    recon : replace resid[L] by its reconstruction; baseline = mean ablation
    delta : replace resid[L+1] by resid[L] + predicted write; baseline = skip block L+1
    loss_recovered = (CE_baseline - CE_dict) / (CE_baseline - CE_clean)

    HF_HOME=/var/tmp/hf_cache python projects/llm_dictionaries/llm_dictionaries.py --layer 6 --n_train_tokens 500000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", "/var/tmp/hf_cache")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "group_tracking")); sys.path.insert(0, str(HERE.parent / "path_transformer")); sys.path.insert(0, str(HERE.parent / "manifold_features"))
from dictionary_compare import HingeAE, SplineAE, TensorSplineAE, train_dict  # noqa: E402

RES = HERE / "results"


def load_text_tokens(tok, n_tokens: int, seq_len: int, seed: int, split: str):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    rng = np.random.default_rng(seed); order = rng.permutation(len(ds))
    ids, buf = [], []
    for i in order:
        t = ds[int(i)]["text"].strip()
        if len(t) < 200:
            continue
        buf.extend(tok(t)["input_ids"] + [tok.eos_token_id])
        while len(buf) >= seq_len:
            ids.append(buf[:seq_len]); buf = buf[seq_len:]
        if len(ids) * seq_len >= n_tokens:
            break
    return torch.tensor(ids[: n_tokens // seq_len])


@torch.no_grad()
def collect(model, X, layer, device, batch=32):
    """Residuals after block `layer` and after block `layer+1` (float32, N*T x d)."""
    A, B = [], []
    for b in range(0, len(X), batch):
        out = model(X[b:b + batch].to(device), output_hidden_states=True)
        A.append(out.hidden_states[layer + 1].float().reshape(-1, model.config.n_embd).cpu())     # hidden_states[0] = embeddings
        B.append(out.hidden_states[layer + 2].float().reshape(-1, model.config.n_embd).cpu())
    return torch.cat(A), torch.cat(B)


@torch.no_grad()
def ce_loss(model, X, device, hook=None, batch=32):
    handle = None
    if hook is not None:
        handle = hook()
    tot, n = 0.0, 0
    for b in range(0, len(X), batch):
        xb = X[b:b + batch].to(device)
        logits = model(xb).logits[:, :-1].float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1), reduction="sum").item(); n += xb[:, 1:].numel()
    if handle is not None:
        handle.remove()
    return tot / n


def splice_hooks(model, layer, objective, fn, stats):
    """Return a callable that installs the splice and returns the handle."""
    blocks = model.transformer.h
    muX, sdX, muY, sdY = stats

    def first(out):   # block output may be a tensor or a tuple whose first element is the hidden state
        return out[0] if isinstance(out, tuple) else out

    def rebuild(out, new):
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    def install():
        if objective == "recon":
            def hook(mod, inp, out):
                h = first(out); z = ((h.float() - muX) / sdX).reshape(-1, h.shape[-1])
                rec = (fn(z) * sdY + muY).reshape(h.shape)
                return rebuild(out, rec.to(h.dtype))
            return blocks[layer].register_forward_hook(hook)
        state = {}

        def stash(mod, inp, out):
            state["h"] = first(out)

        def hook(mod, inp, out):
            h = state["h"]; z = ((h.float() - muX) / sdX).reshape(-1, h.shape[-1])
            new = h.float() + (fn(z) * sdY + muY).reshape(h.shape)
            return rebuild(out, new.to(h.dtype))
        h1 = blocks[layer].register_forward_hook(stash); h2 = blocks[layer + 1].register_forward_hook(hook)

        class Both:
            def remove(self):
                h1.remove(); h2.remove()
        return Both()
    return install


def main(a):
    device = torch.device("cuda"); torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model).to(device).eval()
    d = model.config.n_embd
    t0 = time.time()
    Xtr = load_text_tokens(tok, a.n_train_tokens, a.seq_len, a.seed, "train"); Xte = load_text_tokens(tok, a.n_eval_tokens, a.seq_len, a.seed + 1, "validation")
    Atr, Btr = collect(model, Xtr, a.layer, device); Ate, Bte = collect(model, Xte, a.layer, device)
    print(f"{a.model} layer {a.layer}: train {Atr.shape[0]} tokens, eval {Ate.shape[0]} tokens, d={d}, collected in {time.time() - t0:.0f}s", flush=True)
    clean = ce_loss(model, Xte, device)
    results = dict(model=a.model, layer=a.layer, clean_ce=clean, rows=[])
    for objective in a.objectives:
        Ytr_raw = Atr if objective == "recon" else Btr - Atr; Yte_raw = Ate if objective == "recon" else Bte - Ate
        muX, sdX = Atr.mean(0), Atr.std(0) + 1e-6; muY, sdY = Ytr_raw.mean(0), Ytr_raw.std(0) + 1e-6
        Xn = ((Atr - muX) / sdX).to(device); Yn = ((Ytr_raw - muY) / sdY).to(device)
        Xn_te = ((Ate - muX) / sdX).to(device); Yn_te = ((Yte_raw - muY) / sdY).to(device)
        stats = (muX.to(device), sdX.to(device), muY.to(device), sdY.to(device))
        # baselines for loss recovered
        if objective == "recon":
            base = ce_loss(model, Xte, device, splice_hooks(model, a.layer, objective, lambda z: torch.zeros_like(z), stats))   # mean ablation
        else:
            base = ce_loss(model, Xte, device, splice_hooks(model, a.layer, objective, lambda z, s=stats: (-s[2] / s[3]).expand_as(z), stats))   # skip the block's write
        print(f"objective {objective}: clean CE {clean:.4f}, baseline CE {base:.4f}", flush=True)
        for fam, m, k in a.dicts:
            if fam == "sae":
                net = HingeAE(d, m, k, d).to(device); npar = net.enc.weight.numel() + net.enc.bias.numel()
            elif fam == "spae":
                net = SplineAE(d, m, k, G=a.G).to(device); npar = net.enc.weight.numel() + net.enc.bias.numel() + net.coef.numel()
            else:
                net = TensorSplineAE(d, m, k, G=a.G2).to(device); npar = net.enc_params()
            t1 = time.time(); net, ev_tr, l0 = train_dict(net, Xn, Yn, steps=a.steps, lr=a.lr, batch=a.batch)
            with torch.no_grad():
                rec, h = net(Xn_te[:20000]); ev = 1 - ((rec - Yn_te[:20000]) ** 2).sum() / ((Yn_te[:20000] - Yn_te[:20000].mean(0)) ** 2).sum()
                dead = float(((h != 0).sum(0) == 0).float().mean())
            spliced = ce_loss(model, Xte, device, splice_hooks(model, a.layer, objective, lambda z, net=net: net(z)[0], stats))
            rec_frac = (base - spliced) / max(base - clean, 1e-9)
            row = dict(objective=objective, family=fam, m=m, k=k, enc_params=int(npar), ev=float(ev), l0=float(l0), dead=dead, spliced_ce=spliced, loss_recovered=float(rec_frac), seconds=round(time.time() - t1))
            results["rows"].append(row); print(json.dumps(row), flush=True)
            results["baselines"] = results.get("baselines", {}); results["baselines"][objective] = base
    RES.mkdir(exist_ok=True); out = RES / (a.out or f"{a.model.replace('/', '_')}_L{a.layer}.json"); out.write_text(json.dumps(results, indent=1)); print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--objectives", nargs="+", default=["recon", "delta"])
    ap.add_argument("--dicts", nargs="+", type=lambda s: (s.split(",")[0], int(s.split(",")[1]), int(s.split(",")[2])),
                    default=[("sae", 768, 32), ("spae", 750, 32), ("tsae", 384, 32), ("sae", 3072, 32), ("spae", 3000, 32), ("tsae", 1536, 32)],
                    help="family,m,k triples; defaults are matched at ~0.6M and ~2.4M encoder params")
    ap.add_argument("--G", type=int, default=8); ap.add_argument("--G2", type=int, default=6)
    ap.add_argument("--n_train_tokens", type=int, default=500000); ap.add_argument("--n_eval_tokens", type=int, default=60000); ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=4096); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
    main(ap.parse_args())
