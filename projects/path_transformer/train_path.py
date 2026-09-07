"""Train a path transformer on one shape and evaluate it by greedy decoding.

    python projects/path_transformer/train_path.py --shape single_ring --fmt hier \
        --n_layer 6 --d_model 256 --steps 30000

Quality bar: a model is only useful if its generated paths are valid (every
step moves between adjacent occupied cells), reach the goal, and are close to
geodesic length; those are reported as `valid`, `reached`, `success` and
`len_ratio` on held-out (start, goal) pairs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig  # noqa: E402
from paths_data import BOS, EOS, PAD, SEP, PathData  # noqa: E402

CKPT = Path("/var/tmp/geomech_ckpt")


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * step / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate_generation(model, pd: PathData, X, idx, device, max_batch=512):
    model.eval()
    prompts = X[:, :pd.prompt_len].to(device)
    max_new = pd.block_size - pd.prompt_len
    rows = []
    for b in range(0, len(prompts), max_batch):
        out = model.generate(prompts[b:b + max_batch], max_new, eos=EOS, pad=PAD).cpu().numpy()
        for r, i in zip(out, idx[b:b + max_batch]):
            L = int(pd.lengths[i]); opt = pd.cell_ids[i, :L].tolist()
            cells = pd.decode_path(r[pd.prompt_len:].tolist())
            rows.append(pd.score(cells, opt[0], opt[-1], opt))
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


@torch.no_grad()
def token_accuracy(model, X, pd, device, n=4096):
    model.eval()
    xb = X[:n].to(device)
    logits = model(xb[:, :-1])
    tgt = xb[:, 1:]
    mask = (tgt != PAD)
    mask[:, :pd.prompt_len - 1] = False
    return float(((logits.argmax(-1) == tgt) & mask).sum() / mask.sum())


def main(args):
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    if getattr(args, "data", "paths") == "winding":       # winding-class task (gen_winding.py / winding_data.py)
        from winding_data import WindingData
        pd = WindingData(args.shape, args.fmt)
    else:
        pd = PathData(args.shape, args.fmt)
    Xtr, _ = pd.tensors("train"); Xva, iva = pd.tensors("validation", args.n_eval); Xte, ite = pd.tensors("test", args.n_eval)
    print(f"{args.shape} [{args.fmt}] vocab={pd.vocab_size} block={pd.block_size} train={len(Xtr)} val={len(Xva)} cells={pd.n_cells}", flush=True)
    cfg = GPTConfig(pd.vocab_size, pd.block_size, args.n_layer, args.n_head, args.d_model, args.dropout)
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    tag = args.shape if getattr(args, "data", "paths") == "paths" else f"{args.shape}_{args.data}"
    run = CKPT / f"{tag}_{args.fmt}_L{args.n_layer}_d{args.d_model}"
    run.mkdir(parents=True, exist_ok=True)
    Xtr = Xtr.to(device)
    log = []
    best = -1.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        bi = torch.randint(0, len(Xtr), (args.batch,), device=device)
        xb = Xtr[bi]
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb[:, :-1])
        tgt = xb[:, 1:].clone()
        tgt[:, :pd.prompt_len - 1] = PAD  # no loss on the prompt
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=PAD)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            tok = token_accuracy(model, Xva, pd, device)
            gen = evaluate_generation(model, pd, Xva, iva, device)
            rec = dict(step=step, loss=float(loss), val_tok_acc=tok, **{f"val_{k}": v for k, v in gen.items()}, seconds=round(time.time() - t0))
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if gen["success"] >= best:
                best = gen["success"]
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "args": vars(args), "step": step}, run / "best.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "args": vars(args), "step": args.steps}, run / "last.pt")
    model.load_state_dict(torch.load(run / "best.pt")["model"])
    test = evaluate_generation(model, pd, Xte, ite, device)
    summary = dict(shape=args.shape, fmt=args.fmt, n_params=n_params, n_layer=args.n_layer, d_model=args.d_model,
                   steps=args.steps, best_val_success=best, **{f"test_{k}": v for k, v in test.items()}, log=log)
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    print("TEST", json.dumps({k: v for k, v in summary.items() if k != "log"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="single_ring")
    ap.add_argument("--fmt", default="hier", choices=["hier", "flat"])
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="paths", choices=["paths", "winding"], help="paths: PathData on <shape>.npz; winding: WindingData on <shape>_wind.npz (run dir gets a _winding tag)")
    main(ap.parse_args())
