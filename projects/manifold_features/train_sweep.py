"""Parameter-efficiency sweep: for each shape and model family, train models of
increasing size and record test accuracy so saturation curves can be compared.

Usage (from repo root, inside the project venv):
    python projects/manifold_features/train_sweep.py --shapes single_ring keyring_1 \
        --families sae relu1 mlp2 spline1 spline3 --seeds 0 1 --out results/sweep.csv

Training is full-batch Adam with cosine decay over --steps, stopping early once
validation accuracy has plateaued for --patience steps (after --min_steps); the
checkpoint with the best validation accuracy is evaluated on the test split.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SHAPES, load_shape  # noqa: E402
from models import default_grid, make_model, n_params  # noqa: E402

FIELDS = [
    "shape", "family", "size", "seed", "n_params", "steps", "lr",
    "val_acc", "test_acc", "test_acc_exterior", "test_acc_surface", "test_acc_interior",
    "test_acc_near", "test_acc_surface_Kneg", "test_acc_surface_Kpos", "test_acc_cavity",
    "test_loss", "train_acc", "sparsity", "best_step", "stop_step", "seconds",
]


def accuracy(logits, y):
    return float((logits.argmax(1) == y).float().mean())


def masked_acc(pred, y, mask):
    return float((pred[mask] == y[mask]).float().mean()) if mask.any() else float("nan")


def train_one(model, Xtr, ytr, Xva, yva, steps, lr, device, eval_every=100, patience=0, min_steps=0):
    """Full-batch Adam with cosine decay over `steps`.  If `patience` > 0,
    stop early once validation accuracy has not improved for `patience` steps
    (never before `min_steps`).  Returns the best-validation checkpoint."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.01)
    best_va, best_state, best_step = -1.0, None, 0
    for step in range(1, steps + 1):
        model.train()
        logits = model(Xtr)
        loss = F.cross_entropy(logits, ytr) + model.reg()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                va = accuracy(model(Xva), yva)
            if va > best_va:
                best_va, best_step = va, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif patience and step >= min_steps and step - best_step >= patience:
                break
    model.load_state_dict(best_state)
    return best_va, best_step, step


def evaluate(model, sd, device):
    Xte, yte, sdf, K, cav = sd.subset("test")
    X = torch.from_numpy(Xte).to(device); y = torch.from_numpy(yte).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = float(F.cross_entropy(logits, y))
        pred = logits.argmax(1)
    sdf = torch.from_numpy(sdf).to(device); K = torch.from_numpy(K).to(device); cav = torch.from_numpy(cav).to(device)
    out = {
        "test_acc": float((pred == y).float().mean()),
        "test_loss": loss,
        "test_acc_exterior": masked_acc(pred, y, y == 0),
        "test_acc_surface": masked_acc(pred, y, y == 1),
        "test_acc_interior": masked_acc(pred, y, y == 2),
        "test_acc_near": masked_acc(pred, y, (sdf.abs() < 0.05) & (y != 1)),
        "test_acc_surface_Kneg": masked_acc(pred, y, (y == 1) & (K < 0)),
        "test_acc_surface_Kpos": masked_acc(pred, y, (y == 1) & (K > 0)),
        "test_acc_cavity": masked_acc(pred, y, cav),
        "sparsity": model.sparsity() if hasattr(model, "sparsity") else float("nan"),
    }
    return out


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and not args.overwrite:
        with open(out) as f:
            for r in csv.DictReader(f):
                done.add((r["shape"], r["family"], r["size"], r["seed"]))
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if fh.tell() == 0:
        w.writeheader()

    for shape in args.shapes:
        sd = load_shape(shape)
        Xtr, ytr, *_ = sd.subset("train"); Xva, yva, *_ = sd.subset("validation")
        Xtr = torch.from_numpy(Xtr).to(device); ytr = torch.from_numpy(ytr).to(device)
        Xva = torch.from_numpy(Xva).to(device); yva = torch.from_numpy(yva).to(device)
        for family in args.families:
            for size in default_grid(family, args.grid):
                size_s = json.dumps(size, sort_keys=True)
                for seed in args.seeds:
                    key = (shape, family, size_s, str(seed))
                    if key in done:
                        continue
                    torch.manual_seed(seed); np.random.seed(seed)
                    model = make_model(family, size)
                    lr = args.lr_spline if family.startswith("spline") else args.lr
                    t0 = time.time()
                    va, best_step, stop_step = train_one(model, Xtr, ytr, Xva, yva, args.steps, lr, device,
                                              patience=args.patience, min_steps=args.min_steps)
                    with torch.no_grad():
                        tr_acc = accuracy(model(Xtr), ytr)
                    row = {
                        "shape": shape, "family": family, "size": size_s, "seed": seed,
                        "n_params": n_params(model), "steps": args.steps, "lr": lr,
                        "val_acc": va, "train_acc": tr_acc, "best_step": best_step, "stop_step": stop_step,
                        "seconds": round(time.time() - t0, 1),
                    }
                    row.update(evaluate(model, sd, device))
                    w.writerow(row); fh.flush()
                    print(f"{shape:28s} {family:8s} {size_s:22s} seed={seed} params={row['n_params']:7d} "
                          f"val={va:.4f} test={row['test_acc']:.4f} surf={row['test_acc_surface']:.3f} "
                          f"int={row['test_acc_interior']:.3f} ({row['seconds']}s)", flush=True)
    fh.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+", default=["single_ring"], choices=SHAPES + ["all"])
    p.add_argument("--families", nargs="+", default=["sae", "sae_weak", "relu1", "mlp2", "spline1", "spline3"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--patience", type=int, default=3000, help="stop when val acc has not improved for this many steps (0 = off)")
    p.add_argument("--min_steps", type=int, default=5000)
    p.add_argument("--grid", default="reduced", choices=["reduced", "full"])
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lr_spline", type=float, default=1e-2)
    p.add_argument("--out", default="projects/manifold_features/results/sweep.csv")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.shapes == ["all"]:
        a.shapes = SHAPES
    run(a)
