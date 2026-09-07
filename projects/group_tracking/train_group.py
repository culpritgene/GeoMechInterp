"""Train a GPT (projects/path_transformer/model.py) to track group state on
Cayley walks and evaluate exact final-element accuracy.

    python projects/group_tracking/train_group.py --group Z36 --n_layer 2 --d_model 64 --steps 20000

Loss on the final element token only (group_data.loss_mask).  Plateau
stopping on validation exact accuracy: after `--min_steps`, stop when the
best validation accuracy has not improved for `--patience` evaluations.
Evaluation (a single forward pass, argmax over the full vocabulary at the
final SEP position): fresh words (validation / test), held-out generator
bigrams, non-commutativity minimal pairs (per word and per pair), and per k.
Checkpoints and summary.json go to
/var/tmp/geomech_ckpt/group_<group>_L<n_layer>_d<d_model>[<tag>]/.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "path_transformer"))
from model import GPT, GPTConfig  # noqa: E402
from group_data import GroupData, answer_positions, dataset_path, loss_mask  # noqa: E402

CKPT = Path("/var/tmp/geomech_ckpt")


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * step / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


def run_name(args) -> str:
    return f"group_{args.group}_L{args.n_layer}_d{args.d_model}{args.tag}"


@torch.no_grad()
def predict_final(model: GPT, X: torch.Tensor, device, batch: int = 4096) -> np.ndarray:
    """Predicted element id (argmax over the whole vocabulary, minus 4) at the final SEP of every row."""
    model.eval()
    out = []
    for b in range(0, len(X), batch):
        xb = X[b:b + batch].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(xb)
        pos = answer_positions(xb)
        out.append(logits[torch.arange(len(xb), device=device), pos].float().argmax(-1).cpu().numpy() - 4)
    return np.concatenate(out)


@torch.no_grad()
def evaluate(model: GPT, gd: GroupData, split: str, device, max_n: int | None = None) -> dict:
    X, idx = gd.tensors(split, max_n)
    if len(idx) == 0:
        return {}
    pred = predict_final(model, X, device)
    correct = pred == gd.final_elements(idx)
    res = dict(acc=float(correct.mean()), n=int(len(idx)))
    ks = gd.k[idx]
    res["per_k"] = {int(k): float(correct[ks == k].mean()) for k in np.unique(ks)}
    if split == "minimal_pair":
        pid = gd.pair_id[idx]
        both = np.array([correct[pid == p].all() for p in np.unique(pid)])
        res["pair_acc"] = float(both.mean())
    return res


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data = Path(args.data) if args.data else dataset_path(args.group, args.data_seed)
    gd = GroupData(data)
    Xtr, _ = gd.tensors("train")
    print(f"{gd.group.name}: {json.dumps(gd.summary())}", flush=True)
    cfg = GPTConfig(gd.vocab_size, gd.block_size, args.n_layer, args.n_head, args.d_model, args.dropout)
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.3f}M  device={device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    run = CKPT / run_name(args)
    run.mkdir(parents=True, exist_ok=True)
    Xtr = Xtr.to(device)
    log, best, best_step, bad, stop_reason = [], -1.0, 0, 0, "max_steps"
    t0 = time.time()
    ck_extra = dict(cfg=cfg.__dict__, args=vars(args) | dict(data=str(data)), group=gd.group.name, n_params=n_params)
    for step in range(1, args.steps + 1):
        model.train()
        bi = torch.randint(0, len(Xtr), (args.batch,), device=device)
        xb = Xtr[bi]
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(xb[:, :-1])
        mask = loss_mask(xb)
        loss = F.cross_entropy(logits.float()[mask], xb[:, 1:][mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        timed_out = args.time_limit is not None and time.time() - t0 > args.time_limit
        if step % args.eval_every == 0 or step == args.steps or timed_out:
            val = evaluate(model, gd, "validation", device, args.n_eval)
            hb = evaluate(model, gd, "heldout_bigram", device, args.n_eval)
            rec = dict(step=step, loss=loss.item(), lr=lr_at(step, args), val_acc=val["acc"], heldout_bigram_acc=hb["acc"],
                       seconds=round(time.time() - t0, 1))
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if val["acc"] > best:
                best, best_step, bad = val["acc"], step, 0
                torch.save(dict(model=model.state_dict(), step=step, **ck_extra), run / "best.pt")
            else:
                bad += 1
            if timed_out:
                stop_reason = "time_limit"; break
            if step >= args.min_steps and bad >= args.patience:
                stop_reason = "plateau"; break
            if best >= args.stop_acc and step >= args.min_steps:
                stop_reason = "target_acc"; break
    torch.save(dict(model=model.state_dict(), step=step, **ck_extra), run / "last.pt")
    model.load_state_dict(torch.load(run / "best.pt", map_location=device)["model"])
    tests = {s: evaluate(model, gd, s, device) for s in ("validation", "test", "heldout_bigram", "minimal_pair")}
    summary = dict(group=gd.group.name, data=str(data), n_params=n_params, n_layer=args.n_layer, d_model=args.d_model, n_head=args.n_head,
                   steps_run=step, steps=args.steps, stop_reason=stop_reason, best_val_acc=best, best_step=best_step,
                   seconds=round(time.time() - t0, 1), **{f"test_{k}": v for k, v in tests.items()}, args=vars(args), log=log)
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    brief = {k: (v["acc"] if isinstance(v, dict) and "acc" in v else v) for k, v in summary.items() if k not in ("log", "args", "test_validation")}
    brief["pair_acc"] = tests["minimal_pair"].get("pair_acc")
    brief["test_per_k"] = {k: round(v, 4) for k, v in tests["test"]["per_k"].items()}
    print("TEST", json.dumps(brief), flush=True)
    print("saved", run / "summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="Z36", help="Z<n>, T<n>x<m> or D<n>; the dataset must exist (group_data.py)")
    ap.add_argument("--data", default=None, help="explicit npz path (default /var/tmp/geomech_data/groups/<group>_<data_seed>.npz)")
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=40000, help="maximum steps (cosine schedule horizon)")
    ap.add_argument("--min_steps", type=int, default=20000)
    ap.add_argument("--patience", type=int, default=5, help="evaluations without validation improvement before stopping")
    ap.add_argument("--stop_acc", type=float, default=1.01, help="stop once best validation accuracy reaches this (after min_steps)")
    ap.add_argument("--time_limit", type=float, default=None, help="seconds; stop (with a final eval) once exceeded")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=5000, help="validation rows used during training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the run directory (e.g. _s1)")
    main(ap.parse_args())
