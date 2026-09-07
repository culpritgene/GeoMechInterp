"""Evaluate a trained path model by greedy decoding on any split of its dataset.

    python projects/path_transformer/eval_split.py --run single_ring_gh_flat_L3_d64 --splits test test_goalheld
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig
from paths_data import PathData
from train_path import CKPT, evaluate_generation

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--splits", nargs="+", default=["test", "test_goalheld"]); ap.add_argument("--n_eval", type=int, default=2000)
a = ap.parse_args(); device = torch.device("cuda")
ck = torch.load(CKPT / a.run / "best.pt", map_location=device); args = ck["args"]
pd = PathData(args["shape"], args["fmt"]); model = GPT(GPTConfig(**ck["cfg"])).to(device); model.load_state_dict(ck["model"]); model.eval()
res = {}
for sp in a.splits:
    X, idx = pd.tensors(sp, a.n_eval)
    if len(idx) == 0:
        continue
    res[sp] = dict(n=int(len(idx)), **evaluate_generation(model, pd, X, idx, device)); print(sp, json.dumps(res[sp]))
out = CKPT / a.run / "eval_splits.json"; out.write_text(json.dumps(dict(run=a.run, splits=res), indent=1)); print("saved", out)
