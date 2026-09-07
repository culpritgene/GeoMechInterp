"""Tokenisation of winding-class paths (gen_winding.py) into transformer
sequences, and scoring that checks the homotopy class of a decoded path.

WindingData extends PathData (paths_data.py, same specials BOS=0 EOS=1 SEP=2
PAD=3) with a class token in the prompt:

  flat : BOS s SEP g SEP k SEP w1 ... wL EOS      prompt_len = 7
  hier : BOS s0..s4 SEP g0..g4 SEP k SEP w0..w4 SEP ... EOS   prompt_len = 15

The class token of winding number k in [-K, K] is base_vocab + (k + K), i.e.
4 + n_cells + (k + K) for the flat format.  Loss is taken on the path tokens
only (after the third SEP).  score() recomputes the winding number of the
decoded path by summing the signs of the cut edges it traverses and reports
exact_class (equal to the class of the reference path) and success_class
(valid, reached and exact class) in addition to PathData's metrics.

    python projects/path_transformer/winding_data.py --run single_ring_winding_flat_L3_d64
evaluates a checkpoint on validation / test / test_goalheld by greedy decoding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_data  # noqa: E402  (module reference so that tests can redirect paths_data.GEN)
from paths_data import BOS, EOS, PAD, SEP, PathData  # noqa: E402,F401

CKPT = Path("/var/tmp/geomech_ckpt")
TWO_PI = 2.0 * np.pi


def path_classes(cell_ids: np.ndarray, lengths: np.ndarray, cut_edges: np.ndarray, n_cells: int) -> np.ndarray:
    """Vectorised winding number of every row of a (-1 padded) cell-id matrix:
    sum of the signs of the cut edges traversed by consecutive pairs."""
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    if len(cut_edges) == 0 or cell_ids.shape[1] < 2:
        return np.zeros(len(cell_ids), dtype=np.int64)
    cut = np.asarray(cut_edges, dtype=np.int64)
    key = cut[:, 0] * n_cells + cut[:, 1]; order = np.argsort(key); key_s = key[order]; sgn_s = cut[order, 2]
    a, b = cell_ids[:, :-1], cell_ids[:, 1:]
    t = np.arange(cell_ids.shape[1] - 1)[None, :]
    valid = (a >= 0) & (b >= 0) & (t < np.asarray(lengths)[:, None] - 1)
    pk = np.where(valid, a * n_cells + b, -1)
    pos = np.clip(np.searchsorted(key_s, pk), 0, len(key_s) - 1)
    hit = valid & (key_s[pos] == pk)
    return np.where(hit, sgn_s[pos], 0).sum(1)


class WindingData(PathData):
    """PathData over <shape>_wind.npz with a winding-class token in the prompt."""

    def __init__(self, shape: str, fmt: str = "flat"):
        super().__init__(f"{shape}_wind", fmt)
        self.shape = shape
        z = np.load(paths_data.GEN / f"{shape}_wind.npz", allow_pickle=True)
        self.meta = json.loads(str(z["meta"]))
        self.K = int(self.meta["K"])
        self.n_class = 2 * self.K + 1
        self.wind_class = z["wind_class"].astype(np.int64)
        self.free_class = z["free_class"].astype(np.int64) if "free_class" in z.files else None
        self.cell_theta = z["cell_theta"].astype(np.float64)
        self.cut_edges = z["cut_edges"].astype(np.int64)
        self.sign = {(int(a), int(b)): int(s) for a, b, s in self.cut_edges}
        self.base_vocab = self.vocab_size                      # PathData's vocabulary (cells + specials)
        self.vocab_size = self.base_vocab + self.n_class
        self.base_prompt_len = self.prompt_len                 # BOS s.. SEP g.. SEP
        self.prompt_len = self.base_prompt_len + 2             # + k SEP
        self.block_size = self.prompt_len + int(self.lengths.max()) * self.wp_len + 1

    # ---- class tokens ----------------------------------------------------
    def class_token(self, k: int) -> int:
        assert -self.K <= k <= self.K, k
        return self.base_vocab + int(k) + self.K

    def token_class(self, tok: int) -> int | None:
        k = int(tok) - self.base_vocab - self.K
        return k if -self.K <= k <= self.K else None

    # ---- encoding ----------------------------------------------------------
    def encode(self, i: int) -> list[int]:
        s = super().encode(i)
        return s[:self.base_prompt_len] + [self.class_token(int(self.wind_class[i])), SEP] + s[self.base_prompt_len:]

    def decode_prompt(self, toks: list[int]) -> tuple[int, int, int] | None:
        """Prompt tokens -> (start cell, goal cell, class), flat format only."""
        toks = list(toks)
        if self.fmt != "flat" or len(toks) < self.prompt_len:
            return None
        if toks[0] != BOS or toks[2] != SEP or toks[4] != SEP or toks[6] != SEP:
            return None
        s, g, k = toks[1] - 4, toks[3] - 4, self.token_class(toks[5])
        if not (0 <= s < self.n_cells and 0 <= g < self.n_cells) or k is None:
            return None
        return s, g, k

    # ---- winding numbers ---------------------------------------------------
    def step_signs(self, cells) -> np.ndarray:
        """Crossing sign of every consecutive step of a cell path."""
        cells = np.asarray(cells)
        return np.array([self.sign.get((int(a), int(b)), 0) for a, b in zip(cells[:-1], cells[1:])], dtype=np.int64)

    def path_class(self, cells) -> int:
        return int(self.step_signs(cells).sum()) if len(cells) > 1 else 0

    def all_path_classes(self) -> np.ndarray:
        """Winding number of every stored (clean) path, recomputed from cut edges."""
        return path_classes(self.cell_ids, self.lengths, self.cut_edges, self.n_cells)

    def lifted_angles(self, cells) -> np.ndarray:
        """Unrolled angle Theta_t = theta(cell_t) + 2 pi k_t along a path (k_0 = 0)."""
        cells = np.asarray(cells)
        kt = np.r_[0, np.cumsum(self.step_signs(cells))] if len(cells) > 1 else np.zeros(len(cells), dtype=np.int64)
        return self.cell_theta[cells] + TWO_PI * kt

    # ---- scoring -----------------------------------------------------------
    def score(self, cells: list[int] | None, start: int, goal: int, optimal: list[int]) -> dict:
        base = super().score(cells, start, goal, optimal)
        if cells is None or len(cells) == 0:
            base.update(exact_class=0, success_class=0)
            return base
        exact = int(self.path_class(cells) == self.path_class(optimal))
        base.update(exact_class=exact, success_class=int(base["success"] and exact))
        return base


if __name__ == "__main__":
    import argparse

    import torch
    from model import GPT, GPTConfig  # noqa: E402
    from train_path import evaluate_generation  # noqa: E402

    ap = argparse.ArgumentParser(description="Evaluate a winding checkpoint on several splits by greedy decoding.")
    ap.add_argument("--run", default="single_ring_winding_flat_L3_d64")
    ap.add_argument("--splits", nargs="+", default=["validation", "test", "test_goalheld"])
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT / a.run / a.ckpt, map_location=device)
    args = ck["args"]
    wd = WindingData(args["shape"], args["fmt"])
    model = GPT(GPTConfig(**ck["cfg"])).to(device); model.load_state_dict(ck["model"]); model.eval()
    res = {}
    for split in a.splits:
        X, idx = wd.tensors(split, a.n_eval)
        if len(idx) == 0:
            continue
        res[split] = dict(n=int(len(idx)), **evaluate_generation(model, wd, X, idx, device))
        print(split, json.dumps(res[split]), flush=True)
    out = CKPT / a.run / "eval_winding.json"
    out.write_text(json.dumps(dict(run=a.run, step=ck.get("step"), splits=res), indent=1))
    print("saved", out)
