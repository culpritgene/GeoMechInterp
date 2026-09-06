"""Tokenisation of generated shortest paths into transformer sequences.

Two formats share the same specials (BOS=0, EOS=1, SEP=2, PAD=3):

  hier : each waypoint is its 5 oct-tree level tokens L{l}_{c} = 4 + 8*l + c
         (the shipped 44-token vocabulary).  Sequence
         BOS s0..s4 SEP g0..g4 SEP w0..w4 SEP w0..w4 ... SEP w0..w4 EOS
  flat : each waypoint is one token 4 + cell_id.  Sequence
         BOS s SEP g SEP w1 ... wL EOS

The path always starts at the start cell and ends at the goal cell.  Loss is
taken on tokens after the second SEP (the path) only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "data" / "geodesic_datasets" / "generated"
BOS, EOS, SEP, PAD = 0, 1, 2, 3


class PathData:
    def __init__(self, shape: str, fmt: str = "hier"):
        z = np.load(GEN / f"{shape}.npz", allow_pickle=True)
        self.shape, self.fmt = shape, fmt
        self.cell_ids = z["cell_ids"]; self.lengths = z["lengths"]; self.split = z["split"]
        self.cell_table = z["cell_table"]; self.cell_center = z["cell_center"]
        self.geo_dist = z["geo_dist"]
        self.adj = set(map(tuple, z["cell_adj"].tolist()))
        self.n_cells, self.depth = self.cell_table.shape
        self.code_to_id = {tuple(c): i for i, c in enumerate(self.cell_table.tolist())}
        if fmt == "hier":
            self.vocab_size = 4 + 8 * self.depth
            self.cell_tok_len = self.depth
            self.wp_len = self.depth + 1                      # 5 level tokens + SEP
        else:
            self.vocab_size = 4 + self.n_cells
            self.cell_tok_len = 1
            self.wp_len = 1
        self.prompt_len = 1 + 2 * (self.cell_tok_len + 1)     # BOS s.. SEP g.. SEP
        self.block_size = self.prompt_len + int(self.lengths.max()) * self.wp_len + 1

    # ---- encoding -------------------------------------------------------
    def cell_tokens(self, cid: int) -> list[int]:
        if self.fmt == "hier":
            return [4 + 8 * l + int(c) for l, c in enumerate(self.cell_table[cid])]
        return [4 + int(cid)]

    def encode(self, i: int) -> list[int]:
        L = int(self.lengths[i]); ids = self.cell_ids[i, :L]
        seq = [BOS] + self.cell_tokens(ids[0]) + [SEP] + self.cell_tokens(ids[-1]) + [SEP]
        for c in ids:
            seq += self.cell_tokens(c)
            if self.fmt == "hier":
                seq.append(SEP)
        if self.fmt == "hier":
            seq[-1] = EOS
        else:
            seq.append(EOS)
        return seq

    def tensors(self, split: str, max_n: int | None = None):
        idx = np.where(self.split == split)[0]
        if max_n is not None:
            idx = idx[:max_n]
        X = np.full((len(idx), self.block_size), PAD, dtype=np.int64)
        for r, i in enumerate(idx):
            s = self.encode(i); X[r, :len(s)] = s
        return torch.from_numpy(X), idx

    # ---- decoding -------------------------------------------------------
    def decode_path(self, toks: list[int]) -> list[int] | None:
        """Tokens after the prompt -> list of cell ids, or None if malformed."""
        toks = list(toks)
        if EOS in toks:
            toks = toks[:toks.index(EOS)]
        else:
            return None
        if self.fmt == "flat":
            if any(t < 4 or t - 4 >= self.n_cells for t in toks):
                return None
            return [t - 4 for t in toks]
        cells, cur = [], []
        for t in toks + [SEP]:
            if t == SEP:
                if len(cur) != self.depth:
                    return None
                code = tuple(cur); cid = self.code_to_id.get(code)
                if cid is None:
                    return None
                cells.append(cid); cur = []
            elif 4 <= t < self.vocab_size and (t - 4) // 8 == len(cur):
                cur.append((t - 4) % 8)
            else:
                return None
        return cells

    def polyline_length(self, cells) -> float:
        c = self.cell_center[np.asarray(cells)]
        return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) if len(c) > 1 else 0.0

    def score(self, cells: list[int] | None, start: int, goal: int, optimal: list[int]) -> dict:
        if cells is None or len(cells) == 0:
            return dict(parsed=0, valid=0, reached=0, success=0, len_ratio=np.nan)
        valid = cells[0] == start and all((a, b) in self.adj or a == b for a, b in zip(cells[:-1], cells[1:]))
        reached = cells[-1] == goal
        lr = self.polyline_length(cells) / max(self.polyline_length(optimal), 1e-9)
        return dict(parsed=1, valid=int(valid), reached=int(reached), success=int(valid and reached), len_ratio=lr)
