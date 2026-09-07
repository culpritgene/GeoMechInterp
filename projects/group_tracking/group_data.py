"""Cayley-walk datasets for group state tracking.

Sequence format (specials BOS=0 EOS=1 SEP=2 PAD=3, shared with path_transformer):

    BOS e0 SEP g1 g2 .. gk SEP ek EOS        k uniform in [1, kmax]

e0 is a uniform element, the generators g_i are iid from the group's generator
distribution, and ek = e0 g1 ... gk (right multiplication).  Tokens: 4 specials,
then |G| element tokens (4 + g), then generator tokens (4 + |G| + j).  The loss
is taken on the final element token only (`loss_mask`).

Splits (column `split`):
    train / validation / test   fresh random words; words containing a held-out
                                generator bigram are excluded from all three
    heldout_bigram              every word contains at least one of the held-out
                                ordered generator pairs (5% of pairs, chosen by seed)
    minimal_pair                D_n only: pairs of words (same `pair_id`) that differ
                                by swapping two adjacent non-commuting generators,
                                so their final elements differ

The npz (under /var/tmp/geomech_data/groups/<group>_<seed>.npz) also stores the
prefix products (exact element after every generator, -1 padded) and their
irrep coordinates, for probing.

    python projects/group_tracking/group_data.py --group D36 --seed 0 --n_train 200000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groups import Group, make_group, verify  # noqa: E402

BOS, EOS, SEP, PAD = 0, 1, 2, 3
DATA = Path("/var/tmp/geomech_data/groups")
SPLITS = ("train", "validation", "test", "heldout_bigram", "minimal_pair")


# ---------------------------------------------------------------- tokens
def element_token(G: Group, g) -> int | np.ndarray:
    return 4 + g


def generator_token(G: Group, j) -> int | np.ndarray:
    return 4 + G.order + j


def vocab_size(G: Group) -> int:
    return 4 + G.order + G.n_gen


def block_size(kmax: int) -> int:
    return kmax + 6            # BOS e0 SEP g1..gk SEP ek EOS


def encode(G: Group, e0: int, word, kmax: int) -> np.ndarray:
    """One padded token row for start element e0 and generator-index word."""
    word = np.asarray(word, dtype=np.int64)
    ek = G.prefix_products(e0, word)[-1]
    seq = [BOS, element_token(G, e0), SEP] + generator_token(G, word).tolist() + [SEP, element_token(G, ek), EOS]
    row = np.full(block_size(kmax), PAD, dtype=np.int64); row[:len(seq)] = seq
    return row


def encode_batch(G: Group, e0: np.ndarray, words: np.ndarray, k: np.ndarray, ek: np.ndarray, kmax: int) -> np.ndarray:
    n = len(e0); T = block_size(kmax)
    X = np.full((n, T), PAD, dtype=np.int64)
    X[:, 0] = BOS; X[:, 1] = element_token(G, e0); X[:, 2] = SEP
    pos = np.arange(kmax)[None, :]
    valid = pos < k[:, None]
    X[:, 3:3 + kmax] = np.where(valid, generator_token(G, np.where(valid, words, 0)), PAD)
    rows = np.arange(n)
    X[rows, 3 + k] = SEP; X[rows, 4 + k] = element_token(G, ek); X[rows, 5 + k] = EOS
    return X


def decode(G: Group, row) -> tuple[int, list[int], int] | None:
    """Token row -> (e0, word of generator indices, ek), or None if malformed."""
    row = [int(t) for t in row]
    if len(row) < 6 or row[0] != BOS or row[2] != SEP:
        return None
    e0 = row[1] - 4
    if not (0 <= e0 < G.order):
        return None
    word = []
    i = 3
    while i < len(row) and row[i] != SEP:
        j = row[i] - 4 - G.order
        if not (0 <= j < G.n_gen):
            return None
        word.append(j); i += 1
    if i + 2 >= len(row) or row[i] != SEP or row[i + 2] != EOS or not word:
        return None
    ek = row[i + 1] - 4
    if not (0 <= ek < G.order) or any(t != PAD for t in row[i + 3:]):
        return None
    return e0, word, ek


def answer_positions(X: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Index of the final SEP in every row: the position whose logits must
    produce the final element token."""
    is_sep = X == SEP
    # second SEP = first position where the cumulative SEP count reaches 2
    cnt = is_sep.cumsum(1) if isinstance(X, torch.Tensor) else np.cumsum(is_sep, 1)
    return (is_sep & (cnt == 2)).float().argmax(1) if isinstance(X, torch.Tensor) else (is_sep & (cnt == 2)).argmax(1)


def loss_mask(X: torch.Tensor) -> torch.Tensor:
    """Boolean (B, T-1) mask over the shifted targets X[:, 1:]: True only at the
    final element token, i.e. at the final SEP position of the inputs X[:, :-1]."""
    mask = torch.zeros(X.shape[0], X.shape[1] - 1, dtype=torch.bool, device=X.device)
    mask[torch.arange(X.shape[0], device=X.device), answer_positions(X)] = True
    return mask


# ---------------------------------------------------------------- sampling
def _contains_bigram(words: np.ndarray, k: np.ndarray, forbidden: np.ndarray) -> np.ndarray:
    """Rows of `words` (n, kmax; -1 padded) with a forbidden (n_gen x n_gen bool) adjacent pair."""
    if not forbidden.any():
        return np.zeros(len(words), dtype=bool)
    valid = (np.arange(words.shape[1] - 1)[None, :] + 1) < k[:, None]
    a, b = np.maximum(words[:, :-1], 0), np.maximum(words[:, 1:], 0)
    return (forbidden[a, b] & valid).any(1)


def sample_words(rng: np.random.Generator, G: Group, n: int, kmax: int, forbidden: np.ndarray, kmin: int = 1):
    """iid generator words with k ~ U[kmin, kmax], rejecting forbidden bigrams."""
    k = rng.integers(kmin, kmax + 1, size=n)
    words = rng.choice(G.n_gen, size=(n, kmax), p=G.generator_probs)
    words[np.arange(kmax)[None, :] >= k[:, None]] = -1
    bad = _contains_bigram(words, k, forbidden)
    while bad.any():
        idx = np.where(bad)[0]
        w = rng.choice(G.n_gen, size=(len(idx), kmax), p=G.generator_probs)
        w[np.arange(kmax)[None, :] >= k[idx][:, None]] = -1
        words[idx] = w
        bad[idx] = _contains_bigram(w, k[idx], forbidden)
    return k, words


def sample_heldout_words(rng, G: Group, n: int, kmax: int, pairs: np.ndarray, forbidden: np.ndarray):
    """Words of length >= 2 with one held-out bigram planted at a random position."""
    k, words = sample_words(rng, G, n, kmax, forbidden, kmin=2)
    pos = (rng.random(n) * (k - 1)).astype(np.int64)
    which = pairs[rng.integers(0, len(pairs), size=n)]
    rows = np.arange(n)
    words[rows, pos] = which[:, 0]; words[rows, pos + 1] = which[:, 1]
    return k, words


def sample_minimal_pairs(rng, G: Group, n_pairs: int, kmax: int, forbidden: np.ndarray):
    """Pairs of words differing by swapping adjacent non-commuting generators.
    Returns (k, words) with 2 * n_pairs rows, pair 2i / 2i+1, and `pos`."""
    gens = G.generators
    noncomm = G.table[gens[:, None], gens[None, :]] != G.table[gens[None, :], gens[:, None]]
    if not noncomm.any():
        return np.zeros(0, dtype=np.int64), np.zeros((0, kmax), dtype=np.int64), np.zeros(0, dtype=np.int64)
    K, W, P = [], [], []
    need = n_pairs
    while need > 0:
        k, words = sample_words(rng, G, 2 * need, kmax, forbidden, kmin=2)
        valid = (np.arange(kmax - 1)[None, :] + 1) < k[:, None]
        cand = noncomm[np.maximum(words[:, :-1], 0), np.maximum(words[:, 1:], 0)] & valid
        ok = cand.any(1)
        k, words, cand = k[ok], words[ok], cand[ok]
        # choose one candidate position per row
        u = rng.random(cand.shape) * cand
        pos = u.argmax(1)
        swapped = words.copy(); rows = np.arange(len(words))
        swapped[rows, pos], swapped[rows, pos + 1] = words[rows, pos + 1], words[rows, pos]
        keep = ~_contains_bigram(swapped, k, forbidden)
        k, words, swapped, pos = k[keep][:need], words[keep][:need], swapped[keep][:need], pos[keep][:need]
        K.append(np.repeat(k, 2)); P.append(np.repeat(pos, 2))
        W.append(np.stack([words, swapped], 1).reshape(-1, kmax))
        need -= len(k)
    return np.concatenate(K), np.concatenate(W), np.concatenate(P)


def choose_heldout_bigrams(rng, G: Group, frac: float = 0.05) -> np.ndarray:
    """(P, 2) ordered generator-index pairs never seen in training."""
    n_pairs = G.n_gen ** 2
    n_hold = max(1, int(round(frac * n_pairs)))
    flat = rng.choice(n_pairs, size=n_hold, replace=False)
    return np.stack([flat // G.n_gen, flat % G.n_gen], 1)


def generate(G: Group, seed: int, n_train: int, n_eval: int, kmax: int, bigram_frac: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    pairs = choose_heldout_bigrams(rng, G, bigram_frac)
    forbidden = np.zeros((G.n_gen, G.n_gen), dtype=bool); forbidden[pairs[:, 0], pairs[:, 1]] = True
    parts = []
    for split, n in (("train", n_train), ("validation", n_eval), ("test", n_eval)):
        k, w = sample_words(rng, G, n, kmax, forbidden); parts.append((split, k, w, -np.ones(n, dtype=np.int64)))
    k, w = sample_heldout_words(rng, G, n_eval, kmax, pairs, forbidden); parts.append(("heldout_bigram", k, w, -np.ones(n_eval, dtype=np.int64)))
    k, w, _ = sample_minimal_pairs(rng, G, n_eval // 2, kmax, forbidden)
    parts.append(("minimal_pair", k, w, np.repeat(np.arange(len(k) // 2), 2) if len(k) else np.zeros(0, dtype=np.int64)))
    split = np.concatenate([np.full(len(k), s) for s, k, _, _ in parts])
    k = np.concatenate([p[1] for p in parts]); words = np.concatenate([p[2] for p in parts]); pair_id = np.concatenate([p[3] for p in parts])
    n = len(k)
    e0 = rng.integers(0, G.order, size=n)
    mp = np.where(split == "minimal_pair")[0]
    e0[mp[1::2]] = e0[mp[0::2]]                   # both words of a minimal pair start at the same element
    prefix = np.full((n, kmax + 1), -1, dtype=np.int64); prefix[:, 0] = e0
    for i in range(kmax):
        live = words[:, i] >= 0
        prefix[live, i + 1] = G.table[prefix[live, i], G.generators[words[live, i]]]
    ek = prefix[np.arange(n), k]
    tokens = encode_batch(G, e0, words, k, ek, kmax)
    prefix_irrep = np.where(prefix[..., None] >= 0, G.coords[np.maximum(prefix, 0)], 0).astype(np.float32)
    return dict(tokens=tokens.astype(np.int16), k=k.astype(np.int16), split=split, pair_id=pair_id.astype(np.int32),
                prefix=prefix.astype(np.int16), prefix_irrep=prefix_irrep, heldout_bigrams=pairs,
                group=np.array(G.name), group_params=np.array(json.dumps(G.params)), seed=np.array(seed), kmax=np.array(kmax),
                vocab_size=np.array(vocab_size(G)), block_size=np.array(block_size(kmax)),
                generators=G.generators, generator_names=np.array(G.generator_names), generator_probs=G.generator_probs,
                coord_names=np.array(G.coord_names), coords=G.coords, table=G.table.astype(np.int32))


def dataset_path(group: str, seed: int) -> Path:
    return DATA / f"{group}_{seed}.npz"


def write_dataset(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------- loader
class GroupData:
    """Loads a written npz and serves padded token tensors per split."""

    def __init__(self, path: str | Path):
        z = np.load(path)
        p = json.loads(str(z["group_params"]))
        self.group: Group = make_group(str(z["group"]), **p)
        self.path = Path(path)
        self.tokens = z["tokens"].astype(np.int64); self.k = z["k"].astype(np.int64); self.split = z["split"]
        self.pair_id = z["pair_id"]; self.prefix = z["prefix"].astype(np.int64); self.prefix_irrep = z["prefix_irrep"]
        self.heldout_bigrams = z["heldout_bigrams"]; self.seed = int(z["seed"]); self.kmax = int(z["kmax"])
        self.vocab_size = int(z["vocab_size"]); self.block_size = int(z["block_size"])
        assert self.vocab_size == vocab_size(self.group) and self.block_size == block_size(self.kmax)
        assert (z["table"] == self.group.table).all(), "group table in npz does not match the reconstructed group"

    def indices(self, split: str, max_n: int | None = None) -> np.ndarray:
        idx = np.where(self.split == split)[0]
        return idx if max_n is None else idx[:max_n]

    def tensors(self, split: str, max_n: int | None = None) -> tuple[torch.Tensor, np.ndarray]:
        idx = self.indices(split, max_n)
        return torch.from_numpy(self.tokens[idx]), idx

    def final_elements(self, idx) -> np.ndarray:
        return self.prefix[idx, self.k[idx]]

    def generator_positions(self, i: int) -> np.ndarray:
        """Positions of g_1 .. g_k in row i (position of g_t is 2 + t)."""
        return 3 + np.arange(int(self.k[i]))

    def final_sep_position(self, i: int) -> int:
        return 3 + int(self.k[i])

    def decode(self, i: int):
        return decode(self.group, self.tokens[i])

    def summary(self) -> dict:
        return {s: int((self.split == s).sum()) for s in SPLITS} | dict(group=self.group.name, kmax=self.kmax, vocab=self.vocab_size,
                                                                    block=self.block_size, heldout_bigrams=self.heldout_bigrams.tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="D36")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=200000)
    ap.add_argument("--n_eval", type=int, default=10000, help="size of validation, test and held-out sets (minimal pairs: n_eval/2 pairs)")
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--n_freq", type=int, default=3)
    ap.add_argument("--p_reflect", type=float, default=0.2)
    ap.add_argument("--diagonals", action="store_true")
    ap.add_argument("--bigram_frac", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    G = make_group(a.group, a.n_freq, a.p_reflect, a.diagonals); verify(G)
    arrays = generate(G, a.seed, a.n_train, a.n_eval, a.kmax, a.bigram_frac)
    out = Path(a.out) if a.out else dataset_path(a.group, a.seed)
    write_dataset(out, arrays)
    gd = GroupData(out)
    print(json.dumps(gd.summary()), f"written to {out} in {time.time() - t0:.1f}s")
