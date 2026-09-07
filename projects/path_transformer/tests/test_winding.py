"""Tests for the winding-class extension (gen_winding.py, winding_data.py).

Plain functions with a __main__ runner (pytest is optional):

    .venv/bin/python projects/path_transformer/tests/test_winding.py

Synthetic fixtures: a ring of cells (one circle) and a torus grid graph with
exact centres, so that crossing signs, cover Dijkstra and the generated
dataset can be checked without the shipped data.  When
generated/single_ring_wind.npz exists, its stored classes and paths are
checked as well.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths_data  # noqa: E402
from gen_winding import (TWO_PI, build_cover, cell_angles, cover_node, cut_signs, generate, path_winding,  # noqa: E402
                         paths_from_pred, ring_frame)
from winding_data import BOS, EOS, SEP, WindingData, path_classes  # noqa: E402

SCRATCH = Path(os.environ.get("WINDING_TEST_DIR") or tempfile.mkdtemp(prefix="winding_test_", dir=os.environ.get("TMPDIR")))


# ---------------------------------------------------------------- fixtures
def ring_cells(n: int = 40, offset: float = 0.5, axis=(0, 1, 0), R: float = 1.0):
    """n cells on a circle around `axis`; cell i sits at theta = (i + offset) * 2pi/n."""
    ax, u, v = ring_frame(axis)
    th = (np.arange(n) + offset) * TWO_PI / n
    centers = np.outer(R * np.cos(th), u) + np.outer(R * np.sin(th), v)
    adj = np.array([(i, (i + 1) % n) for i in range(n)] + [((i + 1) % n, i) for i in range(n)], dtype=np.int64)
    return centers, adj


def torus_grid(n_th: int = 24, n_ph: int = 8, R: float = 1.0, r: float = 0.25, axis=(0, 1, 0), center=(0, 0, 0), offset: float = 0.5):
    """Torus grid graph (n_th x n_ph cells, 4-neighbour adjacency) with exact centres."""
    ax, u, v = ring_frame(axis)
    th = (np.arange(n_th) + offset) * TWO_PI / n_th; ph = np.arange(n_ph) * TWO_PI / n_ph
    TH, PH = np.meshgrid(th, ph, indexing="ij"); TH, PH = TH.ravel(), PH.ravel()
    rho = R + r * np.cos(PH)
    centers = np.asarray(center, float) + np.outer(rho * np.cos(TH), u) + np.outer(rho * np.sin(TH), v) + np.outer(r * np.sin(PH), ax)
    idx = lambda i, j: (i % n_th) * n_ph + (j % n_ph)  # noqa: E731
    edges = set()
    for i in range(n_th):
        for j in range(n_ph):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                edges.add((idx(i, j), idx(i + di, j + dj)))
    adj = np.array(sorted(edges), dtype=np.int64)
    z = dict(cell_center=centers.astype(np.float32), cell_adj=adj.astype(np.int32),
             cell_table=np.zeros((len(centers), 5), dtype=np.int8), bounds=np.stack([centers.min(0) - 0.1, centers.max(0) + 0.1]))
    torus = dict(center=np.asarray(center, float), axis=np.asarray(axis, float), R=R, r=r)
    return z, torus


def free_dijkstra(centers, adj, sources):
    w = np.linalg.norm(centers[adj[:, 0]] - centers[adj[:, 1]], axis=1)
    A = sp.csr_matrix((w, (adj[:, 0], adj[:, 1])), shape=(len(centers), len(centers)))
    return dijkstra(A, directed=True, indices=sources, return_predecessors=True)


# ---------------------------------------------------------------- tests
def test_cut_signs_synthetic_ring():
    for offset in (0.5, 0.0, 0.999):
        for axis in ((0, 1, 0), (0, -1, 0), (1, 0, 0), (0.3, 0.9, -0.2)):
            n = 40
            centers, adj = ring_cells(n, offset, axis)
            theta = cell_angles(centers, np.zeros(3), axis)
            expect = np.mod((np.arange(n) + offset) * TWO_PI / n, TWO_PI)
            assert np.allclose(np.mod(theta - expect + np.pi, TWO_PI) - np.pi, 0, atol=1e-9), (offset, axis)
            s = cut_signs(theta, adj)
            lookup = {(int(a), int(b)): int(x) for (a, b), x in zip(adj, s)}
            assert lookup[(n - 1, 0)] == 1, (offset, axis, lookup[(n - 1, 0)])       # 2pi -> 0 forward
            assert lookup[(0, n - 1)] == -1, (offset, axis)                          # 0 -> 2pi backward
            others = [v for (a, b), v in lookup.items() if {a, b} != {0, n - 1}]
            assert all(v == 0 for v in others), (offset, axis)
            assert int(np.abs(s).sum()) == 2
    # a step that is exactly antipodal or exactly 0 never crosses
    theta = np.array([0.0, 0.0, np.pi]); adj = np.array([[0, 1], [1, 0]])
    assert (cut_signs(theta, adj) == 0).all()


def test_wiggle_cancels_and_laps_count():
    n = 12
    centers, adj = ring_cells(n)
    theta = cell_angles(centers, np.zeros(3), (0, 1, 0))
    lookup = {(int(a), int(b)): int(x) for (a, b), x in zip(adj, cut_signs(theta, adj))}
    wiggle = [0, 1, 2, 1, 0, n - 1, 0, n - 1, n - 2, n - 1, 0]
    assert path_winding(wiggle, lookup) == 0
    lap = list(range(n)) + [0]
    assert path_winding(lap, lookup) == 1
    assert path_winding(lap[::-1], lookup) == -1
    assert path_winding(lap + lap[1:], lookup) == 2
    # vectorised version agrees, with padding and lengths handled
    cut = np.array([[a, b, s] for (a, b), s in lookup.items() if s != 0])
    M = np.full((3, 2 * n + 1), -1); M[0, :len(wiggle)] = wiggle; M[1, :len(lap)] = lap; M[2] = lap + lap[1:]
    L = np.array([len(wiggle), len(lap), 2 * n + 1])
    assert path_classes(M, L, cut, n).tolist() == [0, 1, 2]
    # a shorter declared length ignores the tail
    assert path_classes(M, np.array([len(wiggle), n, n + 1]), cut, n).tolist() == [0, 0, 1]


def test_cover_dijkstra_matches_free_geodesic():
    z, torus = torus_grid(24, 8)
    centers = z["cell_center"].astype(np.float64); adj = z["cell_adj"].astype(np.int64); C = len(centers)
    theta = cell_angles(centers, torus["center"], torus["axis"])
    signs = cut_signs(theta, adj)
    w = np.linalg.norm(centers[adj[:, 0]] - centers[adj[:, 1]], axis=1)
    lookup = {(int(a), int(b)): int(x) for (a, b), x in zip(adj, signs)}
    assert int(np.abs(signs).sum()) == 2 * 8                  # one meridian of 8 cells, both directions
    K = 2; S = 2 * K + 1
    G = build_cover(C, adj, signs, w, K)
    assert (G != G.T).nnz == 0                                 # symmetric by construction
    rng = np.random.default_rng(0)
    n_checked = 0
    for _ in range(25):
        s, g = rng.choice(C, 2, replace=False)
        Dc, Pc = dijkstra(G, directed=True, indices=int(cover_node(s, 0, K)), return_predecessors=True)
        Df, Pf = free_dijkstra(centers, adj, [int(s)])
        Df, Pf = Df[0], Pf[0]
        sheets = Dc[cover_node(g, np.arange(-K, K + 1), K)]
        assert np.isclose(sheets.min(), Df[g]), (sheets, Df[g])   # min over classes = free geodesic
        # class of the free path (predecessor chain on the cell graph)
        nodes = [int(g)]
        while nodes[-1] != s:
            nodes.append(int(Pf[nodes[-1]]))
        kf = path_winding(nodes[::-1], lookup)
        if (np.isclose(sheets, sheets.min())).sum() == 1:      # unique free class
            assert np.argmin(sheets) - K == kf
            n_checked += 1
        for k in range(-K, K + 1):
            tgt = int(cover_node(g, k, K))
            if not np.isfinite(Dc[tgt]):
                continue
            M, L = paths_from_pred(Pc, int(cover_node(s, 0, K)), np.array([tgt]))
            cells = (M[0, :L[0]] // S)[::-1]                   # forward: s -> g
            assert cells[0] == s and cells[-1] == g
            assert all((int(a), int(b)) in lookup for a, b in zip(cells[:-1], cells[1:]))
            assert path_winding(cells, lookup) == k
            plen = np.linalg.norm(np.diff(centers[cells], axis=0), axis=1).sum()
            assert np.isclose(plen, Dc[tgt])
            assert Dc[tgt] >= Df[g] - 1e-9
    assert n_checked >= 15


def _synthetic_dataset():
    z, torus = torus_grid(24, 8)
    res = generate(z, torus, K=1, n_goals=20, starts_per_goal=25, min_dist=0.3, heldout_frac=0.2, seed=1, verbose=False)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCRATCH / "synth_wind.npz", **res)
    return res


def _with_scratch_gen(fn):
    old = paths_data.GEN
    paths_data.GEN = SCRATCH
    try:
        return fn()
    finally:
        paths_data.GEN = old


def test_generate_synthetic_consistency():
    res = _synthetic_dataset()
    meta = json.loads(res["meta"]); st = meta["stats"]
    assert st["n_dropped_out_of_range"] == 0
    assert 0.6 < st["binding_fraction"] <= 0.7                 # exactly one of three classes is free
    C = len(res["cell_center"]); adj = {tuple(e) for e in res["cell_adj"].tolist()}
    N = len(res["lengths"])
    assert set(res["split"].tolist()) <= {"train", "validation", "test", "test_goalheld"}
    held = set(meta["heldout_goals"])
    assert len(held) == 4
    for i in range(N):
        L = int(res["lengths"][i]); cells = res["cell_ids"][i, :L]
        assert (res["cell_ids"][i, L:] == -1).all()
        assert cells[0] == res["start_node"][i] and cells[-1] == res["goal_node"][i]
        assert all((int(a), int(b)) in adj for a, b in zip(cells[:-1], cells[1:]))
        assert res["geo_dist"][i] > res["free_dist"][i]
        assert (res["split"][i] == "test_goalheld") == (int(res["goal_node"][i]) in held)
    recomputed = path_classes(res["cell_ids"], res["lengths"], res["cut_edges"], C)
    assert (recomputed == res["wind_class"]).all()
    assert (res["wind_class"] != res["free_class"]).all()
    plen = np.array([np.linalg.norm(np.diff(res["cell_center"][res["cell_ids"][i, :res["lengths"][i]]].astype(float), axis=0), axis=1).sum() for i in range(N)])
    assert np.allclose(plen, res["geo_dist"], rtol=1e-5)
    # every (start, goal, class) triple is unique
    trip = set(zip(res["start_node"].tolist(), res["goal_node"].tolist(), res["wind_class"].tolist()))
    assert len(trip) == N


def test_winding_data_roundtrip_and_sizes():
    if not (SCRATCH / "synth_wind.npz").exists():
        _synthetic_dataset()

    def run():
        wd = WindingData("synth", "flat")
        assert wd.K == 1 and wd.n_class == 3
        assert wd.prompt_len == 7 and wd.base_prompt_len == 5
        assert wd.vocab_size == 4 + wd.n_cells + 3
        assert wd.block_size == wd.prompt_len + int(wd.lengths.max()) + 1
        for k in (-1, 0, 1):
            assert wd.class_token(k) == 4 + wd.n_cells + k + 1 and wd.token_class(wd.class_token(k)) == k
        assert wd.token_class(3) is None and wd.token_class(wd.vocab_size) is None
        assert (wd.all_path_classes() == wd.wind_class).all()
        for i in range(len(wd.lengths)):
            L = int(wd.lengths[i]); cells = wd.cell_ids[i, :L].tolist()
            seq = wd.encode(i)
            assert len(seq) == wd.prompt_len + L + 1 <= wd.block_size
            assert seq[0] == BOS and seq[2] == SEP and seq[4] == SEP and seq[6] == SEP and seq[-1] == EOS
            assert wd.decode_prompt(seq[:wd.prompt_len]) == (cells[0], cells[-1], int(wd.wind_class[i]))
            assert wd.decode_path(seq[wd.prompt_len:]) == cells
            assert wd.path_class(cells) == int(wd.wind_class[i])
            Th = wd.lifted_angles(cells)
            assert np.isclose(Th[-1] - Th[0], wd.cell_theta[cells[-1]] + TWO_PI * wd.wind_class[i] - wd.cell_theta[cells[0]])
        # tensors: prompt columns and padding
        X, idx = wd.tensors("train", 16)
        assert X.shape[1] == wd.block_size and (X[:, 6] == SEP).all()
        # scoring
        i = 0; L = int(wd.lengths[i]); cells = wd.cell_ids[i, :L].tolist()
        sc = wd.score(cells, cells[0], cells[-1], cells)
        assert sc["success"] == 1 and sc["exact_class"] == 1 and sc["success_class"] == 1 and np.isclose(sc["len_ratio"], 1.0)
        none = wd.score(None, cells[0], cells[-1], cells)
        assert none["parsed"] == 0 and none["exact_class"] == 0 and none["success_class"] == 0 and np.isnan(none["len_ratio"])
        # a path of another class of the same pair: valid, reached, but wrong class
        same = np.where((wd.cell_ids[:, 0] == cells[0]) & (np.array([wd.cell_ids[j, wd.lengths[j] - 1] for j in range(len(wd.lengths))]) == cells[-1]) & (wd.wind_class != wd.wind_class[i]))[0]
        if len(same):
            j = int(same[0]); other = wd.cell_ids[j, :wd.lengths[j]].tolist()
            sc = wd.score(other, cells[0], cells[-1], cells)
            assert sc["success"] == 1 and sc["exact_class"] == 0 and sc["success_class"] == 0
        # class token inside the path is malformed
        assert wd.decode_path([wd.class_token(0), EOS]) is None
    _with_scratch_gen(run)


def test_real_single_ring_if_present():
    f = paths_data.GEN / "single_ring_wind.npz"
    if not f.exists():
        print("  (skipped: single_ring_wind.npz not generated yet)")
        return
    wd = WindingData("single_ring", "flat")
    assert wd.prompt_len == 7 and wd.vocab_size == 4 + wd.n_cells + wd.n_class
    assert wd.block_size == wd.prompt_len + int(wd.lengths.max()) + 1
    assert (wd.all_path_classes() == wd.wind_class).all()      # every stored path
    held = set(wd.meta["heldout_goals"])
    z = np.load(f, allow_pickle=True); goal = z["goal_node"]; start = z["start_node"]
    assert set(goal[wd.split == "test_goalheld"].tolist()) <= held
    assert not (set(goal[wd.split != "test_goalheld"].tolist()) & held)
    assert (wd.cell_ids[:, 0] == start).all()
    assert (wd.cell_ids[np.arange(len(wd.lengths)), wd.lengths.astype(int) - 1] == goal).all()
    # validity of a random subset of paths on the cell adjacency, round trip
    rng = np.random.default_rng(0)
    for i in rng.choice(len(wd.lengths), 200, replace=False):
        L = int(wd.lengths[i]); cells = wd.cell_ids[i, :L].tolist()
        assert all((a, b) in wd.adj for a, b in zip(cells[:-1], cells[1:]))
        seq = wd.encode(i)
        assert wd.decode_prompt(seq[:7]) == (cells[0], cells[-1], int(wd.wind_class[i])) and wd.decode_path(seq[7:]) == cells
    # cut geometry: the ring axis of single_ring is dataset axis 1
    assert np.allclose(wd.meta["torus"]["axis"], [0, 1, 0], atol=1e-6)
    assert abs(len(wd.cut_edges) - 2 * len(set(wd.cut_edges[:, 0].tolist()))) <= len(wd.cut_edges)  # both directions present
    assert set(wd.cut_edges[:, 2].tolist()) == {-1, 1}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
