"""Generate large shortest-path datasets on the sampled surface graphs.

The shipped geodesic sets (data/geodesic_datasets/combined_geodesics_v1) hold
~4k train paths per shape, too few to train a model that recovers global
geometry.  Each shape also ships the surface samples (11,000 points), the kNN
manifold graph (k=11, edge weight = Euclidean length) and the oct-tree used for
tokenisation (depth 5).  This script reuses those and samples many more
(start, goal) pairs, solving exact shortest paths with Dijkstra.

Tokenisation follows the shipped format: a point is the sequence of its
oct-tree child indices L0..L4 (child code = bx | by<<1 | bz<<2, verified against
the shipped data), and a path is the sequence of visited cells with
consecutive duplicates removed.

Output (data/geodesic_datasets/generated/<shape>.npz):
    cells       (N, Lmax, D) int8  oct-tree codes per waypoint, -1 padded
    cell_ids    (N, Lmax)    int32 flat cell id per waypoint, -1 padded
    lengths     (N,)         int16 number of waypoints
    start_node, goal_node (N,) int32   surface-point indices
    geo_dist    (N,)         float32  exact geodesic distance on the graph
    split       (N,)         str      train / validation / test
    cell_table  (C, D) int8  flat id -> oct-tree code path
    cell_center (C, 3) float32  cell centre in mesh coordinates
    cell_adj    (E, 2) int32  pairs of flat cell ids joined by a graph edge
    bounds      (2, 3)  octree bounds
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

REPO = Path(__file__).resolve().parents[2]
GEO = REPO / "data" / "geodesic_datasets" / "combined_geodesics_v1"
OUT = REPO / "data" / "geodesic_datasets" / "generated"
SHAPES = sorted(p.name[: -len("_geodesics")] for p in GEO.iterdir() if p.name.endswith("_geodesics"))


def load_shape(shape: str):
    d = GEO / f"{shape}_geodesics"
    md = json.load(open(d / "mesh_data.json"))
    meta = json.load(open(d / "octree_metadata.json"))
    g = pickle.load(open(d / "manifold_graph.pkl", "rb"))
    P = np.asarray(md["surface_points"], dtype=np.float64)
    E = np.array([(u, v, e["weight"]) for u, v, e in g["edges"]], dtype=np.float64)
    n = len(P)
    A = sp.coo_matrix((E[:, 2], (E[:, 0].astype(int), E[:, 1].astype(int))), shape=(n, n))
    A = A.maximum(A.T).tocsr()  # undirected
    return P, A, E[:, :2].astype(int), np.array(meta["octree_bounds"]), meta["octree_depth"]


def octree_codes(P: np.ndarray, bounds: np.ndarray, depth: int) -> np.ndarray:
    """(N, depth) child codes; bit k of a code is the half along axis k."""
    lo, hi = bounds.copy(), bounds.copy()
    lo = np.broadcast_to(bounds[0], P.shape).copy(); hi = np.broadcast_to(bounds[1], P.shape).copy()
    codes = np.zeros((len(P), depth), dtype=np.int8)
    for lvl in range(depth):
        mid = (lo + hi) / 2
        bits = (P >= mid)
        codes[:, lvl] = bits[:, 0] | (bits[:, 1] << 1) | (bits[:, 2] << 2)
        lo = np.where(bits, mid, lo); hi = np.where(bits, hi, mid)
    return codes


def cell_centers(table: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo = np.broadcast_to(bounds[0], (len(table), 3)).copy(); hi = np.broadcast_to(bounds[1], (len(table), 3)).copy()
    for lvl in range(table.shape[1]):
        mid = (lo + hi) / 2
        bits = np.stack([(table[:, lvl] >> k) & 1 for k in range(3)], 1).astype(bool)
        lo = np.where(bits, mid, lo); hi = np.where(bits, hi, mid)
    return ((lo + hi) / 2).astype(np.float32)


def generate(shape: str, n_sources: int, targets_per_source: int, min_dist: float, seed: int, chunk: int = 400):
    rng = np.random.default_rng(seed)
    P, A, E, bounds, depth = load_shape(shape)
    n = len(P)
    codes = octree_codes(P, bounds, depth)
    keys = [tuple(c) for c in codes]
    uniq = sorted(set(keys))
    cid = {k: i for i, k in enumerate(uniq)}
    node_cell = np.array([cid[k] for k in keys], dtype=np.int32)
    table = np.array(uniq, dtype=np.int8)
    adj = {(a, b) for a, b in zip(node_cell[E[:, 0]], node_cell[E[:, 1]]) if a != b}
    adj = np.array(sorted(adj | {(b, a) for a, b in adj}), dtype=np.int32)

    sources = rng.choice(n, size=min(n_sources, n), replace=False)
    paths, starts, goals, dists = [], [], [], []
    t0 = time.time()
    for i in range(0, len(sources), chunk):
        src = sources[i:i + chunk]
        D, pred = dijkstra(A, directed=False, indices=src, return_predecessors=True)
        for row, s in enumerate(src):
            ok = np.where(np.isfinite(D[row]) & (D[row] >= min_dist))[0]
            tg = rng.choice(ok, size=min(targets_per_source, len(ok)), replace=False)
            for t in tg:
                nodes = [t]
                while nodes[-1] != s:
                    nodes.append(pred[row, nodes[-1]])
                nodes = nodes[::-1]
                cells = node_cell[nodes]
                keep = np.r_[True, cells[1:] != cells[:-1]]
                paths.append(cells[keep]); starts.append(s); goals.append(t); dists.append(D[row, t])
        print(f"  {shape}: {i + len(src)}/{len(sources)} sources, {len(paths)} paths, {time.time() - t0:.0f}s", flush=True)

    N = len(paths); Lmax = max(len(p) for p in paths)
    cell_ids = np.full((N, Lmax), -1, dtype=np.int32)
    for i, p in enumerate(paths):
        cell_ids[i, :len(p)] = p
    lengths = np.array([len(p) for p in paths], dtype=np.int16)
    cells = np.where(cell_ids[..., None] >= 0, table[np.maximum(cell_ids, 0)], -1).astype(np.int8)
    split = rng.choice(["train", "validation", "test"], size=N, p=[0.9, 0.05, 0.05])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{shape}.npz", cells=cells, cell_ids=cell_ids, lengths=lengths,
        start_node=np.array(starts, dtype=np.int32), goal_node=np.array(goals, dtype=np.int32),
        geo_dist=np.array(dists, dtype=np.float32), split=split, cell_table=table,
        cell_center=cell_centers(table, bounds), cell_adj=adj, bounds=bounds, node_cell=node_cell,
        surface_points=P.astype(np.float32),
    )
    print(f"{shape}: {N} paths, {len(uniq)} cells, {len(adj) // 2} cell adjacencies, waypoints median {np.median(lengths):.0f} max {Lmax}, "
          f"geodesic dist median {np.median(dists):.2f} max {np.max(dists):.2f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["single_ring"], choices=SHAPES + ["all"])
    ap.add_argument("--n_sources", type=int, default=3000)
    ap.add_argument("--targets_per_source", type=int, default=100)
    ap.add_argument("--min_dist", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for s in (SHAPES if a.shapes == ["all"] else a.shapes):
        generate(s, a.n_sources, a.targets_per_source, a.min_dist, a.seed)
