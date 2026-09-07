"""Generate large shortest-path datasets on sampled surface graphs.

Sources of surface graphs:
  * the shipped geodesic sets (data/geodesic_datasets/combined_geodesics_v1):
    11,000 surface points, kNN manifold graph (k=11), oct-tree of depth 5;
  * analytic manifolds written by gen_manifold.py (--manifold): controllable
    minor radius (curvature / hole size), sampling density, oct-tree depth and
    surface-noise shell.

Paths: for each sampled goal one Dijkstra gives exact distances to all nodes;
starts are sampled among reachable nodes at distance >= --min_dist.
  --temperature 0   exact shortest path (predecessor chain)
  --temperature T>0 stochastic near-shortest path: at every node the next node
                    is drawn with probability proportional to
                    exp(-regret / (T * mean_edge_length)), where regret is the
                    extra distance-to-goal incurred by that step (T=0.5 is a
                    mild wobble, T=1 a noticeable one)
Observation noise (--p_obs): each interior waypoint is replaced with
probability p by a random adjacent occupied cell; the model is trained on the
noisy sequence while scoring and probe targets use the clean one.

Tokenisation follows the shipped format: a point is the sequence of its
oct-tree child indices L0..L4 (child code = bx | by<<1 | bz<<2, verified
against the shipped data) and a path is the sequence of visited cells with
consecutive duplicates removed.

Output (data/geodesic_datasets/generated/<name>.npz):
    cells       (N, Lmax, D) int8  oct-tree codes per observed waypoint, -1 padded
    cell_ids    (N, Lmax)    int32 observed (possibly noisy) cell ids, -1 padded
    cell_ids_clean (N, Lmax) int32 clean cell ids, aligned with cell_ids
    lengths     (N,)         int16 number of waypoints
    start_node, goal_node (N,) int32   surface-point indices
    geo_dist    (N,)         float32  exact geodesic distance on the graph
    path_len    (N,)         float32  length of the generated node path
    split       (N,)         str      train / validation / test
    cell_table  (C, D) int8  flat id -> oct-tree code path
    cell_center (C, 3) float32  cell centre in mesh coordinates
    cell_adj    (E, 2) int32  pairs of flat cell ids joined by a graph edge
    bounds      (2, 3)  octree bounds
    node_cell   (n,)    cell id of every surface point
    surface_points (n, 3)
    tori        (T, 8)  centre, axis, R, r per torus (analytic manifolds only)
    meta        JSON string of the generation arguments
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

REPO = Path(__file__).resolve().parents[2]
GEO = REPO / "data" / "geodesic_datasets" / "combined_geodesics_v1"
OUT = REPO / "data" / "geodesic_datasets" / "generated"
MANIFOLDS = Path("/var/tmp/geomech_data/manifolds")
SHAPES = sorted(p.name[: -len("_geodesics")] for p in GEO.iterdir() if p.name.endswith("_geodesics")) if GEO.exists() else []


def load_shape(shape: str):
    """Shipped surface graph: points, csr adjacency, edge list, bounds, depth, extras."""
    d = GEO / f"{shape}_geodesics"
    md = json.load(open(d / "mesh_data.json"))
    meta = json.load(open(d / "octree_metadata.json"))
    g = pickle.load(open(d / "manifold_graph.pkl", "rb"))
    P = np.asarray(md["surface_points"], dtype=np.float64)
    E = np.array([(u, v, e["weight"]) for u, v, e in g["edges"]], dtype=np.float64)
    n = len(P)
    A = sp.coo_matrix((E[:, 2], (E[:, 0].astype(int), E[:, 1].astype(int))), shape=(n, n))
    A = A.maximum(A.T).tocsr()
    return P, A, E[:, :2].astype(int), np.array(meta["octree_bounds"]), meta["octree_depth"], {}


def load_manifold(name: str):
    """Analytic manifold from gen_manifold.py."""
    z = np.load(MANIFOLDS / f"{name}.npz", allow_pickle=True)
    P = z["points"].astype(np.float64); E = z["edges"].astype(int); w = z["weights"].astype(np.float64)
    n = len(P)
    A = sp.coo_matrix((w, (E[:, 0], E[:, 1])), shape=(n, n)); A = A.maximum(A.T).tocsr()
    extras = {k: z[k] for k in ("tori", "gauss_curv", "theta", "phi", "sdf", "normal", "torus_id") if k in z.files}
    return P, A, E, z["bounds"].astype(np.float64), int(z["depth"]), extras


def octree_codes(P: np.ndarray, bounds: np.ndarray, depth: int) -> np.ndarray:
    """(N, depth) child codes; bit k of a code is the half along axis k."""
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


def rollout(A: sp.csr_matrix, D: np.ndarray, s: int, g: int, T: float, scale: float, rng: np.random.Generator, max_steps: int):
    """Stochastic near-shortest path from s to g given exact distances-to-goal D."""
    nodes = [s]; cur = s
    while cur != g and len(nodes) <= max_steps:
        nb = A.indices[A.indptr[cur]:A.indptr[cur + 1]]; w = A.data[A.indptr[cur]:A.indptr[cur + 1]]
        regret = w + D[nb] - D[cur]
        p = np.exp(-regret / (T * scale)); p /= p.sum()
        cur = int(rng.choice(nb, p=p)); nodes.append(cur)
    return nodes if cur == g else None


def generate(shape: str, n_goals: int, starts_per_goal: int, min_dist: float, seed: int, temperature: float, p_obs: float,
             manifold: bool, out_name: str | None, chunk: int = 200):
    rng = np.random.default_rng(seed)
    P, A, E, bounds, depth, extras = (load_manifold if manifold else load_shape)(shape)
    n = len(P)
    codes = octree_codes(P, bounds, depth)
    keys = [tuple(c) for c in codes]
    uniq = sorted(set(keys)); cid = {k: i for i, k in enumerate(uniq)}
    node_cell = np.array([cid[k] for k in keys], dtype=np.int32)
    table = np.array(uniq, dtype=np.int8)
    adj = {(a, b) for a, b in zip(node_cell[E[:, 0]], node_cell[E[:, 1]]) if a != b}
    adj = np.array(sorted(adj | {(b, a) for a, b in adj}), dtype=np.int32)
    adj_list = [[] for _ in range(len(uniq))]
    for a, b in adj:
        adj_list[a].append(b)
    scale = float(A.data.mean())

    goals = rng.choice(n, size=min(n_goals, n), replace=False)
    paths_clean, paths_obs, starts, gl, dists, plens = [], [], [], [], [], []
    t0 = time.time(); n_fail = 0
    for i in range(0, len(goals), chunk):
        gs = goals[i:i + chunk]
        D, pred = dijkstra(A, directed=False, indices=gs, return_predecessors=True)
        for row, g in enumerate(gs):
            ok = np.where(np.isfinite(D[row]) & (D[row] >= min_dist))[0]
            for s in rng.choice(ok, size=min(starts_per_goal, len(ok)), replace=False):
                if temperature > 0:
                    nodes = rollout(A, D[row], int(s), int(g), temperature, scale, rng, max_steps=int(3 * D[row, s] / scale) + 20)
                    if nodes is None:
                        n_fail += 1; continue
                else:
                    nodes = [int(s)]
                    while nodes[-1] != g:
                        nodes.append(int(pred[row, nodes[-1]]))
                cells = node_cell[nodes]
                keep = np.r_[True, cells[1:] != cells[:-1]]
                clean = cells[keep]
                obs = clean.copy()
                if p_obs > 0 and len(clean) > 2:
                    for j in range(1, len(clean) - 1):
                        if rng.random() < p_obs and adj_list[clean[j]]:
                            obs[j] = rng.choice(adj_list[clean[j]])
                paths_clean.append(clean); paths_obs.append(obs); starts.append(s); gl.append(g); dists.append(D[row, g if False else s])
                plens.append(float(np.linalg.norm(np.diff(P[nodes], axis=0), axis=1).sum()))
        print(f"  {shape}: {i + len(gs)}/{len(goals)} goals, {len(paths_clean)} paths, {n_fail} rollout failures, {time.time() - t0:.0f}s", flush=True)

    N = len(paths_clean); Lmax = max(len(p) for p in paths_clean)
    cell_ids = np.full((N, Lmax), -1, dtype=np.int32); cell_ids_clean = np.full((N, Lmax), -1, dtype=np.int32)
    for i, (c, o) in enumerate(zip(paths_clean, paths_obs)):
        cell_ids_clean[i, :len(c)] = c; cell_ids[i, :len(o)] = o
    lengths = np.array([len(p) for p in paths_clean], dtype=np.int16)
    cells = np.where(cell_ids[..., None] >= 0, table[np.maximum(cell_ids, 0)], -1).astype(np.int8)
    split = rng.choice(["train", "validation", "test"], size=N, p=[0.9, 0.05, 0.05])
    name = out_name or (shape + (f"_T{temperature:g}" if temperature > 0 else "") + (f"_obs{p_obs:g}" if p_obs > 0 else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    meta = dict(shape=shape, manifold=manifold, n_goals=n_goals, starts_per_goal=starts_per_goal, min_dist=min_dist, seed=seed,
                temperature=temperature, p_obs=p_obs, n_cells=len(uniq), depth=depth, mean_edge=scale)
    arrays = dict(cells=cells, cell_ids=cell_ids, cell_ids_clean=cell_ids_clean, lengths=lengths,
                  start_node=np.array(starts, dtype=np.int32), goal_node=np.array(gl, dtype=np.int32),
                  geo_dist=np.array(dists, dtype=np.float32), path_len=np.array(plens, dtype=np.float32), split=split,
                  cell_table=table, cell_center=cell_centers(table, bounds), cell_adj=adj, bounds=bounds, node_cell=node_cell,
                  surface_points=P.astype(np.float32), meta=json.dumps(meta))
    arrays.update(extras)
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    obs_changed = float((cell_ids != cell_ids_clean).sum() / max(1, lengths.sum()))
    print(f"{name}: {N} paths, {len(uniq)} cells, {len(adj) // 2} cell adjacencies, waypoints median {np.median(lengths):.0f} max {Lmax}, "
          f"geodesic dist median {np.median(dists):.2f} max {np.max(dists):.2f}, path/geodesic length {np.mean(np.array(plens) / np.array(dists)):.3f}, "
          f"observed cells changed {obs_changed:.3f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["single_ring"], help="shipped shape names, or manifold names with --manifold, or 'all'")
    ap.add_argument("--manifold", action="store_true", help="names refer to gen_manifold.py outputs in /var/tmp/geomech_data/manifolds")
    ap.add_argument("--n_goals", type=int, default=3000)
    ap.add_argument("--starts_per_goal", type=int, default=100)
    ap.add_argument("--min_dist", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--p_obs", type=float, default=0.0)
    ap.add_argument("--out_name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names = SHAPES if a.shapes == ["all"] else a.shapes
    for s in names:
        generate(s, a.n_goals, a.starts_per_goal, a.min_dist, a.seed, a.temperature, a.p_obs, a.manifold,
                 a.out_name if len(names) == 1 else None)
