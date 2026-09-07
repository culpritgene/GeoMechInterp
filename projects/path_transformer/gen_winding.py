"""Winding-class shortest paths on the universal cover of a ring-shaped cell
graph (Rung 1 of PROPOSALS_language_ladder.md).

Extends gen_paths.py: instead of the free geodesic between two cells the task
asks for the shortest path in a prescribed homotopy class around the ring, so
a global invariant (the net number of laps) must be carried along the whole
sequence.  Everything is computed on the CELL adjacency graph of an existing
generated dataset (data/geodesic_datasets/generated/<shape>.npz).

Cut and crossing signs
    theta(c) = atan2(v . p, u . p) mod 2 pi of the cell centre p (relative to
    the torus centre) in the plane orthogonal to the ring axis, with (u, v,
    axis) a right-handed frame (ring_frame).  The cut is the half-plane
    theta = 0.  A directed cell edge a -> b crosses it with sign +1 when theta
    wraps from just below 2 pi to just above 0 in the positive direction
    (theta_a > theta_b and the wrapped step wrap(theta_b - theta_a) > 0), -1
    the other way, and 0 otherwise.  Summing the signs along a path gives its
    winding number; local wiggles across the cut cancel.  For shapes with
    several tori the cut belongs to one torus (--torus) and only edges whose
    two cells lie on that tube (|distance to the tube surface| < --tube_tol)
    can cross it.

Cover graph
    Nodes (cell, k), k in [-K, K], index cell * (2K+1) + (k + K).  Every cell
    adjacency a -> b with sign s gives the edges (a, k) -> (b, k + s), dropped
    when k + s leaves the range; weight = Euclidean distance between the two
    cell centres.  The class-k shortest path from s to g is the shortest path
    from (s, 0) to (g, k).

Sampling
    For each of n_goals goal cells one Dijkstra from (goal, 0) on a WIDER cover
    (range [-2K, 2K]) gives distances to (start, k') for every start and sheet;
    the predecessor chain from (start, k') back to (goal, 0), read forwards, is
    the class k = -k' path from start to goal.  A path found on the wide cover
    whose intermediate windings all stay inside [-K, K] is also the exact
    shortest path on the clamped [-K, K] cover (subgraph); the few that leave
    the range are dropped and counted.  The free geodesic of a (start, goal)
    pair is the minimum over sheets; a class is *binding* when its shortest
    path is strictly longer than the free geodesic, and only binding triples
    are kept.  starts_per_goal starts are sampled per goal (free distance >=
    min_dist) and every binding class of every sampled start is emitted, so
    N ~ n_goals * starts_per_goal * 2K.

Splits
    Every (start, goal, class) triple occurs at most once and is assigned to
    train / validation / test (0.9 / 0.05 / 0.05); in addition heldout_frac
    of the goal cells are held out entirely and all their triples get the
    split value 'test_goalheld'.

Output (data/geodesic_datasets/generated/<shape>_wind.npz), same keys as the
input dataset plus:
    cells, cell_ids, lengths, split, cell_table, cell_center, cell_adj,
    bounds, node_cell, surface_points  : as in gen_paths.py
    start_node, goal_node (N,) int32   : start / goal CELL ids (not points)
    geo_dist  (N,) float32   : length of the class-optimal path on the cover
    wind_class (N,) int8     : class k of the path (net cut crossings)
    free_class (N,) int8     : class of the free geodesic of the pair
    free_dist  (N,) float32  : free geodesic length on the cell graph
    cell_theta (C,) float32  : ring angle of every cell in [0, 2 pi)
    cut_edges  (E2, 3) int32 : [a, b, sign] for every directed edge with a
                               non-zero crossing sign
    meta                     : JSON string (torus frame, K, counts, stats)

    python projects/path_transformer/gen_winding.py --shape single_ring
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GEN = REPO / "data" / "geodesic_datasets" / "generated"   # symlink to /var/tmp/geomech_data/generated
TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------- geometry
def ring_frame(axis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical right-handed frame (axis, u, v) with u x v = axis.  The axis
    sign is fixed so that its largest component is positive (the SVD fit can
    return either sign); u is the world axis least aligned with `axis`,
    projected onto the orthogonal plane."""
    axis = np.asarray(axis, dtype=np.float64); axis = axis / np.linalg.norm(axis)
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    e = np.zeros(3); e[np.argmin(np.abs(axis))] = 1.0
    u = e - (e @ axis) * axis; u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return axis, u, v


def cell_angles(centers: np.ndarray, center, axis) -> np.ndarray:
    """Ring angle theta in [0, 2 pi) of every centre around `axis` through `center`."""
    axis, u, v = ring_frame(axis)
    p = np.asarray(centers, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    return np.mod(np.arctan2(p @ v, p @ u), TWO_PI)


def wrap_pi(d: np.ndarray) -> np.ndarray:
    """Wrap angle differences to [-pi, pi)."""
    return (np.asarray(d, dtype=np.float64) + np.pi) % TWO_PI - np.pi


def cut_signs(theta: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Crossing sign in {-1, 0, +1} of every directed edge (a, b) w.r.t. the
    cut at theta = 0: +1 if the (shorter, signed) angular step is positive but
    theta decreased (wrapped 2 pi -> 0), -1 if the step is negative but theta
    increased (wrapped 0 -> 2 pi)."""
    edges = np.asarray(edges)
    ta, tb = theta[edges[:, 0]], theta[edges[:, 1]]
    d = wrap_pi(tb - ta)
    return ((ta > tb) & (d > 0)).astype(np.int8) - ((ta < tb) & (d < 0)).astype(np.int8)


def tube_distance(centers: np.ndarray, center, axis, R: float, r: float) -> np.ndarray:
    """Signed distance of every point to the torus surface (negative inside)."""
    ax = ring_frame(axis)[0]
    p = np.asarray(centers, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    h = p @ ax; rho = np.linalg.norm(p - np.outer(h, ax), axis=1)
    return np.sqrt((rho - R) ** 2 + h ** 2) - r


def path_winding(cells, sign_lookup: dict) -> int:
    """Net crossings of a cell path given a {(a, b): sign} lookup (0 if absent)."""
    return int(sum(sign_lookup.get((int(a), int(b)), 0) for a, b in zip(cells[:-1], cells[1:])))


# ---------------------------------------------------------------- cover graph
def cover_node(cell, k, K: int):
    return np.asarray(cell) * (2 * K + 1) + (np.asarray(k) + K)


def build_cover(n_cells: int, edges: np.ndarray, signs: np.ndarray, weights: np.ndarray, K: int) -> sp.csr_matrix:
    """Directed weighted cover graph over (cell, k), k in [-K, K]; edges that
    would leave the range are dropped.  `edges` must contain both directions
    of every adjacency (as cell_adj does), so the result is symmetric."""
    S = 2 * K + 1
    ks = np.arange(-K, K + 1)
    E = len(edges)
    a = np.repeat(edges[:, 0].astype(np.int64), S); b = np.repeat(edges[:, 1].astype(np.int64), S)
    s = np.repeat(signs.astype(np.int64), S); w = np.repeat(np.asarray(weights, dtype=np.float64), S)
    k = np.tile(ks, E); k2 = k + s
    keep = np.abs(k2) <= K
    rows = a[keep] * S + (k[keep] + K); cols = b[keep] * S + (k2[keep] + K)
    return sp.csr_matrix((w[keep], (rows, cols)), shape=(n_cells * S, n_cells * S))


def paths_from_pred(pred: np.ndarray, src: int, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Follow predecessor links from every (reachable) target back to `src`.
    Returns (M, L): M[i] is the node sequence target -> ... -> src, -1 padded,
    L[i] its length.  Read forwards this is the path from target to src."""
    cur = np.asarray(targets, dtype=np.int64)
    cols = [cur]; alive = cur != src
    while alive.any():
        cur = np.where(alive, pred[cur], src)
        cols.append(cur); alive = cur != src
    M = np.stack(cols, 1)
    L = (M == src).argmax(1) + 1
    M[np.arange(M.shape[1])[None, :] >= L[:, None]] = -1
    return M, L


# ---------------------------------------------------------------- generation
def generate(z: Mapping, torus: Mapping, K: int = 1, n_goals: int = 1000, starts_per_goal: int = 150,
             min_dist: float = 0.15, heldout_frac: float = 0.1, seed: int = 0, tube_tol: float = 0.15,
             max_len: int = 0, verbose: bool = True) -> dict:
    """Build the winding dataset from an input dataset `z` (npz or dict with
    cell_center, cell_adj, cell_table, bounds, ...) and a torus frame
    {'center', 'axis', 'R', 'r'}.  Returns a dict of output arrays (see the
    module docstring); 'meta' is a JSON string."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    centers = np.asarray(z["cell_center"], dtype=np.float64); n_cells = len(centers)
    adj = np.asarray(z["cell_adj"], dtype=np.int64)
    table = np.asarray(z["cell_table"])
    axis, u, v = ring_frame(torus["axis"])
    theta = cell_angles(centers, torus["center"], axis)
    on_tube = np.abs(tube_distance(centers, torus["center"], axis, float(torus["R"]), float(torus["r"]))) < tube_tol
    signs = cut_signs(theta, adj)
    signs[~(on_tube[adj[:, 0]] & on_tube[adj[:, 1]])] = 0
    weights = np.linalg.norm(centers[adj[:, 0]] - centers[adj[:, 1]], axis=1)
    cut = np.where(signs != 0)[0]
    cut_edges = np.column_stack([adj[cut], signs[cut]]).astype(np.int32)

    Kw = 2 * K; Sw = 2 * Kw + 1
    ks_w = np.arange(-Kw, Kw + 1)
    order = np.argsort(np.abs(ks_w), kind="stable")          # sheets by |k'| so ties prefer small |k|
    G = build_cover(n_cells, adj, signs, weights, Kw)

    goals = rng.choice(n_cells, size=min(n_goals, n_cells), replace=False)
    n_held = int(round(heldout_frac * len(goals)))
    held = np.zeros(n_cells, dtype=bool); held[goals[:n_held]] = True

    paths, starts_l, goals_l, cls_l, dist_l, kfree_l, fdist_l = [], [], [], [], [], [], []
    n_cand = n_bind = n_drop_range = n_drop_len = 0
    for gi, g in enumerate(goals):
        src = int(cover_node(g, 0, Kw))
        dist, pred = dijkstra(G, directed=True, indices=src, return_predecessors=True)
        D = dist.reshape(n_cells, Sw)                         # D[c, j]: (g, 0) -> (c, k' = j - Kw)
        Dord = D[:, order]
        jmin = Dord.argmin(1); free = Dord[np.arange(n_cells), jmin]; kfree = -ks_w[order][jmin]
        ok = np.isfinite(free) & (free >= min_dist); ok[g] = False
        cand = np.where(ok)[0]
        if len(cand) == 0:
            continue
        starts = rng.choice(cand, size=min(starts_per_goal, len(cand)), replace=False)
        for k in range(-K, K + 1):
            j = Kw - k                                         # sheet index of k' = -k
            d = D[starts, j]
            fin = np.isfinite(d); n_cand += int(fin.sum())
            bind = fin & (d > free[starts] + 1e-6)
            n_bind += int(bind.sum())
            st = starts[bind]
            if len(st) == 0:
                continue
            M, L = paths_from_pred(pred, src, st * Sw + j)
            sheets = np.where(M >= 0, M % Sw, 0); kt = sheets - sheets[:, :1]
            within = ((np.abs(kt) <= K) | (M < 0)).all(1)
            n_drop_range += int((~within).sum())
            for row in np.where(within)[0]:
                if max_len and L[row] > max_len:
                    n_drop_len += 1; continue
                paths.append((M[row, :L[row]] // Sw).astype(np.int32))
                starts_l.append(int(st[row])); goals_l.append(int(g)); cls_l.append(k)
                dist_l.append(float(d[bind][row])); kfree_l.append(int(kfree[st[row]])); fdist_l.append(float(free[st[row]]))
        if verbose and ((gi + 1) % 100 == 0 or gi + 1 == len(goals)):
            print(f"  goal {gi + 1}/{len(goals)}: {len(paths)} paths, binding {n_bind}/{max(n_cand, 1)}, {time.time() - t0:.0f}s", flush=True)

    N = len(paths)
    if N == 0:
        raise RuntimeError("no binding paths generated")
    Lmax = max(len(p) for p in paths)
    cell_ids = np.full((N, Lmax), -1, dtype=np.int32)
    for i, p in enumerate(paths):
        cell_ids[i, :len(p)] = p
    lengths = np.array([len(p) for p in paths], dtype=np.int16)
    cells = np.where(cell_ids[..., None] >= 0, table[np.maximum(cell_ids, 0)], -1).astype(np.int8)
    start_node = np.array(starts_l, dtype=np.int32); goal_node = np.array(goals_l, dtype=np.int32)
    wind_class = np.array(cls_l, dtype=np.int8)
    split = rng.choice(["train", "validation", "test"], size=N, p=[0.9, 0.05, 0.05]).astype("<U13")
    split[held[goal_node]] = "test_goalheld"

    # statistics
    theta_s = theta[start_node]; theta_g = theta[goal_node]
    laps = np.abs((theta_g + TWO_PI * wind_class - theta_s) / TWO_PI)
    stats = dict(
        n_paths=N, n_candidates=n_cand, n_binding=n_bind, binding_fraction=n_bind / max(n_cand, 1),
        n_dropped_out_of_range=n_drop_range, n_dropped_max_len=n_drop_len,
        class_counts={int(k): int((wind_class == k).sum()) for k in range(-K, K + 1)},
        free_class_counts={int(k): int((np.array(kfree_l) == k).sum()) for k in range(-K, K + 1)},
        len_median=float(np.median(lengths)), len_p95=float(np.percentile(lengths, 95)), len_max=int(Lmax),
        geo_dist_median=float(np.median(dist_l)), geo_dist_max=float(np.max(dist_l)),
        excess_over_free_median=float(np.median(np.array(dist_l) - np.array(fdist_l))),
        laps_median=float(np.median(laps)), laps_max=float(laps.max()), frac_over_one_lap=float((laps > 1.0).mean()),
        split_counts={s: int((split == s).sum()) for s in ["train", "validation", "test", "test_goalheld"]},
        n_cut_edges=int(len(cut_edges)), seconds=round(time.time() - t0, 1),
    )
    meta = dict(
        K=K, K_search=Kw, torus=dict(center=np.asarray(torus["center"], dtype=float).tolist(), axis=axis.tolist(), u=u.tolist(), v=v.tolist(),
                                     R=float(torus["R"]), r=float(torus["r"])),
        cut="half-plane theta = 0; theta = atan2(v.p, u.p) mod 2pi; sign(a->b) = +1 when theta wraps 2pi->0 in the positive direction",
        class_token="4 + n_cells + (k + K) for the flat format (base vocab + k + K in general)",
        n_goals=int(len(goals)), n_heldout_goals=int(n_held), heldout_goals=sorted(int(g) for g in goals[:n_held]),
        starts_per_goal=starts_per_goal, min_dist=min_dist, heldout_frac=heldout_frac, tube_tol=tube_tol, max_len=max_len, seed=seed,
        stats=stats,
    )
    out = dict(
        cells=cells, cell_ids=cell_ids, lengths=lengths, start_node=start_node, goal_node=goal_node,
        geo_dist=np.array(dist_l, dtype=np.float32), split=split,
        cell_table=table, cell_center=np.asarray(z["cell_center"]), cell_adj=np.asarray(z["cell_adj"]), bounds=np.asarray(z["bounds"]),
        wind_class=wind_class, free_class=np.array(kfree_l, dtype=np.int8), free_dist=np.array(fdist_l, dtype=np.float32),
        cell_theta=theta.astype(np.float32), cut_edges=cut_edges, meta=json.dumps(meta),
    )
    for key in ("node_cell", "surface_points"):
        if key in z:
            out[key] = np.asarray(z[key])
    if verbose:
        print(json.dumps(stats, indent=1), flush=True)
    return out


def torus_from_obj(shape: str, torus_idx: int) -> dict:
    """Exact torus frame from the OBJ mesh (projects/manifold_features/data.py)."""
    sys.path.insert(0, str(HERE.parent / "manifold_features"))
    from data import fit_tori_from_obj  # noqa: E402
    t = fit_tori_from_obj(shape)[torus_idx]
    return dict(center=t["center"], axis=t["axis"], R=t["R"], r=t["r"])


def main(a: argparse.Namespace) -> None:
    inp = Path(a.inp) if a.inp else GEN / f"{a.shape}.npz"
    out = Path(a.out) if a.out else GEN / f"{a.shape}_wind.npz"
    z = np.load(inp, allow_pickle=True)
    if "tori" in z.files:                                     # analytic manifold: frame stored in the dataset
        t = np.asarray(z["tori"])[a.torus]
        torus = dict(center=t[:3], axis=t[3:6], R=float(t[6]), r=float(t[7]))
    else:
        torus = torus_from_obj(a.shape, a.torus)
    print(f"{a.shape}: torus centre {np.round(torus['center'], 4).tolist()} axis {np.round(torus['axis'], 4).tolist()} R {torus['R']:.3f} r {torus['r']:.3f}", flush=True)
    res = generate(z, torus, K=a.K, n_goals=a.n_goals, starts_per_goal=a.starts_per_goal, min_dist=a.min_dist,
                   heldout_frac=a.heldout_frac, seed=a.seed, tube_tol=a.tube_tol, max_len=a.max_len)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    print(f"saved {out} ({len(res['lengths'])} paths, block ~{int(res['lengths'].max()) + 8})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shape", default="single_ring")
    ap.add_argument("--torus", type=int, default=0, help="index of the torus whose ring axis defines the cut")
    ap.add_argument("--K", type=int, default=1, help="classes k in [-K, K]")
    ap.add_argument("--n_goals", type=int, default=1000)
    ap.add_argument("--starts_per_goal", type=int, default=150)
    ap.add_argument("--min_dist", type=float, default=0.15, help="minimum free geodesic length")
    ap.add_argument("--heldout_frac", type=float, default=0.1, help="fraction of goal cells held out entirely")
    ap.add_argument("--tube_tol", type=float, default=0.15, help="cells within this distance of the torus surface can cross the cut")
    ap.add_argument("--max_len", type=int, default=0, help="drop paths longer than this many cells (0 = keep all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inp", default=None, help="input npz (default generated/<shape>.npz)")
    ap.add_argument("--out", default=None, help="output npz (default generated/<shape>_wind.npz)")
    main(ap.parse_args())
