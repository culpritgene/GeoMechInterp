"""Analytic torus-product manifolds with controllable curvature, sampling
density and surface noise, in the same format the path pipeline consumes.

The shipped Blender shapes all have R=1, r=0.25 and 11,000 clean surface
samples. This generator produces the surface graph directly from the analytic
union of tori so that three axes can be varied:

    curvature / hole size : minor radius r (hole radius = R - r; inner-equator
                            Gaussian curvature K = -1 / (r (R - r)))
    sampling density      : number of surface samples and oct-tree depth
    noise                 : shell thickness (samples displaced off the surface
                            along the normal), see also --temperature and
                            --p_obs in gen_paths.py for path and observation
                            noise

Output /var/tmp/geomech_data/manifolds/<name>.npz with
    points (n,3)   noisy sample positions      normal (n,3)  surface normal
    edges (E,2)    kNN graph (undirected)      weights (E,)  Euclidean length
    bounds (2,3)   cubic oct-tree bounds       depth          oct-tree depth
    sdf (n,)       signed displacement off the surface (0 for clean samples)
    gauss_curv (n,) K of the sampled torus at the sample
    theta (n,)     ring angle of the sample around its torus axis
    phi (n,)       tube angle (0 = outer equator, pi = inner equator)
    torus_id (n,)  which torus the sample came from
    tori (T,8)     centre(3), axis(3), R, r for every torus
    meta           JSON string with the generator arguments

Presets (--preset): ring (one torus), stacked (two parallel tori touching,
like stacked_equal_sign_two), chain (two interlocked tori, like
chain_link_two), plus (two orthogonal tori through each other, like
plus_shape). --R and --r override every torus.

    python projects/path_transformer/gen_manifold.py --preset ring --r 0.4 --name ring_r040
    python projects/path_transformer/gen_manifold.py --preset ring --shell 0.3 --name ring_shell030
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

OUT = Path("/var/tmp/geomech_data/manifolds")


def preset_tori(preset: str, R: float, r: float) -> list[dict]:
    z = np.array([0.0, 0.0, 1.0]); y = np.array([0.0, 1.0, 0.0]); x = np.array([1.0, 0.0, 0.0])
    if preset == "ring":
        return [dict(center=np.zeros(3), axis=y, R=R, r=r)]
    if preset == "stacked":      # two parallel rings touching along the tube (stacked_equal_sign_two)
        return [dict(center=np.array([0, -r, 0.0]), axis=y, R=R, r=r), dict(center=np.array([0, r, 0.0]), axis=y, R=R, r=r)]
    if preset == "chain":        # two interlocked rings (chain_link_two): centres 0.75 apart, axes orthogonal
        d = 0.75 * R
        return [dict(center=np.array([-d, 0, 0.0]), axis=y, R=R, r=r), dict(center=np.array([d, 0, 0.0]), axis=z, R=R, r=r)]
    if preset == "plus":         # two orthogonal rings through each other (plus_shape)
        return [dict(center=np.zeros(3), axis=y, R=R, r=r), dict(center=np.zeros(3), axis=z, R=R, r=r)]
    raise ValueError(preset)


def frame(axis: np.ndarray):
    a = axis / np.linalg.norm(axis)
    h = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(a, h); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    return a, e1, e2


def sample_torus(t: dict, n: int, rng: np.random.Generator):
    """Area-uniform samples on one torus: returns points, normals, K, theta, phi."""
    a, e1, e2 = frame(t["axis"]); R, r = t["R"], t["r"]
    u = rng.uniform(0, 2 * np.pi, 3 * n)
    v = rng.uniform(0, 2 * np.pi, 3 * n)
    keep = rng.uniform(0, 1, 3 * n) < (R + r * np.cos(v)) / (R + r)
    u, v = u[keep][:n], v[keep][:n]
    radial = np.outer(np.cos(u), e1) + np.outer(np.sin(u), e2)
    P = t["center"] + (R + r * np.cos(v))[:, None] * radial + (r * np.sin(v))[:, None] * a
    N = np.cos(v)[:, None] * radial + np.sin(v)[:, None] * a
    K = np.cos(v) / (r * (R + r * np.cos(v)))
    return P, N, K, u, (np.pi - v) % (2 * np.pi)  # phi: 0 at the outer equator, pi at the inner one


def torus_sdf(P: np.ndarray, t: dict) -> np.ndarray:
    a = t["axis"] / np.linalg.norm(t["axis"]); v = P - t["center"]; h = v @ a
    rho = np.linalg.norm(v - np.outer(h, a), axis=1)
    return np.sqrt((rho - t["R"]) ** 2 + h ** 2) - t["r"]


def torus_area(t: dict) -> float:
    return 4 * np.pi ** 2 * t["R"] * t["r"]


def build(args):
    rng = np.random.default_rng(args.seed)
    tori = preset_tori(args.preset, args.R, args.r)
    areas = np.array([torus_area(t) for t in tori]); counts = np.round(args.n_points * areas / areas.sum()).astype(int)
    parts = []
    for i, (t, n) in enumerate(zip(tori, counts)):
        P, N, K, th, ph = sample_torus(t, int(n * 1.5), rng)   # oversample, then drop what is inside another torus
        inside = np.zeros(len(P), bool)
        for j, o in enumerate(tori):
            if j != i:
                inside |= torus_sdf(P, o) < 0
        m = ~inside
        parts.append((P[m][:n], N[m][:n], K[m][:n], th[m][:n], ph[m][:n], np.full(min(n, m.sum()), i)))
    P = np.concatenate([p[0] for p in parts]); N = np.concatenate([p[1] for p in parts]); K = np.concatenate([p[2] for p in parts])
    theta = np.concatenate([p[3] for p in parts]); phi = np.concatenate([p[4] for p in parts]); tid = np.concatenate([p[5] for p in parts])

    # surface noise: displace along the normal within a shell of half-thickness shell * r
    if args.shell > 0:
        s = args.shell * args.r
        d = rng.uniform(-s, s, len(P)) if args.shell_dist == "uniform" else np.clip(rng.normal(0, s / 2, len(P)), -s, s)
    else:
        d = np.zeros(len(P))
    Pn = P + d[:, None] * N

    # kNN graph on the noisy points, symmetric, largest component
    nn = NearestNeighbors(n_neighbors=args.k + 1).fit(Pn)
    dist, idx = nn.kneighbors(Pn)
    rows = np.repeat(np.arange(len(Pn)), args.k); cols = idx[:, 1:].ravel(); w = dist[:, 1:].ravel()
    A = sp.coo_matrix((w, (rows, cols)), shape=(len(Pn), len(Pn))).tocsr(); A = A.maximum(A.T)
    nc, lab = connected_components(A, directed=False)
    keep = lab == np.bincount(lab).argmax()
    if nc > 1:
        print(f"  {nc} components; keeping the largest ({keep.sum()} of {len(keep)} points)")
    A = A[keep][:, keep].tocoo(); Pn, P, N, K, theta, phi, tid, d = (x[keep] for x in (Pn, P, N, K, theta, phi, tid, d))
    upper = A.row < A.col
    edges = np.stack([A.row[upper], A.col[upper]], 1).astype(np.int32); weights = A.data[upper].astype(np.float32)

    half = 1.05 * np.abs(Pn).max()
    bounds = np.array([[-half] * 3, [half] * 3])
    tori_arr = np.array([[*t["center"], *(t["axis"] / np.linalg.norm(t["axis"])), t["R"], t["r"]] for t in tori], dtype=np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = dict(vars(args)); meta["n_kept"] = int(keep.sum()); meta["cell_size"] = float(2 * half / 2 ** args.depth)
    np.savez_compressed(OUT / f"{args.name}.npz", points=Pn.astype(np.float32), clean_points=P.astype(np.float32),
                        normal=N.astype(np.float32), edges=edges, weights=weights, bounds=bounds, depth=args.depth,
                        sdf=d.astype(np.float32), gauss_curv=K.astype(np.float32), theta=theta.astype(np.float32),
                        phi=phi.astype(np.float32), torus_id=tid.astype(np.int8), tori=tori_arr, meta=json.dumps(meta))
    print(f"{args.name}: {len(Pn)} points, {len(edges)} edges, mean edge {weights.mean():.4f}, cell size {meta['cell_size']:.4f} "
          f"(tube diameter / cell = {2 * args.r / meta['cell_size']:.1f}), hole radius {args.R - args.r:.2f}, "
          f"K range [{K.min():.2f}, {K.max():.2f}], shell {args.shell}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="ring", choices=["ring", "stacked", "chain", "plus"])
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--r", type=float, default=0.25)
    ap.add_argument("--n_points", type=int, default=11000)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--k", type=int, default=11)
    ap.add_argument("--shell", type=float, default=0.0, help="shell half-thickness as a fraction of r")
    ap.add_argument("--shell_dist", default="uniform", choices=["uniform", "gauss"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    build(ap.parse_args())
