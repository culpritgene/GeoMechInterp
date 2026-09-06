"""Point-cloud loading for the SAE-vs-spline parameter-efficiency experiment.

Source: data/mesh_datasets/combined_pointcloud_dataset_v3 (HF Arrow), plus the
per-shape torus parameters in data/meshes/<name>.json.  Every shape is a union
of tori (R=1, r=0.25); the OBJ vertices are in the same frame as the sampled
points.

Torus parameters are fitted from the OBJ vertices (see fit_tori_from_obj); the
JSON centres are wrong for the chain_link shapes.  Labels are derived
analytically from the union-of-tori signed distance so that the task is well
defined:
    0 = exterior  (sdf >  eps, off-manifold row)
    1 = surface   (on_manifold row with |sdf| < surf_tol)
    2 = interior  (sdf < -eps, off-manifold row)
Off-manifold rows inside the +-eps band and on-manifold rows far from any torus
(mesh seam artifacts) are dropped and counted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
ARROW = DATA / "mesh_datasets" / "combined_pointcloud_dataset_v3" / "data-00000-of-00001.arrow"
CACHE = DATA / "mesh_datasets" / "cache_v3"

SHAPES = [
    "single_ring", "stacked_equal_sign_two", "double_touching_in_plane",
    "double_intersecting_in_plane", "chain_link_two", "chain_link_three",
    "chain_link_four", "pretzel_three_in_plane", "plus_shape", "sphere_skeleton",
    "circle_border_1", "circle_border_2", "circle_border_3",
    "keyring_1", "keyring_2", "keyring_3",
]
CLASS_NAMES = ["exterior", "surface", "interior"]


def parse_obj_vertices(path: Path) -> np.ndarray:
    return np.array([[float(v) for v in l.split()[1:4]] for l in open(path) if l.startswith("v ")])


def fit_tori_from_obj(name: str) -> list[dict]:
    """Fit torus parameters from the exported OBJ vertices.

    The per-shape JSON lists torus transforms, but for the chain_link shapes
    those centres are wrong (they imply a smaller extent than the mesh bounds),
    so the mesh itself is treated as ground truth.  Vertices come in blocks of
    major_segments * minor_segments, one block per torus; each block is fitted
    with centre = mean vertex, axis = least-variance direction, R = mean axial
    distance, r = sqrt(2 * mean(h^2)).  The fit residual is exactly 0 for all
    16 shapes.
    """
    meta = json.load(open(DATA / "meshes" / f"{name}.json"))
    g = meta["generator"]
    blk = g["major_segments"] * g["minor_segments"]
    V = parse_obj_vertices(DATA / "meshes" / f"{name}.obj")
    assert len(V) % blk == 0, (name, len(V), blk)
    tori = []
    for i in range(len(V) // blk):
        B = V[i * blk:(i + 1) * blk]
        c = B.mean(0)
        _, _, vt = np.linalg.svd(B - c)
        ax = vt[2]
        h = (B - c) @ ax
        rho = np.linalg.norm((B - c) - np.outer(h, ax), axis=1)
        R, r = float(rho.mean()), float(np.sqrt(2 * np.mean(h ** 2)))
        resid = float(np.abs(np.sqrt((rho - R) ** 2 + h ** 2) - r).max())
        assert resid < 1e-3, (name, i, resid)
        tori.append({"center": c, "axis": ax, "R": R, "r": r})
    return tori


def torus_frames(tori: list[dict]):
    """Return centers (T,3), unit axes (T,3), R (T,), r (T,)."""
    return (np.array([t["center"] for t in tori]), np.array([t["axis"] for t in tori]),
            np.array([t["R"] for t in tori]), np.array([t["r"] for t in tori]))


def union_sdf(P: np.ndarray, tori: list[dict]):
    """Signed distance to the union of tori and, per point, the nearest torus'
    Gaussian curvature at the radial projection of the point onto its surface.

    For a torus, K = cos(v) / (r (R + r cos v)) with cos v = (rho - R) / r,
    where rho is the distance from the torus axis.  K < 0 on the inner (hole)
    side, K > 0 on the outer side.
    """
    C, A, R, r = torus_frames(tori)
    best = np.full(len(P), np.inf)
    curv = np.zeros(len(P))
    for c, a, Rj, rj in zip(C, A, R, r):
        v = P - c
        h = v @ a
        rho = np.linalg.norm(v - np.outer(h, a), axis=1)
        d = np.sqrt((rho - Rj) ** 2 + h ** 2) - rj
        cosv = np.clip((rho - Rj) / np.maximum(np.sqrt((rho - Rj) ** 2 + h ** 2), 1e-9), -1, 1)
        K = cosv / (rj * (Rj + rj * cosv))
        upd = d < best
        best[upd] = d[upd]; curv[upd] = K[upd]
    return best, curv


@dataclass
class ShapeData:
    name: str
    X: np.ndarray          # (N,3) float32, normalised to [-1,1]^3
    X_raw: np.ndarray      # (N,3) float32, original coordinates
    y: np.ndarray          # (N,) int64 class labels
    sdf: np.ndarray        # (N,) float32 signed distance to surface
    gauss_curv: np.ndarray # (N,) float32 Gaussian curvature of nearest torus
    in_cavity: np.ndarray  # (N,) bool
    split: np.ndarray      # (N,) str in {train, validation, test}
    center: np.ndarray
    half_extent: float
    dropped: dict

    def subset(self, split: str):
        m = self.split == split
        return self.X[m], self.y[m], self.sdf[m], self.gauss_curv[m], self.in_cavity[m]


def _load_arrow():
    import pyarrow as pa
    with open(ARROW, "rb") as f:
        return pa.ipc.open_stream(f).read_all().to_pandas()


def build_shape(name: str, eps: float = 0.005, surf_tol: float = 0.01, df=None) -> ShapeData:
    df = _load_arrow() if df is None else df
    s = df[df.dataset_name == name]
    P = np.stack(s.x.values).astype(np.float64)
    on = s.on_manifold.values.astype(bool)
    sdf, curv = union_sdf(P, fit_tori_from_obj(name))

    bad_surface = on & (np.abs(sdf) > surf_tol)
    band = (~on) & (np.abs(sdf) < eps)
    keep = ~(bad_surface | band)

    y = np.where(on, 1, np.where(sdf < 0, 2, 0)).astype(np.int64)
    P = P[keep]; y = y[keep]; sdf = sdf[keep]; curv = curv[keep]
    lo, hi = P.min(0), P.max(0)
    center = (lo + hi) / 2
    half = float((hi - lo).max() / 2) * 1.02
    X = (P - center) / half
    return ShapeData(
        name=name, X=X.astype(np.float32), X_raw=P.astype(np.float32), y=y,
        sdf=sdf.astype(np.float32), gauss_curv=curv.astype(np.float32),
        in_cavity=s.in_cavity.values[keep].astype(bool), split=s.split.values[keep].astype(str),
        center=center, half_extent=half,
        dropped={"bad_surface": int(bad_surface.sum()), "eps_band": int(band.sum()), "total": int(len(keep))},
    )


def load_shape(name: str, use_cache: bool = True, **kw) -> ShapeData:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npz"
    if use_cache and f.exists():
        z = np.load(f, allow_pickle=True)
        return ShapeData(**{k: (z[k].item() if z[k].dtype == object and z[k].shape == () else z[k]) for k in z.files})
    sd = build_shape(name, **kw)
    np.savez_compressed(f, **{k: (np.array(v, dtype=object) if isinstance(v, dict) else v) for k, v in sd.__dict__.items()})
    return sd


def build_all(shapes=SHAPES, **kw):
    df = _load_arrow()
    out = {}
    for n in shapes:
        sd = build_shape(n, df=df, **kw)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE / f"{n}.npz", **{k: (np.array(v, dtype=object) if isinstance(v, dict) else v) for k, v in sd.__dict__.items()})
        out[n] = sd
    return out


if __name__ == "__main__":
    for n, sd in build_all().items():
        tr = (sd.split == "train").sum()
        cls = np.bincount(sd.y, minlength=3)
        print(f"{n:30s} N={len(sd.y):6d} train={tr:6d} classes ext/surf/int={cls.tolist()} dropped={sd.dropped}")
