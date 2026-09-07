"""Explicit finite groups for Cayley-walk state tracking (language-ladder rung 2).

Every group is an explicit multiplication table over integer element ids plus
exact real irreducible-representation matrices and coordinates for each element:

  Z_n        element a in 0..n-1, table (a + b) mod n.
             Coordinates (cos 2 pi f a / n, sin 2 pi f a / n) for f = 1..F.
  Z_n x Z_m  element (a, b) -> a * m + b.  Two Fourier circles, one per factor.
  D_n        element r^a s^b -> b * n + a (order 2n) with the law
             r^a s^b . r^c s^d = r^(a + (-1)^b c) s^(b + d).
             2-D irreps rho_f(r^a s^b) = R(2 pi f a / n) diag(1, (-1)^b), f = 1..F,
             whose four entries are (c, -eps s, s, eps c); the coordinates store
             (c, s, eps c, eps s) per frequency plus the sign character eps = (-1)^b.

Walks multiply on the right: e_i = e_{i-1} . g_i, so the prefix product after
generator i is e_0 g_1 ... g_i.  `verify(group)` checks closure (Latin square),
associativity, identity, inverses and the homomorphism rho(g) rho(h) = rho(g h)
for every listed irrep over the whole table, with numpy.

    python projects/group_tracking/groups.py --group D36   # verify and print
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FourierAxis:
    """A cyclic coordinate of the element (used for FFT of embeddings and for
    the 'unused harmonic' probe target).  `index[g]` is the coordinate of
    element g (period `period`); `other[g]` labels the remaining structure so
    that (index, other) identifies g uniquely."""
    name: str
    index: np.ndarray
    period: int
    other: np.ndarray


@dataclass
class Group:
    name: str
    order: int
    table: np.ndarray                 # (order, order) int64, table[g, h] = g . h
    identity: int
    inverse: np.ndarray               # (order,) int64
    element_names: list[str]
    generators: np.ndarray            # (n_gen,) element ids used by the walks
    generator_names: list[str]
    generator_probs: np.ndarray       # (n_gen,) sampling probabilities
    irreps: dict[str, np.ndarray]     # irrep name -> (order, k, k) real matrices
    coords: np.ndarray                # (order, D) exact irrep coordinates
    coord_names: list[str]
    coord_groups: dict[str, list[int]]  # target name -> coordinate columns
    axes: list[FourierAxis]
    n_freq: int
    params: dict = field(default_factory=dict)   # constructor arguments (for round trips)

    # ---- algebra --------------------------------------------------------
    def mul(self, g, h):
        return self.table[g, h]

    def product(self, elements) -> int:
        e = self.identity
        for g in elements:
            e = int(self.table[e, g])
        return e

    def prefix_products(self, e0: int, word) -> np.ndarray:
        """Elements after each generator of `word` (generator indices), starting at e0."""
        out = np.empty(len(word) + 1, dtype=np.int64)
        out[0] = e0
        for i, j in enumerate(word):
            out[i + 1] = self.table[out[i], self.generators[j]]
        return out

    def commute(self, g: int, h: int) -> bool:
        return bool(self.table[g, h] == self.table[h, g])

    @property
    def is_abelian(self) -> bool:
        return bool((self.table == self.table.T).all())

    @property
    def n_gen(self) -> int:
        return len(self.generators)

    @property
    def sign(self) -> np.ndarray | None:
        """The sign character eps(g) for D_n, None otherwise."""
        cols = self.coord_groups.get("eps")
        return None if cols is None else self.coords[:, cols[0]]


# ---------------------------------------------------------------- helpers
def _rot(theta: np.ndarray) -> np.ndarray:
    """(N,) angles -> (N, 2, 2) rotation matrices."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def _check_freqs(n_freq: int, n: int) -> None:
    if not (1 <= n_freq <= n // 2):
        raise ValueError(f"n_freq must be in [1, {n // 2}] for a cycle of length {n}, got {n_freq}")


# ---------------------------------------------------------------- Z_n
def cyclic(n: int, n_freq: int = 3, steps=(1, 2, 3)) -> Group:
    _check_freqs(n_freq, n)
    if max(steps) * 2 >= n:
        raise ValueError("generator steps must be distinct modulo n")
    a = np.arange(n)
    table = (a[:, None] + a[None, :]) % n
    inverse = (-a) % n
    irreps, coords, names, groups = {}, [], [], {}
    for f in range(1, n_freq + 1):
        th = 2 * np.pi * f * a / n
        irreps[f"rho{f}"] = _rot(th)
        groups[f"f{f}"] = [len(names), len(names) + 1]
        coords += [np.cos(th), np.sin(th)]; names += [f"f{f}_cos", f"f{f}_sin"]
    groups["irrep"] = list(range(len(names)))
    gens, gnames = [], []
    for s in steps:
        gens += [s % n, (-s) % n]; gnames += [f"+{s}", f"-{s}"]
    return Group(name=f"Z{n}", order=n, table=table, identity=0, inverse=inverse,
                 element_names=[str(i) for i in a], generators=np.array(gens), generator_names=gnames,
                 generator_probs=np.full(len(gens), 1 / len(gens)), irreps=irreps,
                 coords=np.stack(coords, 1), coord_names=names, coord_groups=groups,
                 axes=[FourierAxis("a", a.copy(), n, np.zeros(n, dtype=np.int64))], n_freq=n_freq,
                 params=dict(n_freq=n_freq))


# ---------------------------------------------------------------- Z_n x Z_m
def torus(n: int, m: int, n_freq: int = 3, diagonals: bool = False) -> Group:
    _check_freqs(n_freq, n); _check_freqs(n_freq, m)
    order = n * m
    g = np.arange(order); a, b = g // m, g % m
    table = ((a[:, None] + a[None, :]) % n) * m + (b[:, None] + b[None, :]) % m
    inverse = ((-a) % n) * m + (-b) % m
    irreps, coords, names, groups = {}, [], [], {}
    for axis, idx, period in (("a", a, n), ("b", b, m)):
        for f in range(1, n_freq + 1):
            th = 2 * np.pi * f * idx / period
            irreps[f"rho_{axis}{f}"] = _rot(th)
            groups[f"{axis}_f{f}"] = [len(names), len(names) + 1]
            coords += [np.cos(th), np.sin(th)]; names += [f"{axis}_f{f}_cos", f"{axis}_f{f}_sin"]
    groups["irrep"] = list(range(len(names)))
    groups["irrep_a"] = [i for i, nm in enumerate(names) if nm.startswith("a_")]
    groups["irrep_b"] = [i for i, nm in enumerate(names) if nm.startswith("b_")]
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonals:
        moves += [(1, 1), (-1, -1), (1, -1), (-1, 1)]
    gens = [((da % n) * m + (db % m)) for da, db in moves]
    gnames = [f"({da:+d},{db:+d})" for da, db in moves]
    return Group(name=f"T{n}x{m}", order=order, table=table, identity=0, inverse=inverse,
                 element_names=[f"({x},{y})" for x, y in zip(a, b)], generators=np.array(gens), generator_names=gnames,
                 generator_probs=np.full(len(gens), 1 / len(gens)), irreps=irreps,
                 coords=np.stack(coords, 1), coord_names=names, coord_groups=groups,
                 axes=[FourierAxis("a", a, n, b), FourierAxis("b", b, m, a)], n_freq=n_freq,
                 params=dict(n_freq=n_freq, diagonals=diagonals))


# ---------------------------------------------------------------- D_n
def dihedral(n: int, n_freq: int = 3, p_reflect: float = 0.2, steps=(1, 2, 3)) -> Group:
    _check_freqs(n_freq, n)
    if max(steps) * 2 >= n:
        raise ValueError("generator steps must be distinct modulo n")
    order = 2 * n
    g = np.arange(order); a, b = g % n, g // n           # element r^a s^b -> b * n + a
    sgn = 1 - 2 * b                                       # (-1)^b
    # r^a s^b . r^c s^d = r^(a + (-1)^b c) s^(b + d)
    aa = (a[:, None] + sgn[:, None] * a[None, :]) % n
    bb = (b[:, None] + b[None, :]) % 2
    table = bb * n + aa
    inverse = np.where(b == 0, (-a) % n, g)               # reflections are involutions
    irreps, coords, names, groups = {}, [], [], {}
    S = np.stack([np.diag([1.0, float(s)]) for s in sgn])   # (order, 2, 2)
    for f in range(1, n_freq + 1):
        th = 2 * np.pi * f * a / n
        irreps[f"rho{f}"] = _rot(th) @ S
        c, s = np.cos(th), np.sin(th)
        groups[f"f{f}"] = list(range(len(names), len(names) + 4))
        coords += [c, s, sgn * c, sgn * s]
        names += [f"f{f}_cos", f"f{f}_sin", f"f{f}_eps_cos", f"f{f}_eps_sin"]
    groups["irrep"] = list(range(len(names)))
    irreps["eps"] = sgn.astype(float)[:, None, None]
    groups["eps"] = [len(names)]
    coords.append(sgn.astype(float)); names.append("eps")
    gens, gnames, probs = [], [], []
    for s_ in steps:
        gens += [s_ % n, (-s_) % n]; gnames += [f"r^+{s_}", f"r^-{s_}"]; probs += [(1 - p_reflect) / (2 * len(steps))] * 2
    gens += [n, n + (n - 1)]                              # s = r^0 s ; s r = r^{-1} s
    gnames += ["s", "sr"]; probs += [p_reflect / 2] * 2
    return Group(name=f"D{n}", order=order, table=table, identity=0, inverse=inverse,
                 element_names=[f"r^{x}s^{y}" for x, y in zip(a, b)], generators=np.array(gens), generator_names=gnames,
                 generator_probs=np.array(probs), irreps=irreps,
                 coords=np.stack(coords, 1), coord_names=names, coord_groups=groups,
                 axes=[FourierAxis("a", a, n, b)], n_freq=n_freq,
                 params=dict(n_freq=n_freq, p_reflect=p_reflect))


# ---------------------------------------------------------------- factory / verification
def make_group(name: str, n_freq: int = 3, p_reflect: float = 0.2, diagonals: bool = False) -> Group:
    """Z<n>, T<n>x<m> or D<n>; e.g. Z36, Z360, T36x12, D36."""
    if m := re.fullmatch(r"Z(\d+)", name):
        return cyclic(int(m[1]), n_freq)
    if m := re.fullmatch(r"T(\d+)x(\d+)", name):
        return torus(int(m[1]), int(m[2]), n_freq, diagonals)
    if m := re.fullmatch(r"D(\d+)", name):
        return dihedral(int(m[1]), n_freq, p_reflect)
    raise ValueError(f"unknown group name {name!r}")


def verify(G: Group, atol: float = 1e-9) -> dict:
    """Assert the table is a group and every irrep is a homomorphism.  Returns
    a dict of the checks performed (all True if it returns)."""
    T, n = G.table, G.order
    ar = np.arange(n)
    assert T.shape == (n, n) and T.dtype.kind in "iu", "table shape/dtype"
    assert T.min() >= 0 and T.max() < n, "closure: entries outside the group"
    assert (np.sort(T, 1) == ar).all() and (np.sort(T, 0) == ar[:, None]).all(), "closure: not a Latin square"
    assert (T[G.identity] == ar).all() and (T[:, G.identity] == ar).all(), "identity"
    assert (T[ar, G.inverse] == G.identity).all() and (T[G.inverse, ar] == G.identity).all(), "inverses"
    for lo in range(0, n, 64):                             # (ab)c == a(bc), chunked over a
        sl = slice(lo, min(n, lo + 64))
        lhs = T[T[sl][:, :, None], ar[None, None, :]]
        rhs = T[ar[sl][:, None, None], T[None, :, :]]
        assert (lhs == rhs).all(), "associativity"
    for name, R in G.irreps.items():
        prod = np.einsum("gij,hjk->ghik", R, R)
        assert np.allclose(prod, R[T], atol=atol), f"irrep {name} is not a homomorphism"
        assert np.allclose(R[G.identity], np.eye(R.shape[1]), atol=atol), f"irrep {name} at identity"
    assert G.coords.shape == (n, len(G.coord_names)), "coordinate table shape"
    assert abs(G.generator_probs.sum() - 1) < 1e-12 and (G.generator_probs > 0).all(), "generator probabilities"
    assert len(set(G.generators.tolist())) == len(G.generators), "duplicate generators"
    return dict(order=n, closure=True, associative=True, identity=True, inverses=True,
                irreps={k: True for k in G.irreps}, abelian=G.is_abelian)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="D36")
    ap.add_argument("--n_freq", type=int, default=3)
    ap.add_argument("--p_reflect", type=float, default=0.2)
    ap.add_argument("--diagonals", action="store_true")
    a = ap.parse_args()
    G = make_group(a.group, a.n_freq, a.p_reflect, a.diagonals)
    print(verify(G))
    print(f"{G.name}: order {G.order}, generators {dict(zip(G.generator_names, G.generators.tolist()))}, "
          f"probs {np.round(G.generator_probs, 3).tolist()}, coords {G.coord_names}")
