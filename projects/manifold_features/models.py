"""Model families for the parameter-efficiency sweep.

All models map a 3-D point in [-1,1]^3 to 3 class logits.

  sae      : x -> ReLU(W x + b) -> linear readout, with an L1 penalty on the
             hidden code.  One hidden unit == one hinge == one first-order
             spline knot in a random direction.  This is the SAE encoder used
             as a supervised feature layer.
  relu1    : same architecture with no sparsity penalty (dense 1-hidden ReLU).
  mlp2     : 3 -> h -> h -> 3 ReLU (deeper piecewise-linear reference).
  spline_k : KAN-style network 3 -> h -> 3 where every edge carries a degree-k
             B-spline with G intervals on [-1,1]; tanh squashes hidden units
             back into the grid range.  k=1 is a piecewise-linear spline net,
             k=3 a cubic spline net.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class SparseReLU(nn.Module):
    def __init__(self, d_in: int, m: int, n_cls: int = 3, l1: float = 0.0, topk: int | None = None):
        super().__init__()
        self.enc = nn.Linear(d_in, m)
        self.dec = nn.Linear(m, n_cls)
        self.l1 = l1
        self.topk = topk
        self.last_code = None

    def forward(self, x):
        h = F.relu(self.enc(x))
        if self.topk is not None and self.topk < h.shape[1]:
            idx = h.topk(self.topk, dim=1).indices
            h = torch.zeros_like(h).scatter(1, idx, h.gather(1, idx))
        self.last_code = h
        return self.dec(h)

    def reg(self):
        if self.l1 == 0 or self.last_code is None:
            return 0.0
        return self.l1 * self.last_code.abs().sum(1).mean()

    def sparsity(self):
        return float((self.last_code > 0).float().mean()) if self.last_code is not None else float("nan")


class MLP(nn.Module):
    def __init__(self, d_in: int, h: int, depth: int = 2, n_cls: int = 3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def reg(self):
        return 0.0


def bspline_basis(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """Cox-de Boor.  x: (B, d) in the grid's interior range; grid: (n_knots,)
    uniform, extended by k knots on each side.  Returns (B, d, n_knots-k-1)."""
    x = x.unsqueeze(-1)
    b = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)
    for p in range(1, k + 1):
        left = (x - grid[: -(p + 1)]) / (grid[p:-1] - grid[: -(p + 1)]) * b[..., :-1]
        right = (grid[p + 1 :] - x) / (grid[p + 1 :] - grid[1:-p]) * b[..., 1:]
        b = left + right
    return b


class KANLayer(nn.Module):
    """Every edge (i -> o) carries its own degree-k B-spline on a shared uniform
    grid of G intervals over [-1, 1].  Params: d_out * d_in * (G + k) + d_out."""

    def __init__(self, d_in: int, d_out: int, G: int = 8, k: int = 3, lo: float = -1.0, hi: float = 1.0):
        super().__init__()
        self.k, self.G = k, G
        step = (hi - lo) / G
        grid = torch.linspace(lo - k * step, hi + k * step, G + 2 * k + 1)
        self.register_buffer("grid", grid)
        self.lo, self.hi = lo, hi
        n_b = G + k
        self.coef = nn.Parameter(torch.randn(d_out, d_in, n_b) * (0.5 / (d_in ** 0.5)))
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x):
        x = x.clamp(self.lo, self.hi - 1e-6)
        b = bspline_basis(x, self.grid, self.k)          # B, d_in, n_b
        return torch.einsum("bik,oik->bo", b, self.coef) + self.bias


class SplineNet(nn.Module):
    def __init__(self, d_in: int, h: int, G: int = 8, k: int = 3, n_cls: int = 3, depth: int = 2):
        super().__init__()
        dims = [d_in] + [h] * (depth - 1) + [n_cls]
        self.layers = nn.ModuleList(KANLayer(a, b, G=G, k=k) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x

    def reg(self):
        return 0.0


def make_model(family: str, size: dict, d_in: int = 3) -> nn.Module:
    if family == "sae":
        return SparseReLU(d_in, size["m"], l1=size.get("l1", 1e-3), topk=size.get("topk"))
    if family == "sae_weak":  # 10x weaker sparsity penalty
        return SparseReLU(d_in, size["m"], l1=size.get("l1", 1e-4), topk=size.get("topk"))
    if family == "relu1":
        return SparseReLU(d_in, size["m"], l1=0.0)
    if family == "mlp2":
        return MLP(d_in, size["h"], depth=2)
    if family.startswith("spline"):
        k = int(family[len("spline"):])
        return SplineNet(d_in, size["h"], G=size["G"], k=k)
    raise ValueError(family)


def default_grid(family: str, grid: str = "reduced") -> list[dict]:
    """Sizes for the sweep, roughly log-spaced in parameter count."""
    full = grid == "full"
    if family in ("sae", "sae_weak", "relu1"):
        ms = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] if full else [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        return [{"m": m} for m in ms]
    if family == "mlp2":
        hs = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128] if full else [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
        return [{"h": h} for h in hs]
    if family.startswith("spline"):
        hs = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32] if full else [1, 2, 4, 8, 16, 32]
        return [{"h": h, "G": G} for G in [4, 8, 16] for h in hs]
    raise ValueError(family)
