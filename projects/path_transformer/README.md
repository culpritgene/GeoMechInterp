# Path transformers on oct-tree tokenised surfaces

**Experiment 2.** Train small decoder-only transformers to produce the
shortest surface path between two points on a torus-product shape, then use
the trained models as a substrate for comparing SAE features against spline
probes. A model is only useful for that if it has actually learned the global
geometry: generated paths must be valid (adjacent occupied cells), reach the
goal, and be close to geodesic length. Those are the reported metrics.

## Data

Shipped per-shape geodesic sets (`data/geodesic_datasets/combined_geodesics_v1`)
contain only ~4k train paths, so `gen_paths.py` regenerates paths from the
shipped surface samples (11,000 points), kNN manifold graph (k=11) and oct-tree
(depth 5): 3,000 random sources x 100 random targets = 300k exact Dijkstra
paths per shape, tokenised identically to the shipped data (verified) and with
consecutive duplicate cells removed. Output lives in
`data/geodesic_datasets/generated/<shape>.npz` (symlink to `/var/tmp`, since
`/home` is full) and includes the cell table, cell centres and the cell
adjacency graph used for validation.

Note: the shipped kNN graph is connected even for interlocked chain links, so
geodesics may hop between links where they nearly touch.

## Sequence formats (`paths_data.py`)

- `hier`: each waypoint is its five oct-tree level tokens (the shipped
  44-token vocabulary); `BOS s0..s4 SEP g0..g4 SEP w.. SEP w.. ... EOS`,
  block size ~330.
- `flat`: one token per cell (vocab = 4 + #cells, 1,100-4,200);
  `BOS s SEP g SEP w1 .. wL EOS`, block size ~60. Cheaper to train and gives a
  token embedding per location, convenient for interpretability.

Loss is taken only on the path tokens after the goal.

## Trained models (6 layers, d=256, 8 heads, 30k steps of batch 256, held-out start/goal pairs)

| shape | format | params | parsed | valid | reached | success | length / geodesic |
|---|---|---|---|---|---|---|---|
| chain_link_two | flat | 5.08M | 1.000 | 1.000 | 1.000 | 1.000 | 1.002 |
| chain_link_two | hier | 4.83M | 0.996 | 0.988 | 0.996 | 0.988 | 1.004 |
| keyring_1 | flat | 5.08M | 1.000 | 0.998 | 1.000 | 0.998 | 1.010 |
| keyring_1 | hier | 4.83M | 0.996 | 0.994 | 0.996 | 0.994 | 1.001 |
| single_ring | flat | 5.24M | 0.999 | 0.998 | 0.999 | 0.998 | 1.016 |
| single_ring | hier | 4.83M | 0.994 | 0.982 | 0.994 | 0.982 | 1.008 |

Both formats learn the task; the flat format trains ~6x faster (block 59 vs 332) and is what the probes use.

## Environment (rebuild after a reboot)

`/var/tmp` (the overlay filesystem) is wiped when the machine reboots, taking
the venv, generated data, checkpoints and any results left there. Results that
matter are copied into `results/runs/` by `sweep.sh`. Rebuild with:

```bash
mkdir -p /var/tmp/venvs /var/tmp/uv_cache /var/tmp/geomech_data/generated
export UV_CACHE_DIR=/var/tmp/uv_cache UV_LINK_MODE=copy
uv --no-config venv /var/tmp/venvs/geomechinterp --python 3.12
uv --no-config pip install --python /var/tmp/venvs/geomechinterp/bin/python --index-url https://pypi.org/simple \
    torch numpy pyarrow datasets scikit-learn scipy pandas matplotlib tqdm einops networkx sympy
uv --no-config pip install --python /var/tmp/venvs/geomechinterp/bin/python --index-url https://pypi.org/simple --no-deps -e .
bash projects/path_transformer/sweep.sh      # resumable: regenerates data, models, probes, winding and group runs
```

## Training

```bash
.venv/bin/python projects/path_transformer/gen_paths.py --shapes all
.venv/bin/python projects/path_transformer/train_path.py --shape single_ring --fmt flat --steps 30000
```

Checkpoints and `summary.json` go to `/var/tmp/geomech_ckpt/<shape>_<fmt>_L<layers>_d<dim>/`.
Evaluation decodes greedily on held-out (start, goal) pairs and reports
`parsed`, `valid`, `reached`, `success` (valid and reached) and `len_ratio`
(polyline length through cell centres relative to the optimal path).

## Residual-stream geometry (`viz_resid.py`)

`results/resid_ring_projection.png`: ridge projection of the residual onto
(cos, sin) of the ring angle for `single_ring`. At the embedding layer the
ring is a clean linear circle (held-out R² 0.98); from layer 1 onward it
collapses into a filled disc (R² 0.73-0.78): the angle survives linearly but
the radius is taken over by path context. The residual is high-dimensional
throughout (100-155 of 256 directions for 90% variance; the top two PCs hold
4-7%), so position is not confined to a small linear subspace.

## Probing (`probe.py`, `summarize_probes.py`)

For each path token of held-out sequences we decode, from the layer-l
residual: current cell centre (`pos`), goal cell centre (`goal`), next cell
centre (`next`) and normalised remaining distance (`remain`). Probe families
(train/val/test split by sequence, 40k training tokens, 3k steps):

| probe | form | params (d=256, m units) |
|---|---|---|
| linear | ridge | 771 |
| top-k SAE + linear | unsupervised top-k SAE (m in 256..4096, k in 8..128), ridge on codes | 66k-1.06M |
| ReLU hinges + linear | supervised, sum_j w_j ReLU(a_j·r + b_j) | 7m + ... |
| cubic splines + linear | same with each hinge replaced by a cubic B-spline (G=8) | ~8m |
| 2-layer cubic KAN | projection to k dims, KAN k -> h -> out | |

Mean over the three shapes of the best held-out R² at or below a parameter
budget, layer 4 (`results/probes_probes_v2_L4.md`; layers 2 and 6 alongside):

| target | probe | <= 1000 | <= 2500 | <= 5000 | <= 10000 | <= 40000 | <= 300000 |
|---|---|---|---|---|---|---|---|
| pos | linear | 0.638 | | | | | |
| pos | top-k SAE + linear | - | - | - | - | - | 0.845 |
| pos | ReLU hinges + linear | 0.449 | 0.762 | 0.807 | 0.840 | 0.871 | 0.888 |
| pos | cubic splines + linear | 0.521 | 0.764 | 0.805 | 0.831 | 0.831 | 0.831 |
| pos | 2-layer cubic KAN | 0.532 | 0.711 | 0.711 | 0.731 | 0.756 | 0.756 |
| goal | linear | 0.468 | | | | | |
| goal | top-k SAE + linear | - | - | - | - | - | 0.610 |
| goal | ReLU hinges + linear | 0.314 | 0.544 | 0.659 | 0.702 | 0.756 | 0.792 |
| goal | cubic splines + linear | 0.344 | 0.567 | 0.630 | 0.675 | 0.677 | 0.677 |
| next | linear | 0.635 | | | | | |
| next | top-k SAE + linear | - | - | - | - | - | 0.846 |
| next | ReLU hinges + linear | 0.447 | 0.764 | 0.807 | 0.842 | 0.872 | 0.887 |
| next | cubic splines + linear | 0.523 | 0.770 | 0.813 | 0.834 | 0.835 | 0.835 |

(the cubic-spline probe was swept to m=128, i.e. ~37k parameters; the ReLU
probe to m=512.)

### v3: fixed spline probe, six models

The v2 spline probe squashed its projection with tanh, which crippled it at
width (m=128: 0.71-0.78 vs 0.85 for ReLU on single_ring, layer 4). Removing
the squash and using a B-spline grid over [-3, 3] on standardised inputs gives
0.849-0.854 at the same width, above the ReLU probe (0.846). All v3 numbers
use the fixed probe, add the unit step direction (`dir`) as a target, and
include the 0.24M-parameter 3-layer d=64 models (which solve the task as well
as the 5M ones: 98.4-99.7% success).

Layer 2, mean over all six models (`results/probes_probes_v3_L2.md`):

| target | probe | <= 1000 | <= 2500 | <= 5000 | <= 10000 | <= 40000 | <= 300000 |
|---|---|---|---|---|---|---|---|
| pos | linear | 0.635 | | | | | |
| pos | top-k SAE + linear | - | - | - | - | - | 0.925 |
| pos | ReLU hinges + linear | 0.604 | 0.848 | 0.889 | 0.915 | 0.941 | 0.945 |
| pos | cubic splines + linear | 0.694 | 0.854 | 0.904 | 0.934 | 0.950 | 0.950 |
| goal | linear | 0.569 | | | | | |
| goal | ReLU hinges + linear | 0.507 | 0.744 | 0.788 | 0.817 | 0.854 | 0.859 |
| goal | cubic splines + linear | 0.589 | 0.747 | 0.796 | 0.832 | 0.850 | 0.850 |
| dir | linear | 0.220 | | | | | |
| dir | top-k SAE + linear | - | - | - | - | - | 0.510 |
| dir | ReLU hinges + linear | 0.249 | 0.416 | 0.486 | 0.550 | 0.636 | 0.665 |
| dir | cubic splines + linear | 0.272 | 0.388 | 0.463 | 0.525 | 0.586 | 0.586 |
| remain | linear | 0.802 | | | | | |
| remain | ReLU hinges + linear | 0.832 | 0.874 | 0.892 | 0.899 | 0.908 | 0.911 |
| remain | cubic splines + linear | 0.849 | 0.884 | 0.896 | 0.900 | 0.900 | 0.900 |

Small (d=64) models, layer 3, position: linear 0.50, ReLU 0.64 / 0.80 / 0.88
and cubic splines 0.68 / 0.81 / 0.88 at <= 1k / 5k / 40k params; SAE + linear
0.61 at 40k and 0.84 at 300k.

Takeaways (v3):

- With a fair probe, cubic splines match or modestly beat hinges on the
  smooth geometric targets (position, remaining distance) at every budget,
  and hinges win on the decision-like targets (step direction; goal at wide
  budgets). Smooth features favour splines, piecewise/categorical decisions
  favour hinges; neither gap is large.
- The unsupervised top-k SAE trails both supervised families by roughly an
  order of magnitude in parameters for every target.
- Compression (d=64) makes the code more non-linear (linear R^2 for position
  drops from 0.64 to 0.50 and non-linear probes add +0.38), but does not widen
  the spline-vs-hinge margin.

Earlier (v2, tanh spline probe) takeaways, kept for the record:

- Geometry in the residual stream is only partly linear: a linear probe
  explains 64-74% of position variance in the middle layers, and a small
  non-linear probe adds 0.2-0.25 R².
- In activation space the input-space advantage of cubic splines mostly
  disappears. Splines win at the smallest budgets (<= 1000 params: 0.52 vs
  0.45 for position), tie around 2,500-5,000, and the ReLU probe pulls ahead
  when wider (0.87 vs 0.83 at ~35k). The residual encodes position in a
  distributed, hinge-friendly way rather than as a low-dimensional curved
  embedding.
- A top-k SAE fitted unsupervised exposes the geometric features, but at a
  high price: to match a 4k-parameter supervised ReLU probe it needs a 1024-
  wide dictionary (266k parameters), and its codes never beat the supervised
  probes. For `goal` the SAE codes are much worse (0.61 vs 0.79).
- Position is decoded best from layer 2 (R² 0.93 for the ReLU probe) and
  degrades toward the output (0.87 at layer 6), while `goal` is progressively
  consumed (0.89 at layer 2, 0.64 at layer 6): the model converts
  (position, goal) into the next step.


## Manifold sweep (2026-09-07; `sweep.sh`, `sweep_analysis.py`, `results/sweep_summary.md`)

Analytic torus products (`gen_manifold.py`) varying minor radius r (hole size
and curvature), sample density, oct-tree depth, surface-shell noise, and
path noise (`gen_paths.py --temperature`, `--p_obs`), two model sizes.

- Task difficulty for the 0.24M model scales with the number of occupied
  cells, not curvature: the thin tube r=0.12 (1,117 cells) is easiest
  (99.6%), r=0.50 (2,470 cells) 95.9%, depth 6 (5,369 cells) 37%; the 5M
  model stays at 98.7-100% everywhere.
- Noise: shell displacement up to 0.6r costs 3-4 points (d=64) / 1 (d=256);
  stochastic training paths at temperature 0.25 still yield greedy paths
  5-7% over geodesic (the model denoises); observation noise p=0.15 costs
  4 points at d=64 and 9 at d=256 (the big model imitates the noise).
- Probes (middle layer): the spline-minus-hinge gap for position stays within
  +-0.05 on every axis; linearity of the position code falls with the number
  of cells (ridge 0.55 -> 0.22) but stays hinge-friendly; path and
  observation noise LINEARISE the code (ridge 0.53 -> 0.73).
- Stratified probes with the fixed spline probe (`probe_strata.py`): near
  link contacts the global spline probe beats hinges by +0.09..+0.24; local
  probes find the region near-linear (0.79-0.93), i.e. a locally linear
  patch that a global linear map cannot fit.
