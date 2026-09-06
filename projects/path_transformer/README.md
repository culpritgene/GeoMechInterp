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

Takeaways:

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
