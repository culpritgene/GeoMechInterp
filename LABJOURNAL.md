# Lab journal

## 2026-09-06 — SAE vs spline networks on torus-product manifolds

Goal: test whether an SAE (a first-order sparse spline: ReLU of a linear
projection) is less parameter-efficient than higher-order spline networks at
modelling curved geometry, first in input space (point clouds), then in the
activation space of transformers trained on the same manifolds.

### Data

- Manifold data lives in GCS under `users/culpritgene/tokenized_manifolds/`
  (`meshes`, `mesh_datasets`, `geodesic_datasets`), not in the prefix that
  `scripts/fetch_data.sh` pulls. Synced into `data/` (gitignored, ~750 MB).
- Point clouds: `combined_pointcloud_dataset_v3`, 16 shapes x 57,600 points.
  `in_shape_volume` is unused (always False); `in_tunnel` is the interior label
  and `in_cavity` marks the ring hole.
- **Bug found:** the per-shape JSON torus centres are wrong for the three
  `chain_link_*` shapes (they imply a smaller extent than the mesh bounds).
  The shipped `in_tunnel` labels for those shapes were computed from the
  JSON and agree with the true interior only 83-90% of the time. Fix: fit
  torus parameters from the OBJ vertices (exact, residual 0 on all 16 shapes)
  and derive labels analytically from the union-of-tori signed distance.
- Surface points lying inside another torus (self-intersections, 0-6% per
  shape) and off-surface points within 0.005 of the surface are dropped so
  the classification task is well defined.
- Oct-tree tokenisation verified: child code = bx | by<<1 | bz<<2, depth 5;
  my tokeniser reproduces the shipped `octree_paths` exactly.
- The shipped kNN surface graphs (k=11, 11,000 points) are connected even for
  interlocked chain links, so geodesics may hop between links where they
  nearly touch.

### Experiment 1 — point classification (`projects/manifold_features`)

Setup: 3-D point -> exterior / surface / interior. Families: SAE (L1 1e-3 and
1e-4), dense 1-hidden ReLU, 2-hidden ReLU MLP, linear-spline KAN, cubic-spline
KAN; sizes swept, full-batch Adam, plateau stopping (up to 15k steps).

Observations:

- A fixed 3,000-step budget under-trained every family (archived in
  `results/steps3000`): with 12k steps a 1024-unit ReLU went 0.965 -> 0.988
  and a 539-param cubic net 0.976 -> 0.999. Ranking unchanged, saturation
  levels very different. Always train to plateau when the claim is about
  parameter efficiency.
- Cubic-spline net reaches 98% on all 16 shapes at ~460 params and 99% on 14
  shapes at ~730. No ReLU family reaches 98% on more than 4 shapes even at
  15-30k params; they keep improving slowly with width (inefficient, not
  incapable).
- The gap is concentrated near the surface: at 1,000 params, near-surface
  accuracy 0.87 (cubic) vs 0.50-0.59 (ReLU families).
- Spline order matters on its own: linear splines sit between ReLU and cubic.
- Negative-curvature (inner) surface is harder than positive-curvature for
  ReLU families (0.90-0.95 vs 0.96-0.98 at <=1k params); splines are uniform.
- The L1 penalty barely matters at small widths and caps the SAE at large
  widths (active fraction 4% at 14k params).

### Experiment 2 — path transformers (`projects/path_transformer`)

Setup: shipped geodesic sets have only ~4k train paths per shape, so 300k
exact Dijkstra paths per shape were regenerated from the shipped surface
graphs (`gen_paths.py`). 6-layer, d=256 GPTs, two sequence formats (flat: one
token per cell; hier: five oct-tree tokens per cell), loss on path tokens only.

Observations:

- Quality bar met: on held-out start/goal pairs, 98.2-100% of generated paths
  are valid (adjacent occupied cells) and reach the goal, with polyline length
  1.00-1.02x geodesic, for single_ring, keyring_1, chain_link_two, both formats.
  Flat trains ~6x faster (block 59 vs 332).
- Residual-stream geometry (single_ring): the ring angle is a clean linear
  circle at the embedding layer (ridge R² 0.98) and collapses to a filled disc
  from layer 1 on (R² 0.73-0.78). The residual is high-dimensional (100-155 of
  256 dims for 90% variance; top-2 PCs hold 4-7%).
- Position is decoded best at layer 2 (ReLU probe R² 0.93) and decays toward
  the output (0.87); goal is progressively consumed (0.89 -> 0.64).
- Probes at matched parameter counts (layer 4, mean over 3 shapes, position):
  linear 0.64; ReLU hinges 0.45 / 0.76 / 0.87 at <=1k / 2.5k / 40k params;
  cubic splines 0.52 / 0.76 / 0.83; top-k SAE + linear 0.85 only at 266k+.
  The input-space spline advantage does not carry over to activation space:
  splines win only at the smallest budgets, ReLU wins when wide. The residual
  encodes geometry in a distributed, hinge-friendly way rather than as a
  low-dimensional curved embedding.
- The SAE story in activation space is inefficiency, not incapacity: an
  unsupervised top-k SAE needs a 1024-wide dictionary to match a 4k-param
  supervised probe, and for the goal feature its codes are much worse
  (0.61 vs 0.79).
- A two-layer KAN on a small learned projection is the weakest supervised
  probe: the projection bottleneck discards information.

### Open questions / next

- Probe the remaining 13 shapes and more seeds; try spline probes on wider
  projections and on attention-head outputs rather than the full residual.
- Hold out regions of goal cells during training to test true generalisation
  of the path models beyond unseen pairs.
- Test whether SAE features trained on the path models are interpretable as
  positions/directions (feature-cell maps) and how many features a hole costs.

### Environment notes

- `/home` is ~100% full; project venv at `/var/tmp/venvs/geomechinterp`
  (`.venv` symlink), generated path data and checkpoints under `/var/tmp`.
- User-level uv config points at a private index; use
  `uv --no-config ... --index-url https://pypi.org/simple`.
