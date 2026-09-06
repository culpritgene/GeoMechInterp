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

### Follow-up: why do splines not help in activation space? (same day)

Hypotheses considered: (a) oct-tree tokenisation, (b) over-parametrisation,
(c) local linearity of single steps on a densely sampled surface.

- Stratified probes (`probe_strata.py`): tokens split by |Gaussian
  curvature| terciles of the current cell and by distance to the nearest
  other torus (< 0.35 = near link contact). Target = unit step direction.
  Every probe decodes the step direction *better* in high-|K| and
  near-contact regions than on flat parts (e.g. keyring, layer 4, ReLU m=32:
  0.54 highK vs 0.32 lowK): on flat regions many directions are near-optimal,
  so the target itself is ambiguous. Splines never beat hinges in any stratum.
- Position near link contacts is genuinely non-linear: the linear probe's
  within-stratum R² is negative there and even the best probe gets 0.25-0.45
  (vs 0.85-0.9 elsewhere). Two surfaces close in 3-D must be told apart by
  a non-linear code. Splines are not better than hinges there either.
- Probe optimisation confound confirmed: the tanh squash in the spline probe
  hurt. With standardised inputs, no tanh and a B-spline grid over [-3, 3],
  the m=32 spline probe rose from 0.79 to 0.83 (ReLU m=32: 0.80-0.81). The
  wide (m=128) tanh version was badly under-optimised (0.71 vs ReLU 0.85).
  Adopted the no-tanh probe for the v3 probes.
- Over-parametrisation (b): 3-layer d=64 models (0.24M params, 20x smaller)
  reach 98.4-99.7% success, so the 5M models were far larger than needed.
  In the small single_ring model the ring angle stays linear through layer 1
  (R² 0.94) and goes non-linear in layers 2-3 (0.55-0.60), and the residual
  uses 44 of 64 dims for 90% variance. Compression pushes the geometric code
  toward non-linear form, which is where higher-order probes could matter;
  v3 probes run on both model sizes.
- Tokenisation (a) looks like a minor factor: the flat embedding already
  learns a linear circle, so the discrete cells did not block a smooth code.

### v3 probes (fixed spline probe; large and small models)

- Spline-probe fix verified at width: m=128 without tanh, grid [-3, 3] gives
  0.849-0.854 vs 0.846 for ReLU m=128 (tanh versions 0.71-0.78 whatever the
  lr, steps or grid). The earlier "splines lose when wide" was the probe.
- Layer 2, mean over six models, position: splines >= hinges at every budget
  (0.69 vs 0.60 at <=1k; 0.934 vs 0.915 at <=10k; 0.950 vs 0.941 at <=40k);
  remaining distance: tie; goal: tie; step direction: hinges win when wide
  (0.64 vs 0.59). Same pattern at layer 4.
- Reading: smooth geometric features favour cubic splines modestly;
  decision-like features (which neighbour to step to) favour hinges. The SAE
  needs ~10x the parameters of either supervised probe.
- Small models (0.24M, 3 layers, d=64) solve the task as well as the 5M ones;
  their code is more non-linear (linear R^2 for position 0.50 vs 0.64) but the
  spline-vs-hinge margin does not widen, so compression alone does not
  create the kind of curved feature where splines dominate.
- Net conclusion for experiment 2: in this task the residual stream's
  geometric features are smooth-but-distributed, and the parameter gap
  between first-order and cubic readouts is small (<= 0.1 R^2). The large
  input-space gap of experiment 1 needs a feature that is genuinely curved
  in a low-dimensional subspace; that is what the next experiments must
  engineer (cyclic categorical features, belief over a manifold, ratio
  features from in-context parameter learning).

### Proposals toward language (`projects/PROPOSALS_language_ladder.md`)

Produced by a propose / merge / adversarial-critique / synthesise pass (20 raw
proposals from five lenses, 13 merged, two critics each; the critics ran code
against the repo). Seven rungs, ordered by distance from the current work:
winding-class geodesics (circle x counter); group state tracking on Cayley
walks (D_36, flat torus, Z_360 under compression); stack of modular-arithmetic
circles with bindings; rank-constrained hidden-tree taxonomy (hyperbolic vs
simplicial codes, exact absorption ground truth); scoped belief grammar
(nested Mess3 scopes with agreement-dependent closers); clock-world relational
deduction (binding names to angles across hops); in-context Bayesian chains
with a latent hyperparameter. Shared protocol: force the curved feature by
construction, verify presence and non-linearity in a pre-registered cell,
subtract a synthetic-linear null control, then compare families.

Two researcher ideas were transformed by the critique rather than adopted
as-is: (i) cyclic categorical causal patterns: any function of a k-valued
categorical is linear in its one-hot when every (value, shift) combination is
seen, so a Fourier code needs held-out combinations and small d (folded into
rungs 2 and 3); (ii) Dirichlet in-context mixtures: the posterior mean is
produced linearly by an averaging head whose softmax absorbs the denominator,
so ratios are expected linear; the non-linear quantity is the posterior over
the concentration hyperparameter (rung 7). The manifold random-walk belief
idea was dropped because the ring belief concentration is nearly constant
(0.996) under the local walk; a hidden-start localisation variant is noted.

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
