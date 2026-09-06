# Manifold features: SAE vs spline networks

**Question.** A sparse autoencoder (SAE) feature is a ReLU of a linear
projection, i.e. a first-order spline knot. How many such features does it
take to model the curved boundary of a surface with holes, compared with a
network whose edge functions are cubic B-splines (KAN-style)? If SAEs are
much less parameter-efficient on curved geometry, that bounds what a
linear-feature dictionary can say about non-linear features in a transformer.

**Experiment 1 (this folder).** Point-cloud classification on 16 Blender
torus-product shapes (`data/meshes`): given a 3-D point, predict
exterior / surface / interior. The surface is a thin shell, so the task is
exactly "model the curvature of the surface".

## Data

- `data/mesh_datasets/combined_pointcloud_dataset_v3` (HF Arrow, 921,600
  points, 16 shapes x 57,600, 80/10/10 split). Fetch with
  `gcloud storage rsync --recursive gs://pm-user-data-us-west1/users/culpritgene/tokenized_manifolds/mesh_datasets/ data/mesh_datasets/`
  and the same for `meshes/`.
- Labels are recomputed analytically in `data.py` from the union-of-tori
  signed distance. Torus parameters are fitted from the OBJ vertices because
  the JSON centres are wrong for the `chain_link_*` shapes (the shipped
  `in_tunnel` labels for those shapes are therefore also wrong; they agree
  with the true interior only 83-90% of the time). Surface points lying inside
  another torus (self-intersections) and off-surface points within 0.005 of
  the surface are dropped.
- Per-shape caches: `data/mesh_datasets/cache_v3/<shape>.npz`.

## Models (`models.py`)

| family | architecture | params |
|---|---|---|
| `sae` | 3 -> m ReLU -> 3, L1 = 1e-3 on the code | 7m + 3 |
| `sae_weak` | same, L1 = 1e-4 | 7m + 3 |
| `relu1` | same, no penalty (dense control) | 7m + 3 |
| `mlp2` | 3 -> h -> h -> 3 ReLU | |
| `spline1` | KAN 3 -> h -> 3, degree-1 B-splines, G intervals | 3h(G+1) + 3h(G+1) + h + 3 |
| `spline3` | KAN 3 -> h -> 3, degree-3 B-splines, G intervals | 3h(G+3) + 3h(G+3) + h + 3 |

All models take normalised coordinates in [-1,1]^3 and are trained
full-batch with Adam + cosine decay for up to 15,000 steps, stopping once
validation accuracy has not improved for 3,000 steps (never before 5,000);
the best-validation checkpoint is scored on the test split.

A first pass with a fixed 3,000-step budget (archived in `results/steps3000/`)
under-trained every family: with 12,000 steps a 1024-unit ReLU net went from
0.965 to 0.988 on `single_ring`, and a 539-parameter cubic-spline net from
0.976 to 0.999. The ranking was unchanged but the saturation levels were not,
hence the plateau-based schedule.

## Running

```bash
.venv/bin/python projects/manifold_features/data.py                 # build caches
.venv/bin/python projects/manifold_features/train_sweep.py --shapes all --seeds 0   # ~2 h on an H100 with 8 parallel chains
.venv/bin/python projects/manifold_features/plot_results.py         # figures + summary.md
```

## Results (16 shapes, seed 0, plateau schedule)

Figures: `results/acc_vs_params.png`, `results/near_acc_vs_params.png`;
tables: `results/summary.md`, `results/params_to_threshold.csv`.

Mean over the 16 shapes of the best test accuracy reachable at or below a
parameter budget:

| family | 250 | 500 | 1000 | 2000 | 4000 | 16000 |
|---|---|---|---|---|---|---|
| SAE, L1=1e-3 | 0.793 | 0.905 | 0.943 | 0.953 | 0.958 | 0.960 |
| SAE, L1=1e-4 | 0.816 | 0.905 | 0.940 | 0.954 | 0.962 | 0.971 |
| dense ReLU, 1 hidden | 0.819 | 0.878 | 0.917 | 0.941 | 0.953 | 0.967 |
| ReLU MLP, 2 hidden | 0.826 | 0.915 | 0.933 | 0.952 | 0.962 | 0.969 |
| linear-spline net | 0.894 | 0.965 | 0.980 | 0.986 | 0.986 | 0.986 |
| cubic-spline net | 0.942 | 0.976 | 0.988 | 0.993 | 0.994 | 0.994 |

Same, restricted to off-surface points within 0.05 of the surface (where the
curvature has to be modelled):

| family | 250 | 500 | 1000 | 2000 | 4000 | 16000 |
|---|---|---|---|---|---|---|
| SAE, L1=1e-3 | 0.380 | 0.486 | 0.588 | 0.639 | 0.667 | 0.677 |
| SAE, L1=1e-4 | 0.386 | 0.482 | 0.577 | 0.625 | 0.667 | 0.724 |
| dense ReLU, 1 hidden | 0.382 | 0.445 | 0.501 | 0.578 | 0.616 | 0.699 |
| ReLU MLP, 2 hidden | 0.495 | 0.548 | 0.590 | 0.661 | 0.711 | 0.746 |
| linear-spline net | 0.489 | 0.708 | 0.817 | 0.864 | 0.869 | 0.869 |
| cubic-spline net | 0.597 | 0.768 | 0.874 | 0.923 | 0.930 | 0.930 |

Minimum parameters to reach a test accuracy (geometric mean over the shapes
that reach it; n = number of shapes out of 16):

| family | 95% | 98% | 99% | 99.5% |
|---|---|---|---|---|
| SAE, L1=1e-3 | 1200 (n=12) | 1798 (n=2) | never | never |
| SAE, L1=1e-4 | 1627 (n=14) | 1133 (n=3) | 14339 (n=1) | never |
| dense ReLU, 1 hidden | 3094 (n=14) | 2261 (n=3) | never | never |
| ReLU MLP, 2 hidden | 1288 (n=15) | 1767 (n=4) | 3523 (n=2) | never |
| linear-spline net | 417 (n=16) | 754 (n=13) | 926 (n=5) | never |
| cubic-spline net | 247 (n=16) | 457 (n=16) | 732 (n=14) | 656 (n=6) |

Takeaways:

- The cubic-spline net reaches 98% on every shape with ~460 parameters; no
  ReLU-based family (SAE at either sparsity, dense ReLU, or the 2-hidden MLP)
  reaches 98% on more than 4 of 16 shapes even at 15-30k parameters. The
  parameter gap at matched accuracy is one to two orders of magnitude.
- The gap is concentrated near the surface: at 1000 parameters the cubic
  spline classifies 87% of near-surface points correctly versus 50-59% for
  the ReLU families.
- Spline order matters on its own: linear splines (k=1) sit between ReLU and
  cubic, so the advantage is not only the per-edge non-linearity but also its
  smoothness order.
- ReLU families keep improving slowly with width and are still gaining at
  the largest sizes; they do not fail, they are inefficient. The sparse L1
  penalty costs little extra at small sizes and caps the SAE at large sizes
  (its active fraction falls to 4% at 14k parameters).
- Negative-curvature (inner) surface points are harder than positive-curvature
  ones for the ReLU families (0.90-0.95 vs 0.96-0.98 at <=1000 params);
  the spline nets are near-uniform (0.99+).
