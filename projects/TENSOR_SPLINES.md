# Tensor-spline dictionaries: second-order interpretable codes

Separate note for the tensor-spline line of work (2026-09-08). Companion to
`LABJOURNAL.md`; code in `projects/group_tracking/dictionary_compare.py`
(synthetic and group models) and `projects/llm_dictionaries/` (GPT-2 on text).

## Why

Everything before this note says the same thing from several angles: the
features where cubic splines beat first-order readouts by a wide margin are
*polynomial functions of linearly stored coordinates*: harmonics of a stored
circle, products across two tokens, the dihedral sign as a quadratic form.
A top-k SAE plus linear readout cannot express such a feature in one unit;
it either tiles the latent into many arc features (which works only when the
code is dense enough and the latent is small) or misses it. A univariate
spline code (`SpAE`) does not fix that on its own, because a reconstruction
objective on a linearly embedded latent is solved by near-linear response
curves. The missing piece is a code that can represent a *product* of two
directions directly: a tensor-product cubic B-spline surface on a pair of
learned projections,

    h_j(x) = sum_{ab} c_{jab} B_a(u_j) B_b(v_j),   u_j = a_j . x + alpha_j,  v_j = b_j . x + beta_j,

with top-k sparsity on |h| and a linear decoder (`TensorSplineAE`, "TsAE").
This is the dictionary counterpart of the cross-token probe that read angle
addition from two tokens with one surface (`probe_pair.py`).

Encoder parameters per code: SAE d+1; SpAE d+1+(G+3); TsAE 2d+2+(G+3)^2.
Comparisons are run at matched *total* encoder parameters, so the TsAE gets
roughly half as many codes as the SAE.

## Objectives

    recon      reconstruct the residual x
    next_layer predict the residual at a later layer from x
    delta      predict what the layers in between write (resid[out] - x)

Read-out test on the synthetic and group data: ridge regression from the
codes to exact targets (irrep coordinates, harmonics, sign). On GPT-2: EV of
the objective, L0, dead fraction, and loss recovered when the dictionary's
output is spliced back into the model (recon: replace the residual, baseline
mean ablation; delta: replace the next residual by x + predicted write,
baseline skip the write).

Dead-feature control: codes that have not fired in the last 1000 steps are
re-initialised (encoder rows pointed at random inputs, spline coefficients
fresh), identically for all families, until 75% of training.

## Results

### Synthetic and group models (`dictionary_compare.py`, `results/runs/dictX_*.json`)

Best held-out ridge R² from the codes over the size grid (SAE (m,k) in
{(256,8),(256,32),(1024,32)}; SpAE same; TsAE (m,k) in {(64,8),(128,16),
(256,32),(512,32)}), 3000 steps, dead-feature resampling; "supervised" is an
8-unit cubic-spline probe (~600 params) for reference.

| case | target | linear | SAE | SpAE | TsAE | supervised |
|---|---|---|---|---|---|---|
| D36 null (linear circle + nuisance), recon | sign | 0.00 | 0.93 | 0.39 | 0.07 | 0.999 |
| D36 null, recon | 2nd harmonic | 0.00 | 0.84 | 0.12 | 0.02 | 0.98 |
| D36 d=64 layer 2, recon | sign | 0.00 | 0.38 | 0.29 | 0.15 | 0.90 |
| D36 d=64 layer 2, delta | sign | 0.00 | 0.44 | 0.31 | 0.30 | 0.89 |
| D36 d=256 layer 2, delta | sign | -0.36 | 0.87 | 0.79 | 0.80 | 0.97 |
| Z36 d=32 layer 2, recon | 2nd harmonic | 0.11 | 0.49 | 0.53 | 0.51 | 0.97 |
| Z36 d=32 layer 2, delta | 2nd harmonic | 0.11 | 0.58 | 0.82 | 0.81 | 0.98 |
| Z36 d=32 layer 2, delta | 3rd harmonic | 0.01 | 0.31 | 0.58 | 0.45 | 0.93 |

Reading:

- Second-order codes never beat univariate spline codes in an unsupervised
  dictionary, and on the product feature (the D36 sign) they are worse than
  or equal to the SAE. Under reconstruction a tensor-spline dictionary fitted
  to a linearly embedded latent stays effectively linear (null: 0.07 / 0.02),
  exactly like the univariate one; only the dense hinge SAE tiles the latent.
- The one place spline dictionaries lead (Z36 harmonics under the
  write-prediction objective, +0.24 / +0.28 over the SAE) is a case where the
  feature is a univariate function of a stored projection, so a surface adds
  nothing over a curve.
- Every dictionary remains far below a supervised spline readout. The
  unsupervised objective, not the code family, decides whether a polynomial
  feature is exposed; the tensor-spline surface that read the composed sign
  across two tokens when *supervised* (`probe_pair.py`: 0.99 at 1k params)
  does not find it when trained to predict the layer's write.
