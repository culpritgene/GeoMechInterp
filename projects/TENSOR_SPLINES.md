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

### GPT-2 small on text (`projects/llm_dictionaries/llm_dictionaries.py`, `results/gpt2_L6.json`)

Residual stream after block 6, 500k training tokens and 60k held-out tokens
of WikiText-103, top-k = 32 for every dictionary, 6000 steps, resampling.
Clean CE 4.141; mean-ablation baseline 9.252 (recon); skip-block-7 baseline
4.259 (delta).

| objective | encoder budget | family | codes | EV | dead | spliced CE | loss recovered |
|---|---|---|---|---|---|---|---|
| recon | 0.6M | SAE | 768 | 0.711 | 0% | 4.458 | 0.938 |
| recon | 0.6M | SpAE | 750 | 0.653 | 0.1% | 4.615 | 0.907 |
| recon | 0.6M | TsAE | 384 | 0.630 | 0% | 4.649 | 0.901 |
| recon | 2.4M | SAE | 3072 | 0.795 | 0% | 4.337 | 0.962 |
| recon | 2.4M | SpAE | 3000 | 0.710 | 18% | 4.561 | 0.918 |
| recon | 2.4M | TsAE | 1536 | 0.733 | 0.1% | 4.451 | 0.939 |
| delta | 0.6M | SAE | 768 | 0.396 | 0% | 4.176 | 0.701 |
| delta | 0.6M | SpAE | 750 | 0.353 | 0% | 4.176 | 0.705 |
| delta | 0.6M | TsAE | 384 | 0.319 | 0% | 4.178 | 0.684 |
| delta | 2.4M | SAE | 3072 | 0.510 | 0% | 4.165 | 0.799 |
| delta | 2.4M | SpAE | 3000 | 0.438 | 24% | 4.170 | 0.753 |
| delta | 2.4M | TsAE | 1536 | 0.427 | 0% | 4.171 | 0.749 |

Reading (layer 6): on real text the hinge SAE is the best dictionary at
matched encoder parameters on both objectives (EV and loss recovered). The
tensor-spline dictionary is the better of the two spline families at the
larger budget (recon 0.94 vs 0.92 loss recovered, with almost no dead codes
where the univariate one loses a quarter of its codes), and roughly ties it
under the write objective. Whatever polynomial structure GPT-2's residual
carries, it is not what these unsupervised objectives reward, so a code
family that can express products earns nothing here either.

### GPT-2 layer 3 (partial; the VM was shut down before the run finished)

Clean CE 4.141; mean-ablation baseline 10.540.

| objective | budget | family | codes | EV | dead | loss recovered |
|---|---|---|---|---|---|---|
| recon | 0.6M | SAE | 768 | 0.744 | 0% | 0.956 |
| recon | 0.6M | SpAE | 750 | 0.684 | 0% | 0.880 |
| recon | 0.6M | TsAE | 384 | 0.660 | 0% | 0.890 |

## Summary

1. Second-order (tensor-spline) codes give an unsupervised dictionary no
   advantage over univariate spline codes on the synthetic and group models,
   and on the one genuine product feature (the D36 sign) they are no better
   than the hinge SAE under either objective. Under reconstruction any spline
   dictionary fitted to a linearly embedded latent stays effectively linear.
2. On GPT-2 small (layer 6, matched encoder parameters, top-k 32) the hinge
   SAE remains the best dictionary on both reconstruction (96% loss
   recovered at 2.4M) and write prediction (80%); the tensor-spline
   dictionary is the better spline family (94% / 75%) and keeps its codes
   alive where the univariate one loses 18-24%.
3. The spline advantage established elsewhere in this repo is a property of
   supervised readouts of polynomial features and of the cross-token
   composition law (two splines vs tens of hinges; the tensor-spline surface
   reads angle addition and the composed dihedral sign where hinges, bilinear
   forms and SAE codes fail). Unsupervised dictionaries are limited by their
   objective before their code family matters; a task-aligned objective
   (write prediction) lets univariate spline codes lead on harmonic features
   of small models, but nothing tried here made second-order codes earn their
   parameters unsupervised.
4. Open: an objective that rewards product features directly (e.g. predicting
   the model's logits or attention-head outputs rather than the residual
   write), and supervised tensor-spline probes on real LLM features with
   known composition structure (e.g. positional or arithmetic circuits).
