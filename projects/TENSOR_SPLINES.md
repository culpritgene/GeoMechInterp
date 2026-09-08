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

(filled in as runs complete; see the tables below)
