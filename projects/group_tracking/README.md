# Group state tracking on Cayley walks (language ladder, rung 2)

Small GPTs (`projects/path_transformer/model.py`) are trained to output the
group element reached by a walk `e0 g1 .. gk` on a Cayley graph, and their
residual streams are probed for the exact state in irreducible-representation
coordinates.  Groups: cyclic `Z<n>`, torus `T<n>x<m>`, dihedral `D<n>`
(order 2n, elements `r^a s^b`, non-abelian).  The design follows
`projects/PROPOSALS_language_ladder.md`, rung 2.

## Files

| file | what |
|---|---|
| `groups.py` | explicit multiplication tables, exact irrep matrices and coordinates, generator sets; `verify()` checks closure, associativity, inverses and `rho(g) rho(h) = rho(gh)` on the whole table |
| `group_data.py` | sequences `BOS e0 SEP g1..gk SEP ek EOS` (k uniform in 1..kmax), splits, held-out generator bigrams, D_n non-commutativity minimal pairs, prefix products + irrep coordinates; npz writer and `GroupData` loader; `loss_mask` (loss on the final element only) |
| `train_group.py` | trainer (bf16 autocast, cosine schedule, plateau stopping on validation exact accuracy); evaluates fresh words, held-out bigrams, minimal pairs, per k |
| `probe_group.py` | residuals at generator positions and the final SEP; targets: irrep coordinates (all / per frequency), element (linear softmax), eps (logistic + grid), the lowest harmonic absent from the model's embedding FFT; ridge / top-k SAE + ridge / ReLU hinges / cubic splines at matched sizes (imported from `path_transformer/probe.py`) |
| `null_control.py` | probe-capacity ceiling: the exact latent embedded orthogonally into R^d with 40 nuisance dims and noise, identical grid, prints spline-minus-hinge R^2 per budget |
| `tests/test_groups.py` | plain-function tests (`python tests/test_groups.py`), no pytest needed |

Conventions: specials BOS=0 EOS=1 SEP=2 PAD=3; element tokens `4 + g`;
generator tokens `4 + |G| + j`; walks multiply on the right
(`e_i = e_{i-1} . g_i`).  Data: `/var/tmp/geomech_data/groups/<group>_<seed>.npz`;
checkpoints: `/var/tmp/geomech_ckpt/group_<group>_L<layers>_d<dim>/`
(`best.pt`, `last.pt`, `summary.json`, `probes.json`).

Generators: `Z_n` {+-1, +-2, +-3} (uniform); `T_nxm` {(+-1,0), (0,+-1)}
(`--diagonals` adds (+-1,+-1)); `D_n` {r^+-1, r^+-2, r^+-3, s, sr} with the two
reflections sharing probability `--p_reflect` (default 0.2).  Irrep
coordinates: `Z_n` (cos, sin)(2 pi f a / n), f = 1..F (default F=3); torus one
circle per factor; `D_n` the entries (c, s, eps c, eps s) of
`rho_f(r^a s^b) = R(2 pi f a / n) diag(1, (-1)^b)` plus `eps = (-1)^b`.

Splits: `train` / `validation` / `test` (fresh random words; 5% of ordered
generator pairs, chosen by seed, never occur in them), `heldout_bigram` (every
word contains one of those pairs), `minimal_pair` (D_n: word pairs differing by
one adjacent swap of non-commuting generators, same start element, different
answers; empty for abelian groups).

## Commands

```bash
cd /home/culpritgene/GeoMechInterp
# tests
.venv/bin/python projects/group_tracking/tests/test_groups.py
# verify a group and print its generators
.venv/bin/python projects/group_tracking/groups.py --group D36
# data (200k training words, 10k per eval split, kmax 12)
.venv/bin/python projects/group_tracking/group_data.py --group D36 --seed 0 --n_train 200000
.venv/bin/python projects/group_tracking/group_data.py --group Z36 --seed 0 --n_train 200000
# training (plateau stopping: --min_steps, --patience; --time_limit for smoke runs)
.venv/bin/python projects/group_tracking/train_group.py --group Z36 --n_layer 2 --d_model 64 --n_head 4 \
    --steps 12000 --min_steps 2000 --patience 5 --eval_every 500 --time_limit 110
.venv/bin/python projects/group_tracking/train_group.py --group D36 --n_layer 4 --d_model 64 --steps 40000 --min_steps 20000
# probes (full grid) and a smoke grid
.venv/bin/python projects/group_tracking/probe_group.py --run group_D36_L4_d64 --layers 0 1 2 3 4
.venv/bin/python projects/group_tracking/probe_group.py --run group_Z36_L2_d64 --tiny --n_seq 3000
# null control (probe-capacity ceiling)
.venv/bin/python projects/group_tracking/null_control.py --group D36 --d 64 --freqs 1
.venv/bin/python projects/group_tracking/null_control.py --group Z36 --d 64 --freqs 1 --tiny --n 20000 --n_train 12000
```

## Smoke result (Z36, 2 layers, d=64, 12k steps of batch 256, 63 s)

Exact final-element accuracy: validation 0.998, test 0.999, held-out bigrams
0.999; per k 1.000 for k <= 9, 0.993 at k = 12.  The embedding FFT puts 90%
of its power on frequencies {1..5}, so the "unused harmonic" probe target is
f = 6.  With the tiny probe grid (300 steps, so under-trained), at the
generator positions of layer 1 the first harmonic is linear (ridge R^2 0.90),
the second and third are not (0.26 / 0.05); the element is linearly decodable
only at the final position (softmax accuracy 0.997 at layer 2 vs 0.18 at the
generator positions), i.e. this small model finishes the computation at the
answer position.  Numbers from the full grid and trained-to-plateau models
are what the rung's claims need; these are only an end-to-end check.

## Notes

- The final SEP is the answer position: with tied embeddings the element is
  linear there by construction (the proposal says not to use answer positions
  for family comparisons); it is collected because it is the only position
  that must hold the full product, and reported as its own cell.
- Layer 0 at a generator position is the generator's token embedding, so all
  state targets are at chance there (a sanity check on the pipeline).
- `--min_step` drops the first t generator positions from the `gen` cell
  (t = 1 is the one-step lookup `e0 . g1`).


## Results (2026-09-07)

Task bar (long schedule: 120k steps, batch 512, lr 5e-4, plateau stop):

| group | d | fresh words | held-out bigrams | minimal pairs |
|---|---|---|---|---|
| D36 | 32 | 0.969 | 0.967 | 0.958 |
| D36 | 64 | 0.999 | 0.9995 | 0.998 |
| D36 | 256 | 0.997 | | |
| T36x12 | 32 / 64 / 256 | 0.9997 / 0.9991 / 0.9995 | | |
| Z36 | 32 / 64 / 256 | 0.998 / 0.999 / 0.999 | | |
| Z360 | 32 | 0.999 | | |

Single-token probes (`probe_group.py`, `results/group_probes.md`, figure
`results/group_probes.png`): in every D36 model the sign character of the
prefix product is linearly undecodable at layer 2 (ridge ~0, logistic at
chance) yet cubic-spline probes read it at 0.96-0.99 under 1k parameters
where ReLU hinges get 0.65-0.97 and top-k SAE codes + ridge 0.16-0.9; the
model linearises it only at layer 4. Z36 at d=32 stores the element as a
harmonic code (4th harmonic dominant) and splines read the fundamental at
0.98 under 1k parameters vs 0.74 for hinges. The torus is stored linearly
and shows no gap.

Null controls (`null_control.py`, a linearly embedded circle plus 40
nuisance dimensions): the linear feature is read equally by all families;
its 2nd/3rd harmonics and the quadratic sign form are read by splines
(0.83-0.93 under 1k) but not by hinges (0.09-0.34) nor by SAE codes
(0.07-0.26). The spline advantage is a property of the feature class:
polynomial functions of linearly stored circles.

Cross-token probes (`probe_pair.py`, `results/pair_surface_Z36_d32.png`):
reading the composed element from the previous state and the new generator.
Z36 d=32: two cubic splines (176 params) reach 0.995; hinges need 16-32
units; the exact bilinear form caps at 0.72; SAE codes + ridge 0.65-0.79 at
18k-72k params. D36 d=64: the composed sign is cubic in the residuals, so
ridge and bilinear score 0.00, hinges 0.07/0.34 at 500/1k params (chance on
reflection steps), cubic splines 0.97/0.99, tensor-spline surfaces 0.95/0.99,
SAE codes 0.09-0.16.
