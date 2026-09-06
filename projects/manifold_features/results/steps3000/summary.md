# Parameter-efficiency summary

Shapes: 8, seeds per config: 1-2

## Minimum parameters to reach a test accuracy (geometric mean over shapes that reach it; n = shapes reaching it)

| family | 95.0% | 98.0% | 99.0% | 99.5% | max acc (mean) |
|---|---|---|---|---|---|
| SAE (sparse ReLU, L1=1e-3) | 1796 (n=2) | not reached | not reached | not reached | 0.9275 |
| dense ReLU, 1 hidden | 1795 (n=2) | not reached | not reached | not reached | 0.9443 |
| ReLU MLP, 2 hidden | 2562 (n=5) | 4611 (n=2) | not reached | not reached | 0.9579 |
| linear-spline net (k=1) | 461 (n=8) | 1264 (n=4) | not reached | not reached | 0.9791 |
| cubic-spline net (k=3) | 470 (n=8) | 846 (n=2) | not reached | not reached | 0.9707 |

## Per shape: parameters to reach 99% test accuracy

| shape | sae | relu1 | mlp2 | spline1 | spline3 |
|---|---|---|---|---|---|
| single_ring | - | - | - | - | - |
| double_touching_in_plane | - | - | - | - | - |
| chain_link_two | - | - | - | - | - |
| plus_shape | - | - | - | - | - |
| sphere_skeleton | - | - | - | - | - |
| circle_border_1 | - | - | - | - | - |
| keyring_1 | - | - | - | - | - |
| keyring_2 | - | - | - | - | - |
