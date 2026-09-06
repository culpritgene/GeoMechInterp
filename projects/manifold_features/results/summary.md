# Parameter-efficiency summary

Shapes: 16, seeds per config: 1-1

## Minimum parameters to reach a test accuracy (geometric mean over shapes that reach it; n = shapes reaching it)

| family | 95.0% | 98.0% | 99.0% | 99.5% | max acc (mean) |
|---|---|---|---|---|---|
| SAE (sparse ReLU, L1=1e-3) | 1200 (n=12) | 1798 (n=2) | not reached | not reached | 0.9598 |
| SAE (sparse ReLU, L1=1e-4) | 1627 (n=14) | 1133 (n=3) | 14339 (n=1) | not reached | 0.9711 |
| dense ReLU, 1 hidden | 3094 (n=14) | 2261 (n=3) | not reached | not reached | 0.9671 |
| ReLU MLP, 2 hidden | 1288 (n=15) | 1767 (n=4) | 3523 (n=2) | not reached | 0.9688 |
| linear-spline net (k=1) | 417 (n=16) | 754 (n=13) | 926 (n=5) | not reached | 0.9862 |
| cubic-spline net (k=3) | 247 (n=16) | 457 (n=16) | 732 (n=14) | 656 (n=6) | 0.9937 |

## Per shape: parameters to reach 99% test accuracy

| shape | sae | sae_weak | relu1 | mlp2 | spline1 | spline3 |
|---|---|---|---|---|---|---|
| single_ring | - | 14339 | - | 4611 | 499 | 233 |
| stacked_equal_sign_two | - | - | - | 2691 | 883 | 347 |
| double_touching_in_plane | - | - | - | - | - | 463 |
| double_intersecting_in_plane | - | - | - | - | 1763 | 539 |
| chain_link_two | - | - | - | - | - | 1075 |
| chain_link_three | - | - | - | - | - | 1075 |
| chain_link_four | - | - | - | - | - | 2147 |
| pretzel_three_in_plane | - | - | - | - | - | 1075 |
| plus_shape | - | - | - | - | 883 | 271 |
| sphere_skeleton | - | - | - | - | 995 | 923 |
| circle_border_1 | - | - | - | - | - | 691 |
| circle_border_2 | - | - | - | - | - | 691 |
| circle_border_3 | - | - | - | - | - | - |
| keyring_1 | - | - | - | - | - | 1075 |
| keyring_2 | - | - | - | - | - | 1843 |
| keyring_3 | - | - | - | - | - | - |
