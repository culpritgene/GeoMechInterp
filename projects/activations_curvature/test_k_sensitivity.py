"""Test how k affects hyperboloid curvature."""
import numpy as np
import sys
sys.path.append('/Users/solar/Code/DL/Mechanistic_Interpretability/GeoMechInterp/projects/activations_curvature')
from curvature_measure import generate_hyperboloid
from curvature_measure_final import fit_local_surface_and_compute_curvature

np.random.seed(42)
n_points = 2500
hyperboloid = generate_hyperboloid(n_points, a=1.0)

z_vals = hyperboloid[:, 2]
K_expected = np.mean(-1 / z_vals**2)

print("Testing k-sensitivity for Hyperboloid Gaussian curvature")
print("=" * 60)
print(f"Expected K: {K_expected:.4f}")
print()

for k in [30, 60, 100, 150, 200, 300]:
    K_hyp = fit_local_surface_and_compute_curvature(hyperboloid, k=k)
    mean_K = np.mean(K_hyp)
    std_K = np.std(K_hyp)
    error = abs(mean_K - K_expected) / abs(K_expected) * 100
    sign = '✓' if mean_K < 0 else '✗'
    
    print(f"k={k:3d}:  K = {mean_K:7.4f} ± {std_K:.4f}  Error: {error:5.1f}%  Sign: {sign}")

