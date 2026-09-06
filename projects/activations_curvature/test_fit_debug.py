"""Debug a single point's curvature calculation."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from curvature_measure import generate_hyperboloid

# Generate hyperboloid
np.random.seed(42)
hyperboloid = generate_hyperboloid(n_points=2500, a=1.0)

# Pick a point near the center
i = 0
center = hyperboloid[i]
print(f"Center point: {center}")
print(f"Analytical K = -1/z² = {-1/center[2]**2:.4f}")
print()

# Get neighbors
k = 50
nbrs = NearestNeighbors(n_neighbors=k+1)
nbrs.fit(hyperboloid)
distances, neighbors = nbrs.kneighbors(hyperboloid[[i]])
neighbors = neighbors[0, 1:]

local_points = hyperboloid[neighbors]
local_centered = local_points - center

# PCA
pca = PCA(n_components=3)
pca.fit(local_centered)

print("PCA explained variance:", pca.explained_variance_)
print("PCA explained variance ratio:", pca.explained_variance_ratio_)
print()

# Components
tangent_basis = pca.components_[:2].T  # (3, 2)
normal = pca.components_[2]

print("Tangent basis (first 2 PCA components):")
print(tangent_basis)
print()
print("Normal (3rd PCA component):")
print(normal)
print()

# Project to 2D
local_2d = local_centered @ tangent_basis
heights = local_centered @ normal

print("Heights above tangent plane:")
print(f"  Mean: {np.mean(heights):.6f}")
print(f"  Std:  {np.std(heights):.6f}")
print(f"  Min:  {np.min(heights):.6f}")
print(f"  Max:  {np.max(heights):.6f}")
print()

# Fit quadratic
X = np.column_stack([
    local_2d[:, 0]**2,
    local_2d[:, 0] * local_2d[:, 1],
    local_2d[:, 1]**2,
    local_2d[:, 0],
    local_2d[:, 1],
    np.ones(len(local_2d))
])

coeffs = np.linalg.lstsq(X, heights, rcond=None)[0]
a, b, c, d, e, f = coeffs

print("Fitted coefficients:")
print(f"  a (x²):  {a:.6f}")
print(f"  b (xy):  {b:.6f}")
print(f"  c (y²):  {c:.6f}")
print(f"  d (x):   {d:.6f}")
print(f"  e (y):   {e:.6f}")
print(f"  f:       {f:.6f}")
print()

# Compute curvature
f_xx = 2 * a
f_yy = 2 * c
f_xy = b

numerator = f_xx * f_yy - f_xy**2
denominator = (1 + d**2 + e**2)**2

K = numerator / denominator

print("Curvature calculation:")
print(f"  f_xx = 2a = {f_xx:.6f}")
print(f"  f_yy = 2c = {f_yy:.6f}")
print(f"  f_xy = b  = {f_xy:.6f}")
print(f"  numerator = {numerator:.6f}")
print(f"  denominator = {denominator:.6f}")
print(f"  K = {K:.6f}")
print()
print(f"Expected K = {-1/center[2]**2:.6f}")
print(f"Sign: {'CORRECT' if K < 0 else 'WRONG'}")

