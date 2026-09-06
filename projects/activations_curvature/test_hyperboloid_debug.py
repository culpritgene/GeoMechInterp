"""Debug the hyperboloid generation."""
import numpy as np
from curvature_measure import generate_hyperboloid

# Generate hyperboloid
np.random.seed(42)
points = generate_hyperboloid(n_points=1000, a=1.0)

print("Hyperboloid Generation Check")
print("=" * 60)

# Check the constraint: z^2 - x^2 - y^2 should be constant (= a^2 = 1)
constraint = points[:, 2]**2 - points[:, 0]**2 - points[:, 1]**2

print("\n1. Constraint check (z² - x² - y² = a² = 1):")
print(f"   Mean:   {np.mean(constraint):.6f}")
print(f"   Std:    {np.std(constraint):.6f}")
print(f"   Min:    {np.min(constraint):.6f}")
print(f"   Max:    {np.max(constraint):.6f}")
print(f"   Expected: 1.0")

# Check z values
print(f"\n2. Z coordinate range:")
print(f"   Min z: {np.min(points[:, 2]):.3f}")
print(f"   Max z: {np.max(points[:, 2]):.3f}")

# Compute analytical Gaussian curvature
print("\n3. Analytical Gaussian curvature (K = -1/z²):")
z_values = points[:, 2]
K_analytical = -1 / (z_values**2)
print(f"   Mean: {np.mean(K_analytical):.4f}")
print(f"   Std:  {np.std(K_analytical):.4f}")
print(f"   Min:  {np.min(K_analytical):.4f}")
print(f"   Max:  {np.max(K_analytical):.4f}")

print("\n4. Sample points with analytical curvature:")
for i in range(min(5, len(points))):
    x, y, z = points[i]
    K = -1 / (z**2)
    print(f"   Point {i}: (x={x:.3f}, y={y:.3f}, z={z:.3f}) → K={K:.4f}")

print("\n" + "=" * 60)
print("Note: For a hyperboloid of one sheet (z² - x² - y² = a²),")
print("the Gaussian curvature K = -a²/z² = -1/z² when a=1")
print("This is NEGATIVE everywhere (hyperbolic surface)")
print("=" * 60)
