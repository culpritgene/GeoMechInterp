"""Debug: Look at the fitted coefficients for a few points."""
import numpy as np
import sys
sys.path.append('/Users/solar/Code/DL/Mechanistic_Interpretability/GeoMechInterp/projects/activations_curvature')
from curvature_measure import generate_hyperboloid, generate_sphere
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def analyze_single_point(points, idx, k=60):
    """Analyze the surface fit at a single point."""
    center = points[idx]
    
    # Get neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1)
    nbrs.fit(points)
    distances, neighbors = nbrs.kneighbors([center])
    neighbor_points = points[neighbors[0, 1:]]
    
    # Center
    local_points = neighbor_points - center
    
    # PCA
    pca = PCA(n_components=3)
    pca.fit(local_points)
    
    tangent_u = pca.components_[0]
    tangent_v = pca.components_[1]
    normal = pca.components_[2]
    
    # Project
    u = local_points @ tangent_u
    v = local_points @ tangent_v
    h = local_points @ normal
    
    # Fit
    A = np.column_stack([u**2, u*v, v**2, u, v, np.ones_like(u)])
    coeffs = np.linalg.lstsq(A, h, rcond=None)[0]
    a, b, c, d, e, f = coeffs
    
    # Curvature
    h_uu = 2 * a
    h_vv = 2 * c
    h_uv = b
    K = (h_uu * h_vv - h_uv**2) / (1 + d**2 + e**2)**2
    
    return {
        'center': center,
        'coeffs': (a, b, c, d, e, f),
        'h_range': (h.min(), h.max()),
        'curvature': K,
        'pca_variance': pca.explained_variance_ratio_
    }

# Test on hyperboloid
print("=" * 70)
print("HYPERBOLOID COEFFICIENT ANALYSIS")
print("=" * 70)

np.random.seed(42)
hyperboloid = generate_hyperboloid(2500, a=1.0)

# Analyze several points
for i in [0, 100, 500, 1000, 1500]:
    result = analyze_single_point(hyperboloid, i, k=100)
    center = result['center']
    a, b, c, d, e, f = result['coeffs']
    
    # Analytical curvature
    K_analytical = -1 / center[2]**2
    
    print(f"\nPoint {i}: center = ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"  Analytical K = {K_analytical:.4f}")
    print(f"  Computed K   = {result['curvature']:.4f}")
    print(f"  Coefficients: a={a:.6f}, b={b:.6f}, c={c:.6f}")
    print(f"  h_uu = {2*a:.6f}, h_vv = {2*c:.6f}, h_uv = {b:.6f}")
    print(f"  h_uu * h_vv - h_uv² = {4*a*c - b**2:.6f}")
    print(f"  Height range: [{result['h_range'][0]:.6f}, {result['h_range'][1]:.6f}]")
    print(f"  PCA variance: {result['pca_variance']}")
    
    # Check signs
    if a > 0 and c > 0:
        print(f"  ⚠️  Both a and c are POSITIVE → convex bowl")
    elif a < 0 and c < 0:
        print(f"  ⚠️  Both a and c are NEGATIVE → concave bowl")
    else:
        print(f"  ✓  Mixed signs (a and c) → saddle shape")

print("\n" + "=" * 70)

