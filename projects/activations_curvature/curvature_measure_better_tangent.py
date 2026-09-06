"""
Better tangent plane estimation for curvature.

Key insight: PCA on all k neighbors can give a tilted plane.
Instead, use only the CLOSEST neighbors for the tangent plane.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def fit_curvature_with_better_tangent(points, k_tangent=10, k_curvature=80):
    """
    Fit curvature using a two-stage approach:
    1. Estimate tangent plane using k_tangent closest neighbors
    2. Fit quadratic surface using k_curvature neighbors
    
    This ensures the tangent plane is actually tangent (not tilted).
    """
    n_points = len(points)
    curvatures = np.zeros(n_points)
    
    # Build k-NN for both scales
    nbrs = NearestNeighbors(n_neighbors=k_curvature+1)
    nbrs.fit(points)
    
    for i in range(n_points):
        try:
            # Get neighborhoods
            distances, neighbors = nbrs.kneighbors([points[i]])
            center = points[i]
            
            # Stage 1: Get tangent plane from CLOSEST neighbors only
            closest_neighbors = points[neighbors[0, 1:k_tangent+1]]
            local_closest = closest_neighbors - center
            
            pca_tangent = PCA(n_components=min(3, local_closest.shape[1]))
            pca_tangent.fit(local_closest)
            
            tangent_u = pca_tangent.components_[0]
            tangent_v = pca_tangent.components_[1]
            normal = pca_tangent.components_[2]
            
            # Stage 2: Fit quadratic using LARGER neighborhood
            all_neighbors = points[neighbors[0, 1:k_curvature+1]]
            local_points = all_neighbors - center
            
            # Project to tangent coordinates
            u = local_points @ tangent_u
            v = local_points @ tangent_v
            h = local_points @ normal
            
            # Fit quadratic
            A = np.column_stack([
                u**2, u*v, v**2, u, v, np.ones_like(u)
            ])
            
            coeffs = np.linalg.lstsq(A, h, rcond=None)[0]
            a, b, c, d, e, f = coeffs
            
            # Gaussian curvature
            h_uu = 2 * a
            h_vv = 2 * c
            h_uv = b
            
            numerator = h_uu * h_vv - h_uv**2
            denominator = (1 + d**2 + e**2)**2
            K = numerator / (denominator + 1e-10)
            
            curvatures[i] = K
            
        except (np.linalg.LinAlgError, ValueError, IndexError):
            curvatures[i] = 0.0
    
    return curvatures


# Test
if __name__ == "__main__":
    import sys
    sys.path.append('/Users/solar/Code/DL/Mechanistic_Interpretability/GeoMechInterp/projects/activations_curvature')
    from curvature_measure import generate_sphere, generate_hyperboloid
    
    print("=" * 70)
    print("TWO-STAGE TANGENT PLANE METHOD")
    print("=" * 70)
    
    np.random.seed(42)
    n_points = 2500
    
    # Sphere
    print("\n[1/2] SPHERE")
    sphere = generate_sphere(n_points, radius=1.0, dim=2)
    K_sphere = fit_curvature_with_better_tangent(sphere, k_tangent=10, k_curvature=80)
    print(f"  K = {np.mean(K_sphere):.4f} ± {np.std(K_sphere):.4f}")
    print(f"  Expected: 1.0000")
    print(f"  Error: {abs(np.mean(K_sphere) - 1.0) / 1.0 * 100:.1f}%")
    
    # Hyperboloid
    print("\n[2/2] HYPERBOLOID")
    hyperboloid = generate_hyperboloid(n_points, a=1.0)
    K_hyp = fit_curvature_with_better_tangent(hyperboloid, k_tangent=10, k_curvature=80)
    
    z_vals = hyperboloid[:, 2]
    K_expected = np.mean(-1 / z_vals**2)
    
    print(f"  K = {np.mean(K_hyp):.4f} ± {np.std(K_hyp):.4f}")
    print(f"  Expected: {K_expected:.4f}")
    print(f"  Error: {abs(np.mean(K_hyp) - K_expected) / abs(K_expected) * 100:.1f}%")
    print(f"  Sign: {'CORRECT ✓' if np.mean(K_hyp) < 0 else 'WRONG ✗'}")
    
    print("\n" + "=" * 70)

