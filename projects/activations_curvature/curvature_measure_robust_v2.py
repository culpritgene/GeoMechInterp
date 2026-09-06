"""
Improved Gaussian curvature estimation using proper surface normal.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_surface_normal(center, neighbors):
    """
    Compute surface normal using local geometry.
    
    Uses PCA to get approximate tangent plane, then verifies with
    cross products.
    """
    from sklearn.decomposition import PCA
    
    # Center the neighborhood
    local_points = neighbors - center
    
    # Use PCA to get initial estimate
    pca = PCA(n_components=3)
    pca.fit(local_points)
    
    # The normal should be the direction of minimum variance
    normal_estimate = pca.components_[2]
    
    # Verify: take two vectors in the approximate tangent plane
    # and compute their cross product
    v1 = pca.components_[0]
    v2 = pca.components_[1]
    normal_cross = np.cross(v1, v2)
    normal_cross = normal_cross / (np.linalg.norm(normal_cross) + 1e-10)
    
    # Make sure they point in the same direction
    if np.dot(normal_estimate, normal_cross) < 0:
        normal_cross = -normal_cross
    
    return normal_cross, v1, v2


def compute_gaussian_curvature_via_shape_operator(points, k=50):
    """
    Compute Gaussian curvature using the shape operator (Weingarten map).
    
    This computes principal curvatures k1 and k2, then K = k1 * k2.
    More robust than Monge patch fitting.
    """
    n_points = len(points)
    curvatures = np.zeros(n_points)
    
    # Build k-NN
    nbrs = NearestNeighbors(n_neighbors=k+1)
    nbrs.fit(points)
    distances, neighbor_indices = nbrs.kneighbors(points)
    
    for i in range(n_points):
        center = points[i]
        neighbors = points[neighbor_indices[i, 1:]]  # Exclude self
        
        # Get surface normal and tangent basis
        normal, t1, t2 = compute_surface_normal(center, neighbors)
        
        # Build tangent basis matrix
        tangent_basis = np.column_stack([t1, t2])  # (3, 2)
        
        # Center neighbors
        local_centered = neighbors - center
        
        # Project to tangent plane
        local_2d = local_centered @ tangent_basis  # (k, 2)
        
        # Compute heights (signed distance along normal)
        heights = local_centered @ normal  # (k,)
        
        # Fit quadratic form in tangent coordinates
        # h(u,v) = au² + buv + cv²
        # (we omit linear and constant terms for principal curvatures)
        try:
            # Use only quadratic terms for cleaner principal curvature extraction
            X_quad = np.column_stack([
                local_2d[:, 0]**2,  # u²
                local_2d[:, 0] * local_2d[:, 1],  # uv
                local_2d[:, 1]**2,  # v²
            ])
            
            # Fit
            coeffs = np.linalg.lstsq(X_quad, heights, rcond=None)[0]
            a, b, c = coeffs
            
            # The second fundamental form matrix (shape operator) is:
            # S = [[a,  b/2],
            #      [b/2, c]]
            #
            # Eigenvalues are the principal curvatures k1, k2
            # Gaussian curvature K = k1 * k2 = det(S) = ac - (b/2)²
            
            K = a * c - (b / 2)**2
            curvatures[i] = K
            
        except (np.linalg.LinAlgError, ValueError):
            curvatures[i] = 0.0
    
    return curvatures


# Test
if __name__ == "__main__":
    from curvature_measure import generate_sphere, generate_hyperboloid
    
    print("=" * 60)
    print("Testing Shape Operator Method")
    print("=" * 60)
    
    np.random.seed(42)
    n_points = 2500
    
    # Test sphere
    print("\n[1/2] SPHERE (expected K=1.0)")
    sphere = generate_sphere(n_points, radius=1.0, dim=2)
    K_sphere = compute_gaussian_curvature_via_shape_operator(sphere, k=50)
    print(f"  Mean: {np.mean(K_sphere):.4f} ± {np.std(K_sphere):.4f}")
    print(f"  Error: {abs(np.mean(K_sphere) - 1.0) / 1.0 * 100:.1f}%")
    
    # Test hyperboloid
    print("\n[2/2] HYPERBOLOID (expected K ≈ -0.60)")
    hyperboloid = generate_hyperboloid(n_points, a=1.0)
    K_hyp = compute_gaussian_curvature_via_shape_operator(hyperboloid, k=50)
    
    # Compute analytical mean
    z_vals = hyperboloid[:, 2]
    K_expected = np.mean(-1 / z_vals**2)
    
    print(f"  Mean: {np.mean(K_hyp):.4f} ± {np.std(K_hyp):.4f}")
    print(f"  Expected: {K_expected:.4f}")
    print(f"  Error: {abs(np.mean(K_hyp) - K_expected) / abs(K_expected) * 100:.1f}%")
    print(f"  Sign: {'CORRECT ✓' if np.mean(K_hyp) < 0 else 'WRONG ✗'}")
    
    print("\n" + "=" * 60)

