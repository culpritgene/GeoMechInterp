"""
Curvature via numerical differentiation of normal vectors.

Based on the shape operator (Weingarten map): dN/dx gives curvature.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def estimate_normal_at_point(center, neighbors):
    """Estimate surface normal using local PCA."""
    local = neighbors - center
    pca = PCA(n_components=min(3, local.shape[1]))
    pca.fit(local)
    normal = pca.components_[-1]  # Direction of minimum variance
    return normal


def compute_curvature_via_shape_operator(points, k=50):
    """
    Estimate Gaussian curvature by measuring how the normal changes.
    
    The shape operator S maps tangent vectors to their normal derivatives.
    Gaussian curvature K = det(S).
    """
    n_points = len(points)
    curvatures = np.zeros(n_points)
    
    nbrs = NearestNeighbors(n_neighbors=k+1)
    nbrs.fit(points)
    
    for i in range(n_points):
        try:
            center = points[i]
            distances, neighbors_idx = nbrs.kneighbors([center])
            neighbors_idx = neighbors_idx[0, 1:]
            neighbor_points = points[neighbors_idx]
            
            # Get normal at center
            n_center = estimate_normal_at_point(center, neighbor_points)
            
            # Get tangent basis at center
            local = neighbor_points - center
            pca = PCA(n_components=3)
            pca.fit(local)
            t1 = pca.components_[0]
            t2 = pca.components_[1]
            
            # For a few neighbors, estimate how normal changes
            # Shape operator approximation: S_ij ≈ -⟨n_j - n_i, t⟩ / ||p_j - p_i||
            shape_matrix = np.zeros((2, 2))
            count = 0
            
            for j_idx in neighbors_idx[:min(20, len(neighbors_idx))]:  # Use subset
                neighbor_j = points[j_idx]
                
                # Get neighbors of neighbor_j for its normal
                _, sub_neighbors_idx = nbrs.kneighbors([neighbor_j])
                sub_neighbors = points[sub_neighbors_idx[0, 1:]]
                n_j = estimate_normal_at_point(neighbor_j, sub_neighbors)
                
                # Make sure normals point in same direction
                if np.dot(n_center, n_j) < 0:
                    n_j = -n_j
                
                # Vector from center to neighbor
                dp = neighbor_j - center
                dist = np.linalg.norm(dp)
                
                if dist < 1e-10:
                    continue
                
                # Change in normal
                dn = n_j - n_center
                
                # Project onto tangent directions
                dt1 = np.dot(dp, t1)
                dt2 = np.dot(dp, t2)
                dn_t1 = -np.dot(dn, t1)  # Negative for shape operator convention
                dn_t2 = -np.dot(dn, t2)
                
                # Accumulate
                if abs(dt1) > 1e-6:
                    shape_matrix[0, 0] += dn_t1 / dt1
                    count += 1
                if abs(dt2) > 1e-6:
                    shape_matrix[1, 1] += dn_t2 / dt2
                    count += 1
            
            if count > 0:
                shape_matrix /= (count / 2)  # Average
                
                # Gaussian curvature = det(shape operator)
                K = shape_matrix[0, 0] * shape_matrix[1, 1] - shape_matrix[0, 1] * shape_matrix[1, 0]
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
    print("SHAPE OPERATOR METHOD (Normal Derivatives)")
    print("=" * 70)
    
    np.random.seed(42)
    n_points = 2500
    
    # Sphere
    print("\n[1/2] SPHERE")
    sphere = generate_sphere(n_points, radius=1.0, dim=2)
    K_sphere = compute_curvature_via_shape_operator(sphere, k=50)
    print(f"  K = {np.mean(K_sphere):.4f} ± {np.std(K_sphere):.4f}")
    print(f"  Expected: 1.0000")
    
    # Hyperboloid
    print("\n[2/2] HYPERBOLOID")
    hyperboloid = generate_hyperboloid(n_points, a=1.0)
    K_hyp = compute_curvature_via_shape_operator(hyperboloid, k=50)
    
    z_vals = hyperboloid[:, 2]
    K_expected = np.mean(-1 / z_vals**2)
    
    print(f"  K = {np.mean(K_hyp):.4f} ± {np.std(K_hyp):.4f}")
    print(f"  Expected: {K_expected:.4f}")
    print(f"  Sign: {'CORRECT ✓' if np.mean(K_hyp) < 0 else 'WRONG ✗'}")
    
    print("\n" + "=" * 70)

