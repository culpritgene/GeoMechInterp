"""
Robust Curvature Estimation with Multi-Scale Smoothing

This module provides improved curvature estimates that:
1. Use multi-scale analysis (multiple k values)
2. Add explicit smoothing
3. Better handle continuous manifolds
4. Distinguish intrinsic vs extrinsic curvature
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_multi_scale_curvature(compute_func, points, k_range=(20, 40, 60), **kwargs):
    """
    Compute curvature at multiple scales and average.
    
    This reduces brittleness by combining local and global information.
    
    Parameters:
    compute_func: Curvature computation function
    points: Point cloud
    k_range: Tuple of k values to try
    kwargs: Additional arguments for compute_func
    
    Returns:
    curvatures: Smoothed multi-scale estimate
    details: Dict with per-scale results
    """
    all_curvatures = []
    
    for k in k_range:
        result = compute_func(points, k=k, **kwargs)
        
        # Handle different return types
        if isinstance(result, tuple):
            curv = result[0]
        else:
            curv = result
            
        if isinstance(curv, dict):
            curv = curv.get('mean', list(curv.values())[0])
            
        all_curvatures.append(curv)
    
    # Average across scales (robust to any single scale)
    curvatures = np.mean(all_curvatures, axis=0)
    
    details = {
        'per_scale': all_curvatures,
        'std_across_scales': np.std(all_curvatures, axis=0),
        'k_values': k_range
    }
    
    return curvatures, details


def smooth_curvature_field(curvatures, neighbors, n_iterations=2):
    """
    Smooth curvature estimates by diffusion on the manifold.
    
    Parameters:
    curvatures: Raw curvature estimates
    neighbors: Neighbor indices from k-NN
    n_iterations: Number of smoothing iterations
    
    Returns:
    smoothed: Smoothed curvatures
    """
    smoothed = curvatures.copy()
    
    for _ in range(n_iterations):
        new_smoothed = np.zeros_like(smoothed)
        
        for i in range(len(curvatures)):
            # Average with neighbors (heat diffusion)
            neighbor_vals = smoothed[neighbors[i]]
            new_smoothed[i] = 0.5 * smoothed[i] + 0.5 * np.mean(neighbor_vals)
        
        smoothed = new_smoothed
    
    return smoothed


def compute_gaussian_curvature_from_fit(points, k=40):
    """
    Estimate Gaussian curvature by fitting a local quadratic surface.
    
    For a 2D surface, Gaussian curvature = k1 * k2 (principal curvatures)
    and Ricci curvature = Gaussian curvature.
    
    Parameters:
    points: Point cloud (n_points, 3) for 2D surface in 3D
    k: Number of neighbors
    
    Returns:
    curvatures: Gaussian curvature estimates
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    
    n_points = len(points)
    curvatures = np.zeros(n_points)
    
    # Build k-NN
    nbrs = NearestNeighbors(n_neighbors=k+1)
    nbrs.fit(points)
    distances, neighbors = nbrs.kneighbors(points)
    neighbors = neighbors[:, 1:]
    
    for i in range(n_points):
        # Get local neighborhood
        local_points = points[neighbors[i]]
        center = points[i]
        
        # Center the points
        local_centered = local_points - center
        
        # Find local tangent plane via PCA
        pca = PCA(n_components=3)
        pca.fit(local_centered)
        
        # Get tangent basis (first 2 components)
        tangent_basis = pca.components_[:2].T  # (3, 2)
        normal = pca.components_[2]  # Normal to surface
        
        # Project to 2D tangent coordinates
        local_2d = local_centered @ tangent_basis  # (k, 2)
        
        # Heights above tangent plane
        heights = local_centered @ normal  # (k,)
        
        # Fit quadratic surface: z = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f
        # This gives us the Monge patch representation
        X = np.column_stack([
            local_2d[:, 0]**2,  # x^2
            local_2d[:, 0] * local_2d[:, 1],  # xy
            local_2d[:, 1]**2,  # y^2
            local_2d[:, 0],  # x
            local_2d[:, 1],  # y
            np.ones(len(local_2d))  # constant
        ])
        
        # Solve least squares
        try:
            coeffs = np.linalg.lstsq(X, heights, rcond=None)[0]
            a, b, c, d, e, f = coeffs
            
            # For Monge patch z = f(x, y), Gaussian curvature is:
            # K = (f_xx * f_yy - f_xy^2) / (1 + f_x^2 + f_y^2)^2
            # 
            # For our quadratic: f(x,y) = ax² + bxy + cy² + dx + ey + f
            # f_x = 2ax + by + d
            # f_y = bx + 2cy + e
            # f_xx = 2a, f_xy = b, f_yy = 2c
            #
            # At center (x=0, y=0): f_x = d, f_y = e
            
            # Compute at center point
            f_xx = 2 * a
            f_yy = 2 * c
            f_xy = b
            f_x_center = d
            f_y_center = e
            
            # Full Gaussian curvature formula
            numerator = f_xx * f_yy - f_xy**2
            denominator = (1 + f_x_center**2 + f_y_center**2)**2
            K = numerator / (denominator + 1e-10)
            
            curvatures[i] = K
            
        except np.linalg.LinAlgError:
            curvatures[i] = 0.0
    
    return curvatures


def compute_robust_pca_curvature(points, k_range=(30, 50, 70), intrinsic_dim=2, 
                                 smooth_iterations=2):
    """
    Robust PCA tangent space curvature with multi-scale and smoothing.
    
    Parameters:
    points: Point cloud
    k_range: Multiple k values for multi-scale analysis
    intrinsic_dim: Manifold dimension
    smooth_iterations: Diffusion smoothing steps
    
    Returns:
    curvatures: Robust curvature estimates
    details: Additional information
    """
    from curvature_measure import compute_tangent_space_curvature, build_knn_graph
    
    # Multi-scale computation
    curvatures, multi_details = compute_multi_scale_curvature(
        compute_tangent_space_curvature,
        points,
        k_range=k_range,
        intrinsic_dim=intrinsic_dim
    )
    
    # Spatial smoothing
    neighbors = build_knn_graph(points, k=k_range[1])  # Use middle k
    curvatures_smooth = smooth_curvature_field(curvatures, neighbors, smooth_iterations)
    
    details = {
        **multi_details,
        'before_smoothing': curvatures.copy(),
        'smoothing_iterations': smooth_iterations
    }
    
    return curvatures_smooth, details


# ============================================================================
# VALIDATION: Test on known manifolds
# ============================================================================

def validate_robust_methods():
    """Test robust methods on sphere and hyperboloid."""
    from curvature_measure import generate_sphere, generate_hyperboloid
    
    print("="*70)
    print("VALIDATION: Robust Multi-Scale Methods")
    print("="*70)
    
    n_points = 2500
    
    # Test on sphere
    print("\n[1/3] Testing on Sphere (K=1.0)...")
    sphere = generate_sphere(n_points, radius=1.0, dim=2)
    
    curv_robust, details = compute_robust_pca_curvature(
        sphere, 
        k_range=(30, 50, 70),
        intrinsic_dim=2,
        smooth_iterations=3
    )
    
    print(f"   Multi-scale PCA: {np.mean(curv_robust):.4f} ± {np.std(curv_robust):.4f}")
    print(f"   Error: {abs(np.mean(curv_robust) - 1.0) / 1.0 * 100:.1f}%")
    print(f"   Variance reduction: {np.std(details['before_smoothing']):.4f} → {np.std(curv_robust):.4f}")
    
    # Test Gaussian curvature (= Ricci for 2D surfaces)
    print("\n[2/3] Testing Gaussian curvature on Sphere...")
    curv_gaussian = compute_gaussian_curvature_from_fit(sphere, k=50)
    print(f"   Gaussian curvature: {np.mean(curv_gaussian):.4f} ± {np.std(curv_gaussian):.4f}")
    print(f"   Expected: 1.0 (for unit sphere)")
    print(f"   Error: {abs(np.mean(curv_gaussian) - 1.0) / 1.0 * 100:.1f}%")
    
    # Test on hyperboloid
    print("\n[3/3] Testing on Hyperboloid (K=-1.0)...")
    hyperboloid = generate_hyperboloid(n_points, a=1.0)
    
    curv_gaussian_hyp = compute_gaussian_curvature_from_fit(hyperboloid, k=50)
    print(f"   Gaussian curvature: {np.mean(curv_gaussian_hyp):.4f} ± {np.std(curv_gaussian_hyp):.4f}")
    print(f"   Expected: -1.0")
    print(f"   Error: {abs(np.mean(curv_gaussian_hyp) - (-1.0)) / 1.0 * 100:.1f}%")
    print(f"   Sign: {'CORRECT ✓' if np.mean(curv_gaussian_hyp) < 0 else 'WRONG ✗'}")
    
    print("\n" + "="*70)
    print("✓ Validation complete")
    print("="*70)


if __name__ == "__main__":
    validate_robust_methods()

