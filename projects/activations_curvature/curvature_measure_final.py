"""
Final robust Gaussian curvature estimation for point clouds.

Based on:
- Cazals & Pouget (2005) "Estimating Differential Quantities Using Polynomial Fitting"
- Uses jet fitting to estimate the osculating paraboloid
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def fit_local_surface_and_compute_curvature(points, k=50):
    """
    Fit local quadratic surface and compute Gaussian curvature.
    
    Algorithm:
    1. For each point, find k nearest neighbors
    2. Fit local coordinate system via PCA
    3. Fit quadratic surface h(u,v) in local coordinates
    4. Extract second fundamental form coefficients
    5. Compute Gaussian curvature K = det(II) / det(I)
    
    For a height function h(u,v) over a tangent plane:
    - First fundamental form:  I ≈ identity (for small patches)
    - Second fundamental form: II = Hessian of h
    - Gaussian curvature: K ≈ det(Hessian(h)) = h_uu * h_vv - h_uv²
    """
    n_points = len(points)
    curvatures = np.zeros(n_points)
    
    # Build k-NN
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n_points))
    nbrs.fit(points)
    
    for i in range(n_points):
        try:
            # Get neighborhood
            distances, neighbors = nbrs.kneighbors([points[i]])
            neighbor_points = points[neighbors[0, 1:]]  # Exclude self
            center = points[i]
            
            if len(neighbor_points) < 6:  # Need at least 6 points for quadratic fit
                continue
            
            # Center the neighborhood
            local_points = neighbor_points - center
            
            # Get local frame via PCA
            pca = PCA(n_components=min(3, local_points.shape[1]))
            pca.fit(local_points)
            
            # Local coordinate system
            # For a 2D surface in 3D: first 2 components span tangent plane
            if local_points.shape[1] == 3:
                tangent_u = pca.components_[0]
                tangent_v = pca.components_[1]
                
                # Use PCA's third component as normal
                # This is the direction of minimum variance
                normal = pca.components_[2]
                
            else:
                # Higher dimensional case
                tangent_u = pca.components_[0]
                tangent_v = pca.components_[1] if len(pca.components_) > 1 else np.zeros_like(tangent_u)
                # For high-d, normal is harder to define
                # We'll work with the projection onto the first 2 components
                normal = None
            
            # Project to 2D local coordinates
            u = local_points @ tangent_u
            v = local_points @ tangent_v
            
            # Heights
            if normal is not None:
                h = local_points @ normal
            else:
                # Use distance from the 2D plane spanned by first 2 PCA components
                projected = np.outer(u, tangent_u) + np.outer(v, tangent_v)
                h = np.linalg.norm(local_points - projected, axis=1)
            
            # Fit quadratic surface: h = a*u² + b*u*v + c*v² + d*u + e*v + f
            # Build design matrix
            A = np.column_stack([
                u**2,
                u * v,
                v**2,
                u,
                v,
                np.ones_like(u)
            ])
            
            # Solve least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(A, h, rcond=None)
            a, b, c, d, e, f = coeffs
            
            # Compute Gaussian curvature via principal curvatures
            # Build the Hessian matrix (second fundamental form)
            hessian = np.array([[2*a, b],
                                [b, 2*c]])
            
            # Eigenvalues of Hessian are the principal curvatures k1, k2
            # Gaussian curvature K = k1 * k2 = det(Hessian)
            eigenvalues = np.linalg.eigvalsh(hessian)
            k1, k2 = eigenvalues[0], eigenvalues[1]
            
            # Gaussian curvature
            K = k1 * k2
            
            # Apply correction for slope at center
            h_u = d
            h_v = e
            denominator = (1 + h_u**2 + h_v**2)**2
            K = K / (denominator + 1e-10)
            
            curvatures[i] = K
            
        except (np.linalg.LinAlgError, ValueError, IndexError):
            curvatures[i] = 0.0
    
    return curvatures


# Test
if __name__ == "__main__":
    import sys
    sys.path.append('/Users/solar/Code/DL/Mechanistic_Interpretability/GeoMechInterp/projects/activations_curvature')
    from curvature_measure import generate_sphere, generate_hyperboloid, generate_torus
    
    print("=" * 70)
    print("FINAL GAUSSIAN CURVATURE TEST")
    print("=" * 70)
    
    np.random.seed(42)
    n_points = 2500
    
    # Sphere
    print("\n[1/3] SPHERE")
    sphere = generate_sphere(n_points, radius=1.0, dim=2)
    K_sphere = fit_local_surface_and_compute_curvature(sphere, k=60)
    print(f"  K = {np.mean(K_sphere):.4f} ± {np.std(K_sphere):.4f}")
    print(f"  Expected: 1.0000")
    print(f"  Error: {abs(np.mean(K_sphere) - 1.0) / 1.0 * 100:.1f}%")
    
    # Hyperboloid
    print("\n[2/3] HYPERBOLOID")
    hyperboloid = generate_hyperboloid(n_points, a=1.0)
    K_hyp = fit_local_surface_and_compute_curvature(hyperboloid, k=60)
    
    z_vals = hyperboloid[:, 2]
    K_expected = np.mean(-1 / z_vals**2)
    
    print(f"  K = {np.mean(K_hyp):.4f} ± {np.std(K_hyp):.4f}")
    print(f"  Expected: {K_expected:.4f} (K = -1/z²)")
    print(f"  Error: {abs(np.mean(K_hyp) - K_expected) / abs(K_expected) * 100:.1f}%")
    print(f"  Sign: {'CORRECT ✓' if np.mean(K_hyp) < 0 else 'WRONG ✗'}")
    
    # Torus
    print("\n[3/3] TORUS (major_radius=2, minor_radius=0.5)")
    torus = generate_torus(n_points, major_radius=2.0, minor_radius=0.5)
    K_torus = fit_local_surface_and_compute_curvature(torus, k=60)
    
    # Analytical: K = cos(θ) / (r*(R + r*cos(θ)))
    # Mean K ≈ 0 (positive on outside, negative on inside)
    
    print(f"  K = {np.mean(K_torus):.4f} ± {np.std(K_torus):.4f}")
    print(f"  Expected: ~0 (mixed sign)")
    print(f"  % Positive: {100 * np.mean(K_torus > 0):.1f}%")
    print(f"  % Negative: {100 * np.mean(K_torus < 0):.1f}%")
    
    print("\n" + "=" * 70)

