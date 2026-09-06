"""Quick test for Gaussian curvature method."""
import numpy as np
import sys
sys.path.append('/Users/solar/Code/DL/Mechanistic_Interpretability/GeoMechInterp/projects/activations_curvature')

# Import the generate functions (which don't need sklearn)
from curvature_measure import generate_sphere, generate_hyperboloid

# Copy the Gaussian curvature function here to avoid import issues
def compute_gaussian_curvature_from_fit(points, k=40):
    """Estimate Gaussian curvature by fitting a local quadratic surface."""
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
            # At center (x=0, y=0): f_x = d, f_y = e
            f_xx = 2 * a
            f_yy = 2 * c
            f_xy = b
            numerator = f_xx * f_yy - f_xy**2
            denominator = (1 + d**2 + e**2)**2
            K = numerator / (denominator + 1e-10)
            
            curvatures[i] = K
            
        except np.linalg.LinAlgError:
            curvatures[i] = 0.0
    
    return curvatures


# Test on sphere
print("Testing Gaussian curvature on SPHERE...")
print("-" * 50)
np.random.seed(42)
sphere = generate_sphere(n_points=2500, radius=1.0, dim=2)
curv_sphere = compute_gaussian_curvature_from_fit(sphere, k=50)

print(f"Mean curvature: {np.mean(curv_sphere):.4f}")
print(f"Std dev:        {np.std(curv_sphere):.4f}")
print(f"Expected:       1.0000 (for unit sphere)")
print(f"Error:          {abs(np.mean(curv_sphere) - 1.0) / 1.0 * 100:.1f}%")
print()

# Test on hyperboloid
print("Testing Gaussian curvature on HYPERBOLOID...")
print("-" * 50)
hyperboloid = generate_hyperboloid(n_points=2500, a=1.0)

# Compute analytical expectation: K = -1/z^2
z_values = hyperboloid[:, 2]
K_expected_mean = np.mean(-1 / z_values**2)

curv_hyp = compute_gaussian_curvature_from_fit(hyperboloid, k=50)

print(f"Mean curvature: {np.mean(curv_hyp):.4f}")
print(f"Std dev:        {np.std(curv_hyp):.4f}")
print(f"Expected:       {K_expected_mean:.4f} (analytical: K=-1/z²)")
print(f"Error:          {abs(np.mean(curv_hyp) - K_expected_mean) / abs(K_expected_mean) * 100:.1f}%")
print(f"Sign:           {'CORRECT ✓' if np.mean(curv_hyp) < 0 else 'WRONG ✗'}")
print()
print("=" * 50)

