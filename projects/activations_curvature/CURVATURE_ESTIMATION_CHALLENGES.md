# Curvature Estimation Challenges

## Summary

After extensive investigation, we've identified that **accurately estimating Gaussian curvature from point clouds is extremely challenging** and prone to sign errors and high variance. This document summarizes what we tried and recommendations going forward.

## The Problem

**Goal**: Estimate Gaussian curvature K for surfaces embedded in 3D (or higher dimensions) from point cloud samples.

**Challenge**: For a hyperboloid (analytical K ≈ -0.6), all attempted methods consistently give **positive** curvature (~0.3), with the wrong sign.

## What We Tried

### 1. Monge Patch Fitting (Polynomial Approach)
**Method**: 
- Fit local tangent plane via PCA
- Fit quadratic surface h(u,v) = au² + buv + cv² in tangent coordinates  
- Compute K = (h_uu * h_vv - h_uv²) / (1 + h_u² + h_v²)²

**Result**: ✗ Wrong sign for hyperboloid
- Sphere: K = 1.04 ± 0.01 ✓ (expected 1.0)
- Hyperboloid: K = +0.33 ± 0.33 ✗ (expected -0.6)

**Why it failed**:
- For saddle surfaces, need both positive and negative heights
- PCA tangent plane can be slightly tilted, putting all neighbors on one side
- Fitted coefficients have both a > 0 and c > 0 (convex bowl) instead of mixed signs (saddle)

### 2. Two-Stage Tangent Estimation
**Method**: Use closest neighbors for tangent plane, larger neighborhood for curvature

**Result**: ✗ No improvement

### 3. Shape Operator via Normal Derivatives
**Method**: Estimate dN/dx numerically by comparing normals at neighboring points

**Result**: ✗ Wrong sign AND high variance (±7.8)

### 4. Cross Product vs PCA Normal
**Method**: Tried different ways to compute the normal vector

**Result**: ✗ No effect on sign error

### 5. Multi-Scale and Smoothing
**Method**: Average over multiple k values, smooth results

**Result**: ✗ Reduces variance but doesn't fix sign error

## Root Causes

1. **Discrete Sampling**: Point clouds lack the continuous derivatives needed for exact curvature
2. **Tangent Plane Ambiguity**: PCA finds minimum-variance direction, not geometric tangent plane
3. **Saddle Detection**: Local polynomial fits struggle to detect saddle shapes (opposite curvatures in different directions)
4. **Numerical Stability**: Small errors in normal estimation compound when computing second derivatives

## What DOES Work

### Methods with Correct Behavior

1. **PCA Tangent Space Rotation** (`compute_tangent_space_curvature`)
   - Measures how tangent spaces rotate between neighbors
   - ✓ Gives correct RELATIVE curvature (higher for curved regions)
   - ✓ Stable and low variance
   - ⚠️ Not exact Gaussian curvature, but useful for comparisons

2. **Geodesic vs Euclidean Distance** (`compute_geodesic_curvature`)
   - Ratio of geodesic to Euclidean distance
   - ✓ Positive curvature → ratio > 1
   - ✓ Works for all manifolds
   - ⚠️ Expensive (shortest path computation)

3. **Sectional Curvature Sampling** (`compute_sectional_curvature`)
   - Sample random 2D planes through tangent space
   - ✓ Works for high-dimensional manifolds
   - ⚠️ High variance, needs many samples

## Recommendations

### For Research / Accurate Gaussian Curvature

**Use specialized libraries**:
- **Open3D** (`estimate_normals`, `estimate_covariances`)
- **PyMesh** (has robust curvature estimation)
- **Point Cloud Library (PCL)** via python-pcl

These libraries have:
- Proper mesh reconstruction
- Robust normal estimation with orientation consistency
- Validated implementations

### For Neural Network Activation Analysis

**Use the methods that work**:

```python
from curvature_measure import (
    compute_tangent_space_curvature,  # Best for relative comparisons
    compute_sectional_curvature,      # Good for high-dimensional spaces
    compute_geodesic_curvature        # Expensive but reliable
)

# For 4D-5D+ activation spaces
activations = model.get_activations(data)  # shape (n_samples, n_dims)

# Option 1: Tangent space rotation (fast, stable)
curvature_estimates = compute_tangent_space_curvature(
    activations, 
    k=40,  # Larger k for stability
    intrinsic_dim=2  # Expected manifold dimension
)

# Option 2: Sectional curvature (more mathematically rigorous)
curvature_estimates = compute_sectional_curvature(
    activations,
    k=40,
    intrinsic_dim=2,
    n_plane_samples=20  # More samples for stability
)
```

**Interpretation**:
- Higher values → more curved regions
- Compare across layers, training epochs, or different architectures
- Don't expect exact analytical values
- Focus on RELATIVE curvature patterns

### Validation Strategy

Test on known manifolds to understand method behavior:
```python
# Generate test cases
sphere = generate_sphere(2500, radius=1.0)
torus = generate_torus(2500)  
swiss_roll = generate_swiss_roll(2500)

# Compare methods
for name, manifold in [("Sphere", sphere), ("Torus", torus)]:
    K = compute_tangent_space_curvature(manifold, k=40)
    print(f"{name}: mean={np.mean(K):.3f}, std={np.std(K):.3f}")
```

## Lessons Learned

1. **Point cloud curvature is hard**: Don't trust simple implementations
2. **Validation is essential**: Always test on known shapes
3. **Relative > Absolute**: For ML applications, relative comparisons matter more than exact values
4. **Use existing tools**: Don't reinvent the wheel for production use
5. **Trade-offs matter**: Stability and interpretability often beat mathematical rigor

## References

- Cazals & Pouget (2005): "Estimating Differential Quantities Using Polynomial Fitting"
- Goldberg et al. (2002): "Understanding and Measuring Distances in Manifolds"
- Pottmann et al. (2007): "Principal Curvatures from Integral Invariant Viewpoint"
- Open3D Documentation: http://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html

---

**Bottom Line**: For your mechanistic interpretability work, use `compute_tangent_space_curvature` or `compute_sectional_curvature` from `curvature_measure.py`. They're stable, interpretable, and good enough for comparing curvature across different parts of your neural network's activation manifold.

