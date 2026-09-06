# Visual Comparison Guide

## New Visualization Functions Added

I've added two powerful visualization functions for comparing curvature methods side-by-side:

### 1. `plot_manifold_curvature_comparison()`

**Purpose:** Show the SAME manifold colored by DIFFERENT curvature methods

**Usage:**
```python
from curvature_measure import (
    generate_sphere,
    compute_tangent_space_curvature,
    compute_geodesic_curvature,
    compute_ollivier_ricci_curvature,
    plot_manifold_curvature_comparison
)

# Generate manifold
sphere_points = generate_sphere(n_points=500, radius=1.0, dim=2)

# Compute curvature with different methods
curv_pca, _ = compute_tangent_space_curvature(sphere_points, k=15, intrinsic_dim=2)
curv_geo, _ = compute_geodesic_curvature(sphere_points, k=15)
_, curv_or = compute_ollivier_ricci_curvature(sphere_points, k=10)

# Compare visualizations
methods = {
    'PCA Tangent': curv_pca,
    'Geodesic': curv_geo,
    'Ollivier-Ricci': curv_or
}

plot_manifold_curvature_comparison(
    sphere_points,
    methods,
    manifold_name="Sphere (K=1.0)",
    cmap='RdYlBu_r'
)
```

**What you'll see:**
- Multiple 3D plots side-by-side
- Same manifold structure, different coloring per method
- Mean curvature displayed on each subplot
- Consistent colormap range for comparison

---

### 2. `plot_manifold_curvature_grid()`

**Purpose:** Show DIFFERENT manifolds colored by the SAME method

**Usage:**
```python
from curvature_measure import (
    generate_sphere,
    generate_torus,
    generate_hyperboloid,
    generate_swiss_roll,
    compute_tangent_space_curvature,
    plot_manifold_curvature_grid
)

# Generate multiple manifolds
manifolds = {
    'Sphere (K=1)': generate_sphere(500, radius=1.0),
    'Torus (K≈0)': generate_torus(500),
    'Hyperboloid (K=-1)': generate_hyperboloid(500),
    'Swiss Roll': generate_swiss_roll(500)
}

# Apply same method to all
plot_manifold_curvature_grid(
    manifolds,
    compute_tangent_space_curvature,
    method_name="PCA Tangent Space Curvature",
    cmap='viridis',
    k=15,
    intrinsic_dim=2
)
```

**What you'll see:**
- Grid of 3D plots (up to 3 columns)
- Each manifold analyzed with the same method
- Consistent curvature interpretation
- Easy comparison of positive vs negative vs zero curvature

---

## Demo Script Integration

The `demo_example.py` script now automatically generates these comparison plots:

### Plot 1: Sphere with 4 Different Methods
Shows how different algorithms perceive the same sphere

### Plot 2: Hyperboloid with 3 Different Methods  
Compares methods on negative curvature manifold

### Plot 3: 4 Manifolds with PCA Tangent Method
Shows PCA method's performance across different geometries

---

## Interpreting the Visualizations

### Color Coding
- **Red/Warm colors:** High positive curvature
- **Blue/Cool colors:** Low or negative curvature
- **Green/Middle:** Near-zero curvature

### What to Look For

**1. Consistency Across Methods:**
- If all methods show similar patterns → strong geometric signal
- If methods disagree → investigate why (noise, scale sensitivity, etc.)

**2. Spatial Patterns:**
- Uniform color → constant curvature (sphere, hyperboloid)
- Varying color → non-constant curvature (torus, swiss roll)
- Patches/regions → local geometric features

**3. Method Characteristics:**
- **PCA Tangent:** Sensitive to tangent space orientation
- **Geodesic:** Global connectivity-aware
- **Ollivier-Ricci:** Graph structure dependent
- **Sectional:** Statistical aggregate
- **Covariance:** Local structure changes

---

## Tips for Best Results

1. **Use appropriate k:**
   - Too small (k<10): Noisy estimates
   - Too large (k>50): Over-smoothed
   - Recommended: k=15-20 for most cases

2. **Choose the right colormap:**
   - `'RdYlBu_r'`: Diverging (negative to positive)
   - `'viridis'`: Sequential (zero to positive)
   - `'coolwarm'`: Diverging with strong contrast

3. **Normalize for comparison:**
   - Functions auto-normalize to 5th-95th percentile
   - Removes outliers for better visualization
   - Mean values shown in titles for quantitative reference

4. **Intrinsic dimension matters:**
   - Set based on manifold (2 for surfaces, higher for volumes)
   - Use `estimate_local_intrinsic_dimension()` if unknown
   - Wrong dimension → biased curvature estimates

---

## Example Output Interpretation

### Sphere Example:
```
PCA Tangent: Mean = 1.05  → Close to analytical K=1.0 ✓
Geodesic:    Mean = 0.23  → Relative measure (positive) ✓
Ollivier:    Mean = 0.45  → Positive (concentration) ✓
```
**Interpretation:** All methods agree on positive curvature!

### Hyperboloid Example:
```
PCA Tangent: Mean = -0.95 → Close to analytical K=-1.0 ✓
Geodesic:    Mean = -0.15 → Relative measure (negative) ✓
Covariance:  Mean = 0.82  → High variation (not direct K) ⚠️
```
**Interpretation:** Most methods detect negative curvature. Covariance method measures change rate, not curvature directly.

---

## Running the Complete Demo

```bash
python demo_example.py
```

This will:
1. Generate synthetic manifolds
2. Compute curvatures with all methods
3. Print statistical summaries
4. Display comparison visualizations (close each to continue)
5. Show grid views of different methods

**Expected runtime:** 2-5 minutes depending on system

---

## For Your Neural Network Activations

```python
# Load your activations
activations = model.get_activations(data)  # Shape: (n_samples, embedding_dim)

# Project to 3D for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
activations_3d = pca.fit_transform(activations)

# Compute with multiple methods
methods = {}
methods['PCA'] = compute_tangent_space_curvature(activations, k=20, intrinsic_dim=8)[0]
methods['Geodesic'] = compute_geodesic_curvature(activations, k=20)[0]
methods['Covariance'] = compute_covariance_curvature(activations, k=20)

# Compare!
plot_manifold_curvature_comparison(
    activations_3d,
    methods,
    manifold_name=f"Layer {layer_num} Activations"
)
```

This helps you:
- Identify high-curvature regions (complex representations)
- Compare curvature across layers
- Validate geometric interpretations
- Detect representational bottlenecks

---

## Questions & Troubleshooting

### Q: Plots look empty/scattered?
**A:** Points might be outside view. Try rotating the 3D plot or adjusting viewing angle.

### Q: Colors all look the same?
**A:** Curvature might be very uniform. Check printed mean values. Adjust colormap normalization if needed.

### Q: Methods give wildly different results?
**A:** Normal! They measure different aspects:
- PCA → extrinsic curvature
- Geodesic → intrinsic via distances
- Ollivier → graph-based discrete
Check if signs agree (positive vs negative).

### Q: Too slow for large datasets?
**A:** 
- Reduce k (try k=10)
- Subsample points
- Use faster methods (Covariance, PCA)
- Avoid Ollivier-Ricci for >1000 points

---

## Next Steps

1. ✅ Run `demo_example.py` to see examples
2. ✅ Try on your own synthetic manifolds
3. ✅ Apply to transformer activations
4. ✅ Compare curvature across layers
5. ✅ Correlate with model performance
6. ✅ Publish your findings! 📊

---

Happy visualizing! 🎨

