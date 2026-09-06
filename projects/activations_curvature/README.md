# Activation Curvature Analysis

Analyzing curvature in neural network activation spaces using methods from differential geometry, manifold learning, and discrete geometry.

## Overview

This project provides comprehensive tools for measuring curvature in high-dimensional point clouds, specifically designed for analyzing transformer activation spaces. It combines:

- **Classical differential geometry** methods for smooth manifolds
- **Discrete Ricci curvature** (Forman, Ollivier) for graph-like structures  
- **Manifold learning** techniques for dimensionality and geometry estimation

## Installation

```bash
pip install -r requirements.txt
```

Required packages: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `networkx`

---

## Quick Start

### Example 1: Analyze Neural Network Activations

```python
from curvature_measure import (
    compute_tangent_space_curvature,
    estimate_local_intrinsic_dimension,
    plot_curvature_on_manifold_3d
)
from sklearn.decomposition import PCA

# Get activations from your model (shape: n_samples × embedding_dim)
activations = model.get_activations(inputs)  # e.g., (1000, 768)

# Compute curvature
curvatures, details = compute_tangent_space_curvature(
    activations, 
    k=20,  # number of nearest neighbors
    intrinsic_dim=8  # estimated manifold dimension
)

# Estimate local dimension
dimensions = estimate_local_intrinsic_dimension(activations, k=20, method='mle')

# Visualize (project to 3D first)
pca = PCA(n_components=3)
activations_3d = pca.fit_transform(activations)
plot_curvature_on_manifold_3d(activations_3d, curvatures)

print(f"Mean curvature: {curvatures.mean():.4f}")
print(f"Mean dimension: {dimensions.mean():.2f}")
```

### Example 2: Test on Synthetic Manifolds

```python
from curvature_measure import (
    generate_sphere,
    analytical_sphere_curvature,
    validate_curvature_method
)

# Generate sphere with known curvature
points = generate_sphere(n_points=500, radius=1.0, dim=2)
analytical_K = analytical_sphere_curvature(1.0)  # = 1.0

# Validate method
results = validate_curvature_method(
    compute_tangent_space_curvature,
    manifold_type='sphere',
    n_points=500,
    k=15
)

print(f"Analytical: {results['analytical']:.4f}")
print(f"Estimated:  {results['estimated_mean']:.4f}")
print(f"Error:      {results['relative_error']*100:.2f}%")
```

### Example 3: Run Full Demo

```python
# Run comprehensive demo script
python demo_example.py
```

---

## Curvature Methods

### 1. **Local PCA Tangent Space Curvature** ⭐ Recommended

Measures how tangent spaces rotate as you move along the manifold.

```python
curvatures, details = compute_tangent_space_curvature(
    points, 
    k=15,                # nearest neighbors
    metric='euclidean',  # distance metric
    intrinsic_dim=2      # manifold dimension
)
```

**When to use:** Smooth manifolds, well-sampled data
**Output:** Curvature scalar per point
**Interpretation:** Higher values = more curved locally

---

### 2. **Geodesic vs Euclidean Distance Ratio**

Compares manifold distances (via shortest paths) to straight-line distances.

```python
curvatures, geodesic_dist = compute_geodesic_curvature(
    points, 
    k=15,
    metric='euclidean'
)
```

**When to use:** Intuitive interpretation needed, robust to noise
**Output:** Ratio - 1.0 per point (0 = flat, positive = curved)
**Interpretation:** High ratio → positive curvature (sphere-like)

---

### 3. **Local Intrinsic Dimension**

Estimates the local dimensionality (related to curvature and complexity).

```python
dimensions = estimate_local_intrinsic_dimension(
    points, 
    k=15,
    method='mle'  # 'mle', 'pca', or 'correlation'
)
```

**When to use:** Understanding manifold structure, identifying singularities
**Output:** Dimension estimate per point
**Interpretation:** Changes in dimension indicate geometric transitions

---

### 4. **Ollivier-Ricci Curvature**

Transport-based discrete curvature for graphs.

```python
edge_curvatures, node_curvatures = compute_ollivier_ricci_curvature(
    points,
    k=10,
    alpha=0.5  # lazy random walk parameter
)
```

**When to use:** Graph-like data, discrete structures
**Output:** Curvature per edge and per node
**Interpretation:** Positive = concentration (sphere), Negative = dispersion (hyperbolic)

---

### 5. **Sectional Curvature**

Samples random 2D planes through tangent space.

```python
sectional_curv = compute_sectional_curvature(
    points,
    k=15,
    intrinsic_dim=3,
    n_plane_samples=10  # planes to sample per point
)
```

**When to use:** Statistical curvature distribution needed
**Output:** Dict with mean, std, min, max per point
**Interpretation:** Provides curvature variability

---

### 6. **Covariance Derivative**

Measures how local covariance changes along the manifold.

```python
curvatures = compute_covariance_curvature(
    points,
    k=15
)
```

**When to use:** Fast estimation, local structure analysis
**Output:** Scalar per point
**Interpretation:** Rate of covariance change

---

## Synthetic Manifolds for Testing

Generate manifolds with known curvature properties:

```python
# Positive curvature (K = 1/r²)
sphere = generate_sphere(n_points=500, radius=1.0, dim=2)

# Zero curvature  
torus = generate_torus(n_points=500, major_radius=2.0, minor_radius=1.0)

# Negative curvature (K = -1/a²)
hyperboloid = generate_hyperboloid(n_points=500, a=1.0)

# Varying curvature
swiss_roll = generate_swiss_roll(n_points=500, noise=0.1)
saddle = generate_saddle(n_points=500, scale=1.0)
```

---

## Visualization Functions

```python
# Distribution histogram
plot_curvature_distribution(curvatures, title="Curvature Distribution")

# 3D manifold colored by curvature
plot_curvature_on_manifold_3d(points_3d, curvatures)

# Compare multiple methods
plot_curvature_comparison({
    'Method 1': curvatures1,
    'Method 2': curvatures2
}, manifold_name="Sphere")

# Local geometry visualization
visualize_local_geometry(center_point, neighborhood, tangent_space)

# Dimension vs curvature relationship
plot_dimension_vs_curvature(dimensions, curvatures)
```

---

## Utilities

```python
# Build k-NN graph
neighbors, distances = build_knn_graph(points, k=15, return_distances=True)

# Compute distance matrix
dist_matrix = compute_distance_matrix(points, metric='euclidean')

# Estimate geodesic distances
geodesic_dist = estimate_geodesic_distance(points, k=15)

# Convert to NetworkX graph (for graph curvature methods)
G = compute_graph_from_points(points, k=15)
```

---

## Integration with Existing Code

### Using with Forman Curvature

```python
from geomechinterp.curvature.balanced_forman_curvature import balanced_forman_curvature
from curvature_measure import compute_graph_from_points

# Convert point cloud to graph
G = compute_graph_from_points(activations, k=15)
adjacency = nx.to_numpy_array(G)

# Compute Forman curvature
forman_curv = balanced_forman_curvature(adjacency)
```

---

## Method Comparison

| Method | Speed | Accuracy | Noise Robust | High-D | Best For |
|--------|-------|----------|--------------|--------|----------|
| PCA Tangent | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | Smooth manifolds |
| Geodesic | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Interpretability |
| Dimension | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | Structure analysis |
| Ollivier-Ricci | ⚡ | ⭐⭐⭐ | ⭐⭐ | ⚠️ | Discrete/graphs |
| Sectional | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ | ✅ | Distribution stats |
| Covariance | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | ✅ | Fast screening |

---

## Files

- **`curvature_measure.py`** - Main library with all implementations
- **`demo_example.py`** - Comprehensive demo script
- **`demo_curvature_methods.ipynb`** - Interactive notebook (in progress)
- **`curvature_measurements.ipynb`** - Original 2D/3D methods
- **`concept_curvature.ipynb`** - Concept-level analysis
- **`implementation_plan.md`** - Development roadmap

---

## Citation & References

### Key Papers

**Differential Geometry:**
- Pottmann et al. (2007) - "Principal curvatures from the integral invariant viewpoint"
- Do Carmo (1992) - "Riemannian Geometry"

**Discrete Curvature:**
- Ollivier (2009) - "Ricci curvature of Markov chains on metric spaces"
- Forman (2003) - "Bochner's method for cell complexes and combinatorial Ricci curvature"

**Dimensionality:**
- Levina & Bickel (2004) - "Maximum Likelihood Estimation of Intrinsic Dimension"
- Tenenbaum et al. (2000) - "A Global Geometric Framework for Nonlinear Dimensionality Reduction" (Isomap)

---

## Future Work

- [ ] GPU acceleration for large datasets
- [ ] Parallel processing for batch computation
- [ ] Integration with TransformerLens
- [ ] Layer-wise curvature tracking
- [ ] Curvature-based model diagnostics
- [ ] Attention pattern curvature analysis

---

## Contributing

Contributions welcome! Areas of interest:
- Performance optimizations
- Additional curvature methods
- Better visualizations
- Real-world applications

---

## License

MIT License (see parent repository)