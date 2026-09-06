Perfect! Now I can see you already have **Forman-Ricci curvature** implemented. Here's a detailed implementation plan:

## 📋 Implementation Plan for High-Dimensional Curvature Methods

### **Phase 1: Foundation & Utilities** (Tasks 7)
**Goal:** Build common infrastructure for all methods

**What to implement:**
- `build_knn_graph(points, k)` - construct k-NN graph from point cloud
- `compute_distance_matrix(points, metric='euclidean')` - various distance metrics
- `compute_local_neighborhoods(points, k, method='knn')` - get local neighborhoods
- `estimate_geodesic_distance(graph)` - shortest path distances on graph

**Dependencies:** `numpy`, `scipy`, `scikit-learn`

---

### **Phase 2: Core Methods** (Tasks 1-6)

#### **Task 1: Local PCA Tangent Space Curvature** ⭐ START HERE
**Algorithm:**
1. For each point, find k-nearest neighbors
2. Compute PCA on neighborhood → tangent space (principal components)
3. Measure how tangent space rotates between neighboring points
4. Curvature = angle between adjacent tangent spaces / distance

**Output:** Curvature scalar per point

**Reference:** Principal Curvatures from the Integral Invariant Viewpoint (Pottmann et al.)

---

#### **Task 2: Local Intrinsic Dimension**
**Algorithm:**
1. For each point's k-NN neighborhood
2. Compute correlation dimension: `d = d(log(C(r)))/d(log(r))`
3. Or use MLE: based on distances to k nearest neighbors
4. Or use PCA: count eigenvalues above threshold

**Output:** Local dimension estimate per point

**Use case:** Identifies where manifold changes dimension (singularities)

---

#### **Task 3: Geodesic vs Euclidean Distance Ratio**
**Algorithm:**
1. Build k-NN graph from point cloud
2. Compute geodesic distances (shortest paths on graph)
3. Compute Euclidean distances
4. Ratio `r = d_geodesic / d_euclidean`
5. High ratio → high positive curvature

**Output:** Curvature metric per point pair (can aggregate per point)

**Reference:** Geometry-aware dimensionality reduction

---

#### **Task 4: Ollivier-Ricci Curvature**
**Algorithm:**
1. Build k-NN graph
2. For each edge (x,y), define probability measures μ_x, μ_y on neighborhoods
3. Compute Wasserstein distance W(μ_x, μ_y)
4. Curvature: `κ(x,y) = 1 - W(μ_x, μ_y) / d(x,y)`

**Output:** Curvature per edge

**Note:** Complements your existing Forman curvature!

**Reference:** Ollivier (2009), Ricci curvature of metric spaces

---

#### **Task 5: Sectional Curvature**
**Algorithm:**
1. Estimate local tangent space via PCA
2. Sample random 2D planes through tangent space
3. Compute 2D Gaussian curvature in each plane (using existing methods)
4. Aggregate statistics (mean, std, min, max)

**Output:** Curvature distribution per point

**Reference:** Do Carmo, Riemannian Geometry

---

#### **Task 6: Covariance Derivative Method**
**Algorithm:**
1. For each point, compute local covariance matrix from k-NN
2. Move to neighboring points, compute covariance there
3. Measure Frobenius norm of covariance change
4. Normalize by distance traveled

**Output:** Scalar curvature per point

**Reference:** Local covariance geometry

---

### **Phase 3: Testing & Validation** (Task 8)

**Create synthetic test manifolds:**
- n-sphere in (n+1)D space (constant positive curvature)
- n-torus (zero curvature)
- n-hyperboloid (constant negative curvature)
- Swiss roll (varying curvature)
- Saddle surface in high-D

**Validation approach:**
- Compare numerical estimates to analytical values
- Check consistency across methods
- Test scalability (1000, 10000, 100000 points)

---

### **Phase 4: Visualization** (Task 9)

**Implement:**
- `plot_curvature_distribution(curvatures)` - histogram
- `plot_curvature_on_manifold_2d(points_2d, curvatures)` - colored scatter for projections
- `plot_curvature_heatmap(curvature_matrix)` - for pairwise curvatures
- `visualize_local_geometry(point, neighborhood, tangent_space)` - 3D plots

---

### **Recommended Order:**
1. ✅ Task 7 (utilities) 
2. ✅ Task 1 (PCA tangent space) - **most practical**
3. ✅ Task 3 (geodesic ratio) - **easiest, interpretable**
4. ✅ Task 2 (local dimension) - **complements curvature**
5. Task 8 (test on synthetic data)
6. Task 4 (Ollivier-Ricci)
7. Task 5 (sectional curvature)
8. Task 6 (covariance derivative)
9. Task 9 (visualization)

Should I start implementing? I'll begin with Task 7 (utilities) and Task 1 (PCA tangent space curvature).