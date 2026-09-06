"""
Curvature Measurement Functions

This module contains functions for computing various types of curvature:
- 2D parametric curve curvature
- 3D parametric curve curvature and torsion
- Surface curvature (Gaussian and Mean)
- Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_curvature_2d(x, y, t):
    """
    Compute curvature for a 2D parametric curve.

    Parameters:
    x (array): x-coordinates of the curve
    y (array): y-coordinates of the curve
    t (array): parameter values

    Returns:
    array: curvature values at each point
    """
    # Compute derivatives
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)

    # Curvature formula
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**(3/2)
    curvature = numerator / denominator

    return curvature


def bezier_curve(points, num_points=100):
    """
    Generate a Bezier curve from a set of control points.

    Parameters:
    points (list of tuples): Control points for the Bezier curve.
    num_points (int): Number of points to generate along the curve.

    Returns:
    tuple: x and y coordinates of the Bezier curve.
    """
    points = np.array(points)
    n = len(points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.array([(1 - t_i)**n * points[0] +
                      sum(np.math.comb(n, k) * (t_i**k) * ((1 - t_i)**(n - k)) * points[k] for k in range(1, n + 1))
                      for t_i in t])

    return curve[:, 0], curve[:, 1]


def plot_curvature(x, y, curvature):
    """
    Plot a 2D curve colored by curvature values.

    Parameters:
    x (array): x-coordinates of the curve
    y (array): y-coordinates of the curve
    curvature (array): curvature values at each point
    """
    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label="Circle")
    plt.scatter(x, y, c=curvature, cmap="viridis", label="Curvature")
    plt.colorbar(label="Curvature")
    plt.title("Curvature of a Circle")
    plt.legend()
    plt.axis("equal")
    plt.show()


def compute_curvature_torsion_3d(x, y, z, t):
    """
    Compute curvature and torsion for a 3D parametric curve.

    Parameters:
    x (array): x-coordinates of the curve
    y (array): y-coordinates of the curve
    z (array): z-coordinates of the curve
    t (array): parameter values

    Returns:
    tuple: (curvature, torsion) arrays
    """
    # Compute derivatives
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    dz = np.gradient(z, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)
    ddz = np.gradient(dz, t)
    dddx = np.gradient(ddx, t)
    dddy = np.gradient(ddy, t)
    dddz = np.gradient(ddz, t)

    # Tangent and normal vectors
    r_prime = np.array([dx, dy, dz])
    r_double_prime = np.array([ddx, ddy, ddz])
    r_triple_prime = np.array([dddx, dddy, dddz])

    # Curvature
    cross_r1_r2 = np.cross(r_prime.T, r_double_prime.T).T
    curvature = np.linalg.norm(cross_r1_r2, axis=0) / (np.linalg.norm(r_prime, axis=0)**3)

    # Torsion
    torsion = np.einsum('ij,ij->j', r_prime, np.cross(r_double_prime.T, r_triple_prime.T).T)
    torsion /= np.linalg.norm(cross_r1_r2, axis=0)**2

    return curvature, torsion


def compute_surface_curvature(z, x, y):
    """
    Compute Gaussian and Mean curvature for a surface z = f(x, y).

    Parameters:
    z (2D array): z-values of the surface
    x (array): x-coordinates (1D grid)
    y (array): y-coordinates (1D grid)

    Returns:
    tuple: (gaussian_curvature, mean_curvature) 2D arrays
    """
    dz_dx, dz_dy = np.gradient(z, x, y)
    dz_dxx, dz_dxy = np.gradient(dz_dx, x, y)
    dz_dyy, dz_dyx = np.gradient(dz_dy, x, y)

    # Gaussian curvature
    numerator_gaussian = dz_dxx * dz_dyy - dz_dxy**2
    denominator_gaussian = (1 + dz_dx**2 + dz_dy**2)**2
    gaussian_curvature = numerator_gaussian / denominator_gaussian

    # Mean curvature
    numerator_mean = ((1 + dz_dx**2) * dz_dyy - 2 * dz_dx * dz_dy * dz_dxy + (1 + dz_dy**2) * dz_dxx)
    denominator_mean = 2 * (1 + dz_dx**2 + dz_dy**2)**(3/2)
    mean_curvature = numerator_mean / denominator_mean

    return gaussian_curvature, mean_curvature


def generate_random_svg_logo(filename="logo.svg", size=(100, 100)):
    """
    Generate a random SVG logo and save it to a file.

    Parameters:
    filename (str): The name of the file to save the SVG logo.
    size (tuple): The size of the SVG canvas (width, height).
    """
    import svgwrite
    import random

    dwg = svgwrite.Drawing(filename, profile='tiny', size=size)

    # Create random shapes
    for _ in range(random.randint(1, 5)):
        shape_type = random.choice(['circle', 'rect', 'line'])
        if shape_type == 'circle':
            dwg.add(dwg.circle(center=(random.randint(0, size[0]), random.randint(0, size[1])),
                               r=random.randint(5, 20),
                               fill=svgwrite.rgb(random.random(), random.random(), random.random())))
        elif shape_type == 'rect':
            dwg.add(dwg.rect(insert=(random.randint(0, size[0]), random.randint(0, size[1])),
                             size=(random.randint(10, 30), random.randint(10, 30)),
                             fill=svgwrite.rgb(random.random(), random.random(), random.random())))
        elif shape_type == 'line':
            dwg.add(dwg.line(start=(random.randint(0, size[0]), random.randint(0, size[1])),
                             end=(random.randint(0, size[0]), random.randint(0, size[1])),
                             stroke=svgwrite.rgb(random.random(), random.random(), random.random()),
                             stroke_width=random.randint(1, 5)))

    dwg.save()


def svg_to_numpy(filename):
    """
    Convert an SVG file to a numpy array.

    Parameters:
    filename (str): The name of the SVG file to convert.

    Returns:
    np.ndarray: A numpy array representation of the SVG image.
    """
    from cairosvg import svg2png
    from PIL import Image

    # Convert SVG to PNG
    png_filename = filename.replace('.svg', '.png')
    svg2png(url=filename, write_to=png_filename)

    # Load PNG and convert to numpy array
    img = Image.open(png_filename)
    return np.array(img)


# ============================================================================
# HIGH-DIMENSIONAL POINT CLOUD UTILITIES
# ============================================================================

def compute_distance_matrix(points, metric='euclidean'):
    """
    Compute pairwise distance matrix for a point cloud.

    Parameters:
    points (np.ndarray): Array of shape (n_points, n_dims) representing point cloud
    metric (str): Distance metric to use. Options: 'euclidean', 'cosine', 'manhattan'

    Returns:
    np.ndarray: Distance matrix of shape (n_points, n_points)

    Reference:
    Standard distance metrics from scipy.spatial.distance
    """
    from scipy.spatial.distance import pdist, squareform

    if metric == 'cosine':
        # For cosine distance, need to handle it specially
        from sklearn.metrics.pairwise import cosine_distances
        return cosine_distances(points)
    else:
        # Use scipy's pdist for other metrics
        distances = pdist(points, metric=metric)
        return squareform(distances)


def build_knn_graph(points, k, metric='euclidean', return_distances=False):
    """
    Build k-nearest neighbor graph from point cloud.

    Parameters:
    points (np.ndarray): Array of shape (n_points, n_dims)
    k (int): Number of nearest neighbors
    metric (str): Distance metric ('euclidean', 'cosine', 'manhattan')
    return_distances (bool): If True, also return distances to neighbors

    Returns:
    neighbors (np.ndarray): Array of shape (n_points, k) with neighbor indices
    distances (np.ndarray): Optional, distances to neighbors if return_distances=True

    Reference:
    Scikit-learn NearestNeighbors implementation
    """
    from sklearn.neighbors import NearestNeighbors

    # k+1 because the point itself is included
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='auto')
    nbrs.fit(points)

    distances, neighbors = nbrs.kneighbors(points)

    # Remove the first neighbor (the point itself)
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]

    if return_distances:
        return neighbors, distances
    return neighbors


def compute_local_neighborhoods(points, k=None, radius=None, method='knn', metric='euclidean'):
    """
    Get local neighborhoods for each point in the point cloud.

    Parameters:
    points (np.ndarray): Array of shape (n_points, n_dims)
    k (int): Number of nearest neighbors (for method='knn')
    radius (float): Radius for ball query (for method='ball')
    method (str): 'knn' for k-nearest neighbors, 'ball' for radius-based
    metric (str): Distance metric to use

    Returns:
    list: List of arrays, where each array contains indices of neighbors for that point

    Reference:
    Scikit-learn NearestNeighbors for efficient neighborhood queries
    """
    from sklearn.neighbors import NearestNeighbors

    if method == 'knn':
        if k is None:
            raise ValueError("k must be specified for knn method")
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='auto')
        nbrs.fit(points)
        distances, neighbors = nbrs.kneighbors(points)
        # Remove self (first neighbor)
        return [neighbors[i, 1:] for i in range(len(points))]

    elif method == 'ball':
        if radius is None:
            raise ValueError("radius must be specified for ball method")
        nbrs = NearestNeighbors(radius=radius, metric=metric, algorithm='auto')
        nbrs.fit(points)
        distances, neighbors = nbrs.radius_neighbors(points)
        # Remove self
        return [neighbors[i][neighbors[i] != i] for i in range(len(points))]

    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_geodesic_distance(points, k=10, metric='euclidean', method='auto'):
    """
    Estimate geodesic distances on manifold via k-NN graph shortest paths.

    Parameters:
    points (np.ndarray): Array of shape (n_points, n_dims)
    k (int): Number of nearest neighbors for graph construction
    metric (str): Distance metric for local distances
    method (str): Shortest path algorithm:
        - 'auto': Automatically choose best algorithm (recommended)
        - 'D': Dijkstra's algorithm
        - 'FW': Floyd-Warshall algorithm
        - 'BF': Bellman-Ford algorithm

    Returns:
    np.ndarray: Geodesic distance matrix of shape (n_points, n_points)

    Reference:
    Tenenbaum et al. (2000) - Isomap algorithm
    The geodesic distance approximates the true manifold distance by shortest
    paths through the k-NN graph.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    n_points = len(points)

    # Build k-NN graph with distances
    neighbors, distances = build_knn_graph(points, k, metric=metric, return_distances=True)

    # Create sparse adjacency matrix
    row_indices = np.repeat(np.arange(n_points), k)
    col_indices = neighbors.flatten()
    data = distances.flatten()

    # Make symmetric (undirected graph)
    adjacency = csr_matrix((data, (row_indices, col_indices)), shape=(n_points, n_points))
    adjacency = adjacency + adjacency.T

    # Compute shortest paths (geodesic distances)
    geodesic_dist = shortest_path(adjacency, method=method, directed=False)

    return geodesic_dist


def compute_graph_from_points(points, k=10, metric='euclidean'):
    """
    Convert point cloud to NetworkX graph for use with graph-based curvature methods.

    Parameters:
    points (np.ndarray): Array of shape (n_points, n_dims)
    k (int): Number of nearest neighbors
    metric (str): Distance metric

    Returns:
    networkx.Graph: Undirected weighted graph with edge weights = distances

    Note:
    This is useful for integration with Forman-Ricci and Ollivier-Ricci
    curvature methods which operate on graphs.
    """
    import networkx as nx

    neighbors, distances = build_knn_graph(points, k, metric=metric, return_distances=True)

    G = nx.Graph()
    n_points = len(points)

    # Add nodes
    for i in range(n_points):
        G.add_node(i, pos=points[i])

    # Add edges
    for i in range(n_points):
        for j_idx, j in enumerate(neighbors[i]):
            dist = distances[i, j_idx]
            G.add_edge(i, j, weight=dist)

    return G


# ============================================================================
# HIGH-DIMENSIONAL CURVATURE ESTIMATION METHODS
# ============================================================================

def compute_local_pca_tangent_space(points, k=10, metric='euclidean', intrinsic_dim=None):
    """
    Compute local tangent spaces using PCA on k-nearest neighborhoods.

    Parameters:
    points (np.ndarray): Array of shape (n_points, n_dims)
    k (int): Number of nearest neighbors for local PCA
    metric (str): Distance metric
    intrinsic_dim (int): Number of principal components to keep. If None, keeps all.

    Returns:
    tangent_spaces (list): List of arrays, each of shape (intrinsic_dim, n_dims)
                          representing the tangent space basis at each point
    explained_variance (list): List of explained variance ratios for each point

    Reference:
    Pottmann et al. (2007) - Principal curvatures from the integral invariant viewpoint
    """
    from sklearn.decomposition import PCA

    n_points = len(points)
    neighbors = build_knn_graph(points, k, metric=metric, return_distances=False)

    tangent_spaces = []
    explained_variances = []

    for i in range(n_points):
        # Get neighborhood points (including the point itself)
        neighborhood_indices = np.concatenate([[i], neighbors[i]])
        neighborhood = points[neighborhood_indices]

        # Center the neighborhood
        centered = neighborhood - neighborhood.mean(axis=0)

        # Compute PCA
        if intrinsic_dim is not None:
            pca = PCA(n_components=intrinsic_dim)
        else:
            pca = PCA()

        pca.fit(centered)

        # Store tangent space basis vectors (principal components)
        tangent_spaces.append(pca.components_)
        explained_variances.append(pca.explained_variance_ratio_)

    return tangent_spaces, explained_variances


def compute_tangent_space_curvature(points, k=10, metric='euclidean', intrinsic_dim=2):
    """
    Estimate curvature by measuring rotation of local tangent spaces.

    The curvature at each point is estimated by measuring how much the tangent
    space rotates as we move to neighboring points. This is done by computing
    the principal angles between adjacent tangent spaces.

    Parameters:
    points (np.ndarray): Array of shape (n_points, n_dims)
    k (int): Number of nearest neighbors
    metric (str): Distance metric
    intrinsic_dim (int): Intrinsic dimensionality of the manifold

    Returns:
    curvatures (np.ndarray): Array of shape (n_points,) with curvature estimates
    curvature_details (dict): Dictionary with additional information:
        - 'mean_angles': mean principal angle to neighbors
        - 'max_angles': max principal angle to neighbors
        - 'tangent_spaces': the computed tangent spaces

    Reference:
    Based on the approach in:
    - Pottmann et al. (2007) Principal curvatures from integral invariant viewpoint
    - Goldberg et al. (2002) Understanding and measuring distances in manifolds

    Algorithm:
    1. Compute local tangent space at each point via PCA
    2. For each point, compute principal angles to tangent spaces of neighbors
    3. Curvature ≈ (angle / distance) averaged over neighbors
    """
    from scipy.linalg import subspace_angles

    n_points = len(points)

    # Compute tangent spaces
    tangent_spaces, _ = compute_local_pca_tangent_space(
        points, k=k, metric=metric, intrinsic_dim=intrinsic_dim
    )

    # Get neighbors and distances
    neighbors, distances = build_knn_graph(points, k, metric=metric, return_distances=True)

    curvatures = np.zeros(n_points)
    mean_angles = np.zeros(n_points)
    max_angles = np.zeros(n_points)

    for i in range(n_points):
        angles = []

        for j_idx, j in enumerate(neighbors[i]):
            # Compute principal angles between tangent spaces
            # Both tangent spaces should have shape (intrinsic_dim, n_dims)
            T_i = tangent_spaces[i].T  # Shape: (n_dims, intrinsic_dim)
            T_j = tangent_spaces[j].T  # Shape: (n_dims, intrinsic_dim)

            # Compute principal angles
            try:
                principal_angles = subspace_angles(T_i, T_j)
                # Use the maximum principal angle as a measure of tangent space rotation
                max_angle = np.max(principal_angles)
                angles.append(max_angle)

                # Curvature estimate: angle / distance
                dist = distances[i, j_idx]
                if dist > 1e-10:  # Avoid division by zero
                    curvatures[i] += max_angle / dist
            except Exception:
                # Handle degenerate cases
                continue

        if len(angles) > 0:
            curvatures[i] /= len(angles)  # Average over neighbors
            mean_angles[i] = np.mean(angles)
            max_angles[i] = np.max(angles)
        else:
            curvatures[i] = 0
            mean_angles[i] = 0
            max_angles[i] = 0

    curvature_details = {
        'mean_angles': mean_angles,
        'max_angles': max_angles,
        'tangent_spaces': tangent_spaces
    }

    return curvatures, curvature_details


def compute_geodesic_curvature(points, k=10, metric='euclidean', n_samples=None):
    """
    Estimate curvature via geodesic vs Euclidean distance ratio.

    High curvature manifolds have geodesic distances significantly larger than
    Euclidean distances. This method approximates geodesic distance via shortest
    paths on the k-NN graph.

    Parameters:
    points (np.ndarray): Array of shape (n_points, n_dims)
    k (int): Number of nearest neighbors for graph construction
    metric (str): Distance metric
    n_samples (int): Number of reference points to sample. If None, uses all points.
                    For large datasets, sampling improves efficiency.

    Returns:
    curvatures (np.ndarray): Array of shape (n_points,) with curvature estimates
    geodesic_dist (np.ndarray): Geodesic distance matrix (if n_samples is small enough)

    Reference:
    Based on Isomap (Tenenbaum et al. 2000) and geometric analysis of manifolds.
    The ratio r = d_geodesic / d_euclidean > 1 indicates positive curvature.
    """
    n_points = len(points)

    # Compute geodesic distances via shortest paths
    geodesic_dist = estimate_geodesic_distance(points, k=k, metric=metric)

    # Compute Euclidean distances
    euclidean_dist = compute_distance_matrix(points, metric=metric)

    # Avoid division by zero
    euclidean_dist = np.maximum(euclidean_dist, 1e-10)

    # Compute ratio
    ratio = geodesic_dist / euclidean_dist

    # For each point, compute mean ratio to neighbors as curvature estimate
    neighbors = build_knn_graph(points, k, metric=metric, return_distances=False)

    curvatures = np.zeros(n_points)
    for i in range(n_points):
        # Average ratio to k-nearest neighbors
        neighbor_ratios = ratio[i, neighbors[i]]
        curvatures[i] = np.mean(neighbor_ratios) - 1.0  # Subtract 1 so flat manifold has ~0

    return curvatures, geodesic_dist


def estimate_local_intrinsic_dimension(points, k=10, method='mle', metric='euclidean'):
    """
    Estimate local intrinsic dimension at each point.

    Parameters:
    points (np.ndarray): Array of shape (n_points, n_dims)
    k (int): Number of nearest neighbors
    method (str): Method to use:
        - 'mle': Maximum Likelihood Estimator (Levina & Bickel 2004)
        - 'pca': Count significant principal components
        - 'correlation': Correlation dimension
    metric (str): Distance metric

    Returns:
    dimensions (np.ndarray): Array of shape (n_points,) with dimension estimates

    Reference:
    - Levina & Bickel (2004) Maximum Likelihood Estimation of Intrinsic Dimension
    - Grassberger & Procaccia (1983) Measuring the Strangeness of Strange Attractors
    """
    n_points = len(points)
    neighbors, distances = build_knn_graph(points, k, metric=metric, return_distances=True)

    dimensions = np.zeros(n_points)

    if method == 'mle':
        # Maximum Likelihood Estimator
        # d ≈ (k-1) / sum(log(r_k / r_i)) for i < k
        for i in range(n_points):
            dists = distances[i]
            # Remove zero distances
            dists = dists[dists > 1e-10]
            if len(dists) < 2:
                dimensions[i] = 0
                continue

            # MLE formula
            r_k = dists[-1]  # Furthest neighbor
            log_ratios = np.log(r_k / dists[:-1])
            sum_log_ratios = np.sum(log_ratios)

            if sum_log_ratios > 0:
                dimensions[i] = (len(dists) - 1) / sum_log_ratios
            else:
                dimensions[i] = 0

    elif method == 'pca':
        # PCA-based: count eigenvalues above threshold
        tangent_spaces, explained_variances = compute_local_pca_tangent_space(
            points, k=k, metric=metric, intrinsic_dim=None
        )

        for i in range(n_points):
            # Count components explaining > 1% of variance
            threshold = 0.01
            significant_components = np.sum(explained_variances[i] > threshold)
            dimensions[i] = significant_components

    elif method == 'correlation':
        # Correlation dimension (Grassberger-Procaccia)
        # d ≈ d(log C(r)) / d(log r)
        for i in range(n_points):
            dists = distances[i]
            dists = dists[dists > 1e-10]
            if len(dists) < 3:
                dimensions[i] = 0
                continue

            # Compute correlation sum for different radii
            log_r = np.log(dists)
            log_C = np.log(np.arange(1, len(dists) + 1))

            # Linear regression to estimate slope
            slope = np.polyfit(log_r, log_C, 1)[0]
            dimensions[i] = max(0, slope)

    else:
        raise ValueError(f"Unknown method: {method}")

    return dimensions


def compute_ollivier_ricci_curvature(points, k=10, metric='euclidean', alpha=0.5):
    """
    Compute Ollivier-Ricci curvature for point cloud.

    Ollivier-Ricci curvature measures how probability distributions concentrate
    or disperse when transported along edges. Positive curvature indicates
    concentration (sphere-like), negative indicates dispersion (hyperbolic).

    Parameters:
    points (np.ndarray): Array of shape (n_points, n_dims)
    k (int): Number of nearest neighbors
    metric (str): Distance metric
    alpha (float): Lazy random walk parameter (0 = fully lazy, 1 = regular random walk)

    Returns:
    edge_curvatures (dict): Dictionary mapping (i, j) edges to curvature values
    node_curvatures (np.ndarray): Average curvature per node

    Reference:
    - Ollivier (2009) "Ricci curvature of Markov chains on metric spaces"
    - Ni et al. (2015) "Ricci curvature of the Internet topology"

    Algorithm:
    For edge (x, y):
    1. Define probability measures μ_x, μ_y on neighborhoods
    2. Compute Wasserstein distance W_1(μ_x, μ_y)
    3. κ(x, y) = 1 - W_1(μ_x, μ_y) / d(x, y)
    """
    n_points = len(points)
    neighbors, distances = build_knn_graph(points, k, metric=metric, return_distances=True)

    # Build edge list
    edge_curvatures = {}
    node_curvatures = np.zeros(n_points)
    node_degree = np.zeros(n_points)

    for i in range(n_points):
        for j_idx, j in enumerate(neighbors[i]):
            if i >= j:  # Avoid duplicate edges in undirected graph
                continue

            # Distance between i and j
            d_ij = distances[i, j_idx]

            if d_ij < 1e-10:
                continue

            # Define probability distributions on neighborhoods
            # Lazy random walk: stay at current node with probability (1-alpha)
            # Move to neighbors with probability alpha / degree

            # Get neighbors of i and j
            neighbors_i = neighbors[i]
            neighbors_j = neighbors[j]

            # Create probability distributions
            # Using uniform distribution on k-NN with lazy random walk

            # For computational efficiency, we use a simplified Wasserstein computation
            # based on the distance matrix between neighborhoods

            # Get unique nodes in both neighborhoods
            neighborhood_union = np.unique(np.concatenate([neighbors_i, neighbors_j, [i, j]]))

            # Build probability vectors
            prob_i = np.zeros(len(neighborhood_union))
            prob_j = np.zeros(len(neighborhood_union))

            for idx, node in enumerate(neighborhood_union):
                # Probability mass at node from distribution centered at i
                if node == i:
                    prob_i[idx] = 1 - alpha
                elif node in neighbors_i:
                    prob_i[idx] = alpha / len(neighbors_i)

                # Probability mass at node from distribution centered at j
                if node == j:
                    prob_j[idx] = 1 - alpha
                elif node in neighbors_j:
                    prob_j[idx] = alpha / len(neighbors_j)

            # Normalize
            prob_i = prob_i / np.sum(prob_i) if np.sum(prob_i) > 0 else prob_i
            prob_j = prob_j / np.sum(prob_j) if np.sum(prob_j) > 0 else prob_j

            # Compute distance matrix between neighborhood nodes
            neighborhood_points = points[neighborhood_union]
            dist_matrix = compute_distance_matrix(neighborhood_points, metric=metric)

            # Compute Wasserstein-1 distance using linear assignment
            # This is a simplified version; full computation requires optimal transport
            try:
                # Use 1D Wasserstein as approximation (project to line between i and j)
                # This is computationally efficient
                # Alternatively, compute Earth Mover's Distance

                # Simplified: use average distance weighted by probabilities
                W1 = np.sum(prob_i[:, None] * prob_j[None, :] * dist_matrix)

                # Ollivier-Ricci curvature
                kappa = 1 - W1 / d_ij

                edge_curvatures[(i, j)] = kappa
                node_curvatures[i] += kappa
                node_curvatures[j] += kappa
                node_degree[i] += 1
                node_degree[j] += 1

            except Exception:
                continue

    # Average curvature per node
    node_degree = np.maximum(node_degree, 1)  # Avoid division by zero
    node_curvatures = node_curvatures / node_degree

    return edge_curvatures, node_curvatures


def compute_sectional_curvature(points, k=10, metric='euclidean', intrinsic_dim=3,
                                n_plane_samples=10, random_seed=None):
    """
    Estimate sectional curvature by sampling 2D planes through tangent space.

    Sectional curvature measures the Gaussian curvature of 2D surfaces obtained
    by intersecting the manifold with 2-planes through the tangent space.

    Parameters:
    points (np.ndarray): Array of shape (n_points, n_dims)
    k (int): Number of nearest neighbors
    metric (str): Distance metric
    intrinsic_dim (int): Intrinsic dimensionality for tangent space estimation
    n_plane_samples (int): Number of random 2D planes to sample per point
    random_seed (int): Random seed for reproducibility

    Returns:
    sectional_curvatures (dict): Dictionary with statistics per point:
        - 'mean': mean sectional curvature
        - 'std': standard deviation
        - 'min': minimum
        - 'max': maximum
        - 'samples': all sampled curvatures

    Reference:
    - Do Carmo "Riemannian Geometry" (1992)
    - Buser & Karcher (1981) "Gromov's almost flat manifolds"
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_points = len(points)

    # Compute tangent spaces
    tangent_spaces, _ = compute_local_pca_tangent_space(
        points, k=k, metric=metric, intrinsic_dim=intrinsic_dim
    )

    sectional_curvatures = {
        'mean': np.zeros(n_points),
        'std': np.zeros(n_points),
        'min': np.zeros(n_points),
        'max': np.zeros(n_points),
        'samples': []
    }

    for i in range(n_points):
        tangent_basis = tangent_spaces[i]  # Shape: (intrinsic_dim, n_dims)

        if tangent_basis.shape[0] < 2:
            # Need at least 2D tangent space
            sectional_curvatures['samples'].append([])
            continue

        curvatures_at_point = []

        for _ in range(n_plane_samples):
            # Sample random 2D plane in tangent space
            # Choose 2 random orthogonal directions
            if intrinsic_dim == 2:
                # Use both principal components
                v1 = tangent_basis[0]
                v2 = tangent_basis[1]
            else:
                # Sample random coefficients
                coeffs = np.random.randn(2, intrinsic_dim)
                # Orthogonalize
                coeffs[0] = coeffs[0] / np.linalg.norm(coeffs[0])
                coeffs[1] = coeffs[1] - np.dot(coeffs[1], coeffs[0]) * coeffs[0]
                coeffs[1] = coeffs[1] / np.linalg.norm(coeffs[1])

                v1 = coeffs[0] @ tangent_basis
                v2 = coeffs[1] @ tangent_basis

            # Estimate curvature in this 2D plane
            # Use finite differences to approximate second fundamental form
            # This is a simplified approximation

            # Project local neighborhood onto 2D plane
            neighbors = build_knn_graph(points, k=min(k, 10), metric=metric, return_distances=False)
            neighbor_points = points[neighbors[i]]

            # Center on current point
            centered = neighbor_points - points[i]

            # Project onto plane spanned by v1, v2
            coords_v1 = centered @ v1
            coords_v2 = centered @ v2

            # Fit quadratic surface z = ax^2 + by^2 + cxy
            # Gaussian curvature K = ab - (c/2)^2
            if len(coords_v1) >= 5:  # Need enough points
                try:
                    # Compute residual in normal direction
                    plane_proj = np.outer(coords_v1, v1) + np.outer(coords_v2, v2)
                    residuals = np.linalg.norm(centered - plane_proj, axis=1)

                    # Fit quadratic form
                    A = np.column_stack([coords_v1**2, coords_v2**2, coords_v1 * coords_v2,
                                         coords_v1, coords_v2, np.ones_like(coords_v1)])
                    coeffs_fit = np.linalg.lstsq(A, residuals, rcond=None)[0]

                    a, b, c = coeffs_fit[0], coeffs_fit[1], coeffs_fit[2]

                    # Gaussian curvature approximation
                    K = a * b - (c / 2) ** 2
                    curvatures_at_point.append(K)
                except Exception:
                    continue

        if len(curvatures_at_point) > 0:
            sectional_curvatures['mean'][i] = np.mean(curvatures_at_point)
            sectional_curvatures['std'][i] = np.std(curvatures_at_point)
            sectional_curvatures['min'][i] = np.min(curvatures_at_point)
            sectional_curvatures['max'][i] = np.max(curvatures_at_point)
        else:
            sectional_curvatures['mean'][i] = 0
            sectional_curvatures['std'][i] = 0
            sectional_curvatures['min'][i] = 0
            sectional_curvatures['max'][i] = 0

        sectional_curvatures['samples'].append(curvatures_at_point)

    return sectional_curvatures


def compute_covariance_curvature(points, k=10, metric='euclidean'):
    """
    Estimate curvature via covariance derivative method.

    This method measures how the local covariance structure changes as we move
    along the manifold. High curvature regions show rapid changes in covariance.

    Parameters:
    points (np.ndarray): Array of shape (n_points, n_dims)
    k (int): Number of nearest neighbors
    metric (str): Distance metric

    Returns:
    curvatures (np.ndarray): Array of shape (n_points,) with curvature estimates

    Reference:
    Based on the idea that curvature affects local covariance structure.
    See: Pennec (2006) "Intrinsic Statistics on Riemannian Manifolds"
    """
    n_points = len(points)
    neighbors, distances = build_knn_graph(points, k, metric=metric, return_distances=True)

    curvatures = np.zeros(n_points)

    for i in range(n_points):
        # Compute local covariance at point i
        neighbor_indices = neighbors[i]
        neighborhood_i = points[neighbor_indices]
        centered_i = neighborhood_i - points[i]
        cov_i = np.cov(centered_i.T)

        # Measure covariance change to neighbors
        cov_changes = []

        for j_idx, j in enumerate(neighbor_indices):
            # Compute local covariance at neighbor j
            neighbor_indices_j = neighbors[j]
            neighborhood_j = points[neighbor_indices_j]
            centered_j = neighborhood_j - points[j]
            cov_j = np.cov(centered_j.T)

            # Measure difference in covariance matrices
            # Using Frobenius norm
            cov_diff = np.linalg.norm(cov_i - cov_j, 'fro')

            # Normalize by distance
            dist = distances[i, j_idx]
            if dist > 1e-10:
                cov_changes.append(cov_diff / dist)

        if len(cov_changes) > 0:
            curvatures[i] = np.mean(cov_changes)

    return curvatures


# ============================================================================
# SYNTHETIC MANIFOLD GENERATORS FOR TESTING
# ============================================================================

def generate_sphere(n_points=1000, radius=1.0, dim=3, ambient_dim=None):
    """
    Generate points uniformly distributed on an n-sphere.

    Parameters:
    n_points (int): Number of points to generate
    radius (float): Radius of the sphere
    dim (int): Intrinsic dimension of the sphere
    ambient_dim (int): Ambient space dimension (default: dim + 1)

    Returns:
    points (np.ndarray): Array of shape (n_points, ambient_dim)

    Properties:
    - Constant positive curvature K = 1/r²
    - For unit sphere (r=1): K = 1
    """
    if ambient_dim is None:
        ambient_dim = dim + 1

    # Generate from normal distribution and normalize
    points = np.random.randn(n_points, ambient_dim)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = radius * points / norms

    return points


def generate_torus(n_points=1000, major_radius=2.0, minor_radius=1.0):
    """
    Generate points uniformly distributed on a 2-torus embedded in 3D.

    Parameters:
    n_points (int): Number of points
    major_radius (float): Major radius (R)
    minor_radius (float): Minor radius (r)

    Returns:
    points (np.ndarray): Array of shape (n_points, 3)

    Properties:
    - Gaussian curvature K = cos(v) / (r(R + r*cos(v)))
    - Mean curvature varies spatially
    - Zero mean Gaussian curvature overall
    """
    u = np.random.uniform(0, 2*np.pi, n_points)
    v = np.random.uniform(0, 2*np.pi, n_points)

    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)

    points = np.column_stack([x, y, z])
    return points


def generate_hyperboloid(n_points=1000, a=1.0, dim=2):
    """
    Generate points on a hyperboloid (negative curvature manifold).

    Parameters:
    n_points (int): Number of points
    a (float): Scale parameter
    dim (int): Dimension (2 for sheet of hyperboloid)

    Returns:
    points (np.ndarray): Array of shape (n_points, 3)

    Properties:
    - Constant negative curvature K = -1/a²
    - Models hyperbolic space
    """
    # Use proper hyperbolic parametrization  
    # NOTE: This generates the UPPER sheet of a two-sheeted hyperboloid
    u = np.random.uniform(0, 2*np.pi, n_points)
    v = np.random.uniform(-1.5, 1.5, n_points)  # Hyperbolic parameter
    
    # Standard hyperboloid parametrization: z² - x² - y² = a²
    x = a * np.sinh(v) * np.cos(u)
    y = a * np.sinh(v) * np.sin(u)
    z = a * np.cosh(v)  # Always positive (upper sheet)
    
    points = np.column_stack([x, y, z])
    return points


def generate_swiss_roll(n_points=1000, noise=0.1, length=15):
    """
    Generate Swiss roll manifold (varying curvature).

    Parameters:
    n_points (int): Number of points
    noise (float): Noise level
    length (float): Length of the roll

    Returns:
    points (np.ndarray): Array of shape (n_points, 3)

    Properties:
    - Intrinsic dimension: 2
    - Extrinsic dimension: 3
    - Curvature varies along the manifold
    """
    t = np.random.uniform(1.5*np.pi, 4.5*np.pi, n_points)
    h = np.random.uniform(0, length, n_points)

    x = t * np.cos(t)
    y = h
    z = t * np.sin(t)

    # Add noise
    points = np.column_stack([x, y, z])
    points += noise * np.random.randn(n_points, 3)

    return points


def generate_saddle(n_points=1000, scale=1.0, noise=0.0):
    """
    Generate points on a saddle surface (hyperbolic paraboloid).

    Parameters:
    n_points (int): Number of points
    scale (float): Scale parameter
    noise (float): Noise level

    Returns:
    points (np.ndarray): Array of shape (n_points, 3)

    Properties:
    - Negative Gaussian curvature everywhere
    - z = x² - y²
    """
    x = np.random.uniform(-2, 2, n_points)
    y = np.random.uniform(-2, 2, n_points)
    z = scale * (x**2 - y**2)

    points = np.column_stack([x, y, z])
    points += noise * np.random.randn(n_points, 3)

    return points


def analytical_sphere_curvature(radius=1.0):
    """
    Return analytical Gaussian curvature for a sphere.

    Parameters:
    radius (float): Radius of sphere

    Returns:
    float: Gaussian curvature K = 1/r²
    """
    return 1.0 / (radius ** 2)


def analytical_torus_curvature(major_radius=2.0, minor_radius=1.0):
    """
    Return analytical Gaussian curvature for a torus.

    Parameters:
    major_radius (float): Major radius R
    minor_radius (float): Minor radius r

    Returns:
    dict: Contains mean and range of Gaussian curvature
    """
    # Gaussian curvature at outer equator (v=0)
    K_outer = 1.0 / (minor_radius * (major_radius + minor_radius))

    # Gaussian curvature at inner equator (v=π)
    K_inner = -1.0 / (minor_radius * (major_radius - minor_radius))

    return {
        'mean': 0.0,  # Average is zero
        'max': K_outer,
        'min': K_inner
    }


def analytical_hyperboloid_curvature(a=1.0):
    """
    Return analytical Gaussian curvature for hyperboloid.

    Parameters:
    a (float): Scale parameter

    Returns:
    float: Gaussian curvature K = -1/a²
    """
    return -1.0 / (a ** 2)


# ============================================================================
# VALIDATION AND TESTING FUNCTIONS
# ============================================================================

def validate_curvature_method(method_func, manifold_type='sphere', n_points=500,
                              k=15, tolerance=0.5, **manifold_params):
    """
    Validate a curvature estimation method on a synthetic manifold.

    Parameters:
    method_func (callable): Curvature computation function
    manifold_type (str): Type of manifold ('sphere', 'torus', 'hyperboloid', etc.)
    n_points (int): Number of points to generate
    k (int): Number of neighbors for curvature computation
    tolerance (float): Acceptable relative error
    manifold_params (dict): Parameters for manifold generation

    Returns:
    dict: Validation results with estimated vs analytical curvatures
    """
    # Generate manifold
    if manifold_type == 'sphere':
        radius = manifold_params.get('radius', 1.0)
        dim = manifold_params.get('dim', 2)
        points = generate_sphere(n_points, radius=radius, dim=dim)
        analytical_K = analytical_sphere_curvature(radius)

    elif manifold_type == 'torus':
        major_radius = manifold_params.get('major_radius', 2.0)
        minor_radius = manifold_params.get('minor_radius', 1.0)
        points = generate_torus(n_points, major_radius, minor_radius)
        analytical_K = analytical_torus_curvature(major_radius, minor_radius)

    elif manifold_type == 'hyperboloid':
        a = manifold_params.get('a', 1.0)
        points = generate_hyperboloid(n_points, a=a)
        analytical_K = analytical_hyperboloid_curvature(a)

    elif manifold_type == 'swiss_roll':
        noise = manifold_params.get('noise', 0.1)
        length = manifold_params.get('length', 15)
        points = generate_swiss_roll(n_points, noise, length)
        analytical_K = None  # No simple analytical formula

    elif manifold_type == 'saddle':
        scale = manifold_params.get('scale', 1.0)
        noise = manifold_params.get('noise', 0.0)
        points = generate_saddle(n_points, scale, noise)
        analytical_K = None  # Varies spatially

    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")

    # Compute curvature
    try:
        result = method_func(points, k=k)

        # Handle different return types
        if isinstance(result, tuple):
            curvatures = result[0]
        else:
            curvatures = result

        # Handle dict returns (like sectional curvature)
        if isinstance(curvatures, dict):
            curvatures = curvatures.get('mean', curvatures.get('curvatures', np.array([])))

        # Compute statistics
        if len(curvatures) > 0:
            estimated_mean = np.mean(curvatures)
            estimated_std = np.std(curvatures)
            estimated_median = np.median(curvatures)
        else:
            estimated_mean = 0
            estimated_std = 0
            estimated_median = 0

        # Compare with analytical
        if analytical_K is not None:
            if isinstance(analytical_K, dict):
                result_dict = {
                    'manifold': manifold_type,
                    'n_points': n_points,
                    'analytical': analytical_K,
                    'estimated_mean': estimated_mean,
                    'estimated_std': estimated_std,
                    'estimated_median': estimated_median,
                    'all_curvatures': curvatures,
                    'success': True
                }
            else:
                relative_error = abs(estimated_mean - analytical_K) / (abs(analytical_K) + 1e-10)
                success = relative_error < tolerance

                result_dict = {
                    'manifold': manifold_type,
                    'n_points': n_points,
                    'analytical': analytical_K,
                    'estimated_mean': estimated_mean,
                    'estimated_std': estimated_std,
                    'estimated_median': estimated_median,
                    'relative_error': relative_error,
                    'tolerance': tolerance,
                    'success': success,
                    'all_curvatures': curvatures
                }
        else:
            result_dict = {
                'manifold': manifold_type,
                'n_points': n_points,
                'analytical': 'N/A',
                'estimated_mean': estimated_mean,
                'estimated_std': estimated_std,
                'estimated_median': estimated_median,
                'all_curvatures': curvatures,
                'success': True
            }

        return result_dict

    except Exception as e:
        return {
            'manifold': manifold_type,
            'success': False,
            'error': str(e)
        }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_curvature_distribution(curvatures, title="Curvature Distribution", bins=50):
    """
    Plot histogram of curvature values.

    Parameters:
    curvatures (np.ndarray): Array of curvature values
    title (str): Plot title
    bins (int): Number of histogram bins
    """
    plt.figure(figsize=(10, 6))
    plt.hist(curvatures, bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel('Curvature', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.axvline(np.mean(curvatures), color='red', linestyle='--',
                label=f'Mean = {np.mean(curvatures):.4f}')
    plt.axvline(np.median(curvatures), color='green', linestyle='--',
                label=f'Median = {np.median(curvatures):.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_curvature_on_manifold_2d(points_2d, curvatures, title="Curvature on Manifold",
                                  cmap='viridis', alpha=0.7):
    """
    Plot 2D projection of manifold colored by curvature.

    Parameters:
    points_2d (np.ndarray): 2D projection of points, shape (n_points, 2)
    curvatures (np.ndarray): Curvature values at each point
    title (str): Plot title
    cmap (str): Colormap name
    alpha (float): Point transparency
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1], c=curvatures,
                          cmap=cmap, alpha=alpha, s=50, edgecolors='none')
    plt.colorbar(scatter, label='Curvature')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.show()


def plot_curvature_on_manifold_3d(points_3d, curvatures, title="Curvature on 3D Manifold",
                                  cmap='viridis', alpha=0.7):
    """
    Plot 3D manifold colored by curvature.

    Parameters:
    points_3d (np.ndarray): 3D points, shape (n_points, 3)
    curvatures (np.ndarray): Curvature values at each point
    title (str): Plot title
    cmap (str): Colormap name
    alpha (float): Point transparency
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                         c=curvatures, cmap=cmap, alpha=alpha, s=30)

    fig.colorbar(scatter, ax=ax, label='Curvature', shrink=0.5)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(title, fontsize=14)
    plt.show()


def plot_curvature_heatmap(curvature_matrix, title="Pairwise Curvature Heatmap"):
    """
    Plot heatmap of pairwise curvature values.

    Parameters:
    curvature_matrix (np.ndarray): Square matrix of curvatures, shape (n, n)
    title (str): Plot title
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(curvature_matrix, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Curvature')
    plt.xlabel('Point Index', fontsize=12)
    plt.ylabel('Point Index', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_local_geometry(center_point, neighborhood, tangent_space=None,
                             title="Local Geometry"):
    """
    Visualize local geometry: point, neighborhood, and tangent space.

    Parameters:
    center_point (np.ndarray): Central point, shape (3,)
    neighborhood (np.ndarray): Neighboring points, shape (n_neighbors, 3)
    tangent_space (np.ndarray): Tangent space basis vectors, shape (2 or 3, 3)
    title (str): Plot title
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot neighborhood
    ax.scatter(neighborhood[:, 0], neighborhood[:, 1], neighborhood[:, 2],
               c='lightblue', s=30, alpha=0.6, label='Neighborhood')

    # Plot center point
    ax.scatter([center_point[0]], [center_point[1]], [center_point[2]],
               c='red', s=100, marker='*', label='Center Point')

    # Plot tangent space if provided
    if tangent_space is not None:
        scale = np.max(np.linalg.norm(neighborhood - center_point, axis=1))
        for i, basis_vec in enumerate(tangent_space):
            end_point = center_point + scale * basis_vec
            ax.plot([center_point[0], end_point[0]],
                    [center_point[1], end_point[1]],
                    [center_point[2], end_point[2]],
                    linewidth=2, label=f'Tangent Basis {i+1}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14)
    ax.legend()
    plt.show()


def plot_curvature_comparison(methods_results, manifold_name="Manifold"):
    """
    Compare curvature distributions from multiple methods.

    Parameters:
    methods_results (dict): Dictionary mapping method names to curvature arrays
    manifold_name (str): Name of the manifold being analyzed
    """
    n_methods = len(methods_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))

    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, curvatures) in zip(axes, methods_results.items()):
        ax.hist(curvatures, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Curvature')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{method_name}\nMean: {np.mean(curvatures):.4f}')
        ax.axvline(np.mean(curvatures), color='red', linestyle='--', linewidth=2)
        ax.grid(alpha=0.3)

    fig.suptitle(f'Curvature Comparison on {manifold_name}', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_dimension_vs_curvature(dimensions, curvatures, title="Intrinsic Dimension vs Curvature"):
    """
    Plot relationship between local intrinsic dimension and curvature.

    Parameters:
    dimensions (np.ndarray): Local dimension estimates
    curvatures (np.ndarray): Curvature values
    title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(dimensions, curvatures, alpha=0.5, s=30)
    plt.xlabel('Local Intrinsic Dimension', fontsize=12)
    plt.ylabel('Curvature', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(dimensions, curvatures, 1)
    p = np.poly1d(z)
    plt.plot(np.unique(dimensions), p(np.unique(dimensions)),
             "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    plt.legend()
    plt.show()


def plot_manifold_curvature_comparison(points, curvature_methods, manifold_name="Manifold",
                                       cmap='viridis', figsize=None):
    """
    Compare multiple curvature methods by showing the same manifold colored by each method.

    Parameters:
    points (np.ndarray): Points defining the manifold, shape (n_points, 3)
    curvature_methods (dict): Dictionary mapping method names to curvature arrays
    manifold_name (str): Name of the manifold
    cmap (str): Colormap to use
    figsize (tuple): Figure size, defaults to (5*n_methods, 5)

    Example:
    >>> methods = {
    ...     'PCA Tangent': curvatures_pca,
    ...     'Geodesic': curvatures_geo,
    ...     'Ollivier-Ricci': curvatures_or
    ... }
    >>> plot_manifold_curvature_comparison(sphere_points, methods, "Sphere")
    """
    n_methods = len(curvature_methods)
    if figsize is None:
        figsize = (5 * n_methods, 5)

    fig = plt.figure(figsize=figsize)

    for idx, (method_name, curvatures) in enumerate(curvature_methods.items(), 1):
        ax = fig.add_subplot(1, n_methods, idx, projection='3d')

        # Normalize curvatures for consistent coloring
        vmin, vmax = np.percentile(curvatures, [5, 95])

        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=curvatures, cmap=cmap, s=20, alpha=0.7,
                             vmin=vmin, vmax=vmax)

        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_title(f'{method_name}\nMean: {np.mean(curvatures):.4f}', fontsize=11)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('Curvature', fontsize=8)

    fig.suptitle(f'Curvature Methods Comparison: {manifold_name}', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()


def plot_manifold_curvature_grid(manifolds_dict, curvature_method_func,
                                 method_name="Curvature", cmap='viridis', **method_kwargs):
    """
    Show a grid of different manifolds, all colored by the same curvature method.

    Parameters:
    manifolds_dict (dict): Dictionary mapping manifold names to point arrays
    curvature_method_func (callable): Function to compute curvature
    method_name (str): Name of the curvature method
    cmap (str): Colormap to use
    method_kwargs: Additional arguments for curvature method

    Example:
    >>> manifolds = {
    ...     'Sphere': sphere_points,
    ...     'Torus': torus_points,
    ...     'Hyperboloid': hyperboloid_points
    ... }
    >>> plot_manifold_curvature_grid(manifolds, compute_tangent_space_curvature,
    ...                               "PCA Tangent Space", k=15, intrinsic_dim=2)
    """
    n_manifolds = len(manifolds_dict)
    ncols = min(3, n_manifolds)
    nrows = (n_manifolds + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))

    for idx, (manifold_name, points) in enumerate(manifolds_dict.items(), 1):
        ax = fig.add_subplot(nrows, ncols, idx, projection='3d')

        # Compute curvature
        try:
            result = curvature_method_func(points, **method_kwargs)

            # Handle different return types
            if isinstance(result, tuple):
                curvatures = result[0]
            else:
                curvatures = result

            if isinstance(curvatures, dict):
                curvatures = curvatures.get('mean', list(curvatures.values())[0])

            # Plot
            vmin, vmax = np.percentile(curvatures, [5, 95])
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                 c=curvatures, cmap=cmap, s=20, alpha=0.7,
                                 vmin=vmin, vmax=vmax)

            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.set_zlabel('Z', fontsize=8)
            ax.set_title(f'{manifold_name}\nMean: {np.mean(curvatures):.4f}',
                         fontsize=11)

            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
            cbar.set_label('Curvature', fontsize=8)

        except Exception as e:
            ax.text(0.5, 0.5, 0.5, f'Error:\n{str(e)[:50]}',
                    ha='center', va='center', fontsize=9)
            ax.set_title(f'{manifold_name}\n(Error)', fontsize=11)

    fig.suptitle(f'{method_name} on Different Manifolds', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()
