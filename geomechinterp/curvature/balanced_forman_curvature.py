# Implementation of balanced Forman curvature and SDRF (Stochastic Discrete Ricci Flow) algorithms
# Taken from https://github.com/jctops/understanding-oversquashing/blob/main/gdl/src/gdl/curvature/numba.py

import numpy as np
import networkx as nx
from numba import jit, prange
import torch
from tqdm import tqdm

from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)


def balanced_forman_curvature(A, C=None):
    """Wrapper function to calculate balanced Forman curvature"""
    N = A.shape[0]
    A2 = np.matmul(A, A)
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = np.zeros((N, N))

    _balanced_forman_curvature(A, A2, d_in, d_out, N, C)
    return C


@jit(nopython=True)
def _curvature_single_edge(A, A2, d_in, d_out, i, j, N):
    """Calculate curvature for a single edge (i, j)"""
    # Skip if no edge exists
    if A[i, j] == 0:
        return 0, 0, 0, 0

    # Get max and min of in/out degrees
    if d_in[i] > d_out[j]:
        d_max = d_in[i]
        d_min = d_out[j]
    else:
        d_max = d_out[j]
        d_min = d_in[i]

    if d_max * d_min == 0:
        return 0, 0, 0, 0

    # Calculate contribution from 4-cycles

    sharp_ij = 0
    lambda_ij = 0
    for k in range(N):
        # Check for 4-cycles through node k
        # A^2 encodes nodes connected by 2 hops
        # what does A^2 - A mean?
        # A^2[i, j] - A[i, j] = A[i, j] @ A[j, i] - A[i, j] =
        # = A[i, j] @ (A[j, i] - 1)
        TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
        if TMP > 0:
            sharp_ij += 1
            if TMP > lambda_ij:
                lambda_ij = TMP

        TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
        if TMP > 0:
            sharp_ij += 1
            if TMP > lambda_ij:
                lambda_ij = TMP

    return d_max, d_min, sharp_ij, lambda_ij


@jit(nopython=True)
def _balanced_forman_curvature(A, A2, d_in, d_out, N, C):
    """Calculate balanced Forman curvature for a directed graph

    Args:
        A: Adjacency matrix
        A2: Square of adjacency matrix (A*A)
        d_in: In-degrees of nodes
        d_out: Out-degrees of nodes
        N: Number of nodes
        C: Output curvature matrix
    """
    # use visual tracker of for loop progress

    for i in prange(N):
        for j in prange(N):
            d_max, d_min, sharp_ij, lambda_ij = _curvature_single_edge(
                A, A2, d_in, d_out, i, j, N
            )
            if d_max * d_min == 0:
                continue
            # Calculate final curvature combining degree terms and 4-cycle terms
            C[i, j] = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
            )
            # lambda_ij is the maximum contribution from 4-cycles
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)
    return C


def softmax(a, tau=1):
    """Compute softmax values with temperature parameter tau"""
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()


@jit(nopython=True)
def _balanced_forman_post_delta(
    A, A2, d_in_x, d_out_y, N, D, x, y, i_neighbors, j_neighbors, dim_i, dim_j
):
    """Calculate change in curvature after adding potential edges

    Args:
        A: Adjacency matrix
        A2: Square of adjacency matrix
        d_in_x: In-degree of node x
        d_out_y: Out-degree of node y
        N: Number of nodes
        D: Output delta matrix
        x,y: Nodes between which we're considering adding edges
        i_neighbors, j_neighbors: Lists of neighbor nodes to consider
        dim_i, dim_j: Dimensions of output matrix
    """
    for row_idx in prange(dim_i):
        for col_idx in prange(dim_j):
            i = i_neighbors[row_idx]
            j = j_neighbors[col_idx]

            # Skip invalid edges
            if (i == j) or (A[i, j] != 0):
                D[row_idx, col_idx] = -1000
                break

            # Update degrees after potential edge addition
            if j == x:
                d_in_x += 1
            elif i == y:
                d_out_y += 1

            if d_in_x * d_out_y == 0:
                D[row_idx, col_idx] = 0
                break

            if d_in_x > d_out_y:
                d_max = d_in_x
                d_min = d_out_y
            else:
                d_max = d_out_y
                d_min = d_in_x

            # Update triangle count after potential edge addition
            A2_x_y = A2[x, y]
            if (x == i) and (A[j, y] != 0):
                A2_x_y += A[j, y]
            elif (y == j) and (A[x, i] != 0):
                A2_x_y += A[x, i]

            # Update 4-cycle count after potential edge addition
            sharp_ij = 0
            lambda_ij = 0
            for z in range(N):
                A_z_y = A[z, y] + 0
                A_x_z = A[x, z] + 0
                A2_z_y = A2[z, y] + 0
                A2_x_z = A2[x, z] + 0

                # Update adjacency values for potential new edge
                if (z == i) and (y == j):
                    A_z_y += 1
                if (x == i) and (z == j):
                    A_x_z += 1
                if (z == i) and (A[j, y] != 0):
                    A2_z_y += A[j, y]
                if (x == i) and (A[j, z] != 0):
                    A2_x_z += A[j, z]
                if (y == j) and (A[z, i] != 0):
                    A2_z_y += A[z, i]
                if (z == j) and (A[x, i] != 0):
                    A2_x_z += A[x, i]

                # Check for new 4-cycles
                TMP = A_z_y * (A2_x_z - A_x_z) * A[x, y]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A_x_z * (A2_z_y - A_z_y) * A[x, y]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            # Calculate final curvature delta
            D[row_idx, col_idx] = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2_x_y * A[x, y]
            )
            if lambda_ij > 0:
                D[row_idx, col_idx] += sharp_ij / (d_max * lambda_ij)


def balanced_forman_post_delta(A, x, y, i_neighbors, j_neighbors, D=None):
    """Wrapper function to calculate curvature change after adding edges"""
    N = A.shape[0]
    A2 = np.matmul(A, A)
    d_in = A[:, x].sum()
    d_out = A[y].sum()
    if D is None:
        D = np.zeros((len(i_neighbors), len(j_neighbors)))

    _balanced_forman_post_delta(
        A,
        A2,
        d_in,
        d_out,
        N,
        D,
        x,
        y,
        np.array(i_neighbors),
        np.array(j_neighbors),
        D.shape[0],
        D.shape[1],
    )
    return D


def sdrf(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
):
    """Stochastic Discrete Ricci Flow algorithm

    Args:
        data: Input graph data
        loops: Number of iterations
        remove_edges: Whether to remove edges with high curvature
        removal_bound: Threshold for edge removal
        tau: Temperature parameter for softmax
        is_undirected: Whether graph is undirected
    """
    # Initialize adjacency matrix
    N = data.x.shape[0]
    A = np.zeros(shape=(N, N))
    if is_undirected:
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            if i != j:
                A[i, j] = A[j, i] = 1.0
    else:
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            if i != j:
                A[i, j] = 1.0
    N = A.shape[0]
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()
    C = np.zeros((N, N))

    for x in range(loops):
        can_add = True
        # Calculate curvature
        balanced_forman_curvature(A, C=C)
        ix_min = C.argmin()
        x = ix_min // N
        y = ix_min % N

        # Get neighbors for edge addition candidates
        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [y]
        candidates = []
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    candidates.append((i, j))

        # Add edge that improves curvature
        if len(candidates):
            D = balanced_forman_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for i, j in candidates:
                improvements.append(
                    (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)]
                )

            k, end_node = candidates[
                np.random.choice(
                    range(len(candidates)), p=softmax(np.array(improvements), tau=tau)
                )
            ]
            G.add_edge(k, end_node)
            if is_undirected:
                A[k, end_node] = A[end_node, k] = 1
            else:
                A[k, end_node] = 1
        else:
            can_add = False
            if not remove_edges:
                break

        # Remove edge with high curvature if enabled
        if remove_edges:
            ix_max = C.argmax()
            x = ix_max // N
            y = ix_max % N
            if C[x, y] > removal_bound:
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
            else:
                if can_add is False:
                    break

    return from_networkx(G)


import networkx as nx

if __name__ == "__main__":
    # test on random graph from networkx
    # G = nx.gnp_random_graph(20, 0.5)
    G = nx.grid_2d_graph(10, 10)
    A = nx.to_numpy_array(G)
    print(A)
    C = balanced_forman_curvature(A)
    print(C)
