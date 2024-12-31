# Implementation of balanced Forman curvature and SDRF (Stochastic Discrete Ricci Flow) algorithms
# Taken from https://github.com/jctops/understanding-oversquashing/blob/main/gdl/src/gdl/curvature/numba.py

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
import networkx as nx

from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)

import pytensor

pytensor.config.floatX = "float32"


def balanced_forman_curvature(A, C=None):
    """Wrapper function to calculate balanced Forman curvature"""
    N = A.shape[0]
    A2 = jnp.matmul(A, A)
    d_in = jnp.sum(A, axis=0)
    d_out = jnp.sum(A, axis=1)
    if C is None:
        C = jnp.zeros((N, N))

    C = _balanced_forman_curvature(A, A2, d_in, d_out, N)
    return C


def _curvature_single_edge(A, A2, d_in, d_out, i, j, N):
    """Calculate curvature for a single edge (i, j)"""

    # Instead of indexing A directly, use jnp.take
    edge_exists = jnp.take(A, i * N + j)

    # Skip if no edge exists
    def zero_case():
        return 0.0, 0.0, 0.0, 0.0

    def compute_case():
        # Get max and min of in/out degrees
        d_in_i = jnp.take(d_in, i)
        d_out_j = jnp.take(d_out, j)
        d_max = jnp.where(d_in_i > d_out_j, d_in_i, d_out_j)
        d_min = jnp.where(d_in_i > d_out_j, d_out_j, d_in_i)

        def zero_deg_case():
            return (
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.float32(0.0),
            )

        def compute_cycles():
            def body_fun(k, carry):
                sharp_ij, lambda_ij = carry

                # Check for 4-cycles through node k
                ak_ = jnp.take(A, k, axis=1)
                ak_i = jnp.take(ak_, i)
                ak_j = jnp.take(ak_, j)
                a2ik = jnp.take(A2, i * N + k)
                a2kj = jnp.take(A2, k * N + j)
                aij = jnp.take(A, i * N + j)
                TMP1 = ak_j * (a2ik - ak_i) * aij
                TMP2 = ak_i * (a2kj - ak_j) * aij

                sharp_ij += jnp.float32((TMP1 > 0) + (TMP2 > 0))
                lambda_ij = jnp.maximum(
                    lambda_ij, jnp.maximum(TMP1 * (TMP1 > 0), TMP2 * (TMP2 > 0))
                )

                return sharp_ij, lambda_ij

            sharp_ij, lambda_ij = lax.fori_loop(
                0,
                N,
                lambda k, carry: body_fun(k, carry),
                (jnp.float32(0), jnp.float32(0.0)),
            )
            return d_max, d_min, sharp_ij, lambda_ij

        return lax.cond(d_max * d_min == 0, zero_deg_case, compute_cycles)

    return lax.cond(edge_exists == 0, zero_case, compute_case)


def _balanced_forman_curvature(A, A2, d_in, d_out, N):
    """Calculate balanced Forman curvature for a directed graph"""

    def compute_curvature(i, j):
        d_max, d_min, sharp_ij, lambda_ij = _curvature_single_edge(
            A, A2, d_in, d_out, i, j, N
        )

        # Calculate final curvature combining degree terms and 4-cycle terms
        aij = jnp.take(A, i * N + j)
        a2ij = jnp.take(A2, i * N + j)
        curv = (2 / d_max) + (2 / d_min) - 2 + (2 / d_max + 1 / d_min) * a2ij * aij
        # lambda_ij is the maximum contribution from 4-cycles
        curv = jnp.where(lambda_ij > 0, curv + sharp_ij / (d_max * lambda_ij), curv)
        return curv

    compiled_compute_curvature = jax.jit(compute_curvature)

    # Create indices for all pairs
    # Create separate i and j arrays for all pairs
    i_indices, j_indices = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing="ij")
    i_indices = i_indices.flatten()
    j_indices = j_indices.flatten()

    # Vectorize computation across all pairs
    curvatures = vmap(compiled_compute_curvature, in_axes=(0, 0))(i_indices, j_indices)

    # Reshape back to matrix
    return curvatures.reshape((N, N))


def softmax(a, tau=1):
    """Compute softmax values with temperature parameter tau"""
    exp_a = jnp.exp(a * tau)
    return exp_a / jnp.sum(exp_a)


@jit
def _balanced_forman_post_delta(
    A, A2, d_in_x, d_out_y, N, D, x, y, i_neighbors, j_neighbors, dim_i, dim_j
):
    """Calculate change in curvature after adding potential edges"""

    def compute_delta(IJ):
        I, J = IJ
        i = i_neighbors[I]
        j = j_neighbors[J]

        # Skip invalid edges
        def invalid_case():
            return -1000.0

        def valid_case():
            # Update degrees after potential edge addition
            d_in_x_new = d_in_x + (j == x)
            d_out_y_new = d_out_y + (i == y)

            def zero_deg_case():
                return 0.0

            def compute_curvature():
                d_max = jnp.where(d_in_x_new > d_out_y_new, d_in_x_new, d_out_y_new)
                d_min = jnp.where(d_in_x_new > d_out_y_new, d_out_y_new, d_in_x_new)

                # Rest of computation similar to original but using jax operations
                # ... (rest of the computation)
                # This part would need careful translation maintaining the same logic
                # but using jax operations

                return d_max  # Placeholder - actual computation needed here

            return lax.cond(
                d_in_x_new * d_out_y_new == 0, zero_deg_case, compute_curvature
            )

        return lax.cond((i == j) | (A[i, j] != 0), invalid_case, valid_case)

    # Create indices for all pairs
    idx = jnp.array([(I, J) for I in range(dim_i) for J in range(dim_j)])

    # Vectorize computation across all pairs
    deltas = vmap(compute_delta)(idx)

    # Reshape back to matrix
    return deltas.reshape((dim_i, dim_j))


# Rest of the code remains similar but using jax.numpy operations
# The SDRF function would need similar treatment, replacing numpy operations
# with their jax equivalents where appropriate


if __name__ == "__main__":
    # test on random graph from networkx
    # G = nx.gnp_random_graph(20, 0.5)
    G = nx.grid_2d_graph(10, 10)
    A = nx.to_numpy_array(G, dtype=jnp.float32)
    print(A)
    C = balanced_forman_curvature(A)
    print(C)
