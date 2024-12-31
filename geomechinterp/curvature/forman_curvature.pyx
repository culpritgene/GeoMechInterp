# curvature.pyx

from cython.parallel import prange
import numpy as np
cimport numpy as np


def curvature_single_edge(
    np.int_t[:, :] A,
    np.int_t[:, :] A2,
    np.int_t[:] d_in,
    np.int_t[:] d_out,
    int i,
    int j,
    int N
):
    """Calculate curvature for a single edge (i, j)"""
    # Skip if no edge exists
    if A[i, j] == 0:
        return 0.0, 0.0, 0.0, 0.0

    cdef double d_max, d_min
    cdef int sharp_ij = 0
    cdef double lambda_ij = 0.0
    cdef double TMP
    cdef int k

    # Get max and min of in/out degrees
    if d_in[i] > d_out[j]:
        d_max = d_in[i]
        d_min = d_out[j]
    else:
        d_max = d_out[j]
        d_min = d_in[i]

    if d_max * d_min == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Calculate contribution from 4-cycles
    for k in range(N):
        # Check for 4-cycles through node k
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


@cython.boundscheck(False)
@cython.wraparound(False)
def _balanced_forman_curvature(
    np.int_t[:, :] A,
    np.int_t[:, :] A2,
    np.int_t[:] d_in,
    np.int_t[:] d_out,
    int N,
    np.double_t[:, :] C
):
    """Calculate balanced Forman curvature for a directed graph

    Args:
        A: Adjacency matrix
        A2: Square of adjacency matrix (A*A)
        d_in: In-degrees of nodes
        d_out: Out-degrees of nodes
        N: Number of nodes
        C: Output curvature matrix
    """
    cdef int i, j
    cdef double d_max, d_min, sharp_ij, lambda_ij

    for i in prange(N, nogil=True):  # Parallelize the outer loop
        for j in range(N):
            d_max, d_min, sharp_ij, lambda_ij = curvature_single_edge(
                A, A2, d_in, d_out, i, j, N
            )

            # Calculate final curvature combining degree terms and 4-cycle terms
            C[i, j] = (
                (2.0 / d_max)
                + (2.0 / d_min)
                - 2.0
                + (2.0 / d_max + 1.0 / d_min) * A2[i, j] * A[i, j]
            )
            # lambda_ij is the maximum contribution from 4-cycles
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)

    return C