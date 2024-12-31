import networkx as nx
import scipy.sparse
import jax
from jax.experimental.sparse import BCOO
import jax.numpy as jnp

# Step 1: Create a graph in networkx
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])  # Add edges

# Step 2: Get the adjacency matrix in sparse format
adj_matrix_scipy = nx.to_scipy_sparse_array(G, format="coo")

# Step 3: Convert the SciPy sparse matrix to JAX sparse format
adj_matrix_jax = BCOO.from_scipy_sparse(adj_matrix_scipy)


# Step 4: Define a JAX function that operates on the sparse matrix
@jax.jit
def get_edge_weight(adj_sparse, i, j):
    # Perform sparse indexing: get the edge weight between nodes i and j
    return jnp.take(adj_sparse, jnp.array([i, j]))


# Test the JAX function with dynamic indices
i, j = 2, 3
edge_weight = get_edge_weight(adj_matrix_jax, i, j)

print(f"Edge weight between nodes {i} and {j}: {edge_weight}")
