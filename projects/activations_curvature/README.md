**Research Focus**: Analyzing Curvature in the residual stream of toy transformers or GPT2, using methods from classical diff geometry, manifold learning and hessian analysis; as well as anylizing curvature of the attention space using discrerte Olivier Curvature suitable to study discretized attention graphs. The future idea is to combine the two approaches into a single unified one. 



**Key Components**:
- **Curvature Computation**: Implementing Forman curvature for activation graphs
- **Manifold Analysis**: Studying geometric properties of activation spaces
- **Complexity Metrics**: Using curvature as a proxy for representational complexity

**Notable Files**:
- `unembedding_curvature.ipynb` - Curvature analysis of unembedding spaces
- `concept_curvature.ipynb` - Curvature analysis of concept representations

**Key Insights**:
- Activation spaces exhibit non-trivial geometric structure
- Curvature correlates with representational complexity
- Geometric properties influence model behavior


### Computing Activation Curvature
```python
from geomechinterp.curvature.balanced_forman_curvature import balanced_forman_curvature

# Compute curvature for activation graph
curvature_matrix = balanced_forman_curvature(adjacency_matrix)
```