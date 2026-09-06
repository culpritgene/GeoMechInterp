This repository contains the code and Jupyter notebooks used to replicate and extend the linear–geometry analysis of large‑language‑model representations presented in the reports below.

As such we largely follow code for [“The Linear Representation Hypothesis and the Geometry of Large Language Models”](https://arxiv.org/abs/2311.03658)
And a bit subsequent [“The Geometry of Categorical and Hierarchical Concepts in Large Language Models”](https://arxiv.org/abs/2406.01506)


**Key Components**:
- **Concept Direction Extraction**: Computing orthogonal concept vectors from counterfactual pairs
- **Interaction Analysis**: Studying how orthogonal concepts interact during inference
- **Geometric Transformations**: Analyzing concept steering as geometric operations


- `prelogit_orthogonal_concepts_I.ipynb` - Initial analysis of orthogonal concept interactions, on a bunch of binary, ordinal and categorical orthogonal features such as `Eng/French`, `Male/Female`, `Large/Small`, `Slow/Fast`, `Made of Material`, 
- `interactions_of_orthogonal_concepts_II.ipynb` - Largely follows baseline Transformer-Lens demo for IOU circuit, but applied to the task of choosing right word pair completion in a sequence of word pairs oscillating according to combination of some *binary* features


### Key Insights 

See [Google Doc report](https://docs.google.com/document/d/1jm-pqVV4TyfpBbCK3hUqmbAiapFpe0a6P2lGRSXqkGw/edit?usp=sharing) for more

**Reproduced highlights**:
-  Whitening → causal inner product – whitening the unembedding layer collapses spurious correlations; the inner product in the whitened basis isolates additive concept directions 
- Binary counter‑factuals – language, gender and verb‑tense directions achieve > 95 % linear separability on Gemma‑2B; tokenisation differences explain the residual error.

**Additional highlights**:
- Categorical attributes generalise – single vectors for material or colour improve token‑rank of unseen object–attribute pairs by one to two orders of magnitude.
- Ordinal size is brittle – varying the size vector samples only a few of anchor words with high frequency and quickly leaves the data manifold 
- It seems that rare words appear conditionally on some topic-specific / jargon vocabulary cues - this indicated `Curvature` in the Concept direciton, with `Frequent` / `Public` words occupying the linear subspace and rare words diverging from it. 
- Concept composition – linear combinations steer several attributes simultaneously, but obscure the exact attribute value and expose head‑specific attention circuits in mid/late layers


## Directory layout
| Module / Notebook                   | Purpose                                                                                                                |

| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prelogit_orthogonal_concepts_I.ipynb`         | Whitening the unembedding matrix, extracting counter‑factual concept vectors and testing basic steering on **Gemma‑2B** model.                                 |
| `interactions_of_orthogonal_concepts_II.ipynb` | Composing and chaining multiple orthogonal concepts, probing non‑binary categorical attributes (colour, material, size) and preliminary intervention analysis. |


### Minimal Code Example
```python
from geomechinterp.counterfact_pairs import generate_concept_directions
from geomechinterp.steering import compare_concept_steering

# Prepare model / tokenizer / unembedding matrix in the notebook first
concept_dirs = generate_concept_directions(
    unembed=unembedding_matrix,
    tokenizer=tokenizer,
    multi_words_pairs=word_pairs,
)

results = compare_concept_steering(
    model=model,
    train_pairs=train_pairs,
    test_pairs=test_pairs,
    concept_vector=concept_dirs["size"],  # any extracted concept
    betas=[0.0, 0.25, 0.5, 0.75, 1.0],
)
```

#### Extending this work
1.	Layer‑wise tracing – integrate with TransformerLens to project concept activations deeper than the final layer norm.
2.	Sparse auto‑encoders – re‑express concept subspaces via SAEs to study superposition and curvature.
3.	KG extraction – experiment with TransE/RotatE style losses on the extracted concept pairs to build explicit knowledge graphs.
4.	Hierarchical simplices – adapt the 2024 paper’s simplicial‑localisation objective to pre‑logit vectors and compare with inductive graph embeddings.

# Concept Geometry Experiments

This repository contains the code and Jupyter notebooks used to replicate and extend the linear–geometry analysis of large‑language‑model representations presented in the reports below.  All figures in **Playing with Orthogonal Concepts in LLM Pre‑logits** were generated directly from these notebooks.

* **Primary inspiration** – *The Linear Representation Hypothesis and the Geometry of Large Language Models* (Hsu et al., 2023)
* **Follow‑up reference** – *The Geometry of Categorical and Hierarchical Concepts in Large Language Models* (Hsu et al., 2024)

