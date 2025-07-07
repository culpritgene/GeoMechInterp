# GeoMechInterp: Geometric Mechanistic Interpretability

A comprehensive research framework for investigating geometric aspects of Large Language Model (LLM) mechanistic interpretability, with a focus on causal patterns, uncertainty quantification, orthogonal concept interactions, and activation space curvature.

A repo accumulating various approaches to geometric, uncertainty-based and concept-dynamics based approaches to mechanistic interpretability. Besides some general utils it contains several mini-projects applying those utils to investigating toy-transformers uncertainty over causal patterns, curvature of tf activation space, analyzing orthogonal directions native to last pre-logit activation layer, replicating Belief-State-Geometry paper, and others.

## Table of Contents

- [Global Overview](#global-overview)
- [Python Module Structure](#python-module)
- [Sub-Projects](#sub-projects)
  - [Causal Patterns & Uncertainty](#causal-patterns--uncertainty)
  - [Orthogonal Concepts](#orthogonal-concepts)
  - [Activation Space Curvature](#activation-space-curvature)
  - [Belief State Geometry](#belief-state-geometry)
- [Installation & Setup](#installation--setup)

## lobal Overview

Most existing mechinterp frameworks rely on first-order term approximation over the residual stream space - here we want to test curvature-based and topology-based approaches to mechinterp, as well as uncertainty quantification (default token based, concept - token groups based and causal, via explicit graphical model given from data) that relate to curvature. Curvature and Uncertainty are deeply linked, with flat regions in activation space implying either high aleatoric uncertainty of the data (e.g. many synonyms or invariance to a subsapce of activations) or being generally uncertain about next steps. The project investigates how geometric properties of activation spaces, causal relationships between features, and concept interactions manifest in model behavior.


### Python Module

#### **Causal Analysis** (`causal/`)
- `mygpt.py` - Custom GPT model with symbolic tokenizer that is trained on generated causal patterns
- `pattern_generator.py` - Generates exhaustive causal patterns iterating over all possible DAGs and Truth Tables connecting input binary features with output binary feature
- `hasse.py` - Implements Hasse diagrams for causal structure representation
- `belief_update.py` - Belief state update mechanisms and geometric modeling
- `utils.py` - Causal analysis utilities and helper functions
- `base_functions.py` - Base feature functions and symbolic operations

#### **Curvature Analysis** (`curvature/`)
- `balanced_forman_curvature.py` - Balanced Forman curvature computation for graphs
- `balanced_forma_curvature_jax.py` - JAX implementation of Forman curvature
- `forman_curvature.pyx` - Cython implementation for performance-critical curvature calculations

#### **Visualization** (`viz/`)
- `attn_viz.py` - Attention visualization tools and heatmaps
- `streamlit_viz.py` - Interactive Streamlit-based visualization for attention with ability to select subspaces of activations and color / cluster by different categorial features of input data patterns
- `plots.py` - General plotting utilities and uncertainty visualization

#### **Transformer Lens based Mechinterp** (`minterp/`)
- `activations.py` - Activation extraction and manipulation utilities
- `utils.py` - Transformer Lens specific utilities
- `logit_diff_utils.py` - Logit difference computation for concept analysis
- `steering.py` - Concept steering and counterfactual analysis
- `counterfact_pairs.py` - Counterfactual pair generation and analysis

#### **Information Theory** (`informat/`)
- `entropy.py` - Entropy estimation for character and token sequences
- `compression.py` - Compression complexity analysis

#### **Graph Analysis** (`graph/`)
- Graph utilities for causal structure analysis

#### Core Utilities
- `utils.py` - General utilities for model evaluation and pattern analysis


## Sub-Projects

Some minimal standalone experiments can be found in `preliminary_experiments`, for instance 
- `preliminary_experiments/token_superposition.ipynb` that tests if we can decode answer to $2$ mixed queries from a single output (not really)
- `preliminary_experiments/uroborosus.ipynb` that tracks convergence wrt logit diff after re-feeding some FFN parameter slice back into TF
- `preliminary_experiments/uncertaity_quantification.ipynb` that tests very simple uncertainty quantification approaches for MDPs

Each `Project` listed above has its own `README.md` file with more details - navigate into subfolder to see it. 
### Causal Patterns & Uncertainty (`projects/causal_patterns/`)

**Location**: `projects/causal_patterns/`

**Research Focus**: Investigating how language models learn and represent causal relationships between symbolic features, with particular emphasis on quantifying uncertainty in these causal patterns.

**Key Components**:
- **Pattern Generation**: Exhaustive generation of causal patterns from binary feature combinations
- **Uncertainty Quantification**: Measuring model uncertainty in causal relationships
- **Symbolic Analysis**: Using custom symbolic tokenizer for precise feature control

**Notable Files**:
- `mygpt_analysis.ipynb` - Comprehensive analysis of causal patterns in custom GPT models
- `generate_causal_patterns.ipynb` - Pattern generation and dataset creation
- `quantify_causal_uncertainty_gpt2.ipynb` - Uncertainty quantification in GPT-2

**Key Insights**:
- Models exhibit systematic uncertainty in causal relationships
- Causal patterns can be systematically generated and analyzed
- Uncertainty varies predictably with pattern complexity
- Structured uncertainty correlates with feature-interaction order
- Disentangled subspaces allow clean causal interventions


### Orthogonal Concepts (`projects/orthogonal_concepts/`)

**Location**: `projects/orthogonal_concepts/`

**Research Focus**: Understanding how independent concepts interact in embedding spaces, particularly focusing on the geometric properties of concept directions and their interactions. This work follows ["The Linear Representation Hypothesis and the Geometry of Large Language Models"](https://arxiv.org/abs/2311.03658) and ["The Geometry of Categorical and Hierarchical Concepts in Large Language Models"](https://arxiv.org/abs/2406.01506).

**Key Insights from Research**:
- Orthogonal concepts can be reliably extracted from embedding spaces using counterfactual pairs
- Concept interactions follow predictable geometric patterns during inference
- Steering vectors enable controlled concept manipulation in pre-logit activation spaces
- Geometric transformations in embedding space correspond to semantic concept operations
- See [detailed analysis report](https://docs.google.com/document/d/1jm-pqVV4TyfpBbCK3hUqmbAiapFpe0a6P2lGRSXqkGw/edit?usp=sharing) for comprehensive findings


### Activation Space Curvature (`projects/activations_curvature/`)

**Location**: `projects/activations_curvature/`

**Research Focus**: Analyzing Curvature in the residual stream of toy transformers or GPT2, using methods from classical diff geometry, manifold learning and hessian analysis; as well as anylizing curvature of the attention space using discrerte Olivier Curvature suitable to study discretized attention graphs. The future idea is to combine the two approaches into a single unified one. 


### Belief State Geometry (`projects/blief_state_geometry/`)

**Location**: `projects/blief_state_geometry/`

**Research Focus**: Uncovering `Latent Knowledge` (applying beyong next token) from toy transformer activation stream, replicating the original [paper by Shai. et al.](https://arxiv.org/abs/2405.15943) on 3-State HMM generated data and observing that transformer linearly encodes a position in a belief state simplex between 3 actual states of the modeled HMM. This connects to `Causal Uncertainty` project directly, as this Mixed State Presentation encodes `Causal Uncertainty`  under a grokked weighted cyclic graphical model. 

### Installation 

**requirements**
- Python 3.8+
- PyTorch
- Transformer Lens
- NetworkX
- NumPy, Pandas, Matplotlib
- Optional JAX (for curvature computations)

### Installation
```bash
# Clone the repository
git clone git@github.com:culpritgene/GeoMechInterp.git
cd GeoMechInterp
pip install -r requirements.txt
pip install -e .
```

**Note**: At the moment this is not a well structured or well document project, but rather a collection of different experiments and utils for them that can be composed together for further projects. 
For example analysing belief state dynamics of models trained on multiple abstract causal graphs with partially dijoint vocabularies.

