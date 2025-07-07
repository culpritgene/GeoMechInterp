#### Research Focus
Uncovering `Latent Knowledge` (applying beyong next token) from toy transformer activation stream, replicating the original [paper by Shai. et al.](https://arxiv.org/abs/2405.15943) on 3-State HMM generated data and observing that transformer linearly encodes a position in a belief state simplex between 3 actual states of the modeled HMM. This connects to `Causal Uncertainty` project directly, as this Mixed State Presentation encodes `Causal Uncertainty`  under a grokked weighted cyclic graphical model. 


### Key Components 

| Module / Notebook                   | Purpose                                                                                                                |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `Belief_State_Chaos_Modeling.ipynb` | Simulate sequential belief updates, estimate Bayes Gap ~ Lyapunov exponent, and visualise state‑space attractors                  |


### Key Insights 
* **Geometric belief updates** — Posterior updates can be cast as affine transformations in a latent "belief manifold" 
* **Mixed‑DAG structure matters** — The models grokks `true` posterior update with extremely tight `Bayesian Gap` between its update and ground truth update after observing next letter. 
