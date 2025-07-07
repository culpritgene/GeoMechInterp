
#### Research Focus
Investigating how language models learn and represent causal relationships between symbolic binary features, and how uncertainty emerges from those causal structures.

### Key components

| Module / Notebook                        | Purpose                                                                                                 |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `generate_causal_patterns.ipynb`         | Exhaustively enumerate binary‑feature combinations and construct datasets with controlled causal graphs |
| `mygpt_analysis.ipynb`                   | Train **ToyGPT** on the synthetic corpus and visualise causal attribution across layers                 |
| `quantify_causal_uncertainty_gpt2.ipynb` | Quantify predictive entropy and mutual information of a pretrained GPT‑2 on the same patterns           |

### Dataset outline

* \~1 M unique causal sequences
* 8 candidate binary features; ≤ 5 active per sequence; ≤ ternary interactions
* Mix of positional parity and non‑positional token features (identity, case, ± sign)
* Stochastic variants enable controlled signal‑to‑noise studies

Importantly, the dataset consists not just of string patterns, but each pattern has a corresponding data generation model in a `json-serializable` format. You can load this into a generator function using the `DisplayChain.from_json` method from `geomechinterp.causal.utils`:

### Notebook flow

1. **Pattern generation** → `generate_causal_patterns.ipynb`
2. **Model training & causal probe** → `mygpt_analysis.ipynb`
3. **Uncertainty quantification in GPT** → `quantify_causal_uncertainty_gpt2.ipynb`
4. **Uncertainty quantification in ToyGPT** → `quantify_causal_uncertainty_mygpt.ipynb`

These reproduces all figures in the *Causal Patterns* [report draft](https://www.overleaf.com/project/686bc307a28e72a48e8d351c)


#### Generate a Dataset of Causal Patterns with different Causal Structures
1. Fundamentally, we work with Sequential models, 1D is our space and we have to predict next token 
2. Thus Features that we inherit "naturally" from the space itself are: 
    - Absolute Position
    - Module Divisibility (odd / even, each 3rd, each 5th, etc)
    - Relative Position (contextual on anchored token / set of tokens)
    - + **non-causal** global statistics, function of `len(seq)`, e.g. odd/even number of tokens
3. Here the only Positional Feature that we use is `Parity` (odd / even)
    - Though we can ignore Positional Features all together as well!
4. Non-Positional Features can be encoded by 
    - Compositions **of N** Tokens (must be `N > 1`)
    - Compositional Structre ***within*** Vocabulary 
5. Below are examples of **Non-Positional** Causal Sequences 
    - Non-Positional means that they should be treated as `Sets`



| Non-Positional Causal Patterns        | Comment                                        |
|----------------|------------------------------------------------|
| `a a a a a ...`| constant                         |
| `a b a b b ...`| **random** - Track model `Uncertaiinty`!       |
| `+a +a -b +a ...`| Two Token, `+:a` and `-:b`, `+/-` are **random** and causally control `a/b` !      |
| `-a +A +A -A ...`| `+:a` and `-:A`, `+/-` are **random** and causally control `a/A`      |
| `-a -b +A +B ...`| `+:lower` and `-:upper`, `+/-`  causally control `lower/upper`, `+/-` **and** `a/b` are **random**         |
| `-a -b +a +B ...`| `+:lower\|a` and `-:upper\|b`, `+/-`  and `a/b` are **random**, `+/-` causally control `lower/upper` only for `b` !!|
| `a- A+ b- B+ ...`| `lower:-` and `upper:+`, `a/b` and `lower/upper` are random, `lower/upper` controls `+/-`|
| `a- A- b- B+ ...`| only `B` leads to `+`, thus `a,A,b` can be considered as *joint vocabulary*, `a/b` and `lower/upper` merge!!|
| `-ab +AB +AB -ab ...`| Three Token, `-:lower` and `+:upper`, `a`  first `b` second       |
| `-ab +Ab +Ab -ab ...`| `+/-` controls `a/A` and `b` is constant (= independent)         |
| `-ab +AB +AB -ab ...`| Three Token, but `ab` / `AB` can be considered as a single one! `+/-` causal!        |
| `-aa +AA +BA -ba ...`|  `-:lower` and `+:upper`, `ll` both position and ***identity*** are **arbitrary**    |
| `+Ab -aB -aB +Ab ...`|  `+:upper\|first` and `-:upper\|second`, `ll` is always `ab`  |
| `-bA +Ab +Bb -bB ...`|  `+:upper\|first` and `-:upper\|second`, `ll` are arbitrary, but `lower:upper` of the third token is controled by `+/-`   |


#### Train ToyGPT model on a dataset of causal patterns 

Overall we have about ~1 Million of distinct causal patterns in the dataset, with `8` unique binary features in total, but only up to `5` features at once and only up to `ternary` direct interaction between features to avoid blow up from generating all possible Truth Tables (grow hyper exponentiall with the number of features so even `5` is already too intense)

Model achieves ~ `1.13` average CE loss on test data, quite good given that most of the patters are `stochastic`. 


#### Analysis of Uncertainty of the Trained Model

Includes:
- qunatifying uncertainty over the sequence from a given ground truth pattern model
- 

### Core insights

* **Structured uncertainty** — model entropy rises predictably with feature‑interaction order and stochastic control tokens.
* **Disentangled subspaces** — certain feature pairs (e.g. *case × sign*) form nearly orthogonal linear directions, allowing clean causal interventions.

- **DAG predictability** — the uncertainty landscape correlates with simple graph metrics of the underlying causal pattern DAG (depth, branching factor), but Sub-Vocabulary instansiation (e.g. `+/-` vs `1/0`) often acts as a stronger linear feature separating activations
- Uncertainty varies predictably with pattern compression based complexity
- There is a point in most sequences where both GPT2 and MyGPT models converge to stable uncertainty per next token, but before that there is a non-linear region where uncertainty oscillates for tokens of same "position" in the overall causal sequence - this may mean configuration of an implicit "causal graph" in model's activations which will later bias distribution over next tokens. 



### Minimal Run with simplest Uncertainty Prediction model
**Minimal Code Example**:
```python
from geomechinterp.causal.pattern_generator import get_exhaustive_pattern_generators, generate_patterns_mp
from geomechinterp.causal.mygpt import SymbolTokenizer, train_model
from geomechinterp.informat.entropy import estimate_entropy_token
import torch

# 1. Generate causal patterns
patterns = get_exhaustive_pattern_generators(
    selected_features=['position_parity', 'ab', 'case', '+-'],
    max_controls=2
)
pattern_strings = generate_patterns_mp(patterns, pattern_length=20)

# 2. Train/evaluate model on patterns
tokenizer = SymbolTokenizer()
model = train_model(pattern_strings, tokenizer, epochs=10)

# 3. Compare uncertainty between optimal and model
def compare_uncertainty(model, pattern, tokenizer):
    # Get model predictions
    logits = model.run_with_cache(model.to_tokens(pattern))[1]['logits']
    model_entropy = estimate_entropy_token(logits.softmax(-1))
    
    # Get optimal (ground truth) uncertainty
    optimal_entropy = estimate_entropy_token(pattern)  # Based on causal structure
    
    return model_entropy, optimal_entropy

# Example usage
model_uncertainty, optimal_uncertainty = compare_uncertainty(model, "a b a b a b", tokenizer)
print(f"Model uncertainty: {model_uncertainty:.3f}")
print(f"Optimal uncertainty: {optimal_uncertainty:.3f}")
```