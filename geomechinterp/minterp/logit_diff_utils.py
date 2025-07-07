import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Union


def get_rank(tensor: torch.Tensor, value_index: int) -> int:
    sorted_indices = torch.argsort(tensor.squeeze(0), descending=True)
    # Find the rank of the value by locating its index in the sorted indices
    rank = (sorted_indices == value_index).nonzero(as_tuple=True)[0].item() + 1
    return rank


def compute_rank_difference(
    model: HookedTransformer,
    prompt: str,
    expected_answer: str,
    concept_vector: torch.Tensor,
    betas: list[float] = None,
    print_all: bool = True,
    sanity_check: bool = False,
    return_stats: bool = False,
) -> Union[list[int], dict]:
    if not betas:
        betas = np.linspace(0, 10, 5)

    prompt_toks = model.to_tokens(prompt, prepend_bos=True)
    answer_toks = model.to_tokens(expected_answer, prepend_bos=False)

    answer_toks = answer_toks[:, 0]  # take first token of the answer

    # Hook function to capture the output before right unembedding
    pre_logits = None

    def capture_activations(module, input, output):
        nonlocal pre_logits
        pre_logits = output

    # Register the hook on the final layer ('ln_final' for GPT-2 and Gemma)
    hook_handle = model.ln_final.register_forward_hook(capture_activations)

    # Generate logits
    logits = model(prompt_toks)[:, prompt_toks.shape[1] - 1, :]

    original_logit = logits[:, answer_toks].item()
    original_rank = get_rank(logits, answer_toks.item())

    if print_all:
        print("expected answer original logit:", original_logit)
        print("expected answer original rank:", original_rank)
        print("top-1 pred orig:", model.to_string(logits.argmax()))

    # we ran for many betas to avoid re-running model
    collect_ranks = []
    stats = []
    pre_logits_f = pre_logits[:, -1, :]
    # pre_logits_f = pre_logits_f @ sqrt_Cov_gamma
    for beta in betas:
        # Modify the logits with the concept vector
        trans_pre_logits_f = pre_logits_f + beta * concept_vector.unsqueeze(0)
        # Remove the hook after capturing the necessary activations
        hook_handle.remove()

        # Generate transformed logits
        # model.unembed(model.ln_final(trans_pre_logits_f))
        # model.unembed(model.ln_final.hook_normalized(trans_pre_logits_f))
        # g @ model.ln_final.hook_normalized(trans_pre_logits_f).T
        # trans_logits = g @ trans_pre_logits_f.T
        trans_logits = model.unembed(model.ln_final.hook_normalized(trans_pre_logits_f))

        # print(trans_logits.shape)
        trans_logits = trans_logits.squeeze()
        perturbed_logit = trans_logits[answer_toks].item()
        perturbed_rank = get_rank(trans_logits, answer_toks.item())
        rank_diff = original_rank - perturbed_rank
        collect_ranks.append(rank_diff)

        if return_stats:
            stats.append(
                {
                    "original_rank": original_rank,
                    "perturbed_rank": perturbed_rank,
                    "original_output": model.to_string(logits.argmax()),
                    "perturbed_output": model.to_string(trans_logits.argmax()),
                    "rank_diff": rank_diff,
                    "beta": beta,
                }
            )

    if len(betas) == 1 and print_all:
        print("expected answer perturbed logit:", perturbed_logit)
        print("expected answer perturbed rank:", perturbed_rank)
        print("top-1 pred perturb:", model.to_string(trans_logits.argmax()))

    if len(betas) == 1 and sanity_check:
        assert torch.allclose(
            trans_pre_logits_f.squeeze() - pre_logits_f.squeeze(),
            concept_vector.unsqueeze(0),
            rtol=0.001,
        )
        print(
            "Pre-Logits distance: ", (trans_pre_logits_f - pre_logits_f).norm().item()
        )
        print(
            "Perturbed Logits equal to original Logits: ",
            torch.allclose(logits, trans_logits, rtol=0.01),
        )
        print(
            "Perturbed best token equal to orig best tok: ",
            torch.allclose(logits.argmax(), trans_logits.argmax(), rtol=0.001),
        )

    if return_stats:
        return stats
    # Return the decrease in rank (where rank=1 is the largest logit) for each of beta
    return collect_ranks
