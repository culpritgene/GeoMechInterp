import torch
from transformers import PreTrainedTokenizer


def get_counterfactual_pairs_original(
    tokenizer: PreTrainedTokenizer,
    words_pairs: list[tuple[str, str]],
    take_first: bool = False,
):
    base_ind = []
    target_ind = []

    add_bos = True if not take_first else False

    for i in range(len(words_pairs)):
        first = tokenizer.encode(words_pairs[i][0], add_special_tokens=add_bos)
        second = tokenizer.encode(words_pairs[i][1], add_special_tokens=add_bos)
        # they take only two-token words with non-equal last tokens
        # and then select only the last token from the two-token word
        # actually the first token for original tokenizer is always <bos>! so they do take the first token!
        if take_first:
            base_ind.append(first[0])
            target_ind.append(second[0])
        else:
            if len(first) == len(second) == 2 and first[1] != second[1]:
                base_ind.append(first[1])
                target_ind.append(second[1])

    base_name = [tokenizer.decode(i) for i in base_ind]
    target_name = [tokenizer.decode(i) for i in target_ind]

    return base_ind, target_ind, base_name, target_name


def concept_direction(base_ind: int, target_ind: int, data: torch.Tensor):
    base_data = data[base_ind, :]
    target_data = data[target_ind, :]

    diff_data = target_data - base_data
    mean_diff_data = torch.mean(diff_data, dim=0)
    mean_diff_data = mean_diff_data / torch.norm(mean_diff_data)

    return mean_diff_data, diff_data


def inner_product_loo(base_ind: int, target_ind: int, data: torch.Tensor):
    # loo ~ least one out
    # compare one with the mean?
    # why? it should just have small projection onto the mean
    base_data = data[base_ind, :]
    target_data = data[target_ind, :]

    diff_data = target_data - base_data
    products = []
    for i in range(diff_data.shape[0]):
        mask = torch.ones(diff_data.shape[0], dtype=bool)
        mask[i] = False
        loo_diff = diff_data[mask]
        mean_diff_data = torch.mean(loo_diff, dim=0)
        loo_mean = mean_diff_data / torch.norm(mean_diff_data)
        products.append(loo_mean @ diff_data[i])
    return torch.stack(products), diff_data


def generate_concept_direction(
    unembed: torch.Tensor,
    rot_unembed: torch.Tensor,
    words_pairs: list[tuple[str, str]],
    tokenizer: PreTrainedTokenizer,
    take_first: bool = False,
):
    # head_token_idx, tail_token_idx, head_cat, tail_cat
    base_ind, target_ind, base_name, target_name = get_counterfactual_pairs_original(
        tokenizer, words_pairs, take_first=take_first
    )

    # computing concept directions by averaging tokens inside counterfactual pairs
    # in orig unembed and in rotated / whitened unembed
    mean_diff_gamma, diff_gamma = concept_direction(base_ind, target_ind, unembed)
    mean_diff_g, diff_g = concept_direction(base_ind, target_ind, rot_unembed)
    return mean_diff_gamma, mean_diff_g, base_ind, target_ind, base_name, target_name


def generate_concept_directions(
    unembed: torch.Tensor,
    rot_unembed: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    multi_words_pairs: dict[str, list[tuple[str, str]]],
    take_first: bool = False,
):
    concept_gamma = {}
    concept_g = {}
    causal_inner_prods = {}

    head_names = {}
    tail_names = {}

    for concept_tuple in multi_words_pairs:
        (
            mean_diff_gamma,
            mean_diff_g,
            base_ind,
            target_ind,
            base_name,
            target_name,
        ) = generate_concept_direction(
            unembed,
            rot_unembed,
            multi_words_pairs[concept_tuple],
            tokenizer=tokenizer,
            take_first=take_first,
        )

        concept_gamma[concept_tuple] = mean_diff_gamma
        concept_g[concept_tuple] = mean_diff_g

        head_names[concept_tuple] = base_name
        tail_names[concept_tuple] = target_name

        # compputin projections onto the PC within each concept group
        inner_product_LOO, diff_data = inner_product_loo(
            base_ind, target_ind, rot_unembed
        )
        causal_inner_prods[concept_tuple] = inner_product_LOO

    return concept_gamma, concept_g, causal_inner_prods, head_names, tail_names
