import torch
from tqdm import tqdm
import pandas as pd
from geomechinterp.logit_diff_utils import compute_rank_difference


def generate_prompt_binary(
    counterfactual_pair: tuple[str, str], connector: str = " ="
) -> tuple[str, str]:
    prompt = counterfactual_pair[0] + connector
    answer = counterfactual_pair[1]
    return prompt, answer


def compare_concept_steering(
    model,
    train_word_pairs: list[tuple[str, str]],
    test_word_pairs: list[tuple[str, str]],
    cocept_vector: torch.Tensor,
    betas: list[float],
    connector: str = " =",
) -> pd.DataFrame:
    rank_decreases_train = []
    rank_decreases_test = []
    # run on "train data" - on which original concept vector is computed
    for word_pair in tqdm(train_word_pairs):
        prompt, answer = generate_prompt_binary(word_pair, connector=connector)
        rank_decrease = compute_rank_difference(
            model, prompt, answer, cocept_vector, betas=betas, print_all=False
        )
        rank_decreases_train.append(rank_decrease)

    # run on "test data" - counterfactual pairs not included in `concept_w` calc
    for word_pair in tqdm(test_word_pairs):
        prompt, answer = generate_prompt_binary(word_pair, connector=connector)
        rank_decrease = compute_rank_difference(
            model, prompt, answer, cocept_vector, betas=betas, print_all=False
        )
        rank_decreases_test.append(rank_decrease)

    df_train = pd.DataFrame(
        torch.Tensor(rank_decreases_train).cpu(),
        columns=betas,
        index=["_".join(w) for w in train_word_pairs],
    )
    df_test = pd.DataFrame(
        torch.Tensor(rank_decreases_test).cpu(),
        columns=betas,
        index=["_".join(w) for w in test_word_pairs],
    )

    df_train["train"] = True
    df_test["train"] = False
    dff = pd.concat([df_train, df_test], axis=0)
    dff["num_toks_answer"] = [
        model.to_tokens(w.split("_"), prepend_bos=False).shape[1] for w in dff.index
    ]
    return dff
