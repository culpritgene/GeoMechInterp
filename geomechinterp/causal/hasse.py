import random
import numpy as np
from itertools import product
from functools import lru_cache


def generate_binary_masks(n: int) -> np.ndarray:
    return np.array(list(product([0, 1], repeat=n)))


def get_all_possible_hasse_diagrams(n: int, first_n_non_active: int = 0) -> np.array:
    """Note that masks are `abstract` causal patterns,
    they still need to be "initialized" by a particular permutation of attributes."""
    assert n < 10, "lets not blow up for now"
    assert first_n_non_active < n, "first_n_non_causal must be smaller than n"
    # size of the causal part of adjacency matrix
    # 0 + 1 + 2 + ... + (n-1)
    s = (n**2 - n) // 2

    # get all possible binary masks
    masks = generate_binary_masks(s)
    if first_n_non_active:
        s_non_causal = (first_n_non_active**2 - first_n_non_active) // 2
        masks = masks[:, s_non_causal:]
        masks = np.unique(masks, axis=0)
    return masks


def unflatten_to_causal_triangle_simple(flat_vector: np.ndarray) -> list[tuple[int]]:
    """Example usage
    unflatten_to_causal_triangle([1, 0, 1, 0, 1, 0])
    >>> [(1,), (0, 1), (0, 1, 0)]
    """
    # infer n
    s = len(flat_vector)
    n = (-1 + np.sqrt(1 + 4 * 2 * s)) / 2
    assert (
        round(n) == n
    ), "flat vector must come from causal triangle (3,6,10,15,21,...)"
    causal_triangle = []
    index = 0
    for i in range(int(n) + 1):
        causal_triangle.append(tuple([flat_vector[index + j] for j in range(i)]))
        index += i
    return causal_triangle


def unflatten_to_causal_triangle_skip_first_n_non_active(
    flat_vector: np.ndarray, first_n_non_active: int = 0
) -> list[tuple[int]]:
    """Example usage
    unflatten_to_causal_triangle_skip_first_n_non_active([1, 0, 1, 0, 1, 0], first_n_non_active=1)
    >>> [(1,), (0, 1), (0, 1, 0)]
    """
    # infer n
    s = len(flat_vector)
    s2 = s + first_n_non_active * (first_n_non_active - 1) // 2
    n = (-1 + np.sqrt(1 + 4 * 2 * s2)) / 2
    assert (
        round(n) == n
    ), "flat vector must come from a [truncated] causal triangle (...,3,6,10,15,21,...)"
    causal_triangle = []
    index = 0
    for i in range(first_n_non_active, int(n) + 1):
        causal_triangle.append(tuple([flat_vector[index + j] for j in range(i)]))
        index += i
    return causal_triangle


def unflatten_to_causal_triangle(
    flat_vector: np.ndarray, nonactive_num: int = 0
) -> list[tuple[int]]:
    """Unflatten a flat causal vector to a causal triangle.
    If nonactive_num is greater than 0, the first n non-active attributes are skipped.
    flat causal vector: [1,0,1,0,1,1] -> causal triangle: [(1,), (0,1), (0,1,1)]
    meaning of [(1,), (0,1), (0,1,1)]:
    - first feature is free (it is not shown in the triangle as it is always free)
    - second is controlled by first
    - third is *directly* controlled by second
    - fourth is *directly* controlled by second and third

    Note that *indirectly* third feature is *also* controlled by first.
    """
    if nonactive_num:
        return unflatten_to_causal_triangle_skip_first_n_non_active(
            flat_vector, nonactive_num
        )
    return unflatten_to_causal_triangle_simple(flat_vector)


def get_all_possible_causal_patterns_abstract(n: int):
    causal_masks = get_all_possible_hasse_diagrams(n)
    all_causal_edge_lists = []
    for hasse_flat in causal_masks:
        hasse_triangle = tuple(unflatten_to_causal_triangle(hasse_flat))
        all_causal_edge_lists.append(hasse_triangle)
    return all_causal_edge_lists


def sample_hasse_diagram_binary(
    reflections: list[str | int],
) -> list[tuple[int, list[int]]]:
    """Sample a hasse diagram from a list of reflections.
    Reflections are binary functions that are used to generate the hasse diagram.
    """
    idx = list(range(len(reflections)))

    causal_edge_list = []
    # causal sampling
    # example for [0,1,2]
    # (1, [])
    # (0, [1])
    # (2, [1,0])
    fixed = []
    remaining = idx
    for _ in range(len(idx)):
        # fix attribute
        att_idx = random.sample(remaining, 1)[0]
        # select control attributes from already fixed
        num_controls = np.random.choice(len(fixed) + 1) if len(fixed) else 0
        controls = random.sample(fixed, num_controls)

        fixed.append(att_idx)
        remaining = [idx for idx in remaining if idx != att_idx]
        causal_edge_list.append((att_idx, controls))

    return causal_edge_list


@lru_cache(maxsize=5)  # Cache results for N=1 to 12 (though 12 will blow up memory)
def generate_truth_tables(N, exclude_non_causal=True):
    assert N < 5, "lets not blow up for now"
    num_combinations = 2**N
    num_functions = 2**num_combinations
    input_combinations = list(product([0, 1], repeat=N))
    input_combinations = np.array(input_combinations, dtype=np.int8)

    # Generate all possible truth tables
    truth_tables = np.zeros((num_functions, num_combinations), dtype=np.int8)

    for i in range(num_functions):
        binary_string = f"{i:0{num_combinations}b}"
        truth_tables[i] = np.array([int(bit) for bit in binary_string], dtype=np.int8)

    if not exclude_non_causal:
        return truth_tables

    # Filtering to keep only causal functions
    causal_truth_tables = []

    for table in truth_tables:
        is_causal = True
        for var_idx in range(N):
            input_combinations_flipped = input_combinations.copy()
            input_combinations_flipped[:, var_idx] = (
                1 - input_combinations_flipped[:, var_idx]
            )
            output_changed = False
            for i, original_input in enumerate(input_combinations):
                flipped_index = np.where(
                    (input_combinations == input_combinations_flipped[i]).all(axis=1)
                )[0][0]
                if table[i] != table[flipped_index]:
                    output_changed = True
                    break
            if not output_changed:
                is_causal = False
                break

        if is_causal:
            causal_truth_tables.append(table)

    return np.stack(causal_truth_tables)


def apply_truth_table(truth_table: np.ndarray, inputs: np.ndarray) -> int:
    """
    Apply a truth table to a set of inputs.

    Parameters:
    - truth_table: A tensor representing the truth table.
    - inputs: A tensor of shape (M, N) where M is the number of input sets and N is the number of inputs.

    Returns:
    - outputs: A tensor of shape (M,) containing the outputs for each input set.
    """
    # Determine the index of each input in the lexicographical order
    indices = np.sum(
        inputs * (2 ** np.arange(inputs.shape[1] - 1, -1, -1)), axis=1
    ).astype(int)
    # Use the indices to get the output from the truth table
    return truth_table[indices]
