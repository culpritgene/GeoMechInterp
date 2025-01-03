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

    return filter_non_causal_truth_tables(input_combinations, truth_tables)


def filter_non_causal_truth_tables(
    input_combinations, truth_tables: np.ndarray
) -> np.ndarray:
    # Filtering to keep only causal functions
    causal_truth_tables = []
    N = truth_tables.shape[1]

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


def filter_truth_tables(
    truth_tables: np.ndarray, global_controls: list[int | None]
) -> np.ndarray:
    """
    Filter truth tables based on global controls.

    Args:
        truth_tables: Array of truth tables, shape (2^2^N, 2^N)
        global_controls: List of global control values (0, 1, or None)

    Returns:
        Filtered truth tables
    """
    # truth table is a 2^N x N matrix
    # but this is a stack of such tables, 2^2^N x 2^N in size
    # first column is split 0/1 in the middle
    # second column is split 0/1 on 1/4 and 3/4
    # third column is split 0/1 on 1/8, 3/8, 5/8, 7/8
    # and so on, we have FFT-style split
    # global controls are in 0|1|None
    # create indexing based on each global control
    # if global control is None, we keep all
    # we keep, depending on the index of the global control:
    # first half for 0-index global control equal 0
    # second half for 0-index global control equal 1
    # first half of each 1/2 for 1-index global control equal 0
    # second half of each 1/2 for 1-index global control equal 1
    # first half of each 1/4 for 2-index global control equal 0
    # second half of each 1/4 for 2-index global control equal 1
    # and so on, we do this for each global control
    n_controls = len(global_controls)
    n_rows, n_cols = truth_tables.shape

    # Verify input dimensions
    assert n_controls == int(
        np.log2(n_cols)
    ), "Number of global controls must match log2 of truth table columns"

    # Build mask for filtering
    mask = np.ones(n_cols, dtype=bool)
    for i, control in enumerate(global_controls):
        if control is not None:
            # Calculate pattern length for this control level
            repeat_count = n_cols // 2 ** (i + 1)

            # Create base pattern [0,1] repeated appropriately
            base_pattern = np.repeat([0, 1] * (2**i), repeats=repeat_count).astype(bool)

            # Flip pattern if control is 0 (since default pattern assumes control=1)
            if control == 0:
                base_pattern = ~base_pattern

            # Combine with overall mask
            mask = mask & base_pattern

    truncated_truth_tables = truth_tables[:, mask]
    # now we select indices of unique truncated rows
    unique_rows, unique_indices = np.unique(
        truncated_truth_tables, axis=0, return_index=True
    )

    selected_truth_tables = truth_tables[unique_indices]

    selected_truth_tables = np.ma.masked_array(
        selected_truth_tables,
        mask=np.repeat(~mask[np.newaxis, :], selected_truth_tables.shape[0], axis=0),
    )
    return selected_truth_tables


def test_filter_truth_tables():
    # Test case 1: Single control
    tt1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert np.array_equal(
        filter_truth_tables(tt1, [0, None]), np.array([[0, 0], [0, 1]])
    )
    assert np.array_equal(
        filter_truth_tables(tt1, [1, None]), np.array([[1, 0], [1, 1]])
    )

    # Test case 2: Two controls with None
    tt2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]])
    assert np.array_equal(
        filter_truth_tables(tt2, [0, None]), np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    )
    assert np.array_equal(
        filter_truth_tables(tt2, [None, 1]), np.array([[0, 1], [1, 1], [0, 1], [1, 1]])
    )


if __name__ == "__main__":
    test_filter_truth_tables()
