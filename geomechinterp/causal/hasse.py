import random
from typing import Literal
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
def generate_truth_tables(N, exclude_non_causal=False):
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

    return filter_non_causal_truth_tables_binary(truth_tables)


def filter_non_causal_truth_tables_binary(truth_tables: np.ndarray) -> np.ndarray:
    # Filtering to keep only causal functions
    causal_truth_tables = []
    N = int(np.log2(truth_tables.shape[1]))  # Number of input variables

    for table in truth_tables:
        is_causal = True
        for var_idx in range(N):
            # For each variable, check if output changes when we flip just that bit
            # For a causal function, flipping any input bit should change the output
            # for at least one configuration of other inputs
            output_changed = False

            # Create masks for all possible configurations of other variables
            other_vars_configs = np.arange(2 ** (N - 1))
            for config in other_vars_configs:
                # Create two input configurations that differ only in var_idx
                input_0 = np.array([(config >> i) & 1 for i in range(N - 1)])
                input_1 = input_0.copy()

                # Insert 0 and 1 for var_idx
                full_input_0 = np.insert(input_0, var_idx, 0)
                full_input_1 = np.insert(input_1, var_idx, 1)

                # Convert binary arrays to indices
                idx_0 = int(sum([bit * (2**i) for i, bit in enumerate(full_input_0)]))
                idx_1 = int(sum([bit * (2**i) for i, bit in enumerate(full_input_1)]))

                # Check if output changes, handling masked arrays
                val_0 = (
                    table[idx_0].item()
                    if hasattr(table[idx_0], "item")
                    else table[idx_0]
                )
                val_1 = (
                    table[idx_1].item()
                    if hasattr(table[idx_1], "item")
                    else table[idx_1]
                )

                if val_0 != val_1:
                    output_changed = True
                    break

            if not output_changed:
                is_causal = False
                break

        if is_causal:
            causal_truth_tables.append(table)

    if len(causal_truth_tables) == 0:
        return np.array([])
    return np.stack(causal_truth_tables)


def filter_non_causal_truth_tables_block_symmetry(
    truth_tables: np.ndarray,
) -> np.ndarray:
    # Filtering to keep only causal functions
    causal_truth_tables = []

    for table in truth_tables:
        tt_syms = truth_table_block_symmetries(table, mode="all")

        # presence of any symmetry in tt is considered to be non-causal
        if not any(tt_syms):
            causal_truth_tables.append(table)

    return np.stack(causal_truth_tables)


def truth_table_block_symmetries(
    table: np.ndarray, mode: Literal["any", "all"] = "all"
) -> bool | list[int]:
    """For each variable, check symmetries at different scales

    mode 'all' considers total symmetry across all branches.
    for N=3 and mode "all":
    first sym : 1/2=2/2
    second sym: 1/4=2/4 AND 3/4=4/4
    third sym : 1/8=2/8 AND 3/8=4/8 AND 5/8=6/8 AND 7/8=8/8

    mode 'any' considers presence of any *conditional symmetry*,
    hence is much more permissive than 'all'.
    for N=3 and mode "any":
    first sym : 1/2=2/2
    second sym: 1/4=2/4 OR 3/4=4/4
    third sym : 1/8=2/8 OR 3/8=4/8 OR 5/8=6/8 OR 7/8=8/8
    """
    N = np.log2(len(table))
    if round(N) != N:
        raise ValueError("Size of Truth Table must be a power of 2!")
    N = int(N)
    block_symmetries = []

    if mode == "any":
        op = any
    else:
        op = all

    for var_idx in range(N):
        # Calculate number of blocks at this level
        num_blocks = 2**var_idx
        block_size = len(table) // num_blocks

        equal_blocks = []
        # Check each block
        for block in range(num_blocks):
            start = block * block_size
            mid = start + block_size // 2
            end = start + block_size

            # Compare first half with second half
            first_half = table[start:mid]
            second_half = table[mid:end]

            if np.array_equal(first_half, second_half):
                equal_blocks.append(1)

        if op(equal_blocks):
            block_symmetries.append(1)
        else:
            block_symmetries.append(0)

    return block_symmetries


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


def classify_tt_size_4(tt: list[int]):
    """
    Classifies a 2x2 truth table (binary string of length 4) into its corresponding logic gate.

    Parameters:
    - tt (str): A binary string of length 4 representing the truth table (e.g., '0001').

    Returns:
    - str: The name of the logic gate.
    """
    tts_classifier = {
        "0000": "constant_0",  # Always outputs 0
        "1111": "constant_1",  # Always outputs 1
        "0010": "A_AND_NOT_B",  # True when A=1 and B=0
        "0001": "AND",  # True only when all inputs are 1
        "0011": "A",  # Output equals first input
        "0100": "NOT_A_AND_B",  # True when A=0 and B=1
        "0101": "B",  # Output equals second input
        "0110": "XOR",  # True when inputs are different
        "0111": "OR",  # True when any input is 1
        "1000": "NOR",  # True when both inputs are 0
        "1001": "XNOR",  # True when inputs are same
        "1010": "NOT_B",  # Negation of second input
        "1011": "A_OR_NOT_B",  # True when A=1 or B=0
        "1100": "NOT_A",  # Negation of first input
        "1101": "NOT_A_OR_B",  # True when A=0 or B=1
        "1110": "NAND",  # False only when both inputs are 1
    }
    assert len(tt) == 4, "tt must be of length 4"
    assert all(s in [0, 1] for s in tt), "tt must be a binary string"
    tt_str = "".join([str(s) for s in tt])
    return tts_classifier.get(tt_str, "UNKNOWN")


def classify_truth_table(
    tt: list[int], full_class_for_size_4: bool = True
) -> tuple[list[str], int]:
    """
    Example approach:
    Extract symmetry-based truth-tables attributes
    such as it being balanced, or 1/2 symmetric, etc.
    Returns list of String Flags.
    """
    tt_flags = []
    # tt symmetries in size log2(len(tt)) e.g. [1,1,0]
    # first 1 means first half == second half
    # second 1 means 1/4 = 2/4 AND 3/4 = 4/4
    tt_symmetries = truth_table_block_symmetries(tt, mode="all")
    symmetry_complexity = len(tt) - np.sum(tt)  # number of missing symmetries

    # for size 4, we use full classification
    if len(tt) == 4 and full_class_for_size_4:
        return [classify_tt_size_4(tt)], symmetry_complexity

    # overwise we flag by each type of symmetry
    for i, sym in enumerate(tt_symmetries):
        if sym:
            tt_flags.append(f"tt_sym_{i}")

    # and add trivial cases
    length = len(tt)
    if all(x == 0 for x in tt):
        tt_flags.append("constant_0")
    if all(x == 1 for x in tt):
        tt_flags.append("constant_1")

    # half 0s, half 1s
    count_ones = sum(tt)
    if count_ones * 2 == length:
        tt_flags.append("balanced")

    return tt_flags, symmetry_complexity


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
