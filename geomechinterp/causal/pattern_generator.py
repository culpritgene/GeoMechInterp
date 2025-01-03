from typing import Callable, List
import numpy as np
import logging
from itertools import permutations, product
from tqdm import tqdm

from geomechinterp.causal.hasse import (
    get_all_possible_hasse_diagrams,
    unflatten_to_causal_triangle,
    filter_truth_tables,
)
from geomechinterp.causal.utils import (
    check_causal_validity,
    sort_prev_first,
    IndependentFeatureWrapper,
    DependentFeatureWrapper,
    DisplayChain,
    NON_ACTIVE_FEATURE_SUFFIX_RE,
    POSITION_PARITY_FEATURE,
)

from geomechinterp.causal.base_functions import (
    all_binary_generators,
    all_binary_features,
)
from multiprocessing import Pool
from .truth_table_cache import truth_table_cache

logging.basicConfig(level=logging.INFO)


def get_all_possible_causal_patterns_labeled(
    reflections: list[str | int], max_controls: int = 3, verbose=False
):
    assert (
        max_controls < 5
    ), "with 5 controls we get 2**(2**5) = 4294967296 possible patterns"
    non_active_features, active_features = sort_prev_first(reflections)

    if len(active_features) > max_controls:
        logging.warning(
            f"Active features {active_features} exceed max_controls {max_controls}. "
            f"We will discard all causal patterns with more than {max_controls} active features."
        )
        active_features = active_features[:max_controls]

    reflections = non_active_features + active_features
    n = len(reflections)
    num_non_active_features = len(non_active_features)

    # TODO: Perhaps truncating Hasse diagrams is not needed, as we can just
    # filter out the ones that are not valid
    causal_masks = get_all_possible_hasse_diagrams(
        n, first_n_non_active=num_non_active_features
    )

    all_permutations_part = [list(perm) for perm in permutations(active_features)]
    all_permutations = [
        tuple(non_active_features + perm) for perm in all_permutations_part
    ]

    # final causal patterns are direct product of the two
    all_causal_edge_lists = {}
    total_count = len(causal_masks) * len(all_permutations)
    for hasse_flat in causal_masks:
        hasse_triangle = tuple(
            unflatten_to_causal_triangle(
                hasse_flat, nonactive_num=num_non_active_features
            )
        )
        all_causal_edge_lists[hasse_triangle] = {}
        for perm_str in all_permutations:
            if not check_causal_validity(perm_str):
                continue
            perm_specific_causal_graph = []
            for i in range(len(perm_str)):
                if i < num_non_active_features:
                    continue
                control_attrs = tuple(
                    sorted(
                        [
                            perm_str[j]
                            for j, mask in enumerate(
                                hasse_triangle[i - num_non_active_features]
                            )
                            if mask
                        ]
                    )
                )
                if len(control_attrs) > max_controls:
                    continue
                fixed_and_controls = (perm_str[i], control_attrs)
                perm_specific_causal_graph.append(fixed_and_controls)
            # again, sorted needed for subsequent deduplication
            # ('a', 'b', 'c): (('a', ()), ('b', ('a',), ('c', ('a','b')))
            all_causal_edge_lists[hasse_triangle][perm_str] = tuple(
                sorted(perm_specific_causal_graph)
            )
    if verbose:
        logging.info("all_causal_edge_lists:", all_causal_edge_lists)
    # deduplicated, flattened version
    deduplicated_dict = {}
    for hasse_triangle in all_causal_edge_lists:
        for perm, perm_specific_causal_graph in all_causal_edge_lists[
            hasse_triangle
        ].items():
            # perm allows to avoid reconstruction of causal sorting
            # (('a', ()), ('b', ('a',), ('c', ('a','b'))): ('a', 'b', 'c')
            if perm_specific_causal_graph:
                deduplicated_dict[perm_specific_causal_graph] = perm

    if verbose:
        logging.info(
            f"Total Count: {total_count}, Deduplicated Count: {len(deduplicated_dict)}"
        )

    deduplicated_by_perm = {}
    for perm_str1 in all_permutations:
        deduplicated_by_perm[perm_str1] = []
        for perm_specific_causal_graph, perm_str2 in deduplicated_dict.items():
            # we sort causal graph according to permutation, just for simplicity
            sorted_perm_sp_causal_graph = []
            if perm_str2 == perm_str1:
                for refl in perm_str1[len(non_active_features) :]:
                    matching_vals = [
                        val for val in perm_specific_causal_graph if val[0] == refl
                    ]
                    if matching_vals:  # Only append if we found a match
                        sorted_perm_sp_causal_graph.append(matching_vals[0])
            if sorted_perm_sp_causal_graph:
                deduplicated_by_perm[perm_str1].append(
                    tuple(sorted_perm_sp_causal_graph)
                )
    deduplicated_by_perm = {k: v for k, v in deduplicated_by_perm.items() if v}

    return all_causal_edge_lists, deduplicated_by_perm


def realize_causal_pattern(
    pattern: list[tuple[str, list[str]]],
    use_stochastic_features: bool = False,
    all_binary_generators: dict[str, Callable] = all_binary_generators,
) -> list[DisplayChain]:
    """
    Takes a causal pattern description and returns all possible realizations of that pattern
    as functions that generate sequences according to the causal dependencies.

    Args:
        pattern: List of tuples where each tuple contains (feature, controlling_features)
        use_stochastic_features: Whether to use stochastic features, with 50% probability of being 0 or 1
        all_binary_features: Dict mapping feature names to their generating functions

    Returns:
        List of functions that each generate a valid sequence following the causal pattern
    """
    # Get all possible truth tables for each feature based on number of controlling features
    independent_pattern_functions: list[list[Callable]] = []
    independent_global_controls: list[list[int]] = []
    indep_pattern_names: list[str] = []
    pattern_functions: list[list[Callable]] = []

    independent_controls = [None, 0, 1] if use_stochastic_features else [0, 1]
    # For each feature in the pattern
    # example pattern = [('position_parity', ()), ('ab', ()), ('case', ('ab', 'position_parity')))]
    # first tuple argument is the "active feature"
    # second tuple argument contains "control" features
    # loop below inserts particular realizations of control
    # since all features are binary, we just iterate *over all possible truth tables*
    for feature, controls in pattern:
        function_realizations: List[Callable[[str, str, int], str]] = []

        # Get base function for this feature
        _check_feature_func(feature, all_binary_generators)
        base_fn = all_binary_generators[NON_ACTIVE_FEATURE_SUFFIX_RE.sub("", feature)]

        # If no controls, just use base function
        if not controls:
            if feature == POSITION_PARITY_FEATURE:
                # position_parity is ignored - it is switched on generation anyway
                continue
            else:
                # overwise add base function
                for global_control in independent_controls:
                    function_realizations.append(
                        IndependentFeatureWrapper(base_fn, global_control)
                    )
            indep_pattern_names.append(feature)
            independent_pattern_functions.append(function_realizations)
            independent_global_controls.append(independent_controls)

    # generate all possible combinations of independent pattern functions
    all_indep_combinations = list(product(*independent_pattern_functions))
    all_indep_global_controls = list(product(*independent_global_controls))

    for indep_combination, indep_global_control in zip(
        all_indep_combinations, all_indep_global_controls
    ):
        dependent_pattern_func_cont = []
        for feature, controls in pattern:
            function_realizations: List[Callable[[str, str, int], str]] = []

            # Get base function for this feature
            _check_feature_func(feature, all_binary_generators)
            base_fn = all_binary_generators[
                NON_ACTIVE_FEATURE_SUFFIX_RE.sub("", feature)
            ]
            if controls:
                # If function *is* dependent on other features
                # Get all possible truth tables for this number of controls
                n_controls = len(controls)
                truth_tables = truth_table_cache.get_table(n_controls)

                # Sort controls according to indep_combination
                control_sequence = [
                    (
                        indep_global_control[indep_pattern_names.index(control)]
                        if control in indep_pattern_names
                        else None
                    )
                    for control in controls
                ]
                # Filter out truth tables that are not used by
                truth_tables = filter_truth_tables(truth_tables, control_sequence)

                # Create a function for each possible truth table
                for tt in truth_tables:
                    function_realizations.append(
                        DependentFeatureWrapper(base_fn, controls, tt)
                    )

                dependent_pattern_func_cont.append(function_realizations)
        dependent_pattern_func_cont = list(product(*dependent_pattern_func_cont))
        for dep_combination in dependent_pattern_func_cont:
            pattern_functions.append(indep_combination + dep_combination)

    return [
        DisplayChain(list(function_sequence)) for function_sequence in pattern_functions
    ]


def _check_feature_func(feature: str, all_binary_generators: dict) -> bool:
    if (
        NON_ACTIVE_FEATURE_SUFFIX_RE.sub("", feature)
        not in all_binary_generators.keys()
    ):
        if feature != POSITION_PARITY_FEATURE:
            raise ValueError(f"Feature '{feature}' not found in all_binary_features")
    return True


def process_single_function(args):
    realized_function, group_of_features, causal_pattern = args
    pattern = generate_pattern(realized_function, pattern_length=7)
    stochastic = (
        True if realized_function.__repr__().find("stochastic") != -1 else False
    )
    pattern_and_generator = {
        "pattern": pattern,
        "generator": realized_function,
        "sequence_of_features": group_of_features,
        "causal_pattern": causal_pattern,
        "stochastic": stochastic,
    }
    return pattern, pattern_and_generator


def process_pattern(
    group_of_features: list[str],
    causal_pattern: list[tuple[str, list[str]]],
    pattern_length: int = 7,
    use_stochastic_features: bool = False,
    all_binary_generators: dict = all_binary_generators,
) -> dict:
    results = {}
    all_realized_functions = realize_causal_pattern(
        causal_pattern,
        all_binary_generators=all_binary_generators,
        use_stochastic_features=use_stochastic_features,
    )
    for realized_function in all_realized_functions:
        pattern = generate_pattern(realized_function, pattern_length=pattern_length)
        stochastic = (
            True if realized_function.__repr__().find("stochastic") != -1 else False
        )
        pattern_and_generator = {
            "pattern": pattern,
            "generator": realized_function,
            "sequence_of_features": group_of_features,
            "causal_pattern": causal_pattern,
            "stochastic": stochastic,
        }
        results[pattern] = pattern_and_generator
    return results


def generate_all_patterns_and_generators(
    selected_features: list[str],
    all_binary_generators: dict = all_binary_generators,
    use_stochastic_features: bool = False,
    max_controls: int = 3,
    verbose: bool = False,
) -> dict:
    assert (
        max_controls < 5
    ), "with 5 controls we get 2**(2**5) = 4294967296 possible patterns"
    assert len(selected_features) < 12, "lets not blow up for now"
    assert all(
        NON_ACTIVE_FEATURE_SUFFIX_RE.sub("", feature) in all_binary_generators
        for feature in selected_features
    ), "some features are not in a list of known binary features"
    if verbose:
        logging.info(
            f"Generating all possible causal patterns for {len(selected_features)} features..."
        )

    _, dedup_pat = get_all_possible_causal_patterns_labeled(
        selected_features, max_controls=max_controls
    )

    all_causal_patterns = []
    for sequence_of_features in dedup_pat.keys():
        for causal_pattern in dedup_pat[sequence_of_features]:
            all_causal_patterns.append((causal_pattern, sequence_of_features))

    if verbose:
        logging.info(f"Found {len(all_causal_patterns)} unique causal patterns")

    all_generators = set()
    for causal_pattern, sequence_of_features in tqdm(all_causal_patterns):
        results = realize_causal_pattern(
            causal_pattern,
            all_binary_generators=all_binary_generators,
            use_stochastic_features=use_stochastic_features,
        )
        all_generators.update(results)
        if verbose:
            if (
                len(all_generators) // 250000
                > (len(all_generators) - len(results)) // 250000
            ):
                logging.info(f"Processed over {len(all_generators)} patterns")
    return all_generators


def generate_pattern(
    realized_function: DisplayChain,
    start_string: str = "",
    pattern_length: int = 10,
    colorize: bool = False,
) -> str:
    """Given a Chain of Wrapped controlled base functions, generate / sample a string pattern.
    If colorize=True, colors tokens depending on their position; even=blue, *O*dd=*O*range.
    """
    from IPython.display import HTML

    start_string = start_string.strip(" ")
    pattern = start_string + " :" if start_string else ""
    current_s = ""
    prev_s = start_string.split(" ")[-1] if start_string else ""
    position_offset = len(start_string.split(" ")) if start_string else 0

    if colorize:
        # Build HTML string with CSS-based coloring
        html_parts = []
        if start_string:
            html_parts.append(start_string + " :")

        for i in range(pattern_length):
            current_s = realized_function(
                "", s_prev=prev_s, position=(i + position_offset)
            )
            color = (
                "#ffa500" if (i + position_offset) % 2 == 0 else "#007bff"
            )  # blue and orange
            html_parts.append(f'<span style="color: {color}">{current_s}</span>')
            prev_s = current_s

        return HTML(" ".join(html_parts))
    else:
        # Regular plain text output
        for i in range(pattern_length):
            current_s = realized_function(
                "", s_prev=prev_s, position=(i + position_offset)
            )
            pattern += " " + current_s
            prev_s = current_s
        return pattern.strip(" ")
