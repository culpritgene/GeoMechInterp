from typing import Callable, List
import numpy as np
import logging
from itertools import permutations, product
from tqdm import tqdm

from geomechinterp.causal.hasse import (
    get_all_possible_hasse_diagrams,
    unflatten_to_causal_triangle,
    filter_truth_tables,
    filter_non_causal_truth_tables_binary,
    classify_truth_table,
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

from geomechinterp.causal.base_functions import all_binary_generators
from multiprocessing import Pool, Manager
from geomechinterp.causal.truth_table_cache import truth_table_cache
from geomechinterp.graph.utils import dag_to_wl_hash
from geomechinterp.utils import one_hot_encode_columns, filter_categories
from geomechinterp.informat.entropy import estimate_entropy_char, estimate_entropy_token
from geomechinterp.informat.compression import compression_complexity

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


def get_possible_controls(
    prefix_funcs: list[Callable], control_features: list[str]
) -> list[int]:
    """Get possible control values for a set of control features given prefix functions."""
    possible_controls = set(range(2 ** len(control_features)))
    test_str = "a"  # Any test string will do since we just need control values

    # For each possible control combination
    for control_idx in range(2 ** len(control_features)):
        control_vals = []
        # Get control values from prefix functions
        for cf in control_features:
            for f in prefix_funcs:
                if (
                    isinstance(f, IndependentFeatureWrapper)
                    and f.feature_fn.__name__ == cf
                ):
                    control_vals.append(f(test_str))
                elif (
                    isinstance(f, DependentFeatureWrapper)
                    and f.feature_fn.__name__ == cf
                ):
                    control_vals.append(f(test_str))

        # If any control value is fixed and doesn't match truth table index, remove it
        control_idx_bin = format(control_idx, f"0{len(control_features)}b")
        for i, val in enumerate(control_vals):
            if val is not None and int(control_idx_bin[i]) != val:
                possible_controls.discard(control_idx)
                break

    return list(possible_controls)


def add_dependent_feature(
    prefix_funcs: list[Callable],
    feature: str,
    controls: list[str],
    base_fn: Callable,
    remove_non_causal_truth_tables: bool = True,
) -> list[list[Callable]]:
    """Recursively add dependent feature to function chain."""
    if not controls:
        return [prefix_funcs]

    n_controls = len(controls)
    truth_tables = truth_table_cache.get_table(n_controls)
    possible_controls = get_possible_controls(prefix_funcs, controls)

    # Filter truth tables based on possible controls
    valid_truth_tables = []
    for tt in truth_tables:
        valid = True
        for control_idx in range(2**n_controls):
            if control_idx not in possible_controls and tt[control_idx] is not None:
                valid = False
                break
        if valid:
            valid_truth_tables.append(tt)

    valid_truth_tables = np.stack(valid_truth_tables)
    if remove_non_causal_truth_tables:
        # print("valid_truth_tables: ", valid_truth_tables)
        valid_truth_tables = filter_non_causal_truth_tables_binary(valid_truth_tables)

    result = []
    for tt in valid_truth_tables:
        new_func = DependentFeatureWrapper(base_fn, controls, tt)
        result.append(prefix_funcs + [new_func])

    return result


def realize_causal_pattern_with_pruning(
    pattern: list[tuple[str, list[str]]],
    use_stochastic_features: bool = False,
    remove_non_causal_truth_tables: bool = True,
    all_binary_generators: dict[str, Callable] = all_binary_generators,
) -> list[DisplayChain]:
    """
    Takes a causal pattern description and returns all possible realizations of that pattern
    as functions that generate sequences according to the causal dependencies.

    Args:
        pattern: List of tuples where each tuple contains (feature, controlling_features)
        use_stochastic_features: Whether to use stochastic features, with 50% probability
            of being 0 or 1
        all_binary_features: Dict mapping feature names to their generating functions

    Returns:
        List of functions that each generate a valid sequence following the causal pattern
    """

    # Initialize independent features
    pattern = _check_pattern_structure(pattern)
    independent_pattern_functions: list[list[Callable]] = []
    independent_global_controls: list[list[int]] = []
    indep_pattern_names: list[str] = []
    pattern_functions: list[list[Callable]] = []

    independent_controls = [None, 0, 1] if use_stochastic_features else [0, 1]

    # Handle independent features first
    for feature, controls in pattern:
        if not controls:
            if feature == POSITION_PARITY_FEATURE:
                continue

            _check_feature_func(feature, all_binary_generators)
            base_fn = all_binary_generators[
                NON_ACTIVE_FEATURE_SUFFIX_RE.sub("", feature)
            ]

            function_realizations = []
            for global_control in independent_controls:
                function_realizations.append(
                    IndependentFeatureWrapper(base_fn, global_control)
                )
            indep_pattern_names.append(feature)
            independent_pattern_functions.append(function_realizations)
            independent_global_controls.append(independent_controls)

    # Generate independent feature combinations
    all_indep_combinations = list(product(*independent_pattern_functions))

    # For each independent combination, recursively add dependent features
    for indep_combination in all_indep_combinations:
        current_chains = [list(indep_combination)]

        # Add each dependent feature recursively
        for feature, controls in pattern:
            if controls:
                _check_feature_func(feature, all_binary_generators)
                base_fn = all_binary_generators[
                    NON_ACTIVE_FEATURE_SUFFIX_RE.sub("", feature)
                ]

                new_chains = []
                for chain in current_chains:
                    new_chains.extend(
                        add_dependent_feature(
                            chain,
                            feature,
                            controls,
                            base_fn,
                            remove_non_causal_truth_tables,
                        )
                    )
                current_chains = new_chains

        pattern_functions.extend(current_chains)

    return [
        DisplayChain(list(function_sequence)) for function_sequence in pattern_functions
    ]


def realize_causal_pattern(
    pattern: list[tuple[str, list[str]]],
    use_stochastic_features: bool = False,
    remove_non_causal_truth_tables: bool = True,
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
    pattern = _check_pattern_structure(pattern)
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

                if remove_non_causal_truth_tables:
                    truth_tables = filter_non_causal_truth_tables_binary(truth_tables)

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


def _check_pattern_structure(pattern: list[tuple[str, list[str]]]) -> bool:
    assert isinstance(pattern, (tuple, list)), "pattern must be a tuple or list"
    assert len(pattern) > 0, "Pattern must be non-empty"
    assert isinstance(
        pattern[0], (tuple, list)
    ), f"each control subpattern must be a tuple or list, got: {pattern[0]} instead"
    assert isinstance(
        pattern[0][0], str
    ), f"first element of each control subpattern must be a string, got: {pattern[0][0]} instead"
    for i, (feature, controls) in enumerate(pattern):
        if not isinstance(controls, (tuple, list)):
            logging.warning(
                f"second element of each control subpattern must be a tuple or list, got: {controls} instead; wrapping in tuple"
            )
            pattern[i] = (feature, tuple([controls]))

    return pattern


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
        remove_non_causal_truth_tables=True,
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


def worker(args):
    # Define worker function at module level to allow pickling
    return realize_causal_pattern(*args)


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

    # Create arguments for parallel processing
    process_args = []
    for causal_pattern, sequence_of_features in all_causal_patterns:
        process_args.append(
            (
                causal_pattern,
                use_stochastic_features,
            )
        )

    # Process patterns in parallel using tqdm progress bar
    all_generators = set()
    last_logged_size = 0
    with Manager() as manager:
        # Create a tqdm instance in the manager
        tqdm_instance = tqdm(total=len(process_args))
        tqdm_lock = manager.Lock()

        # Create the pool and process with progress updates
        with Pool() as pool:
            for results in pool.imap_unordered(worker, process_args):
                with tqdm_lock:
                    tqdm_instance.update()
                all_generators.update(results)

                # Log every 500k new generators
                current_size = len(all_generators)
                if current_size >= last_logged_size + 500000:
                    logging.info(
                        f"Found {current_size} (non-unique) generators so far..."
                    )
                    last_logged_size = current_size

        tqdm_instance.close()

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


import pandas as pd

# Suppose we know how to map function names like 'plus_minus_f' -> the short feature label '+-'
FUNCTION_NAME_TO_FEATURE = {
    "plus_minus_f": "+-",
    "ab_f": "ab",
    "case_f": "case",
    "f12_f": "12",
    # add others if needed
}

# A universal set of possible features:
ALL_KNOWN_FEATURES = ["position_parity", "ab", "case", "+-", "12", "><", "?!", "]["]


def extract_features_from_dataset_entry(
    entry: dict,
    attach_patterns: bool = False,
    attach_patterns_length: int | None = None,
    compute_uncertainty: bool = False,
    compute_pattern_compression: bool = False,
) -> dict:
    """
    Given one dataset entry, return a dictionary of higher-level features:
    {
      'num_functions': ...,
      'num_edges': ...,
      'presence_position_parity': 0 or 1,
      'presence_ab': 0 or 1,
      ...
      'count_constant_0': # of TT labeled as 'constant_0',
      'count_constant_1': ...,
      'count_balanced': ...,
      'count_other': ...
    }
    """

    # 1) Basic info
    generator_dict = entry["generator"]
    generator = DisplayChain.from_json(generator_dict)
    functions = generator_dict.get("functions", [])
    seq_of_features = entry.get("sequence_of_features", [])

    # 2) num_functions
    num_funcs = len(functions)

    # 3) Count edges in the DAG
    #    Summation of len(control_features) for each function
    edge_count = 0

    # We might also track the set of "function feature names" encountered
    function_feature_set = set()
    num_independent_functions = 0
    max_controls = 0

    tt_by_size_ = {2: {}, 4: {}, 8: {}, 16: {}}
    total_tt_symmetry_complexity = 0
    for fn in functions:
        # Identify the short feature name from the function
        fn_name = fn.get("function", "")
        short_feat = FUNCTION_NAME_TO_FEATURE.get(fn_name, fn_name)
        function_feature_set.add(short_feat)

        c_feats = fn.get("control_features", [])
        if not c_feats or c_feats is None:
            num_independent_functions += 1
            continue

        edge_count += len(c_feats)
        max_controls = max(max_controls, len(c_feats))

        tt_values = fn.get("truth_table_values", None)

        # IndependentFeatures will not have truth tables attached
        if tt_values is not None:
            tt_size = len(tt_values)
            tt_size_dict = tt_by_size_[tt_size]
            tt_size_dict["count"] = tt_size_dict.get("count", 0) + 1
            # return string flags characterizing truth table
            tt_flags, symmetry_complexity = classify_truth_table(tt_values)
            # store counts for each non empty flag grouped by tt size
            for flag in tt_flags:
                tt_size_dict[flag] = tt_size_dict.get("count", 0) + 1
            total_tt_symmetry_complexity += symmetry_complexity

    tt_by_size = {}
    for tt_size, tt_by_size_feats in tt_by_size_.items():
        str_prefix = f"tt_size_{tt_size}"
        for tt_flag, tt_flag_count in tt_by_size_feats.items():
            tt_by_size[str_prefix + "_" + tt_flag] = tt_flag_count

    # 4) presence/absence for particular features
    feature_presence = {}
    for feat in ALL_KNOWN_FEATURES:
        # If the feature is in sequence_of_features OR in the function_feature_set
        val = 1 if (feat in seq_of_features or feat in function_feature_set) else 0
        feature_presence[f"presence_{feat}"] = val

    # consider DAG abstract structure
    equivalence_class_hash = dag_to_wl_hash(generator.dag)
    labeled_graph_hash = dag_to_wl_hash(generator.dag, strip_node_labels=False)

    # 6) Consolidate all features into a single dict
    # Consolidate all features into a single dict
    features_dict = {
        "num_functions": num_funcs,
        "num_edges": edge_count,
        "num_indep_funcs": num_independent_functions,
        "stochastic": int(entry.get("stochastic", False)),
        **feature_presence,
        "max_control": max_controls,
        "dag_equivalence_class": equivalence_class_hash,
        "labeled_graph_hash": labeled_graph_hash,
        **tt_by_size,
    }

    if attach_patterns:
        pattern = entry["pattern"]
        features_dict["pattern"] = (
            pattern[:attach_patterns_length] if attach_patterns_length else pattern
        )

    if compute_uncertainty:
        features_dict["pattern_entropy_token"] = estimate_entropy_token(pattern)
        features_dict["pattern_entropy_char"] = estimate_entropy_char(pattern)

    if compute_pattern_compression:
        features_dict.update(compression_complexity(pattern))

    return features_dict


def build_features_dataframe(
    dataset: list[dict],
    one_hot_hash_features: bool = True,
    take_top_freq_cats: int | float | None = 0.85,
    drop_constant_columns: bool = True,
) -> pd.DataFrame:
    """
    one_hot_hash_features - dummify hashes representing dag structure
    take_top_freq_cats - if not None, take top freq categories and dummify only them
        if int - threshold by count
        if float - threshold by quantile
        if None - do not pre-filter

    dataset: a list of entries, each with keys like:
      {
         'text': <string>,
         'generator': {...},
         'stochastic': bool,
         'sequence_of_features': [ ... ],
         ...
      }
    Returns a DataFrame, each row is a single dataset entry,
    columns are the extracted meta-features.
    """

    all_rows = []
    for entry in dataset:
        feats = extract_features_from_dataset_entry(entry)
        all_rows.append(feats)

    df = pd.DataFrame(all_rows)
    # df = df.reindex(columns=all_rows[0].keys(), fill_value=np.nan)

    if take_top_freq_cats is not None:
        for column in df.columns:
            df = filter_categories(df, column, take_top_freq_cats)

    if one_hot_hash_features:
        # dummify hash features using sklearn
        df = one_hot_encode_columns(
            df,
            columns=["dag_equivalence_class", "labeled_graph_hash"],
            rename_dummies=True,
        )

    # fill missing values with 0
    df.fillna(0, inplace=True)
    if drop_constant_columns:
        df = df.drop(columns=[col for col in df.columns if df[col].nunique() == 1])
    return df


# ------------------- Example usage -------------------
if __name__ == "__main__":
    # Suppose 'my_dataset' is a list of entries in the format you showed:
    my_dataset = [
        {
            "text": "+a +a +b -b +a ...",
            "generator": {
                "functions": [
                    {
                        "control_features": ["ab_prev", "position_parity"],
                        "function": "plus_minus_f",
                        "global_control": None,
                        "truth_table_values": [0, 1, 1, 1],
                        "type": "DependentFeatureWrapper",
                    },
                    {
                        "control_features": ["+-", "ab_prev", "position_parity"],
                        "function": "ab_f",
                        "global_control": None,
                        "truth_table_values": [0, 1, 0, 1, 0, 1, 0, 0],
                        "type": "DependentFeatureWrapper",
                    },
                    {
                        "control_features": ["+-", "ab", "position_parity"],
                        "function": "case_f",
                        "global_control": None,
                        "truth_table_values": [0, 1, 0, 0, 1, 0, 1, 0],
                        "type": "DependentFeatureWrapper",
                    },
                ],
                "type": "DisplayChain",
            },
            "stochastic": False,
            "sequence_of_features": ["position_parity", "ab_prev", "ab", "case", "+-"],
        },
        {
            "text": "+a +a -A +a -A +a -A ...",
            "generator": {
                "functions": [
                    {
                        "control_features": ["ab_prev", "position_parity"],
                        "function": "plus_minus_f",
                        "global_control": None,
                        "truth_table_values": [1, 1, 1, 0],
                        "type": "DependentFeatureWrapper",
                    },
                    {
                        "control_features": ["+-", "ab_prev", "position_parity"],
                        "function": "ab_f",
                        "global_control": None,
                        "truth_table_values": [1, 1, 0, 1, 0, 1, 1, 0],
                        "type": "DependentFeatureWrapper",
                    },
                    {
                        "control_features": ["+-", "ab", "ab_prev"],
                        "function": "case_f",
                        "global_control": None,
                        "truth_table_values": [1, 0, 0, 0, 1, 1, 1, 0],
                        "type": "DependentFeatureWrapper",
                    },
                ],
                "type": "DisplayChain",
            },
            "stochastic": False,
            "sequence_of_features": ["position_parity", "ab_prev", "ab", "case", "+-"],
        },
    ]

    df_features = build_features_dataframe(my_dataset)
    print(df_features)
