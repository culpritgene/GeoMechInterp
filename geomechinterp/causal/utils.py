import random
import itertools
import re
from typing import List, Callable, Dict
from geomechinterp.causal.base_functions import all_binary_checks
from .truth_table_cache import truth_table_cache


NON_ACTIVE_FEATURE_SUFFIX = "_prev"
POSITION_PARITY_FEATURE = "position_parity"

NON_ACTIVE_FEATURE_SUFFIX_RE = re.compile(rf"{NON_ACTIVE_FEATURE_SUFFIX}$")


class IndependentFeatureWrapper:
    """If Global Control is None we choose control uniformly at random
    Otherwise global control can be set to 0 or 1."""

    def __init__(self, feature_fn, global_control=None):
        self.feature_fn = feature_fn
        self.global_control = global_control

    def __call__(self, s: str, s_prev: str = None, position: int = None):
        # ugly hack to make s_prev and position arguments,
        # despite them not being used in the function
        if self.global_control is None:
            return self.feature_fn(s, random.choice([0, 1]))
        else:
            return self.feature_fn(s, self.global_control)

    def __repr__(self):
        reprs_str = f"IndependentFeature(function={self.feature_fn.__name__}, global_control={self.global_control})"
        if self.global_control is None:
            reprs_str += " (stochastic)"
        else:
            reprs_str += " (deterministic)"
        return reprs_str


# class DependentFeatureWrapper:
#     def __init__(
#         self,
#         control_features: List[str],
#         all_binary_features: Dict[str, Callable],
#         truth_table_values: List[int],
#     ):
#         """
#         Initialize with control features and truth table values

#         Args:
#             control_features: List of feature names this depends on
#             all_binary_features: Dict mapping feature names to their functions
#             truth_table_values: List of output values for truth table
#         """
#         self.control_features = control_features
#         self.all_binary_features = all_binary_features

#         # Create cache key from control features and values
#         self.cache_key = (tuple(sorted(control_features)), tuple(truth_table_values))

#         # Get or create truth table
#         cached_table = truth_table_cache.get_table(self.cache_key)
#         if cached_table is None:
#             truth_table = np.array(truth_table_values, dtype=np.int8)
#             truth_table_cache.store_table(self.cache_key, truth_table)

#     def __call__(self, s: str, c: int) -> int:
#         """Evaluate the dependent feature"""
#         # Get control values
#         control_vals = []
#         for cf in self.control_features:
#             control_fn = self.all_binary_features[cf]
#             control_vals.append(control_fn(s))

#         # Convert control values to index into truth table
#         control_idx = 0
#         for i, val in enumerate(control_vals):
#             control_idx += val * (2**i)

#         # Get truth table from cache
#         truth_table = truth_table_cache.get_table(self.cache_key)
#         return truth_table[control_idx]


class DependentFeatureWrapper:
    """
    Wraps a feature function to be dependent on a set of control features.
    """

    all_binary_checks = all_binary_checks

    def __init__(self, feature_fn, control_features, truth_table_values):
        self.feature_fn = feature_fn
        self.control_features = control_features
        self.truth_table_values = truth_table_values

    def __call__(self, s: str, s_prev: str = "", position: int = None):
        control_vals = []  # list of control values
        for cf in self.control_features:
            if cf == "position_parity":
                control_vals.append(all_binary_checks[cf](s, position))
            else:
                if NON_ACTIVE_FEATURE_SUFFIX_RE.findall(cf):
                    control_vals.append(
                        all_binary_checks[NON_ACTIVE_FEATURE_SUFFIX_RE.sub("", cf)](
                            s_prev
                        )
                    )
                else:
                    control_vals.append(all_binary_checks[cf](s))
        control_idx = 0
        for i, v in enumerate(control_vals):
            control_idx += v * (2**i)
        return self.feature_fn(s, c=self.truth_table_values[control_idx])

    def __repr__(self):
        controls = ", ".join(self.control_features)
        tt = "[" + ", ".join(str(x) for x in self.truth_table_values) + "]"
        return f"DependentFeature(function={self.feature_fn.__name__}, controls=[{controls}], truth_table={tt})"


class DisplayChain(itertools.chain):
    def __init__(self, functions):
        self.functions = functions

    def __iter__(self):
        return iter(self.functions)

    def __repr__(self):
        repr_str = "\n".join(
            [f.__name__ if hasattr(f, "__name__") else str(f) for f in self.functions]
        )
        return f"Chain of functions:\n{repr_str}"

    def __call__(self, *args, **kwargs):
        for f in self.functions:
            result = f(*args, **kwargs)
            # Handle case where function returns a tuple
            if isinstance(result, tuple):
                args = (result[0],)  # Take first element as positional arg
                # Update kwargs with any additional returned values
                if len(result) > 1:
                    kwargs.update({"position": result[1]})
            else:
                args = (result,)
        return args[0]  # NOTE: Return single value instead of tuple

    def __getitem__(self, index):
        return self.functions[index]


def check_non_prev_before_prev(reflections: list[str]) -> bool:
    non_prev_features = [
        i
        for i, refl in enumerate(reflections)
        if not NON_ACTIVE_FEATURE_SUFFIX_RE.findall(refl)
    ]
    prev_features = [
        i
        for i, refl in enumerate(reflections)
        if NON_ACTIVE_FEATURE_SUFFIX_RE.findall(refl)
    ]
    if non_prev_features and prev_features:
        return min(non_prev_features) < min(prev_features)
    return False


def check_causal_validity(reflections: list[str]) -> bool:
    if "position_parity" in reflections:
        if reflections.index("position_parity") != 0:
            return False
    # if check_non_prev_before_prev(reflections):
    #     return False
    if "case" in reflections:
        if "ab" not in reflections:
            return False
        if reflections.index("case") <= reflections.index("ab"):
            return False
    return True


def print_causal_structures(
    causal_structures: list[tuple[str, list[str]]], print_indirect: bool = False
):
    for causal_structure in causal_structures:
        if print_indirect:
            print_causal_structure_indirect(causal_structure)
        else:
            print_causal_structure_direct(causal_structure)


def print_causal_structure_direct(causal_structure: tuple[str, list[str]]):
    print("[")
    for feature, controls in causal_structure:
        controls_str = "{" + "; ".join(controls) + "}" if controls else "free"
        print(f"  {feature}: {controls_str}")
    print("]")


def print_causal_structure_indirect(causal_structure: tuple[str, list[str]]):
    def build_chain(current_feature):
        chain.append(current_feature)
        for f, c in causal_structure:
            if f == current_feature and c:
                for indirect in c:
                    build_chain(indirect)

    print("[")
    for feature, controls in causal_structure:
        if not controls:
            controls_str = "free"
        else:
            # for each direct control, build a chain of indirect controls
            control_chains = []
            for control in controls:
                chain = []
                # build control sequence recursively
                build_chain(control)
                # format chain with arrows
                control_chains.append(" -> ".join(reversed(chain)))

            # join multiple control chains with newlines
            controls_str = "; ".join(control_chains)
            controls_str = f"{{{controls_str}}}"
        print(f"  {feature}: {controls_str}")
    print("]")


def sort_prev_first(reflections: list[str]) -> tuple[list[str], list[str]]:
    non_active_features = [
        refl
        for refl in reflections
        if NON_ACTIVE_FEATURE_SUFFIX_RE.findall(refl) or refl == POSITION_PARITY_FEATURE
    ]
    active_features = [
        refl
        for refl in reflections
        if not NON_ACTIVE_FEATURE_SUFFIX_RE.findall(refl)
        and refl != POSITION_PARITY_FEATURE
    ]
    return non_active_features, active_features
