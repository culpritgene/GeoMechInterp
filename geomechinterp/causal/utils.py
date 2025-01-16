import random
import itertools
import re
import numpy
from typing import Callable
from geomechinterp.causal.base_functions import (
    all_binary_checks,
    all_binary_generators_func_names,
)
from .truth_table_cache import truth_table_cache


NON_ACTIVE_FEATURE_SUFFIX = "_prev"
POSITION_PARITY_FEATURE = "position_parity"

NON_ACTIVE_FEATURE_SUFFIX_RE = re.compile(rf"{NON_ACTIVE_FEATURE_SUFFIX}$")


class IndependentFeatureWrapper:
    """If Global Control is None we choose control uniformly at random
    Otherwise global control can be set to 0 or 1."""

    def __init__(self, feature_fn: Callable, global_control: int = None):
        self.feature_fn: Callable = feature_fn
        self.global_control: int = global_control

    def __call__(self, s: str, s_prev: str = None, position: int = None):
        # ugly hack to make s_prev and position arguments,
        # despite them not being used in the function
        if self.global_control is None:
            return self.feature_fn(s, random.choice([0, 1]))
        else:
            return self.feature_fn(s, self.global_control)

    def __repr__(self):
        reprs_str = (
            f"IndependentFeature(function={self.feature_fn.__name__}, "
            f"global_control={self.global_control})"
        )
        if self.global_control is None:
            reprs_str += " (stochastic)"
        else:
            reprs_str += " (deterministic)"
        return reprs_str

    def __hash__(self):
        return hash((self.feature_fn.__name__, self.global_control))

    def to_json(self):
        return {
            "type": "IndependentFeatureWrapper",
            "function": self.feature_fn.__name__,
            "global_control": (
                int(self.global_control) if self.global_control is not None else None
            ),
        }

    @classmethod
    def from_json(cls, data, base_functions):
        return cls(
            feature_fn=base_functions[data["function"]],
            global_control=data["global_control"],
        )


class DependentFeatureWrapper:
    """
    Wraps a feature function to be dependent on a set of control features.
    """

    all_binary_checks = all_binary_checks

    def __init__(
        self,
        feature_fn: Callable,
        control_features: list[str],
        truth_table_values: list[int],
    ):
        self.feature_fn: Callable = feature_fn
        self.control_features: list[str] = control_features
        self.truth_table_values: list[int] = truth_table_values

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
            # truth table vvvv|vvvv
            # controls 101
            # index 1*2^0 + 0*2^1 + 1*2^2 = 1 + 0 + 4 = 5
            # controls 111
            # index 1*2^0 + 1*2^1 + 1*2^2 = 1 + 2 + 4 = 7
            control_idx += v * (2**i)
        return self.feature_fn(s, c=self.truth_table_values[control_idx])

    def __repr__(self):
        controls = ", ".join(self.control_features)
        tt = "[" + ", ".join(str(x) for x in self.truth_table_values) + "]"
        return (
            f"DependentFeature(function={self.feature_fn.__name__}, "
            f"controls=[{controls}], truth_table={tt})"
        )

    def __hash__(self):
        return hash(
            (
                self.feature_fn.__name__,
                tuple(self.control_features),
                tuple(
                    None if isinstance(x, numpy.ma.core.MaskedConstant) else x
                    for x in self.truth_table_values
                ),
            )
        )

    def to_json(self):
        return {
            "type": "DependentFeatureWrapper",
            "function": self.feature_fn.__name__,
            "control_features": self.control_features,
            "truth_table_values": [int(x) for x in self.truth_table_values],
        }

    @classmethod
    def from_json(cls, data, base_functions):
        return cls(
            feature_fn=base_functions[data["function"]],
            control_features=data["control_features"],
            truth_table_values=data["truth_table_values"],
        )


class DisplayChain:
    def __init__(
        self,
        functions: list[IndependentFeatureWrapper | DependentFeatureWrapper],
    ):
        self.functions: list[IndependentFeatureWrapper | DependentFeatureWrapper] = (
            functions
        )

    def __iter__(self):
        return iter(self.functions)

    def __repr__(self):
        repr_str = "\n".join(
            [f.__name__ if hasattr(f, "__name__") else str(f) for f in self.functions]
        )
        return f"Chain of functions:\n{repr_str}"

    def __hash__(self):
        return hash(tuple(self.functions))

    def __eq__(self, other):
        if not isinstance(other, DisplayChain):
            return False
        return self.__hash__() == other.__hash__()

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

    @property
    def dag(self):
        """Returns list of tuples (function_name, [control_features])"""
        result = []
        for f in self.functions:
            if isinstance(f, IndependentFeatureWrapper):
                result.append((f.feature_fn.__name__, []))
            else:  # DependentFeatureWrapper
                result.append((f.feature_fn.__name__, f.control_features))
        # sort by the number of controls, with independent features first
        # result.sort(key=lambda x: len(x[1]))
        return result

    def to_json(self):
        return {
            "type": "DisplayChain",
            "functions": [f.to_json() for f in self.functions],
        }

    @classmethod
    def from_json(cls, data, base_functions=all_binary_generators_func_names):
        wrapper_map = {
            "IndependentFeatureWrapper": IndependentFeatureWrapper,
            "DependentFeatureWrapper": DependentFeatureWrapper,
        }
        functions = [
            wrapper_map[f["type"]].from_json(f, base_functions)
            for f in data["functions"]
        ]
        return cls(functions)


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


def get_word_starts(text: str) -> list[int]:
    """Returns a list of word starts in the text, character-wise."""
    words = text.split()
    cur = 0
    word_starts = [cur]
    for i, word in enumerate(words):
        cur += len(word) + 1  # adding space
        word_starts.append(cur)
    return word_starts
