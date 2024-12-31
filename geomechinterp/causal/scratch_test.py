from itertools import product
from typing import Callable, List, Dict, Tuple


# Example placeholders for your wrappers:
class IndependentFeatureWrapper:
    """Wraps a base function with a forced global control (None, 0, or 1)."""

    def __init__(self, base_fn: Callable, global_control):
        self.base_fn = base_fn
        self.global_control = global_control

    def __call__(self, s: str, _: Dict[str, int]) -> str:
        """
        If global_control is None => pick randomly or do something stochastic.
        If global_control is 0 => always apply 'base_fn' with c=0
        If global_control is 1 => always apply 'base_fn' with c=1
        """
        if self.global_control is None:
            # For illustration, let's pick 0 or 1 randomly
            import random

            bit = random.randint(0, 1)
            return self.base_fn(s, bit)
        else:
            return self.base_fn(s, self.global_control)

    def __repr__(self):
        return f"Independent({self.base_fn.__name__}, ctrl={self.global_control})"


class DependentFeatureWrapper:
    """Wraps a base function with a table that maps a subset of controlling bits -> {0,1}."""

    def __init__(
        self,
        base_fn: Callable,
        controls: List[str],
        truth_table: List[int],
        pinned_controls: Dict[str, int],
    ):
        """
        :param base_fn: the function that modifies the string, base_fn(s, bit).
        :param controls: the list of *non-constant* controlling features in consistent order.
        :param truth_table: a list of length 2^len(controls) giving 0/1 outputs.
        :param pinned_controls: dictionary of feature -> pinned value (0 or 1).
               If not present or value=None, that means it's not pinned or it's random.
        """
        self.base_fn = base_fn
        self.controls = controls
        self.tt = truth_table
        self.pinned_controls = pinned_controls

    def __call__(self, s: str, parent_values: Dict[str, int]) -> str:
        """
        parent_values: a dictionary with each parent feature's 0/1 (or random) realization.
        We figure out the index for the "non-constant" parents, read from self.tt,
        then apply base_fn(s, bit).
        Also incorporate pinned_controls = { 'ab': 0, ... } if needed.
        """
        # 1) Build the input bits for just the "non-constant" controls
        idx = 0
        for feature_name in self.controls:
            # The parent_values dict has 0 or 1 for that feature
            bit_val = parent_values[feature_name]
            idx = (idx << 1) | bit_val

        out_bit = self.tt[idx]

        # 2) Incorporate pinned controls if that changes anything
        #    (often you don't need to do anything extra,
        #     because pinned bits are "baked" into the TT).
        #    But if you want to do "AND pinned=1 => force out=1", you'd do it here.
        #    For simplicity, let's do nothing extra.

        return self.base_fn(s, out_bit)

    def __repr__(self):
        # Just a short debug
        return f"Dependent({self.base_fn.__name__}, controls={self.controls}, pinned={self.pinned_controls})"


class DisplayChain:
    """Just a container for the final combination of functions."""

    def __init__(self, fn_list):
        self.fn_list = fn_list

    def __call__(self, s: str, parent_values_seq: List[Dict[str, int]]) -> str:
        """
        For demonstration, assume we have one dictionary of parent_values per function
        in sequence, or some consistent way to gather them.
        """
        out = s
        for fn, pvals in zip(self.fn_list, parent_values_seq):
            out = fn(out, pvals)
        return out

    def __repr__(self):
        return " -> ".join([repr(fn) for fn in self.fn_list])


def generate_truth_tables(num_controls: int) -> List[List[int]]:
    """Naive: return all possible 2^(2^num_controls) truth tables,
    each truth table is a list[int] of length 2^num_controls."""
    results = []
    nrows = 2**num_controls
    nfuncs = 2**nrows
    for i in range(nfuncs):
        # binary representation of length nrows
        table = [(i >> j) & 1 for j in range(nrows)]
        table.reverse()
        results.append(table)
    return results


def realize_causal_pattern_optimized(
    pattern: List[Tuple[str, List[str]]],
    all_binary_generators: Dict[str, Callable],
    pinned_features: Dict[str, int or None] = None,
) -> List[DisplayChain]:
    """
    A more optimized version of realize_causal_pattern that:
      - Skips enumerating full 2^(2^n) if some controlling features are pinned to 0 or 1
      - Only enumerates truth tables for the "non-constant" controls

    :param pattern: e.g. [
        ('position_parity', ()),
        ('ab', ()),
        ('case', ('ab', 'position_parity')),
        ('+-', ('ab', 'position_parity'))
      ]
      where each tuple is (feature, list_of_controlling_features).
    :param pinned_features: dict { feature_name -> 0 or 1 or None }, meaning:
         - 0 or 1 => pinned constant
         - None => random  (like "free" but not enumerated)
         - not in pinned_features => we treat it as non-constant (fully enumerated if it is a parent).
    """
    if pinned_features is None:
        pinned_features = {}

    pattern_functions = []

    for feature, controls in pattern:
        # Check if this feature is in all_binary_generators (i.e., truly "active"):
        base_name = feature  # or strip suffix, if needed
        if base_name not in all_binary_generators:
            # Possibly skip or handle position_parity or other special features
            continue

        base_fn = all_binary_generators[base_name]

        # ========== CASE 1: No controlling features ==========
        if not controls:
            # If a feature has no controls, we interpret it as:
            #  - pinned 0
            #  - pinned 1
            #  - None => random
            # or we just do "IndepFeatureWrapper(base_fn, ctrl)"
            sub_realizations = []
            for ctrl_val in [0, 1, None]:
                sub_realizations.append(IndependentFeatureWrapper(base_fn, ctrl_val))

            pattern_functions.append(sub_realizations)
            continue

        # ========== CASE 2: Some controlling features exist ==========

        # Partition the controlling features into pinned vs. non-constant:
        pinned_dict = {}  # e.g. {'ab':0, 'position_parity':1}
        nonconst_list = []  # e.g. ['ab'] if pinned_features['position_parity']=1

        for c in controls:
            if c in pinned_features:
                val = pinned_features[c]
                if val is not None and val in (0, 1):
                    pinned_dict[c] = val
                else:
                    # If pinned_features[c] is None => "random input bit" => treat as non-constant dimension
                    nonconst_list.append(c)
            else:
                # If not specified in pinned_features => treat as fully non-constant
                nonconst_list.append(c)

        # Now the dimension is len(nonconst_list).
        n_controls = len(nonconst_list)
        if n_controls == 0:
            # That means *all* controlling features are pinned constants => effectively 0 bits of input.
            # So output can only be pinned to 0, pinned to 1, or random.
            # i.e. we skip enumerating big truth tables, because there's no input dimension:
            sub_realizations = []
            for out_val in [0, 1, None]:
                if out_val is None:
                    # random
                    def fn_random_factory(fn=base_fn):
                        def fn_random(s, _):
                            import random

                            return fn(s, random.randint(0, 1))

                        return fn_random

                    sub_realizations.append(fn_random_factory())
                else:
                    # pinned
                    def fn_const_factory(val, fn=base_fn):
                        def fn_const(s, _):
                            return fn(s, val)

                        return fn_const

                    sub_realizations.append(fn_const_factory(out_val))

            pattern_functions.append(sub_realizations)
        else:
            # We need to enumerate the boolean functions of `n_controls` bits
            all_tt = generate_truth_tables(n_controls)
            sub_realizations = []

            for tt in all_tt:
                # Create a DependentFeatureWrapper that bakes in pinned_dict
                wrapper = DependentFeatureWrapper(
                    base_fn=base_fn,
                    controls=nonconst_list,  # only the non-constant parents
                    truth_table=tt,
                    pinned_controls=pinned_dict,
                )
                sub_realizations.append(wrapper)

            pattern_functions.append(sub_realizations)

    # Finally, we do the Cartesian product across all features in the pattern:
    all_combinations = list(product(*pattern_functions))
    return [DisplayChain(list(combo)) for combo in all_combinations]


# --------------- Example usage ---------------
if __name__ == "__main__":

    # Suppose we have some base generator functions:
    def ab_f(s: str, c: int) -> str:
        return s + ("a" if c == 1 else "b")

    def case_f(s: str, c: int) -> str:
        return s.upper() if c == 1 else s.lower()

    all_binary_generators = {"ab": ab_f, "case": case_f}

    # Example pattern: case depends on ab
    pattern_example = [
        ("ab", []),
        ("case", ["ab"]),
    ]

    # pinned_features example:
    #  - "ab" pinned to 0 => always "b"
    #  - "case" is not pinned => we do normal enumeration for it
    pinned = {
        "ab": 0,
    }

    # Generate the optimized chain
    combos = realize_causal_pattern_optimized(
        pattern=pattern_example,
        all_binary_generators=all_binary_generators,
        pinned_features=pinned,
    )

    print("Number of combos:", len(combos))
    for c in combos:
        print("Chain:", c)
