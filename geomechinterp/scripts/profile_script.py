# use cProfile to profile the function generate_all_patterns_and_generators
from geomechinterp.causal_patterns import generate_all_patterns_and_generators
import cProfile
import itertools


def run_profile():
    ALL_PATTERNS_AND_GENERATORS = {}
    for i in range(1, 5):
        for feature_combination in itertools.combinations(
            [
                "position_parity",
                "ab",
                "case",
                "+-",
                "ab_prev",
                "case_prev",
                "+-_prev",
            ],
            i,
        ):
            if feature_combination == ("position_parity",) or feature_combination == (
                "case",
            ):
                continue
            elif all(
                feature.endswith("_prev") or feature == "position_parity"
                for feature in feature_combination
            ):
                continue
            all_patterns_and_generators = generate_all_patterns_and_generators(
                feature_combination
            )
            for k in all_patterns_and_generators.keys():
                all_patterns_and_generators[k][
                    "group_of_features"
                ] = feature_combination
            ALL_PATTERNS_AND_GENERATORS.update(all_patterns_and_generators)


if __name__ == "__main__":
    profile_file = "profile_output.prof"
    cProfile.run("run_profile()", profile_file)
    print(f"Profile saved to {profile_file}")
