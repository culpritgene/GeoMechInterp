import json
from geomechinterp.causal.pattern_generator import (
    get_exhaustive_patterns_and_pack,
    generate_patterns_mp,
)


dataset_config = [
    {
        "selected_features": [
            "position_parity",
            "ab_prev",
            "case_prev",
            "+-_prev",
            "ab",
            "case",
            "+-",
            "12_prev",
            "12",
        ],
        "max_controls": 3,
        "use_stochastic_features": True,
    },
    {
        "selected_features": [
            "position_parity",
            "ab_prev",
            "case_prev",
            "+-_prev",
            "ab",
            "case",
            "><_prev",
            "><",
            "][_prev",
            "][",
        ],
        "max_controls": 2,
        "use_stochastic_features": True,
    },
    {
        "selected_features": [
            "position_parity",
            "ab_prev",
            "case_prev",
            "+-_prev",
            "ab",
            "case",
            "><_prev",
            "><",
            "][_prev",
            "][",
        ],
        "max_controls": 1,
        "use_stochastic_features": True,
    },
]


def generate_dataset(
    dataset_config: list[dict], final_pattern_length: int = 30
) -> dict:
    ALL_PATTERNS_AND_GENERATORS = {}
    for dataset_part in dataset_config:
        selected_features = dataset_part["selected_features"]
        max_controls = dataset_part["max_controls"]
        use_stochastic_features = dataset_part["use_stochastic_features"]

        patterns_and_generators = get_exhaustive_patterns_and_pack(
            selected_features=selected_features,
            verbose=True,
            use_stochastic_features=use_stochastic_features,
            max_controls=max_controls,
        )
        ALL_PATTERNS_AND_GENERATORS.update(patterns_and_generators)

    # regenerate longer patterns
    ALL_PATTERNS_AND_GENERATORS_FINAL = {}
    for pattern, rec in ALL_PATTERNS_AND_GENERATORS.items():
        extended_pat = generate_patterns_mp(
            rec["generator"], pattern_length=final_pattern_length
        )
        rec["pattern"] = extended_pat
        ALL_PATTERNS_AND_GENERATORS_FINAL[extended_pat] = rec

    return ALL_PATTERNS_AND_GENERATORS_FINAL


def patterns_and_generators_to_json(patterns_and_generators: dict):
    # Convert generators to json format and save all patterns
    all_patterns_json = {
        pattern: {
            **{k: v for k, v in generator_dict.items() if k != "generator"},
            "generator": generator_dict["generator"].to_json(),
        }
        for pattern, generator_dict in patterns_and_generators.items()
    }
    return all_patterns_json


def save_dataset(dataset: dict, filename: str):
    json_data = patterns_and_generators_to_json(dataset)
    json.dump(json_data, open(filename, "w"), indent=2)


if __name__ == "__main__":
    dataset = generate_dataset(dataset_config, final_pattern_length=30)
    save_dataset(dataset, "all_patterns_and_generators_final.json")
