from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
import os
import sys
from pathlib import Path
import torch
from datasets import Dataset
from geomechinterp.causal.utils import get_word_starts
from geomechinterp.tflens.utils import HookedTransformer


def accumulate_activations(
    model: HookedTransformer,
    dataset: Dataset,
    selected_hooks: list[str],
    save_logits: bool = False,
    save_loss: bool = False,
    from_word_idx: int = 0,
    select_num_chars: int = 30,
    batch_size: int | None = None,
    samples_per_file: int | None = None,
    save_suffix: str | None = None,
    save_dir: str = "tensors",
):
    if not batch_size and samples_per_file:
        raise ValueError("if samples_per_file is not None, batch_size must be provided")
    assert select_num_chars > 0, "select_num_chars must be greater than 0"
    assert select_num_chars <= 30, "select_num_chars must be less than 30"
    assert from_word_idx >= 0, "from_word_idx must be greater than 0"

    if samples_per_file:
        batches_per_file = samples_per_file // batch_size
    else:
        batches_per_file = 1

    accumulated_activations = {
        hook.format(i=i): [] for hook in selected_hooks for i in range(1, 6)
    }
    if save_logits:
        accumulated_activations["logits"] = []
    if save_loss:
        accumulated_activations["loss"] = []

    accumulated_strings = []
    selected_word_starts = []
    meta_info_template = {
        "selected_word_positions": from_word_idx,
        "selected_word_starts": [],
        "selected_char_positions": [],
        "select_substrings": [],
        "dataset_positions": (0, 0),
        "timestamp": datetime.now().isoformat(),
        "file_counter": 0,
    }
    meta_info = deepcopy(meta_info_template)

    # add word starts to the dataset
    dataset = dataset.map(lambda x: {"word_starts": get_word_starts(x["text"])})

    # Accumulate activations in batches and save periodically
    if not batch_size:
        # use single batch
        total_batches = 1
        batch_size = len(dataset)
    else:
        total_batches = len(dataset) // batch_size + 1

    saved_files = []
    file_counter = 0

    device = model.device

    for batch_idx in tqdm(range(total_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(dataset))

        batch: dict = dataset[start_idx:end_idx]
        out, activations = model.run_with_cache(batch["input_ids"].to(device))

        # the problem is that words can be of different lengths
        # yet we want to accumulate activations for concrete *word* positions, not character positions
        # so we need to loop for each pattern individually instead of slicing once
        for hook in accumulated_activations:
            for i, word_pos in enumerate(batch["word_starts"]):
                char_pos = (
                    word_pos[from_word_idx],
                    word_pos[from_word_idx] + select_num_chars,
                )
                accumulated_strings.append(batch["text"][i][char_pos[0] : char_pos[1]])
                selected_word_starts.append(char_pos)
                accumulated_activations[hook].append(
                    activations[hook][i, char_pos[0] : char_pos[1], :]
                )
                if save_logits:
                    accumulated_activations["logits"].append(
                        out.logits[i, char_pos[0] : char_pos[1], :]
                    )
        if save_loss:
            accumulated_activations["loss"].append(out.loss[i])

        # we need to keep track of positions in the dataset
        meta_info["dataset_positions"] = (
            min(meta_info["dataset_positions"][0], start_idx),
            max(meta_info["dataset_positions"][1], end_idx),
        )
        # and positions within tokenized sequence
        # NOTE: because we select substrings based on *word* positions, not character positions!
        meta_info["select_substrings"].extend(accumulated_strings)
        meta_info["selected_word_starts"].extend(selected_word_starts)

        # accumulated_activations = {hook: torch.tensor(accumulated_activations[hook]) for hook in accumulated_activations}
        # Save after accumulating samples_per_file samples
        if (batch_idx + 1) % batches_per_file == 0 or batch_idx == total_batches - 1:
            # Concatenate accumulated tensors
            for hook in accumulated_activations:
                accumulated_activations[hook] = torch.stack(
                    accumulated_activations[hook], dim=0
                )

            meta_info["file_counter"] = file_counter
            # Save concatenated tensors and meta info to disk
            save_dict = {"activations": accumulated_activations, "meta": meta_info}
            if save_suffix:
                output_file = f"{save_dir}/accumulated_activations_{save_suffix}_{file_counter}.pt"
            else:
                output_file = f"{save_dir}/accumulated_activations_{file_counter}.pt"
            saved_files.append(output_file)
            torch.save(save_dict, output_file)

            # Reset accumulators
            accumulated_activations = {
                hook.format(i=i): [] for hook in selected_hooks for i in range(1, 6)
            }
            meta_info = deepcopy(meta_info_template)
            file_counter += 1
    return saved_files


def load_precomputed_activation(
    input_dir: str | Path,
    selected_hooks: list[str] | None = None,
    subselected_positions: list[int] | None = None,
    merge_meta_batches: bool = True,
    file_substring: str | None = None,
) -> tuple[dict, dict]:
    max_bytes = 8 * 1024 * 1024 * 1024  # 8 GB
    current_bytes = 0
    if selected_hooks:
        accumulated_activations = {hook: [] for hook in selected_hooks}
    else:
        accumulated_activations = {}
    meta_info = []
    for tensor_file in tqdm(os.listdir(input_dir)):
        if file_substring and file_substring not in tensor_file:
            continue
        # measure bytesize of the accumulated activations
        activations = torch.load(f"{input_dir}/{tensor_file}")["activations"]
        meta_info_file = torch.load(f"{input_dir}/{tensor_file}")["meta"]

        if not selected_hooks:
            selected_hooks = list(activations.keys())

        for hook in selected_hooks:
            if subselected_positions:
                new_tensor = activations[hook][:, subselected_positions, :]
            else:
                new_tensor = activations[hook]
            accumulated_activations[hook].append(new_tensor)
            meta_info.append(meta_info_file)
            current_bytes += new_tensor.element_size() * new_tensor.nelement()

        # Check if current bytes exceed the limit
        if current_bytes > max_bytes:
            print(
                f"Warning: Accumulated activations exceed {max_bytes / (1024**3):.2f} GB. Stopping accumulation."
            )
            sys.exit(1)

    for hook in selected_hooks:
        accumulated_activations[hook] = torch.cat(accumulated_activations[hook], dim=0)

    if not merge_meta_batches:
        return accumulated_activations, meta_info

    meta_info_dict = {}
    for meta_info_batch in meta_info:
        for key, value in meta_info_batch.items():
            if isinstance(value, list):
                meta_info_dict[key] = meta_info_dict.get(key, []) + value
            else:
                meta_info_dict[key] = meta_info_dict.get(key, []) + [value]

    # convert back to single value is all values are the same
    for key, value in meta_info_dict.items():
        if len(set(value)) == 1:
            meta_info_dict[key] = value[0]

    return accumulated_activations, meta_info_dict
