from transformers import PreTrainedTokenizerFast, PreTrainedTokenizerBase
from transformers import GPT2Config, GPT2LMHeadModel
from datasets import Dataset, load_from_disk
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorMixin
import json
import argparse
from multiprocessing import Pool
from functools import partial
import torch

from geomechinterp.causal.base_functions import ALL_SYMBOLS, EXTRA_SYMBOLS

import spacy
from spacy.language import Language


@Language.component("symbol_tokenizer")
def symbol_tokenizer(doc):
    # Break text into individual symbols
    tokens = [doc.text[i] for i in range(len(doc.text))]
    if not tokens:
        return spacy.tokens.Doc(doc.vocab, words=[], spaces=[])
    # Create SpaCy tokens with spaces after each symbol except last
    spaces = [False] * len(tokens)  # No spaces between symbols
    return spacy.tokens.Doc(doc.vocab, words=tokens, spaces=spaces)


class SymbolTokenizer(PreTrainedTokenizerBase):
    def __init__(
        self,
        vocab: list[str] = ALL_SYMBOLS + EXTRA_SYMBOLS,
        multiprocessing: bool = False,
        max_length: int = 72,
    ):
        self.vocab = vocab
        self.multiprocessing = multiprocessing
        self.max_length = max_length
        self.model_input_names = ["input_ids", "attention_mask"]
        self.skip_special_tokens = False
        super().__init__()
        self.__post_init__()

    def __post_init__(self):
        self.pad_token = "(PAD)"
        self.unk_token = "(UNK)"
        self.bos_token = "(BOS)"
        self.eos_token = "(EOS)"
        self.vocab.append(self.pad_token)
        self.vocab.append(self.unk_token)
        self.vocab.append(self.bos_token)
        self.vocab.append(self.eos_token)
        self.token_to_id = {symbol: idx for idx, symbol in enumerate(self.vocab)}
        self.id_to_token = {idx: symbol for symbol, idx in self.token_to_id.items()}
        self.special_tokens_ids = {
            self.pad_token: self.token_to_id[self.pad_token],
            self.unk_token: self.token_to_id[self.unk_token],
            self.bos_token: self.token_to_id[self.bos_token],
            self.eos_token: self.token_to_id[self.eos_token],
        }
        # Create spacy pipeline
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("symbol_tokenizer", name="symbol_tokenizer", first=True)

    def encode(
        self, text: str | list[str], padding: bool = False
    ) -> list[int] | list[list[int]]:
        # Handle single string case
        if isinstance(text, str):
            return self._encode_single(text, padding, self.max_length)

        # Create partial function with fixed padding and max_length
        encode_fn = partial(
            self._encode_single, padding=padding, max_length=self.max_length
        )

        if self.multiprocessing:
            # Process texts in parallel
            with Pool() as pool:
                results = pool.map(encode_fn, text)
            results = list(results)
        else:
            results = [self._encode_single(t, padding, self.max_length) for t in text]

        return {"input_ids": results}

    def _encode_single(
        self, text: str, padding: bool = False, max_length: int = 64
    ) -> list[int]:
        # Use spacy to tokenize into symbols
        doc = self.nlp(text)
        # Convert symbols to IDs
        token_ids = [
            self.token_to_id.get(token.text, self.token_to_id["(UNK)"]) for token in doc
        ]
        # Apply padding if required
        if padding:
            token_ids += [self.token_to_id["(PAD)"]] * (max_length - len(token_ids))
            token_ids = token_ids[:max_length]  # Truncate if longer
        return token_ids

    def decode(
        self,
        token_ids: list[int] | list[list[int]],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> str | list[str]:
        # NOTE: kwargs are ignored, mimicking HF tokenizer
        self.skip_special_tokens = skip_special_tokens
        # Handle single sequence case
        if isinstance(token_ids[0], int):
            return self._decode_single(token_ids)

        if self.multiprocessing:
            with Pool() as pool:
                results = pool.map(self._decode_single, token_ids)
        else:
            results = [self._decode_single(t) for t in token_ids]

        return results

    def _decode_single(self, token_ids: list[int]) -> str:
        # Convert IDs back to symbols and join
        if self.skip_special_tokens:
            token_ids = [
                idx for idx in token_ids if idx not in self.special_tokens_ids.values()
            ]
        return "".join(self.id_to_token.get(idx, "[UNK]") for idx in token_ids)

    def __call__(
        self, text: str | list[str], padding: bool = False
    ) -> list[int] | list[list[int]]:
        return self.encode(text, padding=padding)

    def pad(self, features, padding=True, max_length=None, **kwargs):
        """Add padding method to make compatible with HuggingFace's DataCollator"""
        if not padding:
            return features

        if max_length is None:
            max_length = max(len(x["input_ids"]) for x in features)

        for feature in features:
            input_ids = feature["input_ids"]
            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                input_ids.extend([self.token_to_id["(PAD)"]] * pad_length)

        return features


class DataCollator(DataCollatorMixin):
    def __init__(self, dataset, device: str | None = None):
        self.dataset = dataset
        self.device = device
        super().__init__()

    def __call__(self, features):
        batch = []
        for feature in features["input_ids"]:
            batch.append(feature)
        batch = {"input_ids": torch.tensor(batch), "labels": torch.tensor(batch)}
        if self.device:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch


# Define a simple vocabulary (you can extend it as needed)
custom_vocab = ALL_SYMBOLS + EXTRA_SYMBOLS

# Create the tokenizer
tokenizer = SymbolTokenizer(custom_vocab)


def load_dataset(data_or_path):
    """Load and format the dataset."""
    if isinstance(data_or_path, str):
        data = json.load(open(data_or_path))
    elif isinstance(data_or_path, dict):
        data = data_or_path
    else:
        raise ValueError(f"Invalid data type: {type(data_or_path)}")
    data = [
        {
            "text": k,  # pattern
            "generator": v["generator"],
            "stochastic": v["stochastic"],
            "sequence_of_features": v["sequence_of_features"],
        }
        for k, v in data.items()
    ]
    return Dataset.from_list(data)


def create_tokenize_function(tokenizer):
    """Create tokenization function."""

    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], padding=True)
        return tokens

    return tokenize_function


def get_model_config(vocab_size):
    """Create GPT model configuration."""
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=128,
        n_layer=6,
        n_head=4,
        use_cache=True,
        rotary_dim=64,
        return_dict=True,
    )


def get_training_args(output_dir, epochs=20, batch_size=8):
    """Configure training arguments."""
    return TrainingArguments(
        do_eval=True,
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        weight_decay=0.01,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        do_train=True,
    )


def train_model(data_path, tokenizer_path, model_path, epochs=3, batch_size=8):
    """Main training function."""
    # Create vocabulary and tokenizer
    custom_vocab = ALL_SYMBOLS + EXTRA_SYMBOLS
    tokenizer = SymbolTokenizer(custom_vocab)

    # Load and process dataset
    dataset = load_dataset(data_path)

    # First split into train and test+val
    first_split = dataset.train_test_split(test_size=0.15)
    train_dataset = first_split["train"]

    # Split test+val into test and val
    test_val_split = first_split["test"].train_test_split(test_size=0.33)
    val_dataset = test_val_split["train"]  # ~10% of total
    test_dataset = test_val_split["test"]  # ~5% of total

    # Tokenize datasets
    tokenize_func = create_tokenize_function(tokenizer)
    tokenized_train = train_dataset.map(tokenize_func, batched=True)
    tokenized_val = val_dataset.map(tokenize_func, batched=True)
    tokenized_test = test_dataset.map(tokenize_func, batched=True)

    # Save tokenized datasets
    tokenized_train.save_to_disk("tokenized_train")
    tokenized_val.save_to_disk("tokenized_val")
    tokenized_test.save_to_disk("tokenized_test")

    # Load tokenized datasets
    tokenized_dataset = {
        "train": load_from_disk("tokenized_train"),
        "validation": load_from_disk("tokenized_val"),
        "test": load_from_disk("tokenized_test"),
    }

    # Initialize model
    config = get_model_config(len(custom_vocab))
    model = GPT2LMHeadModel(config)

    # Setup training
    data_collator = DataCollator(tokenized_dataset)

    training_args = get_training_args(model_path, epochs, batch_size)

    # Create and run trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    trainer.train()


def test_tokenizer(data_path):
    """Test the tokenizer."""
    # Create vocabulary and tokenizer
    custom_vocab = ALL_SYMBOLS + EXTRA_SYMBOLS
    tokenizer = SymbolTokenizer(custom_vocab)
    # Load and process dataset
    dataset = load_dataset(data_path)

    # Split into train/test
    split_dataset = dataset.train_test_split(test_size=0.15)

    # Tokenize datasets
    tokenize_func = create_tokenize_function(tokenizer)
    tokenized_train = split_dataset["train"].map(tokenize_func, batched=True)
    tokenized_test = split_dataset["test"].map(tokenize_func, batched=True)

    # Save tokenized datasets
    tokenized_train.save_to_disk("tokenized_train")
    tokenized_test.save_to_disk("tokenized_test")

    # Load tokenized datasets
    tokenized_dataset = {
        "train": load_from_disk("tokenized_train"),
        "test": load_from_disk("tokenized_test"),
    }

    # Setup training
    data_collator = DataCollator(tokenized_dataset)

    for i in range(len(tokenized_dataset)):
        print(tokenizer.decode(data_collator(i)))


def main():
    parser = argparse.ArgumentParser(description="Train a custom GPT model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/all_patterns_and_generators_final.json",
        help="Path to training data JSON file",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="./custom_tokenizer",
        help="Path to save tokenizer",
    )
    parser.add_argument(
        "--model-path", type=str, default="./gpt_rope_custom", help="Path to save model"
    )
    parser.add_argument(
        "--epochs", type=int, default=16, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")

    args = parser.parse_args()

    train_model(
        args.data_path,
        args.tokenizer_path,
        args.model_path,
        args.epochs,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
