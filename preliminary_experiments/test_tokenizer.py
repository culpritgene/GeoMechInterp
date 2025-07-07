import sentencepiece

# Test sentencepiece in isolation
import tempfile
import os

# Create a temporary file with some sample text
with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
    f.write("This is a test sentence.\nAnother test sentence.")
    temp_filename = f.name

# Train a simple sentencepiece model
model_prefix = "test_sp"
sentencepiece.SentencePieceTrainer.train(
    f"--input={temp_filename} --model_prefix={model_prefix} "
    "--vocab_size=19 --character_coverage=1.0"
)

# Load and test the model
sp = sentencepiece.SentencePieceProcessor()
sp.load(f"{model_prefix}.model")

# Test encoding/decoding
test_text = "This is a test."
encoded = sp.encode_as_pieces(test_text)
decoded = sp.decode_pieces(encoded)

print("Original:", test_text)
print("Encoded:", encoded)
print("Decoded:", decoded)

# Clean up temporary files
os.unlink(temp_filename)
os.unlink(f"{model_prefix}.model")
os.unlink(f"{model_prefix}.vocab")


from transformers import PreTrainedTokenizerFast

# Define your custom vocabulary
custom_vocab = ["a", "b", "c", " ", ".", ","]  # Add all symbols you need
custom_vocab += ["[PAD]", "[UNK]"]  # Special tokens

# Create a PreTrainedTokenizerFast using the custom vocabulary
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=None,  # No internal tokenization logic
    vocab={symbol: idx for idx, symbol in enumerate(custom_vocab)},
)

# Set special tokens
tokenizer.pad_token = "[PAD]"
tokenizer.unk_token = "[UNK]"

# Test the tokenizer
print(tokenizer.encode("abc a,."))  # Example encoding
print(tokenizer.decode([0, 1, 2, 3, 4, 5]))  # Example decoding
