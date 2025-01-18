import zlib
import bz2
import lzma


def compression_LZ77_complexity(pattern):
    """Approximate "Kolmogorov complexity" using zlib compression.
    Return number of bytes in the compressed pattern."""
    pattern_str = "".join(map(str, pattern))
    compressed_size = len(zlib.compress(pattern_str.encode("utf-8")))
    return compressed_size


def compression_lzma_complexity(pattern):
    """
    Approximate "Kolmogorov complexity" using lzma compression.
    LZMA uses a combination of dictionary encoding and range encoding,
    making it suitable for highly structured data.
    Return number of bytes in the compressed pattern.
    """
    pattern_str = "".join(map(str, pattern))
    compressed_size = len(lzma.compress(pattern_str.encode("utf-8")))
    return compressed_size


def compression_bz2_complexity(pattern):
    """
    Approximate "Kolmogorov complexity" using bz2 compression.
    This uses the Burrows-Wheeler Transform for better compression
    of data with structured patterns.
    Return number of bytes in the compressed pattern.
    """
    pattern_str = "".join(map(str, pattern))
    compressed_size = len(bz2.compress(pattern_str.encode("utf-8")))
    return compressed_size


def normalize_vocab(data):
    """
    Normalize symbols in the data to a consistent alphabet based on order of appearance.
    """
    unique_symbols = {}
    normalized = []
    current_symbol = 65  # Start with 'A'
    for symbol in data:
        if symbol not in unique_symbols:
            unique_symbols[symbol] = chr(current_symbol)
            current_symbol += 1
        normalized.append(unique_symbols[symbol])
    return "".join(normalized)


def bwt_transform(data):
    """
    Perform the Burrows-Wheeler Transform on the input string.
    """
    n = len(data)
    # Generate all rotations of the input string
    rotations = [data[i:] + data[:i] for i in range(n)]
    # Sort rotations lexicographically
    rotations.sort()
    # Return the last column of the sorted rotations
    return "".join(row[-1] for row in rotations)


def vocabulary_insensitive_bwt(data):
    """
    Perform a vocabulary-insensitive BWT by normalizing the vocabulary first.
    """
    # Normalize the vocabulary
    normalized_data = normalize_vocab(data)
    # Apply standard BWT
    return bwt_transform(normalized_data)


def compression_complexity(pattern: str | list[int | str]) -> dict[str, int]:
    """
    Return a dictionary with the compression sizes for each compression algorithm.
    Compression Size ~ number of bits in the compressed pattern,
    this includes metadata, so compression sizes should not be compared across algorithms.
    """
    compression_sizes = {}
    compression_sizes["LZ77"] = compression_LZ77_complexity(pattern)
    compression_sizes["BZ2"] = compression_bz2_complexity(
        vocabulary_insensitive_bwt(pattern)
    )
    compression_sizes["LZMA"] = compression_lzma_complexity(pattern)
    return compression_sizes


def test_compression_complexity():
    def compression_complexity(pattern):

        compression_sizes = {}
        compression_sizes["LZ77"] = compression_LZ77_complexity(pattern)
        compression_sizes["BZ2"] = compression_bz2_complexity(pattern)
        compression_sizes["BZ2_VOCAB"] = compression_bz2_complexity(
            vocabulary_insensitive_bwt(pattern)
        )
        compression_sizes["LZMA"] = compression_lzma_complexity(pattern)
        return compression_sizes

    truth_table = "11110000"
    truth_table_2 = "11010100"  # More complex

    print(compression_complexity(truth_table))
    print(compression_complexity(truth_table_2))
    print("----" * 10)
    # {'LZ77': 14, 'BZ2': 42, 'LZMA': 64}
    # {'LZ77': 13, 'BZ2': 39, 'LZMA': 64}

    pseudo_truth_table = "++++----"
    pseudo_truth_table_2 = "==$=$=$$"  # More complex

    print(compression_complexity(pseudo_truth_table))
    print(compression_complexity(pseudo_truth_table_2))
    print("----" * 10)
    # {'LZ77': 14, 'BZ2': 42, 'LZMA': 64}
    # {'LZ77': 13, 'BZ2': 41, 'LZMA': 64}
    print(
        "Warning: BZ2 is Vocabulary-Sensitive and can not be used to compare patterns with different Letters!"
    )

    pattern_1 = "+b +b +a -a +b +b +a -a +b +b +a -a +b +b +a -a +b -a -a"
    pattern_2 = "+b +b +b +b +b +b +b +b +b +b +b +b +b +b +b +b +b +a +a"

    print(compression_complexity(pattern_1))
    print(compression_complexity(pattern_2))
    print("----" * 10)
    # {'LZ77': 24, 'BZ2': 51, 'LZMA': 80}
    # {'LZ77': 16, 'BZ2': 46, 'LZMA': 72}


if __name__ == "__main__":
    test_compression_complexity()
