import random
from collections import defaultdict, Counter


def train_char_ngram_model(text, n=3):
    """
    Train a char-level n-gram model.
    Returns a dict: model[context] = Counter({next_char: count, ...})
    where 'context' is a tuple of length (n-1) of characters.
    """
    # We'll keep the text as-is (not lowercased, not tokenized by words),
    # but you can tweak as needed.
    # Optionally add special start/end tokens if you like.
    data = list(text)  # list of single characters
    model = defaultdict(Counter)

    # Build counts
    for i in range(len(data) - n + 1):
        context = tuple(data[i : i + n - 1])
        next_char = data[i + n - 1]
        model[context][next_char] += 1

    # Convert counts to probabilities
    for context in model:
        total = float(sum(model[context].values()))
        for ch in model[context]:
            model[context][ch] /= total

    return model


def get_char_distribution(model, context):
    """
    Given a trained n-gram model and a context (tuple of length n-1),
    returns a dict of {char: probability} for the next character.
    If context not found in model, returns an empty dict.
    """
    return dict(model.get(context, {}))


def combine_distributions(dists, weights=None):
    """
    Combine multiple distributions (list of dicts) into a single distribution
    by weighted sum of probabilities. dists[i] is e.g. {char: prob, ...}.
    'weights' must be a list of floats (same length as dists) or None
    (equal weighting).
    Returns a normalized dict {char: combined_prob, ...}.
    """
    if weights is None:
        weights = [1.0] * len(dists)

    combined = Counter()
    for dist, w in zip(dists, weights):
        for char, p in dist.items():
            combined[char] += w * p

    total_prob = sum(combined.values())
    if total_prob > 0:
        for char in combined:
            combined[char] /= total_prob

    return dict(combined)


def generate_from_mixture(models, n_vals, start_text, max_len=50, weights=None):
    """
    Generate text from a mixture of multiple n-gram models:
    - models: list of n-gram models
    - n_vals: list of the 'n' for each model (e.g. [1,2,3])
    - start_text: string to seed generation
    - max_len: max characters to generate
    - weights: optional weighting for each model
    Returns:
        (generated_text, model_usage_counts)
      where model_usage_counts is a dict {n: count_of_times_it_had_max_prob}
    """
    # We'll maintain the generation in a list for easy appending
    generated = list(start_text)

    # Keep track of which model "wins" for each new char
    usage_counts = {n_val: 0 for n_val in n_vals}

    for _ in range(max_len):
        # For each model, we find context=(last n-1 chars) -> distribution
        # Then we combine them
        all_distributions = []
        for model, n in zip(models, n_vals):
            context_size = n - 1
            if len(generated) < context_size:
                # not enough chars for this model's context
                dist = {}
            else:
                context = tuple(generated[-context_size:])
                dist = get_char_distribution(model, context)
            all_distributions.append(dist)

        # Combine into one distribution
        combined_dist = combine_distributions(all_distributions, weights)

        if not combined_dist:
            # no valid next char (all distributions empty) => stop
            break

        # Randomly choose next_char from combined distribution
        chars = list(combined_dist.keys())
        probs = list(combined_dist.values())
        next_char = random.choices(chars, weights=probs, k=1)[0]
        generated.append(next_char)

        # Determine which model assigned the highest probability to next_char
        best_n = None
        best_prob = -1.0
        for dist, n_val in zip(all_distributions, n_vals):
            p = dist.get(next_char, 0.0)
            if p > best_prob:
                best_prob = p
                best_n = n_val
        if best_n is not None:
            usage_counts[best_n] += 1

    return "".join(generated), usage_counts


if __name__ == "__main__":
    # Example text
    sample_text = (
        "Hello, this is a sample text for demonstrating a mixture of n-gram models!"
        " We'll try character-level n-grams of different sizes."
    )

    # Train 1-gram, 2-gram, 3-gram models
    model1 = train_char_ngram_model(sample_text, n=1)  # 1-gram
    model2 = train_char_ngram_model(sample_text, n=2)  # 2-gram
    model3 = train_char_ngram_model(sample_text, n=3)  # 3-gram

    models = [model1, model2, model3]
    n_vals = [1, 2, 3]

    # Start text (seed)
    start_seed = "He"

    # Optional weighting for each model, e.g. (1,2,3)
    # If None, it's a simple average
    weights = None # [0.5, 1.0, 1.5]

    # Generate text
    generated_text, usage_count = generate_from_mixture(
        models, n_vals, start_seed, max_len=100, weights=weights
    )

    print("Generated Text:")
    print(generated_text)
    print("\nWhich n-gram model contributed more often to final picks?")
    for n_val, cnt in usage_count.items():
        print(f"  n={n_val}: {cnt} times")
