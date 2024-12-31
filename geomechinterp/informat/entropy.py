from typing import Tuple

import numpy as np
import torch


def estimate_entropy_char(text: str) -> float:
    """
    Returns the entropy content of a string calculated from character frequencies.
    Note: spaces are removed from the text before calculating entropy.
    """
    # Count character frequencies
    char_counts = {}
    # remove spaces
    text = text.replace(" ", "")
    total_chars = len(text)

    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1

    # Calculate entropy using Shannon's formula
    entropy = 0
    for count in char_counts.values():
        prob = count / total_chars
        entropy -= prob * np.log2(prob)

    return entropy


def estimate_entropy_token(text: str) -> float:
    """
    Returns the entropy content of a string calculated from token frequencies.
    """
    # Count token frequencies
    token_counts = {}
    tokens = text.split()
    total_tokens = len(tokens)

    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    entropy = 0
    for count in token_counts.values():
        prob = count / total_tokens
        entropy -= prob * np.log2(prob)

    return entropy


def get_uncertainty(logits: torch.Tensor) -> Tuple[list[float], list[float], list[int]]:
    """
    Returns the entropy and the maximum probability of the logits.
    """
    softmax_logits = torch.nn.functional.softmax(logits, dim=-1)
    entropy = -torch.sum(softmax_logits * torch.log(softmax_logits + 1e-8), dim=-1)
    entropy = entropy.cpu().detach().numpy()
    max_prob, max_indices = softmax_logits.max(dim=-1)
    max_prob = max_prob.cpu().detach().numpy()
    max_indices = max_indices.cpu().detach().numpy()
    return entropy, max_prob, max_indices
