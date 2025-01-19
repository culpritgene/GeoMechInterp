import torch
import numpy as np


def calculate_gini(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a list or array of values.

    Args:
        values (array-like): A list or numpy array of non-negative values.

    Returns:
        float: The Gini coefficient, ranging from 0 (perfect equality) to 1 (maximum inequality).
    """
    values = np.array(values)
    if np.any(values < 0):
        raise ValueError("Values should be non-negative.")

    # Sort the values
    sorted_values = np.sort(values)
    n = len(values)
    # Gini coefficient formula
    cumulative_sum = np.cumsum(sorted_values, dtype=float)
    return (2.0 * np.sum((np.arange(1, n + 1) * sorted_values))) / (
        n * cumulative_sum[-1]
    ) - (n + 1) / n


def singular_value_concentration(
    tensor: torch.Tensor | np.ndarray,
    metrics=("spectral_ratio", "gini", "frobenius_ratio", "explained_variance"),
    explained_variance_threshold=0.95,
    use_torch_svd_lowrank: bool = False,
    return_singular_values: bool = False,
) -> dict[str, float | int]:
    """
    Compute the distribution of singular values and estimate their concentration.

    Args:
        tensor (torch.Tensor): Input tensor of shape (m, n).
        metrics (tuple): Metrics to compute concentration. Options:
                         - "spectral_ratio": Ratio of the largest singular value to the sum.
                         - "frobenius_ratio": Ratio of spectral norm to Frobenius norm.
                         - "gini": Gini coefficient of singular values.
                         - "explained_variance": Variance explained by each singular value.

    Returns:
        dict: A dictionary containing singular value metrics, explained variance, and the singular values.
    """
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D.")

    # Compute singular values
    if isinstance(tensor, torch.Tensor):
        if use_torch_svd_lowrank:
            u, s, v = torch.svd_lowrank(tensor)
        else:
            u, s, v = torch.svd(tensor)
        singular_values = s.cpu().numpy()
    else:
        singular_values = np.linalg.svd(tensor, compute_uv=False)

    # Metrics
    results = {}
    if return_singular_values:
        results["singular_values"] = singular_values

    if "spectral_ratio" in metrics:
        spectral_ratio = singular_values[0] / singular_values.sum()
        results["spectral_ratio"] = spectral_ratio

    if "frobenius_ratio" in metrics:
        frobenius_norm = np.sqrt((singular_values**2).sum())
        spectral_norm = singular_values[0]
        frobenius_ratio = spectral_norm / frobenius_norm
        results["frobenius_ratio"] = frobenius_ratio

    if "gini" in metrics:
        # Gini coefficient for singular values
        gini_coeff = calculate_gini(singular_values)
        results["singular_values_gini"] = gini_coeff

    if "explained_variance" in metrics:
        # Explained variance: Proportion of variance explained by each singular value
        total_variance = (singular_values**2).sum()
        explained_variance = (singular_values**2) / total_variance
        explained_variance = np.cumsum(explained_variance)
        explained_variance_threshold_index = np.where(
            explained_variance > explained_variance_threshold
        )[0][0]
        results[f"explained_variance_{explained_variance_threshold}"] = (
            explained_variance_threshold_index + 1
        )

    return results
