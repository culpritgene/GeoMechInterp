import torch
import pandas as pd
from geomechinterp.viz.plots import plot_uncertainty


def run_model_on_pattern(model, pattern, exclude_first_k: int = 0, device: str = "mps"):
    token_ids = model.to_tokens(pattern).to(device)  # Convert tokens to token ids
    token_ids = token_ids[:, :512]  # Limit the sequence length if necessary

    # Forward pass through the model
    logits = model(token_ids, return_type="logits")
    # compute cross-entropy loss
    loss = torch.nn.functional.cross_entropy(
        logits[0, exclude_first_k:, :], token_ids[0, exclude_first_k:]
    )
    return loss.item()


def run_model_on_pattern_and_plot(
    model, pattern, annotate_tokens: str = False, device: str = "mps"
):
    token_ids = model.to_tokens(pattern).to(device)  # Convert tokens to token ids
    token_ids = token_ids[:, :512]  # Limit the sequence length if necessary

    # Forward pass through the model
    logits = model(token_ids, return_type="logits")

    print(model.to_str_tokens(logits[0, :, :].argmax(dim=-1)))
    # print(logits.var(dim=-1))
    if annotate_tokens:
        plot_uncertainty(
            logits[0, :, :],
            model,
            true_tokens=model.to_str_tokens(token_ids[0, 1:]),
            annotate_tokens=annotate_tokens,
        )
    else:
        plot_uncertainty(logits[0, :, :])
    return logits


def filter_categories(df, column, threshold):
    """
    Filters categories in a column based on a threshold.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): The column to filter categories.
    - threshold (int, float, or None):
        - If int, keeps categories with at least `threshold` occurrences.
        - If float, keeps categories above the `threshold` quantile of occurrences.
        - If None, no filtering is applied.
    """
    if threshold is None:
        return df

    value_counts = df[column].value_counts()

    if isinstance(threshold, int):
        valid_categories = value_counts[value_counts >= threshold].index
    elif isinstance(threshold, float):
        cutoff = value_counts.quantile(threshold)
        valid_categories = value_counts[value_counts >= cutoff].index
    else:
        raise ValueError("Threshold must be an int, float, or None")

    df.loc[~df[column].isin(valid_categories), column] = "other"
    return df


def one_hot_encode_columns(df, columns: list[str], rename_dummies=False):
    """
    One-hot encode the specified categorical columns after filtering categories and append them to the original DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list of str): List of column names to one-hot encode.
    - rename_dummies (bool): If True, renames dummy columns to `colname_idx`,
         where idx 0 is for the most frequent category.

    Returns:
    - pd.DataFrame: The original DataFrame with one-hot encoded columns appended.
    """
    for column in columns:
        # One-hot encode the column
        encoded = pd.get_dummies(df[column], prefix=column)

        if rename_dummies:
            # Sort categories by frequency and map to idx
            category_order = df[column].value_counts().index.tolist()
            rename_map = {
                f"{column}_{cat}": f"{column}_{idx}"
                for idx, cat in enumerate(category_order)
            }
            encoded = encoded.rename(columns=rename_map)

        # Append to the DataFrame
        df = pd.concat([df, encoded], axis=1)

    # Drop the original columns
    df = df.drop(columns, axis=1)
    # df.drop(["other"], axis=1, inplace=True)
    return df
