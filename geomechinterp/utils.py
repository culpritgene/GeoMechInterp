import matplotlib.pyplot as plt
import networkx as nx
import torch
import pandas as pd

from geomechinterp.viz.plots import plot_uncertainty


def visualize_dict_structure_tree(
    d, graph=None, parent="Root", level=0, max_depth=None
):
    """
    Recursively adds dictionary keys to the graph for visualization in a tree layout.
    Adds an option to cut off visualization at a certain depth (max_depth).
    """
    if graph is None:
        graph = nx.DiGraph()  # Directed graph for hierarchy

    # If max_depth is specified, stop adding nodes beyond this level
    if max_depth is not None and level > max_depth:
        return graph

    for key, value in d.items():
        node_id = key  # Use the key directly as the node id (no depth in name)

        # Connect the node to its parent (Root for the first level)
        graph.add_edge(parent, node_id)

        # If the value is a dictionary, recursively visualize its structure
        if isinstance(value, dict):
            visualize_dict_structure_tree(
                value, graph=graph, parent=node_id, level=level + 1, max_depth=max_depth
            )
        else:
            # Add leaf node for non-dictionary value
            value_node_id = f"{key}_value"
            graph.add_edge(node_id, value_node_id)

    return graph


def calculate_node_depths(graph, root="Root"):
    """
    Calculates the depth of each node in the graph based on its distance from the root.
    """
    return dict(nx.single_source_shortest_path_length(graph, root))


def display_graph_tree(graph, k=0.5, max_depth=None):
    """
    Display the dictionary structure graph using matplotlib in a tree-like layout.
    Adds stronger repulsion (k) to avoid node occlusion and keeps the depth in check.
    Colors nodes by depth and decreases node size with each level.
    """
    pos = nx.spring_layout(graph, k=k, seed=42)  # Spring layout with more repulsion

    # Calculate the depth of each node
    node_depths = calculate_node_depths(graph)

    # Set node sizes and colors based on depth
    max_level = max(node_depths.values())
    node_sizes = [
        2000 * (0.8 ** node_depths[node]) for node in graph.nodes()
    ]  # Decrease size by depth
    node_colors = [
        plt.cm.viridis(node_depths[node] / max_level) for node in graph.nodes()
    ]  # Color by depth

    plt.figure(figsize=(12, 8))

    # Draw nodes, edges, and labels
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        font_size=10,
        font_weight="bold",
        font_family="Arial",
        edge_color="gray",
        width=2,
        alpha=0.9,
        linewidths=1,
        arrows=False,
    )

    # Add a title
    title = "Nested Dictionary Visualization"
    if max_depth is not None:
        title += f" (Max Depth: {max_depth})"
    plt.title(title, fontsize=15, fontweight="bold")

    # Show the final visualization
    plt.show()


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


def one_hot_encode_columns(
    df, columns: list[str], threshold: int | float | None, rename_dummies=False
):
    """
    One-hot encode the specified categorical columns after filtering categories and append them to the original DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list of str): List of column names to one-hot encode.
    - threshold (int | float | None): A threshold for filtering categories.
    - rename_dummies (bool): If True, renames dummy columns to `colname_idx`,
         where idx 0 is for the most frequent category.

    Returns:
    - pd.DataFrame: The original DataFrame with one-hot encoded columns appended.
    """
    for column in columns:
        # threshold = thresholds.get(column, None)
        df = filter_categories(df, column, threshold)

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


if __name__ == "__main__":
    # Example usage
    nested_dict = {
        "key1": {
            "subkey1": {"subsubkey1": "value1", "subsubkey2": "value2"},
            "subkey2": "value3",
        },
        "key2": "value4",
        "key3": {"subkey3": "value5"},
    }

    # Create the graph with the root node and cutoff at max depth 2
    graph = visualize_dict_structure_tree(nested_dict, max_depth=2)

    # Display the tree layout with stronger node repulsion, coloring by depth, and variable node size
    display_graph_tree(graph, k=0.8, max_depth=2)
