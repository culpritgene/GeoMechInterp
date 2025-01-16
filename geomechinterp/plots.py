from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

from geomechinterp.informat.entropy import get_uncertainty


def plot_pca_reps(concept_dirs, labels, title="PCA Projection of Concept Vectors"):
    pca = PCA(n_components=2)
    pca_matrix = pca.fit_transform(concept_dirs)

    # Create a DataFrame for Plotly
    df = pd.DataFrame(
        {
            "x": np.zeros(len(pca_matrix)),  # Start point of vectors at origin (0, 0)
            "y": np.zeros(len(pca_matrix)),
            "x_end": pca_matrix[:, 0],  # End point x-coordinates
            "y_end": pca_matrix[:, 1],  # End point y-coordinates
            "label": labels,
        }
    )

    # Generate a scatter plot with vectors
    fig = px.scatter(df, x="x_end", y="y_end", text="label")

    # Add vectors from origin to the points
    for i in range(len(df)):
        fig.add_trace(
            go.Scatter(
                x=[0, df.loc[i, "x_end"]],  # X-coordinates
                y=[0, df.loc[i, "y_end"]],  # Y-coordinates
                mode="lines+markers",
                marker=dict(color="black"),
                line=dict(color="black"),
                showlegend=False,
            )
        )

    # Add vector labels
    fig.update_traces(textposition="top right")

    # Customize the layout for better visibility
    fig.update_layout(
        title=title,
        xaxis_title="PC1",
        yaxis_title="PC2",
        showlegend=False,
        width=800,
        height=600,
    )

    fig.show()


def random_histogram(
    random_directions: torch.Tensor,
    concept_direction: torch.Tensor,
    inner_prods_loo: torch.Tensor,
    title: str,
):
    target = inner_prods_loo
    baseline = random_directions @ concept_direction
    fig = plt.figure()
    plt.hist(
        baseline.cpu().numpy(),
        bins=50,
        alpha=0.6,
        color="blue",
        label="random pairs",
        density=True,
    )
    plt.hist(
        target.cpu().numpy(),
        alpha=0.7,
        color="red",
        label="counterfactual pairs",
        density=True,
    )

    plt.title(title)
    plt.xlabel("Inner Product")
    plt.ylabel("Freq")
    plt.tight_layout()
    return fig


def plot_word_change(beta_values, words, runs, labels, figsize=(30, 10)):
    """
    Plots horizontal bar plots stacked vertically showing word changes as beta varies.
    """
    xlim0 = -45
    xlim1 = 45
    fig, axs = plt.subplots(runs, 1, figsize=figsize, sharex=True)
    for i in range(runs):
        ax = axs[i]

        # Plot each word's section as a bar
        for j in range(len(beta_values[i]) - 1):
            start, end = beta_values[i][j], beta_values[i][j + 1]
            word = words[i][j]
            if word.strip() in labels[i]:
                color = "green"
            else:
                color = "royalblue"
            ax.barh(
                i, end - start, left=start, height=0.2, color=color, edgecolor="black"
            )
            ax.text(
                (max(start, xlim0) + min(end, xlim1)) / 2,
                i,
                word,
                ha="center",
                va="center",
                fontsize=15,
                color="white",
            )
        ax.barh(i, 10, left=xlim1, height=0.2, color="white", edgecolor="black")
        ax.text(
            xlim1 + 3,
            i,
            labels[i],
            ha="center",
            va="center",
            fontsize=18,
            color="black",
        )

        # Set y-axis to hide labels, only showing word bars
        ax.set_yticks([])
        ax.set_xlim(xlim0, xlim1)
    axs[0].set_title(
        "Adding Size-Concept vector multiplied by Beta to different initial words (shown on the right for each bar)",
        size=20,
    )
    plt.xlabel("Beta")
    plt.tight_layout()
    plt.show()


def plot_uncertainty(
    logits, model, true_tokens: List[str] = None, annotate_tokens: bool = False
):
    entropy, max_prob, max_indices = get_uncertainty(logits)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    if annotate_tokens:
        predicted_tokens = model.to_str_tokens(max_indices)

    axs[0].plot(entropy, color="blue")
    axs[0].set_title("Entropy")
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Entropy Value")

    axs[1].plot(max_prob, color="orange")
    axs[1].set_title("Max Probability")
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("Max Probability Value")
    axs[1].set_ylim(0, 1.2)  # Extend y-axis to 1.2

    # Annotate with tokens at each step
    if annotate_tokens:
        for i, token in enumerate(predicted_tokens):
            # Add predicted tokens
            axs[1].annotate(
                token,
                (i, max_prob[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

            # Add true tokens at y=1.1 if available
            if true_tokens and i < len(true_tokens):
                # Add true token annotation
                axs[1].annotate(
                    true_tokens[i],
                    (i, 1.1),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )
                # Draw dashed line connecting true and predicted tokens
                axs[1].plot(
                    [i, i], [max_prob[i], 1.1], color="red", linestyle="--", alpha=0.3
                )

    plt.tight_layout()
    plt.show()


def plot_pca_activations(
    pca_result: np.ndarray,
    clusters: np.ndarray | None = None,
    title: str = "PCA of Accumulated Activations",
):
    # get colors for arbitrary number of clusters
    cluster_to_color = {i: f"C{i}" for i in range(len(np.unique(clusters)))}
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=[cluster_to_color[c] for c in clusters] if clusters is not None else None,
        alpha=0.5,
    )
    if clusters is not None:
        plt.colorbar(scatter, label="Clustering Colors")
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()


def plot_tsne_activations(
    tsne_result: np.ndarray,
    clusters: np.ndarray | None = None,
    title: str = "t-SNE of Accumulated Activations",
):
    # get colors for arbitrary number of clusters
    cluster_to_color = {i: f"C{i}" for i in range(len(np.unique(clusters)))}
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        tsne_result[:, 0],
        tsne_result[:, 1],
        c=[cluster_to_color[c] for c in clusters] if clusters is not None else None,
        alpha=0.5,
    )
    if clusters is not None:
        plt.colorbar(scatter, label="Clustering Colors")
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()
