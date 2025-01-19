from typing import Literal
import streamlit as st
import pandas as pd
import torch
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import logging


# Set page configuration
st.set_page_config(
    page_title="Interactive Visualization of Transformer Activations",
    layout="wide",  # Use wide layout to reduce margins
    initial_sidebar_state="expanded",
)


class StreamlitHandler(logging.Handler):
    def __init__(self, num_messages_displayed: int = 3):
        super().__init__()
        self.log_placeholder = st.empty()
        self.logs = []  # Store all log messages
        self.num_messages_displayed = num_messages_displayed

    def emit(self, record):
        log_message = self.format(record)
        self.logs.append(log_message)  # Append new message
        # Display all log messages
        self.log_placeholder.text("\n".join(self.logs[-self.num_messages_displayed :]))


# Configure logging
logger = logging.getLogger("streamlit_logger")
logger.setLevel(logging.DEBUG)
streamlit_handler = StreamlitHandler()
formatter = logging.Formatter("%(levelname)s:  %(message)s")
streamlit_handler.setFormatter(formatter)
logger.addHandler(streamlit_handler)


# Inject CSS to reduce padding and margins
st.markdown(
    """
    <style>
        .css-18e3th9 {padding: 1rem 1rem 1rem 1rem;}  /* Content margins */
        .css-1d391kg {padding: 1rem 1rem 1rem 1rem;}  /* Sidebar margins */
        .css-1v3fvcr {gap: 0rem;}  /* Reduce spacing between widgets */
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def compute_projection(
    processed_data: np.ndarray, projection_method: Literal["PCA", "t-SNE", "UMAP"]
):
    """Perform dimensionality reduction and cache the results."""
    if projection_method == "PCA":
        projector = PCA(n_components=2)
    elif projection_method == "t-SNE":
        projector = TSNE(n_components=2, random_state=42)
    elif projection_method == "UMAP":
        projector = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown projection method: {projection_method}")

    projection = projector.fit_transform(processed_data)
    return projection


class ActivationVisualizer:
    def __init__(
        self,
        activations: dict[str, torch.Tensor] | None = None,
        features_df: pd.DataFrame | None = None,
    ):
        self.activations = activations
        self.features_df = features_df
        self.processed_data = None
        self.projection_df = None
        self.projector = None
        self.projection = None  # Store the projection result

    def select_layer_and_agg(self):
        """Select layer(s) and aggregation method from the sidebar."""
        layer_options = ["All Layers"] + list(self.activations.keys())
        self.layer_choice = st.sidebar.selectbox("Select Layer", layer_options)
        self.agg_choice = st.sidebar.selectbox(
            "Select Aggregation Method", ["mean", "stack", "position"]
        )
        if self.agg_choice == "position":
            max_positions = list(self.activations.values())[0].shape[1]
            self.position_choice = st.sidebar.selectbox(
                "Select Position", range(max_positions)
            )
        else:
            self.position_choice = None

        self.process_activations()

    def process_activations(self):
        """Process activations based on selected layer(s) and aggregation method."""
        if self.agg_choice == "position":
            logger.info(
                f"Processing activations for {self.layer_choice} layer(s) and {self.position_choice} position(s)"
            )
        else:
            logger.info(
                f"Processing activations for {self.layer_choice} layer(s) and {self.agg_choice} aggregation method"
            )
        layer_data: list[np.ndarray] = []
        layer_indices: list[str] = []
        datapoint_indices: list[int] = []
        position_indices: list[int] = []

        # assumming activations for all layers have the same batch size
        batch_indices = np.arange(list(self.activations.values())[0].shape[0])
        if self.layer_choice == "All Layers":
            # Combine activations across all layers
            for layer_name, data in self.activations.items():
                data_np = data.cpu().detach().numpy()
                layer_data.append(data_np)
                layer_indices.extend([layer_name] * data_np.shape[0])
            datapoint_indices = np.tile(batch_indices, len(self.activations))
            layer_data = np.concatenate(layer_data, axis=0)
        else:
            # get activations for the specified layer
            layer_data = self.activations[self.layer_choice].cpu().numpy()
            layer_indices = [self.layer_choice] * layer_data.shape[0]
            datapoint_indices = batch_indices

        if self.agg_choice == "mean":
            self.processed_data = layer_data.mean(axis=1)  # Aggregate across L
            position_indices = np.zeros(len(datapoint_indices))
        elif self.agg_choice == "stack":
            self.processed_data = layer_data.reshape(
                -1, layer_data.shape[-1]
            )  # Stack B * L
            position_indices = np.tile(
                np.arange(layer_data.shape[1]), len(datapoint_indices)
            )
            datapoint_indices = np.repeat(datapoint_indices, layer_data.shape[1])
            layer_indices = np.repeat(layer_indices, layer_data.shape[1])
        elif self.agg_choice == "position":
            self.processed_data = layer_data[
                :, self.position_choice, :
            ]  # Select specific position
            position_indices = np.full(len(datapoint_indices), self.position_choice)

        # Ensure all arrays are the same length before creating DataFrame
        n = len(datapoint_indices)
        assert len(layer_indices) == n
        assert len(datapoint_indices) == n
        assert len(position_indices) == n

        # Construct projection_df with reference to original features_df and metadata
        self.projection_df = pd.DataFrame(
            {
                "Dim1": np.zeros(n),  # Placeholder for projection
                "Dim2": np.zeros(n),  # Placeholder for projection
                "Index": datapoint_indices,
                "Layer": layer_indices,
                "Position": position_indices,
            }
        )

    def perform_projection(self):
        projection_method = st.sidebar.selectbox(
            "Projection Method", ["t-SNE", "PCA", "UMAP"]
        )

        # Compute dim-reduction projection if not already done
        logger.info(
            f"Computing {projection_method} projection for {self.processed_data.shape} data with {self.projection_df.shape} projection_df"
        )
        self.projection = compute_projection(
            processed_data=self.processed_data,
            projection_method=projection_method,
        )

        # Update metadata columns without recomputing projection
        self.projection_df["Dim1"] = self.projection[:, 0]
        self.projection_df["Dim2"] = self.projection[:, 1]

    def add_color_options(self):
        """Add coloring options for the scatter plot."""
        available_colors = []
        if self.layer_choice == "All Layers":
            available_colors.append("Layer")
        if self.agg_choice == "stack":
            available_colors.append("Position")

        default_color = "num_functions"
        if self.layer_choice == "All Layers":
            default_color = "Layer"
        elif self.agg_choice == "stack":
            default_color = "Position"

        feats = list(self.features_df.columns)
        feats.remove("pattern")
        available_colors.extend(feats)
        self.color_by = st.sidebar.selectbox(
            "Color By", available_colors, index=available_colors.index(default_color)
        )

    def filter_categories(self):
        """Filter the dataset based on selected categorical features."""
        category_selectors = st.sidebar.multiselect(
            "Select Category Features", self.features_df.columns
        )
        for selector in category_selectors:
            selected_values = st.sidebar.multiselect(
                f"Select values for {selector}", self.features_df[selector].unique()
            )
            if selected_values:
                self.features_df = self.features_df[
                    self.features_df[selector].isin(selected_values)
                ]

    def plot(self):
        """Plot the data using Plotly."""

        # merge features_df with projection_df
        self.projection_df = pd.merge(
            self.projection_df,
            self.features_df,
            left_on="Index",
            right_index=True,
            how="left",
        )

        # add hover data
        self.projection_df["hover_data"] = self.projection_df.apply(
            lambda row: f"Layer: {row['Layer']}, Position: {row['Position']}", axis=1
        )

        # Determine if the color_by column is categorical (integers or binary)
        if (
            self.projection_df[self.color_by].dtype
            in [np.int64, np.int32, np.int8, bool, str]
            or self.projection_df[self.color_by].nunique() <= 6
        ):
            color_discrete_sequence = px.colors.qualitative.Plotly
        else:
            color_discrete_sequence = None

        fig = px.scatter(
            self.projection_df,
            x="Dim1",
            y="Dim2",
            color=self.color_by,
            color_discrete_sequence=color_discrete_sequence,
            width=1200,
            height=800,
        )
        st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Run the visualization app."""
        st.title("Interactive Visualization of Transformer Activations")
        self.select_layer_and_agg()
        self.filter_categories()
        self.perform_projection()
        self.add_color_options()
        self.plot()


if __name__ == "__main__":
    activations_path = os.getenv("ACTIVATIONS_PATH")
    features_path = os.getenv("FEATURES_PATH")

    if activations_path:
        if not os.path.exists(activations_path):
            raise FileNotFoundError(f"Activations file not found: {activations_path}")
        activations = torch.load(activations_path)
    else:
        raise ValueError("ACTIVATIONS_PATH must be provided")

    if features_path and not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")

    features_df = (
        pd.read_csv(features_path)
        if features_path and os.path.exists(features_path)
        else None
    )

    app = ActivationVisualizer(activations, features_df)
    app.run()
