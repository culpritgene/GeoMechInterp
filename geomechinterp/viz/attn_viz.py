from typing import Literal, Optional, Tuple
import streamlit as st
import pandas as pd
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import logging
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Reuse the StreamlitHandler from the original code
# ... existing StreamlitHandler and logging setup code ...


class AttentionVisualizer:
    def __init__(
        self,
        attention_matrices: (
            dict[str, torch.Tensor] | None
        ) = None,  # Shape: [B, H, L, L] per layer
        features_df: pd.DataFrame | None = None,
    ):
        self.attention_matrices = attention_matrices
        self.features_df = features_df
        self.processed_data = None
        self.projection_df = None
        self.viz_mode = None
        self.selected_samples = None
        self.filtered_indices = None  # Track filtered indices

    def run(self):
        """Main execution flow of the visualization app."""
        st.title("Interactive Visualization of Transformer Attention")

        # Top-level visualization mode selection
        self.viz_mode = st.radio(
            "Visualization Mode",
            ["Dimensionality Reduction", "Attention Heatmaps", "Attention Graphs"],
        )

        # Common filtering options
        self.setup_data_filtering()

        # Branch based on visualization mode
        if self.viz_mode == "Dimensionality Reduction":
            self.run_dim_reduction_viz()
        elif self.viz_mode == "Attention Heatmaps":
            self.run_heatmap_viz()
        else:  # Attention Graphs
            self.run_graph_viz()

    def setup_data_filtering(self):
        """Setup common data filtering options in sidebar."""
        st.sidebar.header("Data Selection")

        # Layer selection
        layer_options = ["All Layers"] + list(self.attention_matrices.keys())
        self.layer_choice = st.sidebar.selectbox("Select Layer", layer_options)

        # Aggregation method
        self.agg_choice = st.sidebar.selectbox(
            "Aggregation Method", ["mean", "max", "single sample", "random subset"]
        )

        # Sample selection if applicable
        if self.agg_choice in ["single sample", "random subset"]:
            self.setup_sample_selection()

        # Category filtering
        self.filter_categories()

    def filter_categories(self):
        """Filter the dataset based on selected categorical features."""
        st.sidebar.subheader("Category Filtering")

        # Start with all indices
        self.filtered_indices = np.arange(len(self.features_df))

        # Select features for filtering
        category_selectors = st.sidebar.multiselect(
            "Select Category Features", self.features_df.columns
        )

        # Apply filters
        for selector in category_selectors:
            selected_values = st.sidebar.multiselect(
                f"Select values for {selector}", self.features_df[selector].unique()
            )
            if selected_values:
                mask = self.features_df[selector].isin(selected_values)
                self.filtered_indices = self.filtered_indices[
                    mask.values[self.filtered_indices]
                ]

    def setup_sample_selection(self):
        """Setup sample selection based on filtered features_df."""
        if not hasattr(self, "filtered_indices") or self.filtered_indices is None:
            self.filter_categories()

        n_filtered = len(self.filtered_indices)

        if self.agg_choice == "random subset":
            self.sample_size = st.sidebar.slider(
                "Number of samples",
                min_value=1,
                max_value=n_filtered,
                value=min(10, n_filtered),
            )
            # Randomly sample from filtered indices
            self.selected_samples = np.random.choice(
                self.filtered_indices, size=self.sample_size, replace=False
            )
        elif self.agg_choice == "single sample":
            # Allow selection of a single sample by feature values
            sample_selector = st.sidebar.selectbox(
                "Select sample by index", self.filtered_indices
            )
            self.selected_samples = np.array([sample_selector])

            # Display sample features
            if sample_selector is not None:
                st.sidebar.write("Sample features:")
                st.sidebar.write(self.features_df.iloc[sample_selector])

    def process_attention_data(self):
        """Process attention matrices based on selection and aggregation."""
        logger.info(f"Processing attention data with {self.agg_choice} aggregation")

        # Get data for selected layer(s)
        if self.layer_choice == "All Layers":
            layer_data = {name: data for name, data in self.attention_matrices.items()}
        else:
            layer_data = {self.layer_choice: self.attention_matrices[self.layer_choice]}

        # Apply sample selection
        if self.agg_choice in ["single sample", "random subset"]:
            processed_data = {
                name: data[self.selected_samples] for name, data in layer_data.items()
            }
        else:
            processed_data = {
                name: data[self.filtered_indices] for name, data in layer_data.items()
            }

        # Apply aggregation
        if self.agg_choice == "mean":
            self.processed_data = {
                name: data.mean(dim=0) for name, data in processed_data.items()
            }
        elif self.agg_choice == "max":
            self.processed_data = {
                name: data.max(dim=0)[0] for name, data in processed_data.items()
            }
        else:
            self.processed_data = processed_data

        logger.info(
            f"Processed attention data shape: {[v.shape for v in self.processed_data.values()]}"
        )

    def run_dim_reduction_viz(self):
        """Handle dimensionality reduction visualization pipeline."""
        self.process_attention_for_dimred()
        self.perform_projection()
        self.plot_projection()

    def run_heatmap_viz(self):
        """Handle heatmap visualization pipeline."""
        self.process_attention_for_heatmap()
        self.plot_attention_heatmaps()

    def run_graph_viz(self):
        """Handle graph visualization pipeline."""
        self.process_attention_for_graph()
        self.plot_attention_graphs()

    def process_attention_for_dimred(self) -> None:
        """Process attention matrices for dimensionality reduction visualization."""
        self.process_attention_data()

        st.sidebar.subheader("Dimensionality Reduction Settings")

        # Analysis level selection
        self.analysis_level = st.sidebar.selectbox(
            "Analysis Level",
            ["all_layers", "per_layer", "per_head", "per_query"],
            help="Level at which to analyze attention patterns",
        )

        # Reshape attention matrices based on analysis level
        if self.analysis_level == "all_layers":
            self.prepare_all_layers_data()
        elif self.analysis_level == "per_layer":
            self.prepare_per_layer_data()
        elif self.analysis_level == "per_head":
            self.prepare_per_head_data()
        else:  # per_query
            self.prepare_per_query_data()

    def prepare_all_layers_data(self) -> None:
        """Prepare data for all-layers analysis."""
        data_list = []
        metadata = []

        for layer_name, attn_data in self.processed_data.items():
            # attn_data shape: [batch/1, heads, seq_len, seq_len]
            batch_size = attn_data.shape[0]
            n_heads = attn_data.shape[1]

            # Reshape to [batch * heads, seq_len * seq_len]
            flat_data = attn_data.reshape(batch_size * n_heads, -1)
            data_list.append(flat_data)

            # Track metadata
            for b in range(batch_size):
                for h in range(n_heads):
                    metadata.append({"layer": layer_name, "head": h, "sample": b})

        self.dimred_data = torch.cat(data_list, dim=0)
        self.dimred_metadata = pd.DataFrame(metadata)

    def prepare_per_layer_data(self) -> None:
        """Prepare data for per-layer analysis."""
        # Select specific layer
        layer_options = list(self.processed_data.keys())
        selected_layer = st.sidebar.selectbox("Select Layer", layer_options)

        attn_data = self.processed_data[selected_layer]
        batch_size = attn_data.shape[0]
        n_heads = attn_data.shape[1]

        # Reshape to [batch * heads, seq_len * seq_len]
        self.dimred_data = attn_data.reshape(batch_size * n_heads, -1)

        # Track metadata
        metadata = []
        for b in range(batch_size):
            for h in range(n_heads):
                metadata.append({"head": h, "sample": b})
        self.dimred_metadata = pd.DataFrame(metadata)

    def prepare_per_head_data(self) -> None:
        """Prepare data for per-head analysis."""
        # Select specific layer and head
        layer_options = list(self.processed_data.keys())
        selected_layer = st.sidebar.selectbox("Select Layer", layer_options)
        n_heads = self.processed_data[selected_layer].shape[1]
        selected_head = st.sidebar.selectbox("Select Head", range(n_heads))

        attn_data = self.processed_data[selected_layer][
            :, selected_head
        ]  # [batch, seq_len, seq_len]

        # Reshape to [batch, seq_len * seq_len]
        self.dimred_data = attn_data.reshape(attn_data.shape[0], -1)

        # Track metadata
        self.dimred_metadata = pd.DataFrame({"sample": range(attn_data.shape[0])})

    def prepare_per_query_data(self) -> None:
        """Prepare data for per-query analysis."""
        # Select specific layer, head, and query position
        layer_options = list(self.processed_data.keys())
        selected_layer = st.sidebar.selectbox("Select Layer", layer_options)
        n_heads = self.processed_data[selected_layer].shape[1]
        selected_head = st.sidebar.selectbox("Select Head", range(n_heads))
        seq_len = self.processed_data[selected_layer].shape[2]
        selected_query = st.sidebar.selectbox("Select Query Position", range(seq_len))

        attn_data = self.processed_data[selected_layer][
            :, selected_head, selected_query
        ]  # [batch, seq_len]

        # Use raw attention vectors
        self.dimred_data = attn_data

        # Track metadata
        self.dimred_metadata = pd.DataFrame({"sample": range(attn_data.shape[0])})

    def plot_projection(self) -> None:
        """Plot the dimensionality reduction results."""
        st.subheader(
            f"Attention Pattern Analysis: {self.analysis_level.replace('_', ' ').title()}"
        )

        # Add description of what we're looking at
        analysis_descriptions = {
            "all_layers": "Visualizing positional biases across all layers and heads. Clusters may indicate similar attention patterns.",
            "per_layer": "Examining how different heads within a layer develop distinct positional attention patterns.",
            "per_head": "Analyzing how a specific head's attention pattern varies across different inputs.",
            "per_query": "Showing how attention is distributed from a specific query position across different inputs.",
        }
        st.markdown(f"*{analysis_descriptions[self.analysis_level]}*")

        # Create the projection plot
        fig = go.Figure()

        # Color mapping based on available metadata
        if "layer" in self.dimred_metadata.columns:
            color_col = "layer"
            hover_data = ["layer", "head", "sample"]
        elif "head" in self.dimred_metadata.columns:
            color_col = "head"
            hover_data = ["head", "sample"]
        else:
            color_col = "sample"
            hover_data = ["sample"]

        # Add feature information if available
        if self.features_df is not None:
            for col in self.features_df.columns:
                if col != "pattern":  # Exclude pattern column
                    self.dimred_metadata[col] = (
                        self.features_df[col]
                        .iloc[self.dimred_metadata["sample"]]
                        .values
                    )
                    hover_data.append(col)

        # Create scatter plot
        fig = px.scatter(
            self.dimred_metadata,
            x=self.projection[:, 0],
            y=self.projection[:, 1],
            color=color_col,
            hover_data=hover_data,
            title=f"Attention Pattern Analysis ({self.analysis_level})",
            width=1000,
            height=600,
        )

        # Update layout
        fig.update_layout(
            xaxis_title="Dimension 1", yaxis_title="Dimension 2", showlegend=True
        )

        st.plotly_chart(fig)

        # Add interpretation hints
        st.markdown("### Interpretation Guide")
        interpretation_hints = {
            "all_layers": [
                "Clusters of points indicate similar attention patterns across layers/heads",
                "Distance between points represents how different the attention patterns are",
                "Look for layer-specific or head-specific clustering",
            ],
            "per_layer": [
                "Groups of points show heads with similar positional biases",
                "Outlier heads might have specialized attention patterns",
                "Consider how head patterns relate to input features",
            ],
            "per_head": [
                "Points represent how this head attends differently to various inputs",
                "Clusters might indicate similar input types or patterns",
                "Spread of points shows the head's adaptability to different inputs",
            ],
            "per_query": [
                "Each point shows how attention is distributed from the selected position",
                "Clusters suggest similar attention patterns for certain input types",
                "Wide spread indicates position-sensitive attention patterns",
            ],
        }

        for hint in interpretation_hints[self.analysis_level]:
            st.markdown(f"- {hint}")

    def process_attention_for_heatmap(self) -> None:
        """Process attention matrices for heatmap visualization."""
        self.process_attention_data()

        # For heatmaps, we want to preserve the original attention matrix shape
        if self.agg_choice in ["single sample", "random subset"]:
            logger.info(
                f"Processing {len(self.selected_samples)} samples for heatmap visualization"
            )
        else:
            logger.info("Processing aggregated attention for heatmap visualization")

    def process_attention_for_graph(self) -> None:
        """Process attention matrices for graph visualization."""
        self.process_attention_data()

        # Graph-specific parameters in sidebar
        st.sidebar.subheader("Graph Visualization Settings")
        self.attn_threshold = st.sidebar.slider(
            "Attention Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            help="Only show attention connections above this threshold",
        )

        self.max_nodes = st.sidebar.slider(
            "Max Sequence Length",
            min_value=10,
            max_value=100,
            value=30,
            help="Truncate sequence to this length for visualization",
        )

        self.show_cross_head = st.sidebar.checkbox(
            "Show Cross-Head Connections",
            value=True,
            help="Connect nodes between heads that share position",
        )

        self.cross_head_connection = (
            st.sidebar.selectbox(
                "Cross-Head Connection Type",
                ["query", "key", "both"],
                help="Connect nodes between heads based on matching positions",
            )
            if self.show_cross_head
            else "none"
        )

        self.stabilize_layouts = st.sidebar.checkbox(
            "Stabilize Similar Layouts",
            value=True,
            help="Try to align similar attention patterns across heads",
        )

        if self.stabilize_layouts:
            self.similarity_threshold = st.sidebar.slider(
                "Layout Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="Minimum similarity score to attempt layout alignment",
            )

            self.alignment_method = st.sidebar.selectbox(
                "Layout Alignment Method",
                ["graph_similarity", "attention_pattern"],
                help="Method to determine which layouts to align",
            )

    def create_head_graph(
        self, attn_matrix: np.ndarray, head_idx: int, layer_name: str
    ) -> nx.Graph:
        """Create a graph for a single attention head."""
        G = nx.Graph()

        # Truncate sequence if needed
        seq_len = min(attn_matrix.shape[0], self.max_nodes)
        attn_matrix = attn_matrix[:seq_len, :seq_len]

        # Add nodes
        for i in range(seq_len):
            G.add_node(
                f"{layer_name}_{head_idx}_{i}",
                pos=(i, 0),  # Base position for layout
                token_idx=i,
                head_idx=head_idx,
                layer_name=layer_name,
                node_type="token",
            )

        # Add edges for attention above threshold
        for i, j in combinations(range(seq_len), 2):
            weight = float(attn_matrix[i, j])
            if weight > self.attn_threshold:
                G.add_edge(
                    f"{layer_name}_{head_idx}_{i}",
                    f"{layer_name}_{head_idx}_{j}",
                    weight=weight,
                    edge_type="attention",
                )

        return G

    def compute_graph_similarity(self, G1: nx.Graph, G2: nx.Graph) -> float:
        """Compute similarity score between two graphs based on structure."""
        # Get adjacency matrices
        nodes1 = list(G1.nodes())
        nodes2 = list(G2.nodes())

        adj1 = nx.adjacency_matrix(G1, nodelist=nodes1).todense()
        adj2 = nx.adjacency_matrix(G2, nodelist=nodes2).todense()

        # If different sizes, pad smaller matrix
        max_size = max(adj1.shape[0], adj2.shape[0])
        if adj1.shape[0] < max_size:
            adj1 = np.pad(
                adj1, ((0, max_size - adj1.shape[0]), (0, max_size - adj1.shape[0]))
            )
        if adj2.shape[0] < max_size:
            adj2 = np.pad(
                adj2, ((0, max_size - adj2.shape[0]), (0, max_size - adj2.shape[0]))
            )

        # Compute similarity score
        similarity = 1 - (np.sum(np.abs(adj1 - adj2)) / (max_size * max_size))
        return float(similarity)

    def align_layouts(
        self,
        reference_pos: dict,
        target_pos: dict,
        reference_G: nx.Graph,
        target_G: nx.Graph,
    ) -> dict:
        """Align target layout to reference layout using point set registration."""
        # Extract positions as arrays
        ref_points = np.array([reference_pos[n] for n in reference_G.nodes()])
        target_points = np.array([target_pos[n] for n in target_G.nodes()])

        # Center both point sets
        ref_center = np.mean(ref_points, axis=0)
        target_center = np.mean(target_points, axis=0)
        ref_centered = ref_points - ref_center
        target_centered = target_points - target_center

        # Compute optimal rotation using Procrustes analysis
        if len(ref_points) == len(target_points):
            # Direct alignment
            cost_matrix = cdist(ref_centered, target_centered)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            target_centered = target_centered[col_ind]
        else:
            # Partial alignment using available points
            min_size = min(len(ref_points), len(target_points))
            cost_matrix = cdist(ref_centered[:min_size], target_centered[:min_size])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            target_centered = target_centered[col_ind]

        # Compute rotation matrix
        H = ref_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Apply transformation
        aligned_points = (target_points - target_center) @ R + ref_center

        # Create new position dictionary
        aligned_pos = {}
        for node, pos in zip(target_G.nodes(), aligned_points):
            aligned_pos[node] = tuple(pos)

        return aligned_pos

    def create_multi_head_graph(self) -> nx.Graph:
        """Create a combined graph for all attention heads with stabilized layouts."""
        G_combined = nx.Graph()
        head_graphs = {}  # Store individual head graphs
        head_layouts = {}  # Store layouts for each head

        # First pass: create all head graphs and compute initial layouts
        for layer_name, attn_data in self.processed_data.items():
            if self.agg_choice in ["single sample", "random subset"]:
                sample_idx = st.session_state.get("current_sample_idx", 0)
                attn_data = attn_data[sample_idx]

            n_heads = attn_data.shape[0]

            for head_idx in range(n_heads):
                G_head = self.create_head_graph(
                    attn_data[head_idx].cpu().numpy(), head_idx, layer_name
                )
                head_key = (layer_name, head_idx)
                head_graphs[head_key] = G_head

                # Compute initial layout
                head_layouts[head_key] = nx.spring_layout(
                    G_head, k=1 / np.sqrt(len(G_head)), iterations=50
                )

        # Second pass: stabilize layouts if enabled
        if self.stabilize_layouts:
            # Find reference layout (graph with most edges)
            reference_key = max(
                head_graphs.keys(), key=lambda k: head_graphs[k].number_of_edges()
            )
            reference_graph = head_graphs[reference_key]
            reference_layout = head_layouts[reference_key]

            # Align other layouts to reference if similar enough
            for head_key, G_head in head_graphs.items():
                if head_key == reference_key:
                    continue

                if self.alignment_method == "graph_similarity":
                    similarity = self.compute_graph_similarity(reference_graph, G_head)
                else:  # attention_pattern
                    similarity = self.compute_attention_pattern_similarity(
                        reference_key, head_key, head_graphs
                    )

                if similarity >= self.similarity_threshold:
                    head_layouts[head_key] = self.align_layouts(
                        reference_layout,
                        head_layouts[head_key],
                        reference_graph,
                        G_head,
                    )

        # Combine all graphs with their layouts
        for head_key, G_head in head_graphs.items():
            # Scale and translate layout to grid position
            layer_name, head_idx = head_key
            layer_idx = list(set(k[0] for k in head_graphs.keys())).index(layer_name)

            base_layout = head_layouts[head_key]
            grid_layout = {}
            for node, pos in base_layout.items():
                # Scale and translate to grid position
                x = pos[0] + (layer_idx * (self.max_nodes + 5))
                y = pos[1] + (head_idx * (self.max_nodes + 5))
                grid_layout[node] = (x, y)

            # Add to combined graph
            G_combined.add_nodes_from(G_head.nodes(data=True))
            G_combined.add_edges_from(G_head.edges(data=True))
            nx.set_node_attributes(
                G_combined, {n: {"pos": p} for n, p in grid_layout.items()}
            )

        # Add cross-head connections if enabled
        if self.show_cross_head:
            self.add_cross_head_connections(G_combined)

        return G_combined

    def compute_attention_pattern_similarity(
        self, key1: tuple, key2: tuple, head_graphs: dict
    ) -> float:
        """Compute similarity between attention patterns of two heads."""
        G1 = head_graphs[key1]
        G2 = head_graphs[key2]

        # Get edge weights as attention patterns
        edges1 = {(min(e), max(e)): d["weight"] for e, d in G1.edges(data=True)}
        edges2 = {(min(e), max(e)): d["weight"] for e, d in G2.edges(data=True)}

        # Get all possible edges
        all_edges = set(edges1.keys()) | set(edges2.keys())

        # Compute similarity score
        total = 0
        count = 0
        for edge in all_edges:
            weight1 = edges1.get(edge, 0)
            weight2 = edges2.get(edge, 0)
            total += abs(weight1 - weight2)
            count += 1

        return 1 - (total / count if count > 0 else 0)

    def add_cross_head_connections(self, G: nx.Graph) -> None:
        """Add connections between nodes in different heads based on position."""
        nodes = list(G.nodes(data=True))

        for i, (node1, data1) in enumerate(nodes):
            for node2, data2 in nodes[i + 1 :]:
                # Only connect nodes from different heads
                if data1["head_idx"] != data2["head_idx"]:
                    # Connect based on selected connection type
                    if (
                        self.cross_head_connection in ["query", "both"]
                        and data1["token_idx"] == data2["token_idx"]
                    ):
                        G.add_edge(
                            node1, node2, weight=0.5, edge_type="cross_head_query"
                        )
                    elif (
                        self.cross_head_connection in ["key", "both"]
                        and data1["token_idx"] == data2["token_idx"]
                    ):
                        G.add_edge(node1, node2, weight=0.5, edge_type="cross_head_key")

    def plot_attention_graphs(self) -> None:
        """Plot attention patterns as graphs."""
        st.subheader("Attention Head Graphs")

        # Create the multi-head graph
        G = self.create_multi_head_graph()

        # Calculate grid layout
        unique_layers = sorted(set(nx.get_node_attributes(G, "layer_name").values()))
        max_heads = max(
            sum(
                1
                for _, data in G.nodes(data=True)
                if data["layer_name"] == layer and data["node_type"] == "token"
            )
            // self.max_nodes
            for layer in unique_layers
        )

        # Position nodes in a grid layout
        pos = {}
        for node, data in G.nodes(data=True):
            layer_idx = unique_layers.index(data["layer_name"])
            base_x = data["token_idx"] + (layer_idx * (self.max_nodes + 5))
            base_y = data["head_idx"] * (self.max_nodes + 5)
            pos[node] = (base_x, base_y)

        # Create plotly figure
        edge_traces = []
        node_traces = []

        # Add edges with different colors for attention and cross-head connections
        edge_colors = {
            "attention": "rgba(150,150,150,0.3)",
            "cross_head_query": "rgba(255,0,0,0.2)",
            "cross_head_key": "rgba(0,0,255,0.2)",
        }

        for edge_type, color in edge_colors.items():
            edge_x = []
            edge_y = []
            for edge in G.edges(data=True):
                if edge[2].get("edge_type") == edge_type:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            edge_traces.append(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=0.5, color=color),
                    hoverinfo="none",
                    mode="lines",
                    name=edge_type,
                )
            )

        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(
                f"Layer: {data['layer_name']}<br>"
                f"Head: {data['head_idx']}<br>"
                f"Position: {data['token_idx']}"
            )

        node_traces.append(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                text=node_text,
                marker=dict(
                    size=5,
                    color="rgb(50,50,50)",
                    line=dict(width=0.5, color="rgb(50,50,50)"),
                ),
                name="nodes",
            )
        )

        # Create figure
        fig = go.Figure(
            data=edge_traces + node_traces,
            layout=go.Layout(
                showlegend=True,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=1200,
                height=800,
                title="Attention Head Graph Visualization",
            ),
        )

        st.plotly_chart(fig)


if __name__ == "__main__":
    attention_path = os.getenv("ATTENTION_PATH")
    features_path = os.getenv("FEATURES_PATH")

    # Load attention matrices and features
    if attention_path:
        if not os.path.exists(attention_path):
            raise FileNotFoundError(f"Attention file not found: {attention_path}")
        attention_matrices = torch.load(attention_path)
    else:
        raise ValueError("ATTENTION_PATH must be provided")

    features_df = pd.read_csv(features_path) if features_path else None

    app = AttentionVisualizer(attention_matrices, features_df)
    app.run()
