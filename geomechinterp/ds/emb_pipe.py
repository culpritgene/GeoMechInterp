from typing import Literal, Any
from pathlib import Path
import pandas as pd
from geomechinterp.ds.cluster import (
    cluster_isomap_kmeans,
    cluster_kmeans_elbow,
    random_forest_feature_imp,
)
import torch
from torch.utils.data import Dataset
from jaxtyping import Int, Float
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


from geomechinterp.ds.stats import perform_anova
from geomechinterp.causal.pattern_generator import build_features_dataframe

from geomechinterp.tflens.activations import load_precomputed_activation
from geomechinterp.viz.plots import plot_pca_activations, plot_tsne_activations

from geomechinterp.viz.streamlit_viz import ActivationVisualizer
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)


def single_position(
    activations: Float[torch.Tensor, "batch seq_len d_model"], position: int
) -> Float[torch.Tensor, "batch d_model"]:
    return activations[:, position, :]


def mean_across_positions(
    activations: Float[torch.Tensor, "batch seq_len d_model"]
) -> Float[torch.Tensor, "batch d_model"]:
    return activations.mean(axis=1)


def stack_all_positions(
    activations: Float[torch.Tensor, "batch seq_len d_model"]
) -> Float[torch.Tensor, "batch d_model"]:
    return activations.reshape(activations.shape[0], -1)


class ActivationStatsPipeline:
    def __init__(
        self,
        dataset: Dataset,
        activations: dict[str, torch.Tensor] | None = None,
        activations_dir: str | Path | None = None,
        activations_file_substring: str | None = None,
        selected_hooks: list[str] | None = None,
        feature_imp_threshold: float = 0.02,
        anova_threshold: float = 0.01,
        agg_methods: list[Literal["mean", "stack", "positions"] | int] = [
            "mean",
            "stack",
            "positions",
        ],
    ):
        self.dataset: Dataset = dataset
        self.activations_dir: Path | None = activations_dir
        self.activations_file_substring: str | None = activations_file_substring
        self.selected_hooks: list[str] | None = selected_hooks

        if activations is not None:
            self.activations: dict[str, torch.Tensor] = activations
        elif activations_dir is not None:
            logging.info(f"Using Lazy Loading of Activations from {activations_dir}")
        else:
            raise ValueError("Either activations or activations_dir must be provided")

        self.feature_imp_threshold: float = feature_imp_threshold
        self.anova_threshold: float = anova_threshold
        self.agg_methods: list[Literal["mean", "stack"] | int] = agg_methods

        self.activations: dict[str, torch.Tensor] = {}
        self.features_df: pd.DataFrame | None = None
        self.cursor: str = ""
        self.results: dict[str, Any] = {}

        self.viz: ActivationVisualizer | None = None
        self.__post_init__()

    def __post_init__(self):
        # build feature df
        self.features_df = self.build_feature_df()

        # load activations
        if not self.activations and self.activations_dir is not None:
            if self.selected_hooks is None:
                raise Warning(
                    "selected_hooks are not provided, loading *all* activations"
                )
            self.load_activations(selected_hooks=self.selected_hooks)

        # add position integers to agg_methods
        if "positions" in self.agg_methods:
            self.agg_methods.pop(self.agg_methods.index("positions"))
            self.agg_methods.extend(
                list(range(self.activations[list(self.activations.keys())[0]].shape[1]))
            )

        # initialize viz
        self.viz = ActivationVisualizer(
            activations=self.activations, features_df=self.features_df
        )

    def build_feature_df(
        self,
        one_hot_encode: bool = True,
        take_top_freq_cats: int | float | None = 0.85,
        drop_constant_columns: bool = True,
        compute_uncertainty: bool = False,
        attach_patterns: bool = False,
    ) -> pd.DataFrame:
        features_df = build_features_dataframe(
            self.dataset, one_hot_encode, take_top_freq_cats, drop_constant_columns
        )
        logging.info(f"Built feature dataframe with {features_df.shape[1]} features")
        return features_df

    def load_activations(self, selected_hooks: list[str] | None = None):
        if selected_hooks is None:
            selected_hooks = self.selected_hooks
        else:
            self.selected_hooks = selected_hooks
        if self.activations_dir is not None:
            activations, _ = load_precomputed_activation(
                self.activations_dir,
                selected_hooks=selected_hooks,
                subselected_positions=None,
                file_substring=self.activations_file_substring,
            )
            self.activations = activations
            logging.info(f"Loaded {len(self.activations)} hooked activations.")
        else:
            raise ValueError("activations_dir is not provided")

    def set_cursor(
        self, activations_str: str, agg_func: Literal["mean", "stack"] | int
    ):
        self.cursor = activations_str + "_" + str(agg_func)
        self.results[self.cursor] = {}

    def _split_cursor(self, cursor: str):
        return "_".join(cursor.split("_")[:-1]), int(cursor.split("_")[-1])

    def get_activations(
        self,
        activations_str: str,
    ):
        if activations_str in self.activations:
            return self.activations[activations_str].detach().cpu().numpy()
        else:
            raise ValueError(f"Cursor {activations_str} not found in activations")

    def agg_activations(
        self, activations_str: str, agg_func: Literal["mean", "stack"] | int
    ):
        activations = self.get_activations(activations_str)
        if isinstance(agg_func, int):
            activations = single_position(activations, agg_func)
        elif agg_func == "mean":
            activations = mean_across_positions(activations)
        elif agg_func == "stack":
            activations = stack_all_positions(activations)
        return activations

    def cluster(
        self,
        activations_str: str,
        agg_func: Literal["mean", "stack"] | int = "mean",
        max_clusters: int = 12,
        use_isomap: bool = False,
        isomap_n_components: int = 10,
    ):
        self.set_cursor(activations_str, agg_func)
        activations = self.agg_activations(activations_str, agg_func)
        if use_isomap:
            self.clusters = cluster_isomap_kmeans(
                activations,
                max_clusters=max_clusters,
                n_components=isomap_n_components,
            )
        else:
            self.clusters = cluster_kmeans_elbow(activations, max_clusters=max_clusters)
        self.results[self.cursor]["clusters"] = self.clusters
        return self

    def anova_on_clusters(self):
        if self.clusters is None:
            raise ValueError("Clusters are not computed yet")
        self.anova_results = perform_anova(self.features_df, self.clusters)
        self.results[self.cursor]["anova"] = self.anova_results
        return self

    def feature_imp(self):
        if self.clusters is None:
            raise ValueError("Clusters are not computed yet")
        self.feature_imp_results = random_forest_feature_imp(
            self.features_df, self.clusters
        )
        self.results[self.cursor]["feature_imp"] = self.feature_imp_results
        return self

    def filter_non_important(self, cursor: str):
        res = self.results[cursor]
        res["anova"] = res["anova"][
            res["anova"]["Adjusted p-value"] < self.anova_threshold
        ]
        res["feature_imp"] = res["feature_imp"][
            res["feature_imp"]["Importance"] > self.feature_imp_threshold
        ]
        self.results[cursor] = res
        return self.results

    def run_single(
        self,
        activations_str: str,
        agg_func: Literal["mean", "stack"] | int,
        filter_non_important: bool = True,
    ):
        self.set_cursor(activations_str, agg_func)
        self.cluster(activations_str, agg_func)
        self.anova_on_clusters()
        self.feature_imp()
        if filter_non_important:
            self.filter_non_important(self.cursor)
        return self.results

    def run_all(self, filter_non_important: bool = True):
        for activations_str in self.activations:
            for agg_func in self.agg_methods:
                self.run_single(activations_str, agg_func, filter_non_important)
        return self.results

    def run_pca(
        self,
        activations_str: str | None = None,
        agg_func: Literal["mean", "stack"] | int | None = "mean",
    ):
        if activations_str is None:
            activations_str, agg_func = self._split_cursor(self.cursor)
        else:
            self.set_cursor(activations_str, agg_func)
        activations = self.agg_activations(activations_str, agg_func)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(activations)
        self.results[self.cursor]["pca"] = pca_result
        return self

    def run_tsne(
        self,
        activations_str: str | None = None,
        agg_func: Literal["mean", "stack"] | int | None = "mean",
    ):
        if activations_str is None:
            activations_str, agg_func = self._split_cursor(self.cursor)
        else:
            self.set_cursor(activations_str, agg_func)
        activations = self.agg_activations(activations_str, agg_func)
        tsne = TSNE(n_components=2)
        tsne_result = tsne.fit_transform(activations)
        self.results[self.cursor]["tsne"] = tsne_result
        return self

    def plot_pca(self, cursor: str | None = None):
        if cursor is None:
            cursor = self.cursor
        res = self.results.get(cursor, {})
        if "pca" not in res:
            self.run_pca()
        plot_pca_activations(
            self.results[self.cursor]["pca"],
            self.results[self.cursor].get("clusters", None),
        )

    def plot_tsne(self, cursor: str | None = None):
        if cursor is None:
            cursor = self.cursor
        res = self.results.get(cursor, {})
        if "tsne" not in res:
            self.run_tsne()
        plot_tsne_activations(
            self.results[self.cursor]["tsne"],
            self.results[self.cursor].get("clusters", None),
        )

    def streamlit_viz(self, one_hot_features: bool = False):
        # run as a separate process
        # providing paths to saved activations and features as arguments
        import subprocess
        import os

        # save current activations and features as tmp files
        activations_path = Path("activations.pt")
        features_path = Path("features.csv")
        torch.save(self.activations, activations_path)
        if one_hot_features:
            self.features_df.to_csv(features_path, index=False)
        else:
            features_df = self.build_feature_df(one_hot_encode=False)
            features_df.to_csv(features_path, index=False)

        # Set environment variables
        os.environ["ACTIVATIONS_PATH"] = str(activations_path)
        os.environ["FEATURES_PATH"] = str(features_path)

        # Run Streamlit app
        # find the path to the streamlit_viz.py file
        streamlit_viz_path = Path(__file__).parent.parent / "viz" / "streamlit_viz.py"
        subprocess.run(["streamlit", "run", str(streamlit_viz_path)])

        # Clean up temp files
        activations_path.unlink()
        features_path.unlink()
