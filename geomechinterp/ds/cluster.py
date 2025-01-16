import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from catboost import CatBoostClassifier, Pool
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.manifold import Isomap


def cluster_hierarchical(activations, threshold=10, method="ward", plot=False):
    """
    Perform hierarchical clustering and find clusters using a dendrogram.

    Parameters:
    - activations (ndarray): The activation matrix.
    - threshold (float): Threshold for cutting the dendrogram (distance scale).
    - method (str): Linkage method (e.g., 'ward', 'single', 'complete', 'average').

    Returns:
    - ndarray: Cluster labels for each sample.
    """
    # Compute linkage matrix
    # normalize activations
    activations = activations / activations.max(axis=0)
    linkage_matrix = linkage(activations, method=method)

    # Plot dendrogram
    if plot:
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.axhline(y=threshold, color="r", linestyle="--")
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.xticks([])
        plt.show()

    # Extract cluster labels
    labels = fcluster(linkage_matrix, t=threshold, criterion="distance")
    return labels


def cluster_kmeans_elbow(activations, max_clusters=10, plot=False):
    """
    Perform KMeans clustering and find the optimal number of clusters using the elbow method on silhouette scores.

    Parameters:
    - activations (ndarray): The activation matrix.
    - max_clusters (int): The maximum number of clusters to evaluate.
    - plot (bool): Whether to plot the silhouette scores.

    Returns:
    - ndarray: Cluster labels for each sample.
    """
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        print("activations", activations.shape)
        kmeans = KMeans(n_clusters=k, random_state=42).fit(activations)
        score = silhouette_score(activations, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_k = np.argmax(silhouette_scores) + 2
    if plot:
        # Plot silhouette scores
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker="o")
        plt.axvline(x=optimal_k, color="r", linestyle="--")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Elbow Method (Silhouette Score)")
        plt.show()
    # run with optimal_k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit_predict(activations)
    return kmeans


def cluster_isomap_kmeans(activations, max_clusters=10, n_components=2, plot=False):
    # Reduce dimensionality with Isomap
    isomap = Isomap(n_components=n_components)
    reduced_activations = isomap.fit_transform(activations)

    return cluster_kmeans_elbow(
        reduced_activations, max_clusters=max_clusters, plot=plot
    )


def calculate_feature_importance(X, y, y_labels=None):
    """
    Train a Random Forest classifier and compute feature importance.

    Parameters:
    - X (ndarray): Feature matrix.
    - y (ndarray): Target labels.

    Returns:
    - importance_df (DataFrame): DataFrame with feature importance scores.
    - model (RandomForestClassifier): Trained Random Forest model.
    """
    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(X, y)

    # Compute feature importance
    importances = clf.feature_importances_
    perm_importance = permutation_importance(clf, X, y, n_repeats=10, random_state=42)

    importance_df = pd.DataFrame(
        {
            "Feature": np.arange(X.shape[1]),
            "RandomForestImportance": importances,
            "PermutationImportance": perm_importance.importances_mean,
        }
    )
    if y_labels is not None:
        assert len(y_labels) == X.shape[1]
        importance_df["Feature"] = importance_df["Feature"].replace(
            np.arange(X.shape[1]), y_labels
        )
    return importance_df, clf


def calculate_feature_importance_catboost(X, y):
    """
    Train a CatBoost classifier and compute feature importance.

    Parameters:
    - X (ndarray): Feature matrix.
    - y (ndarray): Target labels.

    Returns:
    - importance_df (DataFrame): DataFrame with feature importance scores.
    - model (CatBoostClassifier): Trained CatBoost model.
    """
    cat_features = list(
        range(X.shape[1])
    )  # Assume all features are categorical for simplicity
    model = CatBoostClassifier(
        iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=0
    )
    model.fit(X, y, cat_features=cat_features)

    # Compute feature importance
    importances = model.get_feature_importance(Pool(X, y))

    importance_df = pd.DataFrame(
        {"Feature": np.arange(X.shape[1]), "CatBoostImportance": importances}
    )
    return importance_df, model


def random_forest_feature_imp(X, y):
    # Train Random Forest classifier
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X, y)

    # Feature importance
    feature_importances = pd.DataFrame(
        {"Feature": X.columns, "Importance": rf.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    return feature_importances
