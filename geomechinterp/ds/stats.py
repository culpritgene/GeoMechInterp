import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


def perform_anova(data: pd.DataFrame, cluster_column: str | np.ndarray) -> pd.DataFrame:
    """
    Perform ANOVA to analyze the contribution of features to clustering.

    Parameters:
    - data (pd.DataFrame): DataFrame containing features and cluster labels.
    - cluster_column (str | np.ndarray): Name of the column containing cluster labels or an array of cluster labels.

    Returns:
    - pd.DataFrame: DataFrame with ANOVA results including F-statistic and p-values.
    """
    # Encode categorical features
    encoded_data = data.copy()
    label_encoders = {}

    if isinstance(cluster_column, str):
        clusters_array = data[cluster_column].values
    else:
        clusters_array = cluster_column
        cluster_column = "cluster"

    clusters_array = clusters_array.astype(np.float32)
    n_clusters = len(np.unique(clusters_array))

    for col in data.columns:
        if col != cluster_column:  # Skip cluster column
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    # Perform ANOVA
    anova_results = []
    for col in encoded_data.columns:
        if col != cluster_column:  # Skip cluster column
            groups = [
                encoded_data[clusters_array == c][col].values for c in range(n_clusters)
            ]
            # Skip if less than 2 valid groups
            if len(groups) < 2:
                continue

            # Check for variance
            if all(len(np.unique(group)) <= 1 for group in groups):
                continue

            # Compute ANOVA
            # print(groups)
            f_stat, p_value = f_oneway(*groups)
            anova_results.append(
                {"Feature": col, "F-statistic": f_stat, "p-value": p_value}
            )

    anova_results = pd.DataFrame(anova_results)
    # Adjust p-values using Bonferroni correction
    anova_results["Adjusted p-value"] = anova_results["p-value"] * len(anova_results)
    anova_results = anova_results.sort_values(by="F-statistic", ascending=False)

    return anova_results
