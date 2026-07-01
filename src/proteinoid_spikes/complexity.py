"""
Section IV of the paper: complexity metrics and meta-metric.

Eight discrete graph metrics (Table II) are computed per dataset, then
combined into a single meta-metric via z-score normalisation, weighted
sum (paper ref. [21]), and sigmoid mapping (Eq. 5).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy, zscore

from . import config


def calculate_modified_resistance(G: nx.Graph) -> float:
    """Average effective resistance over connected components."""
    resistances = []
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if len(subgraph) > 1:
            L = nx.laplacian_matrix(subgraph).toarray()
            L_pseudo_inv = np.linalg.pinv(L)
            resistance = np.mean([
                L_pseudo_inv[i, i] + L_pseudo_inv[j, j] - 2 * L_pseudo_inv[i, j]
                for i in range(len(L)) for j in range(i + 1, len(L))
            ])
            resistances.append(resistance)

    return np.mean(resistances) if resistances else 0


def calculate_complexity_metrics(G: nx.Graph) -> dict:
    """Eight discrete complexity metrics for one graph (Table II)."""
    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": np.mean([d for n, d in G.degree()]),
        "clustering_coefficient": nx.average_clustering(G),
        "density": nx.density(G),
        "num_components": nx.number_connected_components(G),
    }

    degree_counts = nx.degree_histogram(G)
    degree_dist = np.array(degree_counts) / sum(degree_counts)
    metrics["degree_entropy"] = entropy(degree_dist)

    metrics["avg_resistance"] = calculate_modified_resistance(G)

    return metrics


def sigmoid(x):
    """Sigmoid (paper Eq. 5)."""
    return 1 / (1 + np.exp(-x))


def calculate_meta_metric(metrics_df: pd.DataFrame) -> pd.Series:
    """Combine 8 metrics into a single meta-metric in (0, 1).

    Each column is z-scored across datasets, then a weighted sum is taken
    with the weights from ``config.META_METRIC_WEIGHTS``, and the result
    is passed through a sigmoid.
    """
    weights = config.META_METRIC_WEIGHTS

    normalized_scores = metrics_df.apply(zscore)
    weighted_scores = normalized_scores.dot(pd.Series(weights))

    return sigmoid(weighted_scores)