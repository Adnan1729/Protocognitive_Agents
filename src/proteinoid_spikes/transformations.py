"""
Section III of the paper: data transformation.

* F1 (Eq. 1): spiral sampling -- non-uniform sampling of time points.
* F2 (Eq. 2): first significant digit after the decimal point.
* Graph construction (Table I): for each F1-selected primary node, link
  to all secondary nodes sharing the same F2 digit.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# F1 and F2
# ---------------------------------------------------------------------------

def F1(t):
    """Spiral sampling function (paper Eq. 1).

    x(t) = 10 + (10 - 2t) * cos(t * pi)

    Accepts a scalar or numpy array of t values.
    """
    return 10 + (10 - 2 * t) * np.cos(t * np.pi)


def F2(x):
    """First significant digit after the decimal point (paper Eq. 2).

    Vectorised: x may be a scalar, numpy array, or pandas Series.
    """
    frac_part = x - np.floor(x)
    n = np.ceil(np.log10(1 / frac_part)).astype(int)
    return np.floor((10 ** n) * frac_part)


# ---------------------------------------------------------------------------
# Node lookup helpers
# ---------------------------------------------------------------------------

def find_point_less_than(df: pd.DataFrame, value: float):
    """Largest Time in df strictly less than `value`. NaN if none exists."""
    return df[df["Time"] < value]["Time"].max()


def find_points_with_same_digit(df: pd.DataFrame, x: float, d) -> list:
    """All Time values in df whose F2 digit equals d.

    The `x` argument is kept in the signature to mirror the original
    notebook even though it is unused.
    """
    return df[np.isclose(F2(df["Time"]), d)]["Time"].tolist()


# ---------------------------------------------------------------------------
# Graph construction (Table I of the paper)
# ---------------------------------------------------------------------------

def build_graph(df: pd.DataFrame, with_positions: bool = False) -> nx.Graph:
    """Build the multinodal graph for one spike train dataset.

    Parameters
    ----------
    df : pd.DataFrame
        A single dataset's (Time, Spike) DataFrame.
    with_positions : bool
        If True, attach 3D positions to nodes for the multinodal-graph
        plot in Fig. 4. The unweighted graph used for complexity metrics
        does not need positions.

    Returns
    -------
    networkx.Graph
    """
    G = nx.Graph()
    t_values = np.arange(config.F1_T_START, config.F1_T_STOP)
    x_t_values = F1(t_values)
    selected_points = [find_point_less_than(df, x) for x in x_t_values]

    for j, point in enumerate(selected_points):
        if pd.notna(point):
            if with_positions:
                G.add_node(point, pos=(point, j, 0))
            else:
                G.add_node(point)

            d = F2(point)
            connected_points = find_points_with_same_digit(df, point, d)
            for connected_point in connected_points:
                if connected_point != point:
                    if with_positions:
                        G.add_node(connected_point, pos=(connected_point, j, 1))
                    else:
                        G.add_node(connected_point)
                    G.add_edge(point, connected_point)

    return G