# ht.py file

from dataclasses import dataclass
import pickle
import pandas as pd
from typing import Dict, Union, List
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

from base import BaseConfig, BaseRCA, RCAResults


@dataclass
class HTConfig(BaseConfig):
    """
    Configuration class for the HT algorithm.

    Attributes:
        graph: A pandas DataFrame (adjacency matrix) or a filepath (CSV or pickle) representing the causal graph.
        aggregator: Function name for aggregating node scores ("max", "min", or "sum"). Default is "max".
        root_cause_top_k: Maximum number of root causes to return. Default is 3.
    """

    graph: Union[pd.DataFrame, str]
    aggregator: str = "max"
    root_cause_top_k: int = 3


class HT(BaseRCA):
    """
    Regression-based Hypothesis Testing (HT) method for Root Cause Analysis.

    This class replicates the HT algorithm from PyRCA in a lightweight, standalone package.
    """

    config_class = HTConfig

    def __init__(self, config: HTConfig):
        self.config = config
        # Load the causal graph from file or use directly if DataFrame.
        if isinstance(config.graph, str):
            if config.graph.endswith(".csv"):
                graph = pd.read_csv(config.graph)
            elif config.graph.endswith(".pkl"):
                with open(config.graph, "rb") as f:
                    graph = pickle.load(f)
            else:
                raise RuntimeError("Unsupported graph file format. Use CSV or pickle.")
        else:
            graph = config.graph

        self.adjacency_mat = graph
        # Create a directed graph from the adjacency matrix.
        self.graph = nx.from_pandas_adjacency(graph, create_using=nx.DiGraph())
        # Dictionary to hold (regressor, scaler) for each node.
        self.regressors_dict: Dict[str, List] = {}

    @staticmethod
    def _get_aggregator(name):
        if name == "max":
            return max
        elif name == "min":
            return min
        elif name == "sum":
            return sum
        else:
            raise ValueError(f"Unknown aggregator {name}")

    def train(self, normal_df: pd.DataFrame, **kwargs):
        """
        Train a regression model for each node based on its parents.

        For each node in the graph, if it has parent nodes then a LinearRegression model
        is trained to predict its values from its parents. The residual error is then scaled.
        """
        if self.graph is None:
            raise ValueError("Graph is not set.")

        for node in list(self.graph):
            parents = list(self.graph.predecessors(node))
            if parents:
                normal_x = normal_df[parents].values
                # Check if there is at least one parent (and nonzero features)
                if normal_x.shape[1] > 0:
                    regressor = LinearRegression()
                    regressor.fit(normal_x, normal_df[node].values)
                    normal_err = normal_df[node].values - regressor.predict(normal_x)
                    scaler = StandardScaler().fit(normal_err.reshape(-1, 1))
                    self.regressors_dict[node] = [regressor, scaler]
                    continue  # go to next node
            # For nodes with no parents, only scale the data.
            scaler = StandardScaler().fit(normal_df[node].values.reshape(-1, 1))
            self.regressors_dict[node] = [None, scaler]

    def find_root_causes(
        self,
        abnormal_df: pd.DataFrame,
        anomalous_metrics: str = None,
        adjustment: bool = False,
        **kwargs,
    ) -> RCAResults:
        """
        Identify root causes from abnormal data.

        For each node, compute a score based on the regression error on abnormal data.
        Optionally, perform descendant adjustment. If an anomalous metric is provided,
        shortest paths from each candidate root to the anomaly are included.
        """
        node_scores = {}
        for node in list(self.graph):
            parents = list(self.graph.predecessors(node))
            if parents:
                abnormal_x = abnormal_df[parents].values
                if abnormal_x.shape[1] > 0:
                    regressor = self.regressors_dict[node][0]
                    abnormal_err = abnormal_df[node].values - regressor.predict(
                        abnormal_x
                    )
                    scores = self.regressors_dict[node][1].transform(
                        abnormal_err.reshape(-1, 1)
                    )[:, 0]
                else:
                    scores = self.regressors_dict[node][1].transform(
                        abnormal_df[node].values.reshape(-1, 1)
                    )[:, 0]
            else:
                scores = self.regressors_dict[node][1].transform(
                    abnormal_df[node].values.reshape(-1, 1)
                )[:, 0]

            # Aggregate the absolute error scores.
            agg_func = self._get_aggregator(self.config.aggregator)
            score = agg_func(abs(scores))
            # Compute a confidence value (optional, for illustration).
            conf = 1 - 2 * norm.cdf(-abs(score))
            node_scores[node] = [score, conf]

        # Optional descendant adjustment.
        if adjustment:
            H = self.graph.reverse(copy=True)
            topological_sort = list(nx.topological_sort(H))
            child_nodes = {}
            for node in topological_sort:
                child_nodes[node] = list(self.graph.successors(node))
                for child in child_nodes[node]:
                    if node_scores[child][0] < 3:
                        child_nodes[node] = list(
                            set(child_nodes[node]).union(set(child_nodes[child]))
                        )
            for node in list(self.graph):
                if node_scores[node][0] > 3:
                    candidate_scores = [
                        node_scores[child_node][0] for child_node in child_nodes[node]
                    ]
                    if not candidate_scores:
                        candidate_scores.append(0)
                    node_scores[node][0] = node_scores[node][0] + max(candidate_scores)

        # Select top-k root cause candidates.
        root_cause_nodes = [(key, node_scores[key][0]) for key in node_scores]
        root_cause_nodes = sorted(root_cause_nodes, key=lambda r: r[1], reverse=True)[
            : self.config.root_cause_top_k
        ]

        # If an anomalous metric is specified, attempt to record a shortest path from each candidate.
        root_cause_paths = {}
        if anomalous_metrics is not None:
            for root, _ in root_cause_nodes:
                try:
                    path = nx.shortest_path(
                        self.graph, source=root, target=anomalous_metrics
                    )
                except nx.exception.NetworkXNoPath:
                    path = None
                root_cause_paths[root] = path

        return RCAResults(
            root_cause_nodes=root_cause_nodes, root_cause_paths=root_cause_paths
        )
