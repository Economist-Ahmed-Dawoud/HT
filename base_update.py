from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict


class BaseConfig:
    """A minimal base configuration class."""

    pass


@dataclass
class RCAResults:
    """
    Container for RCA results.

    Attributes:
        root_cause_nodes: A list of tuples, each (node, score).
        root_cause_paths: A dictionary mapping a node to its path (if available).
    """

    root_cause_nodes: list = field(default_factory=list)
    root_cause_paths: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "root_cause_nodes": self.root_cause_nodes,
            "root_cause_paths": self.root_cause_paths,
        }

    def to_list(self):
        return [
            {
                "root_cause": node,
                "score": score,
                "paths": self.root_cause_paths.get(node, None),
            }
            for node, score in self.root_cause_nodes
        ]


class BaseRCA(ABC):
    """Abstract base class for RCA algorithms."""

    @abstractmethod
    def train(self, **kwargs):
        """Train the model with normal (non-anomalous) data."""
        pass

    @abstractmethod
    def find_root_causes(self, **kwargs) -> RCAResults:
        """Identify root causes from data."""
        pass
