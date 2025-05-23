from typing import Optional

from core.chain import Chains
from core.constants import IdCluster


class Clusterer:
    def __init__(self, config):
        self.config = config

    def __call__(self, chains: Chains, question: Optional[str] = None) -> list[IdCluster]:
        raise NotImplementedError("Must be implemented in subclasses")


class TrivialClusterer(Clusterer):
    def __call__(self, chains: Chains, question: Optional[str] = None) -> list[IdCluster]:
        """Returns a single cluster with all chains"""
        return [list(range(len(chains)))]