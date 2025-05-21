from core.chain import Chains
from core.constants import IdCluster

from .utils import *
from .prompts import *

class Clusterer:
    def __call__(self, chains: Chains) -> list[IdCluster]:
        raise NotImplementedError("Must be implemented in subclasses")


class TrivialClusterer(Clusterer):
    def __call__(self, chains: Chains) -> list[IdCluster]:
        """Returns a single cluster with all chains"""
        return [list(range(len(chains)))]