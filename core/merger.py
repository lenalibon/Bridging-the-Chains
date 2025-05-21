

from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from core.chain import Chain, Chains
from core.constants import IdCluster
from core.stepper import Stepper # type: ignore



from .utils import *
from .prompts import *

logger = get_logger()

class MergeFunction:
    """Return a single chain from a list of chains"""
    def __call__(self, chain_list: list[Chain]) -> Chain:
        raise NotImplementedError("Must should be implemented in subclasses")


class SummarizingMergeFunction(MergeFunction):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.stepper = Stepper(model, tokenizer)

    def __call__(self, chain_list: list[Chain]) -> Chain:
        """Returns a single chain being the result of an LLM call to summarize the chains"""
        # TODO test this
        # TODO take precautions against infinite while-looping
        prompt = self.prepare_summarizing_prompt(chain_list)
        debug_panel(logger, "Summarizing prompt", prompt)
        sum_chain = self.stepper.first_step_in_one(prompt)
        while not sum_chain.is_complete():
            sum_chain = self.stepper.next_step_in_one(sum_chain)
        return sum_chain
        
        
    def prepare_summarizing_prompt(self, chain_list: list[Chain]) -> str:
        question = chain_list[0].question
        cot_steps_json = ''.join([chain.jsonic_repr for chain in chain_list])
        prompt = SUMMARIZING_PROMPT_TEMPLATE.substitute(question=question, cot_steps=cot_steps_json)
        return prompt
    


class TrivialMergeFunction(MergeFunction):
    def __call__(self, chain_list: list[Chain]) -> Chain:
        """Returns the first (and only) chain in the list, which is expected to be a singleton"""
        assert len(chain_list) == 1
        return chain_list[0]
        

class Merger:
    """
    Merger receives the chains, (incl. their scores), a nested list of chain indices (clusters), and returns a set of chains (may be a singleton)
    """
    def __init__(self, merge_fn: Optional[MergeFunction]):
        """merge_fn is a function that takes a list of chains and returns a single chain (may be concatenation or summarization-based)"""
        self.merge_fn = merge_fn

    def __call__(self, chains: Chains, chain_id_clusters: list[IdCluster]) -> Chains:
        raise NotImplementedError("Must be implemented in subclasses")


class MergerMaxProb(Merger):
    """
    Select one representative chain per cluster –- based on highest probability P(a_i | question, prompt),
    *and* then merges them into a single chain by applying the merge_fn"""
    def __call__(self, chains: Chains, chain_id_clusters: list[IdCluster]) -> Chains:
        assert self.merge_fn
        return self.merge_fn([ self.get_max_prob_chain(chains, id_cluster) for id_cluster in chain_id_clusters ]).as_chains()

    def get_max_prob_chain(self, chains: Chains, id_cluster: IdCluster) -> Chain:
        return max((chains[i] for i in id_cluster), key=lambda chain: chain.get_log_prob())

class MergerClusterCentroid(Merger):
    def __call__(self, chains: Chains, chain_id_clusters: list[IdCluster]) -> Chains:
        # TODO persist and pass cluster centroids from similarity clusterer?
        assert self.merge_fn
        return self.merge_fn([ self.get_closest_to_cluster_centroids(chains, cluster) for cluster in chain_id_clusters ]).as_chains()

    def get_closest_to_cluster_centroids(self, chains: Chains, id_cluster: IdCluster) -> Chain:
        raise NotImplementedError("TODO")


class MergerWithinCluster(Merger):
    def __call__(self, chains, chain_id_clusters) -> Chains:
        assert self.merge_fn
        return Chains.from_list([ self.merge_fn(self.get_chain_cluster(chains, id_cluster)) for id_cluster in chain_id_clusters ])

    def get_chain_cluster(self, chains, id_cluster) -> list[Chain]:
        raise NotImplementedError("TODO")