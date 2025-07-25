

from typing import Optional

from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

from clustering.embedding import EmbeddingCluster
from core.chain import Chain, Chains, ListChains
from core.clusterer import Clusterer
from core.constants import *
from core.experiment_config import ExperimentConfig
from core.stepper import Stepper # type: ignore



from .utils import *
from prompting.prompts import *

logger = get_logger()

class MergeFunction:
    """Return a single chain from a list of chains"""
    def __call__(self, chain_list: list[Chain]) -> Chain:
        raise NotImplementedError("Must should be implemented in subclasses")


class SummarizingMergeFunction(MergeFunction):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: ExperimentConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        # NOTE: when merging use_cache=False to avoid issues with Gemma.
        # I have found that if the cache is large, Gemma becomes deranged and generates at most one valid step.
        # When merging, the cache is usually large because the prompt is long.
        # This may become an issue with AutoCoT as well :/
        self.stepper = Stepper(model, tokenizer, config = config)

    def __call__(self, chain_list: list[Chain]) -> Chain:
        """Returns a single chain being the result of an LLM call to summarize the chains"""
        # TODO take precautions against infinite while-looping
        prompt = self.prepare_summarizing_prompt(chain_list)
        debug_panel(logger, "Summarizing prompt", prompt)
        sum_chain = self.stepper.first_step_in_one_summarization(prompt)
        while not sum_chain.is_complete():
            sum_chain = self.stepper.next_step_in_one(sum_chain)
            #debug_panel("Generated lines", repr(sum_chain.get_generated_lines()))
        new_chain = self.make_new_chain(chain_list, sum_chain)
        return new_chain
        
    def prepare_summarizing_prompt(self, chain_list: list[Chain]) -> str:
        question = chain_list[0].question
        jsonic_solutions = '\n'.join([chain.jsonic_repr for chain in chain_list])
        prompt = SUMMARIZING_PROMPT_TEMPLATE.substitute(question=question, solutions=jsonic_solutions)
        return prompt

    def make_new_chain(self, chain_list: list[Chain], sum_chain: Chain) -> Chain:
        # FIXME: prompt template of the new chain is to be decided
        question = chain_list[0].question
        prompt = SIMPLE_PROMPT_TEMPLATE.substitute(question=question)
        new_prompt = prompt + '\n'.join(sum_chain.get_generated_lines())
        debug_panel(logger, "New merged prompt", new_prompt)
        token_ids = self.tokenizer(new_prompt, return_tensors="pt").input_ids.to(self.config.device) # type: ignore
        prompt_offset = token_len(self.tokenizer, prompt)
        new_chain = Chain(self.tokenizer, 
                          token_ids,
                          prompt_offset,
                          scores=None, # WARNING: log probs should not computed for merged chains
                          pkv=None,
                          index=max([chain.index for chain in chain_list]),
                          n_lines=len(sum_chain.get_generated_lines()),
                          question=chain_list[0].question)
        return new_chain


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
    def __init__(self, merge_fn: Optional[MergeFunction], clusterer: Clusterer):
        super().__init__(merge_fn)
        assert isinstance(clusterer, EmbeddingCluster), "MergerClusterCentroid only works with EmbeddingCluster"
        self.clusterer = clusterer

    def __call__(self, chains: Chains, chain_id_clusters: list[IdCluster]) -> Chains:
        assert self.merge_fn
        return self.merge_fn([ self.get_closest_to_cluster_centroids(chains, cluster, cluster_index) for cluster_index, cluster in enumerate(chain_id_clusters) ]).as_chains()

    def get_closest_to_cluster_centroids(self, chains: Chains, id_cluster: IdCluster, cluster_index: int) -> Chain:
        """
        For each cluster of chains, find the chain which is closest to the centroid
        """
        # 1. Compute cosine similarities for each chain
        centroid = self.clusterer.get_centroids()[cluster_index]
        closest_chain, max_cos_distance = None, -float('inf')
        for chain_id in id_cluster:
            chain = chains[chain_id]
            chain_embedding = self.clusterer.get_embeddings()[chain_id]
            cos_distance = cosine_similarity(chain_embedding, centroid, dim=0)
            if closest_chain is None or cos_distance > max_cos_distance:
                closest_chain = chain
                max_cos_distance = cos_distance
        
        return closest_chain
        


class MergerWithinCluster(Merger):
    def __call__(self, chains, chain_id_clusters) -> Chains:
        assert self.merge_fn
        return ListChains.from_list([ self.merge_fn(self.get_chain_cluster(chains, id_cluster)) for id_cluster in chain_id_clusters ])

    def get_chain_cluster(self, chains, id_cluster) -> list[Chain]:
        return [chains[i] for i in id_cluster]
    

class MergerMajorityThenMaxProb(Merger):
    """
    First select largest cluster (majority vote by size) then within that cluster select highest-P chain
    """
    def __call__(self, chains: Chains, chain_id_clusters: list[IdCluster]) -> Chains:        
        majority_cluster = max(chain_id_clusters, key=len)
        best_chain = max((chains[i] for i in majority_cluster), key=lambda chain: chain.get_log_prob()) 
        return best_chain.as_chains()
    
class MergerMajorityThenCentroid(Merger):
    """
    First select largest cluster (majority vote by size) then within that cluster select closest-to-centroid chain
    """
    def __init__(self, merge_fn: Optional[MergeFunction], clusterer: Clusterer):
        super().__init__(merge_fn)
        assert isinstance(clusterer, EmbeddingCluster), "MergerMajorityThenCentroid only works with EmbeddingCluster"
        self.clusterer = clusterer

    def __call__(self, chains: Chains, chain_id_clusters: list[IdCluster]) -> Chains:
        majority_cluster_index, majority_cluster = max(enumerate(chain_id_clusters), key=lambda x: len(x[1]))

        # 2. Find the chain closest to the centroid of this cluster
        centroid = self.clusterer.get_centroids()[majority_cluster_index]
        closest_chain = None
        max_cos_sim = -float("inf")

        for chain_id in majority_cluster:
            embedding = self.clusterer.get_embeddings()[chain_id]
            cos_sim = cosine_similarity(embedding, centroid, dim=0).item()
            if closest_chain is None or cos_sim > max_cos_sim:
                closest_chain = chains[chain_id]
                max_cos_sim = cos_sim

        return closest_chain.as_chains()