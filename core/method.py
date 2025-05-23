

from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from clustering.embedding import EmbeddingCluster
from core.chain import Chains, ListChains
from core.clusterer import Clusterer, TrivialClusterer
from core.experiment_config import ExperimentConfig
from core.merger import Merger, MergerClusterCentroid, MergerMaxProb, SummarizingMergeFunction, TrivialMergeFunction
from core.prompter import Prompter
from core.stepper import Stepper # type: ignore



from .utils import *

# ### Summary of the planned Methods
# - [DONE] BaselineGreedy: select the answer from the single chain with the highest probability without merging or clustering.
# - [WIP] BaselineAggregate: aggregate all answers into a single output without clustering.
# - MethodMergingDuringGeneration: 
#   - MethodMergingRepresentatives:
#     - MethodMergingRepresentativesMaxProb
#     - MethodMergingRepresentativesClusterCentoid
#   - MethodMergingWithinCluster
# - MethodMergingAfterGeneration

logger = get_logger()

class Method:
    def __init__(self,
                 model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 prompter: 'Prompter',
                 config: ExperimentConfig,
                 clusterer: Optional['Clusterer'] = None,
                 merger: Optional['Merger'] = None,
                 post_merger: Optional['Merger'] = None,
                 label = None,
                 n_init_chains: int = 1
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.clusterer = clusterer
        # The first merger `merger` is applied during generation, if `merge_every` is set to a positive integer.
        self.merger = merger
        # The second merger, `post_merger`, is applied to the chains after generation is complete, if `merge_after==True`.
        # Why not one merger? for flexibility
        self.post_merger = post_merger
        # The number of initial chains to generate, at first step.
        self.n_init_chains = n_init_chains 
        self.merge_every = config.merge_every
        self.merge_after = config.merge_after 
        # The label is used in the filename with the method results
        self.label = label or self.__class__.__name__
        self.stepper = Stepper(model, tokenizer, config = config)

        # WARNING: possibly not all tokenizers tokenize newlines the "right way": tokens for `\n`, `\n\n`, `\n\n\n`, etc.
        # self.stop_token_ids = [tokenizer.convert_tokens_to_ids('\n'), tokenizer.eos_token_id]
        # logger.debug(f"{self.stop_token_ids=}")
        # stop_tokens = tokenizer.convert_ids_to_tokens(self.stop_token_ids)
        # logger.debug(f"{stop_tokens=}")

    def generate_answer(self, question: str) -> Chains:
        """
        Given the question generate one or possibly multiple complete chains.
        """
        set_seed(42)
        assert self.prompter
        chains: ListChains = self.stepper.first_step_in_all(prompter = self.prompter, question=question, n=self.n_init_chains)
        counter = 1
        while not chains.all_complete():
            if self.merge_every and counter % self.merge_every == 0:
                assert self.clusterer
                assert self.merger
                chain_id_clusters = self.clusterer(chains, question)
                chains = self.merger(chains, chain_id_clusters) # type: ignore
            chains = self.stepper.next_step_in_all(chains)
            counter += 1
        
        if self.merge_after:
            assert self.post_merger
            assert self.clusterer
            chain_id_clusters = self.clusterer(chains, question)
            chains = self.post_merger(chains, chain_id_clusters) # type: ignore

        return chains


class BaselineGreedy(Method):
    """Select the answer from the single chain with the highest probability without merging or clustering"""
    def __init__(self, model, tokenizer, prompter, config, **kwargs):
        # In the greedy baseline,
        # 1. "merging" happens after generation,
        # 2. and "merging" is really just selecting the highest-probability chain.
        super().__init__(model, tokenizer, prompter, config,
                         clusterer=TrivialClusterer(),
                         post_merger=MergerMaxProb(TrivialMergeFunction()),
                         **kwargs)

class BaselineAggregation(Method):
    """Aggregation Approach: Aggregating all answers into a single output without clustering."""
    def __init__(self, model, tokenizer, prompter, config, **kwargs):
        super().__init__(model, tokenizer, prompter, config,
                         clusterer=TrivialClusterer(),
                         post_merger=Merger(SummarizingMergeFunction(model, tokenizer, config)),
                         **kwargs)


class BaselineSimple(Method):
    "No branching, no clustering, no merging"
    def __init__(self, model, tokenizer, prompter, config, **kwargs):
        super().__init__(model, tokenizer, prompter, config, **kwargs)

class EmbeddingMethodTest(Method):
    """Dummy method for verifying that embedding clustering works"""
    def __init__(self, model, tokenizer, prompter, config, **kwargs): 
        # TODO: use a more powerful model for summarizing, maybe with an API call
        clusterer = EmbeddingCluster()
        super().__init__(model, tokenizer, prompter, config,
                         clusterer=clusterer,
                         merger=MergerClusterCentroid(SummarizingMergeFunction(model, tokenizer, config), clusterer),
                         post_merger=MergerClusterCentroid(SummarizingMergeFunction(model, tokenizer, config), clusterer),
                         **kwargs)
