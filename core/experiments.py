

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


logger = get_logger()

class Experiment:
    def __init__(self,
                 model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 prompter: 'Prompter',
                 config: ExperimentConfig,
                 clusterer: Optional['Clusterer'] = None,
                 during_merger: Optional['Merger'] = None,
                 post_merger: Optional['Merger'] = None,
                 label = None,
                 n_init_chains: int = 1
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.clusterer = clusterer

        # Either during_merger or post_merger, not both
        assert (during_merger is None or post_merger is None) and (during_merger is not None or post_merger is not None), \
            "You can only use one of the during_merger or post_merger, not both at the same time."
        self.use_during_merger = during_merger is not None
        
        # Applied during generation, `merge_every` tells the number of chain steps after which we merge
        self.during_merger = during_merger
        self.merge_every = config.merge_every if during_merger else None
        if self.during_merger is not None:
            assert config.merge_every > 0, "merge_every must be greater than 0"

        # Applied to the chains after generation is complete
        self.post_merger = post_merger

        # The number of initial chains to generate, at first step.
        self.n_init_chains = n_init_chains 

        # The label is used in the filename with the experiment results
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
            if self.use_during_merger and counter % self.merge_every == 0:
                assert self.clusterer
                assert self.during_merger
                chain_id_clusters = self.clusterer(chains, question)
                chains = self.during_merger(chains, chain_id_clusters) # type: ignore
            chains = self.stepper.next_step_in_all(chains)
            counter += 1
        
        # Post merger
        if not self.use_during_merger:
            assert self.post_merger
            assert self.clusterer
            chain_id_clusters = self.clusterer(chains, question)
            chains = self.post_merger(chains, chain_id_clusters) # type: ignore

        return chains


class ExperimentB1(Experiment):
    """Generate n chains; pick highest-probability chain as the answer."""
    def __init__(self, model, tokenizer, prompter, config, **kwargs):
        # In the greedy baseline,
        # 1. "merging" happens after generation,
        # 2. and "merging" is really just selecting the highest-probability chain.
        super().__init__(model, tokenizer, prompter, config,
                         clusterer=TrivialClusterer(config),
                         post_merger=MergerMaxProb(TrivialMergeFunction()),
                         label="ExperimentB1",
                         n_init_chains=config.n_init_chains)

class ExperimentN1(Experiment):
    """Cluster once after k steps; pick highest-P chain per cluster."""
    def __init__(self, model, tokenizer, prompter, config, **kwargs):
        super().__init__(model, tokenizer, prompter, config, 
                         clusterer=EntailmentCluster(config),
                         during_merger = MergerMaxProb(SummarizingMergeFunction(model, tokenizer, config)),
                         label="ExperimentN1",
                         n_init_chains=config.n_init_chains
        )

# TODO starting from here, also change method -> experiment everywhere
class ExperimentN2(Experiment):
    "Cluster every k steps; summarize all chains in each cluster."
    def __init__(self, model, tokenizer, prompter, config, **kwargs):
        super().__init__(model, tokenizer, prompter, config, 
                         cluster=EntailmentCluster(config),
                         during_merger=

class EmbeddingMethodTest(Method):
    """Dummy method for verifying that embedding clustering works"""
    def __init__(self, model, tokenizer, prompter, config, **kwargs): 
        # TODO: use a more powerful model for summarizing, maybe with an API call
        clusterer = EmbeddingCluster(config)
        super().__init__(model, tokenizer, prompter, config,
                         clusterer=clusterer,
                         merger=MergerClusterCentroid(SummarizingMergeFunction(model, tokenizer, config), clusterer),
                         post_merger=MergerClusterCentroid(SummarizingMergeFunction(model, tokenizer, config), clusterer),
                         **kwargs)
