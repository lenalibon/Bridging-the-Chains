
import fire
import logging

from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, Cache

import torch
from torch import Tensor

from .utils import SIMPLE_PROMPT_TEMPLATE, get_newline_token_id, write_jsonl, get_timestamp, tensor_split


ChainTensor = torch.Tensor # size: (sequence_length,)
ChainsTensor = torch.Tensor # size: (batch_size, sequence_length)

# NOTE: for some reason torch stores scores as a tuple of tensors, not a single tensor
Scores = tuple[torch.Tensor] # size: (sequence_length, (batch_size, vocab_size))
IdCluster = list[int] # list of chain ids
DataGetter = Callable[[], tuple['Dataset', str]] # function that returns a dataset and its label


DEVICE = "cuda"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.9


class Chain:
    def __init__(self, tokenizer, token_ids: ChainTensor, prompt_offset: int):
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.prompt_offset = prompt_offset

    # TODO implement reference-based scores wrt Chains if needed for perf reasons

    def is_complete(self, end = '\n  ]\n') -> bool:
        max_end_len = len(end)
        pad_token = self.tokenizer.pad_token_id
        # NOTE: unpadding can be optimized
        decoded_chain_end = self.tokenizer.decode(self.token_ids[self.token_ids != pad_token][-max_end_len:])
        return decoded_chain_end.endswith(end)

    def get_lines_as_token_ids(self):
        tensors = tensor_split(self.token_ids, get_newline_token_id(self.tokenizer))
        return tensors

    def get_clean_text(self) -> str:
        decoded = self.tokenizer.decode(self.token_ids[self.prompt_offset:], skip_special_tokens=True)
        # HACK TODO refactor with regex
        pyrepr = "[" + decoded
        # logger.debug(f"pyrepr:\n{pyrepr}")
        steps: list[str] = eval(pyrepr)
        assert isinstance(steps, list), f"Expected a list, got {type(steps)}"
        return ' '.join(steps)

    def as_chains(self) -> 'Chains':
        raise NotImplementedError("TODO")
       
 
class Chains:
    """
    Representation of a batch of reasoning chains for a single question.
    Internally represented as a tensor of token ids, of shape (batch_size, sequence_length),
    where `batch_size` is the number of chains and `sequence_length` is the length of each chain (in tokens, including the prompt)

    Probability scores and past-key-values (KV-cache) are stored as well.
    """
    def __init__(self, tokenizer: AutoTokenizer, token_ids: ChainsTensor, prompt_offset: int,
                 scores: Optional[Scores] = tuple(), pkv: Optional[Cache] = None):
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.prompt_offset = prompt_offset
        # May be more efficient to use a list[Tensor] for scores
        self.scores = scores
        self.pkv = pkv
    
    def all_complete(self) -> bool:
        # NOTE: assuming a chain is complete if `  ]\n` was generated (omitting padding)
        # NOTE: this won't work if the model breaks the JSON format
        return all(self[i].is_complete() for i in range(len(self)))

    def __getitem__(self, chain_id: int) -> Chain:
        # TODO what about scores?
        return Chain(self.tokenizer, self.token_ids[chain_id], self.prompt_offset)

    def __len__(self) -> int:
        return self.token_ids.shape[0]

    @staticmethod
    def from_list(chain_list: list[Chain]):
        raise NotImplementedError("TODO")


# ### Summary of the planned Methods
# - BaselineGreedy: select the answer from the single chain with the highest probability without merging or clustering.
# - BaselineAggregate: aggregate all answers into a single output without clustering.
# - MethodMergingDuringGeneration: 
#   - MethodMergingRepresentatives:
#     - MethodMergingRepresentativesMaxProb
#     - MethodMergingRepresentativesClusterCentoid
#   - MethodMergingWithinCluster
# - MethodMergingAfterGeneration


class Method:
    def __init__(self, model, tokenizer: AutoTokenizer, prompter, clusterer, merger,
                 merge_every=False, merge_after=False, label=None):
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.clusterer = clusterer
        self.merger = merger
        self.merge_every = merge_every
        self.merge_after = merge_after
        self.label = label or self.__class__.__name__

        # WARNING: possibly not all tokenizers tokenize newlines the "right way": tokens for `\n`, `\n\n`, `\n\n\n`, etc.
        self.stop_token_ids = [tokenizer.convert_tokens_to_ids('\n'), tokenizer.eos_token_id]
        stop_tokens = tokenizer.convert_ids_to_tokens(self.stop_token_ids)
        logger.debug(f"{stop_tokens=}")
        self.gen_config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            cache_implementation=None,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=self.stop_token_ids,
        )
        # NOTE: On the difference between `scores` and `logits`, see https://huggingface.co/docs/transformers/v4.51.3/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput

    def generate_answer(self, question: str) -> Chain:
        # NOTE: assuming generate_answer returns a single chain
        prompt: str = self.prompter(question)
        logger.debug(f"==== prompt  ====\n {prompt}")
        logger.debug(f"=================")
        chains: Chains = self.first_step(prompt)
        counter = 1
        while not chains.all_complete():
            if self.merge_every and counter % self.merge_every == 0:
                chain_id_clusters = self.clusterer(chains)
                # chain_clusters = [[chains[chain_id] for chain_id in id_cluster] for id_cluster in chain_id_clusters]
                # chains = Chains.from_list([ self.merger(cluster, offset=prompt_offset) for cluster in chain_clusters ])
                chains = self.merger(chains)
            chains = self.next_step(chains)
            counter += 1
        
        if self.merge_after:
            raise NotImplementedError("TODO")

        if len(chains) == 1:
            return chains[0]
        else:
            raise NotImplementedError("TODO")
            

    def first_step(self, prompt: str) -> Chains:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        prompt_offset = input_ids.shape[1]
        out = self.model.generate(input_ids, past_key_values=None, generation_config=self.gen_config)
        chains = Chains(self.tokenizer,
                        out.sequences,
                        prompt_offset=prompt_offset,
                        scores=out.scores,
                        pkv=out.past_key_values)
        self.print_tokens(chains)
        return chains
    

    def next_step(self, chains: Chains) -> Chains:
        out = self.model.generate(chains.token_ids,
                                  cache_implementation=None,
                                  past_key_values=chains.pkv,
                                  generation_config=self.gen_config)
        chains.pkv = out.past_key_values
        chains.token_ids = out.sequences
        # Append the new scores (tuple of tensors) to the existing scores
        assert chains.scores is not None, f"Scores should not be None, got {chains.scores}"
        chains.scores += out.scores
        self.print_tokens(chains)
        return chains

    def print_tokens(self, chains: Chains) -> None:
        # logger.debug(f"{chains.prompt_offset=}")
        for i, text in enumerate(self.tokenizer.batch_decode(chains.token_ids[:, chains.prompt_offset:])):
            logger.debug(f"=== next[{i}] ===\n {text}")
            logger.debug(f"=================")
    

class MergeFunction:
    """Return a single chain from a list of chains"""
    def __call__(self, chain_list: list[Chain]) -> Chain:
        raise NotImplementedError("MergeFunction __call__ should be implemented in subclasses")


class SummarizingMergeFunction(MergeFunction):
    def __call__(self, chain_list: list[Chain]) -> Chain:
        raise NotImplementedError("TODO")
        

class Merger:
    """
    Merger receives the chains, (incl. their scores), a nested list of chain indices (clusters), and returns a set of chains (may be a singleton)
    """
    def __init__(self, merge_fn: MergeFunction):
        """merge_fn is a function that takes a list of chains and returns a single chain (may be concatenation or summarization-based)"""
        self.merge_fn = merge_fn

    def __call__(self, chains: Chains, chain_id_clusters: list[IdCluster]) -> Chains:
        raise NotImplementedError("Merger __call__ should be implemented in subclasses")


class MergerMaxProb(Merger):
    def __call__(self, chains: Chains, chain_id_clusters: list[IdCluster]) -> Chains:
        return self.merge_fn([ self.get_max_prob_chain(chains, id_cluster) for id_cluster in chain_id_clusters ]).as_chains()

    def get_max_prob_chain(self, chains: Chains, id_cluster: IdCluster) -> Chain:
        raise NotImplementedError("TODO")


class MergerClusterCentroid(Merger):
    def __call__(self, chains: Chains, chain_id_clusters: list[IdCluster]) -> Chains:
        # TODO persist and pass cluster centroids from similarity clusterer?
        return self.merge_fn([ self.get_closest_to_cluster_centroids(chains, cluster) for cluster in chain_id_clusters ]).as_chains()

    def get_closest_to_cluster_centroids(self, chains: Chains, id_cluster: IdCluster) -> Chain:
        raise NotImplementedError("TODO")


class MergerWithinCluster(Merger):
    def __call__(self, chains, chain_id_clusters) -> Chains:
        return Chains.from_list([ self.merge_fn(self.get_chain_cluster(chains, id_cluster)) for id_cluster in chain_id_clusters ])

    def get_chain_cluster(self, chains, id_cluster) -> list[Chain]:
        raise NotImplementedError("TODO")
    

class BaselineGreedy(Method):
    """Select the answer from the single chain with the highest probability without merging or clustering"""
    def __init__(self, model, tokenizer, prompter, **kwargs):
        raise NotImplementedError("TODO")
        super().__init__(model, tokenizer, prompter, None, None, **kwargs)
        self.clusterer = None
        self.merger = None


class BaselineSimple(Method):
    "No branching, no clustering, no merging"
    def __init__(self, model, tokenizer, prompter, **kwargs):
        super().__init__(model, tokenizer, prompter, None, None, **kwargs)
        self.clusterer = None
        self.merger = None


class SimplePrompter:
    def __call__(self, question):
        return SIMPLE_PROMPT_TEMPLATE.format(question=question)


class Experiment:
    def eval(self, **kwargs):
        logger.info("Running full evaluation...")
        data_getters: list[DataGetter] = [partial(self.get_gsm8k, n=3)] # FIXME delete n=
        model, tokenizer = self.get_model_and_tokenizer()
        prompter = SimplePrompter() # FIXME: TODO use autocot
        methods = [BaselineSimple(model, tokenizer, prompter, label="BaselineSimple-1b-it-8bit_no-autocot")]
        for get_data in data_getters:
            eval_data, dataset_label = get_data()
            for method in methods:
                eval_label = f"{method.label}___{dataset_label}"
                self.eval_method(method, eval_data, label=eval_label)

    def eval_method(self, method, eval_data, label="eval"):
        results = []
        for i, sample in enumerate(eval_data):
            question = sample['question']
            true_answer = sample['answer']
            pred_chain: Chain = method.generate_answer(question)
            # NOTE: roscoe's gsm8k.json is different
            results.append({
                "premise": question,
                "reasoning": pred_chain.get_clean_text(),
                "true_answer": true_answer,
            })
        Path("results").mkdir(parents=True, exist_ok=True)
        ts = get_timestamp()
        write_jsonl(results, f"results/{label}___{ts}.jsonl")
        
    def get_gsm8k(self, n=None) -> tuple['Dataset', str]:
        # NOTE: using the train split for evaluation
        split = datasets.load_dataset("gsm8k", "main")["test"]
        if n is not None:
            split = split.select(range(n))
        return (split, "gsm8k")

    def get_model_and_tokenizer(self):
        model_name = "google/gemma-3-1b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
        return model, tokenizer
    

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug("logger initialized!")

           
# Usage: python -m core.main
if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    fire.Fire(Experiment().eval)
