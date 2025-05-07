from typing import Iterable
import fire
import datasets
import torch
import json

import itertools

from datetime import datetime
from functools import partial
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

ChainTensor = torch.Tensor # ?
# Chains = Iterable[Chain]
ChainsTensor = torch.Tensor

DEVICE = "cuda"


class ChainsTensorUtils:
    """
    Methods for working with chains (as a tensor of token ids). 
    Chains are assumed to be represented as a tensor of shape (batch_size, sequence_length),
    where `batch_size` is the number of chains and `sequence_length` is the length of each chain (in tokens, including the prompt)
    """
    @staticmethod
    def all_complete(chains: ChainsTensor) -> bool:
        raise NotImplementedError("TODO")

    @staticmethod
    def get(chains: ChainsTensor, chain_id: int) -> ChainTensor:
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
    def __init__(self, model, tokenizer, prompter, clusterer, merger,
                 merge_every=False, merge_after=False, label=None,
                 gen_config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.clusterer = clusterer
        self.merger = merger
        self.merge_every = merge_every
        self.merge_after = merge_after
        self.label = label or self.__class__.__name__

        stop_token_ids = tokenizer.convert_tokens_to_ids('\n') + [tokenizer.eos_token_id]
        gen_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.9,
            cache_implementation=None,
            return_dict_in_generate=True,
            eos_token_id=stop_token_ids,
        )

 

    def generate_answer(self, question):
        prompt: str = self.prompter(question)
        prompt_offset: int = tokenlen(self.tokenizer, prompt)
        chains: ChainsTensor
        scores: torch.Tensor
        chains, scores = self.first_step(prompt)
        counter = 1
        while not ChainsTensorUtils.all_complete(chains):
            if self.merge_every and counter % self.merge_every == 0:
                chain_id_clusters = self.cluster(chains)
                # chain_clusters = [[ChainsTensorUtils.get(chains, chain_id) for chain_id in id_cluster] for id_cluster in chain_id_clusters]
                # chains = ChainsTensorUtils.from_list([ self.merger(cluster, offset=prompt_offset) for cluster in chain_clusters ])
                chains = self.merger(chains, scores, offset=prompt_offset)
            chains, scores = self.next_step(chains)
            counter += 1
        
        if self.merge_after:
            raise NotImplementedError("TODO")

    def first_step(self, prompt: str) -> tuple[ChainsTensor, torch.Tensor]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        pkv = None
        out = self.model.generate(input_ids,
                             past_key_values=pkv,
                             generation_config=self.gen_config)
        pkv = out.past_key_values
        input_ids = out.sequences
           




class MergeFunction:
    def __call__(self, chains: list[ChainTensor], offset: int = 0) -> ChainTensor:
        raise NotImplementedError("MergeFunction __call__ should be implemented in subclasses")


class SummarizingMergeFunction(MergeFunction):
    def __call__(self, chains: list[ChainTensor], offset: int = 0) -> ChainTensor:
        raise NotImplementedError("TODO")
        

class Merger:
    """
    Merger receives the chains, their scores, a nested list of chain indices (clusters), and returns a set of chains (may be a singleton)
    """
    def __init__(self, merge_fn):
        """merge_fn is a function that takes a list of chains and returns a single chain (may be concatenation or summarization-based)"""
        self.merge_fn = merge_fn

    def __call__(self, chains, scores, chain_id_clusters, offset: int = 0) -> ChainsTensor:
        raise NotImplementedError("Merger __call__ should be implemented in subclasses")


class MergerMaxProb(Merger):
    def __call__(self, chains, scores, chain_id_clusters, offset = 0):
        return self.merge_fn([ self.get_max_prob_chain(chains, scores, cluster, offset=offset) for cluster in chain_id_clusters ])

    def get_max_prob_chain(self, chains, scores, cluster, offset: int = 0) -> ChainTensor:
        raise NotImplementedError("TODO")


class MergerClusterCentroid(Merger):
    def __call__(self, chains, scores, chain_id_clusters, offset: int = 0):
        # TODO persist cluster centroid from clusterer
        return self.merge_fn([ self.get_closest_to_cluster_centroids(chains, cluster, offset=offset) for cluster in chain_id_clusters ])

    def get_closest_to_cluster_centroids(self, chains, cluster, offset: int = 0) -> ChainTensor:
        raise NotImplementedError("TODO")


class MergerWithinCluster(Merger):
    def __call__(self, chains, scores, chain_id_clusters, offset: int = 0):
        return ChainsTensorUtils.from_list([ self.merge_fn(self.get_chain_cluster(chains, id_cluster)) for id_cluster in chain_id_clusters ])

    def get_chain_cluster(self, chains, id_cluster) -> list[ChainTensor]:
        raise NotImplementedError("TODO")
    

class BaselineGreedy(Method):
    """Select the answer from the single chain with the highest probability without merging or clustering"""
    def __init__(self, model, tokenizer, prompter, **kwargs):
        super().__init__(model, tokenizer, prompter, None, None, **kwargs)
        self.clusterer = None
        self.merger = None


class Experiment:
    def eval(self, **kwargs):
        print("Running full evaluation...")
        data_getters = [partial(self.get_gsm8k, n=10)] # FIXME delete n=10
        model, tokenizer = self.get_model_and_tokenizer()
        prompter = lambda x: x # FIXME: TODO use autocot
        methods = [BaselineGreedy(model, tokenizer, prompter, label="BaselineGreedy_gemma3-1b-it-8bit_no-autocot")]
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
            pred_chain = method.generate_answer(question)
            # NOTE: roscoe's gsm8k.json is different
            results.append({
                "premise": question,
                "reasoning": pred_chain,
                "true_answer": true_answer,
            })
        Path("results").mkdir(parents=True, exist_ok=True)
        ts = get_timestamp()
        write_jsonl(results, f"results/{label}___{ts}.jsonl")
        
    def get_gsm8k(self, n=None):
        # NOTE: using the train split for evaluation
        split = datasets.load_dataset("gsm8k", "main")["test"]
        if n is not None:
            split = split.select(range(n))
        return split, "gsm8k"

    def get_model_and_tokenizer(self):
        model_name = "google/gemma-3-1b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
        return model, tokenizer
    

def tokenlen(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt)["input_ids"])


def write_jsonl(data, path, indent=None):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, indent=None) + "\n")


def get_timestamp():
    return datetime.now().strftime('%Y%m%d%H%M')


def flat(nested):
    return itertools.chain.from_iterable(nested)
        
           
if __name__ == "__main__":
    fire.Fire(Experiment)
