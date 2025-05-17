import fire
import logging
import re

from rich.logging import RichHandler

from functools import partial
from string import Template
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, Cache, HybridCache

from rich.panel import Panel
from rich.logging import RichHandler

import gc
import torch
import torch.nn.functional as F
from torch import Tensor

from .utils import *
from .prompts import *


TokenIdsTensor = torch.Tensor # size: (sequence_length,)
BatchedTokenIdsTensor = torch.Tensor # size: (batch_size, sequence_length)

# NOTE: for some reason torch stores scores as a tuple of tensors, not a single tensor
Scores = tuple[torch.Tensor] # size: (sequence_length, (batch_size, vocab_size))
IdCluster = list[int] # list of chain ids
DataGetter = Callable[[], tuple['Dataset', str]] # function that returns a dataset and its label


DEVICE = "cuda"
TEMPERATURE = 0.7

MAX_STEPS = 8
MAX_TOKENS_PER_STEP = 100


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


clear_cache()


def set_seed(seed: int):
    torch.manual_seed(42)


class Chain:
    """
    A single reasoning chain, represented as a tensor of token ids, of shape (1, sequence_length,)
    where `sequence_length` is the length of the chain (in tokens, including the prompt), as well as the prompt offset.
    """
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 token_ids: BatchedTokenIdsTensor,
                 prompt_offset: int,
                 scores: Optional[Scores] = tuple(),
                 pkv: Optional[Cache] = None,
                 index: Optional[int] = None,
                 n_lines: Optional[int] = None,
                 question: Optional[str] = None,
                ):
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.prompt_offset = prompt_offset
        self.scores = scores
        self.pkv = pkv
        self.index = index
        # number of lines generated so far
        self.n_lines = n_lines
        self.question = question

    def is_complete_v0(self, ends = ['\n  ]\n', '\n  ]\n}', '\n\n', '<|endoftext|>']) -> bool:
        """Check if the chain is complete, if it ends with one of the given strings"""
        max_end_len = len(max(ends, key=len))
        decoded_chain_end = self.tokenizer.decode(self.token_ids[0, -max_end_len:])
        logger.debug(f"{self.short_repr} ends with: {repr(decoded_chain_end)}")
        is_complete = any(decoded_chain_end.endswith(end) for end in ends)
        if is_complete:
            logger.debug(f"{self.short_repr} is complete!")
        return is_complete

    @cache
    def is_complete(self) -> bool:
        '''
        Return False if the last generated line is a valid JSONic step of the form `"...",` (meaning the chain is still in generation),
        and True otherwise (meaning the chain is complete).
        '''
        # NOTE: this may be more expensive than v0, but more robust to hallucinations in which the model breaks the JSON format
        lines = self.get_generated_text().split('\n')
        # We know that the last line of split must be empty, because we are using a stop string \n
        assert lines[-1] == '' # TODO handle failure gracefully
        # logger.debug(f"{lines=}")
        last_line = lines[-2]
        logger.debug(f"{self.short_repr} ends with line: {repr(last_line)}")
        pattern = r"^\s*\".*\",?\s*$"
        is_complete = re.fullmatch(pattern, last_line) is None
        logger.debug(f"{self.short_repr} complete? {is_complete}")
        return is_complete

    def get_full_text(self, skip_special_tokens=True) -> str:
        return self.tokenizer.decode(self.token_ids[0],
                                     skip_special_tokens=skip_special_tokens)

    def get_generated_text(self, skip_special_tokens=True) -> str:
        return self.tokenizer.decode(self.token_ids[0, self.prompt_offset:],
                                     skip_special_tokens=skip_special_tokens)

    def get_clean_text(self) -> str:
        return '\n'.join(self.get_generated_steps())

    def get_generated_lines(self) -> list[str]:
        return self.get_generated_text().split('\n')

    @property
    def jsonic_repr(self) -> str:
        steps = self.get_generated_steps()
        quoted_steps = [f'"{step}"' for step in steps]
        if not self.is_complete():
            quoted_steps.append('...')
        return CHAIN_JSON_TEMPLATE.substitute(
            question=self.question,
            cot_steps=(',\n' + ' '*4).join(quoted_steps)
        )

    def get_generated_steps(self) -> list[str]:
        steps = [self.line_to_step(line) for line in self.get_generated_lines()]
        steps = [step for step in steps if step is not None]
        return steps

    def line_to_step(self, line: str) -> Optional[str]:
        pattern = r'^\s*"(.*)",?\s*$'
        match = re.match(pattern, line)
        return match.group(1) if match else None

    def as_chains(self) -> 'ListChains':
        return ListChains([self])

    def get_log_prob(self) -> float:
        assert self.scores, f"Scores are not available for this chain: {self.scores=}"
        scores_tuple = self.scores
        total_log_prob = 0.0
        target_token_ids = self.token_ids[0, self.prompt_offset:] # shape (sequence_length,)
        vocab_size = len(self.tokenizer)

        if len(scores_tuple) != len(target_token_ids):
            raise ValueError("Length of scores_tuple must match length of target_token_ids.")

        for i in range(len(scores_tuple)):
            step_logits = scores_tuple[i][0] # shape (vocab_size,)

            # Calculate log probabilities for all tokens in vocab for this step
            log_probs_at_step = F.log_softmax(step_logits, dim=-1)
            # assert log_probs_at_step.shape == (vocab_size,), f"Expected shape {(vocab_size,)}, got {log_probs_at_step.shape}"

            actual_token_id = target_token_ids[i]
            total_log_prob += log_probs_at_step[actual_token_id].item()

        logger.debug(f"{self.short_repr} has total log prob: {total_log_prob}")

        return total_log_prob

    @property
    def short_repr(self) -> str:
        return f"[Chain#{self.index} @ {self.n_lines} lines]"

    def __hash__(self):
        return hash_tensor(self.token_ids)
        

class Chains:
    """Abstract class for chains of the same prompt"""
    def all_complete(self) -> bool:
        # NOTE: assuming a chain is complete if `  ]\n` was generated (omitting padding)
        # NOTE: this won't work if the model breaks the JSON format
        return all(self[i].is_complete() for i in range(len(self)))

    def __getitem__(self, chain_id: int) -> Chain:
        raise NotImplementedError("Must be implemented in subclasses")

    def __len__(self) -> int:
        raise NotImplementedError("Must be implemented in subclasses")

    @staticmethod
    def from_list(chain_list: list[Chain]) -> 'Chains':
        raise NotImplementedError("Must be implemented in subclasses")


class ListChains(Chains):
    """Chains, (trivially) as a list of chains, *not* intended for batch-processing"""
    def __init__(self, chain_list: list[Chain]):
        self.list = chain_list

    def __getitem__(self, chain_id: int) -> Chain:
        return self.list[chain_id]

    def __len__(self) -> int:
        return len(self.list)
    
    @staticmethod
    def from_list(chain_list: list[Chain]) -> 'ListChains':
        return ListChains(chain_list)

    def __iter__(self):
        return iter(self.list)

 
# DEPRECATED
class BatchedTensorChains(Chains):
    """
    Representation of a batch of reasoning chains for a single prompt.
    Internally represented as a tensor of token ids, of shape (batch_size, sequence_length),
    where `batch_size` is the number of chains and `sequence_length` is the length of each chain (in tokens, including the prompt)

    Probability scores and past-key-values (KV-cache) are stored as well.

    NOTE: does not work! padding is the problem. we can revive this class if really needed.
    """
    def __init__(self, tokenizer: AutoTokenizer, token_ids: BatchedTokenIdsTensor, prompt_offset: int,
                 scores: Optional[Scores] = tuple(), pkv: Optional[Cache] = None):
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.prompt_offset = prompt_offset
        # May be more efficient to use a list[Tensor] for scores
        self.scores = scores
        self.pkv = pkv
    
    def __getitem__(self, chain_id: int) -> Chain:
        # TODO what about scores?
        return Chain(self.tokenizer, self.token_ids[chain_id], self.prompt_offset)

    def __len__(self) -> int:
        return self.token_ids.shape[0]



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
    def __init__(self,
                 model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 prompter: 'Prompter',
                 clusterer: Optional['Clusterer'] = None,
                 merger: Optional['Merger'] = None,
                 post_merger: Optional['Merger'] = None,
                 n_init_chains: int = 1,
                 merge_every: int | bool = False,
                 merge_after: bool = False,
                 label = None):
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
        self.merge_every = merge_every
        self.merge_after = merge_after
        # The label is used in the filename with the method results
        self.label = label or self.__class__.__name__

        # WARNING: possibly not all tokenizers tokenize newlines the "right way": tokens for `\n`, `\n\n`, `\n\n\n`, etc.
        # self.stop_token_ids = [tokenizer.convert_tokens_to_ids('\n'), tokenizer.eos_token_id]
        # logger.debug(f"{self.stop_token_ids=}")
        # stop_tokens = tokenizer.convert_ids_to_tokens(self.stop_token_ids)
        # logger.debug(f"{stop_tokens=}")

        stop_string = "\n"
        self.gen_config = GenerationConfig(
            max_new_tokens=MAX_TOKENS_PER_STEP,
            do_sample=True,
            temperature=TEMPERATURE,
            cache_implementation=None,
            return_dict_in_generate=True,
            output_scores=True,
            # eos_token_id=self.stop_token_ids,
            stop_strings=stop_string,
        )
        # NOTE: On the difference between `scores` and `logits`, see https://huggingface.co/docs/transformers/v4.51.3/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput

    def generate_answer(self, question: str) -> Chains:
        """
        Given the question generate one or possibly multiple complete chains.
        """
        set_seed(42)
        assert self.prompter
        prompt: str = self.prompter(question)
        debug_panel("Prompt", prompt)
        chains: ListChains = self.first_step_in_all(prompt, question=question)
        counter = 1
        while not chains.all_complete():
            if self.merge_every and counter % self.merge_every == 0:
                assert self.clusterer
                assert self.merger
                chain_id_clusters = self.clusterer(chains)
                chains = self.merger(chains, chain_id_clusters) # type: ignore
            chains = self.next_step_in_all(chains)
            counter += 1
        
        if self.merge_after:
            assert self.post_merger
            assert self.clusterer
            chain_id_clusters = self.clusterer(chains)
            chains = self.post_merger(chains, chain_id_clusters) # type: ignore

        return chains

    def first_step_in_all(self, prompt: str, **kw) -> ListChains:
        return ListChains.from_list([self.first_step_in_one(prompt, index=index, **kw) for index in range(self.n_init_chains)])

    def first_step_in_one(self, prompt: str, index: Optional[int] = None, question: Optional[str] = None) -> Chain:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        prompt_offset = input_ids.shape[1]
        pkv = prepare_pkv(self.model, prompt_offset)
        out = self.model.generate(input_ids,
                                  past_key_values=pkv,
                                  generation_config=self.gen_config,
                                  tokenizer=self.tokenizer,
                                  cache_implementation=None
                                  )
        chain = Chain(self.tokenizer,
                      out.sequences,
                      prompt_offset=prompt_offset,
                      scores=out.scores,
                      pkv=out.past_key_values,
                      index=index,
                      n_lines=1,
                      question=question)
        self.print_tokens(chain, skip_offset=True)
        return chain

    def next_step_in_all(self, chains: ListChains) -> ListChains: 
        return ListChains.from_list([self.maybe_next_step_in_one(chain) for chain in chains])

    def maybe_next_step_in_one(self, chain: Chain) -> Chain:
        if chain.is_complete():
            logger.debug(f"{chain.short_repr} is complete, skipping...")
            return chain
        else:
            return self.next_step_in_one(chain)

    def next_step_in_one(self, chain: Chain) -> Chain:
        # assert chains.token_ids[0, 0] == self.tokenizer.bos_token_id, f"Expected <bos> token at the beginning, got {chains.token_ids[0, 0]}"
        # input_token_ids = chains.token_ids[:, 1:] # skip <bos>
        # logger.debug(f"{self.tokenizer.pad_token_id=}")
        # input_token_ids = shift_padding_left(chains.token_ids, pad_token=self.tokenizer.pad_token_id)
        input_token_ids = chain.token_ids
        out = self.model.generate(input_token_ids,
                                #   attention_mask=(input_token_ids != self.tokenizer.pad_token_id),
                                  past_key_values=chain.pkv,
                                  generation_config=self.gen_config,
                                  tokenizer=self.tokenizer,
                                  cache_implementation=None,
                                  )
        chain.pkv = out.past_key_values
        chain.token_ids = out.sequences
        # Append the new scores (tuple of tensors) to the existing scores
        chain.scores += out.scores
        assert chain.n_lines
        chain.n_lines += 1
        self.print_tokens(chain, skip_offset=True)
        return chain

    def print_tokens(self, chain: Chain, jsonic=True, skip_offset=False) -> None:
        if jsonic:
            text = chain.jsonic_repr
        else:
            offset = chain.prompt_offset if skip_offset else 0
            to_decode = chain.token_ids[0, offset:]
            text = self.tokenizer.decode(to_decode)
        debug_panel(f"{chain.short_repr}", text)


class MergeFunction:
    """Return a single chain from a list of chains"""
    def __call__(self, chain_list: list[Chain]) -> Chain:
        raise NotImplementedError("Must should be implemented in subclasses")


class SummarizingMergeFunction(MergeFunction):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, chain_list: list[Chain]) -> Chain:
        """Returns a single chain being the result of an LLM call to summarize the chains"""
        prompt = self.prepare_summarizing_prompt(chain_list)
        debug_panel("Summarizing prompt", prompt)
        raise NotImplementedError("TODO")
        
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
    

class BaselineGreedy(Method):
    """Select the answer from the single chain with the highest probability without merging or clustering"""
    def __init__(self, model, tokenizer, prompter, **kwargs):
        # In the greedy baseline,
        # 1. "merging" happens after generation,
        # 2. and "merging" is really just selecting the highest-probability chain.
        super().__init__(model, tokenizer, prompter,
                         clusterer=TrivialClusterer(),
                         merge_after=True,
                         post_merger=MergerMaxProb(TrivialMergeFunction()),
                         **kwargs)


class BaselineSimple(Method):
    "No branching, no clustering, no merging"
    def __init__(self, model, tokenizer, prompter, **kwargs):
        super().__init__(model, tokenizer, prompter, **kwargs)


class Clusterer:
    def __call__(self, chains: Chains) -> list[IdCluster]:
        raise NotImplementedError("Must be implemented in subclasses")


class TrivialClusterer(Clusterer):
    def __call__(self, chains: Chains) -> list[IdCluster]:
        """Returns a single cluster with all chains"""
        return [list(range(len(chains)))]


# TODO adapt for needs of AutoCoT: e.g. does AutoCoT expect question index?
class Prompter:
    def __call__(self, *args, **kwargs) -> str:
        raise NotImplementedError("Must be implemented in subclasses")


class SimplePrompter(Prompter):
    def __init__(self, template: Template = SIMPLE_PROMPT_TEMPLATE):
        self.template = template

    def __call__(self, question):
        return self.template.substitute(question=question)


class Experiment:
    def eval(self, **kwargs):
        logger.info("Running full evaluation...")
        # NOTE: A DataGetter is a function that returns a tuple with dataset and its label.
        # We might want to consider a wrapper class around datasets.Dataset instead.
        data_getters: list[DataGetter] = [partial(self.get_gsm8k, n=1)] # FIXME delete n=
        model, tokenizer = self.get_model_and_tokenizer()
        prompter = SimplePrompter() # FIXME: TODO use autocot
        methods = [
            BaselineGreedy(model, tokenizer, prompter, n_init_chains=8, label=f"BaselineGreedy-8"),
            # BaselineSimple(model, tokenizer, prompter, n_init_chains=4, label=f"BaselineSimple"),
        ]
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
            pred_chains: Chains = method.generate_answer(question)
            # NOTE: roscoe's gsm8k.json is different
            # FIXME: only the first chain is written! I am not sure what we should do if there are more chains. -sb
            clean_text = pred_chains[0].get_clean_text()
            debug_panel("Predicted Answer", pred_chains[0].get_generated_text())
            debug_panel("True Answer", true_answer)
            results.append({
                "premise": question,
                "reasoning": clean_text,
                "true_answer": true_answer,
            })
        Path("results").mkdir(parents=True, exist_ok=True)
        ts = get_timestamp()
        write_jsonl(results, f"results/{label}___{ts}.jsonl")
        
    def get_gsm8k(self, n=None) -> tuple['Dataset', str]:
        # NOTE: using the train split for evaluation
        split = datasets.load_dataset("gsm8k", "main")["test"]
        label = "gsm8k"
        if n is not None:
            split = split.select(range(n))
            label = f"gsm8k-first{n}"
        return (split, label)

    def get_model_and_tokenizer(self):
        # TODO adapt code for Gemma (fix caching crashes)
        model_name = "google/gemma-3-1b-it"
        # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        # logger.debug("padding_side=left")
        # TODO use_fast=True
        # model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        return model, tokenizer


def prepare_pkv(model, n_input_tokens):
    model_name = model.__class__.__name__
    if 'Gemma' in model_name:
        max_cache_len = n_input_tokens + MAX_STEPS * MAX_TOKENS_PER_STEP
        logger.debug(f"Preparing HybridCache for {model_name} with max_cache_len={max_cache_len}")
        return HybridCache(config=model.config,
                           max_batch_size=1,
                           max_cache_len=max_cache_len,
                           device=model.device,
                           dtype=model.dtype)
    else:
        return None

    

# logging.basicConfig(level=logging.WARNING)
# logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[RichHandler()]
)
logger = logging.getLogger("rich")

logger.setLevel(logging.DEBUG)
logger.debug("logger initialized!")

def debug_panel(title, text):
    panel = Panel(text, title=title)
    logger.debug(f"\n{textify(panel)}")


# Usage: python -m core.main
if __name__ == "__main__":
    fire.Fire(Experiment().eval)
