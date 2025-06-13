

from typing import Optional

from transformers import GenerationConfig

from core.chain import Chain, ListChains
from core.constants import PREFER_JSONIC_DEBUG # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    HybridCache,
)

from core.experiment_config import ExperimentConfig
from .prompter import Prompter
from .utils import *

logger = get_logger()

def prepare_pkv(model, n_input_tokens, max_steps, max_tokens_per_step, device): # Add device parameter
    model_name = model.__class__.__name__
    if 'Gemma' in model_name:
        max_cache_len = n_input_tokens + max_steps * max_tokens_per_step 
        logger.debug(f"Preparing HybridCache for {model_name} with max_cache_len={max_cache_len} on device={device}")
        return HybridCache(config=model.config,
                        max_batch_size=1,
                        max_cache_len=max_cache_len,
                        device=device,  # <-- Use the parameter
                        dtype=model.dtype)
    else:
        return None

def move_hybrid_cache(pkv, model, device, n_input_tokens, max_steps, max_tokens_per_step):
    if isinstance(pkv, HybridCache):
        return prepare_pkv(model, n_input_tokens, max_steps, max_tokens_per_step, device=device)

    return pkv.to(device) if hasattr(pkv, 'to') else pkv


class Stepper:
    """
    Functionality for next step generation, in one class

    Used in `Method` and `SummarizingMergeFunction`
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: ExperimentConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        stop_string = "\n"
        self.gen_config = GenerationConfig(
            max_new_tokens=self.config.max_tokens_per_step,
            do_sample=True,
            temperature=self.config.temperature,
            cache_implementation=None,
            return_dict_in_generate=True,
            output_scores=True,
            # eos_token_id=self.stop_token_ids,
            stop_strings=stop_string,
            use_cache=self.config.use_cache,
        )
        # NOTE: On the difference between `scores` and `logits`,
        # see https://huggingface.co/docs/transformers/v4.51.3/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput

    def first_step_in_all(self, prompter: 'Prompter', n = 1, **kw) -> ListChains:
        return ListChains.from_list([self.first_step_in_one(prompter, index=index, **kw) for index in range(n)])

    def first_step_in_one(self, prompter: 'Prompter', index: Optional[int] = None, question: Optional[str] = None) -> Chain:
        if index is not None:
            prompt: str = prompter(question, index)
        else:
            prompt: str = prompter(question)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.config.device) # type: ignore
        prompt_offset = input_ids.shape[1]
        pkv = prepare_pkv(self.model, prompt_offset, self.config.max_steps, self.config.max_tokens_per_step, device='cpu') if self.gen_config.use_cache else None
        if pkv is not None:
            pkv = move_hybrid_cache(pkv, self.model, self.config.device, input_ids.shape[1],
                                    self.config.max_steps, self.config.max_tokens_per_step)
        with torch.inference_mode():
            out = self.model.generate(input_ids, # type: ignore
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
        self.debug_chain(chain, skip_offset=True)
        return chain

    def first_step_in_one_summarization(self, prompt: str, index: Optional[int] = None, question: Optional[str] = None) -> Chain:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.config.device) # type: ignore
        prompt_offset = input_ids.shape[1]
        pkv = prepare_pkv(self.model, prompt_offset, self.config.max_steps, self.config.max_tokens_per_step, device='cpu') if self.gen_config.use_cache else None
        if pkv is not None:
            pkv = move_hybrid_cache(pkv, self.model, self.config.device, input_ids.shape[1],
                                    self.config.max_steps, self.config.max_tokens_per_step)
        with torch.inference_mode():
            out = self.model.generate(input_ids, # type: ignore
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
        self.debug_chain(chain, skip_offset=True)
        return chain

    def first_step_in_one_summarization(self, prompt: str, index: Optional[int] = None, question: Optional[str] = None) -> Chain:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.config.device) # type: ignore
        prompt_offset = input_ids.shape[1]
        pkv = prepare_pkv(self.model, prompt_offset, self.config.max_steps, self.config.max_tokens_per_step, device='cpu') if self.gen_config.use_cache else None
        if pkv is not None:
            pkv = move_hybrid_cache(pkv, self.model, self.config.device, input_ids.shape[1],
                                    self.config.max_steps, self.config.max_tokens_per_step)
        with torch.inference_mode():
            out = self.model.generate(input_ids, # type: ignore
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
        self.debug_chain(chain, skip_offset=True)
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
        input_token_ids = chain.token_ids
        if self.gen_config.use_cache:
            # Get the pkv from the chain (which should be on CPU)
            pkv = chain.pkv or prepare_pkv(self.model, input_token_ids.shape[1], 
                                           max_steps=self.config.max_steps-chain.n_lines, 
                                           max_tokens_per_step=self.config.max_tokens_per_step,
                                           device='cpu') # <-- specify device
        else:
            pkv = None
        if pkv is not None:
            pkv = move_hybrid_cache(pkv, self.model, self.config.device, input_token_ids.shape[1],
                                    self.config.max_steps, self.config.max_tokens_per_step)
        with torch.inference_mode():
            out = self.model.generate(input_token_ids, # type: ignore
                                      past_key_values=pkv,
                                      generation_config=self.gen_config,
                                      tokenizer=self.tokenizer,
                                      cache_implementation=None,
                                      )
        if out.past_key_values is not None:
            chain.pkv = move_hybrid_cache(out.past_key_values, self.model, 'cpu', input_token_ids.shape[1],
                                          self.config.max_steps, self.config.max_tokens_per_step)
        chain.token_ids = out.sequences
        if out.scores is not None:
            cpu_scores = [s.to("cpu", non_blocking=True) for s in out.scores]
            chain.scores += tuple(cpu_scores)
        # chain.scores += out.scores
        assert chain.n_lines
        chain.n_lines += 1
        self.debug_chain(chain, skip_offset=True)
        return chain

    def debug_chain(self, chain: Chain, jsonic=PREFER_JSONIC_DEBUG, skip_offset=False) -> None:
        if jsonic:
            text = chain.jsonic_repr
        else:
            offset = chain.prompt_offset if skip_offset else 0
            to_decode = chain.token_ids[0, offset:]
            text = self.tokenizer.decode(to_decode) # type: ignore
        debug_panel(logger, f"{chain.short_repr}", text)


