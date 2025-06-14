import re


from typing import Optional

from transformers import AutoTokenizer, Cache # type: ignore


import torch.nn.functional as F

from core.constants import BatchedTokenIdsTensor, Scores

from .utils import *
from prompting.prompts import *

logger = get_logger()

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
        decoded_chain_end = self.tokenizer.decode(self.token_ids[0, -max_end_len:]) # type: ignore
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
        debug_panel(logger, "Generated lines", repr(self.get_generated_lines()))
        # assert lines[-1] == '' # TODO handle failure gracefully
        # logger.debug(f"{lines=}")
        index = -2 if lines[-1] == '' else -1
        last_line = lines[index]
        logger.debug(f"{self.short_repr} ends with line: {repr(last_line)}")
        pattern = r"^\s*\".*\",?\s*$" # Quoted line, with an optional comma at the end
        is_complete = re.fullmatch(pattern, last_line) is None
        logger.debug(f"{self.short_repr} complete? {is_complete}")
        return is_complete

    def get_full_text(self, skip_special_tokens=True) -> str:
        return self.tokenizer.decode(self.token_ids[0], # type: ignore
                                     skip_special_tokens=skip_special_tokens)

    def get_generated_text(self, skip_special_tokens=True) -> str:
        return self.tokenizer.decode(self.token_ids[0, self.prompt_offset:], # type: ignore
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
        vocab_size = len(self.tokenizer) # type: ignore

        if len(scores_tuple) != len(target_token_ids):
            raise ValueError("Length of scores_tuple must match length of target_token_ids.")

        # TODO! assert on sequence length. beware bos

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

    def release_memory(self):
        """
        Explicitly delete tensors to release memory.
        Call this after you're done using the Chain object.
        """
        try:
            del self.token_ids
        except AttributeError:
            pass
        try:
            del self.scores
        except AttributeError:
            pass
        try:
            del self.pkv
        except AttributeError:
            pass

        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        

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