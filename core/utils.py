from datetime import datetime
from functools import cache
import itertools
import json
from typing import Iterable

from torch import Tensor
import torch

from rich.panel import Panel
from rich.text import Text
from rich.console import Console


@cache
def get_newline_token_id(tokenizer):
    """Get the token id for the newline character"""
    newline_token = tokenizer("\n")["input_ids"][0]
    if isinstance(newline_token, list):
        newline_token = newline_token[0]
    return newline_token


def token_len(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt)["input_ids"])


def write_jsonl(data, path, indent=None):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, indent=None) + "\n")


def get_timestamp():
    return datetime.now().strftime('%Y%m%d%H%M')


def flat(nested):
    return itertools.chain.from_iterable(nested)


def print_tokens_batch(tokenizer, batch_output_ids):
    """Print a per-token decoding"""
    for i, token_id in enumerate(batch_output_ids):
        token_str = tokenizer.decode(token_id, skip_special_tokens=False)
        print(f'{i:03d}: {repr(token_str)}')


def print_decode_batch(tokenizer, batch_output_ids):
    print(tokenizer.decode(batch_output_ids))
 

def tensor_split(x: Tensor, delimiter: int) -> Iterable[Tensor]:
    # NOT TESTED
    delimiter_indices = torch.where(x == delimiter)[0]
    # TODO optimize with itertools.chain.from_iterable
    # starts = torch.cat([torch.tensor([-1]), delimiter_indices])
    starts = itertools.chain.from_iterable(([-1], delimiter_indices))
    # ends = torch.cat([delimiter_indices, torch.tensor([len(x)])])
    ends = itertools.chain.from_iterable((delimiter_indices, [len(x)]))
    sub_tensors = [x[s+1:e] for s, e in zip(starts, ends) if e - s > 1]
    return sub_tensors


def shift_padding_left(x, pad_token=0):
    # x: (batch_size, seq_length)
    mask = x != pad_token
    lengths = mask.sum(dim=1)  # (batch_size,)
    
    # Create left-padded tensor
    batch_size, seq_length = x.shape
    shifted = torch.full_like(x, pad_token)

    for i in range(batch_size):
        l = lengths[i]
        shifted[i, -l:] = x[i, mask[i]]
    
    return shifted


def textify(panel: Panel) -> Text:
  console = Console(width=120)
  with console.capture() as capture:
      console.print(panel)
  return Text.from_ansi(capture.get())


def hash_tensor(tensor: torch.Tensor) -> int:
    tensor = tensor.detach().cpu().contiguous()
    return hash(tensor.numpy().tobytes())
