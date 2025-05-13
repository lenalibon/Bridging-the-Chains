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


def escape_braces(template):
    '''Templates with braces must be escaped for formatting to work: } becomes }} and { becomes {{, except for the placeholder {question} which must be kept as is.'''
    tmp_str = 'PLACEHOLDER_ASDF'
    return template.replace('{question}', tmp_str).replace('{', '{{').replace('}', '}}').replace(tmp_str, '{question}')


_SIMPLE_PROMPT_TEMPLATE = '''You are a math tutor. You will be given a math question and you need to answer it step by step, in JSON format. Let's think step by step.

{
  "question": "Billy sells DVDs. He has 8 customers on Tuesday. His first 3 customers buy one DVD each.  His next 2 customers buy 2 DVDs each.  His last 3 customers don't buy any DVDs. How many DVDs did Billy sell on Tuesday?",
  "cot_steps": [
    "The first 3 customers buy 1 DVD each, so they buy 3 * 1 = 3 DVDs.",
    "The next 2 customers buy 2 DVDs each, so they buy 2 * 2 = 4 DVDs.",
    "The last 3 customers buy 0 DVDs each, so they buy 3 * 0 = 0 DVDs.",
    "In total, Billy sells 3 + 4 + 0 = 7 DVDs.",
    "So, the answer is 7."
  ]
},
{
  "question": "{question}",
  "cot_steps": [
'''

SIMPLE_PROMPT_TEMPLATE = escape_braces(_SIMPLE_PROMPT_TEMPLATE)

DEBUG_PROMPT = '''The Countess of Sakharovka.
By F. M. Dostoevsky
1868

The cherry garden was cloaked in a silence so profound that even the wind seemed ashamed to stir the blossoming branches. Pale petals fell one by one, like the slow unraveling of memory.

At the edge of the grove, near the marble fountain where no water had flowed since the old Count’s death, the Countess stood motionless, her veil trailing behind her in the grass.'''
    

@cache
def get_newline_token_id(tokenizer):
    """Get the token id for the newline character"""
    newline_token = tokenizer("\n")["input_ids"][0]
    if isinstance(newline_token, list):
        newline_token = newline_token[0]
    return newline_token


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


def print_tokens_batch(tokenizer, batch_output_ids):
    """Print a per-token decoding"""
    for i, token_id in enumerate(batch_output_ids):
        token_str = tokenizer.decode(token_id, skip_special_tokens=False)
        print(f'{i:03d}: {repr(token_str)}')


def print_decode_batch(tokenizer, batch_output_ids):
    print(tokenizer.decode(batch_output_ids))
 

def tensor_split(x: Tensor, delimiter: int) -> Iterable[Tensor]:
    delimiter_indices = torch.where(x == delimiter)[0]
    starts = torch.cat([torch.tensor([-1]), delimiter_indices])
    ends = torch.cat([delimiter_indices, torch.tensor([len(x)])])
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
