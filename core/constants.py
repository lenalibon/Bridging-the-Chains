from typing import Callable

from datasets import Dataset

import torch

from .utils import *
from .prompts import *

# TODO: At some point it would be nicer to have this as a config like a yaml

TokenIdsTensor = torch.Tensor # size: (sequence_length,)
BatchedTokenIdsTensor = torch.Tensor # size: (batch_size, sequence_length)

# NOTE: for some reason torch stores scores as a tuple of tensors, not a single tensor
Scores = tuple[torch.Tensor] # size: (sequence_length, (batch_size, vocab_size))
IdCluster = list[int] # list of chain ids
DataGetter = Callable[[], tuple['Dataset', str]] # function that returns a dataset and its label


DEVICE = "cpu"
TEMPERATURE = 0.7

MAX_STEPS = 8
MAX_TOKENS_PER_STEP = 100

PREFER_JSONIC_DEBUG = False