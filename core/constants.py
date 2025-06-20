from typing import Callable

import torch


TokenIdsTensor = torch.Tensor # size: (sequence_length,)
BatchedTokenIdsTensor = torch.Tensor # size: (batch_size, sequence_length)

# NOTE: for some reason torch stores scores as a tuple of tensors, not a single tensor
Scores = tuple[torch.Tensor] # size: (sequence_length, (batch_size, vocab_size))
IdCluster = list[int] # list of chain ids
DataGetter = Callable[[], tuple['Dataset', str]] # function that returns a dataset and its label


PREFER_JSONIC_DEBUG = False