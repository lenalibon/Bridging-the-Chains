# TODO Lena
# Given M chain of thoughts till step K (in list, not yet decoded):
# 1. Get the prompt to get intermediate answer and prompt the model for every chain
# 2. Cluster the intermediate answers using entailment by Kuhn et al.
# 3. Provide functions to do:
# 3.1. Return indices to chains corresponding to the clusters
# 3.2. Return one index per cluster with the highest score
from typing import List
from transformers import TextGenerationPipeline
from generation.postprocess_chain import prompt_intermediate_answer

import torch
import numpy as np


def get_intermediate_answers(model, tokenizer, chains_tokenized: List[torch.LongTensor], stop_string: str = '.",',
                             max_tokens=64):
    pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, return_full_text=False)
    prompts = [prompt_intermediate_answer(tokenizer, c, stop_string) for c in chains_tokenized]
    generations = pipe(prompts, max_new_tokens=max_tokens, do_sample=False)
    intermediate_answers = [g[0]["generated_text"].strip() for g in generations]
    return intermediate_answers
