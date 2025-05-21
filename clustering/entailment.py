from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, \
    StoppingCriteria, StoppingCriteriaList

from core.chain import ListChains
from core.clusterer import Clusterer
from core.constants import IdCluster
from prompting.prompts import INTERMEDIATE_ANSWER_PROMPT_TEMPLATE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StopAfterIntermediateAnswer(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings=("\n", "\n\n")):
        super().__init__()
        self.stop_strings = [tokenizer.encode(stop_string, add_special_tokens=False)[0] for stop_string in stop_strings]

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] in self.stop_strings


class EntailmentCluster(Clusterer):

    def __init__(self):
        # NLI model for entailment
        nli_model_name = "cross-encoder/nli-deberta-v3-large"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(DEVICE)
        self.nli_model.eval()

        # Autoregressive model to generate intermediate answers
        model_name = "google/gemma-3-1b-it"
        self.tokenizer_generate = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model_generate = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

    def generate_intermediate_answer(self, question: str, cot_steps: list[str]) -> str:
        filled_prompt = INTERMEDIATE_ANSWER_PROMPT_TEMPLATE.format(
            question=question,
            cot_steps='"' + '",\n\t\t"'.join(cot_steps) + '"'
        )

        stop_criteria = StoppingCriteriaList([StopAfterIntermediateAnswer(self.tokenizer_generate)])
        tokenized = self.tokenizer_generate(filled_prompt, return_tensors="pt").to(DEVICE)
        output_tokenized = self.model_generate.generate(**tokenized, max_new_tokens=200, do_sample=True,
                                                        stopping_criteria=stop_criteria)
        output = self.tokenizer_generate.decode(output_tokenized[0])

        # Extract final answer
        return output.split('"final_answer": ')[-1].replace('\n', '').replace('"', '')


    def __call__(self, chains: ListChains, question: Optional[str] = None) -> list[IdCluster]:
        """Generates the intermediate answer for each chain, and then clusters the chains based on entailment"""
        assert question is not None, "Question must be provided for entailment clustering"

        # 1. Generate the intermediate answer for each chain
        intermediate_answers = list()
        for chain in chains:
            cot_steps: list[str] = chain.get_generated_steps()
            intermediate_answer = self.generate_intermediate_answer(question, cot_steps)
            intermediate_answers.append(intermediate_answer)

        # 1. Compute the embeddings for each chain
        chain_embeddings = list()
        for chain in chains:
            cot_steps: list[str] = chain.get_generated_steps()
            thought_embeddings = list()
            for thought in cot_steps:
                thought_embedding = self.compute_embedding(question, thought)
                thought_embeddings.append(thought_embedding)

        clusters = None
        return clusters
