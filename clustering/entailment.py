import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from core.chain import ListChains
from core.clusterer import Clusterer
from core.constants import IdCluster

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    def generate_intermediate_answer(self, cot_steps: list[str]) -> str:
        # TODO stop at "
        return "abc"

    def __call__(self, chains: ListChains) -> list[IdCluster]:
        """Generates the intermediate answer for each chain, and then clusters the chains based on entailment"""
        # 1. Generate the intermediate answer for each chain
        intermediate_answers = list()
        for chain in chains:
            cot_steps: list[str] = chain.get_generated_steps()
            intermediate_answer = self.generate_intermediate_answer(cot_steps)
            intermediate_answers.append(intermediate_answer)



        # 1. Compute the embeddings for each chain
        chain_embeddings = list()
        for chain in chains:
            cot_steps: list[str] = chain.get_generated_steps()
            thought_embeddings = list()
            for thought in cot_steps:
                thought_embedding = self.compute_embedding(thought)
                thought_embeddings.append(thought_embedding)
            # Aggregate embeddings in a chain by mean
            chain_embedding = torch.stack(thought_embeddings, dim=0).mean(dim=0)
            chain_embeddings.append(chain_embedding)

        # 2. Cluster the embeddings
        clusters = self.kmeans(chain_embeddings)
        return clusters

