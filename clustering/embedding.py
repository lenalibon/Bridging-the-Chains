import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer

from core.chain import ListChains
from core.clusterer import Clusterer
from core.constants import IdCluster


class EmbeddingCluster(Clusterer):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.model.eval()

    def compute_embedding(self, thought: str) -> torch.Tensor:
        """Takes a thought of a cot of a chain, and returns the embedding"""
        with torch.no_grad():
            inputs = self.tokenizer(thought, return_tensors="pt", truncation=True, max_length=128)
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding
    
    def kmeans(self, chain_embeddings: list[torch.Tensor], n_iter: int = 3) -> list[int]:
        """
        Simple k-means clustering for a list of 1D torch tensors.
        Returns a list of cluster assignments (same length as embeddings).
        """
        # 1. Get kmeans cluster per chain
        X = np.array(chain_embeddings) # need to convert to numpy for sklearn
        # NOTE: always halfs number of chains, can also specify in the constructor instead
        k = len(chain_embeddings) // 2 + 1 
        labels = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(X)

        # 2. Change to IdCluster format
        km_clusters: list[IdCluster] = [[] for _ in range(k)]
        for i, cluster in enumerate(labels):
            km_clusters[cluster].append(i)
        return km_clusters

    def __call__(self, chains: ListChains) -> list[IdCluster]:
        """Computes an embedding for each chain, and then clusters the chains"""
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

