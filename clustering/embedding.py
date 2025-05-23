from typing import Optional
import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer

from core.chain import ListChains
from core.clusterer import Clusterer
from core.constants import IdCluster
from core.experiment_config import ExperimentConfig
from core.utils import get_logger

logger = get_logger()

class EmbeddingCluster(Clusterer):

    def __init__(self, config: ExperimentConfig):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.model.eval()
        self.embeddings = None
        self.centroids = None

    def _compute_embedding(self, thought: str) -> torch.Tensor:
        """Takes a thought of a cot of a chain, and returns the embedding"""
        with torch.no_grad():
            inputs = self.tokenizer(thought, return_tensors="pt", truncation=True, max_length=128)
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding
    
    def _kmeans(self, chain_embeddings: list[torch.Tensor], n_iter: int = 3) -> list[int]:
        """
        Simple k-means clustering for a list of 1D torch tensors.
        Returns a list of cluster assignments (same length as embeddings).
        """
        # 1. Get kmeans cluster per chain
        X = np.array(chain_embeddings) # need to convert to numpy for sklearn
        # NOTE: always halfs number of chains, can also specify in the constructor instead
        k = len(chain_embeddings) // 2 + 1 
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        kmeans.fit(X)
        labels = kmeans.predict(X)

        # 2. Change to IdCluster format and store centroids
        km_clusters: list[IdCluster] = [[] for _ in range(k)]
        self.centroids = [None for _ in range(k)]
        for i, cluster in enumerate(labels):
            km_clusters[cluster].append(i)
        self.centroids = [torch.from_numpy(center) for center in kmeans.cluster_centers_]
        # logger.info(f"Centroids have shape {self.centroids.shape}")
        logger.info(self.centroids)
        return km_clusters
    
    def get_centroids(self):
        return self.centroids
    
    def get_embeddings(self):
        return self.embeddings

    def __call__(self, chains: ListChains, question: Optional[str] = None) -> list[IdCluster]:
        """Computes an embedding for each chain, and then clusters the chains"""
        # 1. Compute the embeddings for each chain
        chain_embeddings = list()
        for chain in chains:
            cot_steps: list[str] = chain.get_generated_steps()
            thought_embeddings = list()
            for thought in cot_steps:
                thought_embedding = self._compute_embedding(thought)
                thought_embeddings.append(thought_embedding)
            # Aggregate embeddings in a chain by mean
            chain_embedding = torch.stack(thought_embeddings, dim=0).mean(dim=0)
            chain_embeddings.append(chain_embedding)

        self.embeddings = chain_embeddings

        # 2. Cluster the embeddings
        clusters = self._kmeans(chain_embeddings)
        logger.info(f"Type of centroids is {self.centroids[0].shape}")
        logger.info(f"Type of embeddings is {self.embeddings[0].shape}")
        return clusters

