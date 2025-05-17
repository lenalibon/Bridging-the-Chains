# NOTE: whole file is a WIP for this commit
# 1. Use BERT to generate an embedding for each thought in the chain

# 2. Average embedding of each thought in the chain

# 3. k-means clustering on the aggregated embeddings for each chain to cluster them

from core.main import Chain, Clusterer, IdCluster, ListChains
from transformers import AutoTokenizer, AutoModel
import torch

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
            # outputs.last_hidden_state shape: (1, seq_len, hidden_size)
            # Average pooling over the sequence dimension
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # shape: (hidden_size,)
        return embedding

    def __call__(self, chains: ListChains) -> list[IdCluster]:
        """Computes an embedding for each chain, and then clusters the chains"""
        chain: Chain
        s = list()
        for chain in chains:
            print(chain.get_full_text())
            cot_steps: list[str] = chain.get_generated_steps()
            s.append(cot_steps)
        print(s)
        return [list(range(len(chains)))]
    

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# fake_token_ids_1 = torch.randint(0, tokenizer.vocab_size, (1, 5))
# fake_token_ids_2 = torch.randint(0, tokenizer.vocab_size, (1, 7))

print(tokenizer.vocab_size)

text_1 = tokenizer("""[
    "Let 'g' be the number of gold coins and 's' be the number of silver coins.",
    "We know that g + s = 110.",
    "We also know that g = s + 30.",
    "Substitute the second equation into the first equation: (s + 30) + s = 110.",
    "Combine like terms: 2s + 30 = 110.",
    "Subtract 30 from both sides: 2s = 80.",
    "Divide both sides by 2: s = 40.",
    "Now that we know s = 40, we can find g: g = s + 30 = 40 + 30 = 70.",
    "Therefore, Gretchen has 70 gold coins."
  ]""", return_tensors="pt").input_ids
text_2 = tokenizer("[What is the capital of Luxembourg?]", return_tensors="pt").input_ids

chain1 = Chain(tokenizer=tokenizer, token_ids=text_1, prompt_offset=0, index=0, n_lines=1)
chain2 = Chain(tokenizer=tokenizer, token_ids=text_2, prompt_offset=0, index=1, n_lines=1)

# Put them in a ListChains object
toy_chains = ListChains([chain1, chain2])

# Now you can test your EmbeddingCluster
clusterer = EmbeddingCluster()
clusters = clusterer(toy_chains)
print(clusters)