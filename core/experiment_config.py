from munch import Munch
from typing import List, Type, TypedDict

class ExperimentConfig(TypedDict):
    experiment_id:str # as defined in Table 1 of the paper
    prompter: str # which prompter to use
    n_init_chains: int # number of initial chains
    num_samples_eval: int # how many samples to evaluate on, None means all samples in the dataset
    max_steps: int
    max_tokens_per_step: int
    num_few_shots: int # number of few-shot examples to use in the prompter
    temperature: float # temperature for the model generation, 0.0 means greedy decoding, 1.0 means uniform distribution over tokens, etc.

    merge_every: int # number of steps after which we merge chains if we use during merge timer


    dataset: str # name of the dataset to use, so far only "gsm8k" works
    model_name: str # name of the model to use, e.g. "google/gemma-3-1b-it"
    few_shots_folder_path: str # path to the folder with few-shot examples that were generated before the execution
    use_cache: bool
    device: str
    
    nli_model_name: str # name of the NLI model to use for entailment, e.g. "cross-encoder/nli-deberta-v3-large"
    embedding_model_name: str # name of the embedding model to use for clustering, e.g. "bert-base-uncased"

experiment_config_data = {
    "experiment_id": 'B1', #['B1', 'N1', 'N2', 'N3', 'M1', 'M2', 'M3'],
    "prompter": 'diversified_cot', #['simple', 'diversified_cot']
    "n_init_chains": 7,
    "num_samples_eval": 100,
    "max_steps": 7,
    "max_tokens_per_step": 2000,
    "num_few_shots": 1,
    "temperature": 0.7,

    "merge_every": 2,

    "dataset": "gsm8k",
    "model_name": "google/gemma-3-1b-it",
    "few_shots_folder_path": "few-shot/gsm8k_few_shot/",
    "use_cache": True,
    "device": "cuda",
    
    "nli_model_name": "cross-encoder/nli-deberta-v3-large",  
    "embedding_model_name": "bert-base-uncased"
}

experiment_config: ExperimentConfig = Munch.fromDict(experiment_config_data)