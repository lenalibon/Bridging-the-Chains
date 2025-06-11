from munch import Munch
from typing import List, Type, TypedDict

class ExperimentConfig(TypedDict):
    methods: List[str]
    prompter: str 
    n_init_chains: List[int]
    num_samples_eval: int
    max_steps: int
    max_tokens_per_step: int
    num_few_shots: int
    temperature: float

    merge_after: bool
    merge_every: int | bool

    dataset: str
    model_name: str
    few_shots_folder_path: str
    use_cache: bool
    device: str

experiment_config_data = {
    "methods": ['greedy'], #['greedy', 'aggregation', 'simple']
    "prompter": 'diversified_cot', #['simple', 'diversified_cot']
    "n_init_chains": [4, 8],
    "num_samples_eval": 1,
    "max_steps": 8,
    "max_tokens_per_step": 64,
    "num_few_shots": 4,
    "temperature": 0.7,

    "merge_every": 2,
    "merge_after": True,

    "dataset": "gsm8k",
    "model_name": "google/gemma-7b",
    "few_shots_folder_path": "few-shot/gsm8k_few_shot/",
    "use_cache": True,
    "device": "cuda"
}

experiment_config: ExperimentConfig = Munch.fromDict(experiment_config_data)