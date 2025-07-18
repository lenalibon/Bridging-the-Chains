from huggingface_hub import login
import argparse
from functools import partial
from pathlib import Path

from core.experiments import ExperimentB1, ExperimentN1, ExperimentN2, ExperimentN3, ExperimentM1, ExperimentM2, ExperimentM3
from core.prompter import DiversifiedCoTPrompter, SimplePrompter
import datasets
import fire
from datasets import Dataset
import torch
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    HybridCache,
)

from core.chain import Chains
from core.constants import DataGetter
from core.utils import get_logger, get_logger_slurm, get_timestamp, write_jsonl
from core.experiment_config import experiment_config, ExperimentConfig

import gc

from .utils import *

#logger = get_logger()  # Use this for local runs
logger = get_logger_slurm()

def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


clear_cache()
set_seed(42)

experiment_mappings = {
    'B1': ExperimentB1,
    'N1': ExperimentN1,
    'N2': ExperimentN2,
    'N3': ExperimentN3,
    'M1': ExperimentM1,
    'M2': ExperimentM2,
    'M3': ExperimentM3
}

prompter_mappings = {
    'simple': SimplePrompter,
    'diversified_cot': DiversifiedCoTPrompter
}

class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def eval(self, **kwargs):
        login(token=self.config.hf_token)

        logger.info("Running full evaluation...")
        # NOTE: A DataGetter is a function that returns a tuple with dataset and its label.
        # We might want to consider a wrapper class around datasets.Dataset instead.
        data_getters: list[DataGetter] = [partial(self.get_gsm8k, n=self.config.num_samples_eval)] # FIXME delete n=
        model, tokenizer = self.get_model_and_tokenizer()
        prompter = prompter_mappings[self.config.prompter](folder_path = self.config.few_shots_folder_path, n_shots = self.config.num_few_shots) # FIXME: TODO use autocot
        
        experiment = experiment_mappings[self.config.experiment_id](model, tokenizer, prompter, self.config,
                   label=experiment_mappings[self.config.experiment_id].__name__, n_init_chains = self.config.n_init_chains)
        
        for get_data in data_getters:
            eval_data, dataset_label = get_data()
            eval_label = f"{experiment.label}___{dataset_label}"
            self.eval_experiment(experiment, eval_data, label=eval_label)

    def eval_experiment(self, experiment, eval_data, label="eval"):
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        success_file = results_dir / "success_ids.txt"
        error_file = results_dir / "error_ids.txt"
        
        success_ids = set()
        if success_file.exists():
            success_ids = set(line.strip() for line in success_file.open())
        error_ids = set()
        if error_file.exists():
            error_ids = set(line.strip() for line in error_file.open())
        
        ts = get_timestamp()
        result_file = results_dir / f"{label}___{ts}.jsonl"
        
        for i, sample in enumerate(eval_data): 
            qid = str(sample.get("id", i))
            if qid in success_ids:
                # print(f"Skipping already processed sample {qid}")
                continue
            question = sample['question']
            true_answer = sample['answer']
            try:
                pred_chains: Chains = experiment.generate_answer(question)
                # NOTE: roscoe's gsm8k.json is different
                # FIXME: only the first chain is written! I am not sure what we should do if there are more chains. -sb
                clean_text = pred_chains[0].get_clean_text()
                debug_panel(logger, "Predicted Answer", pred_chains[0].get_generated_text())
                debug_panel(logger, "True Answer", true_answer)
                result = {
                    "premise": question,
                    "reasoning": clean_text,
                    "true_answer": true_answer,
                }
                #print(clean_text)
                with result_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                
                with success_file.open("a", encoding="utf-8") as sf:
                    sf.write(qid + "\n")
                    sf.flush()
                success_ids.add(qid)

                for chain in pred_chains:
                    chain.release_memory()
                del pred_chains, clean_text, question, true_answer
                clear_cache()
            except Exception as e:
                print(f"Error processing sample {qid}: {e}")
                with error_file.open("a", encoding="utf-8") as ef:
                    ef.write(qid + "\n")
                    ef.flush()
                error_ids.add(qid)
            finally:
                clear_cache()
        
    def get_gsm8k(self, n=None) -> tuple['Dataset', str]:
        # NOTE: using the train split for evaluation
        split = datasets.load_dataset(self.config.dataset, "main")["train"] # type: ignore
        label = self.config.dataset 
        if n is not None:
            split = split.select(range(n)) # type: ignore
            label = f"{self.config.dataset}-first{n}"
        return (split, label) # type: ignore

    def get_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, padding_side='left')
        # model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=self.config.hf_token
        ).to(self.config.device)
        return model, tokenizer


# Usage: python -m core.main
if __name__ == "__main__":
    with open("hf_token.txt", "r") as f:
        token = f.read().strip()
    login(token=token)
    exp = Experiment(experiment_config)
    fire.Fire(exp.eval)
