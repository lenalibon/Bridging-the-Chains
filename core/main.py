from functools import partial
from pathlib import Path

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
from core.constants import DEVICE, MAX_STEPS, MAX_TOKENS_PER_STEP, DataGetter
from core.method import EmbeddingMethodTest
from core.prompter import SimplePrompter
from core.utils import get_logger, get_timestamp, write_jsonl


import gc

from .utils import *
from .prompts import *

logger = get_logger()

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    if DEVICE == "cuda":
        torch.cuda.ipc_collect()


clear_cache()

set_seed(42)

class Experiment:
    def eval(self, **kwargs):
        logger.info("Running full evaluation...")
        # NOTE: A DataGetter is a function that returns a tuple with dataset and its label.
        # We might want to consider a wrapper class around datasets.Dataset instead.
        data_getters: list[DataGetter] = [partial(self.get_gsm8k, n=1)] # FIXME delete n=
        model, tokenizer = self.get_model_and_tokenizer()
        prompter = SimplePrompter() # FIXME: TODO use autocot
        methods = [
            EmbeddingMethodTest(model, tokenizer, prompter, n_init_chains=8, label="EmbeddingClustering-8")
            # BaselineGreedy(model, tokenizer, prompter, n_init_chains=8, label="BaselineGreedy-8"),
            # BaselineSimple(model, tokenizer, prompter, n_init_chains=4, label=f"BaselineSimple"),
        ]
        for get_data in data_getters:
            eval_data, dataset_label = get_data()
            for method in methods:
                eval_label = f"{method.label}___{dataset_label}"
                self.eval_method(method, eval_data, label=eval_label)

    def eval_method(self, method, eval_data, label="eval"):
        results = []
        for i, sample in enumerate(eval_data):
            question = sample['question']
            true_answer = sample['answer']
            pred_chains: Chains = method.generate_answer(question)
            # NOTE: roscoe's gsm8k.json is different
            # FIXME: only the first chain is written! I am not sure what we should do if there are more chains. -sb
            clean_text = pred_chains[0].get_clean_text()
            debug_panel(logger, "Predicted Answer", pred_chains[0].get_generated_text())
            debug_panel(logger, "True Answer", true_answer)
            results.append({
                "premise": question,
                "reasoning": clean_text,
                "true_answer": true_answer,
            })
        Path("results").mkdir(parents=True, exist_ok=True)
        ts = get_timestamp()
        write_jsonl(results, f"results/{label}___{ts}.jsonl")
        
    def get_gsm8k(self, n=None) -> tuple['Dataset', str]:
        # NOTE: using the train split for evaluation
        split = datasets.load_dataset("gsm8k", "main")["test"] # type: ignore
        label = "gsm8k"
        if n is not None:
            split = split.select(range(n)) # type: ignore
            label = f"gsm8k-first{n}"
        return (split, label) # type: ignore

    def get_model_and_tokenizer(self):
        # TODO adapt code for Gemma (fix caching crashes)
        model_name = "google/gemma-3-1b-it"
        # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        # logger.debug("padding_side=left")
        # TODO use_fast=True
        # model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        return model, tokenizer




# Usage: python -m core.main
if __name__ == "__main__":
    fire.Fire(Experiment().eval)
