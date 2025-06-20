import os
import sys
from core.main import Experiment
from core.experiment_config import experiment_config
import submitit
import fire
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    HybridCache,
    Gemma3ForConditionalGeneration
)
import torch
import time
import subprocess

# Code to submit job in SLURM or run locally
def get_model_and_tokenizer(config):
    model_name = config.model_name 
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.hf_token, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            token=config.hf_token
    ).to(config.device)
    return model, tokenizer



def run_wrapper(config):
    assert config.hf_token != "TODO: fill in", "Please set the Hugging Face token."
    config.model, config.tokenizer = get_model_and_tokenizer(config)

    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    exp = Experiment(experiment_config)
    fire.Fire(exp.eval)


def run(local = True):

    for _ in range(200):
        # Run `squeue -u username` to check for active jobs
        result = subprocess.run(['squeue', '-u', 'TODO: fill in with your username'], capture_output=True, text=True)

        # Check if only the header line is returned (no jobs)
        job_lines = result.stdout.strip().split('\n')
        if len(job_lines) <= 1:
            print("No jobs found. Starting a new experiment run.")
            if local:
                run_wrapper(experiment_config)
                return
        
            log_folder = "log/csnlp"
            os.makedirs(log_folder, exist_ok=True)
            executor = submitit.AutoExecutor(folder="logs/slurm")
            executor.update_parameters(
                    timeout_min=60 * 24,
                    tasks_per_node=1,
                    #cpus_per_task=4,
                    account="csnlp_jobs",
                    name="csnlp_training"
            )
            
            job = executor.submit(run_wrapper, experiment_config)
            print("Submitted job: ", job)
        else:
            print("Jobs are still running. Skipping this iteration.")

        time.sleep(5 * 60)  # Wait 5 minutes before checking again


if __name__ == "__main__":
    # Set `local=False` to submit to SLURM
    run(local=False)
