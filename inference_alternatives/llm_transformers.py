# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import time
from prompting.prompts import GEMMA_SIMPLE_PROMPT_TEMPLATE

start_time = time.time()
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", cache_dir="/work/scratch/imcauliffe")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b", 
    cache_dir="/work/scratch/imcauliffe", 
    quantization_config=quantization_config,
    device_map="auto", 
    torch_dtype=torch.float16)

load_time = time.time()
print(f"Time to load: {load_time - start_time}")
input_text = GEMMA_SIMPLE_PROMPT_TEMPLATE.substitute(
    question="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, return_dict_in_generate=True, output_scores=True)
print("All outputs")
print(tokenizer.decode(outputs['sequences'][0]))
print(f"Inference time: {time.time() - load_time}")

# TODO: check if the new prompt format helps https://ai.google.dev/gemma/docs/core/prompt-structure
# TODO: check if i can use some KV cache
# (maybe) TODO: switch to vLLM