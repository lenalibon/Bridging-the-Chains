import math
import json

from datasets import load_dataset, Dataset
import pandas as pd
from generate_cot_steps import generate_cot_steps

dataset = load_dataset("gsm8k", "main") 

generation_data = pd.DataFrame(dataset['test'])

batch_size = 8
num_batches = math.ceil(len(generation_data) / batch_size)

for i in range(num_batches):
    batch_start = i * batch_size
    batch_end = min((i + 1) * batch_size, len(generation_data))

    batch_df = generation_data.iloc[batch_start:batch_end].copy()

    batch_samples = []
    for question in batch_df["question"]:
        cot_steps = generate_cot_steps(problem=question)
        batch_samples.append({
            "question": question,
            "cot_steps": cot_steps
        })

    with open(f"./gsm8k_few_shot/gsm8k_few_shot_{i}.json", "w") as f:
        json.dump(batch_samples, f, indent=2)