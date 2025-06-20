# Bridging the Chains: Clustering and Chain Fusion for Enhanced Chain-of-Thought Reasoning
## Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/lenalibon/Bridging-the-Chains.git
cd Bridging-the-Chains
```

### 2. Set Up API Tokens
- Create a file named `hf_token.txt` in the root directory and paste your Hugging Face token into it.
- If you are running experiments that require external API calls (N1, N2, N3), add your Gemma API key in `core/experiment_config.py`

### 3. Run an Experiment
**Local Execution:** Run the main script using the default configuration from `core/experiment_config.py`:
```bash
python -m core.main
```
To customize the experiment, edit the configuration file `core/experiment_config.py`.

**Running with SLURM:** Submit the experiment to a SLURM job scheduler: 
```bash
python submit_experiment.py
```

## Output Format 
Results will be saved in the `results/` directory. The filenames follow this format: `results/<EXPERIMENT_ID>___<DATASET>-first<NUMBER-OF-SAMPLES>___<TIMESTAMP>.jsonl`.

Each line in the `.jsonl` file contains:
```javascript
{
  "premise": "Question",
  "reasoning": "Generated final chain",
  "true_answer": "Ground truth reasoning"
}
```
- Successful sample IDs are stored in `results/success_ids.txt` and will be skipped in subsequent runs.
- Failed sample IDs are stored in `results/error_ids.txt`.

## Evaluation
### Optional: Merge Multiple Result Files
If your run was interrupted and you have several `.jsonl` result files, merge them with:
1. Set the `input_folder` and `output_path` in `merge.py`
2. Run:
```bash
python merge.py
```

### Convert to ROSCOE-Compatible Format
To prepare output for ROSCOE evaluation:
```bash
python metrics/change_for_roscoe.py ./experiments_results/b1_2/b1_2.jsonl ./experiments_results/b1_2/b1_2_roscoe.jsonl
```

### Accuracy & Reasoning Metrics
To compute exact match (EM), F1, BERTScore, and LLM-based correctness:
```bash
python metrics/compute_metrics.py --input-path ./experiments_results/b1_2/b1_2.jsonl --suffix jsonl --metrics llm f1 em bert
```

### Chain-Level Consistency and Faithfulness (ROSCOE)
```bash
python metrics/compute_metrics.py --input-path ./experiments_results/b1_2/b1_2_roscoe.jsonl --suffix jsonl --metrics roscoe
```
