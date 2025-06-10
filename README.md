# Bridging the Chains
## Project Structure
TODO

## Getting Started
### 1. Clone the repository
```bash
git clone https://github.com/lenalibon/Bridging-the-Chains.git
cd Bridging-the-Chains
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup HF Token
Create a file called `hf_token.txt` and insert your token.

### 4. Running an Experiment
Run the main script using the default configuration in `core/experiment_config.py`:
```bash
python -m core.main
```

Edit `core/experiment_config.py`to change the experimental setup.

### 4. Output Format 
Results will be saved in the `results/` folder with filenames like: `results/greedy___gsm8k-first1___<timestamp>.jsonl`.

Each line in the `.jsonl` file contains:
```javascript
{
  "premise": "Original question",
  "reasoning": "Model's chain-of-thought",
  "true_answer": "Ground truth"
}
``` 