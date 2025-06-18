import os
import json
import pandas as pd

path_to_experiments = "../experiments_results"
required_files = [
    "{name}.jsonl",
    "{name}_em_results.jsonl",
    "{name}_llm_comparison_results.jsonl",
    "{name}_roscoe.jsonl",
    "{name}_roscoe_results.tsv",
    "{name}_bert_score_results.jsonl",
    "{name}_f1_results.jsonl",
]

def check_existing_folders(experiments):
    # Check which experiment folders exist
    existing_folders = []

    for e in experiments:
        folder_path = os.path.join(path_to_experiments, e)
        if os.path.isdir(folder_path):
            existing_folders.append(e)
        else:
            print(f"ERROR {e} folder does not exist")

    print()
    # Check required files in existing folders
    complete_folders = []
    for e in existing_folders:
        folder_path = os.path.join(path_to_experiments, e)
        missing_files = []
        for file_template in required_files:
            file_name = file_template.format(name=e)
            file_path = os.path.join(folder_path, file_name)
            if not os.path.isfile(file_path):
                missing_files.append(file_name)
        if not missing_files:
            complete_folders.append(e)
        else:
            print(f"ERROR {e} has missing files")
            print(missing_files)
            print()
    print("Folders with all required files:")
    print(complete_folders)
    return complete_folders


def compute_accuracy(exp, metric_file_name="_em_results.jsonl", metric_name="predicted"):
    file_path = os.path.join(path_to_experiments, exp, f"{exp}{metric_file_name}")
    if not os.path.isfile(file_path):
        print(f"ERROR: {file_path} not found. Defaulting to 100% accuracy.")
        return 1.0
    total = 0
    sum = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total += 1
            sum += data.get(metric_name, False)
    if total == 0:
        print(f"ERROR: {file_path} has no data. Defaulting to 100% accuracy.")
        return 1.0
    return sum / total

def compute_roscoe_metric(exp, metric_file_name="_roscoe_results.tsv", metric_column="faithfulness"):
    file_path = os.path.join(path_to_experiments, exp, f"{exp}{metric_file_name}")
    if not os.path.isfile(file_path):
        print(f"ERROR: {file_path} not found. Defaulting to 100% value.")
        return 1.0
    df = pd.read_csv(file_path, sep=r'\s+', engine='python')
    if metric_column not in df.columns:
        print(f"ERROR: Column {metric_column} not found in {file_path}. Defaulting to 100% value.")
        return 1.0
    if len(df) == 0:
        print(f"ERROR: {file_path} is empty. Defaulting to 100% value.")
        return 1.0
    return df[metric_column].mean()