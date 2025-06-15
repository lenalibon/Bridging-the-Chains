#!/usr/bin/env python3

import argparse
import os
import json
from dotenv import load_dotenv

from llm_comparison.llm_comparison_evaluator import LLMComparisonEvaluator 
from f1_score.f1_score_evaluator import F1ScoreEvaluator
from bert.bert_score_evaluator import BertScoreEvaluator
from roscoe.roscoe import ReasoningEvaluator
from roscoe.score import SENT_TRANS
from roscoe.utils import save_scores, print_and_reset_max_gpu_memory

def run_roscoe(input_path, output_path, model_name, model_type, datasets, suffix, scores, discourse_batch, coherence_batch):
    evaluator = ReasoningEvaluator(
        score_types=scores,
        model_type=model_type,
        transformer_model=model_name,
        ppl_model="gpt2-large",
        discourse_batch=discourse_batch,
        coherence_batch=coherence_batch,
    )

    for root, _, files in os.walk(input_path):
        for fname in files:
            if not fname.endswith(suffix) or not any(fname.startswith(d) for d in datasets):
                continue

            in_file = os.path.join(root, fname)
            out_file = os.path.join(output_path, model_name, f"scores_{fname.split('.')[0]}.tsv")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

            print(f"Running Roscoe on {fname}")
            evaluator.update_evaluator(in_file)
            file_scores = evaluator.evaluate(score_types=scores)
            save_scores(file_scores, out_file)
            print_and_reset_max_gpu_memory()

def run_bert_score(input_path):
    print(f"Running BERTScore on {input_path}")
    
    evaluator = BertScoreEvaluator()
    results = evaluator.evaluate(input_path)
    output_file = input_path.replace(".json", "_bert_score_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"BertScore results written to {output_file}")

def run_f1_score(input_path):
    print(f"Running F1 score on {input_path}")

    evaluator = F1ScoreEvaluator()
    results = evaluator.evaluate(input_path)
    output_file = input_path.replace(".json", "_f1_results.json")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    print(f"F1 score results written to {output_file}")

def run_llm_comparison(input_path):
    print(f"Running LLM comparison evaluation on {input_path}")
    evaluator = LLMComparisonEvaluator(model="gemma-3-27b-it")
    results = evaluator.evaluate(input_path)

    output_file = input_path.replace(".json", "_llm_comparison_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Exact Match results written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to a single JSON file")
    parser.add_argument("--output-path", type=str, default="./roscoe/")
    parser.add_argument("--suffix", type=str, default="json")
    parser.add_argument("--metrics", nargs="+", choices=["bert", "f1", "llm", "roscoe"], help="List of metrics to run")
    parser.add_argument("--model-type", type=str, default=SENT_TRANS)
    parser.add_argument("--model-name", type=str, default="facebook/roscoe-512-roberta-base")
    parser.add_argument("--discourse-batch", type=int, default=64)
    parser.add_argument("--coherence-batch", type=int, default=16)
    parser.add_argument("--roscoe-scores", nargs="*", default=["fluency", "coherence", "discourse", "relevance"])

    args = parser.parse_args()
    load_dotenv()

    fpath = args.input_path

    if not fpath.endswith(args.suffix):
        raise ValueError(f"Input file must end with .{args.suffix}")

    if not args.metrics:
        raise ValueError("Please provide at least one metric using --metrics (e.g., --metrics bert f1 em)")

    if "bert" in args.metrics:
        run_bert_score(fpath)

    if "f1" in args.metrics:
        run_f1_score(fpath)

    if "llm" in args.metrics:
        run_llm_comparison(fpath)

    if "roscoe" in args.metrics:
        run_roscoe(
            input_path=fpath,
            output_path=args.output_path,
            model_name=args.model_name,
            model_type=args.model_type,
            datasets=[],
            suffix=args.suffix,
            scores=args.roscoe_scores,
            discourse_batch=args.discourse_batch,
            coherence_batch=args.coherence_batch,
        )

