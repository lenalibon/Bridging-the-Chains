#!/usr/bin/env python3

import argparse
import os
import json

from exact_match.exact_match import ExactMatchEvaluator
from roscoe.roscoe import ReasoningEvaluator
from roscoe.score import SENT_TRANS
from roscoe.utils import save_scores, print_and_reset_max_gpu_memory

# Placeholder imports (you must implement or replace these)
#from metrics.bert import compute_bert_score  # Assume function exists
#from metrics.f1 import compute_f1_score       # Assume function exists
#from metrics.exact_match import compute_em    # Assume function exists

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

def run_bert_score(in_file):
    print(f"Running BERTScore on {in_file}")
    #return compute_bert_score(in_file)

def run_f1_score(in_file):
    print(f"Running F1 score on {in_file}")
    #return compute_f1_score(in_file)

def run_exact_match(input_path):
    print(f"Running Exact Match evaluation on {input_path}")
    evaluator = ExactMatchEvaluator(model="gemma-3-27b-it")
    results = evaluator.evaluate(input_path)

    # Save results
    output_file = input_path.replace(".json", "_em_results.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Exact Match results written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to a single JSON file")
    parser.add_argument("--output-path", type=str, default="./roscoe/")
    parser.add_argument("--suffix", type=str, default="json")
    parser.add_argument("--run-roscoe", action="store_true")
    parser.add_argument("--run-bert", action="store_true")
    parser.add_argument("--run-f1", action="store_true")
    parser.add_argument("--run-em", action="store_true")
    parser.add_argument("--model-type", type=str, default=SENT_TRANS)
    parser.add_argument("--model-name", type=str, default="facebook/roscoe-512-roberta-base")
    parser.add_argument("--discourse-batch", type=int, default=64)
    parser.add_argument("--coherence-batch", type=int, default=16)
    parser.add_argument("--roscoe-scores", nargs="*", default=["fluency", "coherence", "discourse", "relevance"])

    args = parser.parse_args()

    fpath = args.input_path

    if not fpath.endswith(args.suffix):
        raise ValueError(f"Input file must end with .{args.suffix}")

    if args.run_roscoe:
        run_roscoe(
            input_path=fpath,
            output_path=args.output_path,
            model_name=args.model_name,
            model_type=args.model_type,
            datasets=[],  # no longer used
            suffix=args.suffix,
            scores=args.roscoe_scores,
            discourse_batch=args.discourse_batch,
            coherence_batch=args.coherence_batch,
        )

    if args.run_bert:
        run_bert_score(fpath)

    if args.run_f1:
        run_f1_score(fpath)

    if args.run_em:
        run_exact_match(fpath)
