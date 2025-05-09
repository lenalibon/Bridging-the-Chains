import os
import json
import re
from tqdm import tqdm
from collections import Counter

def normalize_text(text):
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

class F1ScoreEvaluator:
    def evaluate_answer(self, problem: str, answer: str, ground_truth: str):
        return compute_f1(answer, ground_truth)

    def evaluate(self, input_file: str):
        results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Evaluating F1", unit="example"):
            item = json.loads(line)
            problem = item['premise']
            answer = item['gpt-3']
            ground_truth = item['hypothesis'].split(
                "IGNORE THIS. Ground truth here for reference. "
            )[1]

            f1_score = self.evaluate_answer(problem, answer, ground_truth)

            results.append({
                "key": item.get("key", ""),
                "f1": f1_score
            })

        print(f"Evaluated {len(results)} examples.")
        return results
