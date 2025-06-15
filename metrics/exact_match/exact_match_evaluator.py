import re
import json
from tqdm import tqdm

NUMBER_REGEX = r'[-+]?\d[\d,]*\.?\d*'

class ExactMatchEvaluator:
    def __init__(self):
        pass

    def extract_last_number(self, text: str):
        matches = re.findall(NUMBER_REGEX, text)
        return matches[-1] if matches else None

    def evaluate_answer(self, answer: str, ground_truth: str):
        last_answer_number = self.extract_last_number(answer)
        last_ground_truth_number = self.extract_last_number(ground_truth)

        return last_answer_number == last_ground_truth_number

    def evaluate(self, input_file: str):
        results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Evaluating", unit="example"):
            item = json.loads(line)
            problem = item['premise']
            answer = item['reasoning']
            ground_truth = item['true_answer']

            predicted = self.evaluate_answer(answer, ground_truth)

            results.append({
                "premise": problem,
                "reasoning": answer,
                "true_answer": ground_truth,
                "predicted": predicted
            })

        print(f"Evaluated {len(results)} examples.")
        return results

# Example Usage:
# evaluator = RobustExactMatchEvaluator()
# evaluation_results = evaluator.evaluate('your_data.jsonl')