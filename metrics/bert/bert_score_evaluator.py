import json
from tqdm import tqdm
from bert_score import score

class BertScoreEvaluator:
    def __init__(self, model_type: str = "bert-base-uncased"):
        self.model_type = model_type

    def evaluate_answer(self, answer: str, ground_truth: str) -> float:
        try:
            _, _, F1 = score([answer], [ground_truth], model_type=self.model_type, verbose=False)
            return F1[0].item()
        except Exception as e:
            print("Error during BERTScore evaluation:", e)
            return 0.0  

    def evaluate(self, input_file: str):
        results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Evaluating", unit="example"):
            item = json.loads(line)
            question = item['premise']
            answer = item['reasoning']
            ground_truth = item['true_answer']

            similarity = self.evaluate_answer(answer, ground_truth)

            results.append({
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "similarity": similarity
            })
        
        print(f"Evaluated {len(results)} examples.")
        return results
