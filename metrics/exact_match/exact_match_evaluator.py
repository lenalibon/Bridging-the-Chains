
import os
from google import genai
import json
from tqdm import tqdm

PROMPT_TEMPLATE = """Given the problem, the answer and the ground truth answer \
tell whether the given answer is correct based on the ground truth one. The 
last word in your answer should be "Yes" if the answer is correct and "No" \
otherwise. Just compare the given answer with the ground truth and don't try
to solve the problem yourself.
Problem: {problem}
Answer: {answer}
Ground truth: {ground_truth}
"""

class ExactMatchEvaluator:
    def __init__(self, model : str = "gemma-3-27b-it"):
        self.__client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.__model = model

    def evaluate_answer(self, problem : str, answer : str, ground_truth : str):
        prompt = PROMPT_TEMPLATE.format(
            problem = problem,
            answer = answer,
            ground_truth = ground_truth
        )
        try:
            response = self.__client.models.generate_content(
                model=self.__model, contents=prompt
            )
            result_text = response.text.strip().lower()
            return result_text.endswith("yes.") or result_text.endswith("yes")
        except Exception as e:
            print("Error in parsing:", e)
            return False
    
    def evaluate(self, input_file: str):
        results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Evaluating", unit="example"):
            item = json.loads(line)
            problem = item['premise']
            answer = item['gpt-3']
            ground_truth = item['hypothesis'].split(
                "IGNORE THIS. Ground truth here for reference. "
            )[1]

            predicted = self.evaluate_answer(problem, answer, ground_truth)

            results.append({
                "key": item.get("key", ""),
                "predicted": predicted
            })
        

        print(f"Evaluated {len(results)} examples.")
        return results