from typing import Optional
import torch
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM

from core.chain import ListChains
from core.clusterer import Clusterer
from core.constants import IdCluster
from core.experiment_config import ExperimentConfig
from prompting.prompts import INTERMEDIATE_ANSWER_PROMPT_TEMPLATE
import requests
import json
import time

from google import genai

from core.experiment_config import experiment_config
client = genai.Client(api_key = experiment_config.gemma_api_key)


# Keep the same stopping criteria
class StopAfterIntermediateAnswer(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings=("\n", "\n\n")):
        super().__init__()
        self.stop_strings = [tokenizer.encode(stop_string, add_special_tokens=False)[0] for stop_string in stop_strings]

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] in self.stop_strings


def call_gemma_entailment_api(prompt: str, config:ExperimentConfig) -> str:
    """
    Sends the prompt to Gemma 3 27B endpoint and returns 'Yes' or 'No' string.
    """
    max_retries = 20
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model = "gemma-3-27b-it", contents = prompt
            )
            return response.text.strip().lower()
        except Exception as e:
            if attempt < max_retries:
                print(f"[Attempt {attempt}] Request failed: {e}. Retrying in 1 second...")
                time.sleep(1)
            else:
                print(f"[Attempt {attempt}] Request failed: {e}. No more retries left.")
                raise

class EntailmentCluster(Clusterer):

    def __init__(self, config: ExperimentConfig, model_generate: AutoModelForCausalLM, tokenizer_generate):
        self.config = config
        self.tokenizer_generate = tokenizer_generate
        self.model_generate = model_generate.to(config.device)
        self.model_generate.eval()

    def generate_intermediate_answer(self, question: str, cot_steps: list[str]) -> str:
        filled_prompt = INTERMEDIATE_ANSWER_PROMPT_TEMPLATE.format(
            question=question,
            cot_steps='"' + '",\n\t\t"'.join(cot_steps) + '"'
        )

        stop_criteria = StoppingCriteriaList([StopAfterIntermediateAnswer(self.tokenizer_generate)])
        tokenized = self.tokenizer_generate(filled_prompt, return_tensors="pt").to(self.config.device)
        with torch.no_grad():
            output_tokenized = self.model_generate.generate(
                **tokenized,
                max_new_tokens=200,
                do_sample=True,
                stopping_criteria=stop_criteria
            )
        output = self.tokenizer_generate.decode(output_tokenized[0], skip_special_tokens=True)
        return output.split('"final_answer": ')[-1].replace('\n', '').replace('"', '')

    def __call__(self, chains: ListChains, question: Optional[str] = None) -> list[IdCluster]:
        assert question is not None, "Question must be provided for entailment clustering"

        intermediate_answers = []
        for chain in chains:
            cot_steps: list[str] = chain.get_generated_steps()
            intermediate_answer = self.generate_intermediate_answer(question, cot_steps)
            intermediate_answers.append(intermediate_answer)

        clusters = []
        for i in range(len(intermediate_answers)):
            found = False
            for j in range(len(clusters)):
                if self.compute_entailment(intermediate_answers[i], intermediate_answers[clusters[j][0]], question):
                    clusters[j].append(i)
                    found = True
                    break
            if not found:
                clusters.append([i])
        return clusters

    def compute_entailment(self, answer1: str, answer2: str, question: str) -> bool:
        """Use Gemma API to check bidirectional entailment between answer1 and answer2."""

        prompt_template = (
            "Question: {question}\n"
            "Answer A: {a1}\n"
            "Answer B: {a2}\n"
            "Does Answer A entail Answer B?\n"
            "Answer with Yes or No."
        )

        def check_entailment(premise: str, hypothesis: str) -> bool:
            prompt = prompt_template.format(question=question, a1=premise, a2=hypothesis)
            response = call_gemma_entailment_api(prompt, self.config).strip().lower()
            return response.startswith("yes")

        return check_entailment(answer1, answer2) and check_entailment(answer2, answer1)