import json
import os
import random
from string import Template
from typing import List

# by field of expertise, and approach
DIVERSIFICATION = ["Alan Turing, the father of computer science",
                   "Dr. Patel, the mathematician who specializes in algebra",
                   "Ada Lovelace, a visionary computer scientist",
                   "Prof. Chen, a logician who emphasizes formal deduction",
                   "Isaac Newton, a physicist with a mathematical mind",
                   "Sophie Germain, a number theory expert",
                   "Prof. Kaur, a computer vision and pattern recognition researcher",
                   "a skeptical peer-reviewer questioning every assumption",
                   "Marie Curie, a physicist and chemist known for scientific rigor",
                   "Grace Hopper, a systems thinker and pioneering computer scientist",
                   "Niels Bohr, a physicist comfortable with paradox and complementarity",
                   "Leonhard Euler, a prolific problem solver and master of elegant solutions"
                   ]

FEW_SHOT_PROMPT = Template(
    """{
    "question": "${example_question}",
    "cot_steps": [
        ${example_cot_steps}
    ]
},
"""
)

MASTER_PROMPT = Template(
    """You will be given math questions. Think step by step and write the \
solution in JSON format like this: ["step_1", "step_2", ..., "step_n"]. \
Every step should contain just the content of the reasoning and nothing else. \
Use double quotes for the steps. The text of the last step needs to have the final answer at the end.

${few_shot_block}{
    "question": "Think like ${expert} and solve the problem. ${question}",
    "cot_steps": [
"""
)


def _load_few_shots(folder_path: str = "../auto-cot/gsm8k_few_shot/", n_shots: int = 8) -> list:
    prompt_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    random.shuffle(prompt_files)

    few_shots = []
    for f in prompt_files:
        with open(os.path.join(folder_path, f), 'r') as file:
            data = json.load(file)
            random.shuffle(data)
            for i in data:
                few_shots.append(i)
                if len(few_shots) >= n_shots:
                    return few_shots

    return few_shots


def _format_few_shots(shot: List[dict]) -> str:
    return "".join(FEW_SHOT_PROMPT.substitute(
        example_question=s["question"],
        example_cot_steps = ",\n\t\t".join(json.dumps(step) for step in s["cot_steps"])
    ) for s in shot)





def build_prompt(question: str, folder_path: str = "../auto-cot/gsm8k_few_shot", n_shots: int = 8,
                 chain_index: int = 0) -> str:
    """
    Builds a single prompt for the given question and chain index using few-shot examples and diversification.
    """
    # Add few shot examples
    few_shots = _load_few_shots(folder_path, n_shots)
    few_shot_block = _format_few_shots(few_shots)

    # Diversify
    random.shuffle(DIVERSIFICATION)
    expert = DIVERSIFICATION[chain_index % len(DIVERSIFICATION)]
    return MASTER_PROMPT.substitute(
        few_shot_block=few_shot_block,
        expert=expert,
        question=question
    )
