import json
import os
import random

from gensim.similarities.docsim import query_shard

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


def _format_one_shot(shot: dict) -> str:
    return f"Q: {shot['question']}\n" \
           f"A: {shot['cot_steps']}\n\n"


def build_prompt(question: str, folder_path: str = "../auto-cot/gsm8k_few_shot", n_shots: int = 8,
                 n_chains: int = 5) -> list:
    """
    Build prompts for creating the chains using few-shot examples and diversification.
    :param question: New question
    :param folder_path: Path to the folder containing few-shot examples
    :param n_shots: Number of few-shot examples to use
    :param n_chains: Number of chains to create the prompts for
    :return:
    """
    # Add few shot examples
    few_shots = _load_few_shots(folder_path, n_shots)
    formatted_few_shots = "\n\n".join([_format_one_shot(shot) for shot in few_shots])

    prompt_chains = []
    # Diversify
    random.shuffle(DIVERSIFICATION)
    for c in range(n_chains):
        expert = DIVERSIFICATION[c % len(DIVERSIFICATION)]
        prompt = f"{formatted_few_shots}\n\nQ: Think like {expert} and solve the problem. {question}\nA: "
        prompt_chains.append(prompt)

    return prompt_chains
