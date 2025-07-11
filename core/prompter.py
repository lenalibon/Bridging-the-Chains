from prompting.create_prompts_cot import build_prompt

from prompting.prompts import *

class Prompter:
    def __call__(self, *args, **kwargs) -> str:
        raise NotImplementedError("Must be implemented in subclasses")


class SimplePrompter(Prompter):
    def __init__(self, template: Template = SIMPLE_PROMPT_TEMPLATE, folder_path=None, n_shots=0):
        self.template = template

    def __call__(self, question, chain_index=0):
        return self.template.substitute(question=question)
    
class DiversifiedCoTPrompter(Prompter):
    """
    Returns one few-shot prompt per chain index, with diversified expert.
    """
    def __init__(self,
                 template: Template = None,
                 folder_path: str = "few-shot/gsm8k_few_shot/",
                 n_shots: int = 8):
        self.template = template
        self.folder_path = folder_path
        self.n_shots = n_shots

    def __call__(self, question, chain_index: int = 0) -> str:
        return build_prompt(
            question,
            folder_path=self.folder_path,
            n_shots=self.n_shots,
            chain_index=chain_index
        )