from string import Template


from .utils import *
from .prompts import *

class Prompter:
    def __call__(self, *args, **kwargs) -> str:
        raise NotImplementedError("Must be implemented in subclasses")


class SimplePrompter(Prompter):
    def __init__(self, template: Template = SIMPLE_PROMPT_TEMPLATE):
        self.template = template

    def __call__(self, question):
        return self.template.substitute(question=question)