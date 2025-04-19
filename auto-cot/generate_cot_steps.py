from google import genai
import ast

client = genai.Client(api_key="AIzaSyAWsCwIgB8MIUHobdz9u3YuQzUoiRIMj-c")

PROMPT_TEMPLATE = """Given the problem below write the solution. Think step by \
step and write the solution in a format of an array of strings like this: \
["step_1", "step_2", ..., "step_n"]. Every step should contain just the content \
of the reasoning and nothing else. Use double quotes for the steps. \
Last step needs to have the answer as last item. Do not write json, python or \
markdown. The text of the last step needs to have the answer at the end.
Example of question: How many fingers does a dog have?
Example of answer: ["Dogs have 4 legs.", "Every paw has 4 fingers.", \
"A dog has 16 fingers.", "So, the answer is 16."]
{problem}"""

def generate_cot_steps(problem, model = "gemma-3-27b-it"):
    prompt = PROMPT_TEMPLATE.format(problem = problem)

    response = client.models.generate_content(
        model = model, contents = prompt
    )
    try:
        steps = ast.literal_eval(response.text)
    except:
        print("error in parsing")
        return []
    return steps