def escape_braces(template):
    '''Templates with braces must be escaped for formatting to work: } becomes }} and { becomes {{, except for the placeholder {question} which must be kept as is.'''
    tmp_str = 'PLACEHOLDER_ASDF'
    return template.replace('{question}', tmp_str).replace('{', '{{').replace('}', '}}').replace(tmp_str, '{question}')


_SIMPLE_PROMPT_TEMPLATE = '''You are a math tutor. You will be given a math question and you need to answer it step by step, in JSON format. Let's think step by step.

{
  "question": "Billy sells DVDs. He has 8 customers on Tuesday. His first 3 customers buy one DVD each.  His next 2 customers buy 2 DVDs each.  His last 3 customers don't buy any DVDs. How many DVDs did Billy sell on Tuesday?",
  "cot_steps": [
    "The first 3 customers buy 1 DVD each, so they buy 3 * 1 = 3 DVDs.",
    "The next 2 customers buy 2 DVDs each, so they buy 2 * 2 = 4 DVDs.",
    "The last 3 customers buy 0 DVDs each, so they buy 3 * 0 = 0 DVDs.",
    "In total, Billy sells 3 + 4 + 0 = 7 DVDs.",
    "So, the answer is 7."
  ]
},
{
  "question": "{question}",
  "cot_steps": [
'''

SIMPLE_PROMPT_TEMPLATE = escape_braces(_SIMPLE_PROMPT_TEMPLATE)
    
