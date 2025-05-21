from string import Template


SIMPLE_PROMPT_TEMPLATE = Template('''You are a math tutor. You will be given a math question and you need to answer it step by step, in JSON format. Let's think step by step.

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
  "question": "${question}",
  "cot_steps": [
''')



SUMMARIZING_PROMPT_TEMPLATE_v1 = Template('''Question: "${question}"
Here are some potential solutions:
${solutions}

Given these solutions, please consider their consistency, and please provide a correct solution to the question with clear reasoning and step-by-step calculations.
Please state the final answer clearly at the last step.

{
  "question": "${question}",
  "cot_steps": [
''')

SUMMARIZING_PROMPT_TEMPLATE_v2 = Template('''You are a math tutor. You will be given a math question and some example solutions and you need to answer the question step by step, in JSON format. Let's think step by step.

{
  "question": "${question}",
}
${solutions}
{
  "question": "${question}",
  "cot_steps": [
''')

SUMMARIZING_PROMPT_TEMPLATE = SUMMARIZING_PROMPT_TEMPLATE_v1



# "question": "${question}",
CHAIN_JSON_TEMPLATE = Template('''{
  "question": "${question}",
  "cot_steps": [
    ${cot_steps}
  ]
}''')



DEBUG_PROMPT = '''The Countess of Sakharovka.
By F. M. Dostoevsky
1868

The cherry garden was cloaked in a silence so profound that even the wind seemed ashamed to stir the blossoming branches. Pale petals fell one by one, like the slow unraveling of memory.

At the edge of the grove, near the marble fountain where no water had flowed since the old Count’s death, the Countess stood motionless, her veil trailing behind her in the grass.'''


from string import Template


SIMPLE_PROMPT_TEMPLATE = Template('''You are a math tutor. You will be given a math question and you need to answer it step by step, in JSON format. Let's think step by step.

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
  "question": "${question}",
  "cot_steps": [
''')




CHAIN_JSON_TEMPLATE = Template('''{
  "question": "${question}",
  "cot_steps": [
    ${cot_steps}
  ]
}''')



DEBUG_PROMPT = '''The Countess of Sakharovka.
By F. M. Dostoevsky
1868

The cherry garden was cloaked in a silence so profound that even the wind seemed ashamed to stir the blossoming branches. Pale petals fell one by one, like the slow unraveling of memory.

At the edge of the grove, near the marble fountain where no water had flowed since the old Count’s death, the Countess stood motionless, her veil trailing behind her in the grass.'''

