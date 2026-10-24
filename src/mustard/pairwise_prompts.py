"""Recovered pairwise judge prompts; judge model is configured separately."""

system_prompt_template = '''
Please act as an impartial judge. You will be given a description of a industrial product, which may contain an anomaly, along with two anomaly detection responses from different systems.
Your task is to evaluate which system identifies the anomaly more accurately and explains its reasoning more clearly. You must choose only one of the two answers.
Your evaluation should be based on the following criteria:
Accuracy - Which response more correctly identifies the anomaly in the scene.
Clarity - Which response provides a clearer, more logical, and more detailed explanation of its reasoning.
Relevance - Which response stays focused on the scene and avoids unrelated or speculative information.
Begin your evaluation by objectively comparing the two responses and providing a concise explanation of your reasoning.
Avoid any bias based on the order of responses, their length, or the names of the systems.
After providing your explanation, give your final verdict in the following strict format:
[[A]] if assistant A is better
[[B]] if assistant B is better
'''

user_prompt_template = '''
[Video description]
{description}
[The Start of System A's anomaly detection result]
{answer_a}
[The End of System A's Answer]
[The Start of System B's anomaly detection result]
{answer_b}
[The End of System B's Answer]
'''

def create_prompt(description, answer_a, answer_b):
    system_prompt = system_prompt_template.strip()
    user_prompt = user_prompt_template.format(
        description=description,
        answer_a=answer_a,
        answer_b=answer_b
    ).strip()
    return system_prompt, user_prompt
