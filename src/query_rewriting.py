import re
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.CascadeRCG_prompt import query_rewriting_prompt
from src.get_json import change_to_json

def extract_question_psyqa(text):
    question_match = re.search(r'\[QUESTION\](.*?)\[ANSWER\]', text)
    if question_match:
        question = question_match.group(1).strip()
        return question
    else:
        raise ValueError("找不到问题部分")
    
def extract_question_smilechat(text):
    client_dialogues = re.findall(r"Client: (.*?)\n", text)
    all_dialogue = " ".join(client_dialogues)
    return all_dialogue
    
def rewriting(question, query_model, single_turn):
    if single_turn:
        question = extract_question_psyqa(question)
    else:
        question = extract_question_smilechat(question)
    prompt = query_rewriting_prompt(question)
    max_retries = 5  # 设置最大重试次数
    attempts = 0
    while attempts < max_retries:
        try:
            res = query_model(prompt)
            content = change_to_json(res)
            data_json = json.loads(content)
            break  
        except AttributeError as e:
            attempts += 1
            if attempts >= max_retries:
                print("Max retries reached. Exiting...")
                raise e
    queriess = []
    queriess.append(question)
    for data in list(data_json.values()):
        queriess.append(data)
    return queriess
    
    