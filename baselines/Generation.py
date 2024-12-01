import dashscope
from dashscope import Generation
from http import HTTPStatus
from dashscope import Generation
import json

dashscope.api_key="sk-9f54f947734647c89c7d1e37a8054c41"
def query_Qwen_7B(prompt, stop_=None):

    device = "cuda" # the device to load the model onto
    messages = [
        {"role": "system", "content": "You are a psychological expert and specialized in psychological knowledge."},
        {"role": "user", "content": prompt}
    ]

    # Directly use generate() and tokenizer.decode() to get the output.
    # Use `max_new_tokens` to control the maximum output length.
    response = Generation.call(
        'qwen1.5-14b-chat',
        messages=messages,
        stop=stop_
    )
    if response.status_code == HTTPStatus.OK:
        response = response.output["text"]
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
    return response

import re
def extraction_question(text):
    question_match = re.search(r'\[QUESTION\](.*?)\[ANSWER\]', text)
    if question_match:
        question = question_match.group(1).strip()
        return question
    else:
        raise ValueError("找不到问题部分")

with open("/data/yangdi/data/psyqa_test.json", encoding='utf-8') as f:
    datas = json.load(f)
res = []
for i in range(50):
    psyqa_data = datas[i]
    quetion = extraction_question(psyqa_data)
    aa = query_Qwen_7B(quetion)
    res.append(quetion + '[ANSWER]' + aa)
    with open('./Res-RAGREK/Qwen-70B-Generation.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        
    
