from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import json
import re
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("/data/pretrained_models/MindChat-Qwen-7B-v2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/data/pretrained_models/MindChat-Qwen-7B-v2", trust_remote_code=True)

def extraction_question(text):
    question_match = re.search(r'\[QUESTION\](.*?)\[LABEL\]', text)
    if question_match:
        question = question_match.group(1).strip()
        return question
    else:
        raise ValueError("找不到问题部分")
    
with open("/data/yangdi/data/psyqa_test.json", 'r', encoding='utf-8') as f:
    datas = json.load(f)
    
res = []
for i in range(0, 50):
    data = datas[i]
    question = extraction_question(data)
    response, history = model.chat(tokenizer, question, history=None)
    print(response)
    with open("./Res-RAGREK/MindChat.json", 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)