import re
import json
import sys
import openai
import argparse
from metrics import *
import os
from openai import OpenAI

try:
    openai_api_key = os.environ["OPENAI_API_KEY"]
    openai_api_base = os.environ["OPENAI_API_BASE"]
except KeyError as e:
    print(f"Missing environment variable: {e}")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='GPT-4 Evaluation Tool')
    parser.add_argument('-e', '--eva_path', type=str, required=True, 
                        help='Path to the data for evaluation.')
    parser.add_argument('-t', '--eva_type', type=str, choices=['ethics', 'rag'], required=True, 
                        help='Type of evaluation: "ethics" or "rag".')
    parser.add_argument('-r', '--res_path', type=str, required=True, 
                        help='Path to save the evaluation results.')
    args = parser.parse_args()
    return args


def gpt4(prompt):
    model = 'gpt-4'
    openai_api_key = openai_api_key
    openai_api_base = openai_api_base
    message = [{"role": "user", "content": prompt}]
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    completion = client.chat.completions.create(model=model, messages=message)
    return completion.choices[0].message.content


def clean_json_string(json_string):
    cleaned_json_string = re.sub(r'[^\x00-\x7F]+', '', json_string)
    cleaned_json_string = re.sub(r'\n{1,}', '', cleaned_json_string).replace("\"\"", "")
    return cleaned_json_string

def evaluation(eva_path, res_path, eval_type):
    results = []
    with open(eva_path, 'r', encoding='utf-8') as f:
        answers = json.load(f)
    
    for answer in answers:
        prompt = evaluate_prompt(answer) if eval_type == "rag" else ethics_evaluation(answer)
        ans = gpt4(prompt)
        matches = re.findall(r'\{[^{}]*\}', ans)
        while not matches:
            ans = gpt4(prompt)
            matches = re.findall(r'\{[^{}]*\}', ans)
        
        cleaned_json_string = clean_json_string(matches[0])
        score = json.loads(cleaned_json_string)
        results.append(score)
        with open(res_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    return


if __name__ == "__main__":
    args = parse_args()
    evaluation(args.eva_path, args.res_path, args.eva_type)