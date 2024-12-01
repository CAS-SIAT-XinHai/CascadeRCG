from openai import OpenAI
import dashscope
from dashscope import Generation
from http import HTTPStatus
from dashscope import Generation
import os
import qianfan

def model_type(name): 
    query = None
    if name == 'GPT':
        try:
            openai_api_key = os.environ["OPENAI_API_KEY"]
            openai_api_base = os.environ["OPENAI_API_BASE"]
        except KeyError as e:
            print(f"Missing environment variable: {e}")
            exit(1)
        query = query_gpt
    elif name == 'Qwen':
        try:
            dashscope_api_key = os.environ["DASHSCOPE_API_KEY"]
        except KeyError as e:
            print(f"Missing environment variable: {e}")
            exit(1)
        query = query_Qwen_7B
    elif name == 'qianfan':
        try:
            openai_api_key = os.environ["QIANFAN_ACCESS_KEY"]
            openai_api_base = os.environ["QIANFAN_SECRET_KEY"]
        except KeyError as e:
            print(f"Missing environment variable: {e}")
            exit(1)
        query = query_qianfan
    
    if query is None:
        raise ValueError(f"Unknown model name: {name}")
    
    return query
    


def query_gpt(prompt):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"]
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 使用的模型
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},  # 系统角色设定
            {"role": "user", "content": prompt}  # 用户输入
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content  # 返回生成的文本


def query_Qwen_7B(prompt, stop_=None):
    dashscope.api_key=os.environ["DASHSCOPE_API_KEY"]
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


def query_qianfan(prompt):

    chat_comp = qianfan.ChatCompletion()

    resp = chat_comp.do(model="Meta-Llama-3-70B", messages=[{
        "role": "user",
        "content": prompt
    }], )

    print(resp["body"]['result'])
    return resp["body"]['result']
