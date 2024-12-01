import string
import re
def vectors_db(embed_path, db_path):
    query_instruction = "为这个句子生成表示以用于检索相关文章："
    embeddings = HuggingFaceBgeEmbeddings(model_name= embed_path,
                                            model_kwargs={'device': 'cuda'},
                                            query_instruction=query_instruction) 
    vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return vectordb
import random
import json
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
import re
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from sentence_transformers import CrossEncoder
import logging
import argparse
from FlagEmbedding import FlagModel
from sklearn.cluster import KMeans
import logging
EMBED_PATH = "/data/yangdi/bge-large-zh-v1.5"
PROBOOKS_DB_PATH = "/data/yangdi/Pro-bge-1.5-300"
KNOW_DB_PATH = "/data/yangdi/SS-bge-1.5-300"
pro_vectordb = vectors_db(EMBED_PATH, PROBOOKS_DB_PATH)
print('Professional knowledge database gets ready !')
know_vectordb = vectors_db(EMBED_PATH, KNOW_DB_PATH)
print('Social science knowledge database gets ready !')

RERANKER_PATH = '/data/pretrained_models/maidalun/bce-reranker-base_v1'
model = CrossEncoder(RERANKER_PATH, max_length=512)

def retrieval_RAGREK(query):    
    pro_chunks = []
    know_chunks = []
    chunks_pro = pro_vectordb.similarity_search_with_score(query, k=70)
    chunks_know = know_vectordb.similarity_search_with_score(query, k=70)
    for i in range(70):
        pro_chunks.append(chunks_pro[i][0].page_content)
    for i in range(70):
        know_chunks.append(chunks_know[i][0].page_content)
    return pro_chunks, know_chunks
def get_sentence_pairs(query, chunks):
    sentence_pairs = [[query, chunks[i]] for i in range(70)]
    return sentence_pairs

def reranker_top2(sentence_pairs):
# calculate scores of sentence pairs
    sort_scores = {}
    scores = model.predict(sentence_pairs)
    for i, score in enumerate(scores):
        sort_scores[i] = score
    scores_ = sorted(sort_scores.items(), key=lambda item: item[1], reverse=True)
    index1, index2 = scores_[0][0], scores_[1][0]
    return index1, index2
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from dashscope import Generation
from http import HTTPStatus
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
import dashscope
dashscope.api_key="sk-9f54f947734647c89c7d1e37a8054c41"
def query_Qwen_7B(prompt, stop_=None):
    # Instead of using model.chat(), we directly use model.generate()
    # But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
    device = "cuda" # the device to load the model onto

    # # Now you do not need to add "trust_remote_code=True"
    # query_model = AutoModelForCausalLM.from_pretrained(
    #     QWEN_PATH,
    #     torch_dtype="auto",
    #     device_map="auto"
    # )
    # tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
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
def get_answer_intermidiate(question, history):
    pro_chunks, know_chunks = retrieval_RAGREK(query=question)
    pro_sentence_pairs = get_sentence_pairs(question,pro_chunks)
    know_sentence_pairs = get_sentence_pairs(question,know_chunks)
    pro_index1, pro_index2 = reranker_top2(pro_sentence_pairs)
    know_index1, know_index2 = reranker_top2(know_sentence_pairs)
    knowledge = pro_chunks[pro_index1] + '\n' + pro_chunks[pro_index2] + '\n' + know_chunks[know_index1] + know_chunks[know_index2]
    prompt = f"""
    你是一名心理学专家，根据对话历史，给出你的回答，回答应简洁明了。
    
    对话历史：{history}
    可参考知识知识：{knowledge}
    
    仅返回回答内容，禁止出现其他字段。

    """
    res = query_Qwen_7B(prompt)
    return res
def get_answer_final(history):

    prompt = f"""
    你是一名心理学专家，根据对话历史，综合之前的问题和中间回答给出最终回答，最终回答应该内容丰富，尽量生成长文本。
    
    对话历史：{history}
    
    仅返回回答内容，禁止出现其他字段。

    """
    res = query_Qwen_7B(prompt)
    return res

with open("/data/yangdi/data/psyqa_test.json", 'r', encoding='utf-8') as f:
    datas = json.load(f)
def judge_follow_up(history):
    prompt = f"""
    根据对话历史，请判断是否需要在下一步提出新的问题。若需要仅返回是，若不需要额外的问题就可以回答Qestion返回否，禁止给出理由。

    对话历史：{history}

    """
    res = query_Qwen_7B(prompt)
    return res    
    
def get_follow_up_question(history):
    prompt = f"""
    根据对话历史，给出后续问题。仅给出问题，以一句话给出问题，禁止出现回答。

    对话历史：{history}

    """
    res = query_Qwen_7B(prompt)
    return res 
res = []
for i in range(0, 50):
    data = datas[i]
    history = ''
    question = extraction_question(data)
    history += "Question: " + question + '\n'
    cnt = 0
    while "是" in judge_follow_up(history) and cnt <= 4:
        history += "是否需要后续问题：" + "是" + '\n'
        follow_question = get_follow_up_question(history)
        history += "后续问题：" + follow_question + '\n'
        history += "中间回答："
        answer = get_answer_intermidiate(follow_question, history)
        history += answer + '\n'
        cnt += 1
    history += "最后答案："
    answer = get_answer_final(history)
    history += answer
    print(history)
    res.append(question + '[ANSWER]' + answer)
    
    with open('./Res-RAGREK/Self-Ask-14B_4.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        
        