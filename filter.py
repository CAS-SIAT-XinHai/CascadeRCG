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
from openai import OpenAI
import openai

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def vectors_db(embed_path, db_path):
    query_instruction = "为这个句子生成表示以用于检索相关文章："
    embeddings = HuggingFaceBgeEmbeddings(model_name= embed_path,
                                            model_kwargs={'device': 'cuda'},
                                            query_instruction=query_instruction) 
    vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return vectordb

def query_gpt(prompt):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="sk-icMEnTNFeYyYU5AvmyJd4EvZuQSYTZKM3uKyUYJGTZRKmBPF",
        base_url="https://api.agicto.cn/v1"
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



import json, random
def generate_random_data(rewriting_path):
    with open(rewriting_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    random_item = random.choice(json_data)
    return random_item

import re
def extraction_question(text):
    question_match = re.search(r'\[QUESTION\](.*?)\[ANSWER\]', text)
    if question_match:
        question = question_match.group(1).strip()
        return question
    else:
        raise ValueError("找不到问题部分")
    
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

import dashscope
from dashscope import Generation
from http import HTTPStatus
from dashscope import Generation

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


def query_rewriting_prompt(question):
    prompt = f"""
    你是一名心理医生，需要根据以下用户的提问来查询百科全书，根据用户的提问，你想向百科全书提出的问题是什么？
    请采用该心理现象是什么，为什么会产生该心理现象，怎么做的逻辑提问,用json的格式返回。
    用户的提问：{question}
    
    ##注意
    问题必须语义完整。
    每个问题应该独立，不依赖前一个问题。问题必须是子问题，即不包含其他问题，不超过三个问题。
    
    ##格式：
    {{"question1": "aaa"，
    "question2": "bbb"}}
    """
    return prompt

def get_json(res):
    pattern = r'\{([^}]*)\}'
    match = re.search(pattern, res, re.DOTALL)
    content = match.group(0)  # 提取匹配到的内容
    return content

def judge_prompt(question, knowledge):
    prompt = f"""
    请判断下面知识片段内容是否对用户提问有帮助（部分内容有帮助也算有帮助）。若有帮助返回"是"，若无帮助返回“否”，仅返回是或否，不要给出理由。
    
    用户的提问：{question}
    知识片段：{knowledge}
    """
    return prompt

def extraction_prompt(test_data):
    extraction_prompt = f"""
    请对下面的问题的回答进行总结。

    ##注意：
    仅对文本进行总结，不是你去回答问题！
    要求保持心理学专业术语，实验内容、包括人名、书籍名、调查数据来源、引用来源，专业心理名词，专业词语解释，案例内容等，当涉及到以上内容时应完整复制上述内容。
    问题应保持一致。

    ##问题回答对：
    {test_data}

    ##格式：
    {{"问题aaa"：“回答bbb”}}
    """
    return extraction_prompt

def without_extraction(retrieval_filter):
    res = []
    for data in retrieval_filter:
        data = list(data)
        res.append("问题："+data[0]+ "回答：" +data[1] )
    return res
        
        


def extraction_by_query(retrieval_filter):
    hashtable = dict()
    for data in retrieval_filter:
        data = list(data)
        data[0] = str(data[0])
        data[0] = "问题: " + data[0]
        if data[0] not in hashtable:
            hashtable[data[0]] = []
            hashtable[data[0]].append(data[1])
        else:
            hashtable[data[0]].append(data[1])
    extraction = []
    for question, ans in hashtable.items():
        prompt = extraction_prompt((question + '\n' ,ans))
        res = query_model(prompt)
        extraction.append(res)
    return extraction

def filter(queries, retrieval_knowledge):
    retrieval_filter = []
    for question, knowledge in zip(queries, retrieval_knowledge):
        prompt = judge_prompt(question, knowledge)
        # res = query_Qwe(prompt, query_model, tokenizer)
        res = query_model(prompt)
        if "是" in res:
            retrieval_filter.append((question,knowledge))
    return retrieval_filter
    
def get_cross_question_psyqa(cross_retrieval):
    question_psyqa = []
    for question in cross_retrieval:
        try:
            content = get_json(question)
            logging.debug("???????????????????????????????????")
            logging.debug(content)
            logging.debug("???????????????????????????????????")
            data_json = json.loads(content)
            logging.debug("???????????????????????????????????")
            logging.debug(type(data_json))
            logging.debug("???????????????????????????????????")
            question_psyqa.append(data_json.keys())
        except Exception as e:
            print(e)
            
    return question_psyqa

def all_question_match(text):
    pattern = r"(.{6,}?)(?=：|:)"
    matches = re.findall(pattern, text, re.DOTALL)
    try:
        return matches[0]
    except Exception as e:
        return ""
    
# 安装包(Python >= 3.7)：pip install qianfan
import os
import qianfan

os.environ["QIANFAN_ACCESS_KEY"] = "ALTAKe3TQb7eEsSRasJhaCwsGA"
os.environ["QIANFAN_SECRET_KEY"] = "1c1e9a6affc841b9bea540791b3ded84"



def query_qianfan(prompt):

    chat_comp = qianfan.ChatCompletion()

    resp = chat_comp.do(model="Meta-Llama-3-70B", messages=[{
        "role": "user",
        "content": prompt
    }], )

    print(resp["body"]['result'])
    return resp["body"]['result']

def all_question_extraction(all):
    pairs = dict()
    for a in all:
        if isinstance(a, list):
            for b in a:
                question = all_question_match(b).replace('{', '').replace('}', '')
                logging.debug("????????????????????????????????????????????")
                logging.debug(question)
                logging.debug(pairs)
                logging.debug("????????????????????????????????????????????")
                if question and question not in pairs:
                    pairs[question] = []
                    pairs[question].append(b.replace(question, "").replace("{", "").replace("}", "").lstrip(":").lstrip("：").lstrip())
                elif question:
                    pairs[question].append(b.replace(question, "").replace("{", "").replace("}", "").lstrip(":").lstrip("：").lstrip())
        else:
                question = all_question_match(a).replace('{', '').replace('}', '')
                if question and question not in pairs:
                    pairs[question] = []
                    pairs[question].append(a.replace(question, "").replace("{", "").replace("}", "").lstrip(":").lstrip("：").lstrip())
                elif question:
                    pairs[question].append(a.replace(question, "").replace("{", "").replace("}", "").lstrip(":").lstrip("：").lstrip())
    questions = list(pairs.keys())
    return questions, pairs

def cluster_K_Means(questions, pairs, encoder_model):
    model = encoder_model
    question_embeddings = model.encode(questions)

    # 使用 K-Means 聚类
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(question_embeddings)
    labels = kmeans.labels_
    clusters = dict()
    for question, label in zip(questions, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(question)
    final_pairs = []
    for i, questions in clusters.items():
        ans = []
        max_question = max(questions, key=len)
        for ques in questions:
            ans.append(pairs[ques])
        final_pairs.append((max_question, ans))
    return final_pairs

def single_turn_rewriting_prompt(dialogue, knowledge):
    single_turn_rewriting_prompt = f"""
    你是一名精通心理学知识的专家。请你基于下述用户的描述，对用户描述分析后，利用下面知识进行回答，以提高回答的专业性和知识性。
    
    ##注意：  
    1. 以表达善意或爱意或安慰的话开头，以提供情感上的支持，回答过程中应始终保持尊重、热情、真诚、共情、积极关注的态度。 
    2. 请确保按照信息的先后顺序、逻辑关系和相关性组织文本，同时在回答中添加适当过渡句帮助读者更好理解内容之间的关系和转变。  
    3. 请你尽可能生成长文本,用中文返回,内容应该知识丰富完整且有深度
    3. 仅返回最终的回答，不要出现其他内容。
    
    ##重点：
    要求保留心理学专业术语，实验内容、包括人名、书籍名、调查数据来源、引用来源，专业心理名词，专业词语解释，案例内容等，当涉及到以上内容时应完整复制上述内容。
    尽可能多得利用下面的知识。
    
    ##用户的描述：{dialogue} 
    
    ##相关知识：{knowledge} 
    最终回答是：
    """
    return single_turn_rewriting_prompt

def RAG_prompt(dialogue, knowledge):
    RAG_prompt = f"""
    你是一名精通心理学知识的专家。请你基于下述用户的描述，对用户描述分析后，利用下面知识进行回答:
    
    ##用户的描述: {dialogue}
    ##相关知识: {knowledge}
    
    最终回答是:
    """
    return RAG_prompt


def muti_turn_rewriting_prompt(dialogue, knowledge):
    muti_turn_rewriting_prompt = f"""
    你是一名具有丰富心理学知识且掌握多种心理咨询技巧的Counselor。请利用下面的知识对下面对话中的每一个Counselor回答进行重写。
    
    ##注意
    1.你应该对每一次Client角色的内容进行回答。
    2.保持Client内容和数量与原对话一致。 
    5.回答内容应提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。
    6.确保回答长度符合咨询场景情况，回答的内容应承接上下文，保持对话符合逻辑且流畅。
    7.回答过程中应始终保持尊重、热情、真诚、共情、积极关注的态度。
    9.用中文返回。
    
    ##重要：
    要求保留心理学专业术语，实验内容、包括人名、书籍名、调查数据来源、引用来源，专业心理名词，专业词语解释，案例内容等，当涉及到以上内容时应完整复制上述内容。
    尽可能多得利用下面的知识。

    
    ##待补全对话:{dialogue}
    ##知识：{knowledge}
    ##对话格式：以Client对话开始，每一个Client后面接一个Counselor的回答，依次交替，每个角色转变应用"\n"隔开。
    
    重写对话是：
    """
    return muti_turn_rewriting_prompt

EMBED_PATH = "/data/yangdi/bge-large-zh-v1.5"
KNOW_DB_PATH = "/data/yangdi/SS-bge-1.5-300"
PRO_DB_PATH = "/data/yangdi/Pro-bge-1.5-300"
know_vectordb = vectors_db(EMBED_PATH, KNOW_DB_PATH)
pro_vectordb = vectors_db(EMBED_PATH, PRO_DB_PATH)
print("all database is get ready!")

def retrieval_RAG(query):    
    all_chunks = []
    chunks_all = all_vectordb.similarity_search_with_score(query, k=6)
    for i in range(6):
        all_chunks.append(chunks_all[i][0].page_content)
    return all_chunks

def model_type(name):
    query = None  
    if name == 'GPT':
        query = query_gpt
    elif name == 'Qwen':
        query = query_Qwen_7B
    elif name == 'qianfan':
        query = query_qianfan
    
    if query is None:
        raise ValueError(f"Unknown model name: {name}")
    
    return query

query_model = model_type('GPT')


if __name__ == "__main__":
    EMBED_PATH = "/data/yangdi/bge-large-zh-v1.5"
    PROBOOKS_DB_PATH = "/data/yangdi/Pro-bge-1.5-300"
    KNOW_DB_PATH = "/data/yangdi/SS-bge-1.5-300"
    pro_vectordb = vectors_db(EMBED_PATH, PROBOOKS_DB_PATH)
    print('Professional knowledge database gets ready !')
    know_vectordb = vectors_db(EMBED_PATH, KNOW_DB_PATH)
    print('Social science knowledge database gets ready !')

    with open("/data/yangdi/RAGREK/Datas/SMILE-50-Test-Query.json", encoding='utf-8') as f:
        datas = json.load(f)
        
    with open("/data/yangdi/RAGREK/Datas/SMILE-50-Test-Answer_2.json", encoding='utf-8') as f:
        dialogues = json.load(f)

    RERANKER_PATH = '/data/pretrained_models/maidalun/bce-reranker-base_v1'
    model = CrossEncoder(RERANKER_PATH, max_length=512)
    encoder_model = FlagModel('/data/yangdi/bge-large-zh-v1.5',
                    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                    use_fp16=True)
    psyqa_questions = []

    for data in datas:
        psyqa_questions.append(data)
    pro = []
    know = []
    anses = []
    for i in range(0, 50):
        question = psyqa_questions[i]
        question = extraction_question(question)
        prompt = query_rewriting_prompt(question)
        max_retries = 5  # 设置最大重试次数
        attempts = 0
        while attempts < max_retries:
            try:
                res = query_model(prompt)
                content = get_json(res)
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
        logging.debug(f"##########################################")
        logging.debug(f"-------Rewriting Queried: {queriess}")
        logging.debug(f"##########################################")
        pro_retrieval = []
        know_retrieval = []
        for query in queriess:
            pro_chunks, know_chunks = retrieval_RAGREK(query=query)
            pro_sentence_pairs = get_sentence_pairs(query,pro_chunks)
            know_sentence_pairs = get_sentence_pairs(query,know_chunks)
            pro_index1, pro_index2 = reranker_top2(pro_sentence_pairs)
            know_index1, know_index2 = reranker_top2(know_sentence_pairs)
            pro_retrieval.append(pro_chunks[pro_index1])
            pro_retrieval.append(pro_chunks[pro_index2])
            know_retrieval.append(know_chunks[know_index1])
            know_retrieval.append(know_chunks[know_index2])
        queriess = [item for item in queriess for _ in range(2)]
        pro_retrieval_filter = filter(queriess, pro_retrieval)
        know_retrieval_filter = filter(queriess, know_retrieval)
        pro.append(pro_retrieval_filter)
        know.append(know_retrieval_filter)
        count_pro = [len(i) for i in pro]
        count_know = [len(i) for i in know]
        with open('./smile_count_pro.json', 'w', encoding='utf-8') as f:
            json.dump(count_pro, f,indent=4, ensure_ascii=False)
        with open('./smile_count_know.json', 'w', encoding='utf-8') as f:
            json.dump(count_know, f,indent=4, ensure_ascii=False)
        with open('./smile_pro.json', 'w', encoding='utf-8') as f:
            json.dump(pro, f,indent=4, ensure_ascii=False)
        with open('./smile_know.json', 'w', encoding='utf-8') as f:
            json.dump(know, f,indent=4, ensure_ascii=False)
        pro_extraction = extraction_by_query(pro_retrieval_filter)
        know_extraction = extraction_by_query(know_retrieval_filter)
        # pro_extraction = without_extraction(pro_retrieval_filter)
        # know_extraction = without_extraction(know_retrieval_filter)
        pro_cross_retrieval = []
        know_cross_retrieval = []
        for query in pro_extraction:
            _, know_chunks = retrieval_RAGREK(query=query)
            know_sentence_pairs = get_sentence_pairs(query,know_chunks)
            know_index1, _ = reranker_top2(know_sentence_pairs)
            know_cross_retrieval.append(know_chunks[know_index1])
        for query in know_extraction:
            pro_chunks, _ = retrieval_RAGREK(query=query)
            pro_sentence_pairs = get_sentence_pairs(query,pro_chunks)
            pro_index1, _ = reranker_top2(pro_sentence_pairs)
            pro_cross_retrieval.append(pro_chunks[pro_index1])
        pro_question_psyqa = get_cross_question_psyqa(pro_extraction)
        logging.debug("***********************************************")
        logging.debug(pro_extraction)
        logging.debug(know_extraction)
        logging.debug("***********************************************")
        know_question_psyqa = get_cross_question_psyqa(know_extraction)
        pro_cross_retrieval_filter = filter(queries=pro_question_psyqa, retrieval_knowledge=pro_cross_retrieval)
        know_cross_retrieval_filter = filter(queries=know_question_psyqa, retrieval_knowledge=know_cross_retrieval)
        logging.debug(f"{know_question_psyqa}")
        logging.debug(f"{pro_question_psyqa}")
        pro_cross_extraction = extraction_by_query(pro_cross_retrieval_filter)
        know_cross_extraction = extraction_by_query(know_cross_retrieval_filter)
        # pro_cross_extraction = without_extraction(pro_cross_retrieval_filter)
        # know_cross_extraction = without_extraction(know_cross_retrieval_filter)
        logging.debug("***********************************************")
        logging.debug(pro_cross_extraction)
        logging.debug(know_cross_extraction)
        logging.debug("***********************************************")
        all = [pro_extraction, know_extraction, pro_cross_extraction, know_cross_extraction]
        logging.info(f"最后检索知识数量：{sum(len(x) for x in all)}")
        questions, pairs = all_question_extraction(all=all)
        if len(pairs) > 4:
            final_pairs = cluster_K_Means(questions, pairs, encoder_model)
            final_knowledge = extraction_by_query(final_pairs)
        else:
            final_knowledge = pairs
        # final_knowledge = retrieval_RAG(question)
        prompt = muti_turn_rewriting_prompt(dialogues[i], final_knowledge)
        res = query_model(prompt)
        anses.append(res)
        # logging.info(res)
        with open('./Res-RAGREK/SMILE_GPT-3.5.json', 'w', encoding='utf-8') as f:
            json.dump(anses, f,indent=4, ensure_ascii=False)
        
        
        
        
        
        
        
            
        


    
    
    
    




