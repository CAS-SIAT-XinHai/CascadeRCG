import random
import json
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
import re
import logging
import argparse
from sklearn.cluster import KMeans
import logging
from inference_models import *
from load_database import *
from prompt import *
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

def reranker_top(sentence_pairs, k):
# calculate scores of sentence pairs
    sort_scores = {}
    scores = model.predict(sentence_pairs)
    for i, score in enumerate(scores):
        sort_scores[i] = score
    scores_ = sorted(sort_scores.items(), key=lambda item: item[1], reverse=True)
    indexes = [scores_[i][0] for i in range(k)]
    return indexes


def get_json(res):
    pattern = r'\{([^}]*)\}'
    match = re.search(pattern, res, re.DOTALL)
    content = match.group(0)  # 提取匹配到的内容
    return content

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

def cluster_K_Means(questions, pairs, encoder_model, J):
    model = encoder_model
    question_embeddings = model.encode(questions)

    kmeans = KMeans(n_clusters=J, random_state=42)
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


def retrieval_RAG(query):    
    all_chunks = []
    chunks_all = all_vectordb.similarity_search_with_score(query, k=6)
    for i in range(6):
        all_chunks.append(chunks_all[i][0].page_content)
    return all_chunks

query_model = None

def CascadeRCG(inference_model_type: str, data_path: str, save_path: str, K_1=2, K_2=1, J=4, ablation=False, single_turn=True):    
    global query_model
    query_model = model_type(inference_model_type)
    if ablation:
        extraction = extraction_by_query
    else:
        extraction = without_extraction
        
    if single_turn:
        answer_prompt = single_turn_rewriting_prompt
    else:
        answer_prompt = muti_turn_rewriting_prompt

    with open(data_path, encoding='utf-8') as f:
        datas = json.load(f)

    psyqa_questions = []

    for data in datas:
        psyqa_questions.append(data)
    pro = []
    know = []
    anses = []
    length = len(psyqa_questions)
    for i in range(length):
        question = psyqa_questions[i]
        if single_turn:
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
            pro_indexes = reranker_top(pro_sentence_pairs, K_1)
            know_indexes = reranker_top(know_sentence_pairs, K_1)
            for pro_index in pro_indexes:
                pro_retrieval.append(pro_chunks[pro_index])
            for know_index in know_indexes:
                know_retrieval.append(know_chunks[know_index])
        queriess = [item for item in queriess for _ in range(2)]
        pro_retrieval_filter = filter(queriess, pro_retrieval)
        know_retrieval_filter = filter(queriess, know_retrieval)
        pro.append(pro_retrieval_filter)
        know.append(know_retrieval_filter)
        pro_extraction = extraction(pro_retrieval_filter)
        know_extraction = extraction(know_retrieval_filter)
        pro_cross_retrieval = []
        know_cross_retrieval = []
        for query in pro_extraction:
            _, know_chunks = retrieval_RAGREK(query=query)
            know_sentence_pairs = get_sentence_pairs(query,know_chunks)
            know_indexes = reranker_top(know_sentence_pairs, K_2)
            for know_index in know_indexes:
                know_cross_retrieval.append(know_chunks[know_index])
        for query in know_extraction:
            pro_chunks, _ = retrieval_RAGREK(query=query)
            pro_sentence_pairs = get_sentence_pairs(query,pro_chunks)
            pro_indexes = reranker_top(pro_sentence_pairs, K_2)
            for pro_index in pro_indexes:
                pro_cross_retrieval.append(pro_chunks[pro_index])
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
        pro_cross_extraction = extraction(pro_cross_retrieval_filter)
        know_cross_extraction = extraction(know_cross_retrieval_filter)
        logging.debug("***********************************************")
        logging.debug(pro_cross_extraction)
        logging.debug(know_cross_extraction)
        logging.debug("***********************************************")
        all = [pro_extraction, know_extraction, pro_cross_extraction, know_cross_extraction]
        logging.info(f"最后检索知识数量：{sum(len(x) for x in all)}")
        questions, pairs = all_question_extraction(all=all)
        if len(pairs) > 4:
            final_pairs = cluster_K_Means(questions, pairs, encoder_model, J)
            final_knowledge = extraction_by_query(final_pairs)
        else:
            final_knowledge = pairs
        prompt = answer_prompt(question, final_knowledge)
        res = query_model(prompt)
        anses.append(res)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(anses, f,indent=4, ensure_ascii=False)
  
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure file paths and model settings.")
    parser.add_argument('-e', '--embed_path', type=str, required=True, help="Path to the embedding model")
    parser.add_argument('-k', '--know_db_path', type=str, required=True, help="Path to the knowledge database")
    parser.add_argument('-p', '--pro_db_path', type=str, required=True, help="Path to the professional knowledge database")
    parser.add_argument('-a', '--all_db_path', type=str, required=True, help="Path to the complete database")
    parser.add_argument('-r', '--reranker_path', type=str, required=True, help="Path to the Reranker model")
    parser.add_argument('--inference_model_type', '-m', type=str, required=True, help="Type of inference model")
    parser.add_argument('--data_path', '-d', type=str, required=True, help="Path to the data")
    parser.add_argument('--save_path', '-s', type=str, required=True, help="Path to save the results")
    parser.add_argument('--K_1', '-k1', type=int, default=2, help="Value of K_1 (default: 2)")
    parser.add_argument('--K_2', '-k2', type=int, default=1, help="Value of K_2 (default: 1)")
    parser.add_argument('--J', '-j', type=int, default=4, help="Value of J (default: 4)")
    parser.add_argument('--ablation', '-a', action='store_true', help="Enable ablation study (default: False)")
    parser.add_argument('--single_turn', '-st', action='store_true', help="Choose  the single-turn dialogue question (default: True)")
    args = parser.parse_args()      
    
    initialize(args.embed_path, args.know_db_path, args.pro_db_path, args.all_db_path, args.reranker_path)
    CascadeRCG(args.inference_model_type, args.data_path, args.save_path, args.K_1, args.K_2, args.J, args.ablation, args.single_turn)
        
        
        
        
        
            
        


    
    
    
    




