import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.CascadeRCG_prompt import judge_prompt, extraction_prompt
from src.get_json import change_to_json

def retrieval_RAGREK(query, pro_vectordb, know_vectordb):    
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

def reranker_top(sentence_pairs, k, reranker_model):
# calculate scores of sentence pairs
    sort_scores = {}
    scores = reranker_model.predict(sentence_pairs)
    for i, score in enumerate(scores):
        sort_scores[i] = score
    scores_ = sorted(sort_scores.items(), key=lambda item: item[1], reverse=True)
    indexes = [scores_[i][0] for i in range(k)]
    return indexes

def filter(queries, retrieval_knowledge, query_model):
    retrieval_filter = []
    for question, knowledge in zip(queries, retrieval_knowledge):
        prompt = judge_prompt(question, knowledge)
        res = query_model(prompt)
        if "是" in res:
            retrieval_filter.append((question,knowledge))
    return retrieval_filter

def extraction_by_query(retrieval_filter, query_model):
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

def get_cross_question_psyqa(cross_retrieval):
    question_psyqa = []
    for question in cross_retrieval:
        try:
            content = change_to_json(question)
            data_json = json.loads(content)
            question_psyqa.append(data_json.keys())
        except Exception as e:
            print(e)
            
    return question_psyqa

def RFS(K_1, pro_vectordb, know_vectordb, reranker_model, query_model):
    pro_retrieval = []
    know_retrieval = []
    for query in queriess:
        pro_chunks, know_chunks = retrieval_RAGREK(query, pro_vectordb, know_vectordb)
        pro_sentence_pairs = get_sentence_pairs(query,pro_chunks)
        know_sentence_pairs = get_sentence_pairs(query,know_chunks)
        pro_indexes = reranker_top(pro_sentence_pairs, K_1, reranker_model)
        know_indexes = reranker_top(know_sentence_pairs, K_1, reranker_model)
        for pro_index in pro_indexes:
            pro_retrieval.append(pro_chunks[pro_index])
        for know_index in know_indexes:
            know_retrieval.append(know_chunks[know_index])
    queriess = [item for item in queriess for _ in range(2)]
    pro_retrieval_filter = filter(queriess, pro_retrieval, query_model)
    know_retrieval_filter = filter(queriess, know_retrieval, query_model)
    pro_extraction = extraction_by_query(pro_retrieval_filter, query_model)
    know_extraction = extraction_by_query(know_retrieval_filter, query_model)
    return pro_extraction, know_extraction

def cross_RFS(K_2, pro_vectordb, know_vectordb, reranker_model, query_model, pro_extraction, know_extraction):
    pro_cross_retrieval = []
    know_cross_retrieval = []
    for query in pro_extraction:
        _, know_chunks = retrieval_RAGREK(query, pro_vectordb, know_vectordb)
        know_sentence_pairs = get_sentence_pairs(query,know_chunks)
        know_indexes = reranker_top(know_sentence_pairs, K_2, reranker_model)
        for know_index in know_indexes:
            know_cross_retrieval.append(know_chunks[know_index])
    for query in know_extraction:
        pro_chunks, _ = retrieval_RAGREK(query, pro_vectordb, know_vectordb)
        pro_sentence_pairs = get_sentence_pairs(query,pro_chunks)
        pro_indexes = reranker_top(pro_sentence_pairs, K_2, reranker_model)
        for pro_index in pro_indexes:
            pro_cross_retrieval.append(pro_chunks[pro_index])
    pro_question_psyqa = get_cross_question_psyqa(pro_extraction)
    know_question_psyqa = get_cross_question_psyqa(know_extraction)
    pro_cross_retrieval_filter = filter(queries=pro_question_psyqa, retrieval_knowledge=pro_cross_retrieval, query_model=query_model)
    know_cross_retrieval_filter = filter(queries=know_question_psyqa, retrieval_knowledge=know_cross_retrieval, query_model=query_model)
    pro_cross_extraction = extraction_by_query(pro_cross_retrieval_filter, query_model)
    know_cross_extraction = extraction_by_query(know_cross_retrieval_filter, query_model)
    return pro_cross_extraction, know_cross_extraction