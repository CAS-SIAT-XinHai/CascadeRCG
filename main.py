import random
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import re
import logging
import argparse
import json
import random
import logging
from src.inference_models import *
from src.load_database import initialize
from src.cluster import clustering_then_summarizing
from src.query_rewriting import rewriting
from src.retrieve_filter_summarize import RFS, cross_RFS
from  prompts.CascadeRCG_prompt import *
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

query_model = None

def generate_random_data(rewriting_path):
    with open(rewriting_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    random_item = random.choice(json_data)
    return random_item

def CascadeRCG(inference_model_type: str, data_path: str, save_path: str, K_1=2, K_2=1, J=4, single_turn=True):    
    global query_model
    query_model = model_type(inference_model_type)
    if single_turn:
        answer_prompt = single_turn_rewriting_prompt
    else:
        answer_prompt = muti_turn_rewriting_prompt
    with open(data_path, encoding='utf-8') as f:
        datas = json.load(f)
    psyqa_questions = []
    for data in datas:
        psyqa_questions.append(data)
    anses = []
    length = len(psyqa_questions)
    for i in range(length):
        question = psyqa_questions[i]
        queriess = rewriting(question, query_model, single_turn)
        logging.debug(f"-------Rewriting Queried: {queriess}")
        pro_extraction, know_extraction = RFS(K_1, pro_vectordb, know_vectordb, reranker_model, query_model)
        pro_cross_extraction, know_cross_extraction = cross_RFS(K_2, pro_vectordb, know_vectordb, reranker_model, query_model, pro_extraction, know_extraction)
        all = [pro_extraction, know_extraction, pro_cross_extraction, know_cross_extraction]
        final_knowledge = clustering_then_summarizing(all, encoder_model, J)
        prompt = answer_prompt(question, final_knowledge)
        res = query_model(prompt)
        logging.debug(f"-------Answer:: {res}")
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
    
    all_vectordb, know_vectordb, pro_vectordb, reranker_model, encoder_model = initialize(args.embed_path, args.know_db_path, args.pro_db_path, args.all_db_path, args.reranker_path)
    CascadeRCG(args.inference_model_type, args.data_path, args.save_path, args.K_1, args.K_2, args.J, args.ablation, args.single_turn)
        
        
        
        
        
            
        


    
    
    
    




