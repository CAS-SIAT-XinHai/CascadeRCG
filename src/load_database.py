import random
import json
import argparse
import logging
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagModel

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def vectors_db(embed_path, db_path):
    query_instruction = "为这个句子生成表示以用于检索相关文章："
    embeddings = HuggingFaceBgeEmbeddings(model_name=embed_path,
                                          model_kwargs={'device': 'cuda'},
                                          query_instruction=query_instruction)
    vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return vectordb

def initialize(embed_path, know_db_path, pro_db_path, all_db_path, reranker_path):
    all_vectordb = vectors_db(embed_path, all_db_path)
    know_vectordb = vectors_db(embed_path, know_db_path)
    pro_vectordb = vectors_db(embed_path, pro_db_path)

    # Initialize Reranker model
    model = CrossEncoder(reranker_path, max_length=512)
    
    # Initialize encoder model
    encoder_model = FlagModel(embed_path,
                              query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                              use_fp16=True)
    return all_vectordb, know_vectordb, pro_vectordb, model, encoder_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure file paths and model settings.")
    parser.add_argument('-e', '--embed_path', type=str, required=True, help="Path to the embedding model")
    parser.add_argument('-k', '--know_db_path', type=str, required=True, help="Path to the knowledge database")
    parser.add_argument('-p', '--pro_db_path', type=str, required=True, help="Path to the professional knowledge database")
    parser.add_argument('-a', '--all_db_path', type=str, required=True, help="Path to the complete database")
    parser.add_argument('-r', '--reranker_path', type=str, required=True, help="Path to the Reranker model")

    args = parser.parse_args()

    # Initialize and optionally do something with the returned objects
    all_vectordb, know_vectordb, pro_vectordb, model, encoder_model = initialize(
        args.embed_path, args.know_db_path, args.pro_db_path, args.all_db_path, args.reranker_path
    )
