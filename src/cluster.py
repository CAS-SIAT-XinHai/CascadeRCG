from sklearn.cluster import KMeans
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieve_filter_summarize import extraction_by_query

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

def clustering_then_summarizing(all, encoder_model, J):
    questions, pairs = all_question_extraction(all=all)
    if len(pairs) > 4:
        final_pairs = cluster_K_Means(questions, pairs, encoder_model, J)
        final_knowledge = extraction_by_query(final_pairs)
    else:
        final_knowledge = pairs
    return final_knowledge