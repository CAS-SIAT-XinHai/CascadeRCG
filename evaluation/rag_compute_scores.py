import json
import sys
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Computing the evaluation result.')
    parse.add_argument('-c','--com_path', required=True, type=str, help='The path of evaluation result.')
    parse.add_argument('-s','--sco_path', required=True, type=str, help='The path of scores result.')
    args = parse.parse_args()
    return args

def compute_score(path, save_path):
    with open(path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    # print(len(datas))
    # print(datas[0])
    pro_scores = []
    Know_scores = []
    Emp_scores = []
    Nohu_scores = []
    Saf_scores = []
    length = len(datas)
    for i in range(length):
        pro_scores.append(datas[i]['Professionalism'])
        Know_scores.append(datas[i]['Knowledgeability'])
        Emp_scores.append(datas[i]['Empathy'])
        print(datas[i])
        Nohu_scores.append(datas[i]['No hallucinations'])
        Saf_scores.append(datas[i]['Safety'])
    score = {
        'Professionalism': sum(pro_scores)/length,
        'Knowledgeability': sum(Know_scores)/length,
        'Empathy': sum(Emp_scores)/length,
        'No hallucinations': sum(Nohu_scores)/length,
        'Safety': sum(Saf_scores)/length
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(score, f, ensure_ascii=False, indent=4)
    return
    
    
    

if __name__ == '__main__':
    args = parse_args()
    compute_score(args.com_path, args.sco_path)