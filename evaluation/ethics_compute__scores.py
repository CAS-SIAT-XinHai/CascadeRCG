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
    A_scores = []
    B_scores = []
    C_scores = []
    D_scores = []
    E_scores = []
    F_scores = []
    length = len(datas)
    for i in range(length):
        A_scores.append(datas[i]['A'])
        B_scores.append(datas[i]['B'])
        C_scores.append(datas[i]['C'])
        D_scores.append(datas[i]['D'])
        E_scores.append(datas[i]['E'])
        F_scores.append(datas[i]['F'])
    score = {
        'A': sum(A_scores)/length,
        'B': sum(B_scores)/length,
        'C': sum(C_scores)/length,
        'D': sum(D_scores)/length,
        'E': sum(E_scores)/length,
        'F': sum(F_scores)/length
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(score, f, ensure_ascii=False, indent=4)
    return
    
    
    

if __name__ == '__main__':
    args = parse_args()
    compute_score(args.com_path, args.sco_path)