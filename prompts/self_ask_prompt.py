def judge_follow_up(history):
    prompt = f"""
    根据对话历史，请判断是否需要在下一步提出新的问题。若需要仅返回是，若不需要额外的问题返回否，禁止给出理由。

    对话历史：{history}

    """
    return prompt

def get_follow_up_question(history):
    prompt = f"""
    根据对话历史，给出后续问题。仅给出问题，以一句话给出问题，禁止出现回答。

    对话历史：{history}

    """
    return prompt

def get_intermidiate_answer(knowledge, history):
    prompt = f"""
    你是一名心理学专家，根据对话历史，给出你的回答，回答应简洁明了。
    
    对话历史：{history}
    可参考知识知识：{knowledge}
    
    仅返回回答内容，禁止出现其他字段。

    """
    return prompt

def get_final_answer(history):
    prompt = f"""
    你是一名心理学专家，根据对话历史，综合之前的问题和中间回答给出最终回答，最终回答应该内容丰富，尽量生成长文本。
    
    对话历史：{history}
    
    仅返回回答内容，禁止出现其他字段。

    """
    return prompt





